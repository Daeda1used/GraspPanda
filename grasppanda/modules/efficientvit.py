"""Native MIT EfficientViT stages with calibrated RGB-D feature alignment."""
import ast
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import hashlib
import math
import sys
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F

from .efficientvit_options import resolve

SOURCE_HASHES = {'models/utils/list.py': 'de0b3b62f0f7eebe6f83679697ea5955a6a502eaae5695bb86c0c0340a6e778e', 'models/utils/network.py': 'b437a1f41c7721e47def883b4e1a8b142bfc295372007ffc0b358bd1d0dfc6aa', 'models/utils/random.py': '6b0428cf88fc6feb5abdb3de4276fe5db4984d1d5a0a76750122af13c14643d7', 'models/nn/triton_rms_norm.py': 'b9ccc6000a3d6cadd0f43614ffe4590cc0bc4838c81de9ad291175a0053b8033', 'models/nn/norm.py': 'efaad2af9fd6805e18c759405b534a9fbb0be39a8e952a7a3c1c52c371f8bdef', 'models/nn/act.py': '4bcf83cb4e28c2b83e000b244948d143feb7e3784e67407fda383604abc6074f', 'models/nn/ops.py': '6e632634ae60eb6b7fb738b393e7a111bc80e2c074cc4f527fb75e13952fffbe', 'models/efficientvit/backbone.py': '027a0bad8beea923d0992d3d75a9033bdd14e7e3fdb1e60f9b6561b250b40ccd'}


@lru_cache(maxsize=1)
def native_module():
    from ..config import ROOT
    root=ROOT/'environments/sources/cv/efficientvit/efficientvit'
    prefix='_grasppanda_efficientvit'
    if prefix in sys.modules:raise ValueError('An EfficientViT namespace is already loaded; restart the worker')
    for name,sha in SOURCE_HASHES.items():
        path=root/name
        if not path.is_file():raise ValueError('EfficientViT source is missing; run ./panda install')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('EfficientViT source differs from its registered file: '+name)
    def package(name):
        module=ModuleType(name);module.__path__=[];sys.modules[name]=module
        return module
    def load(relative):
        name=prefix+'.'+relative.replace('/', '.').removesuffix('.py')
        path=root/relative;tree=ast.parse(path.read_bytes(),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('efficientvit.'):
                node.module=prefix+node.module[len('efficientvit'):]
        module=ModuleType(name);module.__file__=str(path);module.__package__=name.rpartition('.')[0]
        sys.modules[name]=module
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
        return module
    try:
        package(prefix);package(prefix+'.models')
        utils=package(prefix+'.models.utils');ops=package(prefix+'.models.nn')
        package(prefix+'.models.efficientvit')
        for name in ('list','network','random'):
            module=load('models/utils/'+name+'.py')
            utils.__dict__.update({key:getattr(module,key) for key in module.__all__})
        for name in ('triton_rms_norm','norm','act','ops'):
            module=load('models/nn/'+name+'.py')
            ops.__dict__.update({key:getattr(module,key) for key in module.__all__})
        backbone=load('models/efficientvit/backbone.py')
        backbone.ops=ops
        return backbone
    except Exception:
        for name in list(sys.modules):
            if name==prefix or name.startswith(prefix+'.'):sys.modules.pop(name,None)
        raise


@contextmanager
def preserve_norm_buffers(stage):
    """Checkpoint recomputation must not update BN statistics a second time."""
    states=[(layer,{key:value.clone() for key,value in layer.named_buffers(recurse=False)})
            for layer in stage.modules() if isinstance(layer,nn.modules.batchnorm._BatchNorm)]
    try:yield
    finally:
        with torch.no_grad():
            for layer,buffers in states:
                for key,value in buffers.items():getattr(layer,key).copy_(value)


class EfficientViTPyramid(nn.Module):
    def __init__(self, **options):
        super().__init__();p=resolve(options);native=native_module()
        self.options=p;self.large=p['variant'].startswith('l')
        active=[i for i,kind in enumerate(p.get('stage_blocks',[])) if kind.startswith('att')] if self.large else [3,4]
        native_dim=16 if p['variant'] in ('b0','b1') else 32
        # Build valid native placeholders before installing per-block attention.
        # Narrow custom stages may be smaller than the author's default head dim.
        constructor_dim=min([native_dim]+[p['stage_channels'][i] for i in active])
        kwargs=dict(width_list=p['stage_channels'],depth_list=p['stage_depths'],in_channels=3,
                    norm=p['norm'],act_func=p['activation'])
        if self.large:
            self.encoder=native.EfficientViTLargeBackbone(**kwargs,block_list=p['stage_blocks'],
                expand_list=p['stage_expansions'],fewer_norm_list=p['fewer_norm'],qkv_dim=constructor_dim)
        else:self.encoder=native.EfficientViTBackbone(**kwargs,dim=constructor_dim,expand_ratio=p['expand_ratio'])
        index=0
        for stage_id,stage in enumerate(self.stages()):
            for j,block in enumerate(stage.op_list):
                if not isinstance(block,native.ops.EfficientViTBlock):continue
                width=p['stage_channels'][stage_id]
                block.context_module.main=native.ops.LiteMLA(width,width,heads=p['attention_heads'][index],
                    dim=p['attention_dims'][index],scales=tuple(p['attention_scales'][index]),
                    use_bias=p['attention_bias'][index],norm=(None,p['norm']),
                    kernel_func=p['attention_kernel'],eps=p['attention_epsilon'])
                index+=1
        if index!=len(p['attention_dims']):raise ValueError('Native EfficientViT attention layout changed')
        for layer in self.encoder.modules():
            if isinstance(layer,(nn.modules.batchnorm._BatchNorm,nn.LayerNorm)):layer.eps=p['norm_epsilon']
        self.rgb_projections=nn.ModuleList();self.depth_projections=nn.ModuleList()
        for channels,stride in zip(p['stage_channels'],(2,4,8,16,32)):
            target=4*stride
            layers=[nn.Conv2d(channels,target,1,bias=p['projection_norm']=='none')]
            if p['projection_norm']=='batch':layers.append(nn.BatchNorm2d(target))
            elif p['projection_norm']=='group':layers.append(nn.GroupNorm(8,target))
            self.rgb_projections.append(nn.Sequential(*layers))
            self.depth_projections.append(nn.Conv2d(1,target,3,padding=1))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        for stage in self.stages()[:5-p['trainable_stages']]:stage.requires_grad_(False)
        self.train(self.training)

    def stages(self):
        return list(self.encoder.stages) if self.large else [self.encoder.input_stem,*self.encoder.stages]

    def train(self, mode=True):
        super().train(mode)
        for stage in self.stages()[:5-self.options['trainable_stages']]:stage.eval()
        if mode and self.options['freeze_norm_stats']:
            for layer in self.modules():
                if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()
        return self

    def initialize_pretrained(self):
        if not self.options['pretrained']:return None
        from ..weights import fetch_component,component_records
        ident='efficientvit_'+self.options['variant']
        state=torch.load(fetch_component(ident),map_location='cpu',weights_only=True)
        if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
        if not isinstance(state,dict) or not all(key.startswith(('backbone.','head.')) for key in state):
            raise ValueError('Unexpected EfficientViT classification checkpoint layout')
        selected={key.removeprefix('backbone.'):value for key,value in state.items() if key.startswith('backbone.')}
        self.encoder.load_state_dict(selected,strict=True)
        record=component_records()[ident]
        return dict(id=ident,source=record['source'],sha256=record['sha256'],trainable_stages=self.options['trainable_stages'])

    def forward(self, x):
        if x.ndim!=4 or x.shape[1:]!=(4,640,360):raise ValueError('EfficientViT requires native D,R,G,B [B,4,640,360] inputs')
        value=(x[:,1:].transpose(-2,-1)-self.rgb_mean)/self.rgb_std
        result=[]
        for stage,project,depth_project,stride in zip(self.stages(),self.rgb_projections,self.depth_projections,(2,4,8,16,32)):
            if self.options['gradient_checkpointing'] and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                value=checkpoint(stage,value,use_reentrant=False,
                    context_fn=lambda stage=stage:(nullcontext(),preserve_norm_buffers(stage)))
            else:value=stage(value)
            feature=value.transpose(-2,-1)
            shape=tuple(math.ceil(size/stride) for size in x.shape[-2:])
            if feature.shape[-2:]!=shape:raise ValueError('EfficientViT lost the native camera feature lattice')
            result.append(F.leaky_relu(project(feature)+depth_project(x[:,:1,::stride,::stride])))
        return result
