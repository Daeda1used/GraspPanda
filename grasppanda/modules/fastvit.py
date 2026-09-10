"""Author FastViT blocks and a calibrated RGB-D pyramid for HGGD/RNG."""
import ast
from contextlib import nullcontext
from functools import lru_cache, partial
import hashlib
import math
import sys
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F

from .efficientvit import preserve_norm_buffers
from .fastvit_options import resolve, weight_id

SOURCE_HASHES = {'fastvit': {'models/fastvit.py': '80f602b9ad14a8738a6b7f0915b86f64ea325ae61d9034ac59c270be28927740', 'models/modules/mobileone.py': 'e881b6c921fddf6b44acf0230d585460c7d443931aa4984cbdf1015ac7d71b3a', 'models/modules/replknet.py': '6a85668cf1c9870bac04149040f0e4198f5408b49f355cf688f4d4c7838c3da7'}, 'fastvlm': {'llava/model/multimodal_encoder/mobileclip/mci.py': 'f8655a05a423531a903bfd22824eeb744d9c43e5a0500f5b511bfcba009ca7e1'}}


@lru_cache(maxsize=2)
def native_module(hd=False):
    from ..config import ROOT
    family='fastvlm' if hd else 'fastvit';root=ROOT/'environments/sources/cv'/family
    prefix='_grasppanda_'+family
    if prefix in sys.modules:raise ValueError('A FastViT namespace is already loaded; restart the worker')
    def load(relative,name):
        path=root/relative
        if not path.is_file():raise ValueError('FastViT source is missing; run ./panda install')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_HASHES[family][relative]:
            raise ValueError('FastViT source differs from its registered file: '+relative)
        tree=ast.parse(path.read_bytes(),filename=str(path));body=[]
        for node in tree.body:
            # The image definitions do not need optional detection frameworks or
            # global timm model registrations. Native numerical code is retained.
            if isinstance(node,ast.Try):continue
            if isinstance(node,ast.ImportFrom) and node.module in ('timm.models','timm.models.registry'):continue
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                node.decorator_list=[d for d in node.decorator_list if not isinstance(d,ast.Name) or d.id!='register_model']
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('models.modules.'):
                node.module=prefix+'.'+node.module.rsplit('.',1)[-1]
            if isinstance(node,ast.ClassDef) and node.name=='AttentionBlock':
                ctor=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
                ctor.args.kwonlyargs.append(ast.arg(arg='head_dim'));ctor.args.kw_defaults.append(ast.Constant(32))
                calls=[n for n in ast.walk(ctor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='MHSA']
                if len(calls)!=1:raise ValueError('Native FastViT attention constructor changed')
                calls[0].keywords.append(ast.keyword(arg='head_dim',value=ast.Name(id='head_dim',ctx=ast.Load())))
            body.append(node)
        tree.body=body
        module=ModuleType(name);module.__file__=str(path);module.__package__=prefix
        module.has_mmseg=module.has_mmdet=False
        sys.modules[name]=module
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
        return module
    try:
        package=ModuleType(prefix);package.__path__=[];sys.modules[prefix]=package
        if hd:return load('llava/model/multimodal_encoder/mobileclip/mci.py',prefix+'.model')
        load('models/modules/mobileone.py',prefix+'.mobileone')
        load('models/modules/replknet.py',prefix+'.replknet')
        return load('models/fastvit.py',prefix+'.model')
    except Exception:
        for name in list(sys.modules):
            if name==prefix or name.startswith(prefix+'.'):sys.modules.pop(name,None)
        raise


def sample_coarse(feature, shape):
    """Align stride-64, pixel-zero centers to stride-32, pixel-zero centers."""
    height,width=feature.shape[-2:]
    y=(torch.arange(shape[0],device=feature.device,dtype=feature.dtype)*.5+.5)*2/height-1
    x=(torch.arange(shape[1],device=feature.device,dtype=feature.dtype)*.5+.5)*2/width-1
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    grid=torch.stack((xx,yy),-1)[None].expand(feature.shape[0],-1,-1,-1)
    return F.grid_sample(feature,grid,mode='bilinear',padding_mode='border',align_corners=False)


class FastViTPyramid(nn.Module):
    def __init__(self, **options):
        super().__init__();p=resolve(options);self.options=p;self.hd=p['variant']=='hd'
        native=native_module(self.hd);fused=p['parameterization']=='fused'
        self.encoder=nn.Module();self.encoder.network=nn.ModuleList()
        kwargs=dict(use_scale_branch=False) if self.hd else {}
        self.encoder.patch_embed=native.convolutional_stem(3,p['stage_channels'][0],fused,**kwargs)
        self.stage_ranges=[];self.stage_indices=[];index=rep=attention=0
        activation={'gelu':nn.GELU,'relu':nn.ReLU,'silu':nn.SiLU}[p['activation']]
        norm=native.LayerNormChannel if self.hd else nn.BatchNorm2d
        for stage,(width,depth) in enumerate(zip(p['stage_channels'],p['stage_depths'])):
            start=len(self.encoder.network)
            if stage:
                self.encoder.network.append(native.PatchEmbed(patch_size=p['downsample_kernel'],stride=2,
                    in_channels=p['stage_channels'][stage-1],embed_dim=width,inference_mode=fused))
            if p['cpe_kernels'][stage]:
                self.encoder.network.append(native.RepCPE(width,width,spatial_shape=(p['cpe_kernels'][stage],)*2,inference_mode=fused))
            blocks=[]
            for _ in range(depth):
                kwargs=dict(dim=width,mlp_ratio=p['block_mlp_ratios'][index],act_layer=activation,
                    drop=p['block_dropout'][index],drop_path=p['block_drop_path'][index],
                    use_layer_scale=p['block_layer_scale'][index],layer_scale_init_value=p['block_layer_scale_init'][index])
                if p['block_mixers'][index]=='repmixer':
                    block=native.RepMixerBlock(**kwargs,kernel_size=p['repmixer_kernels'][rep],inference_mode=fused);rep+=1
                else:
                    block=native.AttentionBlock(**kwargs,norm_layer=norm,head_dim=p['attention_head_dims'][attention])
                    block.token_mixer=native.MHSA(width,head_dim=p['attention_head_dims'][attention],
                        qkv_bias=p['attention_qkv_bias'][attention],attn_drop=p['attention_dropout'][attention],
                        proj_drop=p['attention_projection_dropout'][attention]);attention+=1
                blocks.append(block);index+=1
            self.encoder.network.append(nn.Sequential(*blocks))
            self.stage_indices.append(len(self.encoder.network)-1)
            self.stage_ranges.append((start,len(self.encoder.network)))
        if self.hd:
            width=p['stage_channels'][-1]
            self.encoder.conv_exp=native.MobileOneBlock(width,width*2,3,stride=1,padding=1,
                groups=width,inference_mode=fused,use_se=True,num_conv_branches=1)
        self.encoder.apply(partial(native.FastViT.cls_init_weights,self.encoder))
        self.stem=nn.Sequential(nn.Conv2d(4,8,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(8),nn.LeakyReLU())
        self.rgb_projections=nn.ModuleList();self.depth_projections=nn.ModuleList()
        for width,stride in zip(p['stage_channels'][:4],(4,8,16,32)):
            target=4*stride;layers=[nn.Conv2d(width,target,1,bias=p['projection_norm']=='none')]
            if p['projection_norm']=='batch':layers.append(nn.BatchNorm2d(target))
            elif p['projection_norm']=='group':layers.append(nn.GroupNorm(8,target))
            self.rgb_projections.append(nn.Sequential(*layers))
            self.depth_projections.append(nn.Conv2d(1,target,3,padding=1))
        if self.hd:self.coarse_projection=nn.Conv2d(p['stage_channels'][-1]*2,128,1)
        self.register_buffer('rgb_mean',torch.tensor([0.,0.,0.] if self.hd else [.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([1.,1.,1.] if self.hd else [.229,.224,.225]).view(1,3,1,1))
        frozen=len(self.stage_ranges)-p['trainable_stages']
        if frozen:self.encoder.patch_embed.requires_grad_(False)
        for stage in range(frozen):
            for module in self.stage_modules(stage):module.requires_grad_(False)
        self.train(self.training)

    def stage_modules(self, stage):
        start,end=self.stage_ranges[stage];modules=list(self.encoder.network[start:end])
        if self.hd and stage==len(self.stage_ranges)-1:modules.append(self.encoder.conv_exp)
        return modules

    def train(self, mode=True):
        super().train(mode)
        frozen=len(self.stage_ranges)-self.options['trainable_stages']
        if frozen:self.encoder.patch_embed.eval()
        for stage in range(frozen):
            for module in self.stage_modules(stage):module.eval()
        if mode and self.options['freeze_norm_stats']:
            for module in self.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def initialize_pretrained(self):
        if not self.options['pretrained']:return None
        from ..weights import fetch_component,component_records
        ident=weight_id(self.options);path=fetch_component(ident)
        if self.hd:
            from safetensors import safe_open
            prefix='model.vision_tower.vision_tower.model.'
            with safe_open(path,framework='pt',device='cpu') as state:
                keys=[key for key in state.keys() if key.startswith(prefix)]
                if not keys:raise ValueError('FastVLM checkpoint lacks the registered native visual encoder')
                selected={key.removeprefix(prefix):state.get_tensor(key) for key in keys if not key.removeprefix(prefix).startswith('head.')}
        else:
            state=torch.load(path,map_location='cpu',weights_only=True)
            if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
            if not isinstance(state,dict) or not all(key.startswith(('patch_embed.','network.','conv_exp.','head.')) for key in state):
                raise ValueError('Unexpected FastViT classification checkpoint layout')
            selected={key:value for key,value in state.items() if key.startswith(('patch_embed.','network.'))}
        self.encoder.load_state_dict(selected,strict=True)
        record=component_records()[ident]
        return dict(id=ident,source=record['source'],sha256=record['sha256'],
                    parameterization=self.options['parameterization'],trainable_stages=self.options['trainable_stages'])

    def native_features(self, rgb):
        value=self.encoder.patch_embed(rgb);result=[]
        for stage in range(len(self.stage_ranges)):
            for module in self.stage_modules(stage):
                if self.options['gradient_checkpointing'] and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    value=checkpoint(module,value,use_reentrant=False,
                        context_fn=lambda module=module:(nullcontext(),preserve_norm_buffers(module)))
                else:value=module(value)
            result.append(value)
        return result

    def forward(self, x):
        if x.ndim!=4 or x.shape[1:]!=(4,640,360):raise ValueError('FastViT requires native D,R,G,B [B,4,640,360] inputs')
        rgb=(x[:,1:].transpose(-2,-1)-self.rgb_mean)/self.rgb_std
        features=self.native_features(rgb);result=[self.stem(x)]
        for value,project,depth_project,stride in zip(features[:4],self.rgb_projections,self.depth_projections,(4,8,16,32)):
            feature=value.transpose(-2,-1)
            expected=tuple(math.ceil(size/stride) for size in x.shape[-2:])
            if feature.shape[-2:]!=expected:raise ValueError('FastViT lost the native camera feature lattice')
            result.append(project(feature)+depth_project(x[:,:1,::stride,::stride]))
        if self.hd:
            coarse=self.coarse_projection(features[-1]).transpose(-2,-1)
            result[-1]=result[-1]+sample_coarse(coarse,result[-1].shape[-2:])
        return [result[0],*[F.leaky_relu(value) for value in result[1:]]]
