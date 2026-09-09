"""Native OctFormer hierarchy with scene-local windows and grasp seed mapping."""
import ast
import hashlib
import importlib.util
import sys
import types
from contextlib import contextmanager, nullcontext
from functools import lru_cache

import torch
import ocnn
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from ..config import ROOT
from ..octformer_options import resolve, stage_value, STAGE_FIELDS

PREFIX = '_grasppanda_octformer'
SOURCE_HASHES = {'octformer.py': 'a9d145a89a18942a0694cab0fd62803d00f4f8f356e902f12b1a0c7074d52b9c', 'octformerseg.py': 'df0f1efbfd444e8f34ed00a729a353e0799d3b243f0cdf79d0be32b993f5cf39'}


@contextmanager
def temporary_bn_buffers(module):
    originals = []
    try:
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                    value = getattr(child, name)
                    if value is not None:
                        originals.append((child, name, value))
                        setattr(child, name, value.clone())
        yield
    finally:
        for child, name, value in originals:
            setattr(child, name, value)


def checkpoint(module, *args, **kwargs):
    if kwargs.get('use_reentrant') is not False:
        raise RuntimeError('Pinned OctFormer requires non-reentrant checkpointing')
    return torch_checkpoint(module, *args, **kwargs,
        context_fn=lambda: (nullcontext(), temporary_bn_buffers(module)))


def scene_octree_type(original):
    class SceneOctreeT(original):
        def build_t(self):
            self.scene_maps = {}
            for depth in range(self.start_depth, self.max_depth + 1):
                batch = self.batch_id(depth, self.nempty).long()
                counts = torch.bincount(batch, minlength=self.batch_size)
                padded = (counts + self.block_num - 1) // self.block_num * self.block_num
                shifts = torch.cat([counts.new_zeros(1), (padded-counts).cumsum(0)[:-1]])
                self.scene_maps[depth] = torch.arange(batch.numel(), device=batch.device) + shifts[batch]
                self.nnum_a[depth] = padded.sum().cpu()
            super().build_t()

        def patch_partition(self, data, depth, fill_value=0):
            out = data.new_full((int(self.nnum_a[depth]),) + data.shape[1:], fill_value)
            return out.index_copy(0, self.scene_maps[depth], data)

        def patch_reverse(self, data, depth):
            return data.index_select(0, self.scene_maps[depth])
    return SceneOctreeT


def source_module(path, name):
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != SOURCE_HASHES[path.name]:
        raise ValueError('OctFormer source differs from the pinned revision; restore it with ./panda install')
    tree = ast.parse(contents, filename=str(path))
    counts = dict(stage=0, block=0, attention=0, backbone=0)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == 'OctFormerStage':
            for key in STAGE_FIELDS:
                expression = "stage_value(use_dwconv, i)" if key == 'use_dwconv' else f"stage_value(kwargs.get({key!r}, DEFAULTS[{key!r}]), i)"
                value = ast.parse(expression, mode='eval').body
                existing = next((kw for kw in node.keywords if kw.arg == key), None)
                if existing:
                    existing.value = value
                else:
                    node.keywords.append(ast.keyword(arg=key, value=value))
            counts['stage'] += 1
        elif node.func.id in ('octformer_block', 'OctreeAttention'):
            node.keywords.append(ast.keyword(arg='use_rpe',
                value=ast.parse("kwargs.get('use_rpe', True)", mode='eval').body))
            counts['block' if node.func.id == 'octformer_block' else 'attention'] += 1
        elif node.func.id == 'OctFormer':
            node.keywords.append(ast.keyword(arg=None, value=ast.Name(id='kwargs', ctx=ast.Load())))
            counts['backbone'] += 1
    expected = dict(stage=1, block=1, attention=1, backbone=0) if path.name == 'octformer.py' else dict(stage=0, block=0, attention=0, backbone=1)
    if counts != expected:
        raise RuntimeError('OctFormer constructor layout differs from the pinned source')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    from ..octformer_options import DEFAULTS
    module.__dict__.update(stage_value=stage_value, DEFAULTS=DEFAULTS)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module


@lru_cache(maxsize=1)
def load():
    root = ROOT/'environments/sources/cv/octformer/models'
    if not all((root/name).is_file() for name in SOURCE_HASHES):
        raise ValueError('OctFormer source is missing; run ./panda install')
    try:
        import dwconv.core
    except ImportError as error:
        raise ValueError('OctFormer depthwise operators are missing; run ./panda install') from error
    package = types.ModuleType(PREFIX)
    package.__package__ = PREFIX
    package.__path__ = [str(root)]
    sys.modules[PREFIX] = package
    native = source_module(root/'octformer.py', PREFIX+'.octformer')
    native.OctreeT = scene_octree_type(native.OctreeT)
    native.checkpoint = checkpoint
    segmentation = source_module(root/'octformerseg.py', PREFIX+'.octformerseg')
    return native, segmentation


def geometry(points,depth=11,full_depth=2):
    if points.ndim!=3 or points.shape[-1]!=3 or points.shape[0]<1 or points.shape[1]<1:
        raise ValueError('Expected nonempty [B,N,3] camera XYZ')
    if points.dtype!=torch.float32 or not points.is_cuda or not torch.isfinite(points).all():
        raise ValueError('Expected finite CUDA float32 points')
    low=points.amin(1,keepdim=True);high=points.amax(1,keepdim=True)
    center=low*.5+high*.5
    radius=(high-low).amax(-1,keepdim=True)*.5
    if not torch.isfinite(radius).all():raise ValueError('Point coordinate extent overflow')
    normalized=(points-center)/radius.clamp_min(1e-6)*.99
    trees=[]
    for p,f in zip(normalized,points):
        tree=ocnn.octree.Octree(depth,full_depth,device=points.device)
        tree.build_octree(ocnn.octree.Points(p,features=f))
        trees.append(tree)
    tree=ocnn.octree.merge_octrees(trees);tree.construct_all_neigh()
    ids=torch.arange(points.shape[0],device=points.device).view(-1,1,1).expand(-1,points.shape[1],1)
    query=torch.cat([normalized,ids],dim=-1).reshape(-1,4)
    return tree,query

class OctFormerBackbone(nn.Module):
    def __init__(self, **options):
        super().__init__()
        from ..module_options import validate_options
        validate_options('graspnet_baseline', 'backbone', 'octformer', options)
        p = resolve(options)
        self.depth = p.pop('depth')
        self.full_depth = p.pop('full_depth')
        self.nempty = p['nempty']
        _, segmentation = load()
        self.network = segmentation.OctFormerSeg(in_channels=3, out_channels=256, **p)

    def dense(self,points):
        tree,query=geometry(points,self.depth,self.full_depth)
        minimum=self.depth-self.network.backbone.stem_down-self.network.backbone.num_stages+1
        counts=tree.nnum_nempty if self.nempty else tree.nnum
        if self.training and int(counts[minimum:self.depth+1].min())<2:
            raise ValueError('OctFormer training requires at least two octree nodes at every resolution; use a non-degenerate point cloud or a larger batch')
        data=tree.get_input_feature('F',self.nempty)
        out=self.network(data,tree,self.depth,query)
        return out.reshape(points.shape[0],points.shape[1],256).transpose(1,2).contiguous()
    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        if points.ndim!=3 or points.shape[-1]!=3 or points.shape[1]<1024:raise ValueError('Grasp seeds require at least 1024 input points')
        points=points.contiguous();dense=self.dense(points)
        indices=_ext.furthest_point_sampling(points,1024)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(2,indices.long()[:,None,:].expand(-1,256,-1)).contiguous()
        ends={} if end_points is None else end_points
        ends.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,ends
