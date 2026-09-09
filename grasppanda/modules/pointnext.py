"""Native PointNeXt encoder/decoder adapted to GraspNet seed features."""
import importlib
from pathlib import Path
import sys
import types
import torch
from torch import nn


def native_module():
    from ..config import ROOT
    root=ROOT/'environments/sources/cv/pointnext/openpoints'
    if not (root/'models/backbone/pointnext.py').exists():
        raise ValueError('PointNeXt source is missing; run the component installer')
    # Load only the native network dependency graph. OpenPoints package-level
    # imports also initialize unrelated datasets, classification and transforms.
    for name, relative in (('openpoints',''),('openpoints.utils','utils'),
                           ('openpoints.models','models'),('openpoints.models.backbone','models/backbone')):
        if name not in sys.modules:
            package=types.ModuleType(name);package.__path__=[str(root/relative)]
            sys.modules[name]=package
    extension=importlib.import_module('_grasppanda_openpoints_cuda')
    for name,relative in (('openpoints.cpp','cpp'),('openpoints.cpp.pointnet2_batch','cpp/pointnet2_batch')):
        if name not in sys.modules:
            package=types.ModuleType(name);package.__path__=[str(root/relative)]
            package.pointnet2_cuda=extension;sys.modules[name]=package
    return importlib.import_module('openpoints.models.backbone.pointnext')


class PointNeXtBackbone(nn.Module):
    def __init__(self,width=32,blocks=(1,1,1,1,1),nsample=32,radius=.05,
                 radius_scaling=2.,expansion=4,activation='relu',reduction='max',decoder_layers=2):
        super().__init__()
        from easydict import EasyDict
        native=native_module()
        self.encoder=native.PointNextEncoder(in_channels=3,width=width,blocks=list(blocks),
            strides=[1,4,4,4,4],nsample=nsample,radius=radius,radius_scaling=radius_scaling,
            expansion=expansion,act_args={'act':activation},conv_args={},
            group_args=EasyDict(NAME='ballquery'),aggr_args={'feature_type':'dp_fj','reduction':reduction})
        self.decoder=native.PointNextDecoder(self.encoder.channel_list.copy(),decoder_layers=decoder_layers)
        self.projection=nn.Conv1d(self.decoder.out_channels,256,1)

    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        if points.ndim!=3 or points.shape[2]!=3 or points.shape[1]<1024:
            raise ValueError(getattr(self,'family','PointNeXt')+' requires camera XYZ [B,N,3] with N >= 1024')
        points=points.contiguous()
        positions,features=self.encoder.forward_seg_feat(points)
        dense=self.projection(self.decoder(positions,features))
        if dense.shape[-1]!=points.shape[1]:raise ValueError(getattr(self,'family','PointNeXt')+' decoder lost original point correspondence')
        from .sampling import sample_indices
        indices=sample_indices(points,1024,getattr(self,"seed_sampling","upstream"),native=_ext.furthest_point_sampling,training=self.training)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(2,indices.long()[:,None,:].expand(-1,256,-1)).contiguous()
        end_points={} if end_points is None else end_points
        end_points.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points
