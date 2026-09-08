"""PointNet-style local/global encoder adapted to the baseline seed contract.

This is a GraspPanda component, not a pretrained author PointNet++ checkpoint.
FPS determines output seed indices; point features use shared MLPs and global max.
"""
import torch
from torch import nn


def mlp(channels):
    layers=[]
    for a,b in zip(channels,channels[1:]):
        layers.extend([nn.Conv1d(a,b,1,bias=False),nn.BatchNorm1d(b),nn.ReLU(inplace=True)])
    return nn.Sequential(*layers)


class PointNetBackbone(nn.Module):
    def __init__(self,num_seeds=1024,local_channels=(64,128),global_channels=(256,512),
                 fusion_channels=(256,),activation='relu',normalization='batch',dropout=0.):
        super().__init__()
        from .layers import point_mlp
        self.num_seeds=num_seeds
        self.local=point_mlp([3,*local_channels],activation=activation,normalization=normalization)
        self.global_features=point_mlp([local_channels[-1],*global_channels],activation=activation,normalization=normalization)
        self.fusion=point_mlp([local_channels[-1]+global_channels[-1],*fusion_channels,256],activation=activation,normalization=normalization)
        self.dropout=nn.Dropout(dropout)

    def forward(self,pointcloud,end_points=None):
        from pointnet2 import _ext
        if pointcloud.ndim!=3 or pointcloud.shape[-1]!=3 or pointcloud.shape[1]<self.num_seeds:
            raise ValueError('PointNet slot expects [B,N,3], N >= num_seeds')
        end_points={} if end_points is None else end_points
        xyz=pointcloud.contiguous()
        features=self.local(xyz.transpose(1,2).contiguous())
        global_features=self.global_features(features).amax(2,keepdim=True).expand(-1,-1,xyz.shape[1])
        features=self.dropout(self.fusion(torch.cat([features,global_features],1)))
        indices=_ext.furthest_point_sampling(xyz,self.num_seeds)
        seeds=xyz.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=features.gather(2,indices.long()[:,None,:].expand(-1,features.shape[1],-1))
        end_points.update(input_xyz=xyz,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points
