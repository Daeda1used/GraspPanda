"""FineGrasp's native local attention and group fusion for Graspness seeds."""
import torch
from torch import nn


class FineGraspCrop(nn.Module):
    def __init__(self,native,nsample=16,radius_factors=(.25,.5,.75,1.)):
        super().__init__()
        from ..finegrasp import native_module
        source=native_module()
        self.groups=nn.ModuleList(source.CylinderGroup(
            nsample=nsample,seed_feature_dim=native.in_dim,
            cylinder_radius=native.grouper.radius*factor,
            hmin=native.grouper.hmin,hmax=native.grouper.hmax)
            for factor in radius_factors)
        self.fusion=source.GroupTransformerFusion(feature_dim=256,num_groups=len(self.groups))

    def forward(self,seed_xyz,seed_features,rotations):
        features=torch.stack([group(seed_xyz,seed_features,rotations) for group in self.groups],dim=1)
        return self.fusion(features)
