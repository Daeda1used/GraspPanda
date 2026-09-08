"""Three-radius cylindrical grouping, preserving the baseline depth-bin contract."""
import torch
from torch import nn


class MultiScaleCrop(nn.Module):
    def __init__(self,native,radius_factors=(0.5,1.0,1.5)):
        super().__init__()
        cls=type(native)
        self.branches=nn.ModuleList([cls(native.nsample,native.in_dim,
            cylinder_radius=native.cylinder_radius*factor) for factor in radius_factors])
        self.fusion=nn.Sequential(nn.Conv2d(256*len(radius_factors),256,1,bias=False),
                                  nn.BatchNorm2d(256),nn.ReLU(inplace=True))

    def forward(self,seed_xyz,pointcloud,rotations):
        return self.fusion(torch.cat([branch(seed_xyz,pointcloud,rotations) for branch in self.branches],dim=1))
