"""Shared image encoders projected onto ZeroGrasp's calibrated dense lattice."""
import torch
from torch import nn
from torch.nn import functional as F

from .image_pyramid import ImagePyramid


def dense_lattice(feature, size, stride):
    """A pyramid cell at index i represents input pixel stride*i."""
    height, width = feature.shape[-2:]
    y = (torch.arange(size[0], device=feature.device, dtype=feature.dtype)/stride+.5)*2/height-1
    x = (torch.arange(size[1], device=feature.device, dtype=feature.dtype)/stride+.5)*2/width-1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xx,yy), dim=-1)[None].expand(feature.shape[0],-1,-1,-1)
    return F.grid_sample(feature, grid, padding_mode='border', align_corners=False)


class DenseImageFeatures(nn.Module):
    def __init__(self, family, **options):
        super().__init__()
        self.pyramid = ImagePyramid(family, input_channels=3, input_size=(480,640), **options)
        self.lateral = nn.ModuleList(nn.Conv2d(4*s,32,1) for s in (2,4,8,16,32))
        self.fuse = nn.Sequential(nn.Conv2d(32,32,3,padding=1,bias=False),
                                  nn.GroupNorm(8,32), nn.GELU(), nn.Conv2d(32,32,1))

    def forward(self, image):
        maps = self.pyramid(image)
        result = sum(dense_lattice(project(value), image.shape[-2:], stride)
                     for value,project,stride in zip(maps,self.lateral,(2,4,8,16,32)))/5
        return self.fuse(result)
