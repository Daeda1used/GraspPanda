"""DINO image features with a depth branch and native grasp feature lattices."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sample_lattice(feature, shape, target_stride, patch_size):
    """Interpolate patch centers onto native pixel centers at stride*i."""
    height, width = feature.shape[-2:]
    offset = (patch_size - 1) / 2
    y = (torch.arange(shape[0], device=feature.device, dtype=feature.dtype)*target_stride-offset)/patch_size
    x = (torch.arange(shape[1], device=feature.device, dtype=feature.dtype)*target_stride-offset)/patch_size
    yy, xx = torch.meshgrid((y+.5)*2/height-1, (x+.5)*2/width-1, indexing='ij')
    grid = torch.stack([xx, yy], -1)[None].expand(feature.shape[0], -1, -1, -1)
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='border', align_corners=False)


class DinoPyramid(nn.Module):
    def __init__(self, family, variant='small', pretrained=True, out_indices=(2,5,8,11),
                 trainable_blocks=12, drop_path=0., gradient_checkpointing=False,
                 projection_norm='batch'):
        super().__init__()
        import timm
        from ..weights import component_records
        self.weight_id = family+'_'+variant
        record = component_records()[self.weight_id]
        self.encoder = timm.create_model(record['model'], pretrained=False, num_classes=0,
            dynamic_img_size=True, drop_path_rate=drop_path)
        self.encoder.set_grad_checkpointing(gradient_checkpointing)
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.out_indices = tuple(out_indices)
        self.pretrained_requested = pretrained
        self.trainable_blocks = trainable_blocks
        self.register_buffer('rgb_mean', torch.tensor(record['mean']).view(1,3,1,1))
        self.register_buffer('rgb_std', torch.tensor(record['std']).view(1,3,1,1))
        if trainable_blocks < len(self.encoder.blocks):
            self.encoder.requires_grad_(False)
            if trainable_blocks:
                self.encoder.blocks[-trainable_blocks:].requires_grad_(True)
                self.encoder.norm.requires_grad_(True)
        def projection(channels, target, kernel):
            layers = [nn.Conv2d(channels, target, kernel, padding=kernel//2, bias=projection_norm=='none')]
            if projection_norm=='batch': layers.append(nn.BatchNorm2d(target))
            elif projection_norm=='group': layers.append(nn.GroupNorm(8,target))
            return nn.Sequential(*layers)
        self.rgb_projections = nn.ModuleList(projection(self.encoder.num_features, 4*s, 1) for s in (4,8,16,32))
        self.depth_projections = nn.ModuleList(projection(1, 4*s, 3) for s in (4,8,16,32))
        self.stem = nn.Sequential(nn.Conv2d(4,8,7,stride=2,padding=3,bias=False),
                                  nn.BatchNorm2d(8), nn.LeakyReLU(inplace=True))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.trainable_blocks < len(self.encoder.blocks):
            self.encoder.eval()
            if mode and self.trainable_blocks:
                self.encoder.blocks[-self.trainable_blocks:].train()
                self.encoder.norm.train()
        return self

    def initialize_pretrained(self):
        if not self.pretrained_requested: return None
        from safetensors.torch import load_file
        from ..weights import component_records, fetch_component
        path = fetch_component(self.weight_id)
        self.encoder.load_state_dict(load_file(path), strict=True)
        record = component_records()[self.weight_id]
        return dict(id=self.weight_id, source=record['source'], sha256=record['sha256'],
                    trainable_blocks=self.trainable_blocks)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (4,640,360):
            raise ValueError('DINO grasp adapters require native D,R,G,B [B,4,640,360] input')
        # Undo the native width/height permutation for the pretrained RGB encoder.
        rgb = (x[:,1:].transpose(-2,-1)-self.rgb_mean)/self.rgb_std
        rgb = F.pad(rgb, (0,(-rgb.shape[-1])%self.patch_size,0,(-rgb.shape[-2])%self.patch_size), mode='replicate')
        maps = self.encoder.forward_intermediates(rgb, indices=self.out_indices,
            norm=True, output_fmt='NCHW', intermediates_only=True)
        result = [self.stem(x)]
        for value, project, depth_project, stride in zip(maps, self.rgb_projections, self.depth_projections, (4,8,16,32)):
            shape = tuple(math.ceil(side/stride) for side in x.shape[-2:])
            feature = sample_lattice(value.transpose(-2,-1), shape, stride, self.patch_size)
            result.append(F.leaky_relu(project(feature)+depth_project(x[:,:1,::stride,::stride])))
        return result
