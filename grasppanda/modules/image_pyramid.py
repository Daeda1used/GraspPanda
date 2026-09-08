"""RGB-D image backbones aligned to native HGGD/RNG feature lattices."""
import math
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = {
    'convnextv2': {'atto': 'convnextv2_atto', 'tiny': 'convnextv2_tiny'},
    'repvit': {'m0_9': 'repvit_m0_9', 'm1_1': 'repvit_m1_1'},
    'mobilenetv4': {'small': 'mobilenetv4_conv_small', 'medium': 'mobilenetv4_conv_medium'},
}


def align_lattice(feature, shape, stride, offset=0.):
    """Sample at native centers (stride*i), without swapping image axes."""
    if offset == 0:
        return feature[..., :shape[0], :shape[1]]
    height, width = feature.shape[-2:]
    y = torch.arange(shape[0], device=feature.device, dtype=feature.dtype) - offset / stride
    x = torch.arange(shape[1], device=feature.device, dtype=feature.dtype) - offset / stride
    yy, xx = torch.meshgrid((y + .5) * 2 / height - 1, (x + .5) * 2 / width - 1, indexing='ij')
    grid = torch.stack([xx, yy], -1)[None].expand(feature.shape[0], -1, -1, -1)
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='border', align_corners=False)


class ImagePyramid(nn.Module):
    def __init__(self, family, variant=None, stage_channels=None, stage_depths=None,
                 drop_path=0., projection_norm='batch'):
        super().__init__()
        import timm
        variants = VARIANTS[family]
        name = variants[variant or next(iter(variants))]
        kwargs = dict(pretrained=False, in_chans=4, features_only=True)
        if family == 'convnextv2':
            kwargs['drop_path_rate'] = drop_path
            if stage_channels is not None: kwargs['dims'] = tuple(stage_channels)
            if stage_depths is not None: kwargs['depths'] = tuple(stage_depths)
        elif family == 'repvit':
            if stage_channels is not None: kwargs['embed_dim'] = tuple(stage_channels)
            if stage_depths is not None: kwargs['depth'] = tuple(stage_depths)
        else:
            kwargs['drop_path_rate'] = drop_path
        self.encoder = timm.create_model(name, **kwargs)
        self.configure_pyramid(family, self.encoder.feature_info.channels(),
                               self.encoder.feature_info.reduction(), projection_norm)

    def configure_pyramid(self, family, channels, strides, projection_norm, pad_to_stride=True):
        self.strides = tuple(strides)
        self.pad_to_stride = pad_to_stride
        if self.strides not in ((4, 8, 16, 32), (2, 4, 8, 16, 32)):
            raise ValueError('Image encoder does not expose the registered feature pyramid')
        self.family = family
        self.projections = nn.ModuleList()
        for channels, stride in zip(channels, self.strides):
            target = 4 * stride
            layers = [nn.Conv2d(channels, target, 1, bias=projection_norm == 'none')]
            if projection_norm == 'batch': layers.append(nn.BatchNorm2d(target))
            elif projection_norm == 'group': layers.append(nn.GroupNorm(8, target))
            layers.append(nn.LeakyReLU(inplace=True))
            self.projections.append(nn.Sequential(*layers))
        self.stem = None
        if self.strides[0] != 2:
            self.stem = nn.Sequential(nn.Conv2d(4, 8, 7, stride=2, padding=3, bias=False),
                                      nn.BatchNorm2d(8), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (4, 640, 360):
            raise ValueError('HGGD/RNG image backbones require native D,R,G,B [B,4,640,360] inputs')
        size = x.shape[-2:]
        # Padding keeps the input coordinate origin. It supplies the boundary
        # cells needed by the native ceil-stride feature maps.
        padded = F.pad(x, (0, (-size[1]) % 32, 0, (-size[0]) % 32)) if self.pad_to_stride else x
        maps = self.encoder(padded)
        result = [self.stem(x)] if self.stem is not None else []
        for value, project, stride in zip(maps, self.projections, self.strides):
            shape = tuple(math.ceil(side / stride) for side in size)
            # ConvNeXt's unpadded patch/merge kernels center at (stride-1)/2;
            # the odd, symmetrically padded RepViT/MobileNet kernels center at 0.
            offset = (stride - 1) / 2 if self.family == 'convnextv2' else 0.
            result.append(project(align_lattice(value, shape, stride, offset)))
        expected = [(4*s, math.ceil(size[0]/s), math.ceil(size[1]/s)) for s in (2,4,8,16,32)]
        if [tuple(value.shape[1:]) for value in result] != expected:
            raise ValueError('Image pyramid lost the native feature lattice')
        return result


def native_resnet(native, variant='18', stage_depths=None):
    """Use the selected method's original residual blocks and channel widths."""
    import importlib
    replacement = type(native)(in_dim=4, planes=8, mode=variant)
    if stage_depths is not None:
        module = importlib.import_module(type(replacement.net).__module__)
        block = module.BottleNeck if variant == '50' else module.BasicBlock
        replacement.net = type(replacement.net)(block, stage_depths, in_dim=4,
                                                planes=2 if variant == '50' else 8)
    return replacement
