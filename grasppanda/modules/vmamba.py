"""Native VMamba selective scans on the HGGD/RNG feature lattice."""
import importlib.util
import sys
from types import MethodType

from torch import nn

from .image_pyramid import ImagePyramid


SCANS = {'cross2d': 'v05', 'unidirectional': 'v051d',
         'bidirectional': 'v052d', 'cascade2d': 'v052dc'}


def checkpointed_block(block, value):
    import torch
    from torch.utils.checkpoint import checkpoint
    if block.training and torch.is_grad_enabled():
        return checkpoint(block._forward, value, use_reentrant=False)
    return block._forward(value)


def native_module():
    from ..config import ROOT
    from ..compat import triton_driver
    from timm.layers import DropPath
    try:
        import selective_scan_cuda_oflex
    except ImportError as error:
        raise ValueError('VMamba CUDA selective scan is missing; run ./panda install') from error
    triton_driver()
    name = '_grasppanda_vmamba'
    if name not in sys.modules:
        path = ROOT / 'environments/sources/cv/vmamba/classification/models/vmamba.py'
        if not path.is_file():
            raise ValueError('VMamba source is missing; run ./panda install')
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        original_repr = DropPath.__repr__
        try:
            spec.loader.exec_module(module)
        except Exception:
            for key in list(sys.modules):
                if key == name or key.startswith(name + '.'):
                    sys.modules.pop(key, None)
            raise
        finally:
            DropPath.__repr__ = original_repr
    return sys.modules[name]


class VMambaPyramid(ImagePyramid):
    def __init__(self, stage_channels=(96, 192, 384, 768), stage_depths=(2, 2, 5, 2),
                 state_dim=1, ssm_ratio=2., dt_rank=None, scan='cross2d',
                 ssm_conv=3, ssm_conv_bias=False, ssm_activation='silu', ssm_dropout=0.,
                 mlp_ratio=4., mlp_activation='gelu', mlp_dropout=0.,
                 drop_path=.2, gradient_checkpointing=False, projection_norm='batch'):
        nn.Module.__init__(self)
        native = native_module()
        self.encoder = native.Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=None,
            in_chans=4, dims=list(stage_channels), depths=list(stage_depths),
            ssm_d_state=state_dim, ssm_ratio=ssm_ratio, ssm_dt_rank='auto' if dt_rank is None else dt_rank,
            ssm_act_layer=ssm_activation, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_dropout, forward_type=SCANS[scan]+'_noz',
            mlp_ratio=mlp_ratio, mlp_act_layer=mlp_activation, mlp_drop_rate=mlp_dropout,
            drop_path_rate=drop_path, use_checkpoint=False,
            norm_layer='ln2d', downsample_version='v3', patchembed_version='v2')
        if gradient_checkpointing:
            for stage in self.encoder.layers:
                for block in stage.blocks:
                    block.forward = MethodType(checkpointed_block, block)
        # Symmetrically padded odd kernels keep each stage's origin at pixel 0.
        self.configure_pyramid('vmamba', stage_channels, (4, 8, 16, 32), projection_norm, pad_to_stride=False)
