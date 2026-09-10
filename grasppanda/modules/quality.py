"""Residual quality logits with native angular, width and coordinate contracts."""
import math

from torch import nn
from torch.nn import functional as F

from .layers import point_mlp


class QualityResidual(nn.Module):
    def __init__(self, native, method, hidden_channels=(128, 128), activation='relu',
                 normalization='batch', initial_probability=.1):
        super().__init__()
        self.method = method
        self.angles, self.depths = native.num_angle, native.num_depth
        self.hidden = point_mlp([256, *hidden_channels], activation=activation, normalization=normalization)
        outputs = self.angles * self.depths if method == 'graspness' else self.angles
        self.output = nn.Conv1d(hidden_channels[-1], outputs, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, math.log(initial_probability / (1 - initial_probability)))

    def forward(self, features):
        batch, _, seeds = features.shape[:3]
        correction = self.output(self.hidden(features.flatten(2)))
        if self.method == 'graspness':
            return correction.reshape(batch, self.angles, self.depths, seeds).permute(0, 3, 1, 2)
        return correction.reshape(batch, self.angles, seeds, self.depths)


def apply_quality(module, inputs, end):
    if not isinstance(end, dict) or 'grasp_score_pred' not in end:
        raise ValueError('Quality scoring requires the native recorded grasp endpoints')
    logits = end['grasp_score_pred'] + module.quality_residual(inputs[0])
    end['grasp_score_logits'] = logits
    # Graspness labels are normalized; dense Baseline labels are log qualities.
    # softplus(z) is the stable inverse of q = 1 - exp(-native_score).
    end['grasp_score_pred'] = logits.sigmoid() if module.quality_residual.method == 'graspness' else F.softplus(logits)
    return end


def install(native, method, options):
    if hasattr(native, 'quality_residual'):
        raise ValueError('Quality scoring is already configured on this head')
    native.add_module('quality_residual', QualityResidual(native, method, **options))
    native.register_forward_hook(apply_quality)
