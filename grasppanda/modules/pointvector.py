"""Native PointVector segmentation features with original-input grasp seeds."""
import importlib

from torch import nn

from .pointnext import PointNeXtBackbone, native_module


class PointVectorBackbone(PointNeXtBackbone):
    family = 'PointVector'

    def __init__(self, width=32, blocks=(1,3,5,3,3), nsample=32, local_nsample=8,
                 radius=.05, radius_scaling=2., normalize_dp=True,
                 sa_layers=1, sa_use_res=False, decoder_layers=2):
        nn.Module.__init__(self)
        from easydict import EasyDict
        native_module()
        native = importlib.import_module('openpoints.models.backbone.pointvector')
        self.encoder = native.PointVectorEncoder(in_channels=3, width=width,
            blocks=list(blocks), strides=[1,4,4,4,4], nsample=nsample,
            radius=radius, radius_scaling=radius_scaling, flag=1,
            sa_layers=sa_layers, sa_use_res=sa_use_res,
            group_args=EasyDict(NAME='ballquery', normalize_dp=normalize_dp),
            conv_args={}, aggr_args={'feature_type':'dp_fj','reduction':'max'},
            norm_args={'norm':'bn'}, act_args={'act':'relu'})
        # Native vector blocks hard-code eight neighbors during construction.
        # Alter their query size only; keep vector rotations and sum reduction.
        for module in self.encoder.modules():
            if isinstance(module, native.LocalAggregation):
                module.grouper.nsample = local_nsample
        self.decoder = native.PointVectorDecoder(self.encoder.channel_list.copy(),
                                                 decoder_layers=decoder_layers)
        self.projection = nn.Conv1d(self.decoder.out_channels, 256, 1)
