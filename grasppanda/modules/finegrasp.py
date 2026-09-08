"""FineGrasp's native local attention and group fusion for Graspness seeds."""
import torch
from torch import nn


class FineGraspCrop(nn.Module):
    def __init__(self,native,nsample=16,radius_factors=(.25,.5,.75,1.),**fusion_options):
        super().__init__()
        from ..finegrasp import native_module
        source=native_module()
        self.groups=nn.ModuleList(source.CylinderGroup(
            nsample=nsample,seed_feature_dim=native.in_dim,
            cylinder_radius=native.grouper.radius*factor,
            hmin=native.grouper.hmin,hmax=native.grouper.hmax)
            for factor in radius_factors)
        self.fusion=source.GroupTransformerFusion(feature_dim=256,num_groups=len(self.groups))
        configure_fusion(self.fusion, fusion_options)

    def forward(self,seed_xyz,seed_features,rotations):
        features=torch.stack([group(seed_xyz,seed_features,rotations) for group in self.groups],dim=1)
        return self.fusion(features)


def configure_fusion(fusion, options):
    """Keep native cross-radius attention and expose its Transformer architecture."""
    if not options: return False
    layer = nn.TransformerEncoderLayer(
        d_model=256, nhead=options.get('fusion_heads', 8),
        dim_feedforward=options.get('fusion_ffn_dim', 1024),
        dropout=options.get('fusion_dropout', .1),
        activation=options.get('fusion_activation', 'relu'),
        norm_first=options.get('fusion_pre_norm', False))
    fusion.transformer = nn.TransformerEncoder(layer, num_layers=options.get('fusion_layers', 2))
    return True
