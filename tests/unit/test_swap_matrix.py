"""The swap matrix: every combination of registered stages either works or is refused.

Interchangeable modules are the reason this toolbox exists, so the claim is tested as a
whole, on tiny synthetic data and on the CPU:

* **encoder × method**: every registered scene encoder under every seed-based method.
  Where the contracts allow the pair, the detector builds, runs ``loss()`` +
  ``backward()`` and ``predict()``. Where they forbid it, the config check says so,
  names the adapter that bridges the two, and the adapted config then runs;
* **grouper × head bundle** and **proposal × encoder kind**: the same, pair by pair;
* **all stage sets**: the full product of detector types, encoder kinds, proposals, view
  stages, assigners, groupers and head bundles (with the option variants that change a
  verdict). No combination may pass the config check and fail afterwards.

The tables below hold one tiny configuration per registered component. A component
added to a registry without an entry here fails ``test_every_registered_stage_is_in_the
_matrix``, which is the point: it cannot be claimed interchangeable until it has been run
in the matrix. ``docs/advanced_guides/swapping.md`` shows the resulting tables; they are
generated from the same contract rules and compared with this module's verdicts.
"""

from __future__ import annotations

import importlib.util
import itertools
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from grasppanda.cli.components import collect
from grasppanda.combine.synthetic import random_batch
from grasppanda.config import Config, ConfigError
from grasppanda.models.base import Backbone
from grasppanda.models.detectors.seed_based import SeedGraspDetector, stage_contract_problems
from grasppanda.registry import (
    BACKBONES,
    DETECTORS,
    GROUPERS,
    HEADS,
    PROPOSALS,
    component_parameters,
)
from grasppanda.structures import ContractError, GraspGroup, Observation, Supervision

from .conftest import REPO

HAS_SPCONV = importlib.util.find_spec("spconv") is not None
needs_spconv = pytest.mark.skipif(not HAS_SPCONV, reason="needs the spconv extra")
needs_ocnn = pytest.mark.skipif(
    importlib.util.find_spec("ocnn") is None, reason="needs the ocnn extra"
)
#: PyGConvGroup is a bridge to PyTorch Geometric; without the library it has no variant
#: in the matrix (its own tests are in tests/unit/test_grouper_options.py).
HAS_PYG = importlib.util.find_spec("torch_geometric") is not None
needs_timm = pytest.mark.skipif(
    importlib.util.find_spec("timm") is None, reason="needs the timm extra"
)
#: The selective scan of mamba-ssm is a CUDA kernel and these tests run on the CPU; the
#: encoder is run under a method by ``tests/unit/test_backbone_point_tokens.py`` on a GPU.
needs_cuda_scan = pytest.mark.skip(reason="PointMamba runs on CUDA only (mamba-ssm)")
needs_pyg = pytest.mark.skipif(
    importlib.util.find_spec("torch_geometric") is None, reason="needs the torch_geometric extra"
)

VIEW_COUNT, ANGLES, DEPTHS, SEEDS, POINTS, FRAMES = 10, 3, 2, 12, 96, 2
#: Channels of every tiny encoder: Scale-Balanced-Grasp adds seed features to crop
#: features, so the two widths must be equal there. A multiple of 6 per attention head,
#: which the rotary embedding of PTv3Utonia needs.
WIDTH = 12


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Tiny inputs gain nothing from intra-op threads and lose a lot on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ------------------------------------------------------------- one tiny config each

_SA = {
    "npoints": [48, SEEDS, 8, 4],
    "radii": [0.2, 0.4, 0.6, 0.9],
    "nsamples": [8, 8, 4, 4],
    "sa_mlps": [[8, 8], [8, 16], [16, 16], [16, 16]],
}
_SA_MSG = {
    "npoints": [48, SEEDS, 8, 4],
    "radii": [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.9, 1.0]],
    "nsamples": [[6, 8], [6, 8], [4, 4], [4, 4]],
    "sa_mlps": [[[8], [8]], [[8], [8]], [[8], [8]], [[8], [8]]],
}
_PTV3 = {
    "in_channels": 3,
    "grid_size": 0.05,
    "stride": [2, 2],
    "enc_depths": [1, 1, 1],
    "enc_channels": [12, 24, 48],
    "enc_num_head": [2, 4, 8],
    "enc_patch_size": [16, 16, 16],
}
_PTV3_DECODER = {
    "dec_depths": [1, 1],
    "dec_channels": [WIDTH, 24],
    "dec_num_head": [2, 4],
    "dec_patch_size": [16, 16],
}

#: Registered scene encoders (adapters are listed separately), each with a tiny config
#: that returns ``WIDTH`` channels and, when sub-sampling, ``SEEDS`` points per cloud.
ENCODERS: dict[str, tuple[dict[str, Any], tuple[Any, ...]]] = {
    "SparseUNet14": (
        {
            "in_channels": 3,
            "out_channels": WIDTH,
            "voxel_size": 0.05,
            "stem_channels": 8,
            "encoder_channels": [8, 8, 16, 16],
            "decoder_channels": [16, 16, 8, 8],
            "stem_kernel": 3,
        },
        (needs_spconv,),
    ),
    "PTv3": ({**_PTV3, **_PTV3_DECODER}, (needs_spconv,)),
    "PTv3Sonata": (
        {**_PTV3, **_PTV3_DECODER, "enc_mode": False, "mask_token": False},
        (needs_spconv,),
    ),
    "PTv3Utonia": (
        {**_PTV3, **_PTV3_DECODER, "enc_mode": False, "mask_token": False},
        (needs_spconv,),
    ),
    "LitePT": (  # the published layout in small: convolve fine, attend coarse, unpool
        {
            **_PTV3,
            "enc_depths": [1, 1, 2],
            "enc_conv": [True, True, False],
            "enc_attn": [False, False, True],
            "enc_rope_freq": [100.0, 100.0, 100.0],
            "dec_depths": [0, 0],
            "dec_channels": [WIDTH, 24],
            "dec_num_head": [2, 4],
            "dec_patch_size": [16, 16],
            "dec_conv": [False, False],
            "dec_attn": [False, False],
            "dec_rope_freq": [100.0, 100.0],
        },
        (needs_spconv,),
    ),
    "OACNNs": (
        {
            "in_channels": 3,
            "out_channels": WIDTH,
            "grid_size": 0.05,
            "embed_channels": 8,
            "enc_channels": [8, 12],
            "enc_depth": [1, 2],
            "dec_channels": [8, 12],
            "point_grid_size": [[2, 4], [2, 3]],
        },
        (needs_spconv,),
    ),
    "SpUNet": (
        {
            "in_channels": 3,
            "out_channels": WIDTH,
            "grid_size": 0.05,
            "base_channels": 6,
            "channels": [8, 12, 10, 8],
            "layers": [1, 2, 1, 1],
        },
        (needs_spconv,),
    ),
    "MinkUNet": (
        {
            "in_channels": 3,
            "out_channels": WIDTH,
            "voxel_size": 0.05,
            "layers": [1, 1, 1, 1, 1, 1, 1, 1],
            "planes": [6, 8, 8, 10, 8, 8, 6, 6],
            "init_dim": 6,
        },
        (needs_spconv,),
    ),
    "PointNet2Dense": ({**_SA, "in_channels": 3, "fp_mlps": [[16], [16], [16], [WIDTH]]}, ()),
    "PointNeXt": (
        {
            "in_channels": 3,
            "width": WIDTH,
            "strides": [1, 2, 2],
            "blocks": [1, 1, 1],
            "radius": 0.2,
            "nsample": 8,
        },
        (),
    ),
    "PointTransformerV1": (
        {
            "in_channels": 3,
            "planes": [WIDTH, 16, 24],
            "blocks": [1, 1, 1],
            "strides": [1, 2, 2],
            "nsamples": [4, 8, 8],
            "share_planes": 4,
        },
        (),
    ),
    "PointTransformerV2": (
        {
            "in_channels": 3,
            "patch_embed_channels": WIDTH,
            "patch_embed_groups": 3,
            "patch_embed_neighbours": 4,
            "enc_depths": [1, 1],
            "enc_channels": [16, 24],
            "enc_groups": [4, 4],
            "enc_neighbours": [8, 8],
            "dec_depths": [1, 1],
            "dec_channels": [WIDTH, 16],
            "dec_groups": [3, 4],
            "dec_neighbours": [8, 8],
            "grid_sizes": [0.15, 0.4],
        },
        (),
    ),
    "Volt": (
        {
            "in_channels": 3,
            "up_mlp_dim": WIDTH,
            "grid_size": 0.05,
            "kernel_size": 2,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 2,
            "rope_split": [1, 2, 1],
        },
        (),
    ),
    "OctFormer": (
        {
            "in_channels": 3,
            "out_channels": WIDTH,
            "voxel_size": 0.05,
            "octree_depth": 6,
            "channels": [8, 16],
            "num_blocks": [2, 2],
            "num_heads": [2, 4],
            "patch_size": 4,
            "dilation": 2,
            "drop_path": 0.0,
            "fpn_channel": 8,
        },
        (needs_ocnn,),
    ),
    "PointNet2": ({**_SA, "fp_mlps": [[16, 16], [16, WIDTH]]}, ()),
    "PointViT": (
        {
            "num_group": SEEDS,
            "group_size": 8,
            "embed_dim": WIDTH,
            "depth": 1,
            "num_heads": 2,
            "token_dim": WIDTH,
        },
        (),
    ),
    "Uni3D": (
        {
            "out_channels": WIDTH,
            "num_group": SEEDS,
            "group_size": 8,
            "timm_model": "eva02_tiny_patch14_224",
            "token_dim": 16,
        },
        (needs_timm,),
    ),
    "PointMamba": (
        {"num_group": SEEDS, "group_size": 8, "embed_dim": WIDTH, "depth": 1, "token_dim": WIDTH},
        (needs_cuda_scan,),
    ),
    "DGCNN": (
        {"in_channels": 3, "k": 6, "width": 8, "emb_dims": 16, "out_channels": WIDTH},
        (),
    ),
    "ASSANet": (
        {
            "in_channels": 3,
            "use_xyz": True,  # the methods feed constant ones: see the class
            "width": 6,
            "strides": [2, 2],
            "radii": [[0.2, 0.3], [0.4, 0.5]],
            "nsamples": [[6, 8], [6, 6]],
            "mlps": [[[6, 6, 9], [9, 9, 12]], [[12, 12, 12], [12, 12, 18]]],
            "fp_mlps": [[WIDTH, WIDTH], [16, 16]],
        },
        (),
    ),
    "DeLA": (
        {
            "in_channels": 3,
            "grid_size": [0.1, 0.3],
            "k": [6, 6],
            "depths": [1, 2],
            "dims": [8, 12],
            "nbr_dims": [8, 8],
            "head_dim": WIDTH,
        },
        (),
    ),
    "PointMLP": (
        {
            "in_channels": 3,
            "embed_dim": 8,
            "dim_expansion": [2, 1],
            "pre_blocks": [1, 1],
            "pos_blocks": [1, 1],
            "k_neighbors": [6, 4],
            "reducers": [2, 2],
            "de_dims": [16, 12],
            "de_blocks": [1, 1],
            "gmp_dim": 8,
            "out_channels": WIDTH,
        },
        (),
    ),
    "RandLANet": (
        {
            "in_channels": 3,
            "num_neighbors": 4,
            "sub_sampling_ratio": [2, 2],
            "dim_features": 4,
            "dim_output": [4, 8],
            "out_channels": WIDTH,
        },
        (),
    ),
    "PyGPointEncoder": (
        {"in_channels": 3, "conv": "pointnet", "channels": [8, 8], "out_channels": WIDTH, "k": 6},
        (needs_pyg,),
    ),
    "PointMetaBase": (
        {
            "in_channels": 3,
            "width": WIDTH,
            "strides": [1, 2, 2],
            "blocks": [1, 2, 1],
            "radius": 0.2,
            "nsample": 8,
        },
        (),
    ),
    "PointVector": (
        {
            "in_channels": 3,
            "width": WIDTH,
            "strides": [1, 2, 2],
            "blocks": [1, 2, 1],
            "radius": 0.2,
            "nsample": 8,
            "block_nsample": 4,
        },
        (),
    ),
    "PointNet2MSG": ({**_SA_MSG, "fp_mlps": [[16, 16], [16, WIDTH]]}, ()),
    "PointNet2MSGDense": (
        {**_SA_MSG, "in_channels": 3, "fp_mlps": [[16], [16], [16], [WIDTH]]},
        (),
    ),
    "PointNet2Transformer": (
        {
            "npoints": [24, SEEDS],
            "radii": [0.4, 0.8],
            "nsamples": [6, 5],
            "sa_mlps": [[8, 8], [8, WIDTH]],
            "attention_neighbours": 4,
        },
        (),
    ),
}
#: Encoder adapters: output kind they produce -> config around an inner backbone node.
ADAPTERS = {
    "dense": lambda inner: {"type": "FeaturePropagation", "backbone": inner, "mlp": [WIDTH]},
    "subsampled": lambda inner: {
        "type": "FurthestPointSubsample",
        "backbone": inner,
        "npoint": SEEDS,
    },
}

_CYLINDER = {"radius": 0.3, "hmin": -0.1, "nsample": 4}
STAGES: dict[str, dict[str, dict[str, Any]]] = {
    "proposal": {
        "GraspnessFPS": {"type": "GraspnessFPS", "num_seeds": SEEDS},
        "SubsampledSeeds": {"type": "SubsampledSeeds", "num_seeds": SEEDS},
        # The two proposals with a sampler slot, each with a sampler that reads what
        # only that kind of proposal can give it (scores, features).
        "ScoredSeeds": {
            "type": "ScoredSeeds",
            "num_seeds": SEEDS,
            "sampler": {"type": "ScoreWeightedFPS"},
        },
        # Graspness only: objectness is left to a view stage that owns it (W4.6).
        "ScoredSeeds+graspness_only": {
            "type": "ScoredSeeds",
            "num_seeds": SEEDS,
            "select_by": "graspness_only",
        },
        "SampledSeeds": {
            "type": "SampledSeeds",
            "num_seeds": SEEDS,
            "sampler": {"type": "FeatureFPS"},
        },
    },
    "view": {
        "TemplateViewStage": {"type": "TemplateViewStage", "num_views": VIEW_COUNT},
        "ObjectnessViewStage": {"type": "ObjectnessViewStage", "num_views": VIEW_COUNT},
        # A prediction lattice of 6 views over the 10 views of the labels.
        "LatticeViewStage": {"type": "LatticeViewStage", "num_views": 6, "label_views": VIEW_COUNT},
        "DirectionViewStage": {"type": "DirectionViewStage", "label_views": VIEW_COUNT},
        "TwoVectorViewStage": {
            "type": "TwoVectorViewStage",
            "label_views": VIEW_COUNT,
            "num_angle": ANGLES,
        },
    },
    "assigner": {
        "NearestLabelAssigner": {"type": "NearestLabelAssigner"},
        "NearestLabelAssigner+object_seeds": {
            "type": "NearestLabelAssigner",
            "crop_centre": "label_point",
            "supervise": "object_seeds",
            "view_widths": True,
        },
        "CompactLabelAssigner": {"type": "CompactLabelAssigner", "max_match_distance": 0.08},
        "ScaleBalancedAssigner": {"type": "ScaleBalancedAssigner"},
    },
    "grouper": {
        "CylinderGroup": {
            "type": "CylinderGroup",
            **_CYLINDER,
            "hmax": 0.2,
            "mlp_channels": [8, WIDTH],
        },
        "CylinderInteractionGroup": {
            "type": "CylinderInteractionGroup",
            **_CYLINDER,
            "hmax": 0.2,
            "mlp_channels": [8, 8],
            "post_mlp_channels": [8, WIDTH],
        },
        "MultiScaleGroup": {
            "type": "MultiScaleGroup",
            "radius": _CYLINDER["radius"],
            "radius_scales": [0.5, 1.0],
            "fusion_layers": 1,
            "fusion_heads": 2,
            "fusion_feedforward_channels": 16,
            "branch": {  # the radius of a branch is set by the scales above
                "type": "CylinderInteractionGroup",
                "hmin": _CYLINDER["hmin"],
                "hmax": 0.2,
                "nsample": _CYLINDER["nsample"],
                "mlp_channels": [8, 8],
                "post_mlp_channels": [8, WIDTH],
            },
        },
        "DepthCylinderGroup": {
            "type": "DepthCylinderGroup",
            **_CYLINDER,
            "hmax": [0.1, 0.25],
            "mlp_channels": [8, WIDTH],
        },
        "DepthCylinderGroup+scales+gate": {
            "type": "DepthCylinderGroup",
            **_CYLINDER,
            "hmax": [0.1, 0.25],
            "mlp_channels": [8, WIDTH],
            "radius_scales": [0.5, 1.0],
            "seed_residual": "gated",
        },
        "LocalGroup": {
            "type": "LocalGroup",
            "shape": "box",
            **_CYLINDER,
            "hmax": 0.2,
            "encoder": "edge",
            "pool": "attentive",
            "mlp_channels": [8, WIDTH],
        },
        "DepthLocalGroup": {
            "type": "DepthLocalGroup",
            "shape": "ball",
            **_CYLINDER,
            "hmax": [0.1, 0.25],
            "mlp_channels": [8, WIDTH],
            "seed_residual": "projected",
        },
        "RoIGridGroup": {
            "type": "RoIGridGroup",
            "radius": 0.3,
            "hmin": -0.1,
            "hmax": 0.2,
            "grid_size": 2,
            "pool_radius": 0.3,
            "nsample": 4,
            "mlp_channels": [8],
            "fc_channels": [WIDTH],
        },
        "RoIAwareGroup": {
            "type": "RoIAwareGroup",
            "radius": 0.3,
            "hmin": -0.1,
            "hmax": 0.2,
            "grid_size": 2,
            "nsample": 4,
            "cell_channels": 4,
            "fc_channels": [WIDTH],
        },
        "VoxelSetAbstractionGroup": {
            "type": "VoxelSetAbstractionGroup",
            "radii": [0.2, 0.4],
            "nsample": [4, 4],
            "mlp_channels": [8],
            "out_channels": WIDTH,
        },
        "DepthEmbedGroup": {
            "type": "DepthEmbedGroup",
            "num_depth": DEPTHS,
            "grouper": {
                "type": "CylinderGroup",
                **_CYLINDER,
                "hmax": 0.2,
                "mlp_channels": [8, WIDTH],
            },
        },
        "DepthPoolGroup": {
            "type": "DepthPoolGroup",
            "grouper": {
                "type": "DepthCylinderGroup",
                **_CYLINDER,
                "hmax": [0.1, 0.25],
                "mlp_channels": [8, WIDTH],
            },
        },
        "MultiRadiusGroup": {
            "type": "MultiRadiusGroup",
            "radius": _CYLINDER["radius"],
            "radius_scales": [0.5, 1.0],
            "fusion": "linear",
            "branch": {
                "type": "CylinderGroup",
                "hmin": _CYLINDER["hmin"],
                "hmax": 0.2,
                "nsample": _CYLINDER["nsample"],
                "mlp_channels": [8, WIDTH],
            },
        },
        "DepthCylinderProjectionGroup": {
            "type": "DepthCylinderProjectionGroup",
            **_CYLINDER,
            "hmax": [0.1, 0.25],
            "mlp_channels": [8, 8],  # narrower than the encoder: the projection maps it
            "radius_scales": [0.5, 1.0],
        },
        "PyGConvGroup": {  # built only where the library is installed (HAS_PYG)
            "type": "PyGConvGroup",
            "conv": "PointTransformerConv",
            **_CYLINDER,
            "hmax": 0.2,
            "hidden_channels": 8,
            "out_channels": WIDTH,
        },
    },
    "head": {
        "SWADHead": {"type": "SWADHead", "num_angle": ANGLES, "num_depth": DEPTHS},
        "EconomicHead": {
            "type": "EconomicHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "token_channels": 8,
        },
        "OperationHead": {
            "type": "OperationHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "hidden_channels": 8,
        },
        "OperationHead+objectness": {
            "type": "OperationHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "hidden_channels": 8,
            "score_supervision": "seeds_with_a_supervised_depth",
            "score_decode": "score_x_objectness_x_tolerance",
        },
        "DecoupledGridHead": {
            "type": "DecoupledGridHead",
            "num_angle": ANGLES,
            "num_depth": DEPTHS,
            "hidden_channels": 8,
        },
    },
}

#: Stages that are bridges to an optional library: slot -> name -> (node, module). They
#: are not part of the product above, whose counts must not depend on what is installed;
#: ``tests/unit/test_view_options.py`` runs them under every method where the library is.
BRIDGE_STAGES: dict[str, dict[str, tuple[dict[str, Any], str]]] = {
    "view": {
        "IcosphereViewStage": (
            {"type": "IcosphereViewStage", "subdivisions": 0, "label_views": VIEW_COUNT},
            "trimesh",
        ),
        "HealpixViewStage": (
            {"type": "HealpixViewStage", "nside": 1, "label_views": VIEW_COUNT},
            "healpix",
        ),
        "RomaRotationViewStage": (
            {"type": "RomaRotationViewStage", "label_views": VIEW_COUNT, "num_angle": ANGLES},
            "roma",
        ),
    },
}

#: The seed-based methods: detector type -> the variant of each stage its reference
#: model uses (``configs/_base_/models/<method>.yaml``, checked below).
METHODS: dict[str, dict[str, str]] = {
    "GSNet": {
        "proposal": "GraspnessFPS",
        "view": "TemplateViewStage",
        "assigner": "NearestLabelAssigner",
        "grouper": "CylinderGroup",
        "head": "SWADHead",
    },
    "EconomicGrasp": {
        "proposal": "GraspnessFPS",
        "view": "TemplateViewStage",
        "assigner": "CompactLabelAssigner",
        "grouper": "CylinderInteractionGroup",
        "head": "EconomicHead",
    },
    "GraspNetBaseline": {
        "proposal": "SubsampledSeeds",
        "view": "ObjectnessViewStage",
        "assigner": "NearestLabelAssigner+object_seeds",
        "grouper": "DepthCylinderGroup",
        "head": "OperationHead",
    },
    "ScaleBalancedGrasp": {
        "proposal": "SubsampledSeeds",
        "view": "ObjectnessViewStage",
        "assigner": "ScaleBalancedAssigner",
        "grouper": "DepthCylinderGroup+scales+gate",
        "head": "OperationHead+objectness",
    },
}


def encoder(name: str) -> dict[str, Any]:
    return {"type": name, **ENCODERS[name][0]}


def kind_of(node: dict[str, Any]) -> str:
    return BACKBONES.get(node["type"]).output


def method(detector: str, backbone: dict[str, Any], **replace: str) -> dict[str, Any]:
    """The tiny reference model of a method around ``backbone``; ``replace`` names
    other variants of single stages."""
    chosen = {**METHODS[detector], **replace}
    return {
        "type": detector,
        "backbone": backbone,
        **{slot: dict(STAGES[slot][variant]) for slot, variant in chosen.items()},
    }


# ---------------------------------------------------------------- one synthetic batch


def batch(seed: int = 0) -> tuple[Observation, Supervision]:
    """Two frames with everything any registered stage reads: dense grids with
    tolerance, compact labels, per-point objectness and graspness, a width histogram.

    The generator lives in the package, where ``grasppanda probe`` uses it too."""
    return random_batch(
        frames=FRAMES, points=POINTS, views=VIEW_COUNT, angles=ANGLES, depths=DEPTHS, seed=seed
    )


def refusal(node: dict[str, Any]) -> list[str]:
    """Messages of the config check for a model node; empty when it is accepted."""
    try:
        Config.from_dict({"model": node})
    except ConfigError as error:
        assert all(issue.path == "model" for issue in error.issues), error.issues
        return [issue.message for issue in error.issues]
    return []


def buildable(node: dict[str, Any]) -> bool:
    """Whether the optional library of every stage of the model is installed."""
    grouper = node["grouper"]
    inner = grouper.get("grouper") if isinstance(grouper.get("grouper"), dict) else {}
    return HAS_PYG or "PyGConvGroup" not in (grouper["type"], inner.get("type"))


def built(node: dict[str, Any]) -> SeedGraspDetector:
    """The detector of a model node, seeded, with a proposal that passes most points."""
    torch.manual_seed(0)
    model = DETECTORS.build(node)
    if hasattr(model.proposal, "score"):  # make most points candidates: real seeds everywhere
        with torch.no_grad():
            model.proposal.score.weight.mul_(0.01)
            bias = model.proposal.score.bias  # [objectness logits,] graspness
            bias.copy_(torch.tensor([0.0, 1.0, 1.0])[-len(bias) :])
    return model


def run(node: dict[str, Any]) -> SeedGraspDetector:
    """Build an accepted model, take one training step's worth of gradients, predict."""
    assert refusal(node) == []
    if not buildable(node):
        pytest.skip("needs the torch_geometric extra")
    model = built(node)
    observation, supervision = batch()
    model.train()
    losses = model.loss(observation, supervision)
    terms = {key: value for key, value in losses.items() if key.startswith("loss/")}
    assert terms and all(torch.isfinite(value) for value in losses.values()), losses
    sum(terms.values()).backward()
    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(bool(g.any()) for g in gradients), "no gradient reaches the scene encoder"
    model.eval()
    grasps = model.predict(observation)
    assert len(grasps) == FRAMES and all(isinstance(group, GraspGroup) for group in grasps)
    for group, key in zip(grasps, observation.keys):
        group.validate()
        assert group.key == key and len(group) <= SEEDS
    return model


def registered(registry_name: str) -> list[str]:
    """In-tree components of a registry (other test modules register their own)."""
    return [entry.name for entry in collect([registry_name], in_tree_only=True)]


# ------------------------------------------------------- the tables cover the registries


def test_every_registered_stage_is_in_the_matrix():
    adapters = {ADAPTERS[kind]({"type": "x"})["type"] for kind in ADAPTERS}
    # Point painting wraps an encoder and reads the colour image, which the clouds of this
    # matrix do not have; it is covered by tests/unit/test_backbone_image_painting.py.
    painted = {"ImagePainted", "ImagePaintedSubsampled"}
    assert set(registered("BACKBONES")) == set(ENCODERS) | adapters | painted
    for slot, registry in (
        ("proposal", "PROPOSALS"),
        ("view", "VIEWS"),
        ("assigner", "ASSIGNERS"),
        ("grouper", "GROUPERS"),
        ("head", "HEADS"),
    ):
        bridges = set(BRIDGE_STAGES.get(slot, {}))
        assert {node["type"] for node in STAGES[slot].values()} | bridges == set(
            registered(registry)
        ), slot
    seed_based = {
        name
        for name in registered("DETECTORS")
        if issubclass(DETECTORS.get(name), SeedGraspDetector)
    }
    assert seed_based == set(METHODS)


#: Shipped reference models that are a method's detector with other stage variants.
VARIANT_MODELS: dict[str, dict[str, str]] = {
    "finegrasp.yaml": {"grouper": "MultiScaleGroup"},  # EconomicGrasp with crops at several radii
}


def test_the_tiny_methods_use_the_stages_of_the_shipped_reference_models():
    for path in sorted((REPO / "configs" / "_base_" / "models").glob("*.yaml")):
        model = Config.load(path)["model"]
        if model["type"] not in METHODS:
            continue
        tiny = method(model["type"], {"type": "PointNet2"}, **VARIANT_MODELS.get(path.name, {}))
        for slot in STAGES:
            assert tiny[slot]["type"] == model[slot]["type"], (path.name, slot)
        assert BACKBONES.get(model["backbone"]["type"]).output == (
            PROPOSALS.get(model["proposal"]["type"]).requires
        )


# ------------------------------------------------------------------ encoder × method


def encoder_method_cases() -> list[Any]:
    return [
        pytest.param(name, detector, id=f"{name}-{detector}", marks=ENCODERS[name][1])
        for name in ENCODERS
        for detector in METHODS
    ]


@pytest.mark.parametrize(("name", "detector"), encoder_method_cases())
def test_encoder_under_method(name, detector):
    """Legal pairs run. Illegal pairs are refused with the adapter by name, and the
    adapted config runs: every encoder can serve every method, never implicitly."""
    backbone = encoder(name)
    wanted = PROPOSALS.get(STAGES["proposal"][METHODS[detector]["proposal"]]["type"]).requires
    node = method(detector, backbone)
    if kind_of(backbone) == wanted:
        model = run(node)
        assert type(model.backbone).__name__ == name and model.backbone.out_channels == WIDTH
        return
    (message,) = refusal(node)
    adapted = ADAPTERS[wanted](backbone)
    proposal = node["proposal"]["type"]
    assert f"{proposal} requires '{wanted}' scene features" in message
    assert f"{name} returns '{kind_of(backbone)}' ones" in message
    assert f"{{type: {adapted['type']}, backbone: {{...}}}}" in message  # the fix, by name
    with pytest.raises(ContractError, match=f"requires '{wanted}' scene features"):
        DETECTORS.build(node)  # the Python API refuses it too
    model = run(method(detector, adapted))
    assert type(model.backbone.backbone).__name__ == name and model.backbone.output == wanted


# ------------------------------------------------------------- grouper × head bundle


@pytest.mark.parametrize("head", list(STAGES["head"]))
@pytest.mark.parametrize("grouper", list(STAGES["grouper"]))
def test_grouper_under_head(grouper, head):
    """Around a fixed front end, a crop and a head bundle combine iff their axes agree."""
    head_cls = HEADS.get(STAGES["head"][head]["type"])
    grouper_cls = GROUPERS.get(STAGES["grouper"][grouper]["type"])
    assigner = (
        "CompactLabelAssigner"
        if head_cls.label_format == "compact"
        else "NearestLabelAssigner+object_seeds"
    )
    node = method("GSNet", encoder("PointNet2Dense"), assigner=assigner, grouper=grouper, head=head)
    if grouper_cls.out_axes == head_cls.in_axes:
        run(node)
        return
    (message,) = refusal(node)
    assert f"{grouper_cls.__name__} emits features with axes {grouper_cls.out_axes}" in message
    assert f"{head_cls.__name__} expects {head_cls.in_axes}" in message

    def names(listing: str) -> set[str]:
        return set(listing.split("): ")[1].rstrip(".").split(", "))

    fitting_groupers = names(message.split("Groupers that emit")[1].split(". Head bundles")[0])
    fitting_heads = names(message.split("Head bundles that expect")[1])
    for other in registered("GROUPERS"):
        assert (other in fitting_groupers) == (GROUPERS.get(other).out_axes == head_cls.in_axes)
    for other in registered("HEADS"):
        assert (other in fitting_heads) == (HEADS.get(other).in_axes == grouper_cls.out_axes)


# ------------------------------------------------------------ proposal × encoder kind


@pytest.mark.parametrize("backbone", ["PointNet2Dense", "PointNet2"])
@pytest.mark.parametrize("proposal", list(STAGES["proposal"]))
def test_proposal_under_encoder_kind(proposal, backbone):
    produced = BACKBONES.get(backbone).output
    kind = STAGES["proposal"][proposal]["type"]
    required = PROPOSALS.get(kind).requires
    node = method("GSNet", encoder(backbone), proposal=proposal)
    if produced == required:
        model = run(node)
        assert model.proposal.requires == model.backbone.output
        return
    (message,) = refusal(node)
    assert f"{kind} requires '{required}' scene features" in message
    assert f"{backbone} returns '{produced}' ones" in message
    assert ADAPTERS[required]({})["type"] in message


# ---------------------------------------------------------------------- all stage sets


def stage_sets() -> Iterator[dict[str, str]]:
    slots = list(STAGES)
    for kind in ("dense", "subsampled"):
        for combination in itertools.product(*(STAGES[slot] for slot in slots)):
            yield {"kind": kind, **dict(zip(slots, combination))}


def node_of(detector: str, chosen: dict[str, str]) -> dict[str, Any]:
    backbone = encoder("PointNet2Dense" if chosen["kind"] == "dense" else "PointNet2")
    return method(detector, backbone, **{slot: chosen[slot] for slot in STAGES})


def test_no_stage_set_passes_the_config_check_and_fails_later():
    """The systematic claim. Before the contracts were lifted into the skeleton, 168 of
    1536 combinations passed the check and failed at the first loss, and 64 failed only
    at build time; a further family trained while ignoring its scale-balanced weights."""
    legal, refused = [], 0
    for chosen in stage_sets():
        verdicts = {detector: bool(refusal(node_of(detector, chosen))) for detector in METHODS}
        assert len(set(verdicts.values())) == 1, (chosen, verdicts)  # the type does not matter
        if verdicts["GSNet"]:
            refused += 1
            if buildable(node_of("GSNet", chosen)):
                with pytest.raises(ContractError):  # the same verdict without the checker
                    DETECTORS.build(node_of("GSNet", chosen))
        else:
            legal.append(chosen)
    assert legal and refused
    for chosen in legal:
        for detector in ("GSNet", "ScaleBalancedGrasp"):  # the barest and the strictest type
            if buildable(node_of(detector, chosen)):
                run(node_of(detector, chosen))
    # Every refusal reads as a sentence about two named stages.
    for chosen in stage_sets():
        for message in refusal(node_of("GSNet", chosen)):
            assert message[0].isalpha() or message.startswith("the "), message
            assert message.rstrip().endswith("."), message
    assert len(legal) == LEGAL_STAGE_SETS, len(legal)


#: Stage sets the contracts allow among the variants above; the count is restated in
#: docs/advanced_guides/swapping.md together with the list, so it is pinned here.
LEGAL_STAGE_SETS = 2000


def test_what_the_contracts_refuse_among_plausible_looking_swaps():
    """One example per rule, as a user would meet it: a single replaced line."""
    dense, sub = encoder("PointNet2Dense"), encoder("PointNet2")
    cases = [
        # PV-RCNN's scene crop under the baseline's head: no adapter gives it a depth axis.
        (
            method("GraspNetBaseline", sub, grouper="VoxelSetAbstractionGroup"),
            "No axis adapter turns a scene-support crop",
        ),
        # GSNet with EconomicGrasp's head only: the labels do not fit.
        (method("GSNet", dense, head="EconomicHead"), "must agree on the label set"),
        # The baseline's view stage after GSNet's proposal: two owners of objectness.
        (
            method("GSNet", dense, view="ObjectnessViewStage"),
            "both predict and supervise the 'objectness'",
        ),
        # The baseline with GSNet's plain assigner: the view stage lacks two fields.
        (
            method("GraspNetBaseline", sub, assigner="NearestLabelAssigner"),
            "`supervise: object_seeds` and `view_widths: true`",
        ),
        # Scale-Balanced-Grasp's decoder without any predicted objectness.
        (
            method(
                "ScaleBalancedGrasp",
                sub,
                view="TemplateViewStage",
                assigner="NearestLabelAssigner",
                grouper="DepthCylinderGroup",
            ),
            "reads the seed score 'objectness' when it decodes",
        ),
        # A crop without a depth axis under a head that needs one.
        (
            method("GraspNetBaseline", sub, grouper="CylinderGroup"),
            "expects ('seed', 'depth', 'channel')",
        ),
    ]
    for node, fragment in cases:
        messages = refusal(node)
        assert any(fragment in message for message in messages), (fragment, messages)


def kinds(node: dict[str, Any]) -> list[str]:
    """The kind of every refusal of the config check for a model node."""
    try:
        Config.from_dict({"model": node})
    except ConfigError as error:
        return [issue.kind for issue in error.issues]
    return []


# ------------------------------------------------------- what an adapter or an option lifts
#
# The refusals below were limits of the toolbox, not of the methods (dev audit of
# 2026-10-14). `grasppanda compose` now inserts the adapter or sets the option, and says so
# (tests/unit/test_combine_compose.py); here each lifted stage set is built, trained one
# step and run to predictions, and what remains refused says why.


@pytest.mark.parametrize("detector", ["GSNet", "GraspNetBaseline"])
@pytest.mark.parametrize("grouper", list(STAGES["grouper"]))
def test_an_axis_mismatch_runs_in_its_axis_adapter_or_says_why(grouper, detector):
    """W4.2: a crop with another axis layout than the head reads is repeated over the depth
    bins (DepthEmbedGroup) or pooled over them (DepthPoolGroup)."""
    from grasppanda.models.detectors.seed_based import axis_adapters

    head = HEADS.get(STAGES["head"][METHODS[detector]["head"]]["type"])
    inner = dict(STAGES["grouper"][grouper])
    cls = GROUPERS.get(inner["type"])
    if cls.out_axes == head.in_axes:
        pytest.skip("the axes agree: test_grouper_under_head")
    backbone = encoder("PointNet2Dense" if detector == "GSNet" else "PointNet2")
    node = method(detector, backbone)
    node["grouper"] = inner
    fitting = axis_adapters(cls.out_axes, head.in_axes, cls.support)
    adapter_itself = getattr(cls, "inner_axes", None) is not None
    if adapter_itself or not fitting:
        assert kinds(node) == ["structural" if adapter_itself else "not-implemented"]
        return
    assert kinds(node) == ["fixed-by-compose"]
    (adapter,) = fitting
    node["grouper"] = {"type": adapter, "grouper": inner}
    if adapter == "DepthEmbedGroup":
        node["grouper"]["num_depth"] = DEPTHS
    model = run(node)
    assert type(model.grouper.grouper).__name__ == inner["type"]
    assert model.grouper.out_axes == head.in_axes


def _losses(model: SeedGraspDetector, matched: Any, out: Any) -> dict[str, float]:
    terms = model.head.loss(out.raw, matched)
    return {key: float(value) for key, value in terms.items() if key.startswith("loss/")}


@pytest.mark.parametrize("head", ["SWADHead", "DecoupledGridHead", "OperationHead", "EconomicHead"])
def test_every_head_applies_the_weight_of_a_seed_as_a_weighted_mean(head):
    """W4.3: a weight per seed weights every loss of the head bundle: uniform weights give
    the unweighted losses, other weights other losses."""
    import dataclasses

    detector = "EconomicGrasp" if head == "EconomicHead" else "GSNet"
    grouper = "DepthCylinderGroup" if head == "OperationHead" else METHODS[detector]["grouper"]
    node = method(detector, encoder("PointNet2Dense"), head=head, grouper=grouper)
    if head == "OperationHead":
        node["assigner"] = dict(STAGES["assigner"]["NearestLabelAssigner+object_seeds"])
    model = built(node)
    observation, supervision = batch()
    model.train()
    out = model.run_stages(observation, supervision)
    matched = out.matched
    assert matched.valid.any()
    plain = _losses(model, dataclasses.replace(matched, weight=None), out)
    ones = torch.ones(len(matched.valid))
    assert _losses(model, dataclasses.replace(matched, weight=ones * 3.0), out) == pytest.approx(
        plain
    )
    skewed = torch.linspace(0.1, 5.0, len(matched.valid))
    weighted = _losses(model, dataclasses.replace(matched, weight=skewed), out)
    assert weighted.keys() == plain.keys()
    assert any(abs(weighted[key] - plain[key]) > 1e-6 for key in plain), (weighted, plain)


@pytest.mark.parametrize("head", ["SWADHead", "DecoupledGridHead"])
def test_the_scale_balanced_assigner_trains_the_flat_heads(head):
    """W4.3/W4.7: Scale-Balanced-Grasp's weights under GSNet's view stage and a flat head,
    through the depth pooling adapter: no refusal, no warning, and the model runs."""
    node = method("ScaleBalancedGrasp", encoder("PointNet2"), view="TemplateViewStage", head=head)
    node["grouper"] = {"type": "DepthPoolGroup", "grouper": node["grouper"]}
    node["head"].pop("score_decode", None)
    from grasppanda.models.detectors.seed_based import stage_contract_warnings

    assert (
        stage_contract_warnings(**{slot: node[slot] for slot in ("view", "assigner", "head")}) == []
    )
    run(node)


def test_a_head_that_reads_a_score_nobody_predicts_decodes_without_it():
    """W4.4: Scale-Balanced-Grasp's decoder multiplies the seeds' objectness in; under a
    view stage that predicts none the head decodes as GraspNet-baseline does."""
    node = method(
        "ScaleBalancedGrasp",
        encoder("PointNet2"),
        view="TemplateViewStage",
        assigner="NearestLabelAssigner",
        grouper="DepthCylinderGroup",
    )
    assert kinds(node) == ["fixed-by-compose"]
    node["head"]["score_decode"] = "score_x_tolerance"
    model = run(node)
    assert model.head.score_kind == "predicted_quality_x_tolerance"


@pytest.mark.parametrize("detector", ["GSNet", "GraspNetBaseline", "ScaleBalancedGrasp"])
def test_a_graspness_proposal_leaves_objectness_to_the_view_stage_that_owns_it(detector):
    """W4.6: the scoring proposal predicts graspness only next to ObjectnessViewStage; the
    view stage owns objectness, and Scale-Balanced-Grasp's decoder reads it from there."""
    with_both = method(
        detector,
        encoder("PointNet2Dense"),
        proposal="ScoredSeeds",
        view="ObjectnessViewStage",
        assigner="NearestLabelAssigner+object_seeds",
    )
    assert kinds(with_both) == ["fixed-by-compose"]
    node = {**with_both, "proposal": dict(STAGES["proposal"]["ScoredSeeds+graspness_only"])}
    model = run(node)
    observation, supervision = batch()
    model.train()
    out = model.run_stages(observation, supervision)
    assert out.point_scores.objectness is None and model.proposal.score.out_features == 1
    assert set(model.proposal.loss(out.point_scores, supervision)) == {"loss/graspness"}
    assert set(out.seeds.scores) == {"graspness", "objectness"}  # the view stage's objectness
    # A proposal whose selection reads objectness cannot leave it: a genuine refusal.
    by_objectness = {**with_both["proposal"], "select_by": "objectness"}
    assert kinds({**with_both, "proposal": by_objectness}) == ["structural"]


def test_a_seed_width_mismatch_is_refused_with_the_fix_and_the_projection_runs():
    """A grouper that adds the seed features to its crops states the width it needs; the
    check compares it with what the encoder returns, building the encoder when its config
    does not say."""
    narrow = {**STAGES["grouper"]["DepthCylinderGroup+scales+gate"], "mlp_channels": [8, 8]}
    node = method("ScaleBalancedGrasp", encoder("PointNet2"))
    node["grouper"] = narrow
    (message,) = refusal(node)
    assert "adds the seed features to its 8-channel crop features" in message
    assert "PointNet2 returns 12-channel features" in message
    assert "`type: DepthCylinderProjectionGroup`" in message and "`mlp_channels` on 12" in message
    # Without the checker the grouper itself refuses, at build time, with the same ways out.
    with pytest.raises(
        ValueError, match=r"12-channel seed features to 8-channel crop features"
    ) as error:
        GROUPERS.build(dict(narrow), in_channels=WIDTH)
    assert "DepthCylinderProjectionGroup" in str(error.value) and "mlp_channels" in str(error.value)
    # An encoder that has no `out_channels` option is built to learn its width.
    wide = encoder("PointNet2Dense")
    assert "out_channels" not in wide
    (problem,) = stage_contract_problems(backbone=wide, grouper=narrow)
    assert "PointNet2Dense returns 12-channel features" in problem
    # The projected variant makes the pair legal, and the model trains and predicts.
    node["grouper"] = {
        **{key: value for key, value in narrow.items() if key != "seed_residual"},
        "type": "DepthCylinderProjectionGroup",
    }
    model = run(node)
    assert model.grouper.project.in_features == WIDTH and model.grouper.out_channels == 8


#: Encoders whose width is "known only once built" for the options named: the class
#: declares ``None`` there, and the config check builds the encoder to learn the width.
#: An entry needs the reason; an encoder whose width follows from its options declares it.
WIDTH_KNOWN_ONCE_BUILT: dict[str, str] = {
    "Uni3D": "without `out_channels` the width is that of the timm model named by `vit`",
}


def encoder_cases() -> list[Any]:
    return [pytest.param(name, id=name, marks=ENCODERS[name][1]) for name in ENCODERS]


@pytest.mark.parametrize("name", encoder_cases())
def test_an_encoder_returns_the_width_it_declares(name):
    """`output_channels` is `out_channels` told from the config: for the tiny encoder of
    the matrix, alone and inside each adapter, and for the constructor defaults. Where
    a class says "known only once built", the check learns the width by building."""
    from grasppanda.models.base import configured_width
    from grasppanda.models.detectors import seed_based

    node = encoder(name)
    built = BACKBONES.build(dict(node))
    assert configured_width(node) == built.out_channels == WIDTH
    adapter = ADAPTERS["dense" if built.output == "subsampled" else "subsampled"](node)
    assert configured_width(adapter) == BACKBONES.build(adapter).out_channels == WIDTH
    default = BACKBONES.build({"type": name}).out_channels
    if name in WIDTH_KNOWN_ONCE_BUILT:
        assert configured_width({"type": name}) is None
        assert seed_based._encoder_width(seed_based._Stage("backbone", {"type": name})) == default
    else:
        assert configured_width({"type": name}) == default
    assert configured_width({"type": "NoSuchEncoder"}) is None


def test_every_registered_encoder_declares_its_width_or_says_that_it_cannot():
    """A new encoder does not get past the seed-width rule by saying nothing: the default
    declaration reads an `out_channels` option, and an encoder without a default for it
    either overrides `output_channels` or is listed above with the reason."""
    from grasppanda.models.base import configured_width

    for name in registered("BACKBONES"):
        cls = BACKBONES.get(name)
        params, _ = component_parameters(cls)
        if "backbone" in params:  # an adapter: the width of what it wraps, or its own
            assert "output_channels" in vars(cls) or any(
                "output_channels" in vars(base)
                for base in cls.__mro__[1:-1]
                if base is not Backbone
            ), name
            continue
        declared = configured_width({"type": name})
        assert (declared is None) == (name in WIDTH_KNOWN_ONCE_BUILT), (name, declared)
