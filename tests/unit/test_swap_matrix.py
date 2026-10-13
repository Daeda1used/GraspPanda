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
from grasppanda.models.detectors.seed_based import SeedGraspDetector, stage_contract_problems
from grasppanda.registry import BACKBONES, DETECTORS, GROUPERS, HEADS, PROPOSALS
from grasppanda.structures import ContractError, GraspGroup, Observation, Supervision

from .conftest import REPO

HAS_SPCONV = importlib.util.find_spec("spconv") is not None
needs_spconv = pytest.mark.skipif(not HAS_SPCONV, reason="needs the spconv extra")

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
    "PointNet2": ({**_SA, "fp_mlps": [[16, 16], [16, WIDTH]]}, ()),
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
    },
    "view": {
        "TemplateViewStage": {"type": "TemplateViewStage", "num_views": VIEW_COUNT},
        "ObjectnessViewStage": {"type": "ObjectnessViewStage", "num_views": VIEW_COUNT},
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


def run(node: dict[str, Any]) -> SeedGraspDetector:
    """Build an accepted model, take one training step's worth of gradients, predict."""
    assert refusal(node) == []
    torch.manual_seed(0)
    model = DETECTORS.build(node)
    if hasattr(model.proposal, "score"):  # make most points candidates: real seeds everywhere
        with torch.no_grad():
            model.proposal.score.weight.mul_(0.01)
            model.proposal.score.bias.copy_(torch.tensor([0.0, 1.0, 1.0]))
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
    assert set(registered("BACKBONES")) == set(ENCODERS) | adapters
    for slot, registry in (
        ("proposal", "PROPOSALS"),
        ("view", "VIEWS"),
        ("assigner", "ASSIGNERS"),
        ("grouper", "GROUPERS"),
        ("head", "HEADS"),
    ):
        assert {node["type"] for node in STAGES[slot].values()} == set(registered(registry)), slot
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
    required = PROPOSALS.get(proposal).requires
    node = method("GSNet", encoder(backbone), proposal=proposal)
    if produced == required:
        model = run(node)
        assert model.proposal.requires == model.backbone.output
        return
    (message,) = refusal(node)
    assert f"{proposal} requires '{required}' scene features" in message
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
            with pytest.raises(ContractError):  # the same verdict without the config checker
                DETECTORS.build(node_of("GSNet", chosen))
        else:
            legal.append(chosen)
    assert legal and refused
    for chosen in legal:
        for detector in ("GSNet", "ScaleBalancedGrasp"):  # the barest and the strictest type
            run(node_of(detector, chosen))
    # Every refusal reads as a sentence about two named stages.
    for chosen in stage_sets():
        for message in refusal(node_of("GSNet", chosen)):
            assert message[0].isalpha() or message.startswith("the "), message
            assert message.rstrip().endswith("."), message
    assert len(legal) == LEGAL_STAGE_SETS, len(legal)


#: Stage sets the contracts allow among the variants above; the count is restated in
#: docs/advanced_guides/swapping.md together with the list, so it is pinned here.
LEGAL_STAGE_SETS = 41


def test_what_the_contracts_refuse_among_plausible_looking_swaps():
    """One example per rule, as a user would meet it: a single replaced line."""
    dense, sub = encoder("PointNet2Dense"), encoder("PointNet2")
    cases = [
        # GSNet with the scale-balanced assigner: the weights would be ignored.
        (
            method("GSNet", dense, assigner="ScaleBalancedAssigner"),
            "TemplateViewStage and SWADHead do not apply seed weights",
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


def test_channel_counts_are_injected_and_a_mismatch_names_the_fix():
    """Widths are known only once the encoder is built, so this one is a build error."""
    narrow = {**STAGES["grouper"]["DepthCylinderGroup+scales+gate"], "mlp_channels": [8, 8]}
    node = method("ScaleBalancedGrasp", encoder("PointNet2"))
    node["grouper"] = narrow
    assert refusal(node) == []
    with pytest.raises(
        ValueError, match=r"12-channel seed features to 8-channel crop features"
    ) as error:
        DETECTORS.build(node)
    assert "mlp_channels" in str(error.value)


def test_pairwise_verdicts_equal_whole_model_verdicts():
    """The tables of the documentation come from pairwise checks; a stage set is legal
    exactly when no pair inside it is refused."""
    for chosen in stage_sets():
        node = node_of("GSNet", chosen)
        pairwise = stage_contract_problems(**{slot: node[slot] for slot in ("backbone", *STAGES)})
        assert pairwise == refusal(node), chosen
