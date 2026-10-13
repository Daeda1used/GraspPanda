"""The labels a model reads against the labels its training dataset serves.

Both sides declare it: a stage what it reads from the training ``Supervision``
(``needs_supervision``), a dataset reader what a config makes it put into a batch
(``served_supervision``). The config check compares the two, so a model on a data recipe
that does not serve its labels is refused before anything is built instead of failing at
the first loss.

Two things are tested. The rule: every shipped config passes, every mismatch is refused
at ``data.train.dataset`` with the field, the stage that reads it and what would serve
it. And the declarations themselves, on the tiny models of the swap matrix: a batch that
carries exactly the declared fields trains, and a batch without any one of them does not.
A stage that reads something it does not declare, or declares something it does not read,
fails here.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch

from grasppanda.config import Config, ConfigError, available_configs
from grasppanda.datasets.scene import GraspSceneDataset
from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY
from grasppanda.registry import DATASETS, DETECTORS
from grasppanda.structures import COMPACT_LABELS, Supervision

from . import test_swap_matrix as matrix

GSNET = "gsnet/gsnet_sunet14_g1b-realsense_10e"
ECONOMIC = "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e"
BASELINE = "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"
SBG = "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e"
HGGD = "hggd/hggd_r34w8_g1b-realsense_15e"

#: Every field the synthetic batch of the swap matrix carries, by its declared name.
FIELDS = (
    "objects",
    "objects.tolerance",
    "point_labels.objectness",
    "point_labels.graspness",
    f"extras.{COMPACT_LABELS}",
    f"extras.{HISTOGRAM_KEY}",
)


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def sides(config: Config) -> tuple[set[str], frozenset[str]]:
    """What the model of a config reads and what its training dataset serves."""
    data = config.to_dict()
    model, dataset = data["model"], data["data"]["train"]["dataset"]
    needs = DETECTORS.get(model["type"]).supervision_needs(model)
    served = DATASETS.get(dataset["type"]).served_supervision(dataset)
    return {field for field, _ in needs}, served


def refused(base: str, overrides: list[str]) -> list[str]:
    """Messages of the config check at ``data.train.dataset``; the config must be refused."""
    with pytest.raises(ConfigError) as error:
        Config.load(base, overrides)
    return [issue.message for issue in error.value.issues if issue.path == "data.train.dataset"]


# ------------------------------------------------------------------------- the rule


def test_every_shipped_config_trains_on_data_that_serves_its_labels():
    trained = 0
    for name in available_configs():
        config = Config.load(name)  # the config check itself passes
        if "data" not in config.to_dict() or "model" not in config.to_dict():
            continue
        needs, served = sides(config)
        assert needs and needs <= served, (name, needs - served)
        trained += 1
    assert trained >= 20
    # The label sets of the shipped methods, as their stages declare them.
    assert sides(Config.load(GSNET))[0] == {
        "objects",
        "point_labels.objectness",
        "point_labels.graspness",
    }
    assert sides(Config.load(ECONOMIC))[0] == {
        f"extras.{COMPACT_LABELS}",
        "point_labels.objectness",
        "point_labels.graspness",
    }
    assert sides(Config.load(BASELINE))[0] == {
        "objects",
        "objects.tolerance",
        "point_labels.objectness",
    }
    assert sides(Config.load(SBG))[0] == sides(Config.load(BASELINE))[0] | {
        f"extras.{HISTOGRAM_KEY}"
    }
    assert sides(Config.load(HGGD))[0] == {"extras.view_grasps"}
    # A dataset serves the dense grids or compact labels, never both.
    assert "objects" not in sides(Config.load(ECONOMIC))[1]
    assert f"extras.{COMPACT_LABELS}" not in sides(Config.load(GSNET))[1]


@pytest.mark.parametrize(
    ("base", "overrides", "fragments"),
    [
        (  # GSNet's assigner and head on EconomicGrasp's recipe: compact labels only
            ECONOMIC,
            ["model.assigner={type: NearestLabelAssigner}", "model.head={type: SWADHead}"],
            [
                "NearestLabelAssigner reads `supervision.objects`, which this training dataset "
                "does not serve.",
                "It serves compact labels in place of the dense label grids",
                "the dense grids need a `SampleGraspLabels` transform",
            ],
        ),
        (  # EconomicGrasp's assigner and head on GSNet's recipe: dense grids only
            GSNET,
            ["model.assigner={type: CompactLabelAssigner}", "model.head={type: EconomicHead}"],
            [
                "CompactLabelAssigner reads `supervision.extras['compact_labels']`",
                "served with a `compact_labels:` section",
                "grasppanda prepare graspnet1b --compact-labels",
            ],
        ),
        (  # a proposal that is supervised with graspness on a recipe without it
            BASELINE,
            [
                "model.backbone={type: FeaturePropagation, backbone: {type: PointNet2}}",
                "model.proposal={type: GraspnessFPS}",
                "model.view={type: TemplateViewStage}",
            ],
            [
                "GraspnessFPS reads `supervision.point_labels['graspness']`",
                "It is served with a `graspness:` section of the dataset",
            ],
        ),
        (  # a head that supervises tolerance on a recipe without tolerance labels
            GSNET,
            ["model.grouper={type: DepthCylinderGroup}", "model.head={type: OperationHead}"],
            [
                "OperationHead reads `supervision.objects.tolerance`",
                "It is served with a `tolerance:` section of the dataset",
                "grasppanda prepare graspnet1b --tolerance",
            ],
        ),
        (  # seed weights from a histogram the recipe does not serve
            BASELINE,
            ["model.assigner={type: ScaleBalancedAssigner}"],
            [
                "ScaleBalancedAssigner reads `supervision.extras['scale_histogram']`",
                "It is served with a `scale_statistic:` section of the dataset",
            ],
        ),
        (  # an image-based detector without its per-view labels
            HGGD,
            ["data.train.dataset.view_grasps=null"],
            [
                "HGGD reads `supervision.extras['view_grasps']`",
                "served with a `view_grasps:` section",
            ],
        ),
        (  # the label transform removed: the objects of a frame would carry no grids
            GSNET,
            ["data.train.dataset.transforms=[{type: RandomSample, num_points: 15000}]"],
            ["reads `supervision.objects`", "add `SampleGraspLabels` to its `transforms`"],
        ),
        (GSNET, ["data.train.dataset.object_labels=false"], ["`object_labels: false`"]),
    ],
)
def test_a_recipe_that_does_not_serve_a_label_is_refused_with_the_fix(base, overrides, fragments):
    messages = refused(base, overrides)
    assert len(messages) == 1, messages
    for fragment in fragments:
        assert fragment in messages[0], (fragment, messages[0])
    # Only training reads the training data.
    assert Config.load(base, overrides, command="test").digest()


def test_one_message_per_field_names_every_reader_and_a_part_is_not_repeated():
    # Two stages read the objectness labels; a dataset without labels serves nothing.
    (message,) = refused(
        GSNET,
        [
            "model.assigner.supervise=object_seeds",
            "data.train.dataset={type: GraspNet1B, camera: realsense, split: train, "
            "load_labels: false, preprocess: [{type: DepthToCloud}, {type: NoWorkspace}], "
            "transforms: []}",
        ],
    )[1:2]
    assert message.startswith(
        "GraspnessFPS and NearestLabelAssigner read `supervision.point_labels['objectness']`"
    )
    assert "`load_labels: false`" in message
    # The tolerance of grids that are not served at all is not a second problem.
    messages = refused(
        ECONOMIC,
        [
            "model.assigner={type: NearestLabelAssigner, supervise: object_seeds}",
            "model.grouper={type: DepthCylinderGroup}",
            "model.head={type: OperationHead}",
        ],
    )
    assert len(messages) == 1 and "`supervision.objects`," in messages[0]


def test_a_side_that_declares_nothing_is_not_compared():
    from .conftest import REPO

    toy = Config.load(REPO / "projects/example_project/configs/toy_train.yaml")
    dataset = toy.to_dict()["data"]["train"]["dataset"]
    assert DATASETS.get(dataset["type"]).served_supervision(dataset) is None
    # A model node the rule cannot read (a stage is missing) is left to the signature check.
    assert DETECTORS.get("GSNet").supervision_needs({"type": "GSNet"}) is None
    assert GraspSceneDataset.supervision_hint("extras.unknown", {}) == (
        "GraspSceneDataset has no option that serves it."
    )


# ------------------------------------------------------- declared is what is read


def without(supervision: Supervision, field: str) -> Supervision:
    """``supervision`` as a dataset that does not serve ``field`` would deliver it."""
    head, _, name = field.partition(".")
    if field == "objects":
        return replace(supervision, objects=[[] for _ in supervision.objects])
    if field == "objects.tolerance":
        stripped = [
            [replace(obj, tolerance=None) for obj in frame] for frame in supervision.objects
        ]
        return replace(supervision, objects=stripped)
    kept = {key: value for key, value in getattr(supervision, head).items() if key != name}
    return replace(supervision, **{head: kept})


@functools.cache
def _legal_sets() -> tuple[dict, ...]:
    return tuple(
        chosen
        for chosen in matrix.stage_sets()
        if not matrix.refusal(matrix.node_of("GSNet", chosen))
    )


def legal_models() -> list[dict]:
    """The legal stage sets of the swap matrix as model nodes (checked once per session)."""
    return [matrix.node_of("GSNet", chosen) for chosen in _legal_sets()]


def test_every_stage_reads_what_it_declares_and_nothing_else():
    observation, everything = matrix.batch()
    seen: set[str] = set()
    for node in legal_models():
        needs = {field for field, _ in DETECTORS.get(node["type"]).supervision_needs(node)}
        assert needs <= set(FIELDS), needs - set(FIELDS)
        seen |= needs
        if not matrix.buildable(node):  # the optional library of a stage is not installed
            continue
        torch.manual_seed(0)
        model = DETECTORS.build(node)
        model.train()
        exact = everything
        for field in FIELDS:
            # A part goes with its whole: the grids without tolerance are still grids.
            if field not in needs and not any(need.startswith(f"{field}.") for need in needs):
                exact = without(exact, field)
        losses = model.loss(observation, exact)  # nothing undeclared is read
        assert all(torch.isfinite(value) for value in losses.values())
        for field in needs:
            with pytest.raises(Exception, match=r"\S"):  # every declared field is read
                model.loss(observation, without(exact, field))
    assert seen == set(FIELDS), "a field of the synthetic batch that no stage set reads"
    assert len(legal_models()) == matrix.LEGAL_STAGE_SETS


def test_a_stage_outside_the_legal_sets_reads_what_a_tested_stage_reads():
    """The test above covers the stages of the legal stage sets. A registered stage that
    is in none of them (a bridge whose library the minimal install lacks) must take its
    loss, and with it what it reads, from a stage that is: a new stage with a loss of
    its own goes into the matrix, where its declaration is tested."""
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    exercised = {
        slot: {node[slot]["type"] for node in legal_models()}
        for slot in STAGE_REGISTRIES
        if slot != "backbone"
    }
    for slot, names in exercised.items():
        registry = STAGE_REGISTRIES[slot]
        for name in registry.names():
            cls = registry.get(name)
            if name in names or not cls.__module__.startswith("grasppanda."):
                continue
            tested = [registry.get(other) for other in sorted(names)]
            twins = [
                other
                for other in tested
                if all(
                    getattr(cls, reader, None) is getattr(other, reader, None)
                    for reader in ("loss", "__call__", "encode_targets")
                )
            ]
            assert twins, (
                f"{name} ({slot}) is in no legal stage set of the swap matrix and has a loss "
                "of its own: what it reads from the supervision is not tested"
            )
            assert cls.needs_supervision == twins[0].needs_supervision, name


def test_the_reader_serves_what_its_sections_say():
    node = Config.load(GSNET).to_dict()["data"]["train"]["dataset"]
    assert GraspSceneDataset.served_supervision(node) == {
        "objects",
        "point_labels.objectness",
        "point_labels.instance",
        "point_labels.graspness",
    }
    with_all = {
        **node,
        "tolerance": {},
        "scale_statistic": {},
        "view_grasps": {},
    }
    assert GraspSceneDataset.served_supervision(with_all) >= {
        "objects.tolerance",
        "extras.scale_histogram",
        "extras.view_grasps",
    }
    assert GraspSceneDataset.served_supervision({**node, "load_labels": False}) == frozenset()
    compact = {**node, "compact_labels": {}, "transforms": []}
    assert f"extras.{COMPACT_LABELS}" in GraspSceneDataset.served_supervision(compact)
    assert "objects" not in GraspSceneDataset.served_supervision(compact)
    # Tolerance is a part of the dense grids: without them there is none to serve.
    assert "objects.tolerance" not in GraspSceneDataset.served_supervision(
        {**compact, "tolerance": {}}
    )
