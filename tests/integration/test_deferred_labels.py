"""Deferred grasp labels on real GraspNet-1Billion training frames: nothing changes.

``SampleGraspLabels(grids="deferred")`` leaves the label grids in the store and lets
label matching read the rows it needs (``docs/advanced_guides/data_loading.md``). The
shipped dataset configs use it, so it has to give exactly what the dense grids give.
For the training recipes of GSNet, GraspNet-baseline and Scale-Balanced-Grasp these
tests build the dataset twice, once per storage, with the same sampling seed, and
require on three batches of real frames:

* the samples are the same (cloud, point labels, object poses, label points) and a
  deferred batch is small;
* the method's own assigner returns bit-identical labels for the same seeds and views;
* inside one forward pass of the method's model, the matched labels and the losses
  computed from them are bit-identical; and a whole training step on a fixed batch,
  losses and gradients, is bit-identical wherever a step repeats itself at all (it does
  for the PointNet++ models under deterministic kernels, not for the sparse
  convolutions of GSNet);
* the scale statistic of Scale-Balanced-Grasp is collected identically.

The tests need the dataset and the derived assets of the recipes
(``grasppanda prepare graspnet1b --labels``, ``--graspness``, ``--tolerance``) and skip
without them. The scale histogram of Scale-Balanced-Grasp is replaced by a made-up one
when its asset is not built: the weights are a function of the matched labels, which
is what is compared.
"""

from __future__ import annotations

import dataclasses
import pickle

import pytest
import torch

from grasppanda.config import Config
from grasppanda.datasets.assets import AssetError
from grasppanda.engine.seeding import determinism_restored, seed_everything
from grasppanda.engine.trainer import total_loss
from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY, collect_seed_scales
from grasppanda.models.views.template import ViewTemplate
from grasppanda.ops import furthest_point_sample
from grasppanda.registry import ASSIGNERS, DATASETS, DETECTORS
from grasppanda.structures import ApproachSet, Batch, Gripper, SampleKey, SeedSet, collate

CONFIGS = {
    "gsnet": "gsnet/gsnet_sunet14_g1b-realsense_10e",
    "baseline": "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e",
    "sbg": "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e",
}
#: ``(scene, frame)`` of twelve training frames; a config takes three batches of its size.
FRAMES = [
    (3, 17), (27, 200), (52, 5), (71, 133), (88, 64), (95, 250),
    (0, 0), (14, 91), (40, 180), (63, 100), (77, 31), (99, 255),
]  # fmt: skip
HISTOGRAM = {
    "edges": torch.linspace(0.0, 0.1, 33, dtype=torch.float64),
    "counts": torch.arange(1, 33) * 7 % 31 + 1,
}

pytestmark = pytest.mark.dataset


@dataclasses.dataclass
class Recipe:
    name: str
    cfg: dict
    batches: list[tuple[Batch, Batch]]  # (dense, deferred) of the same frames
    datasets: tuple


def _build(node: dict, grids: str):
    node = {**node, "seed": 0, "transforms": [dict(step) for step in node["transforms"]]}
    for step in node["transforms"]:
        if step["type"] == "SampleGraspLabels":
            step["grids"] = grids
    return DATASETS.build(node)


@pytest.fixture(scope="module", params=list(CONFIGS))
def recipe(request, graspnet1b_root) -> Recipe:
    cfg = Config.load(CONFIGS[request.param]).resolved()
    node = dict(cfg["data"]["train"]["dataset"])
    made_up_histogram = False
    try:
        dense, deferred = _build(node, "dense"), _build(node, "deferred")
    except AssetError as error:
        if "scale_statistic" not in str(error) and "scale statistic" not in str(error):
            pytest.skip(f"derived assets are not built: {str(error).splitlines()[0]}")
        node["scale_statistic"], made_up_histogram = None, True
        try:
            dense, deferred = _build(node, "dense"), _build(node, "deferred")
        except AssetError as other:
            pytest.skip(f"derived assets are not built: {str(other).splitlines()[0]}")
    size = cfg["data"]["train"]["batch_size"]
    batches = []
    for start in range(0, 3 * size, size):
        keys = [SampleKey("graspnet1b", s, "realsense", f) for s, f in FRAMES[start : start + size]]
        pair = []
        for dataset in (dense, deferred):
            batch = collate([dataset[dataset.keys.index(key)] for key in keys])
            if made_up_histogram:
                batch.supervision.extras[HISTOGRAM_KEY] = [HISTOGRAM] * len(keys)
            pair.append(batch)
        batches.append((pair[0], pair[1]))
    return Recipe(request.param, cfg, batches, (dense, deferred))


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if dataclasses.is_dataclass(value):
        return sum(_tensor_bytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    return 0


def _assert_same(first, second, where: str) -> None:
    for field in dataclasses.fields(first):
        one, two = getattr(first, field.name), getattr(second, field.name)
        if one is None or two is None:
            assert one is None and two is None, (where, field.name)
        else:
            assert one.dtype == two.dtype and torch.equal(one, two), (where, field.name)


def test_samples_are_the_same_and_a_deferred_batch_is_small(recipe):
    for number, (dense, deferred) in enumerate(recipe.batches):
        assert dense.observation.keys == deferred.observation.keys
        assert torch.equal(dense.observation.points.coords, deferred.observation.points.coords)
        assert dense.supervision.point_labels.keys() == deferred.supervision.point_labels.keys()
        for name, values in dense.supervision.point_labels.items():
            assert torch.equal(values, deferred.supervision.point_labels[name]), name
        grids = 0
        for objects, others in zip(dense.supervision.objects, deferred.supervision.objects):
            assert [obj.object_id for obj in objects] == [obj.object_id for obj in others]
            for one, two in zip(objects, others):
                assert two.deferred is not None and one.shapes() == two.shapes()
                assert torch.equal(one.pose, two.pose) and torch.equal(one.points, two.points)
                grids += _tensor_bytes([one.scores, one.widths, one.tolerance])
        # What travels in place of the grids: the readers (file names and rows).
        readers = [[obj.deferred for obj in objects] for objects in deferred.supervision.objects]
        carried = len(pickle.dumps(readers))
        whole = _tensor_bytes(deferred) + carried
        print(
            f"\n{recipe.name} batch {number}: dense grids {grids / 2**20:.0f} MiB, their "
            f"readers {carried / 2**10:.0f} KiB, the whole deferred batch {whole / 2**20:.2f} MiB"
        )
        assert grids > 300 * 2**20 and carried < 2**19 and whole < 4 * 2**20
    # Every grid of one frame, read through the deferred labels, is the dense grid.
    dense, deferred = recipe.batches[0]
    for one, two in zip(dense.supervision.objects[0], deferred.supervision.objects[0]):
        read = two.dense()
        for name in ("scores", "widths", "tolerance"):
            held, got = getattr(one, name), getattr(read, name)
            assert (held is None and got is None) or torch.equal(held, got), name


def _seeds(batch: Batch, template: ViewTemplate, count: int, seed: int):
    """Furthest-point seeds of every frame, as a sub-sampling backbone keeps them, and a
    random view for each."""
    points = batch.observation.points
    offsets = points.batch_offsets.tolist()
    rows = torch.cat(
        [
            furthest_point_sample(points.coords[offsets[f] : offsets[f + 1]], count) + offsets[f]
            for f in range(len(offsets) - 1)
        ]
    )
    seeds = SeedSet(points.subset(rows, torch.arange(len(offsets), device=rows.device) * count))
    generator = torch.Generator().manual_seed(seed)
    view = torch.randint(0, template.num_views, (len(rows),), generator=generator).to(rows.device)
    return seeds, ApproachSet(template.approach(view), template.rotation(view), view)


def test_the_assigner_of_the_method_matches_identical_labels(recipe, device):
    template = ViewTemplate.fibonacci(300).to(device)
    gripper = Gripper(**recipe.cfg["model"]["gripper"])
    assigner = ASSIGNERS.build(recipe.cfg["model"]["assigner"], template=template, gripper=gripper)
    for number, (on_host, deferred) in enumerate(recipe.batches):
        dense, deferred = on_host.to(device), deferred.to(device)
        # the move itself: poses and label points travel stacked and arrive unchanged
        for batch in (dense, deferred):
            moved = [obj for objects in batch.supervision.objects for obj in objects]
            held = [obj for objects in on_host.supervision.objects for obj in objects]
            assert len(moved) == len(held) > 4
            for one, two in zip(moved, held):
                assert one.pose.device.type == one.points.device.type == device.type
                assert torch.equal(one.pose.cpu(), two.pose)
                assert torch.equal(one.points.cpu(), two.points)
        seeds, approach = _seeds(dense, template, count=1024, seed=number)
        reference = assigner(seeds, approach, dense.supervision)
        _assert_same(reference, assigner(seeds, approach, deferred.supervision), recipe.name)
        assert reference.valid.any() and (reference.view_friction > 0).any()
        assert (reference.view_friction == 0).any()  # collisions were written into the scores
        expected = {
            "gsnet": (False, False, False),
            "baseline": (True, True, False),
            "sbg": (True, True, True),
        }[recipe.name]
        found = (reference.view_width, reference.tolerance, reference.weight)
        assert tuple(value is not None for value in found) == expected
        distinct = len(torch.unique(reference.label_point, dim=0))
        print(f"\n{recipe.name} batch {number}: {len(seeds.points)} seeds, {distinct} label points")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Deterministic behavior was enabled")
def test_losses_and_gradients_of_a_fixed_batch_are_identical(recipe, cuda):
    """``loss()`` and its gradients, storage against storage.

    Two things are shown. Inside one forward pass, the labels the model's assigner
    matches, and the view and head losses computed from them, are bit-identical for
    both storages. And where a training step repeats itself bit for bit at all (the
    PointNet++ models, with PyTorch's deterministic kernels), a whole step on deferred
    labels equals a whole step on dense ones, losses and gradients. The sparse
    convolutions of GSNet do not repeat themselves, so two steps on the very same
    batch already differ in the last bits there, whatever the storage.
    """
    if recipe.name == "gsnet":
        pytest.importorskip("spconv")
    dense, deferred = (batch.to(cuda) for batch in recipe.batches[0])
    with determinism_restored():
        seed_everything(0, deterministic=True)
        model = DETECTORS.build(recipe.cfg["model"]).to(cuda).train()

        def step(batch: Batch):
            torch.manual_seed(1)  # the view stage draws the same views every time
            model.zero_grad()
            outputs = model.loss(batch.observation, batch.supervision)
            total_loss(outputs, recipe.name).backward()
            gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            return {name: value.detach().clone() for name, value in outputs.items()}, gradients

        def same(first, second) -> bool:
            return all(torch.equal(first[0][name], second[0][name]) for name in first[0]) and all(
                torch.equal(one, two) for one, two in zip(first[1], second[1])
            )

        reference, again, other = step(dense), step(dense), step(deferred)
        repeatable = same(reference, again)
        print(
            f"\n{recipe.name}: {len(reference[0])} outputs, {len(reference[1])} gradient "
            f"tensors; a step repeats itself bit for bit: {repeatable}"
        )
        assert all(torch.isfinite(value).all() for value in reference[0].values())
        assert any(name.startswith("loss/") for name in reference[0]) and len(reference[1]) > 0
        assert other[0].keys() == reference[0].keys()
        if repeatable:
            assert same(reference, other), "a step on deferred labels differs from a dense one"
        else:
            assert recipe.name == "gsnet", "only the sparse convolutions are known not to repeat"
            for name, value in reference[0].items():  # no further apart than two dense steps
                spread = (again[0][name] - value).abs().max()
                assert (other[0][name] - value).abs().max() <= 8 * spread + 1e-5, name

        # One forward pass, the labels matched twice: the storage is the only difference.
        torch.manual_seed(1)
        with torch.no_grad():
            out = model.run_stages(dense.observation, dense.supervision)
            swapped = model.assigner(out.seeds, out.approach, deferred.supervision)
            _assert_same(out.matched, swapped, recipe.name)
            terms = 0
            for stage, predicted in ((model.view, out.approach), (model.head, out.raw)):
                one, two = stage.loss(predicted, out.matched), stage.loss(predicted, swapped)
                assert one.keys() == two.keys()
                for name, value in one.items():
                    assert torch.equal(torch.as_tensor(value), torch.as_tensor(two[name])), name
                terms += len(one)
        print(f"{recipe.name}: {terms} view and head loss terms identical within one forward pass")


@pytest.mark.parametrize("recipe", ["sbg"], indirect=True)  # the method that has one
def test_the_scale_statistic_is_collected_identically(recipe, device):
    section = recipe.cfg["data"]["train"]["dataset"]["scale_statistic"]
    arguments = {
        "frame_step": len(recipe.datasets[0]) // 5,  # five frames across the split
        "num_seeds": section["num_seeds"],
        "min_graspable_views": section["min_graspable_views"],
        "graspable_score": section["graspable_score"],
        "max_width": section["max_width"],
    }
    dense, deferred = (
        collect_seed_scales(dataset, device=device, **arguments) for dataset in recipe.datasets
    )
    print(f"\nscale statistic: {len(dense)} graspable seeds of 5 frames")
    assert len(dense) > 100 and torch.equal(dense, deferred)


def test_repeated_object_models_of_graspclutter6d_match_identically(graspclutter6d_root, device):
    """GraspClutter6D shares the label format, and a scene can hold one object model
    several times: the instances read the same score and width files and different rows
    of the scene's collision file. Needs `prepare graspclutter6d --labels --split sample`."""
    source = Config.load("_base_/datasets/graspclutter6d_d435.yaml")
    node = {**source.select("data.train.dataset"), "split": "sample"}
    try:
        dense, deferred = _build(node, "dense"), _build(node, "deferred")
    except AssetError as error:
        pytest.skip(f"derived assets are not built: {str(error).splitlines()[0]}")
    indices = list(range(0, len(dense), max(len(dense) // 6, 1)))[:6]
    pair = [collate([dataset[index] for index in indices]) for dataset in (dense, deferred)]
    repeated = 0
    for objects, others in zip(pair[0].supervision.objects, pair[1].supervision.objects):
        models = [obj.object_id for obj in objects]
        repeated += len(models) - len(set(models))
        for one, two in zip(objects, others):
            assert torch.equal(one.pose, two.pose) and torch.equal(one.points, two.points)
    assert repeated > 0, "the sample scenes are meant to hold an object model twice"
    for one, two in zip(pair[0].supervision.objects[0], pair[1].supervision.objects[0]):
        read = two.dense()
        assert torch.equal(one.scores, read.scores) and torch.equal(one.widths, read.widths)
    template = ViewTemplate.fibonacci(300).to(device)
    assigner = ASSIGNERS.build({"type": "NearestLabelAssigner"}, template=template)
    on_device = [batch.to(device) for batch in pair]
    seeds, approach = _seeds(on_device[0], template, count=1024, seed=0)
    reference = assigner(seeds, approach, on_device[0].supervision)
    _assert_same(reference, assigner(seeds, approach, on_device[1].supervision), "gc6d")
    assert (reference.view_friction == 0).any() and (reference.view_friction > 0).any()
    print(f"\ngraspclutter6d: {len(indices)} frames, {repeated} repeated object instances")
