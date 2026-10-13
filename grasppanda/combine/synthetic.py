"""Synthetic inputs with labels, and the wiring check built on them.

Two generators, both free of any dataset content:

* :func:`random_batch`: frames of uniformly random points with every label a registered
  stage reads, at any grid size. It is the batch of the swap matrix and of the label
  contract tests (``tests/unit/test_swap_matrix.py``,
  ``tests/unit/test_supervision_contract.py``), where the stages are a few channels wide;
  it is shipped so that a project can run its own stage through the same batch.
* :func:`scene_batch`: the ray-cast table-top scene of
  :func:`grasppanda.apis.example.example_scene` as a point cloud in metres (and, for an
  image-based detector, as images), with random labels attached to its objects. A
  full-size model sees neighbourhoods of the right scale in it.

The labels are noise. A model fed with them learns nothing, and no number computed on
them is a result; they exist so that every stage runs, receives a target and returns a
gradient. :func:`wiring_check` is that test for one config; its best verdict is
``wired``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from grasppanda.models.assigners.scale_balance import HISTOGRAM_KEY
from grasppanda.registry import DATASETS, PROPOSALS, component_parameters
from grasppanda.structures import (
    COMPACT_LABELS,
    CompactGraspLabels,
    ObjectGraspLabels,
    Observation,
    PointSet,
    SampleKey,
    Supervision,
)

__all__ = ["HISTOGRAM_KEY", "random_batch", "scene_batch", "wiring_check"]

_LEVELS = (-1.0, 0.1, 0.3, 0.6, 1.0)  # friction labels: invalid, then good to poor


def _labels(
    generator: torch.Generator,
    object_id: int,
    pose: torch.Tensor,
    local: torch.Tensor,
    grid: tuple[int, int, int, int],
    view_max_friction: float,
) -> tuple[ObjectGraspLabels, CompactGraspLabels]:
    """Random dense and compact labels of one object; the draws come in a fixed order."""

    def integers(high: int) -> torch.Tensor:
        return torch.randint(0, high, grid[:2], generator=generator).to(torch.uint8)

    dense = ObjectGraspLabels(
        object_id=object_id,
        pose=pose,
        points=local,
        scores=torch.tensor(_LEVELS)[torch.randint(0, 5, grid, generator=generator)],
        widths=torch.rand(grid, generator=generator) * 0.12,
        tolerance=torch.randint(0, 51, grid, generator=generator) * 0.001,
    )
    compact = CompactGraspLabels(
        object_id=object_id,
        pose=pose,
        points=local,
        view_graspness=torch.rand(grid[:2], generator=generator),
        angle=integers(grid[2]),
        depth=integers(grid[3]),
        score=integers(11),
        width=integers(120),
        view_max_friction=view_max_friction,
    )
    return dense, compact


def _histogram() -> dict[str, torch.Tensor]:
    return {
        "edges": torch.linspace(0.0, 0.12, 5, dtype=torch.float64),
        "counts": torch.tensor([3, 20, 50, 9]),
    }


def random_batch(
    *,
    frames: int = 2,
    points: int = 96,
    views: int = 10,
    angles: int = 3,
    depths: int = 2,
    label_points: int = 20,
    seed: int = 0,
    view_max_friction: float = 0.6,
) -> tuple[Observation, Supervision]:
    """Frames of random points in the unit cube with everything any stage reads.

    Dense grids with tolerance, compact labels, per-point objectness and graspness and
    a width histogram; two objects per frame, their label points on the cloud so that
    seeds lie within a matching distance.
    """
    from grasppanda.ops import approach_to_rotation

    generator = torch.Generator().manual_seed(seed)
    cloud = torch.rand(frames, points, 3, generator=generator)
    observation = Observation(
        keys=[SampleKey("synthetic", 7, "none", frame) for frame in range(frames)],
        points=PointSet.from_dense(cloud),
    )
    grid = (label_points, views, angles, depths)
    dense: list[list[ObjectGraspLabels]] = []
    compact: list[list[CompactGraspLabels]] = []
    for frame in range(frames):
        dense.append([])
        compact.append([])
        for _ in range(2):
            pose = torch.eye(4)
            pose[:3, :3] = approach_to_rotation(
                torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
            )
            pose[:3, 3] = torch.rand(3, generator=generator)
            pick = torch.randint(0, points, (label_points,), generator=generator)
            placed = cloud[frame, pick] + 0.01 * torch.randn(label_points, 3, generator=generator)
            local = (placed - pose[:3, 3]) @ pose[:3, :3]
            full, kept = _labels(generator, len(dense), pose, local, grid, view_max_friction)
            dense[-1].append(full)
            compact[-1].append(kept)
    supervision = Supervision(
        objects=dense,
        point_labels={
            "objectness": (torch.rand(frames * points, generator=generator) < 0.7).long(),
            "graspness": torch.rand(frames * points, generator=generator),
        },
        extras={COMPACT_LABELS: compact, HISTOGRAM_KEY: [_histogram()] * frames},
    )
    return observation, supervision


# ------------------------------------------------------------------------ the scene


def _view_grasps(
    generator: torch.Generator,
    mask: np.ndarray,
    depth: torch.Tensor,
    focal: float,
    count: int = 60,
) -> dict[str, torch.Tensor]:
    """Random per-view grasp labels on the object pixels of an image.

    Widths and heights are those of grasps 4 to 9 cm wide and 2 cm high at the depth of
    their pixel, in pixels of the image.
    """
    rows, columns = np.nonzero(mask > 0)
    pick = torch.randint(0, len(rows), (count,), generator=generator).numpy()
    u = torch.from_numpy(columns[pick]).double()
    v = torch.from_numpy(rows[pick]).double()
    metres = depth[rows[pick], columns[pick]].double()

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(count, generator=generator, dtype=torch.float64)

    return {
        "centers": torch.stack([u, v], dim=1),
        "depth": metres + 0.005,
        "theta": uniform(-math.pi, math.pi),
        "gamma": uniform(-0.5, 0.5),
        "beta": uniform(-0.5, 0.5),
        "width": uniform(0.04, 0.09) * focal / metres,
        "height": 0.02 * focal / metres,
        "score": uniform(0.5, 1.0),
        "object_id": torch.from_numpy(mask[rows[pick], columns[pick]].astype(np.int64)) - 1,
    }


def scene_batch(
    *,
    frames: int = 2,
    points: int = 4096,
    views: int = 300,
    angles: int = 12,
    depths: int = 4,
    label_points: int = 48,
    feature_channels: int | None = None,
    images: bool = False,
    image_size: tuple[int, int] = (1280, 720),
    seed: int = 0,
    view_max_friction: float = 0.6,
) -> tuple[Observation, Supervision]:
    """The synthetic table-top scene as a training batch with random labels.

    Args:
        frames: Frames of the batch; each renders the scene with another layout.
        points: Points sampled per frame from the pixels the camera sees.
        views, angles, depths: Size of the label grids; they must be those of the view
            stage and the head bundle that will read them.
        label_points: Annotated points per object, drawn from the cloud.
        feature_channels: With a number, ``point_feats`` holds that many channels: the
            coordinates, then a unit vector towards the camera, then zeros. ``None``
            leaves the features out (a detector that feeds constant ones needs none).
        images: Also return the colour image, the depth image and the camera matrix,
            and per-view grasp labels on the object pixels (``extras["view_grasps"]``),
            for image-based detectors.
        image_size: ``(width, height)`` of those images.
        seed: Seed of the layout, the sampling and the labels.
        view_max_friction: The friction level compact labels claim their view scores
            were computed with; the view stage checks it against its own.

    Returns:
        ``(observation, supervision)`` on the CPU.
    """
    from grasppanda.apis.example import example_scene
    from grasppanda.ops import approach_to_rotation

    generator = torch.Generator().manual_seed(seed)
    size = tuple(image_size) if images else (480, 270)
    grid = (label_points, views, angles, depths)
    clouds, feats, objectness = [], [], []
    dense: list[list[ObjectGraspLabels]] = []
    compact: list[list[CompactGraspLabels]] = []
    pictures: dict[str, list[Any]] = {"rgb": [], "depth": [], "intrinsics": [], "view_grasps": []}
    for frame in range(frames):
        scene = example_scene(seed + frame, size=size)
        depth = torch.from_numpy(scene.depth.astype(np.float32) / scene.depth_scale)
        k = torch.from_numpy(scene.intrinsics).float()
        v, u = torch.meshgrid(torch.arange(size[1]), torch.arange(size[0]), indexing="ij")
        cloud = torch.stack(
            [(u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth], dim=-1
        )
        seen = (depth > 0).reshape(-1)
        candidates = seen.nonzero()[:, 0]
        if len(candidates) >= points:
            order = torch.randperm(len(candidates), generator=generator)[:points]
        else:  # a small image: repeat pixels rather than return fewer points than asked
            order = torch.randint(0, len(candidates), (points,), generator=generator)
        rows = candidates[order]
        sampled = cloud.reshape(-1, 3)[rows]
        instance = torch.from_numpy(scene.mask.astype(np.int64)).reshape(-1)[rows]
        clouds.append(sampled)
        objectness.append((instance > 0).long())
        if feature_channels is not None:
            towards = -sampled / sampled.norm(dim=1, keepdim=True).clamp(min=1e-6)
            wide = torch.cat([sampled, towards, sampled.new_zeros(points, feature_channels)], 1)
            feats.append(wide[:, :feature_channels])
        dense.append([])
        compact.append([])
        for mark in instance.unique().tolist():
            members = (instance == mark).nonzero()[:, 0]
            if mark == 0 or len(members) < 8:
                continue
            pose = torch.eye(4)
            pose[:3, :3] = approach_to_rotation(
                torch.randn(3, generator=generator), torch.rand((), generator=generator) * math.pi
            )
            pose[:3, 3] = sampled[members].mean(dim=0)
            pick = members[torch.randint(0, len(members), (label_points,), generator=generator)]
            placed = sampled[pick] + 0.001 * torch.randn(label_points, 3, generator=generator)
            local = (placed - pose[:3, 3]) @ pose[:3, :3]
            full, kept = _labels(generator, mark - 1, pose, local, grid, view_max_friction)
            dense[-1].append(full)
            compact[-1].append(kept)
        if images:
            pictures["rgb"].append(torch.from_numpy(scene.rgb).permute(2, 0, 1).float() / 255.0)
            pictures["depth"].append(depth)
            pictures["intrinsics"].append(k)
            labels = _view_grasps(generator, scene.mask, depth, float(k[0, 0]))
            pictures["view_grasps"].append(labels)
    observation = Observation(
        keys=[SampleKey("synthetic", 0, "example", frame) for frame in range(frames)],
        points=PointSet.from_dense(torch.stack(clouds)),
        point_feats=torch.cat(feats) if feats else None,
        intrinsics=torch.stack(pictures["intrinsics"]) if images else None,
        depth=torch.stack(pictures["depth"]) if images else None,
        rgb=torch.stack(pictures["rgb"]) if images else None,
    )
    extras: dict[str, list[Any]] = {COMPACT_LABELS: compact, HISTOGRAM_KEY: [_histogram()] * frames}
    if images:
        extras["view_grasps"] = pictures["view_grasps"]
    graspness = torch.rand(frames * points, generator=generator)
    supervision = Supervision(
        objects=dense,
        point_labels={"objectness": torch.cat(objectness), "graspness": graspness},
        extras=extras,
    )
    return observation, supervision


# -------------------------------------------------------------------- the wiring check


def _find(node: Any, key: str) -> Any:
    """First value of ``key`` anywhere in a nested config node."""
    if isinstance(node, Mapping):
        if node.get(key) is not None:
            return node[key]
        for value in node.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _sizes(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """What the synthetic batch must look like for the model of a resolved config."""
    model = cfg["model"]
    transforms = (((cfg.get("data") or {}).get("train") or {}).get("dataset") or {}).get(
        "transforms"
    )
    sampled = _find(transforms, "num_points")
    sizes: dict[str, Any] = {"points": int(sampled) if sampled else 4096}
    dataset = ((cfg.get("data") or {}).get("train") or {}).get("dataset") or {}
    reader = DATASETS.get(dataset["type"]) if dataset.get("type") in DATASETS else None
    grid = getattr(reader, "label_grid", None)
    if grid is not None:
        # The grid of the labels the dataset serves, not of the model: a stage laid out
        # on another grid must fail here as it would on real frames.
        sizes.update(views=grid.num_view, angles=grid.num_angle, depths=grid.num_depth)
    else:
        view = model.get("view") if isinstance(model.get("view"), Mapping) else {}
        lattice = view.get("label_views") or _find(model, "num_views")
        for key, value in (
            ("views", lattice),
            ("angles", _find(model, "num_angle")),
            ("depths", _find(model, "num_depth")),
        ):
            if isinstance(value, int):
                sizes[key] = value
    friction = (model.get("view") or {}).get("max_friction") if "view" in model else None
    if isinstance(friction, (int, float)):
        sizes["view_max_friction"] = float(friction)
    # Images as the dataset of the config delivers them; without one, twice the network
    # image of the model, which is how the shipped image-based configs relate the two.
    sensor = getattr(reader, "image_size", None)
    if not isinstance(sensor, (tuple, list)):  # a property: the size depends on the camera
        sensor = (getattr(reader, "image_sizes", None) or {}).get(dataset.get("camera"))
    network = (
        (model.get("image") or {}).get("size") if isinstance(model.get("image"), Mapping) else None
    )
    if sensor is not None:
        sizes["image_size"] = (int(sensor[1]), int(sensor[0]))
    elif network is not None:
        sizes["image_size"] = (2 * int(network[0]), 2 * int(network[1]))
    return sizes


def _opened(cfg: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """``cfg`` with the score threshold of its proposal lowered, when it has one.

    An untrained proposal may let no point through, and the stages behind it would then
    see no seed and return no gradient. Which points pass means nothing on noise labels,
    so the check lets them all through and says so.
    """
    proposal = cfg["model"].get("proposal")
    if not (isinstance(proposal, Mapping) and proposal.get("type") in PROPOSALS):
        return cfg
    params, _ = component_parameters(PROPOSALS.get(proposal["type"]))
    if "graspness_threshold" not in params:
        return cfg
    notes.append(
        "wiring check: model.proposal.graspness_threshold was lowered so that the untrained "
        "proposal lets points through to the stages behind it."
    )
    model = {**cfg["model"], "proposal": {**proposal, "graspness_threshold": -1.0e9}}
    return {**cfg, "model": model}


def wiring_check(report: Any, config: Any, *, iters: int, device: str | None) -> None:
    """Build the detector of ``config``, train it on the synthetic scene, predict.

    Fills ``report`` (a :class:`~grasppanda.combine.probe.ProbeReport`). The verdict is
    ``wired`` when every loss is finite, a gradient reached every part of the detector
    that has parameters, and prediction returned one grasp group per frame; ``fails``
    otherwise, with the phase and the component.
    """
    from grasppanda.engine.builders import build_model
    from grasppanda.engine.optim import build_optimizers
    from grasppanda.engine.run import resolve_device
    from grasppanda.engine.seeding import determinism_restored, seed_everything
    from grasppanda.engine.trainer import total_loss

    from .probe import _count_grasps, _Enough, _fail, _judge, _Recorder, _summarise

    cfg = _opened(config.resolved(), report.notes)
    target = resolve_device(device or cfg["runtime"]["device"])
    report.device, report.data, report.iterations_requested = str(target), "synthetic", iters
    recorder = _Recorder(iters, target)
    with determinism_restored():
        seed_everything(cfg["runtime"]["seed"])
        phase = "build"
        try:
            model = build_model(cfg).to(target)
            report.parameters = sum(parameter.numel() for parameter in model.parameters())
            needs = getattr(model, "required_inputs", frozenset())
            width = getattr(getattr(model, "backbone", None), "in_channels", 3)
            channels = width if "point_feats" in needs else None
            phase = "data"
            observation, supervision = scene_batch(
                images=bool({"rgb", "depth"} & set(needs)), feature_channels=channels, **_sizes(cfg)
            )
            observation, supervision = observation.to(target), supervision.to(target)
            phase = "optimizer"
            recipe = cfg.get("optimizer") or {"type": "Adam", "lr": 1e-3}
            optimizers = list(build_optimizers(model, recipe).values())
            signal = getattr(model, "on_train_epoch", None)
            if callable(signal):  # what EpochSignalHook does for a detector that listens
                signal(0, 1)
            model.train()
            for index in range(iters):
                phase, when = "loss", f"iteration {index + 1}"
                recorder.begin(observation)
                outputs = model.loss(observation, supervision)
                total = total_loss(outputs, when, observation.keys)
                phase = "backward"
                total.backward()
                recorder.note_gradients(model)
                phase = "step"
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                state = type("State", (), {"outputs": {**outputs, "loss": total.detach()}})
                try:
                    recorder.after_iter(state)
                except _Enough:
                    break
            _summarise(report, recorder)
            if report.no_gradient:
                phase = "backward"
                raise RuntimeError(
                    f"no gradient reached {', '.join(report.no_gradient)} in "
                    f"{report.iterations} iteration(s) on the synthetic scene: a stage before "
                    "it produced nothing to learn from (for example no seed)"
                )
            phase = "predict"
            model.eval()
            with torch.no_grad():
                groups = model.predict(observation)
            _count_grasps(report, observation.keys, groups)
        except Exception as error:
            _summarise(report, recorder)
            _fail(report, error, phase)
            return
        finally:
            if target.type == "cuda":
                report.peak_gpu_mib = round(torch.cuda.max_memory_allocated(target) / 2**20, 1)
    report.real_seeds = []  # seeds on noise labels are not something to judge
    _judge(report, "wired", "a synthetic scene with random labels", len(observation.keys))
