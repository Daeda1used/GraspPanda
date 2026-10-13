"""``grasppanda prepare``: verify a dataset tree and build its derived assets.

Derived assets depend on how a dataset is configured (the preprocessing chain, the
graspness parameters, the per-view grasp parameters), so every action reads a dataset
config: the shipped base config of the dataset and camera unless ``--config`` names
another. What gets built is exactly what that config will later ask for; nothing is
written into the dataset tree.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

from grasppanda.config import Config
from grasppanda.registry import DATASETS
from grasppanda.utils.console import style

__all__ = ["register"]

#: Options that build an asset, in the order they are built.
_ASSET_FLAGS = (
    "--labels",
    "--graspness",
    "--compact-labels",
    "--view-grasps",
    "--tolerance",
    "--scale-statistic",
)


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "prepare",
        help="verify a dataset tree and build derived assets (grasp labels, graspness and the "
        "label assets of individual methods)",
        description=(
            "Verify a dataset tree and build the derived assets a dataset config reads. "
            "Generation is resumable: finished shards are skipped."
        ),
    )
    parser.add_argument("dataset", help="dataset name, e.g. graspnet1b")
    parser.add_argument("--root", help="dataset directory (default: resolved as in configs)")
    parser.add_argument(
        "--check", action="store_true", help="report what the tree and the assets lack"
    )
    parser.add_argument(
        "--labels", action="store_true", help="convert grasp labels into the memory-mapped store"
    )
    parser.add_argument(
        "--graspness", action="store_true", help="generate point-wise graspness maps"
    )
    parser.add_argument(
        "--import-legacy",
        action="store_true",
        help="with --graspness: import upstream-format files instead of generating",
    )
    parser.add_argument(
        "--compact-labels",
        action="store_true",
        help="collapse the grasp labels to one grasp per point and view (needs a --config "
        "whose dataset declares `compact_labels`, e.g. an EconomicGrasp config)",
    )
    parser.add_argument(
        "--view-grasps",
        action="store_true",
        help="generate per-view grasp labels (the `view_grasps` section of the dataset config)",
    )
    parser.add_argument(
        "--tolerance",
        action="store_true",
        help="generate grasp tolerance labels (needs a --config whose dataset has a "
        "`tolerance` section)",
    )
    parser.add_argument(
        "--scale-statistic",
        action="store_true",
        help="build the grasp-width histogram of Scale-Balanced-Grasp (needs a --config whose "
        "dataset has a `scale_statistic` section); with --import-legacy, import the reference "
        "file named by --scale-statistic-file",
    )
    parser.add_argument(
        "--scale-statistic-file", metavar="PATH", help="reference file to import as scale statistic"
    )
    parser.add_argument("--camera", help="camera whose shipped base config is used")
    parser.add_argument("--split", help="split whose scenes are prepared (default: the config's)")
    parser.add_argument("--scenes", help="scene ids instead of a split, e.g. 0-99 or 3,7,100-129")
    parser.add_argument(
        "--config", help="dataset config to prepare for (default: shipped base config)"
    )
    parser.add_argument(
        "--loader",
        default="train",
        choices=["train", "val", "test"],
        help="which `data.<loader>.dataset` of the config to read (default: train)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="processes for --graspness, --view-grasps and --tolerance",
    )
    parser.set_defaults(handler=run)


def _dataset_class(name: str) -> Any:
    known = {}
    for component in DATASETS.names():
        cls = DATASETS.get(component)
        known[getattr(cls, "name", component)] = cls
    if name not in known:
        raise SystemExit(f"unknown dataset '{name}'. Datasets: {', '.join(sorted(known))}")
    return known[name]


def _custom_imports(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """Out-of-tree modules of ``--config`` (imported by loading it), for worker processes."""
    if not args.config:
        return []
    config = Config.load(args.config)
    entries = config.get("custom_imports") or []
    return [(entry, config.source_of(f"custom_imports.{i}")) for i, entry in enumerate(entries)]


def _node(args: argparse.Namespace, cls: Any, camera: str | None) -> dict[str, Any]:
    """The dataset config node the requested action works for."""
    source = args.config or cls.default_config(camera)
    node = Config.load(source).select(f"data.{args.loader}.dataset")
    if not isinstance(node, dict):
        raise SystemExit(f"{source} has no data.{args.loader}.dataset to prepare for")
    node = dict(node)
    node["check_assets"] = False
    node["load_labels"] = True
    if args.root:
        node["root"] = args.root
    if camera:
        node["camera"] = camera
    if args.split:
        node["split"] = args.split
    return node


def _scenes(args: argparse.Namespace, dataset: Any) -> list[int]:
    if args.scenes:
        from grasppanda.datasets.manifest import expand_ranges

        return list(expand_ranges([part for part in args.scenes.split(",") if part]))
    return list(dataset.scenes())


def _progress(total: int, unit: str) -> Any:
    from tqdm import tqdm

    return tqdm(total=total, unit=unit, dynamic_ncols=True, disable=None)


def _ok(text: str) -> None:
    print(style("✓ ", "bold", "green") + text)


def _bad(text: str) -> None:
    print(style("✗ ", "bold", "red") + text)


# ------------------------------------------------------------------------------ check


def _check(args: argparse.Namespace, cls: Any) -> int:
    from grasppanda.config import ConfigError
    from grasppanda.utils.paths import cache_root, dataset_root

    cameras = [args.camera] if args.camera else list(cls.cameras)
    nodes: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        try:
            nodes[camera] = _node(args, cls, camera)
        except ConfigError:
            if args.config:
                raise
    configured = next((node.get("root") for node in nodes.values() if node.get("root")), None)
    root = dataset_root(cls.name, args.root or configured)
    print(style(f"{cls.name} at {root}", "bold"))
    problems = cls.check_tree(root, cameras)
    if problems:
        _bad(f"the tree lacks {len(problems)} item(s):")
        for problem in problems[:40]:
            print(f"    {problem}")
        if len(problems) > 40:
            print(f"    … and {len(problems) - 40} more")
        return 1
    _ok(f"tree is complete for {', '.join(cameras)}")
    print(style(f"derived assets under {cache_root()}", "bold"))
    incomplete = False
    for camera in cameras:
        if camera not in nodes:
            print(f"  {camera}: no shipped dataset config; pass --config to report its assets")
            continue
        dataset = DATASETS.build(nodes[camera])
        scenes = _scenes(args, dataset)
        objects = dataset.object_ids(scenes)
        missing = dataset.label_store.missing(objects, scenes)
        total = len(objects) + len(scenes)
        where = f"{camera} · {dataset.split} · {len(scenes)} scenes"
        report = _ok if not missing else _bad
        report(
            f"{where}: grasp labels {total - len(missing)}/{total} shards "
            f"({dataset.label_store.asset.spec.hash})"
        )
        incomplete |= bool(missing)
        maps = dataset.graspness
        if maps is not None and hasattr(maps, "asset"):
            from grasppanda.datasets.graspness import scene_shard

            shards = [scene_shard(scene) for scene in scenes]
            lacking = shards if not maps.asset.exists() else maps.asset.missing(shards)
            report = _ok if not lacking else _bad
            report(
                f"{where}: graspness {len(shards) - len(lacking)}/{len(shards)} scenes "
                f"({maps.asset.spec.hash}, {maps.asset.spec.provenance}, preprocessing "
                f"{dataset.fingerprint})"
            )
            incomplete |= bool(lacking)
        compact = dataset.compact_labels
        if compact is not None:
            from grasppanda.datasets.compact_labels import scene_shard as compact_shard

            shards = [compact_shard(scene) for scene in scenes]
            lacking = shards if not compact.asset.exists() else compact.asset.missing(shards)
            report = _ok if not lacking else _bad
            report(
                f"{where}: compact grasp labels {len(shards) - len(lacking)}/{len(shards)} "
                f"scenes ({compact.asset.spec.hash})"
            )
            incomplete |= bool(lacking)
        views = dataset.view_grasps
        if views is not None:
            from grasppanda.datasets.view_grasps import scene_shard as view_shard

            shards = [view_shard(scene) for scene in scenes]
            lacking = shards if not views.asset.exists() else views.asset.missing(shards)
            report = _ok if not lacking else _bad
            report(
                f"{where}: view grasps {len(shards) - len(lacking)}/{len(shards)} scenes "
                f"({views.asset.spec.hash})"
            )
            incomplete |= bool(lacking)
        if dataset.tolerance is not None:
            absent = dataset.tolerance.missing(objects)
            report = _ok if not absent else _bad
            report(
                f"{where}: tolerance {len(objects) - len(absent)}/{len(objects)} objects "
                f"({dataset.tolerance.asset.spec.hash})"
            )
            incomplete |= bool(absent)
    if incomplete:
        print(
            f"build what is missing with: grasppanda prepare {cls.name} " + " / ".join(_ASSET_FLAGS)
        )
    return 1 if incomplete else 0


# ----------------------------------------------------------------------------- labels


def _labels(args: argparse.Namespace, cls: Any) -> int:
    dataset = DATASETS.build(_node(args, cls, args.camera))
    scenes = _scenes(args, dataset)
    objects = dataset.object_ids(scenes)
    store = dataset.label_store
    print(f"grasp-label store: {store.asset.path}")
    with _progress(len(objects) + len(scenes), "shard") as bar:

        def advance(shard: str) -> None:
            bar.set_postfix_str(shard)
            bar.update(1)

        counts = store.build(objects, scenes, progress=advance)
    _ok(
        f"{counts['built']} shards built and verified against their sources, "
        f"{counts['skipped']} already present"
    )
    return 0


# -------------------------------------------------------------------------- graspness


def _graspness_scene(
    node: dict[str, Any],
    scene: int,
    workers: int,
    imports: Sequence[tuple[str, str | None]] = (),
) -> tuple[int, dict[str, int]]:
    """Generate or import one scene. Runs in a worker process; writes the shard file only."""
    from grasppanda.config.validate import import_custom
    from grasppanda.datasets.graspness import generate_graspness, import_legacy_graspness

    for entry, declared_in in imports:  # a fresh process has not seen out-of-tree components
        import_custom(entry, declared_in)
    dataset = DATASETS.build(node)
    config, maps = dataset.graspness_config, dataset.graspness
    if config.source == "legacy_import":
        record = import_legacy_graspness(
            dataset, maps, scene, dataset.legacy_graspness(), config.dtype
        )
    else:
        record = generate_graspness(dataset, maps, scene, config, workers=workers)
    return scene, record


def _graspness(args: argparse.Namespace, cls: Any) -> int:
    from grasppanda.datasets.graspness import scene_shard

    node = _node(args, cls, args.camera)
    graspness = dict(node.get("graspness") or {})
    if not graspness:
        raise SystemExit(
            "the dataset config has no `graspness` section, so there is nothing to generate for it"
        )
    graspness["source"] = "legacy_import" if args.import_legacy else "native"
    node["graspness"] = graspness
    dataset = DATASETS.build(node)
    scenes = _scenes(args, dataset)
    maps = dataset.graspness
    if not args.import_legacy:
        objects = dataset.object_ids(scenes)
        dataset.label_store.require(
            objects, scenes, build=f"grasppanda prepare {cls.name} --labels"
        )
    maps.asset.create()
    pending = [scene for scene in scenes if not maps.asset.is_complete(scene_shard(scene))]
    print(f"graspness asset: {maps.asset.path}")
    print(
        f"  {dataset.camera} · preprocessing {dataset.fingerprint} · "
        f"{maps.asset.spec.provenance} · {len(scenes) - len(pending)} of {len(scenes)} scenes "
        "already present"
    )
    if not pending:
        _ok("nothing to do")
        return 0
    workers = max(1, min(args.workers, len(pending)))
    with _progress(len(pending), "scene") as bar:
        if workers == 1:
            for scene in pending:
                _, record = _graspness_scene(node, scene, -1)
                maps.mark_complete(scene, record)
                bar.update(1)
        else:
            # Workers only write shard files; this process alone updates the manifest.
            imports = _custom_imports(args)
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
                jobs = [pool.submit(_graspness_scene, node, scene, 1, imports) for scene in pending]
                for job in as_completed(jobs):
                    scene, record = job.result()
                    maps.mark_complete(scene, record)
                    bar.update(1)
    _ok(f"{len(pending)} scenes written")
    return 0


# --------------------------------------------------------------------- compact labels


def _compact_labels(args: argparse.Namespace, cls: Any) -> int:
    from grasppanda.datasets.compact_labels import generate_compact_labels, scene_shard

    node = _node(args, cls, args.camera)
    if not node.get("compact_labels"):
        raise SystemExit(
            "the dataset config has no `compact_labels` section, so there is nothing to generate "
            "for it. Pass --config with a config that declares one (an EconomicGrasp config)."
        )
    dataset = DATASETS.build(node)
    scenes = _scenes(args, dataset)
    store = dataset.compact_labels
    # The dense grids are the input: the dataset itself no longer asks for them.
    dataset.label_store.require(
        dataset.object_ids(scenes), scenes, build=f"grasppanda prepare {cls.name} --labels"
    )
    store.asset.create()
    pending = [scene for scene in scenes if not store.asset.is_complete(scene_shard(scene))]
    print(f"compact grasp labels: {store.asset.path}")
    print(f"  {len(scenes) - len(pending)} of {len(scenes)} scenes already present")
    if not pending:
        _ok("nothing to do")
        return 0
    kept = annotated = 0
    with _progress(len(pending), "scene") as bar:
        for scene in pending:
            record = generate_compact_labels(dataset, store, scene)
            store.mark_complete(scene, record)
            kept += sum(record["kept"])
            annotated += sum(record["annotated"])
            bar.update(1)
    _ok(f"{len(pending)} scenes written and read back: {kept} of {annotated} annotated points kept")
    return 0


# ------------------------------------------------------------------------ view grasps


def _view_grasps_scene(
    node: dict[str, Any],
    scene: int,
    threads: int,
    imports: Sequence[tuple[str, str | None]] = (),
) -> tuple[int, dict[str, int]]:
    """Generate one scene. Runs in a worker process; writes the shard file only."""
    from grasppanda.config.validate import import_custom
    from grasppanda.datasets.view_grasps import generate_view_grasps

    for entry, declared_in in imports:  # a fresh process has not seen out-of-tree components
        import_custom(entry, declared_in)
    if threads > 0:  # one of several processes: do not start a thread pool in each
        import torch

        torch.set_num_threads(threads)
    dataset = DATASETS.build(node)
    record = generate_view_grasps(dataset, dataset.view_grasps, scene, dataset.view_grasp_config)
    return scene, record


def _view_grasps(args: argparse.Namespace, cls: Any) -> int:
    from grasppanda.datasets.view_grasps import scene_shard

    node = _node(args, cls, args.camera)
    if node.get("view_grasps") is None:
        raise SystemExit(
            "the dataset config has no `view_grasps` section, so there is nothing to generate "
            "for it; name a config that has one with --config"
        )
    dataset = DATASETS.build(node)
    scenes = _scenes(args, dataset)
    store = dataset.view_grasps
    dataset.label_store.require(
        dataset.object_ids(scenes), scenes, build=f"grasppanda prepare {cls.name} --labels"
    )
    store.asset.create()
    pending = [scene for scene in scenes if not store.asset.is_complete(scene_shard(scene))]
    print(f"view-grasp asset: {store.asset.path}")
    print(
        f"  {dataset.camera} · {len(scenes) - len(pending)} of {len(scenes)} scenes already present"
    )
    if not pending:
        _ok("nothing to do")
        return 0
    workers = max(1, min(args.workers, len(pending)))
    grasps = 0
    with _progress(len(pending), "scene") as bar:
        if workers == 1:
            for scene in pending:
                _, record = _view_grasps_scene(node, scene, 0)
                store.mark_complete(scene, record)
                grasps += record["grasps"]
                bar.update(1)
        else:
            # Workers only write shard files; this process alone updates the manifest.
            imports = _custom_imports(args)
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
                jobs = [
                    pool.submit(_view_grasps_scene, node, scene, 1, imports) for scene in pending
                ]
                for job in as_completed(jobs):
                    scene, record = job.result()
                    store.mark_complete(scene, record)
                    grasps += record["grasps"]
                    bar.update(1)
    _ok(f"{len(pending)} scenes written, {grasps} grasp labels")
    return 0


# -------------------------------------------------------------------------- tolerance


def _tolerance_object(
    node: dict[str, Any], object_id: int, imports: Sequence[tuple[str, str | None]] = ()
) -> tuple[int, dict[str, Any]]:
    """Generate one object. Runs in a worker process; writes the shard file only."""
    from grasppanda.config.validate import import_custom

    for entry, declared_in in imports:  # a fresh process has not seen out-of-tree components
        import_custom(entry, declared_in)
    dataset = DATASETS.build(node)
    points, scores, _ = dataset.label_store.object_arrays(object_id)
    return object_id, dataset.tolerance.write(object_id, points, scores)


def _tolerance(args: argparse.Namespace, cls: Any) -> int:
    node = _node(args, cls, args.camera)
    if not node.get("tolerance"):
        raise SystemExit(
            "the dataset config has no `tolerance` section, so there is nothing to generate for "
            "it; pass --config with the config of a method that supervises grasp tolerance"
        )
    dataset = DATASETS.build(node)
    objects = sorted(dataset.object_ids(_scenes(args, dataset)))
    store = dataset.tolerance
    dataset.label_store.require(objects, [], build=f"grasppanda prepare {cls.name} --labels")
    store.asset.create()
    pending = store.missing(objects)
    print(f"tolerance asset: {store.asset.path}")
    print(f"  {len(objects) - len(pending)} of {len(objects)} objects already present")
    if not pending:
        _ok("nothing to do")
        return 0
    workers = max(1, min(args.workers, len(pending)))
    with _progress(len(pending), "object") as bar:
        if workers == 1:
            for object_id in pending:
                store.mark_complete(*_tolerance_object(node, object_id))
                bar.update(1)
        else:
            # Workers only write shard files; this process alone updates the manifest.
            imports = _custom_imports(args)
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
                jobs = [pool.submit(_tolerance_object, node, i, imports) for i in pending]
                for job in as_completed(jobs):
                    store.mark_complete(*job.result())
                    bar.update(1)
    _ok(f"{len(pending)} objects written and verified")
    return 0


# -------------------------------------------------------------------- scale statistic


def _scale_statistic(args: argparse.Namespace, cls: Any) -> int:
    import numpy as np
    import torch

    node = _node(args, cls, args.camera)
    section = dict(node.get("scale_statistic") or {})
    if not section:
        raise SystemExit(
            "the dataset config has no `scale_statistic` section, so there is nothing to build "
            "for it; pass --config with the config of a method that uses scale-balanced weights"
        )
    section["source"] = "legacy_import" if args.import_legacy else "native"
    if not args.import_legacy:
        section.pop("reference_sha256", None)
        node["seed"] = section.get("seed", 0)  # a fixed function of the configuration
    node["scale_statistic"] = section
    dataset = DATASETS.build(node)
    store = dataset.scale_statistic
    print(f"scale statistic: {store.asset.path} ({store.asset.spec.provenance})")
    if store.exists():
        _ok("nothing to do")
        return 0
    if args.import_legacy:
        if not args.scale_statistic_file:
            raise SystemExit("--scale-statistic --import-legacy needs --scale-statistic-file")
        store.import_reference(args.scale_statistic_file)
        edges, counts = store.load()
        _ok(
            f"imported {len(counts)} bins, {int(counts.sum())} entries, verified against its SHA-256"
        )
        return 0
    from grasppanda.models.assigners.scale_balance import collect_seed_scales

    scenes = dataset.scenes()
    dataset.label_store.require(
        dataset.object_ids(scenes), scenes, build=f"grasppanda prepare {cls.name} --labels"
    )
    if dataset.tolerance is not None:
        dataset.tolerance.require(
            dataset.object_ids(scenes), build=f"grasppanda prepare {cls.name} --tolerance"
        )
    config = store.config
    frames = len(range(0, len(dataset), config.frame_step))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  {frames} frames of {dataset.split} (every {config.frame_step}th), seeds on {device}")
    with _progress(frames, "frame") as bar:
        widths = collect_seed_scales(
            dataset,
            frame_step=config.frame_step,
            num_seeds=config.num_seeds,
            min_graspable_views=config.min_graspable_views,
            graspable_score=config.graspable_score,
            max_width=config.max_width,
            device=device,
            progress=lambda done: bar.update(done - bar.n),
        ).numpy()
    if len(widths) == 0:
        raise SystemExit("no graspable seed was found; the statistic cannot be built")
    counts, edges = np.histogram(widths, bins=config.bins)
    store.save(edges, counts, frames=frames, seeds_counted=len(widths))
    _ok(f"{len(widths)} graspable seeds of {frames} frames in {config.bins} bins")
    return 0


def run(args: argparse.Namespace) -> int:
    from grasppanda.datasets.assets import AssetError
    from grasppanda.structures import ContractError

    _custom_imports(args)  # registers out-of-tree datasets named by --config
    cls = _dataset_class(args.dataset)
    actions: Sequence[tuple[bool, Callable[[argparse.Namespace, Any], int]]] = (
        (args.check, _check),
        (args.labels, _labels),
        (args.graspness, _graspness),
        (args.compact_labels, _compact_labels),
        (args.view_grasps, _view_grasps),
        (args.tolerance, _tolerance),
        (args.scale_statistic, _scale_statistic),
    )
    if args.import_legacy and not (args.graspness or args.scale_statistic):
        print("--import-legacy goes with --graspness or --scale-statistic", file=sys.stderr)
        return 2
    if not any(selected for selected, _ in actions):
        print(
            f"nothing to do: pass --check or one of {', '.join(_ASSET_FLAGS)} (those after "
            "--graspness need a --config that declares the asset)",
            file=sys.stderr,
        )
        return 2
    status = 0
    try:
        for selected, action in actions:
            if selected:
                status = max(status, action(args, cls))
    except (AssetError, ContractError, FileNotFoundError) as error:
        print(style("✗ ", "bold", "red") + str(error), file=sys.stderr)
        return 1
    return status
