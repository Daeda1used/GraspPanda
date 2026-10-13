"""``grasppanda demo``: grasps on one frame, as numbers and as a picture.

    grasppanda demo <config> --weights W --example
    grasppanda demo <config> --weights W --depth depth.png --intrinsics fx,fy,cx,cy [--rgb rgb.png]
    grasppanda demo <config> --weights W --points cloud.ply
    grasppanda demo <config> --weights W --frame graspnet1b:scene_0100/realsense/0000

The input goes through the test pipeline of the config
(:class:`~grasppanda.apis.GraspInferencer`). The command prints the best grasps and
writes, into ``--out``:

* ``grasps.npy``: every returned grasp as a ``(G, 17)`` float32 array in the GraspNet
  layout ``[score, width, height, depth, R (9, row-major), t (3), object_id]``, best
  first, in the camera frame (x right, y down, z forward), metres;
* ``grasps.png``: the cloud with the ``--top-k`` best grasps, coloured by score rank
  from red (lowest shown) to green (highest);
* ``grasps.gif``: the same, swinging around the scene.

``--show`` also opens an interactive window; it needs Open3D
(``pip install "grasppanda[vis]"``) and a display.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from grasppanda import __version__
from grasppanda.utils.console import banner, style, table

__all__ = ["register"]

#: Points drawn at most: a full 1280 x 720 image is drawn as it is, a larger cloud is thinned.
_RENDER_POINTS = 1_000_000


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "demo",
        help="predict grasps on one frame and draw them",
        description=(
            "Predict grasps on a depth image, a point cloud, a dataset frame or the built-in "
            "synthetic example scene, through the test pipeline of a config. Writes the grasps "
            "(.npy), a picture and an animation."
        ),
        epilog=(
            "Units: metres. Frame: the camera's optical frame, x right, y down, z forward. "
            "A 16-bit depth PNG is taken as millimetres unless --depth-scale says otherwise."
        ),
    )
    parser.add_argument("config", help="config file or name of a shipped config")
    parser.add_argument("--weights", metavar="FILE", help="weights for the model of the config")
    source = parser.add_argument_group("input (one of)")
    source.add_argument(
        "--example",
        action="store_true",
        help="the synthetic table-top scene that ships with GraspPanda",
    )
    source.add_argument("--depth", metavar="FILE", help="depth image (.png, .tiff or .npy)")
    source.add_argument(
        "--points", metavar="FILE", help="point cloud (.npy, .ply or .pcd), camera frame, metres"
    )
    source.add_argument(
        "--frame", metavar="KEY", help="dataset frame, e.g. graspnet1b:scene_0100/realsense/0000"
    )
    camera = parser.add_argument_group("with --depth")
    camera.add_argument(
        "--intrinsics",
        metavar="fx,fy,cx,cy",
        help="camera intrinsics in pixels, or a file holding them",
    )
    camera.add_argument("--rgb", metavar="FILE", help="colour image registered to the depth image")
    camera.add_argument("--mask", metavar="FILE", help="object mask (non-zero pixels), same size")
    camera.add_argument(
        "--depth-scale",
        type=float,
        metavar="N",
        help="depth units per metre (default: 1000 for an integer image, 1 for a float image)",
    )
    camera.add_argument(
        "--table-from-camera", metavar="FILE", help="4 x 4 camera-to-table transform (text file)"
    )
    parser.add_argument(
        "--workspace",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="keep only the points inside this box of the camera frame, metres "
        "(six numbers separated by spaces)",
    )
    parser.add_argument(
        "--collision-filter",
        action="store_true",
        help="apply the config's collision-filter variant to the predictions",
    )
    parser.add_argument("--variant", metavar="NAME", help="apply this chain of evaluation.variants")
    parser.add_argument(
        "--post",
        nargs="+",
        metavar="STEP",
        help="post-processing steps applied afterwards, e.g. nms top-k=20 "
        "gripper-limits=0.08 (presets of `grasppanda compose --list post` or component "
        "names; name=value sets the main option, name:option=value,... any option)",
    )
    parser.add_argument(
        "--top-k", type=int, default=30, metavar="K", help="grasps to draw (default 30)"
    )
    parser.add_argument(
        "--seed", type=int, metavar="N", help="seed of the point sampling and the model"
    )
    parser.add_argument("--device", metavar="NAME", help="auto, cpu, cuda or cuda:N")
    parser.add_argument(
        "--out", metavar="DIR", help="output directory (default: work_dirs/<config>/demo)"
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=(960, 720), metavar=("W", "H"), help="picture size"
    )
    parser.add_argument(
        "--gif-frames", type=int, default=24, metavar="N", help="animation frames; 0 writes none"
    )
    parser.add_argument(
        "--show", action="store_true", help="open an interactive window (needs Open3D)"
    )
    parser.add_argument(
        "--opts", nargs="+", default=[], metavar="KEY=VALUE", help="override config values"
    )
    parser.set_defaults(handler=run)


# ---------------------------------------------------------------------------- inputs


def _one_input(args: argparse.Namespace) -> str:
    from grasppanda.apis import InputError

    chosen = [
        name
        for name, value in (
            ("--example", args.example),
            ("--depth", args.depth),
            ("--points", args.points),
            ("--frame", args.frame),
        )
        if value
    ]
    if len(chosen) != 1:
        raise InputError(
            "give exactly one input: --example, --depth FILE (with --intrinsics), --points FILE "
            f"or --frame KEY; got {' and '.join(chosen) or 'none'}. Try --example first: it needs "
            "no data."
        )
    only_depth = [
        flag
        for flag, value in (
            ("--intrinsics", args.intrinsics),
            ("--rgb", args.rgb),
            ("--mask", args.mask),
            ("--depth-scale", args.depth_scale),
            ("--table-from-camera", args.table_from_camera),
        )
        if value is not None
    ]
    if chosen[0] != "--depth" and only_depth:
        raise InputError(
            f"{', '.join(only_depth)} belong to --depth; {chosen[0]} does not use them"
        )
    if chosen[0] == "--depth" and args.intrinsics is None:
        raise InputError(
            "--depth needs --intrinsics fx,fy,cx,cy: the focal lengths and the principal point "
            "of the depth camera in pixels. Without them a depth image cannot become a point "
            "cloud. Your camera's SDK reports them (RealSense: rs-enumerate-devices -c)."
        )
    if chosen[0] in ("--example", "--frame") and args.workspace:
        raise InputError(f"--workspace cannot be combined with {chosen[0]}")
    return chosen[0]


def _call_arguments(args: argparse.Namespace, kind: str, inferencer: Any) -> tuple[dict, dict]:
    """Arguments of the inferencer call, and what the pictures need to know about the input."""
    from grasppanda.apis import example_scene, io

    call: dict[str, Any] = {}
    info: dict[str, Any] = {"rgb": None, "intrinsics": None, "colors": None, "caption": None}
    if args.workspace:
        call["workspace"] = args.workspace
    if args.seed is not None:
        call["seed"] = args.seed
    reads_mask = any(step.uses_ground_truth for step in inferencer.preprocess)
    if kind == "--example":
        scene = example_scene()
        call.update(
            depth=scene.depth,
            intrinsics=scene.intrinsics,
            rgb=scene.rgb,
            depth_scale=scene.depth_scale,
            table_from_camera=scene.table_from_camera,
        )
        crop = ""
        if reads_mask:  # the config crops around object pixels; the scene knows its own
            call["mask"] = scene.mask
            crop = "; workspace from the scene's own object mask"
        height, width = scene.depth.shape
        info.update(
            rgb=scene.rgb,
            intrinsics=scene.intrinsics,
            caption="synthetic example scene",
            scene=scene,
            line=f"synthetic example scene: {width} x {height}, {len(scene.objects)} primitives "
            f"on a table, no sensor noise (not a camera image){crop}",
        )
    elif kind == "--depth":
        depth = io.read_depth(args.depth)
        scale = args.depth_scale
        if scale is None:
            scale = 1000.0 if depth.dtype.kind in "ui" else 1.0
        call.update(depth=depth, intrinsics=io.read_intrinsics(args.intrinsics), depth_scale=scale)
        unit = {1000.0: "millimetres", 1.0: "metres"}.get(scale, f"1/{scale:g} m")
        line = f"{args.depth}: {depth.shape[1]} x {depth.shape[0]} {depth.dtype}, read as {unit}"
        if args.rgb:
            call["rgb"] = info["rgb"] = io.read_rgb(args.rgb)
            line += f"; colour {args.rgb}"
        if args.mask:
            call["mask"] = io.read_mask(args.mask)
            line += f"; mask {args.mask}"
        if args.table_from_camera:
            call["table_from_camera"] = np.loadtxt(args.table_from_camera).reshape(4, 4)
        info.update(intrinsics=call["intrinsics"], line=line)
    elif kind == "--points":
        points, colors = io.read_points(args.points)
        call["points"] = points
        info.update(colors=colors, line=f"{args.points}: {len(points):,} points")
    else:
        call = {"frame": args.frame}
        info["line"] = f"dataset frame {args.frame}, read by the config's dataset reader"
    return call, info


# -------------------------------------------------------------------------- pictures


def _frame_colour(inferencer: Any, key: Any) -> np.ndarray | None:
    """The colour image of a dataset frame, for the picture only; ``None`` if unavailable."""
    from grasppanda.registry import DATASETS

    dataset = inferencer.dataset
    if "rgb" not in getattr(dataset, "served_images", ()) or not hasattr(dataset, "read_frame"):
        return None
    try:  # a second reader that serves the image; the one of the pipeline is left as it is
        node = {**inferencer.cfg["data"][inferencer.section]["dataset"], "images": ["rgb"]}
        return DATASETS.build(node).read_frame(key).rgb
    except Exception:
        return None


def _picture_cloud(observation: Any, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    """The cloud to draw and its colours: what the pipeline kept, at sensor resolution."""
    from grasppanda.visualization.viewer import colors_from_image

    intrinsics = info["intrinsics"]
    if intrinsics is None and observation.intrinsics is not None:
        intrinsics = observation.intrinsics[0].cpu().numpy().astype(np.float64)
    elif intrinsics is not None and np.shape(intrinsics) == (4,):
        fx, fy, cx, cy = intrinsics
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    rgb = info["rgb"]
    if rgb is None and observation.rgb is not None:
        rgb = (observation.rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    colors = info["colors"]
    if observation.full_points is not None:
        points = observation.full_points.coords.cpu().numpy()
    elif observation.depth is not None and intrinsics is not None:
        # The pipeline kept no full cloud (HGGD): draw the depth image it was given.
        depth = observation.depth[0].cpu().numpy().astype(np.float64)
        v, u = np.nonzero(depth > 0)
        z = depth[v, u]
        points = np.stack(
            [
                (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ],
            axis=1,
        )
    else:
        points = observation.points.coords.cpu().numpy()
    if colors is not None and len(colors) != len(points):
        colors = None  # the cloud was filtered or cropped; the file's colours no longer line up
    if colors is None and rgb is not None and intrinsics is not None:
        colors = colors_from_image(points, rgb, intrinsics)
    if len(points) > _RENDER_POINTS:
        # A fixed random subset: every n-th pixel of an image would draw stripes.
        keep = np.sort(np.random.default_rng(0).choice(len(points), _RENDER_POINTS, replace=False))
        points, colors = points[keep], None if colors is None else colors[keep]
    return points, colors


def _caption(image: Any, text: str) -> Any:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), text)
    x, y = 10, image.height - (bottom - top) - 14
    draw.rectangle(
        [x - 4, y - 3, x + (right - left) + 4, y + (bottom - top) + 5], fill=(255, 255, 255)
    )
    draw.text((x, y), text, fill=(60, 60, 60))
    return image


def _draw(
    points, colors, grasps, args: argparse.Namespace, out: Path, caption: str | None
) -> list[tuple[Path, str]]:
    from grasppanda.visualization import fit_camera, orbit_cameras, render, save_gif

    size = tuple(args.size)
    shown = min(args.top_k, len(grasps))
    focus = None
    if shown:  # frame the grasps that are drawn, with some of the scene around them
        centres = grasps.sort_by_score()[:shown].translation.numpy().astype(np.float64)
        pad = 0.12 * np.concatenate([np.eye(3), -np.eye(3)])
        focus = (centres[:, None, :] + pad[None, :, :]).reshape(-1, 3)
    view = fit_camera(points, focus, aspect=size[0] / size[1])
    still = render(points, colors, grasps, camera=view, size=size, top_k=args.top_k)
    if caption:
        _caption(still, caption)
    still.save(out / "grasps.png")
    written = [
        (out / "grasps.png", f"the cloud and the {shown} best grasps, red (lowest shown) to green")
    ]
    if args.gif_frames > 0:
        small = (size[0] * 2 // 3, size[1] * 2 // 3)
        frames = [
            render(
                points, colors, grasps, camera=camera, size=small, top_k=args.top_k, supersample=1
            )
            for camera in orbit_cameras(points, frames=args.gif_frames, base=view)
        ]
        if caption:
            frames = [_caption(frame, caption) for frame in frames]
        save_gif(frames, out / "grasps.gif")
        written.append((out / "grasps.gif", f"{args.gif_frames} frames swinging around the scene"))
    return written


# ------------------------------------------------------------------------------ run


def _shown(path: Path | None) -> str:
    """A path as a reader wants it: relative to here, or as the name of a shipped config."""
    from grasppanda.config import configs_root

    if path is None:
        return ""
    resolved = Path(path).resolve()
    for base, note in ((Path.cwd(), ""), (configs_root().resolve(), "  [shipped config]")):
        try:
            return f"{resolved.relative_to(base)}{note}"
        except ValueError:
            continue
    return str(path)


def _grasp_table(grasps: Any, rows: int) -> str:
    best = grasps.sort_by_score()[:rows]
    array = best.to_graspnet_array()
    body = [
        [
            str(rank + 1),
            f"{row[0]:.3f}",
            f"{row[13]:+.3f}",
            f"{row[14]:+.3f}",
            f"{row[15]:+.3f}",
            f"{row[1]:.3f}",
            f"{row[3]:.3f}",
            f"({row[4]:+.2f}, {row[7]:+.2f}, {row[10]:+.2f})",
        ]
        for rank, row in enumerate(array)
    ]
    return table(
        ["rank", "score", "x", "y", "z", "width", "depth", "approach (x, y, z)"],
        body,
        enabled=False,
    )


def _demo(args: argparse.Namespace) -> None:
    from grasppanda.apis import GraspInferencer, InferenceWarning, parse_frame_key
    from grasppanda.engine.checkpoint import file_sha256
    from grasppanda.utils.imports import require

    kind = _one_input(args)
    if args.show:  # fail before the model is built, not after
        require("open3d", extra="vis", purpose="grasppanda demo --show")
    inferencer = GraspInferencer(args.config, args.weights, device=args.device, overrides=args.opts)
    config = inferencer.config
    name = config.path.stem if config.path else "config"
    out = Path(args.out or Path("work_dirs") / name / "demo").expanduser()

    print(banner(__version__, enabled=False))
    print("  demo")
    report = inferencer.weights_report
    summary = (
        f"{len(report.matched)} matched, {len(report.missing)} missing, "
        f"{len(report.unexpected)} unexpected"
    )
    lines = [
        ("config", f"{_shown(config.path) or '<dict>'}  (hash {config.digest()[:12]})"),
        (
            "weights",
            f"{_shown(inferencer.weights)}  (sha256 {file_sha256(inferencer.weights)[:12]}; "
            f"{summary})",
        ),
        *(tuple(line.split(None, 1)) for line in inferencer.describe().splitlines()),
    ]
    call, info = _call_arguments(args, kind, inferencer)
    lines.append(("input", info["line"]))
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        print(f"  {label:<{width}}  {value}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InferenceWarning)
        observation = inferencer.observe(**call)
    for warning in caught:
        if issubclass(warning.category, InferenceWarning):
            print(f"  {'note':<{width}}  {warning.message}")
    kept = len(observation.full_points) if observation.full_points is not None else None
    cloud = f"{len(observation.points):,} points to the model"
    if kept is not None and kept != len(observation.points):
        cloud += f", sampled from {kept:,} in the workspace"
    assisted = observation.gt_assisted()
    if assisted and kind == "--frame":
        cloud += f"; ground-truth-assisted inputs: {', '.join(assisted)}"
    print(f"  {'cloud':<{width}}  {cloud}")

    chain = inferencer.chain(
        collision_filter=args.collision_filter, variant=args.variant, post=args.post
    )
    started = time.perf_counter()
    raw = inferencer.predict(observation, seed=args.seed)
    seconds = time.perf_counter() - started
    counts, grasps, drawn = [], raw, raw
    for step in chain:  # one step at a time, to say what each one left
        grasps = inferencer.postprocess(grasps, observation, [step])
        counts.append(len(grasps))
        if grasps.frame == raw.frame:  # the picture shows the cloud: grasps in its frame
            drawn = grasps
    grasps, drawn = grasps.sort_by_score(), drawn.sort_by_score()
    result = f"{len(raw)} predicted in {seconds:.2f} s (first call, includes warm-up)"
    if len(chain) == 1:
        result += f"; {len(grasps)} left after {type(chain[0]).__name__}"
    elif chain:  # what every step left
        steps = " → ".join(f"{type(step).__name__} {count}" for step, count in zip(chain, counts))
        result += f"; after {steps}"
    print(f"  {'grasps':<{width}}  {result}")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "grasps.npy", grasps.to_graspnet_array())
    written = [(out / "grasps.npy", f"({len(grasps)}, 17) float32, GraspNet layout, best first")]
    if kind == "--example":
        info["scene"].save(out / "example")
        written.append(
            (out / "example", "the scene as files: depth.png, rgb.png, mask.png, intrinsics.txt")
        )
    if kind == "--frame" and info["rgb"] is None and observation.rgb is None:
        info["rgb"] = _frame_colour(inferencer, parse_frame_key(args.frame))
    points, colors = _picture_cloud(observation, info)
    grasps_cpu, drawn_cpu = grasps.to("cpu"), drawn.to("cpu")
    written[1:1] = _draw(points, colors, drawn_cpu, args, out, info["caption"])

    print()
    if len(grasps):
        print(_grasp_table(grasps_cpu, 10))
        if grasps.frame == "camera":
            print(
                "camera frame, metres: x right, y down, z forward · approach = first column "
                "of the rotation"
            )
        else:
            print(
                f"frame '{grasps.frame}' (a post-processing step changed it), metres · approach "
                "= first column of the rotation · the picture shows the grasps before that step"
            )
    else:
        print(
            "No grasp was returned. That is a valid result: the detector found nothing it "
            "would grasp in this cloud" + (" or the filter removed everything." if chain else ".")
        )
    print()
    for index, (path, what) in enumerate(written):
        print(f"{'wrote' if index == 0 else '     '}  {path}  {what}")
    if args.show:
        from grasppanda.visualization.viewer import show

        show(points, colors, drawn_cpu, top_k=args.top_k, title=f"GraspPanda · {name}")


def run(args: argparse.Namespace) -> int:
    from grasppanda.engine.errors import EngineError
    from grasppanda.registry import BuildError
    from grasppanda.structures import ContractError
    from grasppanda.utils.imports import MissingDependencyError

    try:
        _demo(args)
    except (
        EngineError,
        BuildError,
        ContractError,
        MissingDependencyError,
        FileNotFoundError,
    ) as error:
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        print(f"{mark} {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        # How a GPU without free memory fails: outright, or as cuDNN finding no algorithm.
        reason = str(error).strip().splitlines()[0] if str(error).strip() else ""
        if "out of memory" not in reason and "unable to find an engine" not in reason:
            raise  # anything else is a bug and keeps its traceback
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        print(
            f"{mark} {reason}\n  One frame needs little memory, so the GPU is probably in use "
            "by another process: wait for it, or run this on the CPU with --device cpu.",
            file=sys.stderr,
        )
        return 1
    return 0
