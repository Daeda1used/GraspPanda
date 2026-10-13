"""``grasppanda demo`` end to end on a CPU: the example scene, files of a camera, a cloud."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grasppanda.apis import example_scene
from grasppanda.cli.main import PLANNED, main

from .inference_fixtures import IMAGE_CONFIG, POINT_CONFIG, save_weights, write_config
from .test_engine_tiny import single_thread  # noqa: F401  (autouse fixture)

SMALL = ["--size", "240", "180", "--gif-frames", "3"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict[str, Path]:
    """Two configs with weights, and the example scene written as the files a camera gives."""
    work = tmp_path_factory.mktemp("demo")
    point = write_config(work, POINT_CONFIG, "point.yaml")
    image = write_config(work, IMAGE_CONFIG, "image.yaml")
    files = example_scene(size=(320, 180)).save(work / "camera")
    return {
        "work": work,
        "point": point,
        "point_weights": save_weights(point, work / "point.pth"),
        "image": image,
        "image_weights": save_weights(image, work / "image.pth"),
        **{f"file_{name}": path for name, path in files.items()},
    }


def base(setup, kind: str = "point") -> list[str]:
    return ["demo", str(setup[kind]), "--weights", str(setup[f"{kind}_weights"])]


def camera_files(setup) -> list[str]:
    return [
        "--depth",
        str(setup["file_depth"]),
        "--intrinsics",
        str(setup["file_intrinsics"]),
    ]


def test_demo_is_a_command_and_no_longer_planned(capsys):
    assert main([]) == 0
    assert "  demo " in capsys.readouterr().out
    assert "demo" not in {name for name, _, _ in PLANNED}
    with pytest.raises(SystemExit) as stop:
        main(["demo", "--help"])
    usage = capsys.readouterr().out
    assert stop.value.code == 0
    for flag in ("--example", "--depth", "--intrinsics", "--points", "--frame", "--show"):
        assert flag in usage
    assert "x right, y down, z forward" in usage and "millimetres" in usage


def test_the_example_scene_gives_grasps_a_picture_and_an_animation(setup, tmp_path, capsys):
    out = tmp_path / "out"
    assert main([*base(setup), "--example", "--collision-filter", "--out", str(out), *SMALL]) == 0
    printed = capsys.readouterr().out
    assert "synthetic example scene: 1280 x 720, 6 primitives" in printed
    assert (
        "not a camera image" in printed and "workspace from the scene's own object mask" in printed
    )
    assert "ToyDetector on cpu, reads points" in printed
    assert "SegmentationBoxWorkspace" in printed and "RandomSample(num_points=1024)" in printed
    assert "32 predicted in" in printed and "left after ModelFreeCollisionFilter" in printed
    assert "x right, y down, z forward" in printed
    grasps = np.load(out / "grasps.npy")
    assert grasps.ndim == 2 and grasps.shape[1] == 17 and grasps.dtype == np.float32
    assert (np.diff(grasps[:, 0]) <= 0).all()  # best first
    assert 0.3 < grasps[:, 15].mean() < 1.0  # on the table, metres in front of the camera
    with Image.open(out / "grasps.png") as still:
        assert still.size == (240, 180)
        assert len(still.getcolors(maxcolors=1 << 20)) > 50  # a picture, not a blank canvas
    with Image.open(out / "grasps.gif") as animation:
        assert animation.n_frames == 3 and animation.size == (160, 120)
    for name in ("depth.png", "rgb.png", "mask.png", "intrinsics.txt", "table_from_camera.txt"):
        assert (out / "example" / name).is_file()
    assert (out / "example/intrinsics.txt").read_text().strip() == "925,925,640,360"


def test_files_of_a_camera(setup, tmp_path, capsys):
    out = tmp_path / "out"
    arguments = [*base(setup), *camera_files(setup), "--out", str(out), "--gif-frames", "0"]
    workspace = ["--workspace", "-0.25", "-0.2", "0.3", "0.25", "0.2", "0.9"]  # negatives parse
    assert main([*arguments, *workspace, "--size", "200", "150"]) == 0
    printed = capsys.readouterr().out
    assert "320 x 180 uint16, read as millimetres" in printed  # the default for an integer image
    assert "note" not in printed  # a stated workspace needs no remark
    assert (out / "grasps.png").is_file() and not (out / "grasps.gif").exists()
    assert np.load(out / "grasps.npy").shape == (32, 17)

    assert main([*arguments, "--mask", str(setup["file_mask"])]) == 0
    assert "aligned with the camera axes" in capsys.readouterr().out
    table = ["--table-from-camera", str(setup["file_table_from_camera"])]
    assert main([*arguments, "--mask", str(setup["file_mask"]), *table]) == 0
    assert "note" not in capsys.readouterr().out  # now it is the config's own crop
    assert main(arguments) == 0  # no mask at all: every pixel, and a remark that says so
    assert "No mask was given, so every pixel with depth is used" in capsys.readouterr().out


def test_the_usual_mistakes_end_with_a_reason(setup, tmp_path, capsys):
    out = ["--out", str(tmp_path / "out")]
    depth = ["--depth", str(setup["file_depth"])]

    assert main([*base(setup), *out]) == 1
    assert "give exactly one input" in capsys.readouterr().err
    assert main([*base(setup), *depth, *out]) == 1  # no intrinsics
    error = capsys.readouterr().err
    assert "--depth needs --intrinsics fx,fy,cx,cy" in error and "point" in error
    wrong_unit = [*base(setup), *camera_files(setup), "--depth-scale", "1", *out]
    assert main(wrong_unit) == 1
    assert (
        "probably millimetres: pass depth_scale=1000 (--depth-scale 1000)"
        in capsys.readouterr().err
    )
    empty = [
        *base(setup),
        *camera_files(setup),
        "--workspace",
        "0",
        "0",
        "2",
        "0.1",
        "0.1",
        "3",
        *out,
    ]
    assert main(empty) == 1
    error = capsys.readouterr().err
    assert "the workspace is empty" in error and "The cloud spans x" in error
    assert main([*base(setup, "image"), *camera_files(setup), *out]) == 1  # needs the colour image
    error = capsys.readouterr().err
    assert "ImageToyDetector reads the colour image" in error and "--rgb FILE" in error
    assert (
        main(
            [
                *base(setup),
                "--depth",
                str(setup["file_rgb"]),
                "--intrinsics",
                "200,200,160,90",
                *out,
            ]
        )
        == 1
    )
    assert "a picture of depth, not a depth image" in capsys.readouterr().err
    assert main([*base(setup), "--points", str(tmp_path / "missing.ply"), *out]) == 1
    assert "point cloud not found" in capsys.readouterr().err
    assert main([*base(setup), "--example", "--rgb", "x.png", *out]) == 1
    assert "--rgb belong to --depth" in capsys.readouterr().err
    assert main(["demo", str(setup["point"]), "--example", *out]) == 1  # no weights
    assert "needs weights" in capsys.readouterr().err
    assert main([*base(setup), "--example", "--variant", "nms", *out]) == 1
    assert "no post-processing variant 'nms'" in capsys.readouterr().err
    assert main(["demo", "no/such/config", "--example", *out]) == 2  # a config problem: 2
    assert not (tmp_path / "out").exists()  # nothing is written for a refused input


def test_an_image_method_and_a_point_cloud_file(setup, tmp_path, capsys):
    out = tmp_path / "image"
    arguments = [*base(setup, "image"), *camera_files(setup), "--rgb", str(setup["file_rgb"])]
    assert main([*arguments, "--out", str(out), *SMALL]) == 0
    assert (
        "ImageToyDetector on cpu, reads depth, intrinsics, points, rgb" in capsys.readouterr().out
    )
    assert np.load(out / "grasps.npy").shape == (6, 17)
    with Image.open(out / "grasps.png") as still:  # coloured by the image that was given
        colours = np.asarray(still.convert("RGB")).reshape(-1, 3)
    assert len(np.unique(colours, axis=0)) > 100

    scene = example_scene(size=(320, 180))
    v, u = np.nonzero(scene.mask > 0)
    z = scene.depth[v, u] / 1000.0
    cloud = np.stack(
        [(u - 160) * z / scene.intrinsics[0, 0], (v - 90) * z / scene.intrinsics[1, 1], z], 1
    )
    file = tmp_path / "objects.npy"
    np.save(file, cloud)
    out = tmp_path / "cloud"
    assert main([*base(setup), "--points", str(file), "--out", str(out), *SMALL]) == 0
    printed = capsys.readouterr().out
    assert f"{len(cloud):,} points" in printed and "crop it to the workspace yourself" in printed
    assert np.load(out / "grasps.npy").shape == (32, 17)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is not None, reason="Open3D is installed")
def test_show_needs_open3d_and_says_how_to_get_it(setup, tmp_path, capsys):
    assert main([*base(setup), "--example", "--show", "--out", str(tmp_path / "out")]) == 1
    error = capsys.readouterr().err
    assert "needs 'open3d'" in error and 'pip install -e ".[vis]"' in error
    assert not (tmp_path / "out").exists()  # refused before the model was built


def test_a_full_gpu_is_explained_and_other_failures_keep_their_traceback(
    setup, monkeypatch, capsys
):
    from grasppanda.cli import demo

    def out_of_memory(args):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(demo, "_demo", out_of_memory)
    assert main([*base(setup), "--example"]) == 1
    error = capsys.readouterr().err
    assert "CUDA error: out of memory" in error and "--device cpu" in error

    def bug(args):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(demo, "_demo", bug)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        main([*base(setup), "--example"])
