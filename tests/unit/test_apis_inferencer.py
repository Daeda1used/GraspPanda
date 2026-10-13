"""``GraspInferencer``: the config's pipeline on user input, its checks and its messages."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from grasppanda.apis import (
    GraspInferencer,
    InferenceWarning,
    InputError,
    example_scene,
    parse_frame_key,
)
from grasppanda.cli.main import main
from grasppanda.engine.errors import EngineError
from grasppanda.structures import GraspGroup, SampleKey
from grasppanda.transforms.frame import transform_points

from .inference_fixtures import (
    IMAGE_CONFIG,
    MODEL_ONLY_CONFIG,
    POINT_CONFIG,
    TOY_CONFIG,
    save_weights,
    write_config,
)
from .test_engine_tiny import single_thread  # noqa: F401  (autouse fixture)


@pytest.fixture(scope="module")
def scene():
    return example_scene(size=(320, 180))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("inference")


@pytest.fixture(scope="module")
def point_inferencer(workdir) -> GraspInferencer:
    config = write_config(workdir, POINT_CONFIG, "point.yaml")
    return GraspInferencer(config, save_weights(config, workdir / "point.pth"))


@pytest.fixture(scope="module")
def image_inferencer(workdir) -> GraspInferencer:
    config = write_config(workdir, IMAGE_CONFIG, "image.yaml")
    return GraspInferencer(config, save_weights(config, workdir / "image.pth"))


def camera(scene) -> dict:
    """The arguments a calibrated camera with a segmentation would pass."""
    return {
        "depth": scene.depth,
        "intrinsics": scene.intrinsics,
        "depth_scale": scene.depth_scale,
        "mask": scene.mask,
        "table_from_camera": scene.table_from_camera,
    }


# ----------------------------------------------------------------------- the pipeline


def test_a_depth_image_goes_through_the_steps_of_the_config(point_inferencer, scene):
    assert [type(step).__name__ for step in point_inferencer.preprocess] == [
        "DepthToCloud",
        "SegmentationBoxWorkspace",
    ]
    assert [type(step).__name__ for step in point_inferencer.transforms] == [
        "RandomSample",
        "ConstantFeatures",
    ]
    observation = point_inferencer.observe(**camera(scene))
    assert len(observation.points) == 1024 and observation.point_feats.shape == (1024, 3)
    assert observation.points.frame == "camera" and observation.rgb is None

    # The workspace is the configured one: the box around the masked pixels, in the table frame.
    v, u = np.nonzero(scene.depth > 0)
    z = scene.depth[v, u] / 1000.0
    cloud = np.stack(
        [
            (u - scene.intrinsics[0, 2]) * z / scene.intrinsics[0, 0],
            (v - scene.intrinsics[1, 2]) * z / scene.intrinsics[1, 1],
            z,
        ],
        axis=1,
    )
    on_table = transform_points(cloud, scene.table_from_camera)
    objects = on_table[scene.mask[v, u] > 0]
    low, high = objects.min(axis=0) - 0.02, objects.max(axis=0) + 0.02
    expected = cloud[((on_table > low) & (on_table < high)).all(axis=1)]
    full = observation.full_points.coords.numpy()
    assert full.shape == expected.shape and np.allclose(full, expected, atol=1e-6)
    assert 0 < len(full) < len(cloud)  # a crop, with table around the objects left in

    # Sampling draws without replacement from that cloud, under the seed of the config.
    order = np.random.default_rng(0).choice(len(expected), 1024, replace=False)
    assert np.allclose(observation.points.coords.numpy(), expected[order], atol=1e-6)


def test_a_call_returns_grasps_in_the_camera_frame_and_repeats_exactly(point_inferencer, scene):
    grasps = point_inferencer(**camera(scene))
    assert isinstance(grasps, GraspGroup) and len(grasps) == 32 and grasps.frame == "camera"
    grasps.validate()
    array = grasps.to_graspnet_array()
    assert array.shape == (32, 17) and array.dtype == np.float32
    again = point_inferencer(**camera(scene))
    assert np.array_equal(array, again.to_graspnet_array())
    other = point_inferencer(**camera(scene), seed=5)
    assert not np.array_equal(array, other.to_graspnet_array())  # another sample of the cloud
    assert np.array_equal(
        other.to_graspnet_array(), point_inferencer(**camera(scene), seed=5).to_graspnet_array()
    )


def test_integer_millimetres_and_float_metres_are_the_same_scene(point_inferencer, scene):
    millimetres = point_inferencer.observe(**camera(scene))
    arguments = {**camera(scene), "depth": scene.depth / 1000.0, "depth_scale": 1.0}
    metres = point_inferencer.observe(**arguments)
    assert torch.allclose(millimetres.points.coords, metres.points.coords, atol=1e-6)
    four = point_inferencer.observe(
        **{**camera(scene), "intrinsics": [scene.intrinsics[0, 0], scene.intrinsics[1, 1], 160, 90]}
    )
    assert torch.equal(four.points.coords, millimetres.points.coords)


def test_a_point_cloud_skips_the_image_steps_and_keeps_the_point_steps(point_inferencer, scene):
    cloud = point_inferencer.observe(**camera(scene)).full_points.coords.numpy()
    holes = np.concatenate([cloud, np.full((3, 3), np.nan)])  # an organised cloud marks holes so
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observation = point_inferencer.observe(points=holes)
    assert any("crop it to the workspace yourself" in str(w.message) for w in caught)
    assert len(observation.points) == 1024 and observation.intrinsics is None
    assert len(observation.full_points) == len(cloud)
    assert len(point_inferencer(points=cloud)) == 32
    boxed = point_inferencer.observe(points=cloud, workspace=(-1, -1, 0, 0.0, 1, 2))
    assert (boxed.full_points.coords[:, 0] < 0).all()


def test_an_image_method_gets_its_images_and_seeded_draws(image_inferencer, scene):
    assert image_inferencer.required_inputs == {"rgb", "depth", "intrinsics", "points"}
    assert image_inferencer.images == ("rgb", "depth")
    arguments = {
        "depth": scene.depth,
        "intrinsics": scene.intrinsics,
        "rgb": scene.rgb,
        "depth_scale": 1000.0,
    }
    observation = image_inferencer.observe(**arguments)
    assert observation.rgb.shape == (1, 3, 180, 320) and float(observation.rgb.max()) <= 1.0
    assert observation.depth.shape == (1, 180, 320) and observation.full_points is None
    assert np.isclose(float(observation.depth.max()), scene.depth.max() / 1000.0)  # metres
    assert len(observation.points) == 512
    first = image_inferencer(**arguments).to_graspnet_array()
    assert np.array_equal(first, image_inferencer(**arguments).to_graspnet_array())  # same draw
    assert not np.array_equal(first, image_inferencer(**arguments, seed=3).to_graspnet_array())
    state = torch.get_rng_state()
    image_inferencer(**arguments)
    assert torch.equal(state, torch.get_rng_state())  # the caller's random stream is untouched
    as_float = image_inferencer.observe(**{**arguments, "rgb": scene.rgb / 255.0})
    assert torch.equal(as_float.rgb, observation.rgb)


def test_without_a_declared_pipeline_depth_is_back_projected_and_nothing_else(workdir, scene):
    config = write_config(workdir, MODEL_ONLY_CONFIG, "model_only.yaml")
    inferencer = GraspInferencer(config, save_weights(config, workdir / "model_only.pth"))
    assert inferencer.preprocess == [] and inferencer.transforms == []
    assert "none declared" in inferencer.describe()
    observation = inferencer.observe(
        depth=scene.depth, intrinsics=scene.intrinsics, depth_scale=1000
    )
    assert len(observation.points) == int((scene.depth > 0).sum())
    with pytest.raises(InputError, match="names no dataset"):
        inferencer(frame="toy:scene_0200/toy/0000")


# ------------------------------------------------------------- what the input lacks


def test_missing_inputs_are_named(image_inferencer, point_inferencer, scene):
    depth = {"depth": scene.depth, "depth_scale": 1000.0}
    with pytest.raises(InputError, match=r"needs the intrinsics.*--intrinsics fx,fy,cx,cy"):
        point_inferencer(**depth)
    with pytest.raises(InputError, match=r"ImageToyDetector reads the colour image.*rgb="):
        image_inferencer(**depth, intrinsics=scene.intrinsics)
    with pytest.raises(InputError) as raised:
        image_inferencer(points=np.random.default_rng(0).uniform(0.3, 0.6, size=(2000, 3)))
    message = str(raised.value)
    assert "ImageToyDetector reads depth, intrinsics, points, rgb" in message
    assert "the points input provides points" in message
    assert "rgb: the colour image" in message and "depth: a depth image" in message
    with pytest.raises(InputError, match="exactly one input.*got none"):
        point_inferencer()
    with pytest.raises(InputError, match="exactly one input.*depth and points"):
        point_inferencer(depth=scene.depth, intrinsics=scene.intrinsics, points=np.zeros((4, 3)))


def test_a_depth_image_in_the_wrong_unit_is_refused_with_the_fix(point_inferencer, scene):
    with pytest.raises(InputError, match=r"probably millimetres: pass depth_scale=1000"):
        point_inferencer(depth=scene.depth, intrinsics=scene.intrinsics)  # default: metres
    with pytest.raises(InputError, match="units per\\s+metre"):
        point_inferencer(depth=scene.depth / 1000.0, intrinsics=scene.intrinsics, depth_scale=1000)
    with pytest.raises(InputError, match="divided by 1000"):
        point_inferencer(points=np.random.default_rng(0).uniform(300, 600, size=(500, 3)))
    with pytest.raises(InputError, match="no positive value"):
        point_inferencer(depth=np.zeros((180, 320)), intrinsics=scene.intrinsics)
    with pytest.raises(InputError, match="picture of depth|must be a \\(H, W\\) array"):
        point_inferencer(depth=np.zeros((180, 320, 3)), intrinsics=scene.intrinsics)


def test_an_empty_workspace_says_where_the_cloud_is(point_inferencer, scene):
    with pytest.raises(InputError) as raised:
        point_inferencer(
            depth=scene.depth,
            intrinsics=scene.intrinsics,
            depth_scale=1000,
            workspace=(-0.1, -0.1, 2.0, 0.1, 0.1, 3.0),
        )
    message = str(raised.value)
    assert "the workspace is empty" in message and "The cloud spans x " in message
    assert "x right, y down, z forward" in message
    plain = {"depth": scene.depth, "intrinsics": scene.intrinsics, "depth_scale": 1000}
    with pytest.raises(InputError, match="workspace is empty: lower corner"):
        point_inferencer(**plain, workspace=(0, 0, 1, 0, 0, 0))
    with pytest.raises(InputError, match="six numbers"):
        point_inferencer(**plain, workspace=(0, 1, 2))


def test_mismatched_images_and_calibration_are_refused(image_inferencer, point_inferencer, scene):
    arguments = {"depth": scene.depth, "intrinsics": scene.intrinsics, "depth_scale": 1000.0}
    with pytest.raises(InputError, match="rgb is 100 x 90 but the depth image is 320 x 180"):
        image_inferencer(**arguments, rgb=np.zeros((90, 100, 3), dtype=np.uint8))
    with pytest.raises(InputError, match="mask is \\(90, 100\\)"):
        point_inferencer(**arguments, mask=np.ones((90, 100)))
    with pytest.raises(InputError, match="marks no object"):
        point_inferencer(**arguments, mask=np.zeros_like(scene.mask))
    with pytest.raises(InputError, match="do not fit a 320 x 180 image"):
        point_inferencer(**{**arguments, "intrinsics": (925, 925, 640, 360)})  # of a larger image
    with pytest.raises(InputError, match="belong to a depth image"):
        point_inferencer(points=np.zeros((10, 3)), rgb=scene.rgb)


def test_differences_from_the_test_conditions_are_warned_about_once(workdir, scene):
    config = write_config(workdir, POINT_CONFIG, "warned.yaml")
    inferencer = GraspInferencer(config, workdir / "point.pth")
    plain = {"depth": scene.depth, "intrinsics": scene.intrinsics, "depth_scale": 1000.0}
    with pytest.warns(InferenceWarning, match="No mask was given, so every pixel with depth"):
        uncropped = inferencer.observe(**plain)
    assert len(uncropped.full_points) == int((scene.depth > 0).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the second time it stays quiet
        inferencer.observe(**plain)
        boxed = inferencer.observe(**plain, workspace=(-0.2, -0.2, 0.3, 0.2, 0.2, 0.8))
    assert len(boxed.full_points) < len(uncropped.full_points)
    with pytest.warns(InferenceWarning, match="aligned with the camera axes"):
        camera_box = inferencer.observe(**plain, mask=scene.mask)
    exact = inferencer.observe(**camera(scene))
    assert len(camera_box.full_points) != len(
        exact.full_points
    )  # the same kind of crop, not the same


def test_an_image_pipeline_says_that_it_ignores_a_mask(image_inferencer, scene):
    with pytest.warns(InferenceWarning, match="mask is ignored.*NoWorkspace"):
        image_inferencer.observe(
            depth=scene.depth,
            intrinsics=scene.intrinsics,
            rgb=scene.rgb,
            depth_scale=1000,
            mask=scene.mask,
        )


# -------------------------------------------------------------------- post-processing


def test_post_processors_of_the_config_run_on_request(point_inferencer, image_inferencer, scene):
    assert point_inferencer.variants == ["raw", "cd", "centre"]
    observation = point_inferencer.observe(**camera(scene))
    raw = point_inferencer.predict(observation)
    filtered = point_inferencer.predict(observation, collision_filter=True)
    (step,) = point_inferencer.chain(collision_filter=True)
    assert type(step).__name__ == "ModelFreeCollisionFilter"
    assert np.array_equal(filtered.to_graspnet_array(), step(raw, observation).to_graspnet_array())
    assert len(filtered) <= len(raw)
    assert np.array_equal(
        point_inferencer.predict(observation, variant="cd").to_graspnet_array(),
        filtered.to_graspnet_array(),
    )
    centre = point_inferencer(**camera(scene), variant="centre")
    assert len(centre) <= len(raw)
    with pytest.raises(InputError, match="no post-processing variant 'nms'.*raw, cd, centre"):
        point_inferencer(**camera(scene), variant="nms")
    with pytest.raises(InputError, match="this config has none"):
        image_inferencer.chain(collision_filter=True)


# --------------------------------------------------------------------------- weights


def test_weights_are_required_and_must_fit(workdir):
    config = write_config(workdir, POINT_CONFIG, "weights.yaml")
    with pytest.raises(EngineError, match="needs weights.*untrained model is never run"):
        GraspInferencer(config)
    with pytest.raises(EngineError, match="weights file not found"):
        GraspInferencer(config, workdir / "nowhere.pth")
    with pytest.raises(EngineError, match="trained for \\(or converted to\\) the model"):
        GraspInferencer(config, workdir / "image.pth")
    report = GraspInferencer(config, workdir / "point.pth").weights_report
    assert report.clean and len(report.matched) == 8


def test_the_zoo_is_asked_when_no_weights_are_given():
    with pytest.raises(EngineError) as raised:
        GraspInferencer("graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e")
    message = str(raised.value)
    assert "no weights trained by GraspPanda" in message and "not downloaded for you" in message
    assert "tools.convert_checkpoint" in message and "README.md" in message


# -------------------------------------------------------------------- dataset frames


def test_frame_keys():
    key = SampleKey("graspnet1b", 100, "realsense", 0)
    assert parse_frame_key("graspnet1b:scene_0100/realsense/0000") == key
    assert parse_frame_key(str(key)) == key and parse_frame_key(key) is key
    assert parse_frame_key("graspclutter6d:scene_12/realsense-d435/3").camera == "realsense-d435"
    with pytest.raises(InputError, match="does not name a dataset frame"):
        parse_frame_key("scene_0100/realsense/0000")


def test_a_dataset_frame_gives_what_the_engine_dumps_for_it(workdir, capsys):
    """Same weights, same frames: ``grasppanda infer`` (the predict of ``test``) and the inferencer."""
    weights = save_weights(TOY_CONFIG, workdir / "toy.pth")
    output = workdir / "dumped"
    arguments = ["infer", TOY_CONFIG, "--weights", str(weights), "--input", "test"]
    assert main([*arguments, "--output", str(output), "--opts", "data.test.batch_size=1"]) == 0
    capsys.readouterr()
    inferencer = GraspInferencer(TOY_CONFIG, weights)
    for frame in ("scene_0200/toy/0000", "scene_0200/toy/0005", "scene_0201/toy/0007"):
        for variant in ("raw", "centre"):
            dumped = np.load(output / "predictions" / variant / f"{frame}.npy")
            mine = inferencer(frame=f"toy:{frame}", variant=variant).to_graspnet_array()
            assert np.array_equal(mine, dumped), (frame, variant)
    with pytest.raises(InputError, match="is not one of the 16 frames"):
        inferencer(frame="toy:scene_0300/toy/0000")
    with pytest.raises(InputError, match="reads 'toy'"):
        inferencer(frame="graspnet1b:scene_0100/realsense/0000")
    with pytest.raises(InputError, match="cannot be combined with frame="):
        inferencer(frame="toy:scene_0200/toy/0000", seed=1)
