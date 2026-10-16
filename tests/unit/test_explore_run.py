"""The Run panel of ``grasppanda explore``: one frame through a method, under the server's rules.

Everything runs on the CPU with the toy detectors of ``inference_fixtures`` (a point-cloud
method on the GraspNet-1Billion recipe with a collision-filter variant, and an image method
that reads the colour image like HGGD); nothing is downloaded.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from grasppanda.apis import GraspInferencer, example_scene
from grasppanda.cli.main import main
from grasppanda.combine import explore
from grasppanda.combine.explore_run import (
    BODY_LIMIT,
    STAGES,
    UPLOAD_LIMIT,
    Busy,
    RunError,
    Runner,
    WeightsSource,
    _no_downloads,
    suggest,
)

from .inference_fixtures import IMAGE_CONFIG, POINT_CONFIG, save_weights, write_config
from .test_engine_tiny import single_thread  # noqa: F401  (autouse fixture)
from .test_explore_live import call

SIZE = (320, 180)  # the uploaded camera files: the example scene, small
K = [231.25, 231.25, 160.0, 90.0]  # its intrinsics at that size
BOX = [-0.25, -0.2, 0.35, 0.25, 0.2, 0.8]


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    """Two configs, a weights folder with traps in it, a file outside it, camera files."""
    work = tmp_path_factory.mktemp("run")
    point = write_config(work, POINT_CONFIG, "point.yaml")
    image = write_config(work, IMAGE_CONFIG, "image.yaml")
    folder = work / "weights"
    (folder / "sub" / "deeper").mkdir(parents=True)
    (folder / ".hidden").mkdir()
    save_weights(point, folder / "toy_point.pth")
    save_weights(image, folder / "toy_image.pth")
    save_weights(point, folder / "sub" / "deeper" / "deep.pth")
    save_weights(point, folder / ".hidden" / "secret.pth")
    (folder / "notes.txt").write_text("not weights")
    outside = save_weights(point, work / "outside.pth")
    (folder / "link.pth").symlink_to(outside)  # a link that leaves the folder
    camera = example_scene(size=SIZE).save(work / "camera")
    return {
        "work": work,
        "point": point,
        "image": image,
        "folder": folder,
        "outside": outside,
        **camera,
    }


@pytest.fixture(scope="module")
def runner(files) -> Runner:
    source = WeightsSource.from_arguments(
        files["folder"], [f"toy={files['folder'] / 'toy_point.pth'}"]
    )
    methods = {"Toy": str(files["point"]), "ImageToy": str(files["image"])}
    return Runner(source, methods=methods, device="cpu", picture_size=(160, 120))


@pytest.fixture(scope="module")
def live(runner, tmp_path_factory):
    saves = tmp_path_factory.mktemp("saves")
    httpd = explore.server(0, saves, runner)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", saves
    httpd.shutdown()
    httpd.server_close()


def upload(path: Path, name: str | None = None) -> dict[str, str]:
    return {"name": name or path.name, "data": base64.b64encode(path.read_bytes()).decode()}


def answer(base: str, payload: dict, headers: dict | None = None) -> tuple[int, dict]:
    status, _, body = call(base, "/api/run", payload, headers)
    return status, json.loads(body)


def finished(base: str, payload: dict) -> dict:
    """Start a run and wait for its end; returns its last status."""
    status, started = answer(base, payload)
    assert status == 202, started
    # A run that fails on its input at once may have ended before the answer is written
    # (seen on a loaded machine); the page polls whatever state it gets.
    assert started["state"] in ("running", "failed", "done"), started["state"]
    assert [s["name"] for s in started["stages"]] == list(STAGES)
    deadline = time.monotonic() + 120
    while True:
        code, _, body = call(base, f"/api/run/{started['id']}")
        assert code == 200
        state = json.loads(body)
        if state["state"] != "running":
            return state
        assert time.monotonic() < deadline, "the run did not end"
        time.sleep(0.05)


def example(**extra) -> dict:
    return {"method": "Toy", "weights": "toy", "input": "example", **extra}


def depth_request(files, **extra) -> dict:
    return {
        "method": "Toy",
        "weights": "toy_point.pth",
        "input": "depth",
        "depth": upload(files["depth"]),
        "intrinsics": K,
        **extra,
    }


# ------------------------------------------------------------------------- weights


def test_the_weights_are_the_files_the_command_line_named_and_nothing_else(files):
    folder, outside = files["folder"], files["outside"]
    source = WeightsSource.from_arguments(folder, [f"toy={folder / 'toy_point.pth'}"])
    entries, cut = source.entries()
    assert [entry["id"] for entry in entries] == [
        "toy",
        "toy_image.pth",
        "toy_point.pth",
        "sub/deeper/deep.pth",
    ]
    assert not cut and entries[0]["origin"] == "command line"
    assert {entry["origin"] for entry in entries[1:]} == {"directory"}

    assert source.resolve("toy") == (folder / "toy_point.pth").resolve()
    assert source.resolve("sub/deeper/deep.pth") == (folder / "sub/deeper/deep.pth").resolve()
    assert source.resolve(str(folder / "toy_image.pth")) == (folder / "toy_image.pth").resolve()
    assert source.resolve(" toy_image.pth ") == (folder / "toy_image.pth").resolve()

    refused = {
        "../outside.pth": "does not read",
        "sub/../../outside.pth": "does not read",
        str(outside): "does not read",
        "link.pth": "does not read",  # inside by name, outside once resolved
        "/etc/passwd": "does not read",
        str(files["work"] / "missing.pth"): "does not read",
        "notes.txt": "not a weights file",
        "missing.pth": "there is no weights file",
        "": "choose a weights file",
        "a\x00b.pth": "no weights file",
    }
    for text, reason in refused.items():
        with pytest.raises(RunError, match=reason) as error:
            source.resolve(text)
        assert error.value.kind == "weights", text
    with pytest.raises(RunError, match="choose"):
        source.resolve(None)
    # A path outside is refused in the same words whether it exists or not.
    words = []
    for path in (str(outside), str(files["work"] / "missing.pth")):
        with pytest.raises(RunError) as error:
            source.resolve(path)
        words.append(str(error.value).replace(path, "P"))
    assert words[0] == words[1]


def test_a_large_folder_is_listed_in_part_and_still_readable(files, monkeypatch):
    from grasppanda.combine import explore_run

    source = WeightsSource.from_arguments(files["folder"])
    monkeypatch.setattr(explore_run, "_WALK_LIMIT", 1)  # the top folder only
    entries, cut = source.entries()
    assert cut and [entry["id"] for entry in entries] == ["toy_image.pth", "toy_point.pth"]
    assert source.resolve("sub/deeper/deep.pth").name == "deep.pth", "typed, it is read"
    monkeypatch.setattr(explore_run, "_LIST_LIMIT", 1)
    monkeypatch.setattr(explore_run, "_WALK_LIMIT", 100)
    assert source.entries()[1] is True and len(source.entries()[0]) == 1


def test_the_command_line_is_checked_when_the_server_starts(files):
    folder = files["folder"]
    bare = WeightsSource.from_arguments(None, [str(folder / "toy_point.pth")])
    assert list(bare.named) == ["toy_point"] and bare.directory is None
    for directory, named, reason in (
        (files["work"] / "nope", [], "no such directory"),
        (None, ["x=" + str(folder / "missing.pth")], "no such file"),
        (None, ["bad name=" + str(folder / "toy_point.pth")], "plain word"),
        (None, [f"a={folder / 'toy_point.pth'}", f"a={folder / 'toy_image.pth'}"], "twice"),
        (None, ["=x.pth"], "write NAME=PATH"),
    ):
        with pytest.raises(ValueError, match=reason):
            WeightsSource.from_arguments(directory, named)
    assert WeightsSource().empty and not bare.empty


def test_a_name_suggests_weights_and_the_run_decides():
    entries = [
        {"id": "gsnet", "label": "gsnet · epoch_0010.pth", "origin": "command line"},
        {
            "id": "gsnet_author_realsense.pth",
            "label": "gsnet_author_realsense.pth",
            "origin": "directory",
        },
        {"id": "hggd_author_kinect.pth", "label": "hggd_author_kinect.pth", "origin": "directory"},
        {
            "id": "hggd_author_realsense.pth",
            "label": "hggd_author_realsense.pth",
            "origin": "directory",
        },
        {
            "id": "baseline_author_realsense.pth",
            "label": "baseline_author_realsense.pth",
            "origin": "directory",
        },
    ]
    assert suggest("GSNet", "gsnet/gsnet_sunet14_g1b-realsense_10e", entries) == "gsnet"
    assert (
        suggest("HGGD", "hggd/hggd_r34w8_g1b-realsense_15e", entries) == "hggd_author_realsense.pth"
    )
    assert suggest("HGGD", "hggd/hggd_r34w8_g1b-kinect_15e", entries) == "hggd_author_kinect.pth"
    assert suggest("GraspNet-baseline", "x", entries) == "baseline_author_realsense.pth"
    assert suggest("FineGrasp", "finegrasp/x", entries) == ""


def test_a_composed_selection_downloads_nothing():
    body = {
        "model": {
            "image_encoder": {"type": "Timm", "pretrained": True},
            "heads": [{"pretrained": True}],
        }
    }
    switched, changed = _no_downloads(body)
    assert changed and switched["model"]["image_encoder"]["pretrained"] is False
    assert switched["model"]["heads"][0]["pretrained"] is False
    assert body["model"]["image_encoder"]["pretrained"] is True  # the composition is not touched
    assert _no_downloads({"a": {"pretrained": False}}) == ({"a": {"pretrained": False}}, False)


# ---------------------------------------------------------------------------- runs


def test_the_panel_says_what_it_offers(live):
    base, _ = live
    status, _, body = call(base, "/api/run/setup")
    setup = json.loads(body)
    assert status == 200
    assert [m["name"] for m in setup["methods"]] == ["Toy", "ImageToy"]
    assert setup["methods"][0]["needs_rgb"] is False and setup["methods"][1]["needs_rgb"] is True
    assert setup["methods"][0]["suggested"] == "toy"  # named after the method, on the command line
    assert [entry["id"] for entry in setup["weights"]][:2] == ["toy", "toy_image.pth"]
    assert setup["limits"]["upload_bytes"] == UPLOAD_LIMIT == 20 * 1024 * 1024
    assert setup["stages"] == list(STAGES) and setup["running"] is None
    page = call(base, "/")[2].decode()
    assert 'data-view="run"' in page, "a server with weights opens on the Run panel"


def test_a_run_on_the_example_scene_gives_what_demo_gives(live, monkeypatch, tmp_path):
    base, saves = live
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    state = finished(base, example(collision_filter=True))
    assert state["state"] == "done", state.get("error")
    assert [stage["state"] for stage in state["stages"]] == ["done"] * len(STAGES)
    result = state["result"]
    with Image.open(io.BytesIO(base64.b64decode(result["picture"]))) as picture:
        assert picture.format == "PNG" and picture.size == (160, 120)
    array = np.load(io.BytesIO(base64.b64decode(result["array"])))
    assert array.shape == (result["total"], 17) == tuple(result["array_shape"])
    assert array.dtype == np.float32 and (np.diff(array[:, 0]) <= 0).all()  # best first
    assert result["predicted"] == 32 and 0 < result["total"] <= 32
    assert result["counts"] == [{"step": "ModelFreeCollisionFilter", "count": result["total"]}]
    rows = result["grasps"]
    assert len(rows) == min(10, result["total"]) and rows[0]["rank"] == 1
    assert rows[0]["score"] == pytest.approx(float(array[0, 0]), abs=1e-4)
    assert rows[0]["position"] == pytest.approx(array[0, 13:16].tolist(), abs=1e-4)
    assert rows[0]["approach"] == pytest.approx(array[0, [4, 7, 10]].tolist(), abs=1e-3)
    lines = dict(result["lines"])
    assert "synthetic example scene: 1280 x 720" in lines["input"]
    assert "ToyDetector on cpu" in lines["model"] and "8 matched, 0 missing" in lines["weights"]
    assert result["command"].startswith(f"grasppanda demo {state['target'] and ''}")
    assert result["command"].endswith("--example --collision-filter --device cpu")
    assert result["array_name"] == "toy_grasps.npy" and result["input"] == "the example scene"
    assert not list(saves.iterdir()), "a run writes nothing where configs are saved"
    # PyTorch 2.14 puts its compiler cache (torchinductor_<user>) into the temporary
    # directory the first time it loads inductor: torch's, not the run's.
    left = [entry for entry in tmp_path.iterdir() if not entry.name.startswith("torchinductor_")]
    assert not left, "the temporary directory of a run is removed"

    again = finished(base, example())  # same method and weights: the model is kept
    assert again["state"] == "done"
    assert again["stages"][0]["note"] == "kept from the last run" and again["result"]["reused"]
    assert again["result"]["counts"] == [] and again["result"]["total"] == 32


def test_the_example_scene_is_passed_as_demo_passes_it(runner, files):
    from grasppanda.cli.demo import _call_arguments

    inferencer = GraspInferencer(files["point"], files["folder"] / "toy_point.pth", device="cpu")
    ours, info = runner._call(SimpleNamespace(kind="example"), inferencer)
    theirs, their_info = _call_arguments(
        SimpleNamespace(workspace=None, seed=None), "--example", inferencer
    )
    assert sorted(ours) == sorted(theirs)
    for key, value in theirs.items():
        assert np.array_equal(np.asarray(ours[key]), np.asarray(value)), key
    for key in ("caption", "line"):
        assert info[key] == their_info[key]


def test_an_uploaded_depth_image_in_millimetres_or_in_metres(live, files, tmp_path):
    base, _ = live
    state = finished(base, depth_request(files, workspace=BOX, collision_filter=True))
    assert state["state"] == "done", state.get("error")
    result = state["result"]
    assert result["input"] == "depth.png" and result["notes"] == []
    assert dict(result["lines"])["input"].startswith(
        "depth.png: 320 x 180 uint16, read as millimetres"
    )
    assert (
        "--depth depth.png --intrinsics 231.25,231.25,160,90 --workspace -0.25 -0.2 0.35 0.25 0.2 0.8"
        in result["command"]
    )

    metres = tmp_path / "scene.npy"
    with Image.open(files["depth"]) as image:
        np.save(metres, np.asarray(image).astype(np.float32) / 1000.0)
    request = depth_request(files, depth=upload(metres), intrinsics="231.25, 231.25, 160, 90")
    request["rgb"] = upload(files["rgb"])
    state = finished(base, request)
    assert state["state"] == "done", state.get("error")
    result = state["result"]
    assert "float32, read as metres; colour rgb.png" in dict(result["lines"])["input"]
    # No box and no mask: the run says what the config's crop would have done.
    assert any("every pixel with depth is used" in note for note in result["notes"])


def test_input_problems_are_refused_before_anything_runs(live, files, tmp_path):
    base, _ = live
    colour = tmp_path / "colour.png"
    Image.new("RGB", SIZE, (200, 10, 10)).save(colour)
    wide = tmp_path / "wide.png"
    Image.new("I;16", (4097, 2)).save(wide)
    pickled = tmp_path / "objects.npy"
    np.save(pickled, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"\xff\xd8\xff" + b"0" * 64)
    cases = [
        ({"intrinsics": None}, "needs the intrinsics"),
        ({"intrinsics": [925, 925, 640]}, "needs the intrinsics"),
        ({"intrinsics": [925, "x", 640, 360]}, "needs the intrinsics"),
        ({"workspace": [0, 0, 0, 1, 1]}, "six numbers"),
        ({"workspace": [0, 0, 1, 1, 1, 0.5]}, "workspace is empty"),
        ({"depth": upload(colour)}, "colour.png has 3 channels, so it is a picture of depth"),
        ({"depth": upload(wide)}, "up to 4096 pixels"),
        ({"depth": upload(pickled)}, "cannot be read as a NumPy array"),
        ({"depth": upload(fake)}, "fake.png is not a PNG file"),
        ({"depth": upload(files["depth"], "depth.exr")}, "read from .png, .npy"),
        ({"depth": {"name": "d.png", "data": "not base64!"}}, "did not arrive as base64"),
        ({"depth": None}, "choose a depth image"),
        ({"rgb": upload(files["depth"], "rgb.gif")}, "read from .png, .jpg"),
    ]
    for change, reason in cases:
        status, body = answer(base, depth_request(files, **change))
        assert status == 400 and reason in body["error"], (change, body)
        assert str(tmp_path) not in body["error"] and "grasppanda-run-" not in body["error"]
    for request, reason in (
        (example(intrinsics=K), "cannot be combined"),
        (example(input="video"), "`input` is `example` or `depth`"),
        (example(method="Nobody"), "unknown method"),
        (example(sets="post=nms"), "list of `slot=option`"),
        (example(collision_filter="yes"), "true or false"),
    ):
        status, body = answer(base, request)
        assert status == 400 and reason in body["error"], (request, body)


def test_weights_that_do_not_fit_and_missing_inputs_fail_in_the_words_of_demo(live, files):
    base, _ = live
    state = finished(base, example(method="ImageToy", weights="toy_point.pth"))
    assert state["state"] == "failed"
    error = state["error"]
    assert error["kind"] == "weights" and error["title"].startswith("These weights do not fit")
    assert error["message"].startswith("WeightMismatchError:")
    assert (
        "must have been trained for (or converted to) the model of this config" in error["message"]
    )
    assert [stage["state"] for stage in state["stages"]] == ["failed"] + ["waiting"] * 4

    request = depth_request(files, method="ImageToy", weights="toy_image.pth", workspace=BOX)
    state = finished(base, request)
    assert state["state"] == "failed" and state["error"]["kind"] == "input"
    assert "ImageToyDetector reads the colour image" in state["error"]["message"]

    state = finished(base, example(sets=["post=nope"]))
    assert state["state"] == "failed" and state["error"]["kind"] == "config"
    assert "no post-processing step 'nope'" in state["error"]["message"]


def test_a_composed_selection_runs_with_weights_that_fit(live):
    base, _ = live
    state = finished(base, example(sets=["post=nms"], collision_filter=True))
    assert state["state"] == "done", state.get("error")
    result = state["result"]
    assert [step["step"] for step in result["counts"]] == ["ModelFreeCollisionFilter", "GraspNMS"]
    assert result["target"] == "Toy with post=nms"
    first, second = result["command"].splitlines()
    assert first.startswith("grasppanda compose ") and first.endswith(
        "--set post=nms -o selection.yaml"
    )
    assert second.startswith("grasppanda demo selection.yaml ") and "--post nms" in second
    assert dict(result["lines"])["config"].startswith("composed: Toy with post=nms")


# -------------------------------------------------------------------------- safety


def test_a_run_answers_its_own_page_only(live):
    base, _ = live
    foreign = {"Host": "attacker.example"}
    assert call(base, "/api/run/setup", headers=foreign)[0] == 403
    assert answer(base, example(), foreign)[0] == 403
    assert answer(base, example(), {"Origin": "http://attacker.example"})[0] == 403
    assert answer(base, example(), {"Origin": "null"})[0] == 403
    status, _, _ = call(base, "/api/run", None, None)  # a GET of the action
    assert status == 404
    request = http.client.HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=30)
    request.request(
        "POST", "/api/run", body=json.dumps(example()), headers={"Content-Type": "text/plain"}
    )
    assert request.getresponse().status == 415, "a form of another site cannot start a run"
    request.close()
    status, _, body = call(base, "/api/run/not-a-run")
    assert status == 404 and "keeps the last run only" in json.loads(body)["error"]


def test_an_oversize_request_is_refused_unread(live, files):
    base, _ = live
    port = int(base.rsplit(":", 1)[1])
    for path, limit in (("/api/run", BODY_LIMIT), ("/api/compose", 65536)):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(limit + 1))
        connection.endheaders()  # and no body: the server must not wait for it
        response = connection.getresponse()
        assert response.status == 413 and json.loads(response.read())["kind"] == "too-large"
        connection.close()
    big = {
        "name": "big.png",
        "data": base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(UPLOAD_LIMIT)).decode(),
    }
    status, body = answer(base, depth_request(files, depth=big))
    assert status == 413 and body["kind"] == "too-large" and "20.0 MB" in body["error"]
    assert call(base, "/api/run/setup")[0] == 200, "the server still answers"


def test_weights_are_named_never_opened_from_a_path_of_the_page(live, files):
    base, _ = live
    for text in ("nope.pth", "../outside.pth", str(files["outside"]), "link.pth", "/etc/passwd"):
        status, body = answer(base, example(weights=text))
        assert status == 400 and body["kind"] == "weights", text
    status, body = answer(base, example(weights=str(files["outside"])))
    assert "does not read" in body["error"] and "--weights NAME=FILE" in body["error"]
    state = finished(base, example(weights="sub/deeper/deep.pth"))
    assert state["state"] == "done", "a file deeper in the folder is read"


def test_one_run_at_a_time(live, runner, monkeypatch):
    base, _ = live
    gate = threading.Event()
    entered = threading.Event()

    def slow(job, plan):
        job.enter(0)
        entered.set()
        assert gate.wait(60)
        return {"slow": True}

    monkeypatch.setattr(runner, "_predict", slow)
    answers: list[tuple[int, dict]] = []
    racers = [
        threading.Thread(target=lambda: answers.append(answer(base, example()))) for _ in range(2)
    ]
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join(60)
    assert sorted(status for status, _ in answers) == [202, 409]
    busy = next(body for status, body in answers if status == 409)
    assert busy["kind"] == "busy" and "another run is in progress" in busy["error"]
    assert entered.wait(30)
    setup = json.loads(call(base, "/api/run/setup")[2])
    assert setup["running"]["state"] == "running" and "result" not in setup["running"]
    with pytest.raises(Busy, match="one at a time"):
        runner.start(example())
    gate.set()
    first = next(body for status, body in answers if status == 202)
    deadline = time.monotonic() + 30
    while json.loads(call(base, f"/api/run/{first['id']}")[2])["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    status, again = answer(base, example())
    assert status == 202, "the next run is accepted once the first has ended"
    deadline = time.monotonic() + 30
    while json.loads(call(base, f"/api/run/{again['id']}")[2])["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_a_server_without_weights_says_how_to_give_some(files, tmp_path):
    httpd = explore.server(0, tmp_path, Runner(methods={"Toy": str(files["point"])}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert "data-view" not in call(base, "/")[2].decode()
        setup = json.loads(call(base, "/api/run/setup")[2])
        assert setup["weights"] == [] and setup["weights_dir"] == ""
        status, body = answer(base, example(weights="anything.pth"))
        assert status == 400 and "--weights-dir" in body["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------------------- command, page


def test_the_command_describes_the_run_panel(capsys, tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["explore", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for words in ("--weights-dir DIR", "--weights NAME=PATH", "--device", "Run panel", "20 MB"):
        assert words in text
    assert main(["explore", "--no-browser", "--weights-dir", str(tmp_path / "no")]) == 2
    assert "no such directory" in capsys.readouterr().err
    assert main(["explore", "--no-browser", "--weights", f"x={tmp_path / 'no.pth'}"]) == 2
    assert main(["explore", "--no-browser", "--weights", "=x"]) == 2


def test_the_documentation_page_shows_the_panel_greyed_with_the_command():
    root = explore.explorer_root()
    script = (root / "explorer.js").read_text(encoding="utf-8")
    assert (
        '"grasppanda explore --weights-dir <dir>"' in script and "Runs on your machine." in script
    )
    assert '["run", "Run a method", "Run"]' in script
    assert "grasppanda explore --weights-dir" in (root / "index.md").read_text(encoding="utf-8")
