"""``grasppanda explore``: the explorer served locally, answering with the real compose."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from grasppanda.cli.main import main
from grasppanda.combine import explore
from grasppanda.combine.grid import METHODS

GSNET = next(iter(METHODS))


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    directory = tmp_path_factory.mktemp("explore")
    httpd = explore.server(0, directory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", directory, httpd
    httpd.shutdown()
    httpd.server_close()


def call(base: str, path: str, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {**({"Content-Type": "application/json"} if data else {}), **(headers or {})}
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type"), error.read()


def test_it_listens_on_the_loopback_address_only(live):
    _, _, httpd = live
    assert httpd.server_address[0] == explore.HOST == "127.0.0.1"


def test_it_serves_the_page_and_the_files_of_the_documentation_page(live):
    base, _, _ = live
    status, kind, body = call(base, "/")
    assert status == 200 and kind.startswith("text/html")
    page = body.decode()
    assert 'data-live="api/"' in page and 'data-src="explorer.json"' in page
    assert "http" not in page.replace(explore.DOCS, ""), "the page loads nothing from elsewhere"
    for name in explore.STATIC:
        status, _, body = call(base, "/" + name)
        assert status == 200 and body == (explore.explorer_root() / name).read_bytes()
    assert call(base, "/../pyproject.toml")[0] == 404
    assert call(base, "/index.md")[0] == 404


def test_a_selection_gets_the_verdict_of_the_real_compose(live):
    base, _, _ = live
    status, _, body = call(
        base,
        "/api/compose",
        {"method": GSNET, "sets": ["backbone=pointnet2", "grouper=multi-scale"]},
    )
    report = json.loads(body)
    assert status == 200 and report["accepted"] is True
    assert report["adapter"] == "FeaturePropagation"
    assert {stage["slot"] for stage in report["stages"]} >= {"backbone", "grouper", "head"}
    assert "type: MultiScaleGroup" in report["config"] and report["config_hash"]

    status, _, body = call(base, "/api/compose", {"method": GSNET, "sets": ["head=economic"]})
    report = json.loads(body)
    assert status == 200 and report["accepted"] is False and report["config"] == ""
    assert any("EconomicHead" in problem for problem in report["problems"])

    status, _, body = call(base, "/api/compose", {"method": GSNET, "sets": ["backbone=nope"]})
    assert status == 400 and "nope" in json.loads(body)["error"]
    status, _, body = call(base, "/api/compose", {"method": "Nobody", "sets": []})
    assert status == 400 and "unknown method" in json.loads(body)["error"]


def test_an_accepted_config_is_saved_on_request_and_never_overwritten_silently(live):
    base, directory, _ = live
    request = {"method": GSNET, "sets": ["grouper=multi-scale"], "name": "mine.yaml"}
    status, _, body = call(base, "/api/save", request)
    assert status == 200 and json.loads(body)["written"] == "mine.yaml"
    written = (directory / "mine.yaml").read_text(encoding="utf-8")
    assert "type: MultiScaleGroup" in written and "_base_" in written
    assert main(["print-config", str(directory / "mine.yaml"), "--check"]) == 0

    answer = json.loads(call(base, "/api/save", request)[2])
    assert answer["exists"] is True and answer["written"] == ""
    assert json.loads(call(base, "/api/save", {**request, "overwrite": True})[2])["written"]

    refused = {"method": GSNET, "sets": ["head=economic"], "name": "refused.yaml"}
    assert call(base, "/api/save", refused)[0] == 400 and not (directory / "refused.yaml").exists()
    for name in ("../up.yaml", "a/b.yaml", "/abs.yaml", "mine.txt", ".hidden.yaml", ""):
        status, _, _ = call(base, "/api/save", {**request, "name": name})
        assert status == 400, name
    assert sorted(path.name for path in directory.iterdir()) == ["mine.yaml"]


def test_it_answers_its_own_page_only(live):
    base, _, _ = live
    payload = {"method": GSNET, "sets": ["grouper=multi-scale"]}
    assert call(base, "/", headers={"Host": "attacker.example"})[0] == 403
    assert call(base, "/api/compose", payload, {"Host": "attacker.example"})[0] == 403
    assert call(base, "/api/compose", payload, {"Origin": "http://attacker.example"})[0] == 403
    # A form of another site cannot send JSON without a preflight; nothing else is accepted.
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        base + "/api/compose", data=data, headers={"Content-Type": "text/plain"}
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=30)
    assert refused.value.code == 415
    assert call(base, "/api/probe", payload)[0] == 404, "the server never trains anything"


def test_the_command_is_registered(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["explore", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "--port" in text and "--no-browser" in text and "Nothing is trained" in text
    assert main(["explore", "--save-dir", "no/such/dir", "--no-browser"]) == 2
