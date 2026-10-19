"""``grasppanda explore`` answers for every method, not only for the one a page opens with.

The live explorer composes what a visitor selects with the real compose. Under each of the
six methods it accepts the method as shipped and reports the stages and the contract
groups of the seed-based skeleton where the method has them; HGGD, a detector family of
its own, has none of them and is checked through its own slot.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from grasppanda.combine import explore
from grasppanda.combine import options as option_lists
from grasppanda.combine.compose import CONTRACT_GROUPS, component_slots
from grasppanda.combine.grid import FAMILIES, METHODS, family_slots
from grasppanda.config import Config
from grasppanda.registry import DETECTORS

ALL = {**METHODS, **FAMILIES}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    httpd = explore.server(0, tmp_path_factory.mktemp("explore"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def ask(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + "/api/compose",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.mark.parametrize("method", ALL)
def test_the_live_explorer_composes_every_method_as_shipped(live, method):
    status, report = ask(live, {"method": method, "sets": []})
    assert status == 200 and report["accepted"] is True, report.get("refusal")
    assert report["config"] and report["config_hash"] == Config.load(ALL[method]).digest()
    assert report["refusal"] == [] and report["problems"] == []
    detector = DETECTORS.get(Config.load(ALL[method])["model"]["type"])
    stages = {stage["slot"] for stage in report["stages"]}
    if method in METHODS:  # the six slots of the skeleton and every contract group
        assert stages == set(component_slots(detector))
        assert [group["slots"] for group in report["contracts"]] == [
            list(slots) for slots in CONTRACT_GROUPS
        ]
        assert not any(group["problems"] for group in report["contracts"])
    else:  # no contract of the skeleton; what it lists are slots of its own
        assert report["contracts"] == [] and stages <= set(family_slots(method))


@pytest.mark.parametrize("method", FAMILIES)
def test_the_live_explorer_swaps_the_slot_of_a_family_of_its_own(live, method):
    """One request into HGGD's image encoder: an option whose record runs."""
    slot = next(iter(family_slots(method)))
    records = option_lists.measured().by_id
    option = next(
        option
        for option in option_lists.options(method, slot)
        if option.cell and records.get(option.cell, {}).get("verdict") == "runs"
    )
    status, report = ask(live, {"method": method, "sets": list(option.sets)})
    assert status == 200 and report["accepted"] is True, report.get("refusal")
    assert f"type: {option.component}" in report["config"]
    assert report["config_hash"][:16] == records[option.cell]["config_hash"]
