"""Level 2 under every method: which loss terms a method has, and a summary that covers all six.

Level 2 asks whether an option learns like the method's own component, term by term. Which
stage owns which term (``tools/learn_check.py``, :func:`term_owners`) decides what is
compared; it must name exactly the terms each method's reference run recorded, for all
five seed-based methods (HGGD has no stage of the skeleton: every term of it is compared).
The committed summary must cover every method of the grid.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from grasppanda.combine import reach
from grasppanda.combine.grid import FAMILIES, METHODS
from grasppanda.config import Config

from .conftest import REPO

ALL = {**METHODS, **FAMILIES}
MEASURED = REPO / "docs" / "advanced_guides" / "measured_combinations.json"


def tool(name: str):
    """A module of ``tools/``, loaded by path (the directory is not a package on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def references() -> dict[str, dict]:
    """Method → the level-1 record of the method as shipped."""
    records = {record["id"]: record for record in json.loads(MEASURED.read_text())["cells"]}
    group = {method: "reference" if method in METHODS else "family" for method in ALL}
    return {method: records[f"{group[method]}/{method}"] for method in ALL}


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(tool("learn_check").SUMMARY.read_text(encoding="utf-8"))


def model_of(method: str) -> dict:
    return json.loads(json.dumps(Config.load(ALL[method]).resolved()["model"]))


@pytest.mark.parametrize("method", ALL)
def test_the_loss_terms_a_stage_owns_are_the_terms_the_reference_run_recorded(
    method, references, summary
):
    learn = tool("learn_check")
    recorded = set(references[method]["losses"]) - {"loss"}
    assert recorded, method
    owners = learn.term_owners(model_of(method))
    if method in METHODS:
        assert set(owners) == recorded, (method, sorted(set(owners) ^ recorded))
        assert {owner[0] for owner in owners.values()} <= set(learn.STAGES_WITH_TERMS)
    else:  # no stage of the skeleton: nothing is redefined by a swap, every term compared
        assert owners == {}
    # The level-2 reference runs report the same terms as the level-1 record.
    group = "reference" if method in METHODS else "family"
    for line in summary["references"]:
        if line["id"] in (f"{group}/{method}", f"{group}/{method}@seed1"):
            assert set(line["terms"]) - {"loss"} == recorded, line["id"]


def test_the_level_2_summary_covers_every_method(summary):
    about = summary["about"]
    assert set(about["calibrated"]) == set(about["noise"]) == set(ALL)
    wanted = set()
    for method in ALL:
        group = "reference" if method in METHODS else "family"
        wanted |= {f"{group}/{method}", f"{group}/{method}@seed1"}
    assert {line["id"] for line in summary["references"]} == wanted
    assert {line["method"] for line in summary["references"]} == set(ALL)
    assert {line["method"] for line in summary["cells"]} == set(ALL)
    for method in ALL:  # cells of a method are compared with that method's reference
        assert all(
            line["id"].split("/")[1] == method
            for line in summary["cells"]
            if line["method"] == method
        ), method
    # The pages and the explorer read the same file.
    assert reach.level2_file().resolve() == tool("learn_check").SUMMARY.resolve()
