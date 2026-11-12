"""The pages that show model-zoo numbers say what the metafiles say.

``docs/model_zoo.md`` and the method READMEs are written by hand. These tests hold them
to ``configs/<method>/metafile.yaml``, whose measured rows ``test_zoo_records.py`` holds
to the committed result records: so a number is on a page only when a record backs it,
and a measured row that is missing from a page is named by the failure, as the line to
add.
"""

from __future__ import annotations

import importlib.util
import re
import sys

import pytest

from .conftest import REPO
from .test_zoo_records import G1B_SCENES, MEASURED, METAFILES


def numbers_text(result) -> str:
    """The numbers of a row as the pages print them: ``a / b / c``.

    GraspNet-1Billion: AP on seen / similar / novel. GraspClutter6D: AP / AP0.8 / AP0.4.
    """
    numbers = result.numbers()
    if set(result.ap) == set(G1B_SCENES):
        values = [numbers["AP"][split] for split in G1B_SCENES]
    else:
        values = [numbers[metric]["test"] for metric in ("AP", "AP0.8", "AP0.4")]
    return " / ".join(f"{value:.2f}" for value in values)


def difference(measured: str, reported: str) -> str:
    """Measured minus reported, number by number, of two ``a / b / c`` texts."""
    pairs = zip(measured.split(" / "), reported.split(" / "))
    return " / ".join(f"{round(float(m) - float(r), 2):+.2f}".replace("-", "−") for m, r in pairs)


def table_under(page: str, heading: str) -> list[list[str]]:
    """The cells of the first Markdown table below ``heading``, without the two header lines
    (a table may be indented, inside a fold)."""
    lines = [line.strip() for line in page[page.index(heading) :].splitlines()]
    start = next(index for index, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[2:]


CAMERAS = {"realsense": "RealSense", "kinect": "Kinect", "realsense-d435": "RealSense D435"}


#: The measured provenances, in the order of their columns on the zoo page.
MEASURED_COLUMNS = ("author-ckpt", "reproduced")


def expected_rows(protocol: str) -> list[list[str]]:
    """The lines of the zoo table of one benchmark, from the metafiles alone.

    Per model with a measured row: one line per collision-filter state (off, on) with the
    reported numbers of that state, what the released weights score and what the model
    trained here scores; then one line per reported row whose source does not state the
    filter, which stands beside no measured number.
    """
    lines = []
    for meta in METAFILES:
        for model in meta.models:
            rows: dict[tuple[str, bool | None], list] = {}
            for result in model.results:
                rows.setdefault((result.provenance, result.collision_filter), []).append(result)
            measured = any(provenance in MEASURED_COLUMNS for provenance, _ in rows)
            if model.protocol != protocol or not measured:
                continue
            head = [meta.method, CAMERAS[model.camera]]
            for state, word in ((False, "off"), (True, "on")):
                cells = []
                for provenance in ("reported", *MEASURED_COLUMNS):
                    found = rows.get((provenance, state), [])
                    assert len(found) <= 1, model.name
                    cells.append(numbers_text(found[0]) if found else "—")
                if set(cells) != {"—"}:
                    lines.append([*head, word, *cells])
            for result in rows.get(("reported", None), []):
                lines.append([*head, "not stated", numbers_text(result), "—", "—"])
    return lines


def expected_differences(protocol: str) -> list[list[str]]:
    """Measured minus reported, for every line whose reported number states the filter."""
    lines = []
    for *head, reported, author, trained in expected_rows(protocol):
        if reported == "—" or head[-1] == "not stated":
            continue
        cells = [difference(m, reported) if m != "—" else "—" for m in (author, trained)]
        lines.append([*head, *cells])
    return lines


def disagreement(shown: list[list[str]], expected: list[list[str]]) -> str:
    """The lines to add to a page and the lines to take from it, as Markdown."""
    missing = [line for line in expected if line not in shown]
    extra = [line for line in shown if line not in expected]
    if not missing and not extra and len(shown) == len(expected):
        return ""
    return (
        "docs/model_zoo.md and the metafiles disagree.\nLines the metafiles ask for:\n"
        + "\n".join("| " + " | ".join(line) + " |" for line in missing)
        + "\nLines of the page without a metafile row:\n"
        + "\n".join("| " + " | ".join(line) + " |" for line in extra)
    )


#: Heading of each benchmark's table on the zoo page → its protocol.
ZOO_TABLES = {
    "## GraspNet-1Billion": "graspnet1b-sv-v1",
    "## GraspClutter6D transfers": "graspclutter6d-sv-v1",
}
#: The fold of the zoo page with measured minus reported.
DIFFERENCES = '??? note "Measured minus reported"'


@pytest.mark.parametrize(("heading", "protocol"), ZOO_TABLES.items())
def test_the_zoo_page_shows_exactly_the_rows_of_the_metafiles(heading, protocol):
    """One line per model and filter state: reported, released weights, trained here.

    A benchmark scored with one camera names it above its table instead of on every line.
    """
    page = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    expected = expected_rows(protocol)
    cameras = {line[1] for line in expected}
    if len(cameras) == 1:
        start = page.index(heading)
        assert cameras.pop() in " ".join(page[start : page.index("\n|", start)].split()), heading
        expected = [[line[0], *line[2:]] for line in expected]
    problem = disagreement(table_under(page, heading), expected)
    assert not problem, problem


def test_the_differences_on_the_zoo_page_are_measured_minus_reported():
    """The fold below the GraspNet-1Billion table, computed from the same rows. No
    GraspClutter6D number is reported with its collision filter, so nothing is paired."""
    page = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    assert page.count(DIFFERENCES) == 1
    assert page.index("## GraspNet-1Billion") < page.index(DIFFERENCES)
    assert page.index(DIFFERENCES) < page.index("## GraspClutter6D transfers")
    expected = expected_differences("graspnet1b-sv-v1")
    problem = disagreement(table_under(page, DIFFERENCES), expected)
    assert not problem, problem
    assert expected_differences("graspclutter6d-sv-v1") == []


def test_every_number_on_the_zoo_page_is_stated_by_a_metafile():
    page = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    stated = {
        numbers_text(result)
        for meta in METAFILES
        for model in meta.models
        for result in model.results
    }
    shown = set(re.findall(r"\d+\.\d\d / \d+\.\d\d / \d+\.\d\d", page))
    assert shown <= stated, f"numbers on the page that no metafile states: {sorted(shown - stated)}"


def test_each_method_readme_shows_the_rows_of_its_metafile_and_links_the_records():
    for meta in METAFILES:
        readme = (meta.path.parent / "README.md").read_text(encoding="utf-8")
        for model in meta.models:
            if all(result.provenance == "reported" for result in model.results):
                continue
            for result in model.results:
                numbers = numbers_text(result)  # as "a / b / c" or as three table cells
                shown = numbers in readme or numbers.replace(" / ", " | ") in readme
                assert shown, f"{meta.method}: README.md lacks {numbers} ({model.name})"
                if result.record:
                    assert f"({result.record})" in readme, f"{meta.method}: {result.record}"


def test_the_transfer_differences_stated_in_prose_are_those_of_the_metafiles():
    """Both pages say how far each measured variant is from the reported GraspClutter6D AP."""
    zoo = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    guide = (REPO / "docs" / "user_guides" / "cross_dataset.md").read_text(encoding="utf-8")
    sections = (
        zoo[zoo.index("## GraspClutter6D transfers") :],
        guide[guide.index("## The whole test split") : guide.index("## Checking a run")],
    )
    for meta in METAFILES:
        for model in meta.models:
            if model.protocol != "graspclutter6d-sv-v1":
                continue
            rows = {
                r.collision_filter: r.ap["test"]
                for r in model.results
                if r.provenance in ("reported", "author-ckpt")
            }
            if not {None, False, True} <= set(rows):
                continue
            for state in (False, True):
                distance = f"{abs(round(rows[state] - rows[None], 2)):.2f}"
                for section in sections:
                    assert distance in section, (meta.method, state, distance)


def test_the_whole_split_table_of_the_transfer_guide_is_that_of_the_metafiles():
    guide = (REPO / "docs" / "user_guides" / "cross_dataset.md").read_text(encoding="utf-8")
    section = guide[guide.index("## The whole test split") : guide.index("## Checking a run")]
    transfers = {
        numbers_text(result)
        for meta, model, result in MEASURED
        if model.protocol == "graspclutter6d-sv-v1"
    }
    reported = {
        numbers_text(result)
        for meta in METAFILES
        for model in meta.models
        for result in model.results
        if model.protocol == "graspclutter6d-sv-v1" and result.provenance == "reported"
    }
    shown = set(re.findall(r"\d+\.\d\d / \d+\.\d\d / \d+\.\d\d", section))
    assert transfers <= shown, f"measured transfers missing: {sorted(transfers - shown)}"
    assert shown <= transfers | reported, (
        f"not in a metafile: {sorted(shown - transfers - reported)}"
    )


#: How the zoo tables of the READMEs write the collision filter.
FILTER_WORDS = {"off": False, "on": True, "关": False, "开": True}


def reproduced_by_target() -> dict[tuple[str, bool], str]:
    """``(method, collision filter)`` -> numbers of every GraspNet-1Billion reproduced row
    that has a target: a reported row of its model with the same collision filter."""
    return {
        (meta.method, result.collision_filter): numbers_text(result)
        for meta, model, result in MEASURED
        if result.provenance == "reproduced"
        and model.protocol == "graspnet1b-sv-v1"
        and any(
            r.provenance == "reported" and r.collision_filter is result.collision_filter
            for r in model.results
        )
    }


@pytest.mark.parametrize(
    ("page", "heading"),
    [("README.md", "## Model zoo"), ("README_zh-CN.md", "## 模型库")],
)
def test_the_reproduced_column_of_the_targets_shows_exactly_the_reproduced_rows(page, heading):
    """A target line shows the reproduced numbers of its method and filter, and only those."""
    expected = reproduced_by_target()
    shown = {}
    for cells in table_under((REPO / page).read_text(encoding="utf-8"), heading):
        method = cells[0].split(" (")[0]
        state = FILTER_WORDS.get(cells[-3])  # the column before "Reported"
        if cells[-1] != "—":
            shown[(method, state)] = cells[-1]
    assert shown == expected, f"{page}: the last column of the targets"


def test_the_zoo_page_says_how_each_reproduction_stands_against_the_admission_rule():
    page = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    assert "Gate: `|reproduced − reported| ≤ 1.0 AP` on each of seen / similar / novel" in page
    assert "No method has been trained and evaluated here" not in page
    assert "provenance `author-ckpt`; they are not reproductions" in page
    start = page.index("## GraspNet-1Billion")
    section = page[start : page.index("\n## ", start + 1)]
    for meta, model, result in MEASURED:
        if result.provenance != "reproduced" or model.protocol != "graspnet1b-sv-v1":
            continue
        reported = [
            r
            for r in model.results
            if r.provenance == "reported" and r.collision_filter is result.collision_filter
        ]
        outside = reported and any(
            abs(result.ap[split] - reported[0].ap[split]) > 1.0 for split in result.ap
        )
        if outside:  # the page names the gate and says what is known about the gap
            assert "**Against the admission gate.**" in section, meta.method
        if not model.weights.published:
            assert "**The weights are not published yet.**" in section, meta.method
            opening = " ".join(page[: page.index("\n## ")].split())
            assert "weights are not published yet" in opening, meta.method


def test_the_zoo_page_opens_with_each_method_under_the_provenance_of_its_numbers():
    """The opening names, per measured provenance, exactly the methods with rows of it."""
    page = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    opening = page[: page.index("\n## ")]
    methods = {meta.method for meta in METAFILES}
    wanted: dict[str, set[str]] = {}
    for meta, _, result in MEASURED:
        wanted.setdefault(result.provenance, set()).add(meta.method)
    for provenance, named in wanted.items():
        item = next(part for part in opening.split("\n- ") if f"(`{provenance}`)" in part)
        assert {method for method in methods if method in item} == named, provenance


#: The encoders' page: the status of an encoder in its table → the words of the summary
#: above it (the same mapping as the evidence card, tools/make_combination_docs.py).
ENCODER_SUMMARY = {
    "`weights-verified`": "Weights verified",
    "`tested` + parity": "Compared on synthetic weights",
    "`tested`": "Unit-tested only",
}


def _tool(name: str):
    """A module of ``tools/``, loaded by path (the directory is not a package on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_encoders_page_says_how_far_each_encoder_was_checked_from_its_own_table():
    """The summary above the table names, per depth of evidence, exactly the encoders of
    that status and their number: released weights loaded (weights-verified), outputs
    compared on synthetic weights (+ parity), unit tests only (tested). The evidence card
    counts the same table."""
    page = (REPO / "docs" / "model_zoo_backbones.md").read_text(encoding="utf-8")
    rows = table_under(page, "## The encoders and how each was checked")
    by_status: dict[str, list[str]] = {}
    for row in rows:
        by_status.setdefault(row[3], []).append(row[0].strip("`"))
    assert set(by_status) == set(ENCODER_SUMMARY), set(by_status)
    opening = page[: page.index("\n## ")]
    items = [" ".join(item.split()) for item in opening.split("\n- ")[1:]]
    for status, words in ENCODER_SUMMARY.items():
        names = by_status[status]
        item = next(item for item in items if item.startswith(f"**{words}** ({len(names)}):"))
        listed = item.split(":", 1)[1].split(". ")[0]
        assert re.findall(r"`([^`]+)`", listed) == names, status
    # Strict loading of released weights is claimed for the weights-verified rows only.
    assert "load strictly" not in opening.split("\n- ")[0]
    generator = _tool("make_combination_docs")
    assert generator.encoder_checks() == {s: by_status[s] for s in generator.ENCODER_CHECKS}
    assert {s: words for s, (words, _) in generator.ENCODER_CHECKS.items()} == ENCODER_SUMMARY
