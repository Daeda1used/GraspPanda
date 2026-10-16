"""The pages that show model-zoo numbers say what the metafiles say.

``docs/model_zoo.md`` and the method READMEs are written by hand. These tests hold them
to ``configs/<method>/metafile.yaml``, whose measured rows ``test_zoo_records.py`` holds
to the committed result records: so a number is on a page only when a record backs it,
and a measured row that is missing from a page is named by the failure, as the line to
add.
"""

from __future__ import annotations

import re

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
    """The cells of the first Markdown table below ``heading``, without the two header lines."""
    lines = page[page.index(heading) :].splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[2:]


CAMERAS = {"realsense": "RealSense", "kinect": "Kinect", "realsense-d435": "RealSense D435"}


def expected_rows(protocol: str, provenance: str = "author-ckpt") -> list[list[str]]:
    """The lines of a zoo table of one measured provenance, from the metafiles alone.

    Per model with a measured row: one line per collision-filter state (off, on) with
    the measured numbers, the reported numbers of that state beside them and measured
    minus reported; then one line per reported row whose source does not state the
    filter, which is paired with no measured number.
    """
    lines = []
    for meta in METAFILES:
        for model in meta.models:
            rows = {provenance: {}, "reported": {}}
            for result in model.results:
                if result.provenance in rows:
                    rows[result.provenance].setdefault(result.collision_filter, []).append(result)
            if model.protocol != protocol or not rows[provenance]:
                continue
            head = [meta.method, CAMERAS[model.camera]]
            for state, word in ((False, "off"), (True, "on")):
                measured = rows[provenance].get(state, [])
                reported = rows["reported"].get(state, [])
                assert len(measured) <= 1 and len(reported) <= 1, model.name
                if not measured and not reported:
                    continue
                ours = numbers_text(measured[0]) if measured else "—"
                theirs = numbers_text(reported[0]) if reported else "—"
                delta = difference(ours, theirs) if measured and reported else "—"
                lines.append([*head, word, ours, theirs, delta])
            for result in rows["reported"].get(None, []):
                lines.append([*head, "not stated", "—", numbers_text(result), "—"])
    return lines


@pytest.mark.parametrize(
    ("heading", "protocol", "provenance"),
    [
        ("## Trained and evaluated here", "graspnet1b-sv-v1", "reproduced"),
        ("## Authors' checkpoints scored by this toolbox", "graspnet1b-sv-v1", "author-ckpt"),
        ("## GraspClutter6D transfers", "graspclutter6d-sv-v1", "author-ckpt"),
        ("### The model trained here, on GraspClutter6D", "graspclutter6d-sv-v1", "reproduced"),
    ],
)
def test_the_zoo_page_shows_exactly_the_measured_rows_of_the_metafiles(
    heading, protocol, provenance
):
    page = (REPO / "docs" / "model_zoo.md").read_text(encoding="utf-8")
    shown = table_under(page, heading)
    expected = expected_rows(protocol, provenance)
    missing = [line for line in expected if line not in shown]
    extra = [line for line in shown if line not in expected]
    assert not missing and not extra, (
        "docs/model_zoo.md and the metafiles disagree.\nLines the metafiles ask for:\n"
        + "\n".join("| " + " | ".join(line) + " |" for line in missing)
        + "\nLines of the page without a metafile row:\n"
        + "\n".join("| " + " | ".join(line) + " |" for line in extra)
    )
    assert len(shown) == len(expected)


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
            rows = {r.collision_filter: r.ap["test"] for r in model.results}
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


#: How the zoo tables of the READMEs and of the "Targets" write the collision filter.
FILTER_WORDS = {"off": False, "on": True, "关": False, "开": True}


def reproduced_by_target() -> dict[tuple[str, bool], str]:
    """``(method, collision filter)`` -> numbers of every GraspNet-1Billion reproduced row."""
    return {
        (meta.method, result.collision_filter): numbers_text(result)
        for meta, model, result in MEASURED
        if result.provenance == "reproduced" and model.protocol == "graspnet1b-sv-v1"
    }


@pytest.mark.parametrize(
    ("page", "heading"),
    [
        ("docs/model_zoo.md", "## Targets"),
        ("README.md", "## 🏆 Model zoo"),
        ("README_zh-CN.md", "## 🏆 Model zoo"),
    ],
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
    section = page[page.index("## Trained and evaluated here") : page.index("## Authors'")]
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
            opening = " ".join(page[: page.index("## Admission gate")].split())
            assert "weights are not published yet" in opening, meta.method
