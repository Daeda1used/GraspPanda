"""Every page holds the length budget of its kind (``tools/check_page_budgets.py``).

A reader sees a page with its folds closed; the visible budget is what they read before
opening anything, the total budget keeps folds from turning into a store for what belongs
in the explorer or a data file. The test runs the checker's own logic on every page of the
documentation, the READMEs and the config READMEs, and fails with the list of pages over
budget: ``python -m tools.check_page_budgets`` prints the whole table.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from .conftest import REPO


def budgets():
    name = "check_page_budgets"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_page_has_a_kind_and_holds_its_budget():
    tool = budgets()
    pages = tool.pages()
    over = []
    for path, kind in pages.items():
        relative = path.relative_to(REPO).as_posix()
        if not kind:
            over.append(f"{relative}: no kind (add it to KINDS in tools/check_page_budgets.py)")
            continue
        found = tool.over(kind, tool.count(path.read_text(encoding="utf-8")))
        if found:
            over.append(f"{relative} ({kind}): {'; '.join(found)}")
    assert set(tool.KINDS.values()) <= set(tool.BUDGETS)
    stale = [page for page in tool.KINDS if not (tool.DOCS / page).is_file()]
    assert not stale, f"KINDS names page(s) that do not exist: {stale}"
    if over:
        pytest.fail(
            f"{len(over)} of {len(pages)} page(s) over budget; shorten them, fold the detail "
            "or move it to the explorer (python -m tools.check_page_budgets prints the "
            "table):\n  " + "\n  ".join(over),
            pytrace=False,
        )


def test_what_a_reader_sees_is_counted_apart_from_what_is_folded():
    """The rules of the count: folds (`???`, `<details>`) and later content tabs count in
    the totals only; an open fold (`???+`, `<details open>`) and the first tab are seen;
    table headers are no rows; code is no words, and only a visible block is measured."""
    tool = budgets()
    page = "\n".join(
        [
            "# Title words",  # 2
            "",
            "| a | b |",
            "|---|---|",
            "| one | two |",  # a visible row, 2 words
            "",
            '??? note "Folded"',
            "    hidden words here",  # 3 hidden
            "",
            '???+ note "Open"',  # its title is seen: 2
            "    seen words",  # 2
            "",
            '=== "First"',
            "    first tab",  # 2
            "",
            '=== "Second"',
            "    second tab words",  # 3 hidden
            "",
            "<details><summary>More</summary>",
            "",
            "| x |",
            "|---|",
            "| folded row |",  # a hidden row, 2 hidden words
            "",
            "</details>",
            "",
            "```bash",
            "one line of code",
            "another",
            "```",
        ]
    )
    counted = tool.count(page)
    assert (counted.visible_words, counted.visible_rows) == (10, 1)
    # `More` in the summary line is inside the closed <details>, so it is hidden too.
    assert (counted.total_words, counted.total_rows) == (10 + 3 + 3 + 1 + 2, 2)
    assert counted.longest_code == 2
    assert tool.over("home", counted) == ["visible rows 1 > 0"]
