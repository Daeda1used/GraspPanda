"""Hold every page of the documentation to the length of its kind.

A reader sees a page with its folds closed: what is inside a collapsed block (``??? note``,
``<details>`` without ``open``) or in a content tab other than the first is one click away.
So each page has two budgets, both counted on the Markdown source:

* **visible**: words and table rows a reader sees without opening anything;
* **total**: everything, folds included, so that folds do not become a store for pages that
  belong in the explorer or a data file.

Words are counted outside fenced code, HTML comments and tags; a CJK character counts as
half a word. Table rows are data rows (the header and its separator line are not counted).
Snippets (``--8<-- "file"``) are expanded. Visible code blocks of guides, method and slot
pages are held to :data:`CODE_LINES` lines.

Usage::

    python -m tools.check_page_budgets            # the table of every page
    python -m tools.check_page_budgets --check    # exit 1 when a page is over a budget
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: Kind → (visible words, visible table rows, total words, total table rows); ``None``
#: for a limit that does not apply.
BUDGETS: dict[str, tuple[int | None, int | None, int | None, int | None]] = {
    "home": (300, 0, 600, 10),
    "section": (700, 15, 2000, 60),
    "get-started": (700, 15, 2500, 60),
    "method": (900, 20, 9000, 300),
    "slot": (1000, 40, 6000, 120),
    "guide": (1200, 15, 4000, 60),
    "evidence": (1200, 40, 12000, 400),
    "reference": (1500, 40, 12000, 400),
    "long-reference": (1500, 40, None, None),
    "readme": (800, 45, 3000, 80),
    "config-readme": (600, 20, 2500, 60),
    "exempt": (None, None, None, None),
}
#: The longest visible code block of a guide, a method page or a slot page, in lines.
CODE_LINES = 12
CODE_KINDS = ("home", "get-started", "guide", "method", "slot")

#: Every page and its kind. A Markdown file under ``docs/`` that is not here fails the check.
KINDS: dict[str, str] = {
    "index.md": "home",
    "get_started/index.md": "section",
    "get_started/installation.md": "get-started",
    "get_started/quickstart.md": "get-started",
    "get_started/run_a_method.md": "get-started",
    "get_started/swap.md": "get-started",
    "explorer/index.md": "exempt",  # the app; its data is included as JSON
    "methods/index.md": "section",
    "methods/graspnet_baseline.md": "method",
    "methods/gsnet.md": "method",
    "methods/scale_balanced_grasp.md": "method",
    "methods/economicgrasp.md": "method",
    "methods/finegrasp.md": "method",
    "methods/hggd.md": "method",
    "slots/index.md": "section",
    "slots/scene_encoders.md": "slot",
    "slots/scene_encoders_point.md": "slot",
    "slots/scene_encoders_sparse.md": "slot",
    "slots/scene_encoders_transformer.md": "slot",
    "slots/image_encoders.md": "slot",
    "slots/proposal.md": "slot",
    "slots/view.md": "slot",
    "slots/grouper.md": "slot",
    "slots/head_loss.md": "slot",
    "slots/losses.md": "slot",
    "slots/post.md": "slot",
    "slots/recipe.md": "slot",
    "user_guides/index.md": "section",
    "user_guides/config.md": "guide",
    "user_guides/datasets.md": "guide",
    "user_guides/train_test.md": "guide",
    "user_guides/cross_dataset.md": "guide",
    "user_guides/inference.md": "guide",
    "user_guides/analysis.md": "guide",
    "user_guides/combinations.md": "guide",
    "advanced_guides/add_component.md": "guide",
    "evidence/index.md": "section",
    "model_zoo.md": "evidence",
    "advanced_guides/measured_combinations.md": "evidence",
    "advanced_guides/learns_like_reference.md": "evidence",
    "advanced_guides/swapping.md": "evidence",
    "model_zoo_backbones.md": "evidence",
    "components.md": "long-reference",
    "advanced_guides/data_flow.md": "reference",
    "advanced_guides/data_loading.md": "reference",
    "advanced_guides/operators.md": "reference",
    "api/config.md": "exempt",
    "api/registry.md": "exempt",
    "api/structures.md": "exempt",
    "api/ops.md": "exempt",
    "design/ARCHITECTURE.md": "long-reference",
    "faq.md": "reference",
    "changelog.md": "long-reference",
}
#: Files outside ``docs/`` that are pages too (GitHub shows them), and their kind.
OUTSIDE = {"README.md": "readme", "README_zh-CN.md": "readme"}

SNIPPET = re.compile(r'^([ \t]*)--8<--[ \t]+"([^"]+)"[ \t]*$', re.M)
CJK = re.compile(r"[　-鿿＀-￯]")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’._/-]*")


@dataclass
class Count:
    visible_words: int = 0
    visible_rows: int = 0
    total_words: int = 0
    total_rows: int = 0
    longest_code: int = 0


def expand(text: str, depth: int = 0) -> str:
    """The text with every snippet include replaced by the file it names."""
    if depth > 3:
        return text

    def one(match: re.Match[str]) -> str:
        path = ROOT / match.group(2)
        if not path.is_file() or path.suffix == ".json":
            return ""
        inner = expand(path.read_text(encoding="utf-8"), depth + 1)
        return "\n".join(match.group(1) + line for line in inner.splitlines())

    return SNIPPET.sub(one, text)


def words(line: str) -> float:
    plain = re.sub(r"<[^>]+>", " ", line)
    plain = re.sub(r"\]\([^)]*\)", "]", plain)  # link targets are not read
    return len(WORD.findall(plain)) + len(CJK.findall(plain)) / 2


def count(text: str) -> Count:
    """Visible and total words and rows of a Markdown page."""
    text = re.sub(r"<!--.*?-->", "", expand(text), flags=re.S)
    result = Count()
    words_visible = words_total = 0.0
    fence = ""  # the fence of the open code block
    code_lines, code_visible = 0, True
    folded: int | None = None  # the indent of the marker of the fold we are in
    tabs: dict[int, int] = {}  # indent of a tab set → tabs seen in it
    details = 0  # depth of closed <details> blocks
    table_line = 0  # line number within the current table
    for raw in text.splitlines():
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip(" "))
        if fence:
            if stripped.startswith(fence):
                fence = ""
                if code_visible:
                    result.longest_code = max(result.longest_code, code_lines)
            else:
                code_lines += 1
            continue
        if folded is not None and stripped and indent <= folded:
            folded = None
        if stripped and not stripped.startswith("==="):
            for level in [level for level in tabs if indent <= level]:
                del tabs[level]
        if stripped.startswith(("```", "~~~")):
            fence, code_lines = stripped[:3], 0
            code_visible = folded is None and details == 0
            continue
        if re.match(r"\?\?\?(?!\+)", stripped):
            folded = indent if folded is None else folded
            continue
        if stripped.startswith("==="):
            if tabs.get(indent, 0) and folded is None:
                folded = indent
            tabs[indent] = tabs.get(indent, 0) + 1
            continue
        if re.search(r"<details(?![^>]*\bopen\b)[^>]*>", stripped):
            details += 1
        if "</details>" in stripped:
            details = max(0, details - 1)
            continue
        hidden = folded is not None or details > 0
        if stripped.startswith("|"):
            table_line += 1
            if table_line > 2:  # the header and its separator are not rows
                result.total_rows += 1
                result.visible_rows += not hidden
            if table_line <= 2:
                continue
        else:
            table_line = 0
        n = words(stripped)
        words_total += n
        words_visible += 0 if hidden else n
    result.visible_words, result.total_words = round(words_visible), round(words_total)
    return result


def pages() -> dict[Path, str]:
    """Every page and its kind; a page without a kind is returned with kind ``""``."""
    found: dict[Path, str] = {}
    for path in sorted(DOCS.rglob("*.md")):
        relative = path.relative_to(DOCS).as_posix()
        if relative.startswith("includes/"):
            continue
        found[path] = KINDS.get(relative, "")
    for name, kind in OUTSIDE.items():
        found[ROOT / name] = kind
    for readme in sorted((ROOT / "configs").glob("*/README.md")):
        found[readme] = "config-readme"
    return found


def over(kind: str, counted: Count) -> list[str]:
    """What exceeds the budget of ``kind``."""
    limits = BUDGETS[kind]
    values = (
        counted.visible_words,
        counted.visible_rows,
        counted.total_words,
        counted.total_rows,
    )
    names = ("visible words", "visible rows", "total words", "total rows")
    found = [
        f"{name} {value} > {limit}"
        for name, value, limit in zip(names, values, limits, strict=True)
        if limit is not None and value > limit
    ]
    if kind in CODE_KINDS and counted.longest_code > CODE_LINES:
        found.append(f"a visible code block of {counted.longest_code} lines > {CODE_LINES}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when a page is over")
    parser.add_argument(
        "--only", nargs="*", default=(), help="pages to report: a path, or a path under docs/"
    )
    args = parser.parse_args(argv)
    problems = 0
    rows = []
    for path, kind in pages().items():
        relative = path.relative_to(ROOT).as_posix()
        if args.only and not any(
            relative == part or relative.endswith("/" + part) or relative == f"docs/{part}"
            for part in args.only
        ):
            continue
        if not kind:
            rows.append((relative, "?", None, ["no kind: add the page to KINDS"]))
            problems += 1
            continue
        counted = count(path.read_text(encoding="utf-8"))
        found = over(kind, counted)
        problems += bool(found)
        rows.append((relative, kind, counted, found))
    for relative, kind, counted, found in rows:
        if counted is None:
            print(f"✗ {relative}: {'; '.join(found)}")
            continue
        mark = "✗" if found else "✓"
        numbers = (
            f"{counted.visible_words:>6} w {counted.visible_rows:>4} r visible · "
            f"{counted.total_words:>6} w {counted.total_rows:>5} r total"
        )
        print(
            f"{mark} {kind:<14} {numbers}  {relative}"
            + (f"  ← {'; '.join(found)}" if found else "")
        )
    print(f"\n{problems} page(s) over budget")
    return 1 if args.check and problems else 0


if __name__ == "__main__":
    sys.exit(main())
