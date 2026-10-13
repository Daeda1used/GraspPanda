"""Repository-level invariants that keep the public face of the project consistent."""

from __future__ import annotations

import re

from .conftest import REPO


def headings(name: str) -> list[int]:
    """Heading levels of a Markdown file, in order, ignoring fenced code blocks."""
    levels, fenced = [], False
    for line in (REPO / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (match := re.match(r"(#{1,6}) ", line)):
            levels.append(len(match.group(1)))
    return levels


def test_readmes_have_the_same_structure():
    english, chinese = "README.md", "README_zh-CN.md"
    assert headings(english) == headings(chinese)
    count = {
        name: (REPO / name).read_text(encoding="utf-8").count("\n|") for name in (english, chinese)
    }
    assert count[english] == count[chinese], "tables differ in size between the two READMEs"


def test_brand_assets_referenced_by_the_readme_exist():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    for asset in re.findall(r'(?:src|srcset)="(resources/[^"]+)"', text):
        assert (REPO / asset).is_file(), asset


def test_shipped_configs_are_valid():
    from grasppanda.config import Config, configs_root

    for path in sorted(configs_root().rglob("*.yaml")):
        if path.name != "metafile.yaml":  # model-zoo metadata, validated by grasppanda.hub
            Config.load(path)
