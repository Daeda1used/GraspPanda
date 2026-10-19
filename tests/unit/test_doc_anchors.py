"""Links from outside a page land where they point: the frozen anchors of the documentation.

Links between pages of the site are checked by ``mkdocs build --strict``. These are the
anchors that are reached from elsewhere and that no docs build sees: the command line and
the inferencer print them, the explorer links them, an installed wheel quotes them, and
the READMEs on GitHub link headings of their own and of other files. Pages may be rebuilt
freely as long as these anchors stay (a heading with that slug, or an explicit ``{#id}``):

* ``advanced_guides/measured_combinations.md``: ``#measured-classes``, ``#sets-of-stages``,
  ``#several-slots-at-once``;
* every method page: ``#run-in-five-minutes``, ``#the-slots`` and one anchor per slot the
  method has options for (``#backbone`` ... ``#image-encoder``);
* ``user_guides/inference.md``: ``#your-own-camera``, ``#from-python``, ``#run-in-the-browser``;
* ``model_zoo.md``: ``#graspclutter6d-transfers``, ``#recording-a-result``;
* ``slots/index.md#words-used-on-these-pages`` while a page links it;
* every anchor a README, CONTRIBUTING, the changelog or a config README links, by GitHub's
  rules for headings;
* every ``docs/...md`` reference in the package (with its anchor, ``section N`` or quoted
  headings) and the anchors the explorer links.

The anchors are read from the Markdown source (snippets expanded) by the rules of the toc
extension; where Python-Markdown is installed they are also checked on the rendered page.
"""

from __future__ import annotations

import ast
import html
import importlib
import re
import unicodedata
from functools import cache
from pathlib import Path

import pytest

from grasppanda.combine import options as option_lists
from grasppanda.combine.grid import FAMILIES, METHODS

from .conftest import REPO

DOCS = REPO / "docs"
SITE = "https://daeda1used.github.io/GraspPanda/"

#: Page (relative to ``docs/``) → anchors that are linked from outside the site.
FROZEN: dict[str, tuple[str, ...]] = {
    "advanced_guides/measured_combinations.md": (
        "measured-classes",
        "sets-of-stages",
        "several-slots-at-once",
    ),
    "user_guides/inference.md": ("your-own-camera", "from-python", "run-in-the-browser"),
    "model_zoo.md": ("graspclutter6d-transfers", "recording-a-result"),
}
#: Files that GitHub renders: their links follow GitHub's heading anchors.
GITHUB_FILES = (
    "README.md",
    "README_zh-CN.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    *sorted(path.relative_to(REPO).as_posix() for path in (REPO / "configs").rglob("README.md")),
)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
ATTRIBUTES = re.compile(r"\s*\{:?\s*([^{}]*)\}\s*$")
ATTRIBUTE_LIST = re.compile(r"""^(\s*(#[\w-]+|\.[\w-]+|[\w-]+=("[^"]*"|'[^']*'|[^\s"']+)))+\s*$""")
EXPLICIT_ID = re.compile(r"\{:?[^{}]*?#([\w-]+)[^{}]*\}")
HTML_ID = re.compile(r"""<[a-zA-Z][^>]*?\b(?:id|name)\s*=\s*["']([^"']+)["']""")
SNIPPET = re.compile(r'^(\s*)--8<--\s+"([^"]+)"\s*$', re.M)
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)|\bhref=\"([^\"]+)\"")


# ------------------------------------------------------------------- reading a page


def expand(text: str, depth: int = 0) -> str:
    """The text with every snippet include replaced by the file it names."""
    if depth > 3:
        return text

    def one(match: re.Match[str]) -> str:
        path = REPO / match.group(2)
        if not path.is_file() or path.suffix not in (".md", ".html", ".txt"):
            return ""
        inner = expand(path.read_text(encoding="utf-8"), depth + 1)
        return "\n".join(match.group(1) + line for line in inner.splitlines())

    return SNIPPET.sub(one, text)


def plain(text: str) -> str:
    """The text of a heading as it is rendered: links, code, tags and entities resolved."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text).replace("`", "")
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    return html.unescape(text).strip()


def headings(text: str) -> list[tuple[str, str, str]]:
    """``(text as written, text without its attribute list, explicit id or "")`` of every
    heading outside code. GitHub shows an attribute list as text; the site reads it."""
    found, fence = [], ""
    for line in re.sub(r"<!--.*?-->", "", text, flags=re.S).splitlines():
        match = FENCE.match(line)
        if match:
            fence = "" if fence and match.group(1) == fence else (fence or match.group(1))
            continue
        heading = HEADING.match(line) if not fence else None
        if heading:
            written = heading.group(2).rstrip("#").rstrip()
            title, explicit = written, ""
            attributes = ATTRIBUTES.search(written)
            if attributes and ATTRIBUTE_LIST.match(attributes.group(1)):
                ids = re.findall(r"#([\w-]+)", attributes.group(1))
                explicit = ids[0] if ids else ""
                title = written[: attributes.start()]
            found.append((plain(written), plain(title), explicit))
    return found


def toc_slug(text: str) -> str:
    """``markdown.extensions.toc.slugify`` with the separator ``-``."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def github_slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation and symbols dropped."""
    kept = "".join(
        char for char in text.lower() if char in "-_ " or unicodedata.category(char)[0] in "LMN"
    )
    return kept.replace(" ", "-")


def unique(slugs: list[str], separator: str) -> set[str]:
    seen: dict[str, int] = {}
    found = set()
    for slug in slugs:
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}{separator}{seen[slug]}"
        else:
            seen[slug] = 0
        found.add(slug)
    return found


def outside_code(text: str) -> str:
    lines, fence = [], ""
    for line in text.splitlines():
        match = FENCE.match(line)
        if match:
            fence = "" if fence and match.group(1) == fence else (fence or match.group(1))
            continue
        if not fence:
            lines.append(line)
    return "\n".join(lines)


@cache
def site_anchors(page: str) -> frozenset[str]:
    """Every id a page of the site has: heading slugs, ``{#id}`` and HTML ids."""
    text = expand((DOCS / page).read_text(encoding="utf-8"))
    found = headings(text)
    ids = unique([explicit or toc_slug(title) for _, title, explicit in found], "_")
    body = outside_code(re.sub(r"<!--.*?-->", "", text, flags=re.S))
    ids |= set(EXPLICIT_ID.findall(body)) | set(HTML_ID.findall(body))
    return frozenset(ids)


@cache
def github_anchors(path: Path) -> frozenset[str]:
    """Every anchor GitHub gives a file: heading slugs and HTML ids or names."""
    text = path.read_text(encoding="utf-8")
    ids = unique([github_slug(written) for written, _, _ in headings(text)], "-")
    ids |= set(HTML_ID.findall(outside_code(text)))
    return frozenset(ids)


def heading_titles(page: str) -> list[str]:
    text = expand((DOCS / page).read_text(encoding="utf-8"))
    return [title for _, title, _ in headings(text)]


# ------------------------------------------------------------------- the frozen list


def method_pages() -> dict[str, tuple[str, ...]]:
    """Method page → its frozen anchors: the five-minute run, the slots, one per slot."""
    found = {}
    for method, config in {**METHODS, **FAMILIES}.items():
        slots = [slot.replace("_", "-") for slot in option_lists.method_slots(method)]
        found[f"methods/{config.split('/')[0]}.md"] = ("run-in-five-minutes", "the-slots", *slots)
    return found


@cache
def frozen() -> dict[str, tuple[str, ...]]:
    return {**FROZEN, **method_pages()}


@pytest.mark.parametrize(("page", "anchors"), list(frozen().items()), ids=list(frozen()))
def test_the_frozen_anchors_of_a_page_exist(page, anchors):
    missing = [anchor for anchor in anchors if anchor not in site_anchors(page)]
    assert not missing, (
        f"docs/{page} lost the anchor(s) {missing}, which are linked from outside the site. "
        "Keep the heading, or give the new one the old id: `## New title {#old-id}`."
    )


#: An anchor that may move, together with every link to it: page → anchor.
MOVABLE = {"slots/index.md": "words-used-on-these-pages"}


@pytest.mark.parametrize(("page", "anchor"), list(MOVABLE.items()))
def test_a_movable_anchor_exists_while_a_page_links_it(page, anchor):
    linking = []
    for path in sorted(DOCS.rglob("*.md")):
        for target in re.findall(rf"\(([^)\s#]*\.md)#{anchor}\)", path.read_text("utf-8")):
            if (path.parent / target).resolve() == (DOCS / page).resolve():
                linking.append(path.relative_to(DOCS).as_posix())
    if linking:
        assert anchor in site_anchors(page), f"{linking[:3]} link docs/{page}#{anchor}: gone"


# ------------------------------------------------------------------- READMEs on GitHub


def readme_links() -> list:
    cases = []
    for name in GITHUB_FILES:
        path = REPO / name
        if not path.is_file():
            continue
        for match in LINK.finditer(outside_code(path.read_text(encoding="utf-8"))):
            link = match.group(1) or match.group(2)
            target, _, anchor = link.partition("#")
            if anchor and "=" not in anchor:  # `#m=GSNet` is a route of the explorer
                cases.append(pytest.param(name, target, anchor, id=f"{name}->{link}"))
    return cases


@pytest.mark.parametrize(("source", "target", "anchor"), readme_links())
def test_an_anchor_linked_from_a_file_on_github_exists(source, target, anchor):
    if target.startswith(SITE):  # a page of the documentation site
        page = target[len(SITE) :].strip("/")
        page = f"{page}/index.md" if (DOCS / page / "index.md").is_file() else f"{page}.md"
        assert anchor in site_anchors(page), f"{source}: docs/{page} has no #{anchor}"
        return
    if target.startswith(("http://", "https://", "mailto:")):
        pytest.skip("an anchor of another site")
    path = (REPO / source).parent / target if target else REPO / source
    assert path.is_file(), f"{source} links {target}, which does not exist"
    if path.suffix != ".md":
        pytest.skip("an anchor of a file GitHub does not render as Markdown")
    assert anchor in github_anchors(path.resolve()), (
        f"{source} links #{anchor} of {path.relative_to(REPO)}, which has no such heading "
        f"(its anchors: {sorted(github_anchors(path.resolve()))[:12]} ...)"
    )


# ------------------------------------------------------------------- the code


REFERENCE = re.compile(
    r"docs/(?P<page>[\w./-]+?\.md)"
    r"(?:#(?P<anchor>[\w-]+)"
    r"|,\s*section\s+(?:(?P<number>\d+)(?:\s*\((?P<about>[^)]+)\))?|[\"'‘“](?P<title>[^\"'’”]+)[\"'’”])"
    r"|:\s*(?P<quoted>(?:'[^']+'(?:,\s*)?)+))?"
)


def strings(path: Path) -> list[str]:
    """Every string literal of a module, adjacent literals joined (f-strings in parts)."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            found.append(
                "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
            )
    return found


def code_references() -> list:
    cases = {}
    for path in sorted((REPO / "grasppanda").rglob("*.py")):
        if "docs/" not in path.read_text(encoding="utf-8"):
            continue
        for text in strings(path):
            for match in REFERENCE.finditer(text):
                key = match.group(0)
                cases.setdefault(
                    key, pytest.param(path.relative_to(REPO).as_posix(), match.groupdict(), id=key)
                )
    return list(cases.values())


@pytest.mark.parametrize(("source", "reference"), code_references())
def test_a_page_the_package_names_has_what_it_names(source, reference):
    page = reference["page"]
    assert (DOCS / page).is_file(), f"{source} names docs/{page}, which does not exist"
    titles = heading_titles(page)
    if reference["anchor"]:
        anchor = reference["anchor"]
        assert anchor in site_anchors(page), f"{source}: docs/{page} has no #{anchor}"
    if reference["number"]:
        numbered = [t for t in titles if re.match(rf"{reference['number']}[.)\s]", t)]
        assert numbered, f"{source} names section {reference['number']} of docs/{page}: none"
        if reference["about"]:
            assert any(reference["about"].lower() in t.lower() for t in numbered), (
                f"{source}: section {reference['number']} of docs/{page} is {numbered}, "
                f"not about {reference['about']!r}"
            )
    if reference["title"]:
        assert reference["title"].lower() in [t.lower() for t in titles], (source, page)
    for title in re.findall(r"'([^']+)'", reference["quoted"] or ""):
        assert title in titles, f"{source} names the heading {title!r} of docs/{page}: none"


def test_the_anchors_the_explorer_links_exist():
    script = (DOCS / "explorer" / "explorer.js").read_text(encoding="utf-8")
    links = re.findall(r'docHref\(\s*"([^"]+)"\s*\)\s*\+\s*"#([\w-]+)"', script)
    links += re.findall(r'"(?:\.\./)?([\w/]+)\.md#([\w-]+)"', script)
    missing = [(page, anchor) for page, anchor in links if anchor not in site_anchors(f"{page}.md")]
    assert not missing, f"the explorer links anchors that do not exist: {missing}"


# ------------------------------------------------------------------- the rendered pages


def rendered_ids(page: str) -> set[str]:
    """The ids of a page rendered with the Markdown extensions of ``mkdocs.yml``."""
    markdown = pytest.importorskip("markdown")
    pytest.importorskip("pymdownx")
    yaml = pytest.importorskip("yaml")

    class Loader(yaml.SafeLoader):
        pass

    def python_name(loader, suffix, node):
        module, _, attribute = suffix.rpartition(".")
        try:
            return getattr(importlib.import_module(module), attribute)
        except ImportError:
            pytest.skip(f"needs {module} (the docs extra)")

    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", python_name)
    config = yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=Loader)
    extensions, settings = [], {}
    for item in config["markdown_extensions"]:
        name, options = (item, {}) if isinstance(item, str) else next(iter(item.items()))
        extensions.append(name)
        settings[name] = dict(options or {})
    settings.setdefault("pymdownx.snippets", {})["base_path"] = [str(REPO)]
    text = (DOCS / page).read_text(encoding="utf-8")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    body = markdown.Markdown(extensions=extensions, extension_configs=settings).convert(text)
    return set(re.findall(r'\bid="([^"]+)"', body))


@pytest.mark.parametrize("page", list(frozen()))
def test_the_rendered_page_has_the_frozen_anchors_too(page):
    """The reader above against Python-Markdown with the extensions of the site."""
    rendered = rendered_ids(page)
    missing = [anchor for anchor in frozen()[page] if anchor not in rendered]
    assert not missing, f"docs/{page} rendered has no {missing}"
