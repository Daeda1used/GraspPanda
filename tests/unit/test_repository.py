"""Repository-level invariants that keep the public face of the project consistent.

The READMEs and the getting-started pages are written by hand. These tests hold every
number and every "works under these methods" statement they make to what computes it: the
method metafiles, the measured records (:mod:`grasppanda.combine.reach`), the slots each
method has (:mod:`tools.docs_chips`), the configs and ``pyproject.toml``. A README that
prints a number no test computes fails :func:`test_the_readmes_print_no_number_that_is_not_computed`.
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10: the backport, which pip and pytest bring along
    import tomli as tomllib

from .conftest import REPO

READMES = ("README.md", "README_zh-CN.md")
#: The heading of the model zoo in each README (GitHub anchors ``#-model-zoo``, ``#-模型库``).
ZOO = {"README.md": "## 🏆 Model zoo", "README_zh-CN.md": "## 🏆 模型库"}
#: The documentation site the READMEs link: on GitHub a page of ``docs/`` is MkDocs source
#: (folds, tabs and chips do not render), so the READMEs link the built page.
SITE = "https://daeda1used.github.io/GraspPanda/"


def site_page(page: str) -> str:
    """The URL of the built page ``docs/<page>`` (``methods/gsnet.md`` → ``…/methods/gsnet/``)."""
    return f"{SITE}{page.removesuffix('.md')}/"


def text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def flat(words: str) -> str:
    """Whitespace collapsed, so a phrase may wrap at any line."""
    return " ".join(words.split())


def headings(name: str) -> list[int]:
    """Heading levels of a Markdown file, in order, ignoring fenced code blocks."""
    levels, fenced = [], False
    for line in text(name).splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (match := re.match(r"(#{1,6}) ", line)):
            levels.append(len(match.group(1)))
    return levels


def table(page: str, start: str) -> list[list[str]]:
    """The cells of the first Markdown table at or below the line ``start``, without its
    two header lines."""
    lines = page[page.index(start) :].splitlines()
    first = next(index for index, line in enumerate(lines) if line.lstrip().startswith("|"))
    rows = []
    for line in lines[first:]:
        if not line.lstrip().startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[2:]


def join(names: list[str]) -> str:
    """``a, b and c``, as the pages write a list of names."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@pytest.fixture(scope="module")
def chips():
    from tools import docs_chips

    return docs_chips


# --------------------------------------------------------------------------- structure


def test_readmes_have_the_same_structure():
    english, chinese = READMES
    assert headings(english) == headings(chinese)
    count = {name: text(name).count("\n|") for name in READMES}
    assert count[english] == count[chinese], "tables differ in size between the two READMEs"


@pytest.mark.parametrize("name", READMES)
def test_brand_assets_referenced_by_the_readme_exist(name):
    for asset in re.findall(r'(?:src|srcset)="(resources/[^"]+)"', text(name)):
        assert (REPO / asset).is_file(), asset


@pytest.mark.parametrize("name", READMES)
def test_the_readme_has_one_hero_and_few_code_blocks(name):
    """One picture above the fold (the explorer recording), at most three visible code blocks."""
    page = re.sub(r"<details.*?</details>", "", text(name), flags=re.S)
    pictures = re.findall(r'<img src="(resources/[^"]+)"', page)
    assert pictures == ["resources/banner-light.png", "resources/explorer-light.webp"]
    assert page.count("```") // 2 <= 3
    assert "showcase" not in text(name) and "demo.webp" not in text(name)


def test_shipped_configs_are_valid():
    from grasppanda.config import Config, configs_root

    for path in sorted(configs_root().rglob("*.yaml")):
        if path.name != "metafile.yaml":  # model-zoo metadata, validated by grasppanda.hub
            Config.load(path)


def test_the_brand_pictures_are_what_their_tool_writes():
    from tools import make_brand_assets

    assert not make_brand_assets.stale(), "stale: run `python -m tools.make_brand_assets`."


def test_the_pipeline_figure_draws_both_families_from_the_slots_of_the_methods(chips):
    """Lane one holds the slots of the seed-based skeleton, lane two HGGD's, the shared band
    the slots every method has; nothing is drawn that no shipped method has."""
    from tools import make_brand_assets

    found = make_brand_assets.lanes()
    seed = [info.name for info in chips.methods() if info.family == "seed-based"]
    assert list(found.seed_methods) == seed
    assert [box.key for box in found.seed_boxes] + ["loss"] == [
        slot for slot in chips.SLOT_ORDER if slot != "recipe" and chips.scope(slot) == "seed-based"
    ]
    assert [box.key for box in found.own_boxes if box.key] == ["image_encoder"]
    assert set(found.shared) == {"post", "optimizer", "schedule", "hooks", "augment"}
    figure = (REPO / "resources" / "pipeline.svg").read_text(encoding="utf-8")
    for name in [*seed, *found.own_methods, "GraspNet-1Billion", "GraspClutter6D"]:
        assert name in figure
    assert "dense-contact" not in figure and "Contact-GraspNet" not in figure


# ------------------------------------------------------------- numbers of the READMEs


def gate() -> float:
    """The AP gate of the model zoo, as its page states it."""
    match = re.search(r"Gate: `\|reproduced − reported\| ≤ ([\d.]+) AP`", text("docs/model_zoo.md"))
    assert match, "docs/model_zoo.md no longer states the gate"
    return float(match.group(1))


def reported_seen(method: str) -> list[tuple[bool | None, float]]:
    """``(collision filter, AP on seen)`` of every number ``method`` reports on
    GraspNet-1Billion (RealSense), in metafile order."""
    from grasppanda.hub import load_metafiles

    return [
        (result.collision_filter, result.ap["seen"])
        for meta in load_metafiles()
        if meta.method == method
        for model in meta.models
        if model.protocol == "graspnet1b-sv-v1" and model.camera == "realsense"
        for result in model.results
        if result.provenance == "reported"
    ]


#: What ``tests/integration/test_eval_parity.py::test_recorded_frames_match_stage_by_stage``
#: holds the GraspNet-1Billion evaluator to. Its recordings of ``graspnetAPI`` live in the
#: dataset cache, not in the repository, so two facts of the last run (2026-10-18: 13 suites
#: passed, no mismatch at any stage) are pinned here: the edge suite records one frame per
#: constructed case of ``tools/make_eval_fixtures.edge_cases`` (10), and two dump frames have
#: tied scores, so no single official result, and are not compared (scene 100 frame 13,
#: scene 170 frame 204).
EDGE_CASES, TIED_FRAMES = 10, 2
#: Later papers that re-ran HGGD on GraspNet-1Billion (RealSense, seen): name → (arXiv id,
#: AP). RNGNet prints a re-implementation "without CD" (Table 1), GraRe its own re-run
#: (Table 1). No metafile holds third-party numbers, so they are pinned with their source.
HGGD_RERUNS = {"RNGNet": ("2406.01767", 58.35), "GraRe": ("2608.00946", 63.37)}


def evaluator_parity_frames() -> int:
    """The frames the recorded parity suites compare: a dump suite holds a whole scene, a
    label suite the frames it lists, the edge suite one frame per case; tied frames are left
    out."""
    from grasppanda.evaluation.graspnet1b import GRASPNET1B_SV_V1

    from ..integration.test_eval_parity import RECORDINGS

    frames = 0
    for (suite, _, _), arguments in RECORDINGS.items():
        if suite == "dump":
            frames += GRASPNET1B_SV_V1.frames_per_scene
        elif suite == "label":
            listed = arguments[arguments.index("--frames") + 1 :]
            frames += len(list(itertools.takewhile(lambda item: not item.startswith("--"), listed)))
        else:
            assert suite == "edge", suite
            frames += EDGE_CASES
    return frames - TIED_FRAMES


def claims(chips) -> dict[str, dict[str, bool]]:
    """Per README: every phrase with a number in it, and whether the records back it."""
    from grasppanda.combine.reach import level2, reach

    every = len(chips.methods())
    seed = sum(info.family == "seed-based" for info in chips.methods())
    families = len({info.family for info in chips.methods()})
    ptv3, ptv3_learns = reach("backbone", "ptv3"), level2("backbone", "ptv3")
    dino = reach("image_encoder", "dinov2-small-reg-depth-stem")
    dino_learns = level2("image_encoder", "dinov2-small-reg-depth-stem")
    par = len(ptv3_learns.on_a_par)
    words = {2: "two", 5: "five", 6: "six"}
    chinese = {2: "两", 5: "五", 6: "六"}
    ap = f"{gate():g} AP"
    hggd = reported_seen("HGGD")
    gsnet = dict(reported_seen("GSNet"))
    papers = {
        name: re.findall(r"\[(\w+)\]\(https://arxiv\.org/abs/([\d.]+)\)", text(name))
        for name in READMES
    }
    reruns = [(paper, arxiv) for paper, (arxiv, _) in HGGD_RERUNS.items()]
    rerun_ap = " and ".join(f"{value:.2f}" for _, value in HGGD_RERUNS.values())
    frames = f"{evaluator_parity_frames():,}"
    shared = {
        # PTv3 reaches every seed-based method at runs; trains like shipped except where worse
        "ptv3 runs everywhere": ptv3.fraction == f"{seed}/{seed}" and ptv3.tone == "ok",
        "ptv3 level 2": len(ptv3_learns.compared) == seed and ptv3_learns.excluded == {},
        "ptv3 worse": ptv3_learns.tail == "worse under FineGrasp",
        "dino runs": dino.fraction == "HGGD ✓",
        "dino level 2": dino_learns.fraction == "HGGD: on a par",
        # the collision-filter mix-up: GSNet's seen AP without and with the filter
        "gsnet off": f"{gsnet[False]:.2f}" == "65.70",
        "gsnet on": f"{gsnet[True]:.2f}" == "67.12",
        # HGGD's two sets (paper, repository) and two re-runs by later papers
        "hggd sets": [f"{value:.2f}" for _, value in hggd] == ["59.36", "64.45"],
        "hggd re-runs": all(found == reruns for found in papers.values()),
    }
    return {
        "README.md": {
            f"{words[every].capitalize()} published methods": True,
            f"{words[every].capitalize()} methods in {words[families]} families": True,
            f"PTv3 runs under {ptv3.fraction} seed-based methods": shared["ptv3 runs everywhere"],
            f"under {par}/{seed} it trains like shipped": shared["ptv3 level 2"],
            "(worse under FineGrasp)": shared["ptv3 worse"],
            "The DINOv2 encoder runs under HGGD": shared["dino runs"],
            "trains on a par with HGGD's own": shared["dino level 2"],
            f"All {every} methods": chips.SCOPE_TEXT["all"] == f"All {every} methods",
            f"Seed-based · {seed} methods": chips.SCOPE_TEXT["seed-based"]
            == f"Seed-based · {seed} methods",
            f"within {ap} of the reported number": True,
            f"within {ap} on each split": True,
            f"HGGD** publishes {words[len(hggd)]} sets of numbers": True,
            f"value for value on {frames} recorded frames": True,
            "GSNet reports 65.70 AP": shared["gsnet off"],
            "without the filter and 67.12 with it": shared["gsnet on"],
            "lists 59.36 AP": shared["hggd sets"],
            "and 64.45 as its repository result": shared["hggd sets"],
            f"{words[len(reruns)]} later papers that re-ran it": shared["hggd re-runs"],
            f"report {rerun_ap}": shared["hggd re-runs"],
        },
        "README_zh-CN.md": {
            f"{chinese[every]}个已发表的方法": True,
            f"{chinese[families]}个家族共{chinese[every]}个方法": True,
            f"PTv3 在全部 {seed} 个基于种子点的方法下都能跑": shared["ptv3 runs everywhere"],
            f"其中 {par} 个方法下训练得和原装组件不相上下": shared["ptv3 level 2"],
            "（FineGrasp 下较差）": shared["ptv3 worse"],
            "DINOv2 编码器在 HGGD 下能跑": shared["dino runs"],
            "训练得和 HGGD 的原装组件不相上下": shared["dino level 2"],
            f"全部 {every} 个方法": True,
            f"基于种子点 · {seed} 个方法": True,
            f"相差不超过 {ap}": True,
            f"**HGGD** 公开了{chinese[len(hggd)]}组数字": True,
            f"在 {frames} 个记录帧上": True,
            "不加过滤是 65.70 AP": shared["gsnet off"],
            "加了是 67.12": shared["gsnet on"],
            "把 59.36 AP": shared["hggd sets"],
            "把 64.45 列为代码仓库的结果": shared["hggd sets"],
            f"给出的是 {rerun_ap.replace(' and ', ' 和 ')}": shared["hggd re-runs"],
        },
    }


def visible_prose(name: str, chips) -> str:
    """The README as prose: no code, tags, link targets, targets table (held by its own test),
    venues (held to the metafiles), dates of the news, or names that contain digits."""
    page = text(name)
    zoo = page.index(ZOO[name])
    start = page.index("\n|", zoo)
    end = page.index("\n\n", start)
    page = page[:start] + page[end:]  # the targets table
    page = re.sub(r"```.*?```", " ", page, flags=re.S)
    page = re.sub(r"<!--.*?-->|<[^>]+>|`[^`]*`|\]\([^)]*\)", " ", page, flags=re.S)
    for info in chips.methods():
        page = page.replace(info.venue, " ")
    page = re.sub(r"\b\d{4}-\d{2}\b|Apache License 2\.0", " ", page)
    return flat(page)


def without_names(prose: str) -> str:
    """Prose without the names that contain digits (6-DoF, PTv3, GraspNet-1Billion, M0)."""
    return re.sub(r"[\w.+-]*[A-Za-z][\w.+-]*\d[\w.+-]*|\d[\w.+-]*[A-Za-z][\w.+-]*", " ", prose)


@pytest.mark.parametrize("name", READMES)
def test_every_number_the_readme_prints_is_computed(name, chips):
    """A count or a fraction in a README is a phrase that the records and metafiles back."""
    found = claims(chips)[name]
    prose = visible_prose(name, chips)
    for phrase, backed in found.items():
        assert phrase in prose, f"{name}: the phrase {phrase!r} is gone or the records changed"
        assert backed, f"{name}: {phrase!r} is no longer what the records say"


@pytest.mark.parametrize("name", READMES)
def test_the_readmes_print_no_number_that_is_not_computed(name, chips):
    prose = visible_prose(name, chips)
    for phrase in claims(chips)[name]:
        prose = prose.replace(phrase, " ")
    prose = without_names(prose)
    numbers = re.findall(r"\d+(?:[./]\d+)*", prose)
    spelled = re.findall(r"\b(two|three|four|five|six|seven|eight|nine|ten)\b", prose, re.I)
    spelled += re.findall(r"(?<!第)[两二三四五六七八九十][个组种项]", prose)
    assert not numbers and not spelled, (
        f"{name} prints {numbers + spelled} outside a phrase a test computes: add the phrase "
        "to claims() in tests/unit/test_repository.py, computed from the records"
    )


@pytest.mark.parametrize("name", READMES)
def test_the_methods_of_the_readme_are_those_of_the_metafiles_in_page_order(name, chips):
    family = {
        "README.md": {"seed-based": "seed-based skeleton", "own": "own family"},
        "README_zh-CN.md": {"seed-based": "基于种子点的骨架", "own": "自成一族"},
    }[name]
    reads = {"README.md": {"depth": "depth"}, "README_zh-CN.md": {"depth": "深度"}}[name]
    expected = [
        [
            f"[{info.name}]({site_page(info.page)})",
            info.venue,
            family[info.family],
            reads.get(info.inputs, info.inputs),
        ]
        for info in chips.methods()
    ]
    heading = {"README.md": "## 🧩 What's inside", "README_zh-CN.md": "## 🧩 内容一览"}[name]
    assert table(text(name), heading) == expected, f"{name}: the methods table"


#: The words of the scope table of each README → the slots they name.
SLOT_WORDS = {
    "README.md": {
        "post-processing": ["post"],
        "training recipe": ["optimizer", "schedule", "hooks", "augment"],
        "scene encoder": ["backbone"],
        "seed proposal": ["proposal"],
        "view": ["view"],
        "assigner": ["assigner"],
        "grouper": ["grouper"],
        "head bundle": ["head"],
        "loss": ["loss"],
        "image encoder": ["image_encoder"],
    },
    "README_zh-CN.md": {
        "后处理": ["post"],
        "训练方案": ["optimizer", "schedule", "hooks", "augment"],
        "场景编码器": ["backbone"],
        "种子点提议": ["proposal"],
        "视角": ["view"],
        "分配器": ["assigner"],
        "分组": ["grouper"],
        "检测头": ["head"],
        "损失": ["loss"],
        "图像编码器": ["image_encoder"],
    },
}


@pytest.mark.parametrize("name", READMES)
def test_the_scope_of_every_slot_in_the_readme_is_computed(name, chips):
    every = len(chips.methods())
    seed = sum(info.family == "seed-based" for info in chips.methods())
    scope_words = {
        "README.md": dict(chips.SCOPE_TEXT),
        "README_zh-CN.md": {
            "all": f"全部 {every} 个方法",
            "seed-based": f"基于种子点 · {seed} 个方法",
            "hggd": "仅 HGGD",
        },
    }[name]
    start = {"README.md": "**Slots**", "README_zh-CN.md": "**槽位**"}[name]
    named = []
    for slots_text, scope_text in table(text(name), start):
        for word in re.split(r", |、", slots_text):
            slots = SLOT_WORDS[name][word]
            named += slots
            for slot in slots:
                assert scope_words[chips.scope(slot)] == scope_text, (name, slot, scope_text)
    listed = [slot for slot in chips.SLOT_ORDER if slot != "recipe" and chips.slot_methods(slot)]
    assert sorted(named) == sorted(listed), f"{name}: the scope table must name every slot once"


@pytest.mark.parametrize("name", READMES)
def test_every_slot_row_of_the_readme_links_its_page(name, chips):
    """Each row of the slot table links the built page of its slot; the loss row names the
    key a loss term is set with (``<stage>.<term>_loss``), which compose accepts."""
    page = text(name)
    keys = set()
    for slot_page, cell in re.findall(
        rf"\]\({re.escape(SITE)}(slots/[a-z_]+)/\) \| ((?:`[a-z_<>.]+`[,、 ]*)+)", page
    ):
        for key in re.findall(r"`([a-z_<>.]+)`", cell):
            key = "loss" if key == "<stage>.<term>_loss" else key
            keys.add(key)
            assert chips.SLOT_PAGES[key] == f"{slot_page}.md", (name, key)
    assert keys == set(chips.SLOT_ORDER)
    assert "| `loss` |" not in page, f"{name}: `--set loss=` is no key; a loss term is"


def test_the_loss_key_of_the_readmes_is_one_compose_accepts(chips):
    """``head.score_loss``, the example of the loss row, is a loss term compose sets under
    every method that has the loss slot; ``--set loss=`` is no key."""
    from grasppanda.combine.compose import ComposeError, compose

    for name, example in zip(READMES, ("e.g. `head.score_loss`", "例如 `head.score_loss`")):
        assert example in text(name), name
    for info in chips.methods():
        if info.name in chips.slot_methods("loss"):
            assert compose(info.config, ["head.score_loss=uncertainty-weighting"]).accepted
            with pytest.raises(ComposeError, match="no slot 'loss'"):
                compose(info.config, ["loss=focal"])


#: How the targets table of each README writes the collision filter.
FILTER = {"on": True, "off": False, "not stated": None, "开": True, "关": False, "未说明": None}


@pytest.mark.parametrize("name", READMES)
def test_the_targets_of_the_readme_are_the_reported_rows_of_the_metafiles(name, chips):
    """Every reported RealSense row on GraspNet-1Billion, once, in page order, with the venue
    and the collision filter its metafile states."""
    from grasppanda.hub import load_metafiles

    from .test_zoo_pages import numbers_text

    order = [info.name for info in chips.methods()]
    venue = {info.name: info.venue for info in chips.methods()}
    expected = set()
    for meta in load_metafiles():
        for model in meta.models:
            if model.protocol != "graspnet1b-sv-v1" or model.camera != "realsense":
                continue
            for result in model.results:
                if result.provenance == "reported":
                    expected.add((meta.method, result.collision_filter, numbers_text(result)))
    shown, positions = set(), []
    for cells in table(text(name), ZOO[name]):
        method = cells[0]
        assert cells[1] == venue[method], (name, method)
        state = FILTER[cells[2]]
        for numbers in re.findall(r"\d+\.\d\d / \d+\.\d\d / \d+\.\d\d", cells[3]):
            shown.add((method, state, numbers))
        positions.append(order.index(method))
    assert shown == expected, f"{name}: the targets table and the metafiles disagree"
    assert positions == sorted(positions), f"{name}: the targets are not in page order"


def test_the_core_dependencies_named_by_the_readmes_are_those_of_the_package():
    project = tomllib.loads(text("pyproject.toml"))["project"]
    names = {re.split(r"[<>=!~ \[]", spec)[0] for spec in project["dependencies"]}
    for name in READMES:
        cell = next(line for line in text(name).splitlines() if "📦" in line).split("📦")[1]
        assert set(re.findall(r"`([a-z_-]+)`", cell)) == names, name


def test_every_option_is_probed_under_each_method_it_applies_to_or_says_why():
    """The pitch: "Each option was composed under every method it fits, those accepted were
    probed on real frames, and the few without a record say why"."""
    from grasppanda.combine.levels import NOT_PROBED
    from grasppanda.combine.options import _class_index
    from grasppanda.combine.reach import UNDECIDED_WORDS, reach

    for slot, name in _class_index():
        found = reach(slot, name)
        assert found.note or UNDECIDED_WORDS[NOT_PROBED] not in found.undecided.values(), (
            slot,
            name,
            found.tail,
        )
        assert found.note or found.applies, (slot, name)


# ---------------------------------------------------- examples of the front door pages


def compose_requests(page: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """``(config, [(slot, option), ...])`` of every ``grasppanda compose`` command of a page."""
    joined = re.sub(r"\\\n\s*", " ", page)
    found = []
    for match in re.finditer(r"grasppanda compose (\S+)((?:\s+--set\s+\S+)+)", joined):
        sets = [tuple(item.split("=", 1)) for item in re.findall(r"--set\s+(\S+)", match.group(2))]
        found.append((match.group(1), sets))
    return found


FRONT_DOOR = ("README.md", "README_zh-CN.md", "docs/get_started/swap.md")


@pytest.mark.parametrize("name", FRONT_DOOR)
def test_every_swap_shown_on_the_front_door_is_recorded_runs(name, chips):
    from grasppanda.combine.options import _class_index, measured

    records = measured()
    configs = {info.config: info.name for info in chips.methods()}
    requests = compose_requests(text(name))
    assert requests, name
    for config, sets in requests:
        method = configs[config]
        for slot, option in sets:
            cells = _class_index()[(slot, option)][method]
            verdicts = {records.get(cell)["verdict"] for cell in cells}
            assert verdicts == {"runs"}, (name, method, slot, option, verdicts)


@pytest.mark.parametrize("name", READMES)
def test_the_readme_swaps_a_method_other_than_gsnet_and_an_image_encoder(name, chips):
    configs = {info.config: info for info in chips.methods()}
    methods = [(configs[config], dict(sets)) for config, sets in compose_requests(text(name))]
    assert any(
        info.family == "seed-based" and info.name != "GSNet" and "backbone" in sets
        for info, sets in methods
    )
    assert any(info.name == "HGGD" and "image_encoder" in sets for info, sets in methods)


def test_the_swap_page_has_a_tab_per_method_in_page_order(chips):
    page = text("docs/get_started/swap.md")
    names = [info.name for info in chips.methods()]
    sets = re.split(r"\n(?!=== |    |\n)", page)
    tab_sets = [re.findall(r'^=== "([^"]+)"', block, re.M) for block in sets]
    tab_sets = [labels for labels in tab_sets if labels]
    assert tab_sets == [names, names]
    assert "A swapped model has no weights until you train it" in page
    assert '--8<-- "docs/includes/caveat_probe.md"' in page


def test_the_probe_example_reads_the_sample_split_of_one_scene():
    """The swap page says the probe reads a few frames of scene 0 (the `sample` split)."""
    import yaml

    manifest = yaml.safe_load(
        (REPO / "grasppanda" / "datasets" / "splits" / "graspnet1b.yaml").read_text()
    )
    assert manifest["splits"]["sample"]["scenes"] == [0]
    page = flat(text("docs/get_started/swap.md"))
    assert "the `sample` split, a few frames of scene 0" in page
    assert "scene 0 of [GraspNet-1Billion](../user_guides/datasets.md)" in page


def test_the_probe_assets_of_each_method_are_those_its_config_reads(chips):
    """Each tab of the probe step builds exactly the label assets the method's config reads."""
    from grasppanda.config import Config

    flags = {
        "tolerance": "--tolerance",
        "graspness": "--graspness",
        "compact_labels": "--compact-labels",
        "view_grasps": "--view-grasps",
    }
    page = text("docs/get_started/swap.md")
    probe = page[page.index("## 2. Probe") : page.index("## 3. Train")]
    tabs = re.split(r'^=== "([^"]+)"$', probe, flags=re.M)[1:]
    commands = dict(zip(tabs[::2], tabs[1::2], strict=True))
    for info in chips.methods():
        dataset = Config.load(info.config)["data"]["train"]["dataset"]
        expected = {"--labels"} | {flag for key, flag in flags.items() if dataset.get(key)}
        shown = set(
            re.findall(
                r"--(?:labels|tolerance|graspness|compact-labels|view-grasps)\b",
                commands[info.name].split("grasppanda probe")[0],
            )
        )
        assert shown == expected, (info.name, shown, expected)
        if dataset.get("scale_statistic"):
            assert "histogram" in commands[info.name], info.name


def test_the_front_door_lists_the_methods_in_page_order_with_their_venues(chips):
    home = text("docs/index.md")
    shown = re.findall(r"\[([A-Za-z-]+) <small>([^<]+)</small>\]\(methods/(\w+)\.md\)", home)
    assert shown == [(info.name, info.venue, info.folder) for info in chips.methods()]
    run = text("docs/get_started/run_a_method.md")
    cards = re.findall(r"-   \*\*([A-Za-z-]+)\*\* · ([^·\n]+) · ", run)
    assert cards == [(info.name, info.venue) for info in chips.methods()]
    links = re.findall(r"\(\.\./methods/(\w+)\.md#run-in-five-minutes\)", run)
    assert links == [info.folder for info in chips.methods()]
    assert '--8<-- "docs/includes/choose_method.md"' in run


def extras_of(config: str) -> set[str]:
    """The extras the registered classes of a config declare."""
    from grasppanda.config import Config

    return model_extras(Config.load(config)["model"])


def model_extras(model: dict) -> set[str]:
    """The extras the registered classes of a model node declare."""
    import grasppanda.registry as registry

    found, types = set(), []

    def walk(node):
        if isinstance(node, dict):
            if "type" in node:
                types.append(node["type"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(model)
    registries = [
        getattr(registry, name)
        for name in dir(registry)
        if name.isupper() and hasattr(getattr(registry, name), "get")
    ]
    for name in types:
        for found_in in registries:
            try:
                cls = found_in.get(name)
            except Exception:
                continue
            if cls is not None:
                found |= set(getattr(cls, "extras", ()) or ())
    return found


def test_what_each_method_needs_installed_is_what_its_config_declares(chips):
    plain = [info.name for info in chips.methods() if not extras_of(info.config)]
    sparse = [info.name for info in chips.methods() if extras_of(info.config) == {"spconv"}]
    assert len(plain) + len(sparse) == len(chips.methods())
    install = flat(text("docs/get_started/installation.md"))
    assert f"{join(plain)} need nothing more. {join(sparse)} use sparse convolution" in install
    faq = flat(text("docs/faq.md"))
    assert f"{join(plain)} need no extra; {join(sparse)} need spconv" in faq
    # Run a method says it through the generated chooser it includes
    assert '--8<-- "docs/includes/choose_method.md"' in text("docs/get_started/run_a_method.md")
    chooser = text("docs/includes/choose_method.md")
    assert f"{join(plain)} need no extra install." in flat(chooser)
    rows = {
        line.split("](")[0].removeprefix("| ["): line
        for line in chooser.splitlines()
        if line.startswith("| [")
    }
    assert all("`spconv`" in rows[name] for name in sparse), rows
    assert not any("`spconv`" in rows[name] for name in plain), rows


def test_the_example_scene_fold_says_what_the_scene_is():
    """Six primitives, the image size of the benchmark's cameras, no dataset involved."""
    import inspect

    from grasppanda.apis import example
    from grasppanda.apis.example import example_scene

    scene = example_scene()
    words = {6: "six"}
    height, width = scene.depth.shape
    run = flat(text("docs/get_started/run_a_method.md"))
    assert f"{words[len(scene.objects)]} geometric primitives on a table" in run
    assert f"ray-cast into a {width} × {height} depth image in millimetres" in run
    assert scene.depth_scale == 1000.0 and "../assets/example-scene.png" in run
    source = inspect.getsource(example)
    assert "no dataset image" in source and "outside its training distribution" in source


def test_the_workspace_sentence_of_run_a_method_is_what_the_metafiles_state(chips):
    """The seed-based methods were scored on a cloud cropped to the objects (their results
    say `gt_assisted: workspace`); HGGD crops none (`gt_assisted: none`)."""
    from grasppanda.hub import load_metafiles

    crops = {}
    for meta in load_metafiles():
        for model in meta.models:
            for result in model.results:
                if result.provenance != "reported" and result.inputs:
                    crops.setdefault(meta.method, set()).add(
                        "gt_assisted: workspace" in result.inputs
                    )
    seed = {info.name for info in chips.methods() if info.family == "seed-based"}
    assert {name for name, found in crops.items() if found == {True}} == seed
    assert {name for name, found in crops.items() if found == {False}} == {"HGGD"}
    run = flat(text("docs/get_started/run_a_method.md"))
    assert (
        "For the seed-based methods, `--workspace` crops the cloud to the objects, as they "
        "were trained; HGGD reads the whole image and crops no workspace." in run
    )


@pytest.mark.parametrize("name", READMES)
def test_the_quick_start_installs_the_extras_its_examples_need(name):
    """Every extra a composed example of the README declares is in its `pip install` line
    (spconv as the build for a CUDA version), and that line names only real extras."""
    from grasppanda.combine.compose import compose

    page = text(name)
    line = re.search(r'pip install -e "\.\[([a-z0-9,_-]+)\]"', page)
    assert line, f"{name}: the quick start installs no extra"
    named = set(line.group(1).split(","))
    package = set(tomllib.loads(text("pyproject.toml"))["project"]["optional-dependencies"])
    assert named <= package, named - package
    needed = set()
    for config, sets in compose_requests(page):
        result = compose(config, [f"{slot}={option}" for slot, option in sets])
        assert result.accepted, (name, config, sets)
        needed |= model_extras(result.merged["model"])
    assert needed, name
    for extra in needed:
        builds = {item for item in named if item.startswith(f"{extra}-")}
        assert extra in named or builds, f"{name}: the examples need the extra {extra}"


def test_every_extra_of_the_package_is_on_the_install_page():
    extras = tomllib.loads(text("pyproject.toml"))["project"]["optional-dependencies"]
    page = text("docs/get_started/installation.md")
    for name in extras:
        assert f"`{name}`" in page, name


def test_every_method_reports_with_and_without_the_collision_filter(chips):
    """The home page and the swap page say `test` scores with the filter on and off."""
    from grasppanda.config import Config

    for info in chips.methods():
        variants = Config.load(info.config)["evaluation"]["variants"]
        assert {"raw", "cd"} <= set(variants), info.name


# ----------------------------------------------------------------- tour of the config


def cli(*args: str) -> str:
    env = {key: value for key, value in os.environ.items() if key != "FORCE_COLOR"}
    result = subprocess.run(
        [sys.executable, "-m", "grasppanda.cli.main", *args],
        cwd=REPO,
        env={**env, "NO_COLOR": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def blocks(page: str) -> list[str]:
    """The fenced blocks of a page, without their indent."""
    found = re.findall(r"^( *)```[a-z]*\n(.*?)^\1```", page, flags=re.S | re.M)
    return ["\n".join(line[len(indent) :] for line in body.splitlines()) for indent, body in found]


def test_the_tour_of_the_config_shows_what_the_cli_prints_now():
    """Every output on the page is the CLI's, whole, as it prints it today."""
    toy = "projects/example_project/configs/toy_detector.yaml"
    shown = blocks(text("docs/get_started/quickstart.md"))
    for args in (
        [toy, "--check"],
        [toy, "--defaults"],
        [toy, "--opts", "model.head.num_angles=6"],
        [toy, "--opts", "model.grouper.type=ToyDepthGrouper"],
    ):
        printed = without_torch_version_keys(cli("print-config", *args).strip())
        pages = [without_torch_version_keys(block.strip()) for block in shown]
        assert printed in pages, f"stale output of {args}"


#: Constructor arguments PyTorch added or dropped between the versions CI runs (Adam's
#: ``decoupled_weight_decay`` since 2.6, the schedulers' ``verbose`` until 2.7): `--defaults`
#: prints them only under the versions that have them.
TORCH_VERSION_KEYS = ("decoupled_weight_decay:", "verbose:")


def without_torch_version_keys(output: str) -> str:
    return "\n".join(
        line for line in output.splitlines() if not line.strip().startswith(TORCH_VERSION_KEYS)
    )


def test_the_toy_inherits_the_method_neutral_recipe_the_tour_names():
    from grasppanda.config import Config

    toy = Config.load(REPO / "projects/example_project/configs/toy_detector.yaml")
    assert toy["optimizer"]["type"] == "Adam" and float(toy["optimizer"]["lr"]) == 1e-3
    assert toy["train"]["epochs"] == 10
    page = flat(text("docs/get_started/quickstart.md"))
    assert "(`_base_/schedules/adam_10e.yaml`: Adam, a learning rate of 1e-3, 10 epochs)" in page
    assert "gsnet" not in text("projects/example_project/configs/toy_detector.yaml")
