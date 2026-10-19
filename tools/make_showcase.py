"""Generate the animated showcase: compose, a refusal, a probe, on one seed-based method.

``resources/showcase-{light,dark}.svg`` (and the same files under ``docs/assets/`` for the
documentation) are one loop of twenty-five seconds: a terminal in which a combination is
composed, a wrong swap is refused with the conflict and the fix, and a probe ends with its
verdict, under the pipeline of the method whose slots flip to the chosen options. They are
SVG files animated with CSS keyframes: no script, no font download, a few dozen kilobytes.

The method is :data:`METHOD` and the combination :data:`SWAP`, a cell of the measured grid
(:data:`RECORD`) whose record says ``runs``: the picture shows a combination that was
measured, on a method that is one of six. ``tests/unit/test_showcase.py`` holds the record,
the composed config hash and the pictures to it.

Every line of terminal text is real output. ``resources/showcase_transcript.json`` holds
what three commands printed, whole; the picture shows a selection of those lines
(:data:`SCENES`), and :func:`shown` refuses a line that is not in the transcript. Lines
are dropped, never invented; a line that is too long for the picture is wrapped at a
word. Shown still (no animation, or reduced motion asked for), the picture is its last
frame: the probe with its verdict.

    python -m tools.make_showcase            # rewrite the two pictures from the transcript
    python -m tools.make_showcase --check    # exit 1 when a picture is stale
    python -m tools.make_showcase --capture  # run the three commands again (needs
                                             # GraspNet-1Billion and a GPU for the probe)

To show another method, set :data:`METHOD`, :data:`SWAP`, :data:`REFUSED` and
:data:`RECORD`, capture, and adjust the line patterns of :data:`SCENES` to its output.

``tests/unit/test_showcase.py`` runs the check.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "resources"
#: The documentation's copy of the pictures (``docs/assets/showcase-{light,dark}.svg``).
DOCS = REPO / "docs" / "assets"
TRANSCRIPT = OUT / "showcase_transcript.json"

SANS = "'Inter', 'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'DejaVu Sans Mono', 'SFMono-Regular', Consolas, Menlo, monospace"
GREEN, DEEP_GREEN, RED = "#2da44e", "#1a7f37", "#cf222e"

#: The method of the picture: a seed-based method (its stages are the boxes drawn).
METHOD = "FineGrasp"
#: Its shipped config (``grasppanda.combine.grid.METHODS``).
CONFIG = "finegrasp/finegrasp_sunet14_g1b-realsense_10e"
#: The combination that is composed and probed: two slots swapped at once.
SWAP = ("backbone=point-transformer-v1", "grouper=cylinder-invres")
#: The cell of the measured grid that is this combination; its record says ``runs``.
RECORD = "cross/FineGrasp/point-transformer-v1+cylinder-invres"
#: A swap the contracts refuse under the method, whatever compose inserts.
REFUSED = "view=objectness-view"
#: The file the combination is written to.
COMPOSED = "finegrasp_ptv1_invres.yaml"
#: The line in the title bar of the terminal. The exact version the transcript was
#: captured with is in the transcript file; it names a development build.
CAPTION = "real output of grasppanda compose and probe · lines dropped, none invented"
#: The three commands, as a user types them after ``grasppanda``.
COMMANDS: dict[str, list[str]] = {
    "compose": [
        "compose",
        CONFIG,
        *[word for pair in SWAP for word in ("--set", pair)],
        "-o",
        COMPOSED,
    ],
    "refuse": ["compose", CONFIG, "--set", REFUSED],
    "probe": ["probe", COMPOSED, "--iters", "20", "--frames", "8"],
}
#: The slots the combination swaps, and the slot of the refused swap.
SWAPPED = {pair.split("=", 1)[0]: "ok" for pair in SWAP}
#: Scene → the seconds it is on screen, the slots it changes (``slot: state``), and the
#: lines of its output that are shown, each a regular expression that must match a line
#: of the transcript, in this order. A line that continues the one above it in the
#: transcript (the command line tool wraps at 100 columns) is one paragraph with it.
SCENES: dict[str, dict[str, Any]] = {
    "compose": {
        "seconds": 8.0,
        "flips": SWAPPED,
        "lines": [
            r"in_channels: 6\. The base config feeds real point features",
            r"^    point\), so the encoder reads those instead of the constant ones",
            r"✓ backbone · proposal$",
            r"✓ grouper · head$",
            r"✓ valid",
            r"^✓ written",
        ],
    },
    "refuse": {
        "seconds": 8.0,
        "flips": {REFUSED.split("=", 1)[0]: "no"},
        "lines": [
            r"✗ view · assigner \[structural\]:",
            r"^    labels, but CompactLabelAssigner returns 'compact' ones",
            r"^    DirectionViewStage, HealpixViewStage",
            r"^    RomaRotationViewStage, TemplateViewStage",
            r"^✗ refused: nothing was written",
        ],
    },
    "probe": {
        "seconds": 9.0,
        "flips": SWAPPED,
        "lines": [
            r"^  data ",
            r"^    loss/view ",
            r"^    loss/width ",
            r"^    loss  ",
            r"^  predicted ",
            r"^  gradient ",
            r"^✓ runs",
        ],
    },
}
#: Columns of the drawn terminal. The type is large enough to read at the width of a
#: README, so the terminal is narrower than the one the transcript was captured in, and
#: long lines are wrapped at a word, never cut.
COLUMNS = 74
#: Seconds between two lines of output, and the least a scene's last line stays.
STEP, HOLD = 0.7, 3.0


# ----------------------------------------------------------------------------- capture


def capture() -> dict[str, Any]:
    """Run the three commands in an empty directory and return what they printed."""
    import grasppanda

    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "100", "PYTHONPATH": str(REPO)}
    env.pop("FORCE_COLOR", None)
    from grasppanda.config import git_info

    git = git_info()
    commit = (git.get("commit") or "")[:7] + ("+dirty" if git.get("dirty") else "")
    # The version string is whatever the last build wrote; the commit names the code.
    found: dict[str, Any] = {
        "grasppanda": grasppanda.__version__,
        "commit": commit or None,
        "commands": {},
    }
    with tempfile.TemporaryDirectory() as directory:
        for name, args in COMMANDS.items():
            done = subprocess.run(
                [sys.executable, "-m", "grasppanda.cli.main", *args],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            output = (done.stdout + done.stderr).replace(directory + os.sep, "")
            found["commands"][name] = {
                "command": "grasppanda " + " ".join(args),
                "exit_status": done.returncode,
                "output": output.rstrip("\n").splitlines(),
            }
    return found


def transcript() -> dict[str, Any]:
    return json.loads(TRANSCRIPT.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- lines


def shown(name: str, record: dict[str, Any]) -> list[str]:
    """The lines of one command's output that the picture shows, verbatim."""
    output = record["output"]
    patterns = SCENES[name]["lines"]
    lines = []
    position = 0
    for pattern in patterns:
        hits = [i for i, line in enumerate(output) if i >= position and re.search(pattern, line)]
        if not hits:
            raise ValueError(f"{name}: no line of the transcript matches {pattern!r} in order")
        position = hits[0] + 1
        lines.append(output[hits[0]])
    return lines


def paragraphs(name: str, record: dict[str, Any]) -> list[list[str]]:
    """The shown lines grouped: a line and the transcript lines that continue it."""
    output = record["output"]
    groups: list[list[str]] = []
    previous = -2
    for line in shown(name, record):
        position = output.index(line, max(previous, 0))
        # A continuation hangs under a head that is indented less. A row of an aligned
        # table (the losses of a probe) starts at the same column as the row above it
        # and is never joined to it.
        head = groups[-1][0] if groups else ""
        hangs = len(head) - len(head.lstrip()) < 4
        if hangs and position == previous + 1 and line.startswith("    ") and line[4] != " ":
            groups[-1].append(line)
        else:
            groups.append([line])
        previous = position
    return groups


def wrap(lines: list[str]) -> list[str]:
    """One paragraph of the transcript at the width of the picture.

    A line that fits is drawn as it is. A longer one, or one the command line tool had
    already wrapped, is wrapped again at a word with a hanging indent: the words and
    their order are those of the transcript.
    """
    if len(lines) == 1 and len(lines[0]) <= COLUMNS:
        return list(lines)
    first = lines[0]
    indent = len(first) - len(first.lstrip())
    hang = len(lines[1]) - len(lines[1].lstrip()) if len(lines) > 1 else indent + 2
    text = " ".join([first.rstrip(), *(line.strip() for line in lines[1:])])
    return textwrap.wrap(
        text,
        width=COLUMNS,
        subsequent_indent=" " * hang,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _text(line: str) -> str:
    """Escaped for SVG, with the spaces of the terminal kept (no-break spaces)."""
    return html.escape(line).replace(" ", "\u00a0")


# ----------------------------------------------------------------------------- picture


def _docs_chips() -> Any:
    """``tools/docs_chips.py``: the words the pages use for a slot."""
    import importlib.util

    name = "docs_chips"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pipeline() -> list[dict[str, str]]:
    """The stages of the method and the option it ships in each, as ``grasppanda compose
    <config> --list`` lists them, under the titles of the documentation.

    Losses, post-processing and the training recipe are around the stages and are not drawn.
    """
    from grasppanda.combine import options
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    titles = _docs_chips().SLOT_TITLES
    stages = []
    for slot in options.method_slots(METHOD):
        if slot not in STAGE_REGISTRIES:
            continue
        found = options.slot_listing(METHOD, slot)
        stages.append(
            {"slot": slot, "title": titles[slot], "ships": found.shipped_option or found.shipped}
        )
    return stages


def _sets(name: str) -> dict[str, str]:
    args = COMMANDS[name]
    pairs = [args[i + 1] for i, arg in enumerate(args) if arg == "--set"]
    return dict(pair.split("=", 1) for pair in pairs)


class Frames:
    """Per-element keyframes: visible between two moments of the loop.

    An element that is visible when the loop ends is visible without the animation too
    (its own opacity is 1): where the picture is shown still, or where the reader asked
    for reduced motion, it is the last frame, complete, and not an empty terminal.
    """

    #: Seconds the last frame takes to leave when the loop starts again.
    OPENING = 0.4

    def __init__(self, total: float) -> None:
        self.total = total
        self.rules: list[str] = []
        self.known: dict[tuple[float, float, float], str] = {}

    def between(self, start: float, end: float, fade: float = 0.25) -> str:
        """A class that shows its element from ``start`` to ``end`` seconds."""
        key = (round(start, 3), round(end, 3), fade)
        if key in self.known:
            return self.known[key]
        name = f"k{len(self.rules)}"
        last = end >= self.total
        # The loop closes on its last frame and opens with a cross-fade out of it, so
        # that no moment of the loop, the first included, is an empty terminal.
        start = start if last else max(start, self.OPENING)

        def pct(seconds: float) -> str:
            return f"{max(0.0, min(100.0, 100 * seconds / self.total)):.2f}%"

        if last:
            stops = [
                ("0%", 1),
                (f"{pct(self.OPENING)},{pct(start)}", 0),
                (f"{pct(start + fade)},100%", 1),
            ]
        else:
            stops = [
                (f"0%,{pct(start)}", 0),
                (f"{pct(start + fade)},{pct(end - fade)}", 1),
                (f"{pct(end)},100%", 0),
            ]
        body = "".join(f"{at}{{opacity:{value}}}" for at, value in stops)
        self.rules.append(
            f"@keyframes {name}{{{body}}}"
            f".{name}{{opacity:{int(last)};animation:{name} {self.total:g}s linear infinite}}"
        )
        self.known[key] = name
        return name

    def css(self) -> str:
        still = "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
        return "".join(self.rules) + still


def _two_lines(option: str) -> list[str]:
    """A long option name on two lines, broken at a hyphen or before a capital."""
    if len(option) <= 15:
        return [option]
    places = [m.start() for m in re.finditer(r"(?<=-)\w|(?<=[a-z])(?=[A-Z])", option)]
    if not places:
        return [option]
    at = min(places, key=lambda place: abs(place - len(option) / 2))
    return [option[:at], option[at:]]


def _colour(line: str, palette: dict[str, str]) -> str:
    stripped = line.lstrip()
    if stripped.startswith("✓"):
        return palette["ok"]
    if stripped.startswith("✗"):
        return palette["no"]
    return palette["text"]


def svg(dark: bool) -> str:
    """The animated picture, light or dark."""
    record = transcript()["commands"]
    stages = _pipeline()
    total = sum(scene["seconds"] for scene in SCENES.values())
    frames = Frames(total)
    width = 1280
    page = {
        "bg": "#0d1117" if dark else "#ffffff",
        "card": "#161b22" if dark else "#ffffff",
        "line": "#30363d" if dark else "#d0d7de",
        "ink": "#e6edf3" if dark else "#1f2328",
        "muted": "#8b949e" if dark else "#57606a",
        "ok": "#3fb950" if dark else DEEP_GREEN,
        "ok_bg": "#12301c" if dark else "#dafbe1",
        "no": "#ff7b72" if dark else RED,
        "no_bg": "#3c1618" if dark else "#ffebe9",
        "term": "#010409" if dark else "#0d1117",
        "bar": "#161b22",
        "brand": "#3fb950" if dark else GREEN,
    }
    term = {"text": "#e6edf3", "ok": "#3fb950", "no": "#ff7b72", "muted": "#8b949e"}
    drawn = {
        name: (_typed(record[name]["command"]), [wrap(p) for p in paragraphs(name, record[name])])
        for name in SCENES
    }
    longest = max(len(typed) + sum(map(len, groups)) for typed, groups in drawn.values())
    box_w, box_h, gap, top = 190, 104, 16, 64
    t_top, t_left, pad, size, leading = top + box_h + 22, 28, 24, 24, 35
    t_height = 36 + pad + longest * leading + 8 + pad - 6
    height = t_top + t_height + 28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="GraspPanda: a combination is composed, a wrong '
        'swap is refused with the conflict and the fix, and a probe ends with its verdict">',
        "<style>/*KEYFRAMES*/</style>",
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="16" fill="{page["bg"]}" '
        f'stroke="{page["line"]}" stroke-width="2"/>',
    ]

    # ---- the pipeline: one box per slot, flipping to the option of the scene
    left = (width - (len(stages) * box_w + (len(stages) - 1) * gap)) / 2
    parts.append(
        f'<text x="{left}" y="42" font-family="{SANS}" font-size="24" font-weight="700">'
        f'<tspan fill="{page["brand"]}">GraspPanda</tspan>'
        f'<tspan fill="{page["muted"]}" font-weight="400" font-size="21">  ·  {METHOD}, one of '
        f"{len(_docs_chips().methods())} methods: every stage is a slot</tspan></text>"
    )
    starts, moment = {}, 0.0
    for name, scene in SCENES.items():
        starts[name] = moment
        moment += scene["seconds"]
    captions = {
        "compose": f"1 · compose: swap {len(SWAP)} slots",
        "refuse": "2 · a wrong swap is refused",
        "probe": "3 · probe: train it on real frames",
    }
    for name, text in captions.items():
        css = frames.between(starts[name], starts[name] + SCENES[name]["seconds"])
        parts.append(
            f'<text class="{css}" x="{width - left}" y="42" text-anchor="end" font-family="{SANS}" '
            f'font-size="21" font-weight="700" fill="{page["ink"]}">{html.escape(text)}</text>'
        )
    for index, stage in enumerate(stages):
        x = left + index * (box_w + gap)
        centre = x + box_w / 2
        if index:
            parts.append(
                f'<path d="M{x - gap + 2} {top + box_h / 2} h{gap - 5} m-5 -5 l5 5 l-5 5" '
                f'fill="none" stroke="{page["muted"]}" stroke-width="2.5" stroke-linecap="round"/>'
            )
        parts.append(
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="14" fill="{page["card"]}" '
            f'stroke="{page["line"]}" stroke-width="2.5"/>'
            f'<text x="{centre}" y="{top + 36}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="18" font-weight="700" fill="{page["ink"]}">{html.escape(stage["title"])}</text>'
        )
        # what the slot holds, scene by scene
        windows = []
        for name, scene in SCENES.items():
            state = scene["flips"].get(stage["slot"])
            sets = _sets("compose" if name == "probe" else name)
            begin = starts[name] + (1.2 if state and name != "probe" else 0.0)
            end = starts[name] + scene["seconds"]
            if state and begin > starts[name]:
                windows.append((starts[name], begin, stage["ships"], ""))
            windows.append(
                (begin, end, sets[stage["slot"]] if state else stage["ships"], state or "")
            )
        merged: list[tuple[float, float, str, str]] = []
        for window in windows:
            if merged and merged[-1][2:] == window[2:] and abs(merged[-1][1] - window[0]) < 1e-6:
                merged[-1] = (merged[-1][0], window[1], *window[2:])
            else:
                merged.append(window)
        for begin, end, option, state in merged:
            css = frames.between(begin, end) if len(merged) > 1 else ""
            colour = {"ok": page["ok"], "no": page["no"]}.get(state, page["muted"])
            if state:
                fill = page["ok_bg"] if state == "ok" else page["no_bg"]
                parts.append(
                    f'<rect class="{css}" x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="14" '
                    f'fill="{fill}" stroke="{colour}" stroke-width="3"/>'
                    f'<text class="{css}" x="{centre}" y="{top + 36}" text-anchor="middle" '
                    f'font-family="{SANS}" font-size="18" font-weight="700" fill="{page["ink"]}">'
                    f"{html.escape(stage['title'])}</text>"
                )
            mark = {"ok": "✓ ", "no": "✗ "}.get(state, "")
            rows = _two_lines(mark + option)
            for row, text in enumerate(rows):
                y = top + (72 if len(rows) == 1 else 62 + 22 * row)
                parts.append(
                    f'<text class="{css}" x="{centre}" y="{y}" text-anchor="middle" '
                    f'font-family="{MONO}" font-size="{19 if len(rows) == 1 else 17}" '
                    f'font-weight="{700 if state else 400}" fill="{colour}">{html.escape(text)}</text>'
                )

    # ---- the terminal
    t_width = width - 2 * t_left
    parts += [
        f'<rect x="{t_left}" y="{t_top}" width="{t_width}" height="{t_height}" rx="12" fill="{page["term"]}" '
        f'stroke="{page["line"]}" stroke-width="{2 if dark else 0}"/>',
        f'<path d="M{t_left} {t_top + 36} v-24 a12 12 0 0 1 12 -12 h{t_width - 24} a12 12 0 0 1 12 12 v24 z" '
        f'fill="{page["bar"]}"/>',
        f'<circle cx="{t_left + 22}" cy="{t_top + 18}" r="6" fill="#ff5f57"/>'
        f'<circle cx="{t_left + 42}" cy="{t_top + 18}" r="6" fill="#febc2e"/>'
        f'<circle cx="{t_left + 62}" cy="{t_top + 18}" r="6" fill="#28c840"/>',
        f'<text x="{width / 2}" y="{t_top + 24}" text-anchor="middle" font-family="{SANS}" font-size="17" '
        f'fill="#8b949e">{html.escape(CAPTION)}</text>',
        f'<g font-family="{MONO}" font-size="{size}" xml:space="preserve">',
    ]
    for name, scene in SCENES.items():
        begin, end = starts[name], starts[name] + scene["seconds"]
        typed, groups = drawn[name]
        if 1.4 + STEP * (len(groups) - 1) > scene["seconds"] - HOLD:
            raise ValueError(f"{name}: too many lines for its {scene['seconds']} seconds")
        y = t_top + 36 + pad + size - 4
        for index, text in enumerate(typed):
            css = frames.between(begin + 0.2 + 0.4 * index, end)
            prompt = f'<tspan fill="{term["ok"]}">$\u00a0</tspan>' if index == 0 else "\u00a0" * 4
            parts.append(
                f'<text class="{css}" x="{t_left + pad}" y="{y}">{prompt}<tspan fill="{term["text"]}" '
                f'font-weight="700">{_text(text)}</tspan></text>'
            )
            y += leading
        y += 8
        for index, group in enumerate(groups):
            css = frames.between(begin + 1.4 + STEP * index, end)
            last = index == len(groups) - 1
            for text in group:
                parts.append(
                    f'<text class="{css}" x="{t_left + pad}" y="{y}" fill="{_colour(group[0], term)}"'
                    f"{' font-weight=' + chr(34) + '700' + chr(34) if last else ''}>{_text(text)}</text>"
                )
                y += leading
    parts.append("</g></svg>\n")
    return "\n".join(parts).replace("/*KEYFRAMES*/", frames.css())


def _typed(command: str) -> list[str]:
    return [command] if len(command) + 2 <= COLUMNS else _break(command)


def _break(command: str) -> list[str]:
    """A command that is too long for one line, broken before an option with a backslash."""
    units: list[str] = []
    for word in command.split(" "):
        value = units and units[-1].startswith("-") and " " not in units[-1]
        if value and not word.startswith("-"):
            units[-1] += " " + word  # an option stays with its value
        else:
            units.append(word)
    lines, current = [], ""
    for unit in units:
        # the prompt or the indent in front, the backslash behind
        if current and len(current) + len(unit) + 1 > COLUMNS - 6:
            lines.append(current + " \\")
            current = unit
        else:
            current = f"{current} {unit}".strip()
    return [*lines, current]


def expected() -> dict[Path, str]:
    """Every picture the tool writes: light and dark, for the README and for the docs."""
    light, dark = svg(dark=False), svg(dark=True)
    return {
        folder / f"showcase-{theme}.svg": content
        for folder in (OUT, DOCS)
        for theme, content in (("light", light), ("dark", dark))
    }


def stale() -> list[Path]:
    return [
        path
        for path, content in expected().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 when stale")
    parser.add_argument("--capture", action="store_true", help="run the commands again first")
    args = parser.parse_args(argv)
    if args.capture:
        found = capture()
        TRANSCRIPT.write_text(
            json.dumps(found, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {TRANSCRIPT.relative_to(REPO)}")
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"stale: {path.relative_to(REPO)} (run python -m tools.make_showcase)")
        return 1 if outdated else 0
    for path, content in expected().items():
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)} ({len(content.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
