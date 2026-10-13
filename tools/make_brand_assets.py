"""Regenerate the brand assets in ``resources/``.

``resources/logo.svg`` is the hand-drawn source. Everything else here is derived from
it or generated: the dark-mode logo, the banners, the pipeline diagram, the terminal
demo (captured from the real CLI) and the PNG exports the README uses.

    python tools/make_brand_assets.py          # needs: pip install cairosvg
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "resources"

INK, PAPER, GREEN, DEEP_GREEN = "#1f2328", "#ffffff", "#2da44e", "#1a7f37"
SANS = "'Inter', 'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'DejaVu Sans Mono', 'SFMono-Regular', Consolas, monospace"
TAGLINE = "The toolbox for 6-DoF grasp detection in clutter"
PILLARS = "Swappable components  ·  Faithful ports  ·  One pinned evaluation"


# ------------------------------------------------------------------------------ logo


def logo_body(dark: bool) -> str:
    """Inner shapes of the logo on a 512 × 512 canvas."""
    source = (OUT / "logo.svg").read_text(encoding="utf-8")
    body = source[source.index("</title>") + len("</title>") : source.rindex("</svg>")]
    if dark:  # dark shapes need a light rim to stay visible; the head needs no outline
        body = body.replace(
            'r="68" fill="#1f2328"', 'r="64" fill="#1f2328" stroke="#8b949e" stroke-width="8"'
        )
        body = body.replace('stroke="#1f2328" stroke-width="16"/>', "/>", 1)
    return body


def logo_svg(dark: bool) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" '
        'role="img" aria-label="GraspPanda logo">' + logo_body(dark) + "</svg>\n"
    )


# ---------------------------------------------------------------------------- banner


def banner_svg(dark: bool) -> str:
    text, muted = ("#e6edf3", "#8b949e") if dark else (INK, "#57606a")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 320" width="1280" height="320" role="img" aria-label="GraspPanda: {TAGLINE}">
  <g transform="translate(70 28) scale(0.515)">{logo_body(dark)}</g>
  <text x="372" y="158" font-family="{SANS}" font-size="104" font-weight="700" letter-spacing="-2">
    <tspan fill="{GREEN}">Grasp</tspan><tspan fill="{text}">Panda</tspan>
  </text>
  <text x="378" y="212" font-family="{SANS}" font-size="31" fill="{text}">{TAGLINE}</text>
  <text x="378" y="258" font-family="{SANS}" font-size="21" fill="{muted}">{PILLARS}</text>
</svg>
"""


# -------------------------------------------------------------------------- pipeline

SLOTS = [
    ("Backbone", "model.backbone", "PointFeatures"),
    ("Proposal", "model.proposal", "SeedSet"),
    ("View", "model.view", "ApproachSet"),
    ("Grouper", "model.grouper", "GroupFeatures"),
    ("Head bundle", "model.head", "GraspGroup"),
]
NOTES = [
    (
        "Head bundle = head + target encoder + loss + decoder.",
        "One grasp parameterization; swapped as a unit.",
    ),
    (
        "Observation ≠ Supervision.",
        "predict() never sees ground truth; GT-derived inputs are declared and stamped on results.",
    ),
    (
        "GraspGroup → post-processors → evaluator.",
        "Collision filter on and off are always reported side by side.",
    ),
    (
        "Not every method is this pipeline.",
        "Image-to-region (HGGD) and dense-contact detectors keep only the outer contract.",
    ),
]


def pipeline_svg() -> str:
    width, height = 1280, 470
    box_w, box_h, gap, top, start = 180, 112, 34, 118, 176
    middle = top + box_h / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="GraspPanda pipeline slots and the data contracts between them">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#57606a"/></marker></defs>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="16" fill="#ffffff" stroke="#d0d7de" stroke-width="2"/>',
        f'<text x="36" y="52" font-family="{SANS}" font-size="26" font-weight="700" fill="{INK}">One pipeline, typed contracts between slots</text>',
        f'<text x="36" y="84" font-family="{SANS}" font-size="18" fill="#57606a">Each box is a config slot and names what it emits. A swap is accepted only when the contracts still match.</text>',
        f'<rect x="24" y="{middle - 28}" width="124" height="56" rx="28" fill="#f6f8fa" stroke="#d0d7de" stroke-width="2"/>',
        f'<text x="86" y="{middle + 6}" text-anchor="middle" font-family="{SANS}" font-size="17" font-weight="700" fill="{INK}">Observation</text>',
    ]

    def arrow(x1: float, x2: float) -> None:
        parts.append(
            f'<line x1="{x1}" y1="{middle}" x2="{x2 - 3}" y2="{middle}" stroke="#57606a" stroke-width="2.5" marker-end="url(#a)"/>'
        )

    arrow(148, start)
    for index, (name, key, emits) in enumerate(SLOTS):
        x = start + index * (box_w + gap)
        last = index == len(SLOTS) - 1
        fill, stroke = ("#dafbe1", GREEN) if last else ("#ffffff", INK)
        centre = x + box_w / 2
        parts.append(
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
            f'<text x="{centre}" y="{top + 38}" text-anchor="middle" font-family="{SANS}" font-size="22" font-weight="700" fill="{INK}">{name}</text>'
            f'<text x="{centre}" y="{top + 65}" text-anchor="middle" font-family="{MONO}" font-size="15" fill="{DEEP_GREEN}">{key}</text>'
            f'<text x="{centre}" y="{top + 93}" text-anchor="middle" font-family="{MONO}" font-size="14" fill="#57606a">→ {emits}</text>'
        )
        if not last:
            arrow(x + box_w, x + box_w + gap)
    for index, (title, body) in enumerate(NOTES):
        y = 284 + index * 42
        parts.append(
            f'<circle cx="44" cy="{y - 6}" r="5" fill="{GREEN}"/>'
            f'<text x="60" y="{y}" font-family="{SANS}" font-size="18" fill="{INK}"><tspan font-weight="700">{html.escape(title)}</tspan> <tspan fill="#57606a">{html.escape(body)}</tspan></text>'
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


# -------------------------------------------------------------------- terminal demo

ANSI = re.compile(r"\x1b\[([0-9;]*)m")
COLORS = {"31": "#ff7b72", "32": "#3fb950", "33": "#d29922", "36": "#79c0ff", "90": "#8b949e"}
COMMANDS = [
    ["print-config", "configs/toy_detector.yaml", "--check"],
    ["print-config", "configs/toy_detector.yaml", "--opts", "model.head.num_angles=6"],
    ["print-config", "configs/toy_detector.yaml", "--opts", "model.grouper.type=ToyDepthGrouper"],
]


def run_cli(args: list[str]) -> str:
    """Run the real CLI with colour forced on and return its combined output."""
    result = subprocess.run(
        [sys.executable, "-m", "grasppanda.cli.main", *args],
        cwd=REPO / "projects" / "example_project",
        env={**os.environ, "FORCE_COLOR": "1", "NO_COLOR": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def wrap(line: str, columns: int) -> list[str]:
    plain = ANSI.sub("", line)
    if len(plain) <= columns:
        return [line]
    indent = " " * (len(plain) - len(plain.lstrip()))
    wrapped = textwrap.wrap(plain, columns, subsequent_indent=indent, break_on_hyphens=False)
    codes = ANSI.findall(line)
    prefix = f"\x1b[{codes[0]}m" if codes and line.lstrip().startswith("\x1b") else ""
    return [prefix + part + ("\x1b[0m" if prefix else "") for part in wrapped]


def spans(line: str) -> str:
    """Convert one ANSI-coloured line into SVG tspans."""
    out, position, fill, bold = [], 0, "#e6edf3", False
    for match in [*ANSI.finditer(line), None]:
        chunk = line[position : match.start() if match else len(line)]
        if chunk:
            weight = ' font-weight="700"' if bold else ""
            out.append(f'<tspan fill="{fill}"{weight}>{html.escape(chunk)}</tspan>')
        if match is None:
            break
        position = match.end()
        codes = [c for c in match.group(1).split(";") if c]
        if not codes or codes == ["0"]:
            fill, bold = "#e6edf3", False
        for code in codes:
            if code == "1":
                bold = True
            elif code in COLORS:
                fill = COLORS[code]
    return "".join(out)


def terminal_svg() -> str:
    columns, size, leading, pad = 92, 15, 22, 26
    lines: list[str] = []
    for args in COMMANDS:
        lines.append("\x1b[32m$\x1b[0m \x1b[1mgrasppanda " + " ".join(args) + "\x1b[0m")
        for raw in run_cli(args).rstrip("\n").splitlines():
            lines.extend(wrap(raw, columns))
        lines.append("")
    lines.pop()
    width = int(columns * size * 0.602) + 2 * pad
    height = 44 + len(lines) * leading + pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="grasppanda print-config validating a config and rejecting two mistakes">',
        f'<rect width="{width}" height="{height}" rx="12" fill="#0d1117"/>',
        f'<rect width="{width}" height="36" rx="12" fill="#161b22"/><rect y="24" width="{width}" height="12" fill="#161b22"/>',
        '<circle cx="22" cy="18" r="6" fill="#ff5f57"/><circle cx="42" cy="18" r="6" fill="#febc2e"/><circle cx="62" cy="18" r="6" fill="#28c840"/>',
        f'<text x="{width / 2}" y="23" text-anchor="middle" font-family="{SANS}" font-size="13" fill="#8b949e">projects/example_project — strict config validation</text>',
        f'<g font-family="{MONO}" font-size="{size}" xml:space="preserve">',
    ]
    for index, line in enumerate(lines):
        parts.append(f'<text x="{pad}" y="{62 + index * leading}">{spans(line)}</text>')
    parts.append("</g></svg>\n")
    return "\n".join(parts)


# ----------------------------------------------------------------------------- main


def main() -> None:
    import cairosvg

    assets = {
        "logo-dark.svg": logo_svg(dark=True),
        "banner-light.svg": banner_svg(dark=False),
        "banner-dark.svg": banner_svg(dark=True),
        "pipeline.svg": pipeline_svg(),
        "terminal-demo.svg": terminal_svg(),
    }
    for name, content in assets.items():
        (OUT / name).write_text(content, encoding="utf-8")
    exports = {
        "logo.svg": 512,
        "logo-dark.svg": 512,
        "banner-light.svg": 2560,
        "banner-dark.svg": 2560,
        "pipeline.svg": 2560,
        "terminal-demo.svg": 1900,
    }
    for name, width in exports.items():
        target = OUT / name.replace(".svg", ".png")
        cairosvg.svg2png(url=str(OUT / name), write_to=str(target), output_width=width)
        print(f"wrote {target.relative_to(REPO)}")
    for name in ("logo.svg", "pipeline.png", "terminal-demo.png"):  # used by the docs site
        shutil.copyfile(OUT / name, REPO / "docs" / "assets" / name)


if __name__ == "__main__":
    main()
