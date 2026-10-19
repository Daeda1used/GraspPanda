"""Regenerate the brand assets in ``resources/`` and the pictures the docs site uses.

``resources/logo.svg`` is the hand-drawn source. Everything else is derived from it or
generated: the dark-mode logo, the banners, the pipeline figure (light and dark), the
hero picture of the docs home page and the PNG exports.

The pipeline figure is computed, not drawn by hand: its two lanes, the methods in each and
the slots of each lane come from :mod:`tools.docs_chips` (the method metafiles and the
slots each method has), so the picture changes when a method or a slot does.

    python -m tools.make_brand_assets            # needs: pip install cairosvg
    python -m tools.make_brand_assets --check    # exit 1 when a committed SVG is stale
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "resources"
DOCS_ASSETS = REPO / "docs" / "assets"
HERO = REPO / "overrides" / "partials" / "gp-hero-pipeline.html"

INK, PAPER, GREEN, DEEP_GREEN = "#1f2328", "#ffffff", "#2da44e", "#1a7f37"
SANS = "'Inter', 'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'DejaVu Sans Mono', 'SFMono-Regular', Consolas, monospace"
TAGLINE = "A toolbox for 6-DoF grasp detection in cluttered scenes"
PILLARS = "Run published methods  ·  Swap any stage  ·  See what was measured"
DATASETS = ("GraspNet-1Billion", "GraspClutter6D")


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
  <text x="378" y="210" font-family="{SANS}" font-size="28" fill="{text}">{TAGLINE}</text>
  <text x="378" y="256" font-family="{SANS}" font-size="21" fill="{muted}">{PILLARS}</text>
</svg>
"""


# ------------------------------------------------------------------- what is drawn


@dataclass(frozen=True)
class Box:
    """One box of the pipeline figure.

    Attributes:
        lines: The label, one or two lines.
        key: The slot as ``--set`` spells it; ``""`` for a fixed part of a method.
    """

    lines: tuple[str, ...]
    key: str = ""


#: Slot → its label in the figure (two short lines where the page title is long).
LABELS = {
    "backbone": ("Scene", "encoder"),
    "proposal": ("Seed", "proposal"),
    "view": ("View",),
    "assigner": ("Assigner",),
    "grouper": ("Grouper",),
    "head": ("Head", "bundle"),
    "loss": ("Losses",),
    "image_encoder": ("Image", "encoder"),
    "post": ("Post-processing",),
}
#: The parts of HGGD that are fixed (not slots), in the order data flows.
HGGD_FIXED = (Box(("Heat-map", "stage")), Box(("Region", "stage")))
#: The slots of the training recipe, as one band.
RECIPE = ("optimizer", "schedule", "hooks", "augment")


@dataclass(frozen=True)
class Lanes:
    """What the figure shows, computed from the methods and their slots."""

    seed_methods: tuple[str, ...]
    own_methods: tuple[str, ...]
    seed_boxes: tuple[Box, ...]  # the stages, in data order; the losses are drawn apart
    losses: Box | None
    own_boxes: tuple[Box, ...]
    shared: tuple[str, ...]  # slots every method has


def lanes() -> Lanes:
    """The lanes of the figure: the seed-based skeleton, HGGD, and what all six share."""
    sys.path.insert(0, str(REPO))
    from tools import docs_chips

    infos = docs_chips.methods()
    seed = tuple(info.name for info in infos if info.family == "seed-based")
    own = tuple(info.name for info in infos if info.family != "seed-based")
    by_scope: dict[str, list[str]] = {}
    for slot in docs_chips.SLOT_ORDER:
        if slot == "recipe" or not docs_chips.slot_methods(slot):
            continue
        by_scope.setdefault(docs_chips.scope(slot), []).append(slot)
    stages = [slot for slot in by_scope.get("seed-based", []) if slot != "loss"]
    return Lanes(
        seed_methods=seed,
        own_methods=own,
        seed_boxes=tuple(Box(LABELS[slot], slot) for slot in stages),
        losses=Box(LABELS["loss"], "loss") if "loss" in by_scope.get("seed-based", []) else None,
        own_boxes=tuple(Box(LABELS[slot], slot) for slot in by_scope.get("hggd", [])) + HGGD_FIXED,
        shared=tuple(by_scope.get("all", [])),
    )


# -------------------------------------------------------------------------- pipeline

LIGHT = {
    "card": "#ffffff",
    "card_line": "#d0d7de",
    "lane": "#f6f8fa",
    "lane_line": "#d0d7de",
    "ink": INK,
    "muted": "#57606a",
    "box": "#ffffff",
    "box_line": INK,
    "key": DEEP_GREEN,
    "shared": "#dafbe1",
    "shared_line": GREEN,
    "fixed": "#f6f8fa",
    "fixed_line": "#8c959f",
    "arrow": "#57606a",
    "pill": "#ffffff",
}
DARK = {
    "card": "#0d1117",
    "card_line": "#3d444d",
    "lane": "#151b23",
    "lane_line": "#3d444d",
    "ink": "#e6edf3",
    "muted": "#9198a1",
    "box": "#0d1117",
    "box_line": "#c9d1d9",
    "key": "#3fb950",
    "shared": "#12261e",
    "shared_line": "#3fb950",
    "fixed": "#151b23",
    "fixed_line": "#656c76",
    "arrow": "#9198a1",
    "pill": "#0d1117",
}


def _text(x: float, y: float, words: str, size: float, fill: str, **extra: str) -> str:
    attributes = "".join(f' {name.replace("_", "-")}="{value}"' for name, value in extra.items())
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="{SANS}" font-size="{size:g}" '
        f'fill="{fill}"{attributes}>{html.escape(words)}</text>'
    )


def _key(x: float, y: float, words: str, size: float, fill: str) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="{size:g}" fill="{fill}">{html.escape(words)}</text>'
    )


def _box(
    parts: list[str],
    c: dict[str, str],
    box: Box,
    x: float,
    y: float,
    w: float,
    h: float,
    shared: bool = False,
) -> None:
    """A slot (its key in green), a shared slot (green) or a fixed part (grey, dashed)."""
    if shared:
        fill, line, dash = c["shared"], c["shared_line"], ""
    elif box.key:
        fill, line, dash = c["box"], c["box_line"], ""
    else:
        fill, line, dash = c["fixed"], c["fixed_line"], ' stroke-dasharray="7 5"'
    parts.append(
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="12" fill="{fill}" '
        f'stroke="{line}" stroke-width="2.5"{dash}/>'
    )
    centre, lines = x + w / 2, len(box.lines)
    if box.key:  # the label above, the key at the bottom
        first = y + 27 if h < 70 else y + (34 if lines == 2 else 44)
    else:  # the label in the middle
        first = y + h / 2 + 7 - (lines - 1) * 11.5
    for index, words in enumerate(box.lines):
        parts.append(
            _text(
                centre,
                first + index * 23,
                words,
                19 if box.key else 17,
                c["ink"] if box.key else c["muted"],
                text_anchor="middle",
                font_weight="700",
            )
        )
    if box.key:
        parts.append(_key(centre, y + h - (12 if h < 70 else 16), box.key, 13, c["key"]))


def _arrow(parts: list[str], c: dict[str, str], points: list[tuple[float, float]]) -> None:
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:g} {y:g}" for i, (x, y) in enumerate(points))
    parts.append(
        f'<path d="{path}" fill="none" stroke="{c["arrow"]}" stroke-width="2.5" '
        'stroke-linejoin="round" marker-end="url(#a)"/>'
    )


def pipeline_svg(dark: bool = False) -> str:
    """The two lanes (seed-based skeleton, HGGD), the parts all six share, the datasets."""
    c = DARK if dark else LIGHT
    found = lanes()
    width = 1280
    lane_x, lane_w = 228, 802
    box_w, box_h, gap = 113, 100, 18
    inner = lane_x + 16
    post_x, post_w = 1058, 198
    seed_y, seed_h = 118, 256
    own_y, own_h = 394, 190
    recipe_y, recipe_h = 604, 64
    legend_y = 712
    height = 744
    seed_box_y, own_box_y = seed_y + 72, own_y + 72
    seed_mid, own_mid = seed_box_y + box_h / 2, own_box_y + box_h / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{html.escape(alt_text(found), quote=True)}">',
        f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
        f'fill="{c["arrow"]}"/></marker></defs>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="16" fill="{c["card"]}" '
        f'stroke="{c["card_line"]}" stroke-width="2"/>',
        _text(32, 56, "Two detector families, one toolbox", 28, c["ink"], font_weight="700"),
        _text(
            32,
            90,
            "A box with a green name is a slot: swap it with one line, "
            "--set <slot>=<option>. Grey boxes are fixed parts of a method.",
            18,
            c["muted"],
        ),
    ]

    # the datasets, feeding both lanes
    pill_x, pill_w, pill_h = 24, 180, 44
    mid = (seed_mid + own_mid) / 2
    parts.append(
        _text(
            pill_x + pill_w / 2,
            mid - 62,
            "Datasets",
            16,
            c["muted"],
            text_anchor="middle",
            font_weight="700",
        )
    )
    for index, name in enumerate(DATASETS):
        y = mid - 46 + index * 56
        parts.append(
            f'<rect x="{pill_x}" y="{y:g}" width="{pill_w}" height="{pill_h}" rx="22" '
            f'fill="{c["pill"]}" stroke="{c["card_line"]}" stroke-width="2"/>'
        )
        parts.append(
            _text(
                pill_x + pill_w / 2,
                y + 28,
                name,
                16,
                c["ink"],
                text_anchor="middle",
                font_weight="700",
            )
        )
    fork = pill_x + pill_w + 12
    _arrow(parts, c, [(pill_x + pill_w, mid), (fork, mid), (fork, seed_mid), (inner, seed_mid)])
    _arrow(parts, c, [(fork, mid), (fork, own_mid), (inner, own_mid)])

    # lane 1: the seed-based skeleton
    parts.append(
        f'<rect x="{lane_x}" y="{seed_y}" width="{lane_w}" height="{seed_h}" rx="14" '
        f'fill="{c["lane"]}" stroke="{c["lane_line"]}" stroke-width="2"/>'
    )
    parts.append(
        _text(
            inner,
            seed_y + 32,
            f"Seed-based skeleton · {len(found.seed_methods)} methods",
            20,
            c["ink"],
            font_weight="700",
        )
    )
    parts.append(_text(inner, seed_y + 56, " · ".join(found.seed_methods), 15, c["muted"]))
    for index, box in enumerate(found.seed_boxes):
        x = inner + index * (box_w + gap)
        _box(parts, c, box, x, seed_box_y, box_w, box_h)
        if index:
            _arrow(parts, c, [(x - gap, seed_mid), (x, seed_mid)])
    seed_end = inner + len(found.seed_boxes) * (box_w + gap) - gap
    if found.losses is not None:
        x = seed_end - box_w
        y = seed_box_y + box_h + 16
        _box(parts, c, found.losses, x, y, box_w, 60)
        parts.append(
            f'<line x1="{x + box_w / 2:g}" y1="{seed_box_y + box_h:g}" x2="{x + box_w / 2:g}" '
            f'y2="{y:g}" stroke="{c["arrow"]}" stroke-width="2.5"/>'
        )

    # lane 2: HGGD, a family of its own
    parts.append(
        f'<rect x="{lane_x}" y="{own_y}" width="{lane_w}" height="{own_h}" rx="14" '
        f'fill="{c["lane"]}" stroke="{c["lane_line"]}" stroke-width="2"/>'
    )
    own = " · ".join(found.own_methods)
    parts.append(_text(inner, own_y + 32, f"{own} · own family", 20, c["ink"], font_weight="700"))
    parts.append(_text(inner, own_y + 56, "reads a colour image and its depth", 15, c["muted"]))
    for index, box in enumerate(found.own_boxes):
        x = inner + index * (box_w + gap)
        _box(parts, c, box, x, own_box_y, box_w, box_h)
        if index:
            _arrow(parts, c, [(x - gap, own_mid), (x, own_mid)])
    own_end = inner + len(found.own_boxes) * (box_w + gap) - gap

    # what all six share: post-processing, then the grasps; the recipe under both lanes
    post_h = 108
    post_y = mid - post_h / 2
    turn = post_x - 14
    _arrow(parts, c, [(seed_end, seed_mid), (turn, seed_mid), (turn, mid - 20), (post_x, mid - 20)])
    _arrow(parts, c, [(own_end, own_mid), (turn, own_mid), (turn, mid + 20), (post_x, mid + 20)])
    _box(parts, c, Box(LABELS["post"], "post"), post_x, post_y, post_w, post_h, shared=True)
    parts.append(
        _text(
            post_x + post_w / 2,
            post_y + 66,
            f"all {len(found.seed_methods) + len(found.own_methods)} methods",
            14,
            c["muted"],
            text_anchor="middle",
        )
    )
    out_y = post_y + post_h + 64
    _arrow(parts, c, [(post_x + post_w / 2, post_y + post_h), (post_x + post_w / 2, out_y)])
    parts.append(
        f'<rect x="{post_x + 14}" y="{out_y:g}" width="{post_w - 28}" height="44" rx="22" '
        f'fill="{c["pill"]}" stroke="{c["card_line"]}" stroke-width="2"/>'
    )
    parts.append(
        _text(
            post_x + post_w / 2,
            out_y + 28,
            "6-DoF grasps",
            17,
            c["ink"],
            text_anchor="middle",
            font_weight="700",
        )
    )

    every = len(found.seed_methods) + len(found.own_methods)
    recipe = [slot for slot in RECIPE if slot in found.shared]
    parts.append(
        f'<rect x="{lane_x}" y="{recipe_y}" width="{post_x + post_w - lane_x}" '
        f'height="{recipe_h}" rx="14" fill="{c["shared"]}" stroke="{c["shared_line"]}" '
        'stroke-width="2.5"/>'
    )
    parts.append(_text(inner, recipe_y + 40, "Training recipe", 19, c["ink"], font_weight="700"))
    parts.append(_text(inner + 172, recipe_y + 40, f"all {every} methods", 15, c["muted"]))
    parts.append(
        f'<text x="{post_x + post_w - 20}" y="{recipe_y + 40}" text-anchor="end" '
        f'font-family="{MONO}" font-size="15" fill="{c["key"]}">'
        f"{html.escape(' · '.join(recipe))}</text>"
    )

    # legend
    entries = [
        ("slot", c["box"], c["box_line"], ""),
        (f"shared by all {every} methods", c["shared"], c["shared_line"], ""),
        ("fixed part of a method", c["fixed"], c["fixed_line"], ' stroke-dasharray="5 4"'),
    ]
    x = lane_x
    for label, fill, line, dash in entries:
        parts.append(
            f'<rect x="{x}" y="{legend_y - 15}" width="26" height="18" rx="5" fill="{fill}" '
            f'stroke="{line}" stroke-width="2"{dash}/>'
        )
        parts.append(_text(x + 36, legend_y, label, 15, c["muted"]))
        x += 60 + len(label) * 8
    parts.append("</svg>\n")
    return "\n".join(parts)


def alt_text(found: Lanes | None = None) -> str:
    """What the pipeline figure shows, in words (its aria-label and the alt text of pages)."""
    found = found or lanes()
    seed = ", ".join(" ".join(box.lines).lower() for box in found.seed_boxes)
    own = ", ".join(" ".join(box.lines).lower() for box in found.own_boxes)
    every = len(found.seed_methods) + len(found.own_methods)
    return (
        f"GraspNet-1Billion and GraspClutter6D feed two detector families. The seed-based "
        f"skeleton of {len(found.seed_methods)} methods has the slots {seed}, and losses. "
        f"HGGD has its own family: {own}. Post-processing and the training recipe are shared "
        f"by all {every} methods."
    )


# ------------------------------------------------------------------------ home hero


#: The option each lane of the hero swaps in, as a cross-fade: slot → the label drawn.
HERO_SWAPS = {"backbone": "ptv3", "image_encoder": "dinov2"}


def hero_html() -> str:
    """The picture on the right of the docs home hero, in the theme's colours (CSS variables
    of ``docs/stylesheets/extra.css``): the datasets on a rail into both lanes (the
    seed-based skeleton with its losses, HGGD), a rail from both lanes into the band all
    methods share, and one swap per lane shown as a cross-fade into an option drawn in the
    "swapped option" style (still when the reader prefers reduced motion)."""
    found = lanes()
    every = len(found.seed_methods) + len(found.own_methods)
    width, height = 560, 400
    box_w, box_h, gap = 76, 52, 9
    rail_left, rail_right = 12, 548  # the rails run outside the lanes
    lane_x, lane_w = 22, 516
    inner = lane_x + 6  # the first box of a lane
    seed_y, seed_h = 56, 130
    own_y, own_h = 198, 90
    shared_y, shared_h = 300, 88
    seed_top, own_top = seed_y + 28, own_y + 28
    seed_mid, own_mid = seed_top + box_h / 2, own_top + box_h / 2
    parts = [
        "{#- Generated by tools/make_brand_assets.py (hero_html). Do not edit by hand. -#}",
        f'<svg class="gp-hero-figure" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(alt_text(found), quote=True)}">',
        "<style>",
        ".gp-hf-card{fill:var(--md-default-bg-color);stroke:var(--gp-line)}",
        ".gp-hf-lane{fill:var(--gp-surface);stroke:var(--gp-line)}",
        ".gp-hf-box{fill:var(--md-default-bg-color);stroke:var(--gp-ink);stroke-width:1.6}",
        ".gp-hf-fixed{fill:var(--gp-surface);stroke:var(--gp-muted);stroke-width:1.4;"
        "stroke-dasharray:5 4}",
        ".gp-hf-shared{fill:var(--gp-ok-bg);stroke:var(--gp-ok);stroke-width:1.6}",
        ".gp-hf-swapped{fill:var(--md-default-bg-color);stroke:var(--md-typeset-a-color);"
        "stroke-width:2}",
        ".gp-hf-pill{fill:var(--md-default-bg-color);stroke:var(--gp-line);stroke-width:1.4}",
        ".gp-hf-t{fill:var(--gp-ink);font:600 12px var(--md-text-font-family,sans-serif)}",
        ".gp-hf-m{fill:var(--gp-muted);font:400 11.5px var(--md-text-font-family,sans-serif)}",
        ".gp-hf-h{fill:var(--gp-ink);font:700 13px var(--md-text-font-family,sans-serif)}",
        ".gp-hf-k{fill:var(--gp-ok);font:600 11.5px var(--md-code-font-family,monospace)}",
        ".gp-hf-s{fill:var(--md-typeset-a-color);font:600 12px var(--md-code-font-family,monospace)}",
        ".gp-hf-a{stroke:var(--gp-muted);stroke-width:1.4;fill:none}",
        ".gp-hf-swap{opacity:0;animation:gp-hf-swap 7s ease-in-out infinite both}",
        ".gp-hf-orig{animation:gp-hf-orig 7s ease-in-out infinite both}",
        ".gp-hf-late{animation-delay:3.5s}",
        "@keyframes gp-hf-swap{0%,35%{opacity:0}45%,85%{opacity:1}95%,100%{opacity:0}}",
        "@keyframes gp-hf-orig{0%,35%{opacity:1}45%,85%{opacity:0}95%,100%{opacity:1}}",
        "@media (prefers-reduced-motion:reduce){.gp-hf-swap{animation:none;opacity:0}"
        ".gp-hf-orig{animation:none;opacity:1}}",
        "</style>",
        f'<rect class="gp-hf-card" x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14"/>',
    ]

    # the datasets, joined on a rail that runs down the left into both lanes
    pills = []
    x = inner
    for name in DATASETS:
        pill_w = 22 + 7 * len(name)
        parts.append(
            f'<rect class="gp-hf-pill" x="{x}" y="12" width="{pill_w}" height="24" rx="12"/>'
        )
        parts.append(
            f'<text class="gp-hf-t" x="{x + pill_w / 2:g}" y="28" text-anchor="middle">{name}</text>'
        )
        pills.append(x + pill_w / 2)
        x += pill_w + 10
    for centre in pills:
        parts.append(f'<path class="gp-hf-a" d="M{centre:g} 36 V44"/>')
    parts.append(
        f'<path class="gp-hf-a" d="M{pills[-1]:g} 44 H{rail_left} V{own_mid:g}"/>'
        f'<path class="gp-hf-a" d="M{rail_left} {seed_mid:g} H{inner}"/>'
        f'<path class="gp-hf-a" d="M{rail_left} {own_mid:g} H{inner}"/>'
    )

    # the legend, top right
    legend_x = 330
    for index, (kind, words) in enumerate(
        (
            ("gp-hf-box", "a slot: swap it with one line"),
            ("gp-hf-fixed", "a fixed part of the method"),
            ("gp-hf-swapped", "a swapped option"),
        )
    ):
        y = 10 + index * 14
        parts.append(f'<rect class="{kind}" x="{legend_x}" y="{y}" width="18" height="10" rx="3"/>')
        parts.append(f'<text class="gp-hf-m" x="{legend_x + 26}" y="{y + 9}">{words}</text>')

    def lane(y: float, h: float, title: str) -> None:
        parts.append(
            f'<rect class="gp-hf-lane" x="{lane_x}" y="{y}" width="{lane_w}" height="{h}" rx="11"/>'
        )
        parts.append(
            f'<text class="gp-hf-h" x="{inner + 2}" y="{y + 19}">{html.escape(title)}</text>'
        )

    def box(item: Box, x: float, top: float, h: float, late: bool = False) -> None:
        kind = "gp-hf-box" if item.key else "gp-hf-fixed"
        parts.append(
            f'<rect class="{kind}" x="{x}" y="{top}" width="{box_w}" height="{h}" rx="8"/>'
        )
        centre = x + box_w / 2
        label = "".join(
            f'<tspan x="{centre:g}" dy="{0 if i == 0 else 14}">{html.escape(words)}</tspan>'
            for i, words in enumerate(item.lines)
        )
        first = top + h / 2 + 4 - (len(item.lines) - 1) * 7
        text = (
            f'<text class="{"gp-hf-t" if item.key else "gp-hf-m"}" x="{centre:g}" '
            f'y="{first:g}" text-anchor="middle">{label}</text>'
        )
        swap = HERO_SWAPS.get(item.key)
        if swap is None:
            parts.append(text)
            return
        delay = " gp-hf-late" if late else ""
        parts.append(f'<g class="gp-hf-orig{delay}">{text}</g>')
        parts.append(
            f'<g class="gp-hf-swap{delay}"><rect class="gp-hf-swapped" x="{x}" y="{top}" '
            f'width="{box_w}" height="{h}" rx="8"/><text class="gp-hf-s" x="{centre:g}" '
            f'y="{top + h / 2 + 4:g}" text-anchor="middle">{html.escape(swap)}</text></g>'
        )

    def row(items: tuple[Box, ...], top: float, late: bool = False) -> float:
        """Boxes left to right with a connector between each; returns the right edge."""
        for index, item in enumerate(items):
            x = inner + index * (box_w + gap)
            box(item, x, top, box_h, late)
            if index:
                parts.append(f'<path class="gp-hf-a" d="M{x - gap} {top + box_h / 2:g} H{x}"/>')
        return inner + len(items) * (box_w + gap) - gap

    # lane 1: the seed-based skeleton; its losses hang under the head bundle
    lane(seed_y, seed_h, f"Seed-based skeleton · {len(found.seed_methods)} methods")
    seed_end = row(found.seed_boxes, seed_top)
    if found.losses is not None:
        x = seed_end - box_w
        parts.append(
            f'<path class="gp-hf-a" d="M{x + box_w / 2:g} {seed_top + box_h} V{seed_top + box_h + 10}"/>'
        )
        box(found.losses, x, seed_top + box_h + 10, 26)

    # lane 2: HGGD, a family of its own
    lane(own_y, own_h, " · ".join(found.own_methods) + " · own family")
    own_end = row(found.own_boxes, own_top, late=True)

    # what every method shares; both lanes run into its post-processing down the right rail
    lane(shared_y, shared_h, f"Shared by all {every} methods")
    half = (lane_w - 12 - gap) / 2
    top = shared_y + 28
    post_x = inner + half + gap
    for x, title, keys in (
        (inner, "Training recipe", "optimizer · schedule · …"),
        (post_x, "Post-processing", "post"),
    ):
        parts.append(
            f'<rect class="gp-hf-shared" x="{x:g}" y="{top}" width="{half:g}" height="48" rx="9"/>'
        )
        parts.append(
            f'<text class="gp-hf-t" x="{x + half / 2:g}" y="{top + 20}" text-anchor="middle">{title}</text>'
        )
        parts.append(
            f'<text class="gp-hf-k" x="{x + half / 2:g}" y="{top + 37}" text-anchor="middle">{html.escape(keys)}</text>'
        )
    post_mid = top + 24
    parts.append(
        f'<path class="gp-hf-a" d="M{seed_end} {seed_mid:g} H{rail_right} V{post_mid} '
        f'H{post_x + half:g}"/>'
        f'<path class="gp-hf-a" d="M{own_end} {own_mid:g} H{rail_right}"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------- main


def svgs() -> dict[Path, str]:
    """Every generated text file and its content."""
    return {
        OUT / "logo-dark.svg": logo_svg(dark=True),
        OUT / "banner-light.svg": banner_svg(dark=False),
        OUT / "banner-dark.svg": banner_svg(dark=True),
        OUT / "pipeline.svg": pipeline_svg(dark=False),
        OUT / "pipeline-dark.svg": pipeline_svg(dark=True),
        HERO: hero_html(),
    }


def stale() -> list[Path]:
    """The generated text files whose committed content differs from what is generated."""
    return [
        path
        for path, content in svgs().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when an SVG is stale")
    args = parser.parse_args(argv)
    if args.check:
        found = stale()
        for path in found:
            print(f"stale: {path.relative_to(REPO)}")
        return 1 if found else 0

    import cairosvg

    for path, content in svgs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    exports = {
        "logo.svg": 512,
        "logo-dark.svg": 512,
        "banner-light.svg": 2560,
        "banner-dark.svg": 2560,
        "pipeline.svg": 2560,
        "pipeline-dark.svg": 2560,
    }
    for name, width in exports.items():
        target = OUT / name.replace(".svg", ".png")
        cairosvg.svg2png(url=str(OUT / name), write_to=str(target), output_width=width)
        print(f"wrote {target.relative_to(REPO)}")
    for name in ("logo.svg", "pipeline.png", "pipeline-dark.png"):  # used by the docs site
        shutil.copyfile(OUT / name, DOCS_ASSETS / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
