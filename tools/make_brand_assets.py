"""Regenerate the brand assets in ``resources/`` and the pictures the docs site uses.

``resources/logo.svg`` is the hand-drawn source. Everything else is derived from it or
generated: the dark-mode logo, the banners, the pipeline figure (light and dark), the
hero picture of the docs home page and the PNG exports.

The pipeline figure is computed, not drawn by hand: its two lanes, the methods in each and
the slots of each lane come from :mod:`tools.docs_chips` (the method metafiles and the
slots each method has), so the picture changes when a method or a slot does.

    python -m tools.make_brand_assets            # needs: pip install cairosvg
    python -m tools.make_brand_assets --check    # exit 1 when a committed SVG is stale

The PNG exports are drawn with Inter and JetBrains Mono (the faces of the documentation
site). Install both before exporting (https://rsms.me/inter, https://jetbrains.com/mono):
without them the exporter falls back to another face and refuses to write, so that a
picture never changes its typeface by accident.
"""

from __future__ import annotations

import argparse
import html
import math
import shutil
import subprocess
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
  <text x="372" y="158" font-family="{SANS}" font-size="104" font-weight="700" letter-spacing="-3.5">
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


def centred_dashes(w: float, h: float, rx: float, period: float = 9.0) -> str:
    """Stroke attributes of a dashed rounded box whose dashes are centred on the middle of
    its left and right edges.

    A connector that meets a dashed border at the start of a dash reads as a hook. The
    period is adjusted so that half the outline holds a whole number of dashes, and the
    pattern is shifted so that a dash sits across each of the two points a connector meets.
    """
    outline = 2 * (w - 2 * rx) + 2 * (h - 2 * rx) + 2 * math.pi * rx
    step = outline / 2 / max(1, round(outline / 2 / period))
    dash = step * 5 / 9
    right_middle = (w - 2 * rx) + math.pi * rx / 2 + (h - 2 * rx) / 2  # a rect starts top left
    offset = (dash / 2 - right_middle) % step
    return f'stroke-dasharray="{dash:.3f} {step - dash:.3f}" stroke-dashoffset="{offset:.3f}"'


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


def pipeline_svg(dark: bool = False) -> str:
    """The figure of the README and the FAQ: the datasets on a rail into two lanes (the
    seed-based skeleton with its losses, HGGD), a rail from both lanes into the
    post-processing all methods share, and the training recipe beside it.

    It is the picture of the docs home hero (:func:`hero_html`) with the key of every slot
    written out. The canvas is narrow on purpose: at the width a README shows it, the
    smallest text (the slot keys) is still 11 px.
    """
    c = DARK if dark else LIGHT
    found = lanes()
    every = len(found.seed_methods) + len(found.own_methods)
    width = 680
    rail_left, rail_right = 14, 666  # the rails run outside the lanes
    lane_x, lane_w = 26, 628
    inner = lane_x + 10
    box_w, box_h, gap = 93, 70, 10
    seed_y, seed_h = 122, 160
    own_y, own_h = 294, 114
    shared_y, shared_h = 420, 98
    legend_y = 546
    height = 568
    seed_top, own_top, shared_top = seed_y + 32, own_y + 32, shared_y + 32
    seed_mid, own_mid = seed_top + box_h / 2, own_top + box_h / 2

    def line(d: str, arrow: bool = False) -> str:
        end = ' marker-end="url(#a)"' if arrow else ""
        return (
            f'<path d="{d}" fill="none" stroke="{c["arrow"]}" stroke-width="1.4" '
            f'stroke-linejoin="round"{end}/>'
        )

    def lane(y: float, h: float, title: str, note: str = "") -> str:
        tail = (
            f'<tspan dx="10" font-size="12" font-weight="400" fill="{c["muted"]}">'
            f"{html.escape(note)}</tspan>"
        )
        return (
            f'<rect x="{lane_x}" y="{y}" width="{lane_w}" height="{h}" rx="11" '
            f'fill="{c["lane"]}" stroke="{c["lane_line"]}" stroke-width="1.2"/>'
            f'<text x="{inner + 2}" y="{y + 21}" font-family="{SANS}" font-size="14" '
            f'font-weight="700" fill="{c["ink"]}">{html.escape(title)}{tail if note else ""}</text>'
        )

    def box(item: Box, x: float, y: float, w: float = box_w, h: float = box_h) -> str:
        """A slot (ink outline, its key in green) or a fixed part (grey, dashed)."""
        centre = x + w / 2
        if not item.key:
            first = y + h / 2 + 4.5 - (len(item.lines) - 1) * 8
            label = "".join(
                _text(centre, first + i * 16, words, 13, c["muted"], text_anchor="middle")
                for i, words in enumerate(item.lines)
            )
            return (
                f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="9" '
                f'fill="{c["fixed"]}" stroke="{c["fixed_line"]}" stroke-width="1.4" '
                f"{centred_dashes(w, h, 9)}/>" + label
            )
        compact = h < 60  # the losses: one line and the key
        first = y + 17 if compact else y + (23 if len(item.lines) == 2 else 31)
        label = "".join(
            _text(
                centre,
                first + i * 16,
                words,
                13.5,
                c["ink"],
                text_anchor="middle",
                font_weight="600",
            )
            for i, words in enumerate(item.lines)
        )
        return (
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="9" fill="{c["box"]}" '
            f'stroke="{c["box_line"]}" stroke-width="1.6"/>'
            + label
            + _key(centre, y + h - (8 if compact else 11), item.key, 10.5, c["key"])
        )

    def row(items: tuple[Box, ...], top: float) -> float:
        """Boxes left to right with a connector between each; returns the right edge."""
        for index, item in enumerate(items):
            x = inner + index * (box_w + gap)
            parts.append(box(item, x, top))
            if index:
                parts.append(line(f"M{x - gap} {top + box_h / 2:g} H{x}"))
        return inner + len(items) * (box_w + gap) - gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{html.escape(alt_text(found), quote=True)}">',
        f'<defs><marker id="a" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="8" '
        f'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
        f'<path d="M1.5 1.6 L9.6 5 L1.5 8.4 z" fill="{c["arrow"]}"/></marker></defs>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14" fill="{c["card"]}" '
        f'stroke="{c["card_line"]}" stroke-width="1.5"/>',
        _text(
            24,
            40,
            "Two detector families, one toolbox",
            22,
            c["ink"],
            font_weight="700",
            letter_spacing="-0.5",
        ),
        f'<text x="24" y="63" font-family="{SANS}" font-size="13" fill="{c["muted"]}">'
        "A box with a green key is a slot: swap it with one line, "
        f'<tspan font-family="{MONO}" font-size="12">--set &lt;slot&gt;=&lt;option&gt;</tspan>'
        "</text>",
    ]

    # the datasets, joined on a rail that runs down the left into both lanes
    x, centres = float(inner), []
    for name in DATASETS:
        pill_w = 26 + 7.3 * len(name)
        parts.append(
            f'<rect x="{x:g}" y="78" width="{pill_w:g}" height="26" rx="13" fill="{c["pill"]}" '
            f'stroke="{c["card_line"]}" stroke-width="1.4"/>'
        )
        parts.append(
            _text(
                x + pill_w / 2,
                95.5,
                name,
                12.5,
                c["ink"],
                text_anchor="middle",
                font_weight="600",
            )
        )
        centres.append(x + pill_w / 2)
        x += pill_w + 10

    # the lanes first: the rails run into them and would be covered by a lane drawn later
    own = " · ".join(found.own_methods)
    parts.append(lane(seed_y, seed_h, f"Seed-based skeleton · {len(found.seed_methods)} methods"))
    parts.append(lane(own_y, own_h, f"{own} · own family", "reads a colour image and its depth"))
    parts.append(lane(shared_y, shared_h, f"Shared by all {every} methods"))
    parts += [line(f"M{centre:g} 104 V112") for centre in centres]
    parts.append(line(f"M{centres[-1]:g} 112 H{rail_left} V{own_mid:g} H{inner}", arrow=True))
    parts.append(line(f"M{rail_left} {seed_mid:g} H{inner}", arrow=True))

    # lane 1: the seed-based skeleton; its losses hang under the head bundle, its methods
    # are named beside them
    seed_end = row(found.seed_boxes, seed_top)
    if found.losses is not None:
        x, y = seed_end - box_w, seed_top + box_h + 10
        parts.append(line(f"M{x + box_w / 2:g} {seed_top + box_h} V{y}"))
        parts.append(box(found.losses, x, y, h=40))
    names_y = seed_top + box_h + 34
    parts.append(_text(inner + 2, names_y, " · ".join(found.seed_methods), 12, c["muted"]))

    # lane 2: HGGD, a family of its own
    own_end = row(found.own_boxes, own_top)

    # what every method shares: the recipe, and the post-processing both lanes run into
    recipe = [slot for slot in RECIPE if slot in found.shared]
    recipe_w, post_w, shared_box_h = 318, 160, 54
    post_x = inner + recipe_w + gap
    for x, w, title, keys in (
        (inner, recipe_w, "Training recipe", " · ".join(recipe)),
        (post_x, post_w, LABELS["post"][0], "post"),
    ):
        parts.append(
            f'<rect x="{x:g}" y="{shared_top}" width="{w}" height="{shared_box_h}" rx="9" '
            f'fill="{c["shared"]}" stroke="{c["shared_line"]}" stroke-width="1.6"/>'
        )
        parts.append(
            _text(
                x + w / 2,
                shared_top + 23,
                title,
                13.5,
                c["ink"],
                text_anchor="middle",
                font_weight="600",
            )
        )
        parts.append(_key(x + w / 2, shared_top + 41, keys, 10.5, c["key"]))
    gutter = (own_y + own_h + shared_y) / 2  # between the lanes
    parts.append(
        line(
            f"M{seed_end} {seed_mid:g} H{rail_right} V{gutter:g} H{post_x + post_w / 2:g} "
            f"V{shared_top}",
            arrow=True,
        )
    )
    parts.append(line(f"M{own_end} {own_mid:g} H{rail_right}"))
    out_x = post_x + post_w + 22
    out_w = seed_end - out_x
    out_mid = shared_top + shared_box_h / 2
    parts.append(line(f"M{post_x + post_w} {out_mid:g} H{out_x:g}", arrow=True))
    parts.append(
        f'<rect x="{out_x:g}" y="{out_mid - 14:g}" width="{out_w:g}" height="28" rx="14" '
        f'fill="{c["pill"]}" stroke="{c["card_line"]}" stroke-width="1.4"/>'
    )
    parts.append(
        _text(
            out_x + out_w / 2,
            out_mid + 4.5,
            "6-DoF grasps",
            12.5,
            c["ink"],
            text_anchor="middle",
            font_weight="600",
        )
    )

    # legend, in the words of the hero
    entries = [
        ("a slot", f'fill="{c["box"]}" stroke="{c["box_line"]}" stroke-width="1.6"'),
        (
            f"shared by all {every} methods",
            f'fill="{c["shared"]}" stroke="{c["shared_line"]}" stroke-width="1.6"',
        ),
        (
            "a fixed part of the method",
            f'fill="{c["fixed"]}" stroke="{c["fixed_line"]}" stroke-width="1.4" '
            'stroke-dasharray="4 3"',
        ),
    ]
    x = float(lane_x)
    for label, style in entries:
        parts.append(
            f'<rect x="{x:g}" y="{legend_y - 10}" width="18" height="12" rx="3.5" {style}/>'
        )
        parts.append(_text(x + 26, legend_y, label, 12, c["muted"]))
        x += 26 + 5.9 * len(label) + 20
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
        ".gp-hf-fixed{fill:var(--gp-surface);stroke:var(--gp-muted);stroke-width:1.4}",
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
        dashes = ' stroke-dasharray="4 3"' if kind == "gp-hf-fixed" else ""
        parts.append(
            f'<rect class="{kind}" x="{legend_x}" y="{y}" width="18" height="10" rx="3"{dashes}/>'
        )
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
        dashes = "" if item.key else " " + centred_dashes(box_w, h, 8)
        parts.append(
            f'<rect class="{kind}" x="{x}" y="{top}" width="{box_w}" height="{h}" rx="8"{dashes}/>'
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
    # the methods of the lane, beside the losses
    names = list(found.seed_methods)
    for index, words in enumerate((" · ".join(names[:3]), " · ".join(names[3:]))):
        parts.append(
            f'<text class="gp-hf-m" x="{inner + 2}" y="{seed_top + box_h + 20 + index * 14}">'
            f"{html.escape(words)}</text>"
        )

    # lane 2: HGGD, a family of its own
    lane(own_y, own_h, " · ".join(found.own_methods) + " · own family")
    own_end = row(found.own_boxes, own_top, late=True)

    # what every method shares; both lanes run into its post-processing down the right rail
    lane(shared_y, shared_h, f"Shared by all {every} methods")
    recipe = " · ".join(slot for slot in RECIPE if slot in found.shared)
    recipe_w = 300  # wide enough for the keys of the recipe, written out
    post_w = lane_w - 12 - gap - recipe_w
    top = shared_y + 28
    post_x = inner + recipe_w + gap
    for x, w, title, keys in (
        (inner, recipe_w, "Training recipe", recipe),
        (post_x, post_w, "Post-processing", "post"),
    ):
        parts.append(
            f'<rect class="gp-hf-shared" x="{x:g}" y="{top}" width="{w:g}" height="48" rx="9"/>'
        )
        parts.append(
            f'<text class="gp-hf-t" x="{x + w / 2:g}" y="{top + 20}" text-anchor="middle">'
            f"{title}</text>"
        )
        parts.append(
            f'<text class="gp-hf-k" x="{x + w / 2:g}" y="{top + 37}" text-anchor="middle">'
            f"{html.escape(keys)}</text>"
        )
    post_mid = top + 24
    parts.append(
        f'<path class="gp-hf-a" d="M{seed_end} {seed_mid:g} H{rail_right} V{post_mid} '
        f'H{post_x + post_w:g}"/>'
        f'<path class="gp-hf-a" d="M{own_end} {own_mid:g} H{rail_right}"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------- main


#: The faces the PNG exports are drawn with (the first names of SANS and MONO).
FACES = ("Inter", "JetBrains Mono")


def missing_faces() -> list[str]:
    """The faces of :data:`FACES` that fontconfig does not know on this machine."""
    listing = shutil.which("fc-list")
    if listing is None:
        return []  # no fontconfig to ask: the exporter decides
    known = subprocess.run(
        [listing, ":", "family"], capture_output=True, text=True, check=False
    ).stdout
    families = {name.strip() for line in known.splitlines() for name in line.split(",")}
    return [face for face in FACES if face not in families]


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
    parser.add_argument(
        "--allow-fallback-font",
        action="store_true",
        help="export the PNG files even when Inter or JetBrains Mono is not installed",
    )
    args = parser.parse_args(argv)
    if args.check:
        found = stale()
        for path in found:
            print(f"stale: {path.relative_to(REPO)}")
        return 1 if found else 0

    import cairosvg

    missing = missing_faces()
    if missing and not args.allow_fallback_font:
        print(
            f"not installed: {', '.join(missing)}. The PNG exports would be drawn with a "
            "fallback face. Install the faces (see the top of this file) or pass "
            "--allow-fallback-font.",
            file=sys.stderr,
        )
        return 2

    for path, content in svgs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    exports = {
        "logo.svg": 512,
        "logo-dark.svg": 512,
        "banner-light.svg": 2560,
        "banner-dark.svg": 2560,
        "pipeline.svg": 2040,
        "pipeline-dark.svg": 2040,
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
