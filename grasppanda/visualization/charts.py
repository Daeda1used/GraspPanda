"""Charts for ``grasppanda analyze``, drawn with Pillow alone.

Two figures: what one result is made of (:func:`draw_analysis`) and how two results
differ scene by scene (:func:`draw_comparison`). Both take plain numbers, so they can be
used on anything that has scenes and columns.

Conventions, kept the same in every panel: one hue for one series; a second hue only
where colour carries a second meaning (the sign of a difference, the reason a share was
lost); text in ink, never in a series colour; hairline grid; values written on the few
marks that matter (extremes, the median) and on the axis for the rest. The colours are
a fixed order that stays distinguishable under the common colour-vision deficiencies;
the numbers behind every chart are in the text report of the command.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["draw_analysis", "draw_comparison"]

_SURFACE = (252, 252, 251)
_INK = (11, 11, 11)
_INK_SOFT = (82, 81, 78)
_GRID = (229, 228, 224)
#: Fixed series order: blue, orange, aqua, yellow, magenta.
_SERIES = ((42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164))
_BLUE, _RED = _SERIES[0], (227, 73, 72)
_SCALE = 2  # drawn at twice the size and scaled down, for smooth edges and text


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size * _SCALE)
    except TypeError:  # Pillow before 10.1 has only the small bitmap font
        return ImageFont.load_default()


class _Canvas:
    """A drawing surface in output pixels; everything is scaled by ``_SCALE`` on the way in."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width * _SCALE, height * _SCALE), _SURFACE)
        self.pen = ImageDraw.Draw(self.image)

    def text(
        self, xy: tuple[float, float], text: str, size: int = 12, fill=_INK, anchor: str = "la"
    ) -> None:
        self.pen.text(
            (xy[0] * _SCALE, xy[1] * _SCALE), text, font=_font(size), fill=fill, anchor=anchor
        )

    def text_width(self, text: str, size: int = 12) -> float:
        return self.pen.textlength(text, font=_font(size)) / _SCALE

    def line(
        self, start: tuple[float, float], end: tuple[float, float], fill=_GRID, width: int = 1
    ) -> None:
        points = [(start[0] * _SCALE, start[1] * _SCALE), (end[0] * _SCALE, end[1] * _SCALE)]
        self.pen.line(points, fill=fill, width=width * _SCALE)

    def bar(
        self, box: tuple[float, float, float, float], fill, round_end: str | None = None
    ) -> None:
        """A filled rectangle; ``round_end`` rounds the corners of its ``left``/``right``/``top`` end."""
        left, top, right, bottom = (value * _SCALE for value in box)
        if right - left < 1 or bottom - top < 1:
            return
        # Pillow needs room for both arcs of a rounded end: 2 * (radius + 1) pixels.
        radius = int(min(4 * _SCALE, (right - left) / 2 - 1, (bottom - top) / 2 - 1))
        corners = {
            "right": (False, True, True, False),
            "left": (True, False, False, True),
            "top": (True, True, False, False),
        }.get(round_end or "")
        if corners is None or radius < 1:
            self.pen.rectangle([left, top, right, bottom], fill=fill)
        else:
            self.pen.rounded_rectangle(
                [left, top, right, bottom], radius=radius, fill=fill, corners=corners
            )

    def finish(self) -> Image.Image:
        return self.image.resize((self.width, self.height), Image.LANCZOS)


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values that cover ``[low, high]``."""
    span = max(high - low, 1e-12)
    raw = span / count
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = np.ceil(low / step - 1e-9) * step
    return [float(round(value, 10)) for value in np.arange(first, high + step * 1e-6, step)]


def _number(value: float) -> str:
    value = 0.0 if abs(value) < 1e-12 else value  # never "-0"
    return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.1f}"


def _row_height(count: int) -> int:
    return int(np.clip(900 // max(count, 1), 7, 22))


def _scene_bars(
    canvas: _Canvas,
    origin: tuple[int, int],
    width: int,
    names: Sequence[str],
    values: Sequence[float],
    *,
    signed: bool,
    marked: Sequence[int],
) -> int:
    """Horizontal bars, one per scene, from a zero line; returns the y below the axis."""
    x0, y0 = origin
    row = _row_height(len(names))
    thickness = min(row - 2, 14)
    label_every = max(1, int(np.ceil(13 / row)))
    gutter = max((canvas.text_width(name, 11) for name in names), default=0) + 12
    left, right = x0 + gutter, x0 + width - 56
    low = min(0.0, min(values, default=0.0)) if signed else 0.0
    high = max(max(values, default=0.0), 0.0)
    if signed:  # symmetric about zero: equal distances mean equal differences
        high = max(abs(low), abs(high), 1e-9)
        low = -high
    ticks = _ticks(low, high if high > low else low + 1.0)
    low, high = min(low, ticks[0]), max(high, ticks[-1])

    def at(value: float) -> float:
        return left + (value - low) / (high - low) * (right - left)

    bottom = y0 + row * len(names)
    for tick in ticks:
        canvas.line((at(tick), y0 - 4), (at(tick), bottom + 2))
        canvas.text((at(tick), bottom + 6), _number(tick), 11, _INK_SOFT, "ma")
    for index, (name, value) in enumerate(zip(names, values)):
        middle = y0 + row * index + row / 2
        if index % label_every == 0 or index in marked:
            canvas.text((left - 8, middle), name, 11, _INK_SOFT, "rm")
        colour = (_BLUE if value >= 0 else _RED) if signed else _BLUE
        start, end = sorted((at(0.0), at(value)))
        canvas.bar(
            (start, middle - thickness / 2, end, middle + thickness / 2),
            colour,
            "right" if value >= 0 else "left",
        )
        if index in marked:
            text = f"{value:+.4f}" if signed else f"{value:.3f}"
            if value >= 0:
                canvas.text((end + 6, middle), text, 11, _INK, "lm")
            else:
                canvas.text((start - 6, middle), text, 11, _INK, "rm")
    canvas.line((at(0.0), y0 - 4), (at(0.0), bottom + 2), fill=_INK_SOFT)
    return bottom + 26


def _marked(values: Sequence[float]) -> list[int]:
    """Rows worth a written value: the two ends and the middle of a sorted list."""
    if not values:
        return []
    return sorted({0, len(values) // 2, len(values) - 1})


def draw_analysis(
    title: str,
    subtitle: str,
    column_labels: Sequence[str],
    column_values: Sequence[float],
    scene_names: Sequence[str],
    scene_values: Sequence[float],
    shares: Mapping[str, float] | None = None,
    *,
    width: int = 1100,
) -> Image.Image:
    """One result: precision per friction column, where the ranks went, and every scene.

    Args:
        title, subtitle: What was measured and under which conditions.
        column_labels: Name of each column of the precision table (the friction levels).
        column_values: Mean precision of each column; their mean is the AP.
        scene_names: Scenes in the order to draw them (lowest value first reads best).
        scene_values: Mean precision of each scene.
        shares: Ordered ``{reason: share}`` that add up to one: the share of AP earned
            first, then every way a share was lost. Omit when the record has no split.
        width: Width of the figure in pixels; its height follows the number of scenes.
    """
    top = 150 + 230
    height = top + 44 + _row_height(len(scene_names)) * len(scene_names) + 40
    canvas = _Canvas(width, height)
    canvas.text((32, 22), title, 19)
    canvas.text((32, 52), subtitle, 12, _INK_SOFT)

    # Precision per friction column: one series, one colour, the value on every cap.
    panel_left, panel_top, panel_width, panel_height = 32, 100, 440, 190
    canvas.text((panel_left, panel_top - 8), "Mean precision by friction threshold", 13)
    base = panel_top + 28 + panel_height
    peak = max(max(column_values, default=0.0), 1e-9)
    ticks = _ticks(0.0, peak, 4)
    peak = max(peak, ticks[-1])
    plot_left = panel_left + 38
    for tick in ticks:
        y = base - tick / peak * panel_height
        canvas.line((plot_left, y), (panel_left + panel_width, y))
        canvas.text((plot_left - 8, y), _number(tick), 11, _INK_SOFT, "rm")
    slot = (panel_width - 38) / max(len(column_values), 1)
    for index, (label, value) in enumerate(zip(column_labels, column_values)):
        centre = plot_left + slot * (index + 0.5)
        y = base - value / peak * panel_height
        canvas.bar((centre - 11, y, centre + 11, base), _BLUE, "top")
        canvas.text((centre, y - 5), f"{value:.3f}", 11, _INK, "mb")
        canvas.text((centre, base + 6), str(label), 11, _INK_SOFT, "ma")
    canvas.line((plot_left, base), (panel_left + panel_width, base), fill=_INK_SOFT)

    # Where the ranks went: parts of one whole, so one stacked bar and a legend with numbers.
    if shares:
        x0, y0, bar_width = 540, panel_top + 28, width - 540 - 40
        canvas.text((x0, panel_top - 8), "Share of the AP earned, and lost by cause", 13)
        total = sum(shares.values()) or 1.0
        cursor = float(x0)
        for index, (name, value) in enumerate(shares.items()):
            span = value / total * bar_width
            colour = _SERIES[index % len(_SERIES)]
            if span > 3:  # a 2 px gap in the surface colour separates neighbours
                canvas.bar((cursor, y0, cursor + span - 2, y0 + 24), colour)
            cursor += span
            row = y0 + 46 + index * 24
            canvas.bar((x0, row, x0 + 12, row + 12), colour)
            canvas.text((x0 + 20, row + 6), name, 12, _INK, "lm")
            canvas.text((x0 + bar_width, row + 6), f"{100 * value / total:.1f} %", 12, _INK, "rm")

    canvas.text((32, top), f"Mean precision per scene ({len(scene_names)} scenes)", 13)
    _scene_bars(
        canvas,
        (32, top + 30),
        width - 64,
        scene_names,
        scene_values,
        signed=False,
        marked=_marked(scene_values),
    )
    return canvas.finish()


def draw_comparison(
    title: str,
    subtitle: str,
    scene_names: Sequence[str],
    differences: Sequence[float],
    labels: tuple[str, str] = ("A", "B"),
    *,
    width: int = 1100,
) -> Image.Image:
    """Two results, scene by scene: the difference ``B - A`` of every paired scene.

    Args:
        title, subtitle: What is compared, and the paired summary.
        scene_names: Scenes in the order to draw them (largest difference first reads best).
        differences: ``B - A`` of the mean precision of each scene.
        labels: Short names of the two results, for the legend.
        width: Width of the figure in pixels; its height follows the number of scenes.
    """
    top = 128
    height = top + 30 + _row_height(len(scene_names)) * len(scene_names) + 44
    canvas = _Canvas(width, height)
    canvas.text((32, 22), title, 19)
    canvas.text((32, 52), subtitle, 12, _INK_SOFT)
    canvas.text((32, top - 30), "Difference in mean precision per scene (B - A)", 13)
    x = width - 40
    for colour, text in ((_RED, f"{labels[0]} higher"), (_BLUE, f"{labels[1]} higher")):
        x -= canvas.text_width(text, 12)
        canvas.text((x, top - 23), text, 12, _INK, "lm")
        canvas.bar((x - 18, top - 29, x - 6, top - 17), colour)
        x -= 40
    _scene_bars(
        canvas,
        (32, top + 8),
        width - 64,
        scene_names,
        differences,
        signed=True,
        marked=_marked(differences),
    )
    return canvas.finish()
