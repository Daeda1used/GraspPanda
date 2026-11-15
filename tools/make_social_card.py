"""The card a link to GraspPanda shows: 1280 × 640, the lockup, the headline and the hero.

    python -m tools.make_social_card

writes ``resources/social-preview.png`` (upload it under the repository's Settings → Social
preview) and the same file as ``docs/assets/social-card.png``, which every page of the
documentation names as its ``og:image``. It reads ``resources/banner-light.svg``
(``tools/make_brand_assets.py``) and ``resources/hero.webp`` (``tools/make_hero.py``), and
needs Inter and JetBrains Mono installed, as the brand pictures do.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "resources" / "social-preview.png"
SITE_COPY = REPO / "docs" / "assets" / "social-card.png"
SIZE = (1280, 640)
INK, MUTED, GREEN = (31, 35, 40), (87, 96, 106), (45, 164, 78)
HEADLINE = "6-DoF grasp detection in clutter, one stage at a time"
LINES = ("Six published methods in one PyTorch pipeline,", "scored by one pinned evaluator.")
ADDRESS = "github.com/Daeda1used/GraspPanda"


def face(pattern: str, size: int) -> ImageFont.FreeTypeFont:
    found = subprocess.run(
        ["fc-match", "-f", "%{file}", pattern], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not found or ("Inter" in pattern and "Inter" not in found):
        raise SystemExit(f"not installed: {pattern} (see tools/make_brand_assets.py)")
    return ImageFont.truetype(found, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line]


def card() -> Image.Image:
    import cairosvg

    canvas = Image.new("RGBA", SIZE, (255, 255, 255, 255))
    lockup = Image.open(
        io.BytesIO(
            cairosvg.svg2png(url=str(REPO / "resources" / "banner-light.svg"), output_width=420)
        )
    ).convert("RGBA")
    canvas.alpha_composite(lockup, (62, 70))

    hero = Image.open(REPO / "resources" / "hero.webp").convert("RGBA")
    width = 760
    hero = hero.resize((width, width * hero.height // hero.width), Image.LANCZOS)
    canvas.alpha_composite(hero, (SIZE[0] - width + 20, (SIZE[1] - hero.height) // 2 + 70))

    draw = ImageDraw.Draw(canvas)
    bold, regular = face("Inter:weight=700", 44), face("Inter", 22)
    y = 236
    for line in wrap(draw, HEADLINE, bold, 470):
        draw.text((72, y), line, font=bold, fill=INK)
        y += 56
    y += 14
    for line in LINES:
        draw.text((72, y), line, font=regular, fill=MUTED)
        y += 31
    mono = face("JetBrains Mono", 19)
    draw.text((72, SIZE[1] - 72), ADDRESS, font=mono, fill=MUTED)
    draw.rectangle([0, SIZE[1] - 8, SIZE[0], SIZE[1]], fill=GREEN)
    return canvas.convert("RGB")


def main() -> int:
    image = card()
    image.save(OUT, optimize=True)
    shutil.copyfile(OUT, SITE_COPY)
    print(f"wrote {OUT.relative_to(REPO)} and {SITE_COPY.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
