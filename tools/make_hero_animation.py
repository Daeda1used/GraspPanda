"""The animation that opens the README: methods as published, one taken apart, the best run.

    python -m tools.make_hero_animation predict --gpu 0   # grasps of every run on one scene
    python -m tools.make_hero_animation render            # resources/hero-swap.webp, docs/assets/hero-swap.{mp4,webp}

Act 1 shows the six methods as trained here, act 2 GSNet with one stage swapped at a time
(its scene encoder, then its training recipe), act 3 holds the run with the highest AP. Every
step is a model trained here and scored on the whole GraspNet-1Billion test split: its AP is
read from its committed result record (:data:`RUNS`), and its grasps are what that model
predicts on one test scene, drawn by ``tools/make_hero.py`` from one camera for all steps.
``predict`` checks that the weights it runs are the file of the record (SHA-256), and runs
``grasppanda demo`` on each; ``render`` draws the pictures, fills ``tools/hero_animation.html``
with the records and records it frame by frame in headless Chrome (Playwright), then writes
an animated WebP for the README and, when an ``ffmpeg`` is found, an MP4 for the site.

The weights are not published (``docs/model_zoo.md``): ``predict`` runs where they are,
named by :data:`WEIGHTS` below ``$GRASPPANDA_WEIGHTS`` and the sweep runs below ``--sweeps``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "tools" / "hero_animation.html"
WEBP = REPO / "resources" / "hero-swap.webp"
MP4 = REPO / "docs" / "assets" / "hero-swap.mp4"
POSTER = REPO / "docs" / "assets" / "hero-swap-poster.webp"
SCENE, FRAME = 163, 0  # a scene of the test split whose objects are all novel
METRIC = "AP · seen · collision filter on"
VARIANT, SPLIT = "cd", "seen"

#: One step of the animation: act, key, the label of the chart (two lines), the record (a
#: stem below the repository; ``_cd.json`` is added) and the config that ``predict`` runs.
RUNS = [
    (0, "graspnet-baseline", ("Baseline", ""),
     "configs/graspnet_baseline/records/graspnet-baseline_pointnet2_g1b-realsense_18e_reproduced_test",
     "graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e"),
    (0, "gsnet", ("GSNet", ""), "configs/gsnet/records/gsnet_sunet14_g1b-realsense_10e_reproduced_test",
     "gsnet/gsnet_sunet14_g1b-realsense_10e"),
    (0, "sbg", ("SBG", ""),
     "configs/scale_balanced_grasp/records/sbg_pointnet2-transformer_g1b-realsense_18e_reproduced_test",
     "scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e"),
    (0, "economicgrasp", ("Economic", ""),
     "configs/economicgrasp/records/economicgrasp_sunet14_g1b-realsense_10e_reproduced_test",
     "economicgrasp/economicgrasp_sunet14_g1b-realsense_10e"),
    (0, "finegrasp", ("FineGrasp", ""),
     "configs/finegrasp/records/finegrasp_sunet14_g1b-realsense_10e_reproduced_test",
     "finegrasp/finegrasp_sunet14_g1b-realsense_10e"),
    (0, "hggd", ("HGGD", ""), "configs/hggd/records/hggd_r34w8_g1b-realsense_15e_reproduced_test",
     "hggd/hggd_r34w8_g1b-realsense_15e"),
    (1, "gsnet", ("GSNet", ""), "configs/gsnet/records/gsnet_sunet14_g1b-realsense_10e_reproduced_test",
     "gsnet/gsnet_sunet14_g1b-realsense_10e"),
    (1, "spunet", ("SpUNet", ""), "projects/encoder_comparison/records/spunet_test",
     "sweep:gsnet_encoders/spunet"),
    (1, "ptv3", ("PTv3", ""), "projects/encoder_comparison/records/ptv3_test",
     "sweep:gsnet_encoders/ptv3"),
    (1, "litept", ("LitePT", ""), "projects/encoder_comparison/records/litept_test",
     "sweep:gsnet_encoders/litept"),
    (1, "ptv3_onecycle", ("PTv3", "one-cycle"),
     "projects/encoder_comparison/records/ptv3_adamw-onecycle_test", "sweep:gsnet_recipe/ptv3_adamw-onecycle"),
    (1, "litept_onecycle", ("LitePT", "one-cycle"),
     "projects/encoder_comparison/records/litept_adamw-onecycle_test",
     "sweep:gsnet_recipe/litept_adamw-onecycle"),
    (1, "sunet14_onecycle", ("SUNet14", "one-cycle"),
     "projects/encoder_comparison/records/sunet14_adamw-onecycle_test",
     "sweep:gsnet_recipe/sunet14_adamw-onecycle"),
]  # fmt: skip
#: The weights of the method rows on the machine that trained them, below $GRASPPANDA_WEIGHTS.
WEIGHTS = {
    "graspnet-baseline": "trained/graspnet-baseline_pointnet2_g1b-realsense_18e_x6-seed0.pth",
    "gsnet": "trained/gsnet_sunet14_g1b-realsense_10e_a6000-seed0.pth",
    "sbg": "trained/sbg_pointnet2-transformer_g1b-realsense_18e_x6-seed0.pth",
    "economicgrasp": "trained/economicgrasp_sunet14_g1b-realsense_10e_x6-seed0.pth",
    "finegrasp": "trained/finegrasp_sunet14_g1b-realsense_10e_4x4x2_x6-seed0.pth",
    "hggd": "trained/hggd_r34w8_g1b-realsense_15e_labels-1cm-30deg_x6-seed0.pth",
}
#: What act 2 swaps in GSNet, per step: (scene encoder, recipe).
SWAPS = {
    "gsnet": ("sparse-unet14", None),
    "spunet": ("spunet", None),
    "ptv3": ("ptv3", None),
    "litept": ("litept", None),
    "ptv3_onecycle": ("ptv3", "adamw-onecycle"),
    "litept_onecycle": ("litept", "adamw-onecycle"),
    "sunet14_onecycle": ("sparse-unet14", "adamw-onecycle"),
}
ACTS = [
    {"title": "Run published methods", "sub": "Six methods in one pipeline, scored by one evaluator",
     "pill": "Run"},
    {"title": "Swap one stage at a time", "sub": "GSNet taken apart: its scene encoder, then its recipe",
     "pill": "Swap"},
    {"title": "Keep the best", "sub": "The highest AP of these runs, and its grasps", "pill": "Best"},
]  # fmt: skip
NOTE = (
    "Each point: a model trained here, scored on the whole GraspNet-1Billion test split. "
    "Grasps: its predictions on a test scene of unseen objects."
)
TIMING = {"intro": 0.8, "between": 0.6, "step": [1.35, 1.45], "finale": 4.6}
FPS = 20


def record(stem: str) -> dict:
    return json.loads((REPO / f"{stem}_{VARIANT}.json").read_text(encoding="utf-8"))


def ap_of(stem: str) -> float:
    sys.path.insert(0, str(REPO))
    from tools import record_result

    return round(record_result.headline(record(stem))["AP"][SPLIT], 2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_files(key: str, config: str, stem: str, sweeps: Path) -> tuple[str, Path]:
    """The config and the weights ``predict`` runs for one step, checked against its record."""
    if config.startswith("sweep:"):
        path = record(stem)["weights"]["path"]  # work_dirs/<sweep>/<cell>/seed0/<run>/checkpoints/…
        weights = sweeps / path
        config = str(weights.parent.parent / "config.resolved.yaml")
    else:
        weights = Path(os.environ["GRASPPANDA_WEIGHTS"]) / WEIGHTS[key]
    expected = record(stem)["weights"]["sha256"]
    if file_sha256(weights) != expected:
        raise SystemExit(f"{weights} is not the file of {stem} (sha256 {expected[:12]}…)")
    return config, weights


def predict(args: argparse.Namespace) -> int:
    work = args.work
    for _, key, _, stem, config in RUNS:
        out = work / "demo" / key
        if (out / "grasps.npy").is_file():
            continue
        config, weights = run_files(key, config, stem, args.sweeps)
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "grasppanda.cli.main", "demo", config, "--weights", str(weights),
            "--frame", f"graspnet1b:scene_{SCENE:04d}/realsense/{FRAME:04d}", "--collision-filter",
            "--post", "nms", "--gif-frames", "0", "--out", str(out),
        ]  # fmt: skip
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO)}
        with (out / "log.txt").open("w") as log:
            status = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        print(f"{key:18s} exit {status}")
        if status:
            return status
    return 0


def labels() -> dict[str, list[dict]]:
    """The stages of each method's card: what it ships, from the explorer's data."""
    data = json.loads((REPO / "docs" / "explorer" / "explorer.json").read_text(encoding="utf-8"))
    cards = {}
    for method in data["methods"]:
        ships = method["ships"]

        def name(slot: str, ships: dict = ships) -> str:
            entry = ships.get(slot) or {}
            option = entry.get("option") or ""
            return option if option and option == option.lower() else entry.get("component", "")

        recipe = f"{name('optimizer')} · {name('schedule')}"
        if method["family"] == "seed-based":
            rows = [
                {"slot": "Scene encoder", "value": name("backbone")},
                {"slot": "Seed proposal", "value": name("proposal")},
                {"slot": "View", "value": name("view")},
                {"slot": "Grouper", "value": name("grouper")},
                {"slot": "Head", "value": name("head")},
                {"slot": "Recipe", "value": recipe},
            ]
        else:
            rows = [
                {"slot": "Image encoder", "value": ships["image_encoder"].get("label", "")},
                {"slot": "Heat-map", "value": "fixed stage", "fixed": True},
                {"slot": "Region", "value": "fixed stage", "fixed": True},
                {"slot": "Recipe", "value": recipe},
            ]
        cards[method["name"]] = {"venue": method["venue"], "rows": rows}
    return cards


NAMES = {
    "graspnet-baseline": "GraspNet-baseline",
    "gsnet": "GSNet",
    "sbg": "Scale-Balanced-Grasp",
    "economicgrasp": "EconomicGrasp",
    "finegrasp": "FineGrasp",
    "hggd": "HGGD",
}


def steps(images: dict[str, str]) -> list[dict]:
    cards = labels()
    out = []
    for act, key, (short, short2), stem, _ in RUNS:
        if act == 0:
            method = NAMES[key]
            card = cards[method]
            step = {"card": method, "chip": card["venue"], "rows": card["rows"], "tag": method}
        else:
            encoder, recipe = SWAPS[key]
            rows = [dict(row) for row in cards["GSNet"]["rows"]]
            for row in rows:
                if row["slot"] == "Scene encoder":
                    row.update(value=encoder, swapped=True, best=True)
                if row["slot"] == "Recipe":
                    row.update(value=recipe or row["value"], swapped=True, best=True)
            tag = (
                "GSNet"
                + ("" if key == "gsnet" else f" · {encoder}")
                + (" · one-cycle" if recipe else "")
            )
            step = {"card": "GSNet", "chip": "taken apart", "hot": True, "rows": rows, "tag": tag}
        step.update(act=act, key=key, ap=ap_of(stem), short=short, short2=short2, image=images[key])
        out.append(step)
    return out


def pictures(work: Path, root: Path) -> dict[str, str]:
    """One picture per run, all from one camera."""
    sys.path.insert(0, str(REPO))
    from tools.make_hero import Stage

    stage = Stage(root, SCENE, FRAME)
    keys = list(dict.fromkeys(key for _, key, *_ in RUNS))
    grasps = {key: stage.grasps(work / "demo" / key / "grasps.npy", 10, 0.05) for key in keys}
    size = (1840 * 2, 920 * 2)
    view = stage.view(size, list(grasps.values()), 25.0, 40.0, 28.0, 0.94)
    folder = work / "pictures"
    folder.mkdir(parents=True, exist_ok=True)
    images = {}
    for key in keys:
        target = folder / f"{key}.png"
        stage.picture(grasps[key], view, 2).save(target)
        images[key] = f"pictures/{key}.png"
        print(f"drew {target.name}")
    return images


def page(work: Path, images: dict[str, str]) -> Path:
    found = steps(images)
    best = max(range(len(found)), key=lambda i: found[i]["ap"])
    values = [step["ap"] for step in found]
    data = {
        "steps": found,
        "acts": ACTS,
        "timing": TIMING,
        "note": NOTE,
        "metric": METRIC,
        "split": sum(step["act"] == 0 for step in found),
        "range": [10 * (min(values) // 10) - 5, 10 * (max(values) // 10) + 10],
        "gridlines": list(range(int(10 * (min(values) // 10)), int(max(values)) + 1, 10)),
        "best": {
            "index": best,
            "ap": found[best]["ap"],
            "label": f"Highest AP of these {len({s['key'] for s in found})} models",
            "short": " · ".join(x for x in (found[best]["short"], found[best]["short2"]) if x),
        },
    }
    logo = (REPO / "resources" / "logo-dark.svg").read_text(encoding="utf-8")
    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace("{{FONTS}}", (REPO / "docs" / "assets" / "fonts").as_uri())
    text = text.replace("{{LOGO}}", logo).replace("{{DATA}}", json.dumps(data))
    target = work / "hero_animation.html"
    target.write_text(text, encoding="utf-8")
    return target


async def record_frames(target: Path, frames_dir: Path, fps: int) -> list[Path]:
    from playwright.async_api import async_playwright

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    written = []
    async with async_playwright() as play:
        browser = await play.chromium.launch(channel="chrome", headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 900})
        tab = await context.new_page()
        await tab.goto(target.as_uri())
        await tab.evaluate("document.fonts.ready")
        total = await tab.evaluate("window.TOTAL")
        count = int(total * fps)
        stage = await tab.query_selector("#stage")
        for index in range(count):
            await tab.evaluate(f"window.render({index / fps})")
            await tab.evaluate(
                "Promise.all([...document.images].filter(i => i.src).map(i => i.decode().catch(() => 0)))"
            )
            path = frames_dir / f"{index:04d}.png"
            await stage.screenshot(path=str(path), omit_background=True)
            written.append(path)
        await browser.close()
    return written


def encode(frames: list[Path], fps: int, ffmpeg: str | None) -> None:
    from PIL import Image

    images = [Image.open(path).convert("RGBA") for path in frames]
    WEBP.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        WEBP,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        quality=72,
        method=6,
        minimize_size=True,
    )
    print(
        f"wrote {WEBP.relative_to(REPO)} ({WEBP.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)"
    )
    images[-round(1.2 * fps)].save(POSTER, quality=88, method=6)  # the finale, before the fade
    print(f"wrote {POSTER.relative_to(REPO)}")
    if ffmpeg:
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
            str(frames[0].parent / "%04d.png"), "-vf", "format=yuv420p", "-c:v", "libx264",
            "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(MP4),
        ]  # fmt: skip
        subprocess.run(command, check=True)
        print(f"wrote {MP4.relative_to(REPO)} ({MP4.stat().st_size / 1e6:.2f} MB)")


def find_ffmpeg(given: str | None) -> str | None:
    if given:
        return given
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def render(args: argparse.Namespace) -> int:
    root = args.root or Path(os.environ["GRASPPANDA_GRASPNET1B_ROOT"])
    images = pictures(args.work, root)
    target = page(args.work, images)
    frames = asyncio.run(record_frames(target, args.work / "frames", args.fps))
    encode(frames, args.fps, find_ffmpeg(args.ffmpeg))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "render"):
        sub = commands.add_parser(name)
        sub.add_argument("--work", type=Path, default=REPO / "work_dirs" / "hero_animation")
    commands.choices["predict"].add_argument("--gpu", default="0")
    commands.choices["predict"].add_argument(
        "--sweeps", type=Path, default=Path(os.environ.get("GP_RUNS", ".")) / "sweeps",
        help="the directory the sweeps ran in (their records name work_dirs/... below it)",
    )  # fmt: skip
    commands.choices["render"].add_argument("--root", type=Path, default=None)
    commands.choices["render"].add_argument("--fps", type=int, default=FPS)
    commands.choices["render"].add_argument("--ffmpeg", default=None)
    args = parser.parse_args(argv)
    return {"predict": predict, "render": render}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
