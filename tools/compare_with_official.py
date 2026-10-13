"""Score a prediction dump with GraspPanda's evaluator and compare it with the official tool.

The official result is the array saved by ``tools/oracle/gsnet_reference.py`` (or by any
script that stores what ``GraspNetEval.eval_seen`` / ``eval_similar`` / ``eval_novel``
returns): shape ``(scenes, 256, 50, 6)``. Frames are compared table by table; frames
that differ are listed together with whether their scores are tied, because the official
sorts are unstable and tied inputs have no unique reference result.

    python tools/compare_with_official.py --predictions DIR --camera realsense \
        --split test_seen --official official_realsense_test_seen_raw.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grasppanda.evaluation import prediction_path
from grasppanda.registry import EVALUATORS
from grasppanda.structures import SampleKey

SCENES = {
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--split", choices=sorted(SCENES), required=True)
    parser.add_argument("--official", help="array saved from the official evaluator")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out", help="where to save our (scenes, 256, 50, 6) array and a JSON summary"
    )
    args = parser.parse_args()

    keys = [
        SampleKey("graspnet1b", scene, args.camera, frame)
        for scene in SCENES[args.split]
        for frame in range(256)
    ]
    result = EVALUATORS.build({"type": "GraspNet1BEvaluator"}).evaluate(
        args.predictions, keys, num_workers=args.workers
    )
    if not result.complete:
        raise SystemExit(
            f"{len(result.missing)} frames have no prediction, e.g. {result.missing[0]}"
        )
    ours = np.stack([result.precision[key] for key in keys]).reshape(
        len(SCENES[args.split]), 256, 50, 6
    )
    summary = {
        "split": args.split,
        "camera": args.camera,
        "frames": len(keys),
        "ours": {
            "AP": ours.mean() * 100,
            "AP0.8": ours[..., 3].mean() * 100,
            "AP0.4": ours[..., 1].mean() * 100,
        },
    }
    if args.official:
        official = np.load(args.official)
        different = np.argwhere(np.abs(ours - official).reshape(len(keys), -1).max(axis=1) > 0)[
            :, 0
        ]
        tied = 0
        for index in different:
            scores = np.load(prediction_path(args.predictions, keys[index]))[:, 0]
            tied += int(len(np.unique(scores)) < len(scores))
        summary["official"] = {
            "AP": official.mean() * 100,
            "AP0.8": official[..., 3].mean() * 100,
            "AP0.4": official[..., 1].mean() * 100,
        }
        summary["frames_different"] = len(different)
        summary["frames_different_with_tied_scores"] = tied
        summary["max_abs_table_difference"] = float(np.abs(ours - official).max())
        summary["first_different"] = [str(keys[i]) for i in different[:10]]
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, ours)
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
