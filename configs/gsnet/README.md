# GSNet

> [Graspness Discovery in Clutters for Fast and Accurate Grasp Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html)
> (ICCV 2021). Reference code:
> [graspnet/graspness_unofficial](https://github.com/graspnet/graspness_unofficial) at `b5abf5a`
> (academic, non-commercial; a third-party implementation, there is no official training code).

GSNet predicts a point-wise *graspness* score, samples seeds only where it is high,
classifies one of 300 approach views per seed, crops a cylinder along that view and
regresses grasp score and width over a grid of in-plane angles and depths.

**Run it in five minutes:** [the method page](https://daeda1used.github.io/GraspPanda/methods/gsnet/#run-in-five-minutes): the weights, grasps on the example scene, the same from Python and in the browser, and a first number on the benchmark, each with what it printed when it was measured.

## Configs

| Config | Trains on | Tests on |
|---|---|---|
| [gsnet_sunet14_g1b-realsense_10e](gsnet_sunet14_g1b-realsense_10e.yaml) | GraspNet-1Billion, RealSense | GraspNet-1Billion, RealSense |
| [gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e](gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml) | GraspNet-1Billion, RealSense | GraspClutter6D, RealSense D435 |
| [gsnet_sunet14_gc6d-d435_10e](gsnet_sunet14_gc6d-d435_10e.yaml) | GraspClutter6D, RealSense D435 | GraspClutter6D, RealSense D435 |

The options of every slot, with what was measured for each:
[the method page](https://daeda1used.github.io/GraspPanda/methods/gsnet/#the-slots), or in a terminal
`grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --list` (one slot: `--list grouper`).

## Results on GraspNet-1Billion

AP on seen / similar / novel over the whole test split (90 scenes, 23,040 frames, all
scored), protocol `graspnet1b-sv-v1`. The workspace is cropped with the ground-truth
segmentation, as in the reference (`gt_assisted: workspace`).

| Config | Camera | Collision filter | Provenance | Seen | Similar | Novel |
|---|---|:---:|---|---|---|---|
| [gsnet_sunet14_g1b-realsense_10e](gsnet_sunet14_g1b-realsense_10e.yaml) | RealSense | off | reported (paper) | 65.70 | 53.75 | 23.98 |
| | RealSense | on | reported (paper) | 67.12 | 54.81 | 24.31 |
| | RealSense | off | reproduced ([record](records/gsnet_sunet14_g1b-realsense_10e_reproduced_test_raw.json)) | 68.83 | 60.35 | 25.20 |
| | RealSense | on | reproduced ([record](records/gsnet_sunet14_g1b-realsense_10e_reproduced_test_cd.json)) | 71.23 | 62.29 | 26.42 |

`reproduced` is a model trained from scratch by this toolbox. **Measured minus reported**:
+3.13 / +6.60 / +1.22 AP without the collision filter, +4.11 / +7.48 / +2.11 AP with it,
above the paper on every split by more than the 1 AP of the zoo's admission gate. The gap
is not explained.

<details markdown="1"><summary>How it was trained, and what is known about the gap</summary>

It was trained with `grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e` at commit
`0125baa`, the recipe of the config unchanged: 10 epochs over the 25,600 RealSense frames
of the training scenes, batch size 4, Adam at 1e-3 decayed by 0.95 per epoch, seed 0, one
GPU, about 3.4 hours. The checkpoint of epoch 10 (SHA-256 `829ad72d130d…`, in full in the
metafile) was scored through `grasppanda test` at commit `4bc5333`: `off` is the
prediction as emitted (variant `raw`), `on` adds the reference's model-free collision
filter (variant `cd`). The result records hold the per-scene values, the config hash and
the hashes of code and weights.

What is known: this implementation equals the public reference stage by stage, including
every loss term of a recorded training batch (see *Equivalence with the reference code*
below), so its training step is the reference's. A checkpoint trained independently with
the reference code (on a modified graspness label set, for 18 epochs) is above the paper
as well, at 69.79 / 60.83 / 25.52 without the filter. The paper's numbers come from the
authors' original code, which is not public, so what separates the two cannot be checked
from here. This is one training run with one seed; a training with the unmodified
reference code was not run here.

</details>

## Results on GraspClutter6D

The same model on the whole GraspClutter6D test split (235 scenes, 3,055 frames of the
RealSense D435), protocol `graspclutter6d-sv-v1`, config
[gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e](gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml),
scored at commit `4bc5333`. Percent.

| Collision filter | AP | AP0.8 | AP0.4 | Provenance |
|---|---|---|---|---|
| off | 25.58 | 31.38 | 19.42 | reproduced ([record](records/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_raw.json)) |
| on | 26.87 | 33.00 | 20.04 | reproduced ([record](records/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_cd.json)) |

The GraspClutter6D paper reports no GSNet number, so there is nothing to compare with.

<details markdown="1"><summary>The input on this dataset</summary>

The input is cropped to the image rectangle around the ground-truth segmentation, as the
official tool does (`gt_assisted: workspace`); this is not the crop the weights were
trained with. What else changes with the dataset is described in
`docs/user_guides/cross_dataset.md`.

</details>

## Weights

**The weights trained here are not published yet.** They were trained on
GraspNet-1Billion, whose licence is CC BY-NC-SA 4.0, and whether and where they are
published has not been decided. The metafile lists them with their SHA-256 and
`url: null`, so `grasppanda demo gsnet/gsnet_sunet14_g1b-realsense_10e` without
`--weights` says that they are not published and downloads nothing.

The reference repository releases `minkuresunet_realsense.tar`. It loads strictly and
runs, but far below the paper: use it with the collision filter. Convert it once:

```bash
python -m tools.convert_checkpoint minkuresunet_realsense.tar gsnet_realsense.pth \
    --source-format graspness-unofficial --camera realsense
```

<details markdown="1"><summary>The checkpoints released with the reference code</summary>

The reference repository links two files, `minkuresunet_realsense.tar` and
`minkuresunet_kinect.tar`; its README states no AP for them and recommends the RealSense
one for real-world use. The RealSense file (184,429,769 bytes, SHA-256 `a0b9d85b300e…`)
loads strictly into this model, but it does not give the numbers of the paper, neither
here nor in the reference code. Measured on 2026-10-13 with the reference code at
`b5abf5a` and the functions of the official evaluator, on 16 frames of test scene 100 (a
subset, not a benchmark number), beside a model trained with the same reference code:

| | released file | trained with the reference code |
|---|---|---|
| AP, no collision filter | 9.8 | 59.0 |
| AP, with the reference's collision detector | 23.4 | 61.4 |
| AP if the evaluator did not reject colliding grasps (a diagnostic) | 63.1 | 74.0 |
| best 50 grasps rejected by the evaluator for collision | 93.5 % | 25.4 % |
| predicted graspness against the reference's label, correlation on 16 training frames | 0.45 | 0.98 |
| the same against a label computed without the scene-collision mask | 0.93 | 0.46 |

The archive is sound: epoch 10 of the default schedule, every tensor present, nothing
non-finite; its objectness is as accurate as the other model's (0.94). Its graspness,
view and score outputs follow labels that ignore collisions with the scene. Our reading
is that it was trained without the scene-collision labels, so its best-ranked grasps run
into the table and into neighbouring objects. This reading is not confirmed, and
neither the Kinect file nor a fresh download of the RealSense file could be fetched to
check. A user of another implementation reports the same low score for the released
Kinect file
([rhett-chen/graspness_implementation#38](https://github.com/rhett-chen/graspness_implementation/issues/38)).

So the released file is no reference for the benchmark and has no row in the results.
On a robot it should run with a collision filter (`--collision-filter`, the config's `cd`
variant), as the reference's own demo script does. Weights at the level of the paper come
from training, which applies the collision labels by default.
[The method page](https://daeda1used.github.io/GraspPanda/methods/gsnet/#run-in-five-minutes) has every step, with the AP it gets
on 120 frames of the benchmark.

</details>

## Train and test

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense
grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e
grasppanda test  gsnet/gsnet_sunet14_g1b-realsense_10e --weights <checkpoint>
```

Both evaluation variants are always reported: `raw` (predictions as emitted) and `cd`
(the reference's optional model-free collision filter).

## Notes

<details markdown="1"><summary>How it is built</summary>

| Slot | Component | What it does |
|---|---|---|
| `backbone` | `SparseUNet14` | 14-layer sparse U-Net on spconv, 5 mm voxels, per-point 512-d features |
| `proposal` | `GraspnessFPS` | objectness and graspness per point, then furthest point sampling to 1024 seeds |
| `view` | `TemplateViewStage` | 300 template views; a sampled view in training, the arg-max in inference |
| `assigner` | `NearestLabelAssigner` | nearest label point per seed, labels at the chosen view |
| `grouper` | `CylinderGroup` | one cylinder per seed among the seeds, radius 0.05 m, 16 neighbours |
| `head` | `SWADHead` | score and width grids over 12 angles and 4 depths |

The full default model is [`configs/_base_/models/gsnet.yaml`](../_base_/models/gsnet.yaml).
Which method ships what in each slot:
[Methods](https://daeda1used.github.io/GraspPanda/methods/#what-each-method-ships).

</details>

<details markdown="1"><summary>Equivalence with the reference code</summary>

Checked with `tools/make_gsnet_fixtures.py` and `tests/integration/test_gsnet_parity.py`
against the reference at `b5abf5a`, with weights converted by `tools/convert_checkpoint.py`:

| What | Result |
|---|---|
| Parameters | 15,353,775, equal to the reference |
| Seeds and chosen views | identical on 72 of 72 recorded frames |
| Backbone features, view scores, head outputs | within 5e-6 relative |
| Decoded grasps | score within 1.1e-6, width within 9e-8 m, rotation within 1.2e-7 |
| Training batch: every loss term | within 2e-7 relative |

The same weights give the same predictions; this is not an accuracy claim. The same holds
on the whole test split (90 scenes, 23,040 frames, RealSense): one checkpoint was run
through the reference code (`tools/oracle/gsnet_reference.py`) and through
`grasppanda test`, and both sets of predictions were scored with GraspPanda's evaluator,
which equals the official one frame by frame (`tests/integration/test_eval_parity.py`).
Seen / similar / novel: 69.79 / 60.83 / 25.52 (reference code) and 69.78 / 60.88 / 25.52
(GraspPanda) without the filter, 72.04 / 62.85 / 26.60 and 72.09 / 62.93 / 26.62 with it.
They differ by at most 0.08 AP; the two draw their 15,000 input points with different
random generators. The checkpoint is not the authors': it was trained with the reference
code on a modified graspness label set, which is why these numbers are evidence about the
pipeline and do not appear in the results.

</details>

<details markdown="1"><summary>Reference behaviours kept as named options</summary>

The options marked "reference quirk" in
[`configs/_base_/models/gsnet.yaml`](../_base_/models/gsnet.yaml) reproduce behaviour of
the reference implementation that changes numbers.

</details>

<details markdown="1"><summary>How this port differs from the reference</summary>

- The reference implementation is third-party; no official training code exists.
- The reference test script enables its collision filter by default, while the published
  numbers are without it. Both variants are always evaluated here.
- Point-wise graspness labels must come from `grasppanda prepare`. Label files made by
  other generators are row-aligned to a preprocessing they do not record.

</details>

## Citation

<details markdown="1"><summary>BibTeX</summary>

```bibtex
@inproceedings{wang2021graspness,
  title     = {Graspness Discovery in Clutters for Fast and Accurate Grasp Detection},
  author    = {Wang, Chenxi and Fang, Hao-Shu and Gou, Minghao and Fang, Hongjie and Gao, Jin and Lu, Cewu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```

</details>
