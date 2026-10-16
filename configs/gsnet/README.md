# GSNet

> [Graspness Discovery in Clutters for Fast and Accurate Grasp Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html) (ICCV 2021)

GSNet predicts a point-wise *graspness* score, samples seeds only where it is high,
classifies one of 300 approach views per seed, crops a cylinder along that view and
regresses grasp score and width over a grid of in-plane angles and depths.

**Run it in five minutes:** [docs/methods/gsnet.md](../../docs/methods/gsnet.md) gives the authors' checkpoint, its licence and the conversion command, the demo on the example scene with what it prints, the same from Python and in the browser, and a first AP on 120 frames of the benchmark, each with the date, commit and machine it was measured on.

## Components

| Slot | Component | Notes |
|---|---|---|
| `backbone` | `SparseUNet14` | 14-layer sparse U-Net on spconv, 5 mm voxels, per-point 512-d features |
| `proposal` | `GraspnessFPS` | objectness and graspness per point, then furthest point sampling to 1024 seeds |
| `view` | `TemplateViewStage` | 300 template views; a sampled view in training, the arg-max in inference |
| `assigner` | `NearestLabelAssigner` | nearest label point per seed, labels at the chosen view |
| `grouper` | `CylinderGroup` | one cylinder per seed among the seeds, radius 0.05 m, 16 neighbours |
| `head` | `SWADHead` | score and width grids over 12 angles and 4 depths |

The full default model is [`configs/_base_/models/gsnet.yaml`](../_base_/models/gsnet.yaml).
Options marked "reference quirk" there reproduce behaviour of the reference
implementation that changes numbers.

## Equivalence with the reference implementation

Checked with `tools/make_gsnet_fixtures.py` and `tests/integration/test_gsnet_parity.py`
against [graspnet/graspness_unofficial](https://github.com/graspnet/graspness_unofficial)
at `b5abf5a`, using weights converted with `tools/convert_checkpoint.py`:

| What | Result |
|---|---|
| Parameters | 15,353,775, equal to the reference |
| Seeds and chosen views | identical on 72 of 72 recorded frames |
| Backbone features, view scores, head outputs | within 5e-6 relative |
| Decoded grasps | score within 1.1e-6, width within 9e-8 m, rotation within 1.2e-7 |
| Training batch: every loss term | within 2e-7 relative |

This shows that the same weights give the same predictions. It is not an accuracy
claim: see the results below.

The same holds on the whole test split (90 scenes, 23,040 frames, RealSense). One
checkpoint was run through the reference code (`tools/oracle/gsnet_reference.py`) and
through `grasppanda test`; both sets of predictions were scored with GraspPanda's
evaluator, which equals the official one frame by frame
(`tests/integration/test_eval_parity.py`):

| Pipeline | Collision filter | Seen | Similar | Novel |
|---|:---:|---|---|---|
| reference code | off | 69.79 | 60.83 | 25.52 |
| GraspPanda | off | 69.78 | 60.88 | 25.52 |
| reference code | on | 72.04 | 62.85 | 26.60 |
| GraspPanda | on | 72.09 | 62.93 | 26.62 |

The two differ by at most 0.08 AP; they draw their 15,000 input points with different
random generators. The checkpoint is not the authors': it was trained with the reference
code on a modified graspness label set, which is why these numbers are evidence about the
pipeline and do not appear in the results table.

## Results on GraspNet-1Billion

AP on seen / similar / novel, protocol `graspnet1b-sv-v1`: the whole test split (90
scenes, 23,040 frames, all scored). Inputs follow the reference: the workspace crop is
derived from the ground-truth segmentation at test time (`gt_assisted: workspace`).

| Config | Camera | Collision filter | Provenance | Seen | Similar | Novel |
|---|---|:---:|---|---|---|---|
| [gsnet_sunet14_g1b-realsense_10e](gsnet_sunet14_g1b-realsense_10e.yaml) | RealSense | off | reported (paper) | 65.70 | 53.75 | 23.98 |
| | RealSense | on | reported (paper) | 67.12 | 54.81 | 24.31 |
| | RealSense | off | reproduced ([record](records/gsnet_sunet14_g1b-realsense_10e_reproduced_test_raw.json)) | 68.83 | 60.35 | 25.20 |
| | RealSense | on | reproduced ([record](records/gsnet_sunet14_g1b-realsense_10e_reproduced_test_cd.json)) | 71.23 | 62.29 | 26.42 |

`reproduced` is the first model trained from scratch by this toolbox. It was trained with
`grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e` at commit `0125baa`, the recipe of
the config unchanged: 10 epochs over the 25,600 RealSense frames of the training scenes,
batch size 4, Adam at 1e-3 decayed by 0.95 per epoch, seed 0, one GPU, about 3.4 hours.
The checkpoint of epoch 10 (SHA-256 `829ad72d130d…`, in full in the metafile) was scored
through `grasppanda test` at commit `4bc5333`: `off` is the prediction as emitted (variant
`raw`), `on` adds the reference's model-free collision filter (variant `cd`). The linked
result records hold the per-scene values, the config hash and the hashes of code and
weights.

**Measured minus reported**, seen / similar / novel: +3.13 / +6.60 / +1.22 AP without the
collision filter, +4.11 / +7.48 / +2.11 AP with it. The model trained here is above the
paper on every split, by more than the 1 AP of the zoo's admission gate, most on similar,
and the gap is not explained. What is known: this implementation equals the public
reference implementation stage by stage, including every loss term of a recorded training
batch (see the equivalence section above), so its training step is the reference's. The
checkpoint of that section, trained independently with the reference code (on a modified
graspness label set, for 18 epochs), is above the paper as well, at 69.79 / 60.83 / 25.52
without the filter. The paper's numbers come from the authors' original code, which is not
public, so what separates the two cannot be checked from here. This is one training run
with one seed; a training with the unmodified reference code was not run here.

**The weights are not published yet.** They were trained on GraspNet-1Billion, whose
licence is CC BY-NC-SA 4.0, and whether and where they are published has not been
decided. The metafile lists them with their SHA-256 and `url: null`, so
`grasppanda demo gsnet/gsnet_sunet14_g1b-realsense_10e` without `--weights` says that they
are not published and downloads nothing. To get a model, train one with the commands
under [Usage](#usage).

### On GraspClutter6D

The same model on the whole GraspClutter6D test split (235 scenes, 3,055 frames of the
RealSense D435, all scored), protocol `graspclutter6d-sv-v1`, config
[gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e](gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml),
scored at commit `4bc5333`. Percent.

| Collision filter | AP | AP0.8 | AP0.4 | Provenance |
|---|---|---|---|---|
| off | 25.58 | 31.38 | 19.42 | reproduced ([record](records/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_raw.json)) |
| on | 26.87 | 33.00 | 20.04 | reproduced ([record](records/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_cd.json)) |

The GraspClutter6D paper reports no GSNet number, so there is nothing to compare with.
The input is cropped to the image rectangle around the ground-truth segmentation, as the
official tool does (`gt_assisted: workspace`); this is not the crop the weights were
trained with. What else changes with the dataset is described in
`docs/user_guides/cross_dataset.md`.

## The checkpoints released with the reference code

The reference repository links two files, `minkuresunet_realsense.tar` and
`minkuresunet_kinect.tar`; its README states no AP for them and recommends the RealSense
one for real-world use. The RealSense file (184,429,769 bytes, SHA-256 `a0b9d85b300e…`)
loads strictly into this model and `tools/convert_checkpoint.py` converts it, but it does
not give the numbers of the paper, neither here nor in the reference code. What was
measured on 2026-10-13 with the reference code at `b5abf5a` and the functions of the
official evaluator, on 16 frames of test scene 100 (a subset, not a benchmark number),
beside a model that was trained with the same reference code:

| | released file | trained with the reference code |
|---|---|---|
| AP, no collision filter | 9.8 | 59.0 |
| AP, with the reference's collision detector | 23.4 | 61.4 |
| AP if the evaluator did not reject colliding grasps (a diagnostic) | 63.1 | 74.0 |
| best 50 grasps rejected by the evaluator for collision | 93.5 % | 25.4 % |
| predicted graspness against the reference's label, correlation on 16 training frames | 0.45 | 0.98 |
| the same against a label computed without the scene-collision mask | 0.93 | 0.46 |

The archive is sound: epoch 10 of the default schedule, every tensor present, nothing
non-finite; the input pipeline fits it as well as it fits the other model, and its
objectness is as accurate (0.94). Its graspness, view and score outputs follow labels
that ignore collisions with the scene. Our reading is that the released file was trained
without the scene-collision labels, so that its best-ranked grasps run into the table and
into neighbouring objects. The authors have not confirmed this, and neither the Kinect
file nor a fresh download of the RealSense file could be fetched to check them. A user
of another implementation reports the same low score for the released Kinect file
([rhett-chen/graspness_implementation#38](https://github.com/rhett-chen/graspness_implementation/issues/38)).

What follows from it: the released file is no reference for the benchmark and has no row
in the table above. On a robot it should run with a collision filter
(`--post collision`), as the reference's own demo script does. Weights at the level of
the paper come from training, which applies the collision labels by default.

To run it anyway, convert it as below and pass `--collision-filter` (the config's `cd`
variant); [docs/methods/gsnet.md](../../docs/methods/gsnet.md) has every step, with the
AP it gets on 120 frames of the benchmark:

```bash
python -m tools.convert_checkpoint minkuresunet_realsense.tar gsnet_realsense.pth \
    --source-format graspness-unofficial --camera realsense
```

## Options of every slot

Every slot of this method, the options that fit it and those that do not, each with what
was measured for it (the verdict of its probe, the levels *composes*, *runs*, *learns* and
*predicts*, its class, the grasps per frame, the reason and kind of a refusal):
[docs/methods/gsnet.md](../../docs/methods/gsnet.md), or in a terminal:

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --list            # every slot
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --list grouper    # one slot (--json: everything)
```

A class counts verdicts of one run (seed 0, 20 iterations, 8 frames of one scene), not
accuracy.

## Usage

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense
grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e
grasppanda test  gsnet/gsnet_sunet14_g1b-realsense_10e --weights <checkpoint>
```

## Notes

- The reference implementation is third-party; no official training code exists.
- The reference test script enables its collision filter by default, while its
  published numbers are without it. Both variants are always evaluated here.
- Point-wise graspness labels must come from `grasppanda prepare`. Label files produced
  by other generators are row-aligned to a preprocessing they do not record.

## Citation

```bibtex
@inproceedings{wang2021graspness,
  title     = {Graspness Discovery in Clutters for Fast and Accurate Grasp Detection},
  author    = {Wang, Chenxi and Fang, Hao-Shu and Gou, Minghao and Fang, Hongjie and Gao, Jin and Lu, Cewu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```
