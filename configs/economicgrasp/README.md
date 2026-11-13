# EconomicGrasp

> [An Economic Framework for 6-DoF Grasp Detection](https://arxiv.org/abs/2407.08366)
> (ECCV 2024). Reference code:
> [iSEE-Laboratory/EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) at
> `7361bb6` (MIT; the repository vendors operator code derived from a non-commercial project).

A seed-based detector (a sparse U-Net encoder, graspness-guided seeds, 300 template views)
that changes how grasps are supervised and predicted:

- **Economic supervision.** The dense label grid (300 views × 12 angles × 4 depths per
  annotated point) is collapsed, once per scene, to the best grasp of every point and
  view. A seed is supervised with one grasp instead of 48, and a training batch carries
  megabytes of labels instead of gigabytes.
- **Interaction.** The neighbours of a cylinder crop attend to each other before they
  are pooled, and the head splits the crop feature into four task tokens (angle, depth,
  width, quality) that attend to each other before each predicts its quantity.
- **Classification.** Angle, depth and quality are classified; only the width is
  regressed.

**Run it in five minutes:** [the method page](https://daeda1used.github.io/GraspPanda/methods/economicgrasp/#run-in-five-minutes): the weights, grasps on the example scene, the same from Python and in the browser, and a first number on the benchmark, each with what it printed when it was measured.

## Configs

| Config | Trains on | Tests on |
|---|---|---|
| [economicgrasp_sunet14_g1b-realsense_10e](economicgrasp_sunet14_g1b-realsense_10e.yaml) | GraspNet-1Billion, RealSense | GraspNet-1Billion, RealSense |
| [economicgrasp_sunet14_g1b-kinect_10e](economicgrasp_sunet14_g1b-kinect_10e.yaml) | GraspNet-1Billion, Kinect | GraspNet-1Billion, Kinect |
| [economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e](economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml) | GraspNet-1Billion, RealSense | GraspClutter6D, RealSense D435 |
| [economicgrasp_sunet14_g1b-kinect-to-gc6d-d435_10e](economicgrasp_sunet14_g1b-kinect-to-gc6d-d435_10e.yaml) | GraspNet-1Billion, Kinect | GraspClutter6D, RealSense D435 |
| [economicgrasp_sunet14_gc6d-d435_10e](economicgrasp_sunet14_gc6d-d435_10e.yaml) | GraspClutter6D, RealSense D435 | GraspClutter6D, RealSense D435 |

The options of every slot, with what was measured for each:
[the method page](https://daeda1used.github.io/GraspPanda/methods/economicgrasp/#the-slots), or in a
terminal `grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e --list`
(one slot: `--list grouper`).

## Results on GraspNet-1Billion

AP on seen / similar / novel over the whole test split (90 scenes, 23,040 frames),
protocol `graspnet1b-sv-v1`. The workspace is cropped with the ground-truth
segmentation, as in the reference (`gt_assisted: workspace`).

| Config | Camera | Collision filter | Seen | Similar | Novel | Provenance |
|---|---|---|---|---|---|---|
| [economicgrasp_sunet14_g1b-realsense_10e](economicgrasp_sunet14_g1b-realsense_10e.yaml) | RealSense | off | 68.21 | 61.19 | 25.48 | reported |
| | RealSense | off | 69.28 | 61.50 | 25.17 | author-ckpt ([record](records/economicgrasp_sunet14_g1b-realsense_10e_author-ckpt_test_raw.json)) |
| | RealSense | on | 70.83 | 62.56 | 25.80 | author-ckpt ([record](records/economicgrasp_sunet14_g1b-realsense_10e_author-ckpt_test_cd.json)) |
| | RealSense | off | 69.34 | 61.41 | 26.21 | reproduced ([record](records/economicgrasp_sunet14_g1b-realsense_10e_reproduced_test_raw.json)) |
| | RealSense | on | 71.05 | 62.54 | 26.81 | reproduced ([record](records/economicgrasp_sunet14_g1b-realsense_10e_reproduced_test_cd.json)) |
| [economicgrasp_sunet14_g1b-kinect_10e](economicgrasp_sunet14_g1b-kinect_10e.yaml) | Kinect | off | 62.59 | 51.73 | 19.54 | reported |
| | Kinect | off | 63.71 | 52.28 | 19.73 | author-ckpt ([record](records/economicgrasp_sunet14_g1b-kinect_10e_author-ckpt_test_raw.json)) |
| | Kinect | on | 65.47 | 53.64 | 20.36 | author-ckpt ([record](records/economicgrasp_sunet14_g1b-kinect_10e_author-ckpt_test_cd.json)) |

`author-ckpt` is the authors' checkpoint of upstream release `v1` for each camera,
converted and run through `grasppanda test`, one run each with seed 0. **Measured minus
reported, filter off**: Kinect +1.12 / +0.55 / +0.19 AP, RealSense +1.07 / +0.31 / −0.31
AP: both are about 1.1 AP above the published table on seen, unexplained.

`reproduced` is trained here from scratch with the config's recipe. **It lands on the
authors' checkpoint**: +0.06 / −0.09 / +1.04 AP from it (filter off) and
+0.22 / −0.02 / +1.01 (on). Trained again from the same config and seed, it moved by up to
1.44 AP: the +1.04 on novel is spread, not gain. From the table: +1.13 / +0.22 / +0.73, on
seen 0.13 AP outside the zoo's 1 AP gate, like the checkpoint.

<details markdown="1"><summary>How it was trained, trained twice, and what is known about the gap</summary>

**The training.** `grasppanda train economicgrasp/economicgrasp_sunet14_g1b-realsense_10e`
at commit `7ac91b5`, nothing overridden: 10 epochs, batch size 4, seed 0, 17.0 hours on one
RTX 3090 shared with other jobs. The checkpoint of epoch 10 (SHA-256 `22f17fd571a2…`) was
scored at commit `400b733`.

**Trained twice.** The same config and seed were trained again on an RTX 3090, and the
checkpoint of epoch 10 was scored on the whole test split:

| Training | Collision filter off | on |
|---|---|---|
| first, the rows above | 69.34 / 61.41 / 26.21 | 71.05 / 62.54 / 26.81 |
| second: trained and scored at `7b3ddd4`, `c1c760135460…` ([off](repeats/economicgrasp_sunet14_g1b-realsense_10e_run2_test_raw.json), [on](repeats/economicgrasp_sunet14_g1b-realsense_10e_run2_test_cd.json)) | 69.01 / 60.02 / 25.09 | 70.65 / 61.10 / 25.74 |

Second minus first: −0.33 / −1.39 / −1.12 AP without the filter, −0.40 / −1.44 / −1.07
with it. One config and one seed gave two models, as for GSNet, and on similar and novel
they lie further apart than GSNet's three (0.36 / 0.37 AP). From the authors' checkpoint
the second is −0.27 / −1.48 / −0.08 AP without the filter: the two trainings lie on either
side of it on novel, and both on or below it on similar. The training loss of the two over
the last epoch is 2.98 and 2.98. Other seeds were not trained.

**The gap to the published table.** The checkpoints were scored at commit `ccb8704`
(Kinect) and `4bc5333` (RealSense).

The `reported` numbers are those of the upstream README. Neither it nor the paper says
whether a collision filter was applied; the shipped test command leaves it off, and the
table above takes that reading. No number with the filter is reported.

- It is not the port, as far as the Kinect checkpoint shows: on the frame sample under
  *Equivalence with the reference code*, the upstream code at `7361bb6` and GraspPanda
  agree with the same checkpoint (64.30 and 64.24 on seen when fed the same points), and
  both lie above the published 62.59. The upstream code has not been run with the
  RealSense checkpoint.
- It is not the friction levels: the paper describes AP as an average over µ from 0.2 to
  1.0, but averaging the five levels up to 1.0 instead of the benchmark's six gives
  60.90 / 49.34 / 17.23 (Kinect) and 66.57 / 58.50 / 21.81 (RealSense) from the same
  records, further from the reported values, not closer.
- It is not the released RealSense file: the model trained here is above the table by as
  much on seen (+1.13) as that checkpoint (+1.07).
- Not investigated: whether the released checkpoints are the ones behind the published
  table, and how much a number moves with the seed of the point sampling.

</details>

## Results on GraspClutter6D

The RealSense weights on the whole GraspClutter6D test split (235 scenes, 3,055 frames
of the RealSense D435), protocol `graspclutter6d-sv-v1`, config
[economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e](economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml).
Percent.

| Collision filter | AP | AP0.8 | AP0.4 | Provenance |
|---|---|---|---|---|
| not stated | 21.67 | 26.62 | 15.90 | reported |
| off | 21.69 | 26.64 | 15.79 | author-ckpt ([record](records/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_author-ckpt_test_raw.json)) |
| on | 22.25 | 27.30 | 16.06 | author-ckpt ([record](records/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_author-ckpt_test_cd.json)) |
| off | 23.24 | 28.58 | 17.04 | reproduced ([record](records/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_raw.json)) |
| on | 23.72 | 29.16 | 17.20 | reproduced ([record](records/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_cd.json)) |

The authors' checkpoint matches the reported row unfiltered (0.11 AP); the model trained
here is 1.5 AP above it.

<details markdown="1"><summary>Against the reported row</summary>

`reported` is Table IX of the GraspClutter6D paper (arXiv 2504.06866 v3), which states
neither the collision filter nor which checkpoint was scored. Measured minus reported
(AP / AP0.8 / AP0.4): +0.02 / +0.02 / −0.11 without the filter, +0.58 / +0.68 / +0.16
with it. Whether the paper ran without a filter was not investigated and cannot be read
from it. The input is cropped to the image rectangle around the ground-truth
segmentation, as the official tool does (`gt_assisted: workspace`); this is not the crop
the weights were trained with. What else changes with the dataset is described in
`docs/user_guides/cross_dataset.md`.

</details>

## Weights

GraspPanda publishes no weights for this method yet; for those trained here that is not
decided (the training data is CC BY-NC-SA 4.0). Download the authors' files of upstream
release `v1` (links and SHA-256 in the [metafile](metafile.yaml)) and convert them once:

```bash
python -m tools.convert_checkpoint economicgrasp_realsense.tar economicgrasp_realsense.pth \
    --source-format economicgrasp --camera realsense
python -m tools.convert_checkpoint economicgrasp_kinect.tar economicgrasp_kinect.pth \
    --source-format economicgrasp --camera kinect
```

## Train and test

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera kinect
grasppanda prepare graspnet1b --compact-labels \
    --config economicgrasp/economicgrasp_sunet14_g1b-kinect_10e
grasppanda train economicgrasp/economicgrasp_sunet14_g1b-kinect_10e
grasppanda test  economicgrasp/economicgrasp_sunet14_g1b-kinect_10e --weights <checkpoint>
```

The compact labels are one asset for both cameras, 1.2 GB, generated in about four
minutes; training no longer reads the dense store.

## Notes

<details markdown="1"><summary>How it is built</summary>

| Slot | Component |
|---|---|
| `backbone` | `SparseUNet14` |
| `proposal` | `GraspnessFPS` |
| `view` | `TemplateViewStage` (it reads its targets from the compact labels) |
| `assigner` | `CompactLabelAssigner` |
| `grouper` | `CylinderInteractionGroup` |
| `head` | `EconomicHead` |

Which method ships what in each slot:
[Methods](https://daeda1used.github.io/GraspPanda/methods/#what-each-method-ships). The
implementation is clean-room: written from the paper and from the behaviour of the
upstream code, with no upstream source in it.

</details>

<details markdown="1"><summary>Equivalence with the reference code</summary>

Checked with the authors' Kinect checkpoint (`economicgrasp_kinect.tar` of upstream release
`v1`, epoch 10, sha256 `a47b717a…5804a`) against the upstream code at `7361bb6`, by
`tests/integration/test_economicgrasp_parity.py` and
`tests/integration/test_economicgrasp_labels.py`; parity measurements, not benchmark
results.

| What | Result |
|---|---|
| Checkpoint conversion | strict: 260 of 260 tensors, 15,750,238 parameters, the same per stage as upstream |
| Inference, 4 frames | identical voxels, candidates, seeds and views; activations within 2.7e-6 of the largest value of a stage; decoded grasps identical up to 1.5e-6 (score) |
| Inference, 32 further frames | identical seeds on 32 of 32; of 32,768 grasps, one angle class was decided by round-off (the two best logits of the reference were within 2e-5) |
| Training batch (4 frames) | activations within 2.3e-6, identical seeds; every loss term within 2e-8 relative and their sum within 6e-8, total gradient norm within 2.8e-6 |
| Compact labels, 2 scenes | every value equal to the output of the upstream generator (18 objects, 8,297 kept points) |
| Dataset reader, 4 frames | poses and per-object labels equal to those of the upstream loader |
| Collision filter of the `cd` variant | the same decisions as the upstream detector on the 4 frames |

The training comparison replays three things the reference draws at random or leaves
undefined: the sampled view of every seed, the dropped attention weights, and the choice
described under `assigner.shared_views` below.

**A sample of the benchmark.** Every 64th frame of every test scene: 120 frames per
split, Kinect, AP in percent, without / with the collision filter; 1.6 % of the protocol,
taken to see whether the checkpoint and the port are sound. It is **not** an
`author-ckpt` result.

| | Seen | Similar | Novel |
|---|---|---|---|
| Upstream code at `7361bb6`, its own point sampling | 64.30 / 65.91 | 52.00 / 53.66 | 19.30 / 19.67 |
| GraspPanda fed the same points | 64.24 / 65.86 | not run | not run |
| `grasppanda test`, its own point sampling | 64.60 / 66.37 | 52.35 / 54.20 | 19.72 / 20.42 |
| Reported, whole split, for orientation | 62.59 | 51.73 | 19.54 |

Fed the same points, GraspPanda selects the same seeds on all 120 frames and its grasps
agree with upstream's within 2e-5 in every parameter. Four frames are nevertheless scored
differently, by −6.0 to +0.3 AP: the collision and force-closure tests of the benchmark
are discontinuous in the grasp parameters, so a difference in the seventh digit can
change the verdict on a top-ranked grasp. Parity is therefore established on activations
and grasps; an AP comparison can only confirm it up to that noise.

Not verified: the upstream code with the RealSense checkpoint. The complete training run
is the `reproduced` row of *Results on GraspNet-1Billion*.

</details>

<details markdown="1"><summary>Reference behaviours kept as named options</summary>

Each of these changes numbers. The default reproduces upstream `7361bb6`; the alternative
is a controlled variant with its own experiment name, not a fix to apply silently.

| Option | Default | What upstream does |
|---|---|---|
| `view.train_selection` | `multinomial` | The training view of a seed is sampled from the predicted scores. |
| `assigner.max_match_distance` | `0.005` | A seed further than 5 mm from its label point is not supervised by the grasp losses (the view loss still uses it). |
| `assigner.shared_views` | `lowest` | The object pose maps the 300 template views onto the labelled views, and the map is not one-to-one: for six random poses, 44 to 64 template views shared a labelled view with a neighbour. Upstream supervises only one view of such a pair and picks it with an indexed assignment that has repeated indices, whose result is undefined on a GPU. In a recorded batch the lower view won for 737 of the 773 seeds concerned (95 %), and 14 of 4,096 seeds changed between two runs on the same input. `lowest` makes that a rule; `all` supervises both views. |
| `head.unmatched_class` | `true` | The angle and depth classifiers have 13 and 5 outputs. The extra class is the label of seeds without a label, which the loss mask excludes, so it is never a target. |
| `head.decode_argmax` | `all_logits` | At `7361bb6` the extra class takes part in the decoder's arg-max; upstream `4119bdc` (2026-04-12) restricts it to the real bins (`supervised`). With the Kinect checkpoint the extra class did not win once on the 396 test frames we looked at, so there the two decode alike. |
| `head.score_decode` | `logit_weighted` | The decoded score is `Σ class value × raw logit`, without a softmax: unbounded, and not an expectation. |
| `head.width_decode_gain` | `1.2` | The decoder opens the gripper 20 % wider than regressed; the loss target has no such factor. |
| `head.num_score` | `6` | Label qualities come in steps of 0.1; two neighbouring levels share a class, and "no usable grasp" shares class 0 with the worst level. |
| `grouper.centre_coordinates` | `true` | The crop feature is given the seed's coordinates in the camera frame. |
| `compact_labels.*` (dataset) | `0.4`, `0.6`, `0.1` | Annotated points without a collision-free grasp of friction ≤ 0.4 are dropped; view graspness counts grasps of friction ≤ 0.6; grasps wider than 0.1 m cannot be the kept grasp. |

</details>

<details markdown="1"><summary>How this port differs from the reference</summary>

- One model instance trains and predicts.
- A frame with fewer graspable candidates than seeds yields fewer grasps instead of
  repeats, and a frame with none yields an empty result.
- Compact labels are stored in view order. Upstream sorts the views of every point by
  view graspness and stores the permutation; the order of equal values in that sort is
  not defined, so its files cannot be reproduced bit for bit, while their content can.
- The schedule base trains 10 epochs, as the command of the upstream README does; the
  published Kinect checkpoint is stored as epoch 10. The default of the upstream training
  script is 20 (`_base_/schedules/economicgrasp_20e.yaml`); because the cosine spans the
  run, that is a different schedule.
- The `cd` variant uses the collision filter of the upstream test script, which takes the
  sparse-convolution voxel size (0.005 m); the other five methods' configs use 0.01 m.

</details>

## Citation

<details markdown="1"><summary>BibTeX</summary>

```bibtex
@inproceedings{wu2024economic,
  title     = {An Economic Framework for 6-DoF Grasp Detection},
  author    = {Wu, Xiao-Ming and Cai, Jia-Feng and Jiang, Jian-Jian and Zheng, Dian and
               Wei, Yi-Lin and Zheng, Wei-Shi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

</details>
