# EconomicGrasp

> [An Economic Framework for 6-DoF Grasp Detection](https://arxiv.org/abs/2407.08366)
> (ECCV 2024). Upstream: [iSEE-Laboratory/EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp),
> pinned at `7361bb6`.

EconomicGrasp keeps the scene encoder, the graspness proposal and the view stage of
[GSNet](../gsnet) and changes what comes after them:

- **Economic supervision.** The dense label grid (300 views × 12 angles × 4 depths per
  annotated point) is collapsed, once per scene, to the best grasp of every point and
  view. A seed is then supervised with one grasp instead of 48, and a training batch
  carries megabytes of labels instead of gigabytes.
- **Interaction.** The neighbours of a cylinder crop attend to each other before they
  are pooled, and the head splits the crop feature into four task tokens (angle, depth,
  width, quality) that attend to each other before each predicts its quantity.
- **Classification.** Angle, depth and quality are classified; only the width is
  regressed.

| slot | component | shared with GSNet |
|---|---|---|
| `backbone` | `SparseUNet14` | yes |
| `proposal` | `GraspnessFPS` | yes |
| `view` | `TemplateViewStage` | yes (it reads its targets from the compact labels) |
| `assigner` | `CompactLabelAssigner` | no |
| `grouper` | `CylinderInteractionGroup` | cylinder query and first shared MLP |
| `head` | `EconomicHead` | codec only |

The implementation is clean-room: written from the paper and from the behaviour of the
upstream code, with no upstream source in it.

## Results

GraspNet-1Billion, single view, protocol `graspnet1b-sv-v1`. AP in percent.

| Config | Camera | Collision filter | Seen | Similar | Novel | Provenance |
|---|---|---|---|---|---|---|
| [economicgrasp_sunet14_g1b-kinect_10e](economicgrasp_sunet14_g1b-kinect_10e.yaml) | Kinect | off | 62.59 | 51.73 | 19.54 | reported |
| | Kinect | off | 63.71 | 52.28 | 19.73 | author-ckpt ([record](records/economicgrasp_author_kinect_g1b-test_raw.json)) |
| | Kinect | on | 65.47 | 53.64 | 20.36 | author-ckpt ([record](records/economicgrasp_author_kinect_g1b-test_cd.json)) |
| [economicgrasp_sunet14_g1b-realsense_10e](economicgrasp_sunet14_g1b-realsense_10e.yaml) | RealSense | off | 68.21 | 61.19 | 25.48 | reported |

`reported` numbers are those of the upstream README. Neither it nor the paper says
whether a collision filter was applied; the shipped test command leaves it off, and the
table above takes that reading.

`author-ckpt` numbers are the authors' Kinect checkpoint, converted with
`tools/convert_checkpoint.py` and run through `grasppanda test` on the whole test split
(90 scenes, 23,040 frames, all scored) at commit `ccb8704`; the linked result records
hold the per-scene values, the config hash and the hashes of code and weights. The
workspace crop uses the ground-truth segmentation, as in the reference.

The seen split comes out 1.12 AP above the reported value, similar 0.55 and novel 0.19
above. The gap is not explained. What is known:

- It is not the port: on the frame sample below, the upstream code at `7361bb6` and
  GraspPanda agree with the same checkpoint (64.30 and 64.24 on seen when fed the same
  points), and both lie above the published 62.59.
- It is not the friction levels: the paper describes AP as an average over µ from 0.2 to
  1.0, but averaging the five levels up to 1.0 instead of the benchmark's six gives
  60.90 / 49.34 / 17.23 from the same record, further from the reported values, not
  closer.
- Not checked: whether the released checkpoint is the one behind the published table.

**No `reproduced` number exists yet**: nothing has been trained here. The RealSense
checkpoint has not been evaluated on the full protocol.

## Status of the verification

What has been checked, with the authors' Kinect checkpoint (`economicgrasp_kinect.tar`
of upstream release `v1`, epoch 10, sha256 `a47b717a…5804a`) against the upstream code
at `7361bb6`. The numbers come from `tests/integration/test_economicgrasp_parity.py` and
`tests/integration/test_economicgrasp_labels.py`; they are parity measurements, not
benchmark results.

| What | Result |
|---|---|
| Checkpoint conversion | strict: 260 of 260 tensors, 15 750 238 parameters, the same per stage as upstream |
| Inference, 4 frames | identical voxels, candidates, seeds and views; activations within 2.7e-6 of the largest value of a stage; decoded grasps identical up to 1.5e-6 (score) |
| Inference, 32 further frames | identical seeds on 32 of 32; of 32 768 grasps, one angle class was decided by round-off (the two best logits of the reference were within 2e-5) |
| Training batch (4 frames) | activations within 2.3e-6, identical seeds; every loss term within 2e-8 relative and their sum within 6e-8, total gradient norm within 2.8e-6 |
| Compact labels, 2 scenes | every value equal to the output of the upstream generator (18 objects, 8 297 kept points) |
| Dataset reader, 4 frames | poses and per-object labels equal to those of the upstream loader |
| Collision filter of the `cd` variant | the same decisions as the upstream detector on the 4 frames |

The training comparison replays three things the reference draws at random or leaves
undefined: the sampled view of every seed, the dropped attention weights, and the choice
described under `assigner.shared_views` below.

### A sample of the benchmark

Every 64th frame of every test scene: 120 frames per split, Kinect, AP in percent, without
/ with the collision filter. This is a sample of 1.6 % of the protocol, taken to see
whether the checkpoint and the port are sound. It is **not** an `author-ckpt` result.

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

Not verified: any complete training run (two epochs on eight frames ran through
`grasppanda train` as a smoke test) and AP of the RealSense checkpoint on the full test
splits. For GraspClutter6D see `docs/user_guides/cross_dataset.md`.

## Usage

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera kinect
grasppanda prepare graspnet1b --compact-labels \
    --config economicgrasp/economicgrasp_sunet14_g1b-kinect_10e
grasppanda train economicgrasp/economicgrasp_sunet14_g1b-kinect_10e
grasppanda test  economicgrasp/economicgrasp_sunet14_g1b-kinect_10e --weights <checkpoint>
```

The compact labels are one asset for both cameras: 1.2 GB for the 100 training scenes,
generated from the dense label store in about four minutes on the development machine;
39 % of the annotated points are kept. Once the asset exists the dense store is no
longer read during training.

To evaluate the authors' weights, convert them first:

```bash
python -m tools.convert_checkpoint economicgrasp_kinect.tar economicgrasp_kinect.pth \
    --source-format economicgrasp --camera kinect
```

## Reference behaviour kept as named options

Each of these changes numbers. The default reproduces upstream `7361bb6`; the
alternative is a controlled variant with its own experiment name, not a fix to apply
silently.

| Option | Default | What upstream does |
|---|---|---|
| `view.train_selection` | `multinomial` | The training view of a seed is sampled from the predicted scores. |
| `assigner.max_match_distance` | `0.005` | A seed further than 5 mm from its label point is not supervised by the grasp losses (the view loss still uses it). |
| `assigner.shared_views` | `lowest` | The object pose maps the 300 template views onto the labelled views, and the map is not one-to-one: for six random poses, 44 to 64 template views shared a labelled view with a neighbour. Upstream supervises only one view of such a pair and picks it with an indexed assignment that has repeated indices, whose result is undefined on a GPU. In a recorded batch the lower view won for 737 of the 773 seeds concerned (95 %), and 14 of 4 096 seeds changed between two runs on the same input. `lowest` makes that a rule; `all` supervises both views. |
| `head.unmatched_class` | `true` | The angle and depth classifiers have 13 and 5 outputs. The extra class is the label of seeds without a label, which the loss mask excludes, so it is never a target. |
| `head.decode_argmax` | `all_logits` | At `7361bb6` the extra class takes part in the decoder's arg-max; upstream `4119bdc` (2026-04-12) restricts it to the real bins (`supervised`). With the Kinect checkpoint the extra class did not win once on the 396 test frames we looked at, so there the two decode alike. |
| `head.score_decode` | `logit_weighted` | The decoded score is `Σ class value × raw logit`, without a softmax: unbounded, and not an expectation. |
| `head.width_decode_gain` | `1.2` | The decoder opens the gripper 20 % wider than regressed; the loss target has no such factor. |
| `head.num_score` | `6` | Label qualities come in steps of 0.1; two neighbouring levels share a class, and "no usable grasp" shares class 0 with the worst level. |
| `grouper.centre_coordinates` | `true` | The crop feature is given the seed's coordinates in the camera frame. |
| `compact_labels.*` (dataset) | `0.4`, `0.6`, `0.1` | Annotated points without a collision-free grasp of friction ≤ 0.4 are dropped; view graspness counts grasps of friction ≤ 0.6; grasps wider than 0.1 m cannot be the kept grasp. |

## Differences from the reference

- One model instance trains and predicts.
- A frame with fewer graspable candidates than seeds yields fewer grasps instead of
  repeats, and a frame with none yields an empty result.
- Compact labels are stored in view order. Upstream sorts the views of every point by
  view graspness and stores the permutation; the order of equal values in that sort is
  not defined, so its files cannot be reproduced bit for bit, while their content can.
- The schedule base trains 10 epochs, as the command of the upstream README does; the
  published Kinect checkpoint is stored as epoch 10. The default of the upstream
  training script is 20
  (`_base_/schedules/economicgrasp_20e.yaml`); because the cosine spans the run, that is
  a different schedule.
- The `cd` variant uses the collision filter of the upstream test script, which takes
  the sparse-convolution voxel size (0.005 m) rather than the 0.01 m of GSNet.

## Citation

```bibtex
@inproceedings{wu2024economic,
  title     = {An Economic Framework for 6-DoF Grasp Detection},
  author    = {Wu, Xiao-Ming and Cai, Jia-Feng and Jiang, Jian-Jian and Zheng, Dian and
               Wei, Yi-Lin and Zheng, Wei-Shi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
