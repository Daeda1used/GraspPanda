# HGGD

> [Efficient Heatmap-Guided 6-DoF Grasp Detection in Cluttered Scenes](https://arxiv.org/abs/2403.18546)
> (RA-L 2023). Reference implementation: [THU-VCLab/HGGD](https://github.com/THU-VCLab/HGGD),
> MIT, read at commit `69580ff`.

HGGD reads an RGB-D **image**, not only a cloud. A small encoder-decoder predicts a heat-map
of grasp centres together with the in-plane angle, width and depth offset at each centre.
Up to 48 peaks become *partial grasps*; each is lifted to 3-D, a ball of scene points is cut
around it, and a PointNet completes the grasp by classifying 7 × 7 learned rotation anchors
with a centre offset per anchor.

In this toolbox HGGD is a detector family of its own (`grasppanda/models/detectors/hggd/`).
It uses none of the seed-based slots: there is no scene backbone, no graspness, no view
template and no approach-aligned crop. It shares the outer contracts only: `loss` /
`predict`, `Observation` / `Supervision`, `GraspGroup`, the dataset reader, the trainer,
the post-processors and the evaluator.

| Stage | Module | Config section |
|---|---|---|
| network input `[depth, R, G, B]` at 640 × 360 | `image.py` | `model.image` |
| image network (ResNet-34 block layout at base width 8, from scratch) | `ImageNet` | `model.encoder` |
| heat-map → partial grasps, dense 2-D targets | `HeatmapStage` | `model.proposal` |
| lifting, image features on points, ball regions | `RegionStage` | `model.region` |
| region network (PointNet + partial-grasp info) | `LocalNet` | `model.local` |
| rotation anchors, decoding, matching, anchor shift | `AnchorHead` | `model.anchors` |

## Results

GraspNet-1Billion, AP on seen / similar / novel. Every number states where it comes from
and whether a collision filter was applied.

| Config | Camera | Provenance | Collision filter | Seen | Similar | Novel |
|---|---|---|:---:|---:|---:|---:|
| [hggd_r34w8_g1b-realsense_15e](hggd_r34w8_g1b-realsense_15e.yaml) | RealSense | `reported` (paper; README row "In paper") | not stated | 59.36 | 51.20 | 22.17 |
| | | `reported` (README row "In repo") | not stated | 64.45 | 53.59 | 24.59 |
| [hggd_r34w8_g1b-kinect_15e](hggd_r34w8_g1b-kinect_15e.yaml) | Kinect | `reported` (paper; README row "In paper") | not stated | 60.26 | 48.59 | 18.43 |
| | | `reported` (README row "In repo") | not stated | 61.17 | 47.02 | 19.37 |

There is no `author-ckpt` or `reproduced` row yet: neither a full-split evaluation of the
authors' checkpoints nor a training run has been done in this toolbox.

**Is the released checkpoint healthy?** Yes. A partial check, *not* a headline number: 20
frames per row (every 64th frame of five scenes; a benchmark split has 7 680). "Reference"
is the pristine upstream test path (`tools/oracle/hggd_reference.py`); "here" is this
implementation from the dataset reader to the evaluator, with the converted checkpoint.
Both are scored by `GraspNet1BEvaluator`. The reference draws random numbers at inference
(scene sub-sampling, region jitter, sampling start points), so two runs of either
implementation differ; where several runs were made the table gives mean ± standard
deviation over the runs.

| Checkpoint | Scenes | Variant | Reference | Here |
|---|---|---|---:|---:|
| RealSense | 100–104 (seen) | `cd` (the script's dump) | 66.26 ± 0.41 (5 runs) | 66.68 ± 0.64 (6 runs) |
| | | `raw` | 60.72 ± 0.62 (5 runs) | 61.40 ± 0.69 (6 runs) |
| | 130–134 (similar) | `cd` | 48.00 | 48.74 |
| | | `raw` | 43.48 | 44.50 |
| | 160–164 (novel) | `cd` | 21.96 | 22.67 |
| | | `raw` | 19.10 | 19.17 |
| Kinect | 100–104 (seen) | `cd` | 59.50 | 60.17 |
| | | `raw` | 55.38 | 55.53 |

The two implementations are compared rigorously stage by stage (below), not through this
table: on 20 frames a difference of half an AP point is within the spread of the draws.
With the reference's random draws replayed, predictions made here score what the
reference's own dump scores on the six recorded frames (35.56 against 35.48 `raw` and
41.91 against 41.83 `cd` on four RealSense frames; 31.00 against 31.00 and 35.36 against
35.25 on two Kinect frames; `tests/integration/test_hggd_data.py`).

## Use

```bash
# Per-view grasp labels (a derived asset; nothing is written into the dataset).
grasppanda prepare graspnet1b --view-grasps --camera realsense \
    --config _base_/datasets/graspnet1b_realsense_hggd.yaml

grasppanda train hggd/hggd_r34w8_g1b-realsense_15e
grasppanda test  hggd/hggd_r34w8_g1b-realsense_15e --weights <checkpoint>

# The authors' checkpoint (two networks and the learned anchors) for this config:
python -m tools.convert_checkpoint checkpoint-realsense.tar hggd_realsense.pth \
    --source-format hggd --camera realsense
grasppanda test hggd/hggd_r34w8_g1b-realsense_15e --weights hggd_realsense.pth
```

Every run reports two variants. `raw` is every decoded grasp. `cd` is what the released
test script dumps: its collision filter, which also drops grasps with at most two points
between the fingers, followed by NMS. HGGD crops no workspace, so nothing in its input is
derived from ground truth and its results are not `gt_assisted`.

## What is reproduced, and how it was checked

`tools/make_hggd_fixtures.py` records every stage of the reference with the authors'
checkpoints (four RealSense and two Kinect test frames, one training batch per camera);
`tests/integration/test_hggd_parity.py` compares stage by stage and writes
`parity_report_<camera>.json` next to the fixtures.

| Stage | Result on the recorded frames (both cameras together) |
|---|---|
| Checkpoint conversion | strict and complete: 375 tensors; 1 618 921 + 1 806 669 parameters, as the reference; anchors included |
| Network input from the dataset reader | colour identical, depth within 5e-7 |
| Image network, blurred heat-map | within 2e-6 of the largest value |
| Candidate pixels, partial grasps | identical pixels (2 880 of 2 880 candidates, 288 of 288 grasps); angle, width and offsets within 2e-6 |
| Image features on points | identical for all but 20 of 153 600 points (a near-tie for the 8th neighbour) |
| Lifted centres | within 1e-7 m, including 13 centres on pixels without depth |
| Regions, with the reference's random draws replayed | the same points in 1 294 of 1 295 regions, the same sample order in 1 292 |
| Region network | within 1e-6 of the largest value |
| Decoding | equal to 1e-15 in double precision |
| Collision filter | 32 680 of 32 681 voxels identical after half-float rounding; 2 603 of 2 603 decisions identical |
| NMS and the final dump | identical |
| Dense 2-D targets | heat-map, label counts and width offsets identical; angle and depth offsets within 4e-6 |
| Label matching, anchor targets | the same 2 957 matches; targets within 4e-6; the same positive anchors |
| Loss terms of both networks | within 2e-7; total loss within 4e-6 |
| Gradients (six tensors) | within 8e-3 of each tensor's largest entry |
| Anchor shift | within 3e-16 |

Not reproducible bit for bit, by construction: the reference draws random numbers *at
inference* (scene sub-sampling, a 0.5 mm jitter of region points, the start point of the
region sampling). Here they come from PyTorch's generators and are reproducible under a
seed; for comparisons the recorded draws are replayed.

### Reference behaviours kept as named options

Each default is the reference value; the alternative is a controlled variant.

| Option | Reference behaviour |
|---|---|
| `image.resize: pil_bicubic` | Pillow's resize, which rounds colour to 8 bits after each of its two passes. |
| `encoder.detach_features: true` | The region loss does not reach the image network ("joint training" sums two losses). |
| `proposal.train_score_threshold: 0.0` | No heat threshold while training; 128 centres instead of 48. |
| `proposal.positive_count_threshold: 0.5` | A cell with a single label is a *negative* of its angle class, yet its offsets are supervised. |
| `region.jitter: 0.0005` | Region points are jittered at inference too. |
| `region.zero_depth_centres: append` | Centres of pixels without depth move to the end of the centre list while the grasp attributes keep their order. |
| `anchors.frame: pixel_keypoints` | The in-plane axes are orthogonal only up to `\|fx² − fy²\| / (fx² + fy²)` (2e-4 RealSense, 5e-4 Kinect). |
| `anchors.max_gamma: 60` | With the released RealSense anchors the steepest `gamma` anchor (−61°) is never decoded. |
| collision filter `precision: float16`, `cloud: points` | The filter works in half floats on the 25 600 sampled points, voxelised with cupoch's single-precision arithmetic. |

Also as in the reference: the last row and column of the heat-map are never searched for
a centre; the width offset is clamped to ±0.5 in the loss and to ±1 in the decoder; the
bin mean of the anchor update divides by the count plus one.

### The shell script and the argument defaults disagree

The reference's `train_graspnet.sh` / `test_graspnet.sh` and the defaults in its argument
parser differ. The configs follow the scripts:

| Setting | Script (used) | Argument default |
|---|---|---|
| learning rate | 1e-2 | 3e-3 |
| batch size × accumulation | 4 × 2 | 8 × 1 |
| regression weight (`proposal.offset_weight`) | 5 | 1 |
| offset weight (`anchors.offset_weight`) | 1 | 5 |
| anchors decoded per region (`anchors.top_k`) | 10 | 3 |

### Deliberate differences

- One model instance trains and predicts; the mode selects the proposal budget.
- Camera intrinsics come from the frame, not from per-camera constants rounded to two
  decimals.
- A frame without proposals, or whose regions all lack support, yields an explicit empty
  group (the reference writes no file, or fails). A training batch without regions still
  trains the image network (the reference skips the step).
- Gradient accumulation averages the two batches where the reference sums them, so the
  value clip is 0.5 instead of 1 (AdamW is invariant to the scale). The reference's first
  step of an epoch spans three batches and it drops the gradient of the last batch;
  neither is reproduced.
- In-training evaluation uses the `val_dev` scenes, not test scene 100.
- The anchor label pool is per process: under `torchrun` each process moves its own copy
  of the anchors. Train the reference recipe on one GPU.

## Citation

```bibtex
@article{chen2023efficient,
  title   = {Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes},
  author  = {Chen, Siang and Tang, Wei and Xie, Pengwei and Yang, Wenming and Wang, Guijin},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2023}
}
```
