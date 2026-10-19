# HGGD

> [Efficient Heatmap-Guided 6-DoF Grasp Detection in Cluttered Scenes](https://arxiv.org/abs/2403.18546)
> (RA-L 2023). Reference code: [THU-VCLab/HGGD](https://github.com/THU-VCLab/HGGD) at
> `69580ff` (MIT).

HGGD reads an RGB-D **image**, not only a cloud. A small encoder-decoder predicts a heat-map
of grasp centres together with the in-plane angle, width and depth offset at each centre.
Up to 48 peaks become *partial grasps*; each is lifted to 3-D, a ball of scene points is cut
around it, and a PointNet completes the grasp by classifying 7 × 7 learned rotation anchors
with a centre offset per anchor. It is a detector family of its own: it has none of the
seed-based slots, and shares the outer contracts (inputs, grasps, datasets, trainer,
post-processing, evaluator) with the other five methods.

**Run it in five minutes:** [the method page](https://daeda1used.github.io/GraspPanda/methods/hggd/#run-in-five-minutes): the weights, grasps on the example scene, the same from Python and in the browser, and a first number on the benchmark, each with what it printed when it was measured.

## Configs

| Config | Trains on | Tests on |
|---|---|---|
| [hggd_r34w8_g1b-realsense_15e](hggd_r34w8_g1b-realsense_15e.yaml) | GraspNet-1Billion, RealSense | GraspNet-1Billion, RealSense |
| [hggd_r34w8_g1b-kinect_15e](hggd_r34w8_g1b-kinect_15e.yaml) | GraspNet-1Billion, Kinect | GraspNet-1Billion, Kinect |
| [hggd_r34w8_g1b-realsense-to-gc6d-d435_15e](hggd_r34w8_g1b-realsense-to-gc6d-d435_15e.yaml) | GraspNet-1Billion, RealSense | GraspClutter6D, RealSense D435 |

The options of every slot, with what was measured for each:
[the method page](https://daeda1used.github.io/GraspPanda/methods/hggd/#the-slots), or in a terminal
`grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --list` (one slot:
`--list image_encoder`).

## Results on GraspNet-1Billion

AP on seen / similar / novel over the whole test split (90 scenes, 23,040 frames per
camera, all scored), protocol `graspnet1b-sv-v1`. Nothing in HGGD's input is derived from
ground truth (`gt_assisted` is empty in all four records).

| Config | Camera | Provenance | Collision filter | Seen | Similar | Novel |
|---|---|---|:---:|---:|---:|---:|
| [hggd_r34w8_g1b-realsense_15e](hggd_r34w8_g1b-realsense_15e.yaml) | RealSense | `reported` (paper; README row "In paper") | not stated | 59.36 | 51.20 | 22.17 |
| | | `reported` (README row "In repo") | not stated | 64.45 | 53.59 | 24.59 |
| | | `author-ckpt` ([record](records/hggd_r34w8_g1b-realsense_15e_author-ckpt_test_raw.json)) | off | 58.33 | 47.96 | 22.03 |
| | | `author-ckpt` ([record](records/hggd_r34w8_g1b-realsense_15e_author-ckpt_test_cd.json)) | on, then NMS | 63.70 | 52.90 | 24.18 |
| [hggd_r34w8_g1b-kinect_15e](hggd_r34w8_g1b-kinect_15e.yaml) | Kinect | `reported` (paper; README row "In paper") | not stated | 60.26 | 48.59 | 18.43 |
| | | `reported` (README row "In repo") | not stated | 61.17 | 47.02 | 19.37 |
| | | `author-ckpt` ([record](records/hggd_r34w8_g1b-kinect_15e_author-ckpt_test_raw.json)) | off | 56.98 | 43.97 | 18.63 |
| | | `author-ckpt` ([record](records/hggd_r34w8_g1b-kinect_15e_author-ckpt_test_cd.json)) | on, then NMS | 60.34 | 46.83 | 19.27 |

`author-ckpt` is the authors' checkpoint of each camera, converted and run through
`grasppanda test` at commit `4bc5333`, one run each with seed 0. `off` is every decoded
grasp (variant `raw`); `on, then NMS` is what the released test script dumps (variant
`cd`). No model has been trained here yet. Neither source states the collision filter;
the closest pair is `cd` with "In repo", within 0.83 AP on every split and below it on
all six.

<details markdown="1"><summary>Against the reported rows, and a 20-frame check with the reference code</summary>

Measured minus reported, on seen / similar / novel:

| Camera | Measured | minus "In paper" | minus "In repo" |
|---|---|---|---|
| RealSense | `cd` | +4.34 / +1.70 / +2.01 | −0.75 / −0.69 / −0.41 |
| RealSense | `raw` | −1.03 / −3.24 / −0.14 | −6.12 / −5.63 / −2.56 |
| Kinect | `cd` | +0.08 / −1.76 / +0.84 | −0.83 / −0.19 / −0.10 |
| Kinect | `raw` | −3.28 / −4.62 / +0.20 | −4.19 / −3.05 / −0.74 |

That fits a test script that dumps filtered grasps, but it does not establish how the
"In repo" row was produced. Why `cd` lies below it was not investigated. The network draws
random numbers at inference; the spread over seeds was measured on 20 frames only (0.4 to
0.7 AP standard deviation), not on a split.

**Is the released checkpoint healthy?** Yes. A partial check, *not* a headline number: 20
frames per row (every 64th frame of five scenes). "Reference" is the pristine upstream
test path (`tools/oracle/hggd_reference.py`); "here" is this implementation from the
dataset reader to the evaluator, with the converted checkpoint; both are scored by
`GraspNet1BEvaluator`. Where several runs were made, mean ± standard deviation over them.

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

On 20 frames a difference of half an AP point is within the spread of the draws; the two
implementations are compared rigorously stage by stage instead (*Equivalence with the
reference code*). With the reference's random draws replayed, predictions made here score
what the reference's own dump scores on the six recorded frames (35.56 against 35.48 `raw`
and 41.91 against 41.83 `cd` on four RealSense frames; 31.00 against 31.00 and 35.36
against 35.25 on two Kinect frames; `tests/integration/test_hggd_data.py`).

</details>

## Results on GraspClutter6D

The RealSense checkpoint on the whole GraspClutter6D test split (235 scenes, 3,055 frames
of the RealSense D435), protocol `graspclutter6d-sv-v1`, config
[hggd_r34w8_g1b-realsense-to-gc6d-d435_15e](hggd_r34w8_g1b-realsense-to-gc6d-d435_15e.yaml). Percent.

| Collision filter | AP | AP0.8 | AP0.4 | Provenance |
|---|---|---|---|---|
| off | 14.25 | 17.41 | 9.23 | author-ckpt ([record](records/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e_author-ckpt_test_raw.json)) |
| on, then NMS | 14.56 | 17.79 | 9.33 | author-ckpt ([record](records/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e_author-ckpt_test_cd.json)) |

No number is published for HGGD on this dataset, so there is nothing to compare with.

<details markdown="1"><summary>The input on this dataset</summary>

One run with seed 0. As on GraspNet-1Billion, nothing in the input is derived from ground
truth, whereas the seed-based methods are scored on this dataset with the workspace
cropped around the ground-truth segmentation: rows of the two kinds are not like-for-like.
What changes with the dataset is described in `docs/user_guides/cross_dataset.md`.

</details>

## Weights

GraspPanda publishes no weights for this method. Download `HGGD_realsense_checkpoint` and
`HGGD_kinect_checkpoint` from the authors' folder (link and SHA-256 in the
[metafile](metafile.yaml)): each holds two networks and the learned anchors. Convert them
once:

```bash
python -m tools.convert_checkpoint HGGD_realsense_checkpoint hggd_realsense.pth \
    --source-format hggd --camera realsense
python -m tools.convert_checkpoint HGGD_kinect_checkpoint hggd_kinect.pth \
    --source-format hggd --camera kinect
```

## Train and test

```bash
# Per-view grasp labels (a derived asset; nothing is written into the dataset).
grasppanda prepare graspnet1b --view-grasps --camera realsense \
    --config _base_/datasets/graspnet1b_realsense_hggd.yaml
grasppanda train hggd/hggd_r34w8_g1b-realsense_15e
grasppanda test  hggd/hggd_r34w8_g1b-realsense_15e --weights hggd_realsense.pth
grasppanda test  hggd/hggd_r34w8_g1b-kinect_15e --weights hggd_kinect.pth
```

Every run reports two variants. `raw` is every decoded grasp. `cd` is what the released
test script dumps: its collision filter, which also drops grasps with at most two points
between the fingers, followed by NMS.

## Notes

<details markdown="1"><summary>How it is built</summary>

In `grasppanda/models/detectors/hggd/`; there is no scene encoder, no graspness, no view
template and no approach-aligned crop.

| Stage | Module | Config section |
|---|---|---|
| network input `[depth, R, G, B]` at 640 × 360 | `image.py` | `model.image` |
| image network (ResNet-34 block layout at base width 8, from scratch) | `ImageNet` | `model.encoder` |
| heat-map → partial grasps, dense 2-D targets | `HeatmapStage` | `model.proposal` |
| lifting, image features on points, ball regions | `RegionStage` | `model.region` |
| region network (PointNet + partial-grasp info) | `LocalNet` | `model.local` |
| rotation anchors, decoding, matching, anchor shift | `AnchorHead` | `model.anchors` |

The image network is the slot `image_encoder`; which method ships what in each slot:
[Methods](https://daeda1used.github.io/GraspPanda/methods/#what-each-method-ships).

</details>

<details markdown="1"><summary>Equivalence with the reference code</summary>

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

</details>

<details markdown="1"><summary>Reference behaviours kept as named options</summary>

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

**The shell script and the argument defaults disagree.** The reference's
`train_graspnet.sh` / `test_graspnet.sh` and the defaults of its argument parser differ;
the configs follow the scripts: learning rate 1e-2 (default 3e-3); batch size ×
accumulation 4 × 2 (default 8 × 1); regression weight `proposal.offset_weight` 5 (default
1); offset weight `anchors.offset_weight` 1 (default 5); anchors decoded per region
`anchors.top_k` 10 (default 3).

</details>

<details markdown="1"><summary>How this port differs from the reference</summary>

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

</details>

## Citation

<details markdown="1"><summary>BibTeX</summary>

```bibtex
@article{chen2023efficient,
  title   = {Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes},
  author  = {Chen, Siang and Tang, Wei and Xie, Pengwei and Yang, Wenming and Wang, Guijin},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2023}
}
```

</details>
