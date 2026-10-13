# GraspNet-baseline

> [GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)
> (CVPR 2020). Reference implementation:
> [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) @ `fb2f089`
> (non-commercial research licence).

The baseline that came with the dataset: a PointNet++ encoder keeps 1024 seeds, one head
scores objectness and 300 approach views per seed, a cylinder crop of the scene cloud is
taken at four insertion depths along the best view, and two small networks predict score,
in-plane angle class, width and grasp tolerance per depth.

## Status

**Support level: author weights verified, not trained here yet.** The implementation is
clean-room (written from the paper and the behaviour of the reference code). With the
authors' RealSense checkpoint converted by `tools/convert_checkpoint.py` it reproduces the
reference stage by stage; see [Verification](#verification). No model has been trained
with this code yet, so the table below has no `reproduced` row and the method is not in
the model zoo.

## Model

`configs/_base_/models/graspnet_baseline.yaml`, slot by slot:

| Slot | Component | Contract |
|---|---|---|
| `backbone` | `PointNet2` (4 set-abstraction, 2 propagation levels) | `output = subsampled`: 1024 seeds, 256 channels |
| `proposal` | `SubsampledSeeds` | `requires = subsampled`; every kept point is a seed, nothing is selected |
| `view` | `ObjectnessViewStage` | 2 objectness logits + 300 view scores from one head; arg-max view in training and inference |
| `assigner` | `NearestLabelAssigner` | crop centre `label_point`, supervision `object_seeds`, widths of all views |
| `grouper` | `DepthCylinderGroup` | `support = scene`, axes `(seed, depth, channel)`; radius 0.05 m, far ends 0.01–0.04 m, 64 points |
| `head` | `OperationHead` | expects `(seed, depth, channel)`; score, angle class, width, tolerance over 12 angles × 4 depths |

The contracts are what keeps these parts from being mixed with GSNet's by accident:
`GraspnessFPS` needs dense features and refuses a sub-sampling backbone; `SWADHead` has no
depth axis and refuses this grouper; `OperationHead` refuses `CylinderGroup`. Each of
these is a config error before anything is built.

Objectness is predicted by the view stage, not by a proposal: the reference computes it
with the view scores from one shared trunk and one output layer, and never selects seeds
with it. It masks losses and, at inference, drops seeds classified as background.

## Reference behaviours kept as named options

| Option | Reference value | What it does |
|---|---|---|
| `assigner.crop_centre` | `label_point` | Training crops are centred at the matched label point, inference crops at the seed. |
| `assigner.supervise` | `object_seeds` | View and grasp losses only for seeds whose scene point is labelled as object. |
| `view.score_target`, `head.score_target` | `log_friction_batch_max` | Score target `log(μ_worst / μ)` with `μ_worst` the largest label value in the batch. |
| `head.min_score_target` | `0.1` | A (seed, depth) is supervised only if its best score target exceeds it. |
| `head.width_decode_gain` | `1.2` | Widths are widened in the decoder only. |
| `head.score_decode` | `score_x_tolerance` | Emitted score = regressed score × predicted tolerance / 0.05. |
| `view.reject_background` | `true` | No grasp for a seed whose objectness arg-max is background. |
| `backbone.fps_skip_near_origin` | `true` | The sampling kernel never picks points within 3.2 cm of the camera origin. |
| `backbone.fps_tie_break` | `pointnet2_cuda` | Equally distant candidates are sampled in the order of that kernel (the component default is `lowest_index`). |

Read from the reference and **not** found there, although often attributed to it: the
objectness loss is a plain cross-entropy against the segmentation label of the seed (the
"more than 10 graspable views" rule belongs to Scale-Balanced-Grasp), and the 0.7
threshold is used for a logged count only.

Deliberate differences: one model instance trains and predicts; losses are means over
the supervised entries (the reference adds 1e-6 to the count) and zero, not NaN, when
nothing is supervised; validation uses the `val_dev` scenes, not `test_seen`.

## Data

`configs/_base_/datasets/graspnet1b_realsense_tolerance.yaml`: 20 000 points, workspace
crop from the ground-truth segmentation (results are stamped `gt_assisted: [workspace]`),
label points drawn from the visible side of each object (within 1 cm of its visible
points), object 18 never supervised, flip and ±30° rotation about the camera x-axis.

Grasp tolerance is a derived label. It is generated into the asset store, value for value
equal to the reference generator's output:

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --tolerance \
    --config graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e
```

On the development machine 23 of the 40 objects of the training scenes took 54 s with
four workers.

## Use

```bash
# the authors' checkpoint
python -m tools.convert_checkpoint checkpoint-rs.tar baseline_realsense.pth \
    --source-format graspnet-baseline --camera realsense
grasppanda test graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --weights baseline_realsense.pth

# training with the reference recipe: Adam 1e-3, x0.1 before epochs 8 / 12 / 16, 18 epochs,
# batch 2, batch-norm momentum 0.5 halved every 2 epochs
grasppanda train graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e
```

Both evaluation variants are always reported: `raw` (predictions as emitted) and `cd`
(the model-free collision filter the reference test script applies by default).

## Results

GraspNet-1Billion, RealSense, protocol `graspnet1b-sv-v1`: the whole test split (90
scenes, 23,040 frames, all scored). AP on seen / similar / novel.

| Provenance | Collision filter | AP | Source |
|---|---|---|---|
| `reported` | not stated | 27.56 / 26.11 / 10.55 | paper, as quoted in the reference README ("In paper") |
| `author-ckpt` | off | 32.64 / 27.48 / 10.42 | [record](records/graspnet-baseline_pointnet2_g1b-realsense_18e_author-ckpt_test_raw.json) |
| `reported` | on | 47.47 / 42.27 / 16.61 | reference README, "In repo" |
| `author-ckpt` | on | 47.90 / 42.70 / 16.98 | [record](records/graspnet-baseline_pointnet2_g1b-realsense_18e_author-ckpt_test_cd.json) |
| `reproduced` | — | not trained here yet | |

`author-ckpt` is the authors' `checkpoint-rs.tar`, converted with
`tools/convert_checkpoint.py` and run through `grasppanda test` at commit `4bc5333`, one
run with seed 0. The workspace crop uses the ground-truth segmentation, as in the
reference (`gt_assisted: workspace`).

**Measured minus reported, with the collision filter**: +0.43 / +0.43 / +0.37 AP against
the "In repo" row, which the reference README describes as the model with collision
detection at threshold 0.01, the setting of the `cd` variant. The released checkpoint
scores slightly above its repository's table on every split. Why was not investigated;
each number is one run with seed 0, and the spread over the seed of the point sampling
was not measured.

**Without the collision filter there is no like-for-like reported number.** The "In
paper" row does not state the filter, and the reference README does not say whether it
belongs to the released checkpoint. The unfiltered run lies +5.08 / +1.37 / −0.13 AP from
it; that difference was not investigated.

What was measured with the pristine reference code is a **20-frame subset**, not a
split: scenes 100–104, frames 0 / 64 / 128 / 192, scored by `GraspNet1BEvaluator`.

| Predictions from | Collision filter | AP on the 20 frames |
|---|---|---|
| pristine reference code | off | 30.69 |
| pristine reference code | on (reference filter) | 46.37 |
| GraspPanda, converted weights, same input points | off | 30.69 |

Fed the same points, the reference and this implementation score the same on these
frames. The subset is not a benchmark number and must not be quoted as one.

### On GraspClutter6D

The same checkpoint on the whole GraspClutter6D test split (235 scenes, 3,055 frames of
the RealSense D435, all scored), protocol `graspclutter6d-sv-v1`, config
[graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e](graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e.yaml). Percent.

| Collision filter | AP | AP0.8 | AP0.4 | Provenance |
|---|---|---|---|---|
| not stated | 17.95 | 21.54 | 13.53 | reported |
| off | 8.69 | 10.35 | 6.96 | author-ckpt ([record](records/graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e_author-ckpt_test_raw.json)) |
| on | 17.14 | 20.60 | 12.86 | author-ckpt ([record](records/graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e_author-ckpt_test_cd.json)) |

`reported` is Table IX of the GraspClutter6D paper (arXiv 2504.06866 v3), which states
neither the collision filter nor which checkpoint was scored. Measured minus reported
(AP / AP0.8 / AP0.4): −9.26 / −11.19 / −6.57 without the filter, −0.81 / −0.94 / −0.67 with
it. The collision filter is worth 8.45 AP for this method on this dataset, and only the
filtered run comes near the reported row: the evidence says that row was produced with a
collision filter. The remaining 0.81 AP was not investigated. One observation, also not
investigated: the same table gives 23.43 / 27.81 / 19.56 for this method on
GraspNet-1Billion, and the GraspNet-1Billion records above, taken over all 90 test
scenes, give 23.51 / 27.92 / 19.59 without the filter and 35.86 / 43.09 / 28.68 with it.
The paper's number is thus matched by the unfiltered run on GraspNet-1Billion and
approached only by the filtered run on GraspClutter6D.

The input is cropped to the image rectangle around the ground-truth segmentation, as the
official tool does (`gt_assisted: workspace`); this is not the crop the weights were
trained with (`docs/user_guides/cross_dataset.md`).

## Verification

`tests/integration/test_baseline_parity.py` compares this implementation with activations
recorded from the pristine reference (`tools/make_baseline_fixtures.py`), in true float32:

- conversion is strict and complete: 162 tensors, 1 025 964 parameters on both sides;
- four test frames: identical seeds, views and objectness decisions; every activation
  within 3e-6 of the stage's largest value; decoded grasps within 1e-6;
- 1 of 16 384 crops holds a point on a cylinder surface on which the two float32
  implementations disagree; it is checked in double precision and everything after the
  crop is compared with the reference's neighbour lists as well;
- one training batch of four frames: label matching exact (score, width, tolerance and
  view targets differ by 0), all six loss terms within 1e-7 relative, total gradient norm
  within 1e-6 relative;
- tolerance labels: identical to the reference generator on 1 036 800 grasps of three
  objects (`tests/integration/test_baseline_data.py`).

For scale: with PyTorch's default TF32 convolutions the reference differs from itself by
up to 9e-3 on the seed features and chooses other views for 6 to 11 seeds per frame.
