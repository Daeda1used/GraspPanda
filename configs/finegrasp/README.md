# FineGrasp

> [FineGrasp: Towards Robust Grasping for Delicate Objects](https://arxiv.org/abs/2507.05978)
> (arXiv 2025, Horizon Robotics). Reference code:
> [HorizonRobotics/robo_orchard_lab](https://github.com/HorizonRobotics/robo_orchard_lab),
> `projects/finegrasp_graspnet1b`, read at `221ec0e` and run as PyPI release 0.4.0
> (Apache-2.0). Label tools:
> [HorizonRobotics/robo_orchard_grasp_label](https://github.com/HorizonRobotics/robo_orchard_grasp_label)
> at `977be36`.

FineGrasp is [EconomicGrasp](../economicgrasp/README.md) with three changes, and it is
implemented here as exactly that: the EconomicGrasp detector, one new component, one new
parameter of a label asset and two data transforms. There is no `FineGrasp` class.

- **Instance-normalised graspness.** Graspness is normalised within every object before
  the frame, so a small object next to an easy one stays above the seed threshold.
- **Multi-range attention.** Every seed is cropped at four radii instead of one; the
  four crop features are fused by attention over the scales.
- **Normal prior.** The encoder is fed `[x, y, z | surface normal]` per point instead
  of three ones.

**Run it in five minutes:** [the method page](https://daeda1used.github.io/GraspPanda/methods/finegrasp/#run-in-five-minutes): the weights, grasps on the example scene, the same from Python and in the browser, and a first number on the benchmark, each with what it printed when it was measured.

## Configs

| Config | Trains on | Tests on |
|---|---|---|
| [finegrasp_sunet14_g1b-realsense_10e](finegrasp_sunet14_g1b-realsense_10e.yaml) | GraspNet-1Billion, RealSense | GraspNet-1Billion, RealSense |
| [finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e](finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml) | GraspNet-1Billion, RealSense | GraspClutter6D, RealSense D435 |

The options of every slot, with what was measured for each:
[the method page](https://daeda1used.github.io/GraspPanda/methods/finegrasp/#the-slots), or in a terminal
`grasppanda compose finegrasp/finegrasp_sunet14_g1b-realsense_10e --list` (one slot:
`--list grouper`).

## Results on GraspNet-1Billion

AP on seen / similar / novel over the whole test split (90 scenes, 23,040 frames),
protocol `graspnet1b-sv-v1`. The workspace is cropped with the ground-truth
segmentation, as in the reference (`gt_assisted: workspace`).

| Config | Camera | Collision filter | Seen | Similar | Novel | Provenance |
|---|---|---|---|---|---|---|
| [finegrasp_sunet14_g1b-realsense_10e](finegrasp_sunet14_g1b-realsense_10e.yaml) | RealSense | off | 71.67 | 62.83 | 27.40 | reported |
| | RealSense | off | 71.56 | 62.80 | 27.42 | author-ckpt ([record](records/finegrasp_sunet14_g1b-realsense_10e_author-ckpt_test_raw.json)) |
| | RealSense | off | 69.77 | 59.72 | 25.77 | reproduced ([record](records/finegrasp_sunet14_g1b-realsense_10e_reproduced_test_raw.json)) |
| | RealSense | on | 73.71 | 64.56 | 28.14 | reported |
| | RealSense | on | 73.69 | 64.62 | 28.18 | author-ckpt ([record](records/finegrasp_sunet14_g1b-realsense_10e_author-ckpt_test_cd.json)) |
| | RealSense | on | 72.43 | 61.87 | 26.94 | reproduced ([record](records/finegrasp_sunet14_g1b-realsense_10e_reproduced_test_cd.json)) |

`reported` is Table I of the paper. `author-ckpt` is the weight file its model card names,
converted and run through `grasppanda test` at commit `4bc5333`, one run with seed 0.
**Measured minus reported**: −0.11 / −0.03 / +0.02 AP without the collision filter and
−0.02 / +0.06 / +0.04 AP with it.

`reproduced` is a model trained from scratch here: −1.90 / −3.11 / −1.63 AP and
−1.28 / −2.69 / −1.20 AP from the paper, outside the zoo's admission gate of 1 AP on
every split. The gap is not explained: run the method from the authors' weights.

<details markdown="1"><summary>What the authors' weights show, and what is known about the training</summary>

**The authors' weights.** The normal images they were trained on are not published and
are reconstructed here (*The normals* under Notes); their agreement with the paper is the
whole-split evidence for that reconstruction.

**The training.** At commit `400b733` with the four-GPU command of *Train and test*: four
processes of four frames and the gradients of two batches per step, 32 frames a step as
in the published recipe; 10 epochs, seed 0, 5.1 hours on four RTX 3090 shared with other
jobs. The checkpoint of epoch 10 (SHA-256 `e8b99d4390d9…`) was scored at the same commit.

**The gap.** The authors' weights reach the paper through the same test path, so it comes
from the training. On a recorded batch the losses and gradients equal the reference's
(*Equivalence with the reference code*). What a training here does not share with the
authors': their normal images and graspness labels, generated here from the paper's
equations, and their eight processes, replaced by four with accumulated gradients. Which
of these, or the seed, accounts for it was not investigated: one run, one seed.

</details>

## Results on GraspClutter6D

The whole test split (235 scenes, 3,055 frames of the RealSense D435), protocol
`graspclutter6d-sv-v1`, config
[finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e](finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e.yaml).
AP / AP0.8 / AP0.4.

| Weights | Features | Collision filter off | on | Records |
|---|---|---|---|---|
| authors' | as published | 12.56 / 15.47 / 9.01 | 13.14 / 16.20 / 9.19 | author-ckpt: [off](records/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_author-ckpt_test_raw.json), [on](records/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_author-ckpt_test_cd.json) |
| | depth moved | 24.39 / 29.95 / 17.40 | 25.02 / 30.75 / 17.60 | [off](variants/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_author-ckpt_median-depth-0.42_test_raw.json), [on](variants/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_author-ckpt_median-depth-0.42_test_cd.json) |
| trained here | as published | 16.12 / 19.78 / 11.53 | 17.00 / 20.85 / 11.86 | reproduced: [off](records/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_raw.json), [on](records/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_test_cd.json) |
| | depth moved | 23.90 / 29.13 / 17.62 | 25.18 / 30.73 / 18.17 | [off](variants/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_median-depth-0.42_test_raw.json), [on](variants/finegrasp_sunet14_g1b-realsense-to-gc6d-d435_10e_reproduced_median-depth-0.42_test_cd.json) |

**The weights know their camera distance.** The encoder reads coordinates, and this split
lies farther from the camera than any training cloud. *Depth moved* shifts the depth of
the features back, nothing else: `--opts data.test.dataset.transforms.2.median_depth=0.42`.

<details markdown="1"><summary>The camera distance: what was measured, and your own camera</summary>

The cloud the model is fed has a median depth of 0.42 m on GraspNet-1Billion (0.34 to
0.46 m over 300 training frames) and of 0.75 m on this split (0.51 to 1.61 m over 200
test frames). `CoordinateFeatures.median_depth` shifts the depth coordinate of the
features so that the median of a frame is the given value; points, voxels and grasps stay.
It is a setting of the test, not of the published method: the zoo lists the rows *as
published*. One run each, seed 0; the dataset's paper has no FineGrasp row.

The override also serves `demo`, `infer` and `GraspInferencer(..., overrides=[...])` on a
camera of your own. EconomicGrasp, the same model with constant input features, scores
21.69 AP here; the other methods read no coordinates either (HGGD centres its depth).

</details>

## Weights

GraspPanda publishes no weights for this method; those trained here are below the paper.
Download `finegrasp_pipeline/model.safetensors` from the authors' model repository (link
and SHA-256 in the [metafile](metafile.yaml)) and convert it once; testing needs no asset:

```bash
python -m tools.convert_checkpoint finegrasp_pipeline/model.safetensors \
    finegrasp_realsense.pth --source-format finegrasp --camera realsense
grasppanda test finegrasp/finegrasp_sunet14_g1b-realsense_10e --weights finegrasp_realsense.pth \
    --opts data.test.dataset.split=test_seen data.test.dataset.frame_step=64   # 120 frames
```

## Train and test

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda prepare graspnet1b --compact-labels \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda train finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda test  finegrasp/finegrasp_sunet14_g1b-realsense_10e --weights <checkpoint>
```

The compact labels are EconomicGrasp's asset; the graspness asset is new (100 training
scenes: 70 minutes with 14 worker processes).

**Memory decides how to launch it.** At about 1.5 GB of GPU memory per frame the config
as written, one process with 32 frames, needs a 48 GB card. The published recipe is eight
processes of four frames; with fewer GPUs, accumulate the gradients of several batches: a
step still sees 32 frames, batch norm 4, and the schedule stays.

```bash
# 8 GPUs: the published recipe
torchrun --nproc_per_node=8 -m grasppanda.cli.main train finegrasp/finegrasp_sunet14_g1b-realsense_10e \
    --opts data.train.batch_size=4
# 4 GPUs of 24 GB: how the model of Results was trained
torchrun --nproc_per_node=4 -m grasppanda.cli.main train finegrasp/finegrasp_sunet14_g1b-realsense_10e \
    --opts data.train.batch_size=4 train.accumulate=2
# 1 GPU of 24 GB
grasppanda train finegrasp/finegrasp_sunet14_g1b-realsense_10e \
    --opts data.train.batch_size=4 train.accumulate=8
```

## Notes

<details markdown="1"><summary>How it is built: FineGrasp against EconomicGrasp, stage by stage</summary>

Read from the code of both references (`robo_orchard_lab` at `221ec0e`,
`iSEE-Laboratory/EconomicGrasp` at `7361bb6`). "Same component" means the same class with
the same values in both model configs (compared by `tests/unit/test_finegrasp_detector.py`).

| Stage | EconomicGrasp | FineGrasp | Here |
|---|---|---|---|
| Input features | three ones per point | `[x, y, z, n_x, n_y, n_z]`: coordinates and the normal of the point's pixel | **option** `model.point_features: observation` (was `ones`), fed by two **new transforms**, `DepthNormals` and `CoordinateFeatures` |
| Encoder | MinkUNet14D, 3 → 512 channels, 5 mm voxels | the same, 6 → 512 channels | **same component** (`SparseUNet14`), `in_channels: 6`; `voxel_features: first` now matters, and the reference uses the first point's features |
| Graspness | fraction of usable grasps per label point, spread to the cloud, min-max over the frame | the same, with a min-max within every object before the spreading | **option** `graspness.normalization: instance` of the asset generator (was `scene`); another asset with another hash |
| Proposal, view, head, labels, label matching, losses, decoder | `GraspnessFPS`, `TemplateViewStage`, `EconomicHead`, compact labels, nearest label point within 5 mm, the same terms and weights, arg-max over all logits | the same | **same components**. The reference view adds a guard for a seed whose 300 scores are all equal (arg-max for the frame); `TemplateViewStage` samples that seed uniformly. Neither happened in the recorded batches |
| Grouping | one cylinder among the seeds, radius 0.05 m, 16 neighbours, neighbour attention | four such modules, radii 0.25 / 0.5 / 0.75 / 1 × 0.07 m; the four 256-d features pass two transformer encoder layers (8 heads, width 1024) and are summed with soft-max weights | **new component** `MultiScaleGroup`, whose branches are `CylinderInteractionGroup` unchanged; its encoder layers start with EconomicGrasp's `TokenInteraction`, which gained the **option** `residual_dropout` (0 there) |
| Augmentation | flip of the cloud and the poses | the same; the normals are not mirrored | **same component** (`RandomFlip`); normals are computed before it and are features, which a flip does not touch |
| Training recipe | Adam 1e-3, cosine per epoch, batches of 4, 10 epochs | Adam 8e-3 over 8 processes × 4 frames, weight decay 1e-5, one warm-up epoch then cosine, per step; gradient norm clipped at 10 | a schedule config (`_base_/schedules/finegrasp_10e.yaml`) and one **new scheduler**, `WarmupCosineLR`; clipping and per-step scheduling are **options** of the trainer |
| Collision filter | optional, voxel 0.005 m | optional, voxel 0.01 m | **same component** (`ModelFreeCollisionFilter`), another value |

So a FineGrasp model differs from an EconomicGrasp model in three config entries
(`point_features`, `backbone.in_channels`, `grouper`), and each of the paper's ideas can
be switched on alone, for example instance-normalised graspness under EconomicGrasp:

```bash
grasppanda prepare graspnet1b --graspness --camera realsense \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda train economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
    --opts data.train.dataset.graspness.normalization=instance
```

`MultiScaleGroup` takes any grouper that crops among the seeds as its branch and emits the
axes of a single cylinder, so it fits other seed-based heads. Neither variant has been
trained here. What each method ships:
[Methods](https://daeda1used.github.io/GraspPanda/methods/#what-each-method-ships).

</details>

<details markdown="1"><summary>The normals: what the released weights expect</summary>

The reference dataset reads normal files (`normal/<frame>.npy`, divided by 255) that are
not published, nor is the code that made them. The paper gives the estimator (its
equation 5, the cross product of the two tangents of the organised cloud); orientation
and value range were identified by running the released weights in the reference code on
20 test frames (scenes 100 to 104, every 64th frame) with different normal inputs:

| Normal input | AP, 20 frames, no collision filter |
|---|---|
| paper's estimator, towards the camera, as an 8-bit normal image: `(n + 1) / 2` in [0, 1] | **69.4** |
| the same unrounded / neighbours 2 pixels apart / forward differences | 69.0 / 69.3 / 68.6 |
| Open3D normals of the full cloud, oriented to the camera, same encoding | 68.9 |
| paper's estimator, *away* from the camera, same encoding | 59.5 |
| paper's estimator, towards the camera, colour channels reversed | 58.7 |
| no normals (zeros) | 43.1 |
| unit vectors in [−1, 1], away from / towards the camera | 36.8 / 31.7 |
| what the released inference processor feeds: Open3D normals of the sampled cloud, unit vectors, sign as returned | **35.0** |

`DepthNormals` with `towards_camera: true` and `encoding: image8` is the first row. The
upper rows cannot be told apart on 20 frames; the paper's estimator was chosen. The last
row is a finding about the reference: its inference processor (`FineGraspProcessor`) feeds
normals in another range and orientation than the weights were trained with, which costs
half of the AP.

</details>

<details markdown="1"><summary>Equivalence with the reference code</summary>

Checked with the authors' weights (sha256 `30c1f335…8f79`) against release 0.4.0 of
`robo_orchard_lab` (its FineGrasp files are those of `221ec0e`; Python 3.10, torch 2.4.0,
MinkowskiEngine 0.5.4). `tools/oracle/finegrasp_reference.py` supplies what the reference
does not ship: the `knn` wrapper of its label matching, the `pointnet2` sources and the
normal images above. From `tests/integration/test_finegrasp_parity.py`; parity
measurements, not benchmark results.

| What | Result |
|---|---|
| Checkpoint conversion | strict: 388 of 388 tensors consumed, 396 of 396 filled (the stacked query / key / value projections of the two fusion layers become three tensors each), 19 145 441 parameters as counted in the reference |
| Normals, 4 test frames | `DepthNormals` equals the recorded normal image at every point of the frame |
| Inference, 4 frames | identical voxels, voxel representatives, candidates, seeds and views; activations at every level of the backbone, of the four crops, of both fusion layers, of the fusion weights and of the head within 3.7e-6 of the largest value of a stage; decoded grasps identical up to 2.2e-5 (score) |
| Inference, 32 further frames | identical seeds on 32 of 32; all 32 768 grasps decoded alike |
| Inference, the 120 frames of the sample below | identical seeds on 120 of 120; of 122 880 grasps, one arg-max was decided by round-off |
| Training inputs, 4 frames (two mirrored) | the runnable config builds the reference loader's points and features: mirrored coordinates, unmirrored normals |
| Training batch (4 frames) | activations within 3.3e-6, identical seeds; every loss term within 7.2e-8 relative and their sum within 2.6e-8; total gradient norm within 3.5e-7, the 19 compared gradients within 1.4e-3 of their largest entry |
| Collision filter of the `cd` variant | the same decisions as the reference detector on the 4 frames |
| Scale fusion against `nn.TransformerEncoder` | equal to 2e-6 in evaluation mode and, without dropout, in training mode |

The training comparison replays what the reference draws at random or leaves undefined
(sampled views, dropout masks, the choice among template views that share a labelled
view). Two parts are weaker: the attention dropout inside the fusion layers, drawn in a
compiled kernel of the reference, was off on both sides, and the graspness labels of that
batch are the dataset's scene-normalised files; the instance-normalised asset is verified
against its definition on synthetic data (`tests/unit/test_finegrasp_graspness.py`).

**A sample of the benchmark.** Every 64th frame of every seen test scene: 120 frames,
RealSense, AP without / with the collision filter; not an `author-ckpt` result.

| | Seen, 120 frames |
|---|---|
| Reference code, its own point sampling | 71.48 / 73.87 |
| GraspPanda fed the same points and normals | 71.48 / 73.85 |
| `grasppanda test`, its own point sampling | 71.21 / 72.99 |
| Reported, whole split, for orientation | 71.67 / 73.71 |

Fed the same input, the two differ on one frame without the filter (the arg-max above)
and by −0.02 AP on average with it: the evaluation engine hands the filter a
single-precision cloud, the reference voxelises in double (given that cloud: 73.87).

Not verified: that a training reaches the paper (the run of *Results* ends below it), the
Kinect camera (the authors publish RealSense only), the weights trained with simulation
data (`finegrasp_pipeline_sim`, no normals), and GraspClutter6D.

</details>

<details markdown="1"><summary>Reference behaviours kept as named options</summary>

Each of these changes numbers; an alternative is a controlled variant with its own
experiment name. The options inherited from EconomicGrasp are listed in
[its README](../economicgrasp/README.md) and have the same values here.

| Option | Value | What the reference does |
|---|---|---|
| `DepthNormals.towards_camera`, `.encoding` (data) | `true`, `image8` | Identified with the released weights (see *The normals*), not read from code. `unit` keeps unit vectors. |
| order of `DepthNormals`, `RandomFlip`, `CoordinateFeatures` (data) | normals first, coordinates last | The flip mirrors the cloud and the poses and leaves every normal as it was in the sensor frame, so half of the training samples pair mirrored coordinates with unmirrored normals. |
| `backbone.voxel_features` | `first` | The input feature of a voxel is that of its first point, not a mean. |
| `grouper.radius`, `.radius_scales` | `0.07`, `[0.25, 0.5, 0.75, 1.0]` | Radii are the products in double precision (`0.07 * 0.75` is not `0.0525`); membership of a neighbour on a cylinder boundary depends on it. |
| `grouper.fusion_dropout` | `0.1` | The default of `nn.TransformerEncoderLayer`, in four places per layer. |
| `graspness.normalization` (data) | `instance` | An object whose label points all have the same graspness keeps 0. The published generator divides by zero there, which makes every label of the frame NaN. |
| `scheduler.cosine_steps` | the whole run | The cosine starts after the warm-up but its period is the length of the run, so the last step trains at 2.4 % of the rate instead of the minimum. |
| `evaluation.variants.cd.voxel_size` | `0.01` | EconomicGrasp's test script uses 0.005. |

</details>

<details markdown="1"><summary>How this port differs from the reference</summary>

- No model class of its own; one model instance trains and predicts.
- The normal images and the instance-normalised graspness files of the authors' training
  set are not published. Both are generated here from the paper's equations; the normals
  are checked through the AP of the released weights, the graspness labels cannot be
  checked against the authors' at all.
- The published recipe uses eight processes of four frames; the config as written uses
  one process with 32, which gives batch norm 32 frames instead of 4 and needs 48 GB.
  *Train and test* gives the launch that keeps both on the GPUs you have.
- A frame with fewer graspable candidates than seeds yields fewer grasps instead of
  repeats, and a frame with none an empty result (as for EconomicGrasp).
- The released inference processor also drops grasps as wide as the gripper, applies
  grasp NMS and sorts by score: deployment post-processing, not part of the benchmark
  nor of these configs (`GraspNMS` is a post-processor).

</details>

## Citation

<details markdown="1"><summary>BibTeX</summary>

```bibtex
@misc{du2025finegrasp,
  title         = {FineGrasp: Towards Robust Grasping for Delicate Objects},
  author        = {Du, Yun and Zhao, Mengao and Lin, Tianwei and Jin, Yiwei and
                   Huang, Chaodong and Su, Zhizhong},
  year          = {2025},
  eprint        = {2507.05978},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

</details>
