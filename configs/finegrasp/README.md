# FineGrasp

> [FineGrasp: Towards Robust Grasping for Delicate Objects](https://arxiv.org/abs/2507.05978)
> (arXiv 2025, Horizon Robotics). Upstream:
> [HorizonRobotics/robo_orchard_lab](https://github.com/HorizonRobotics/robo_orchard_lab),
> `projects/finegrasp_graspnet1b`, read at `221ec0e` and run as PyPI release 0.4.0
> (Apache-2.0). Label tools:
> [HorizonRobotics/robo_orchard_grasp_label](https://github.com/HorizonRobotics/robo_orchard_grasp_label)
> at `977be36`.

FineGrasp is [EconomicGrasp](../economicgrasp) with three changes, and it is implemented
here as exactly that: the EconomicGrasp detector, one new component, one new parameter of
a label asset and two data transforms. There is no `FineGrasp` class.

- **Instance-normalised graspness.** Point-wise graspness is normalised within every
  object before it is normalised over the frame, so that a small object next to an easy
  one is not pushed under the seed threshold.
- **Multi-range attention.** Every seed is cropped at four radii instead of one; the
  four crop features are fused by attention over the scales.
- **Normal prior.** The encoder is fed `[x, y, z | surface normal]` per point instead
  of three ones.

**Run it in five minutes:** [docs/methods/finegrasp.md](../../docs/methods/finegrasp.md) gives the authors' checkpoint, its licence and the conversion command, the demo on the example scene with what it prints, the same from Python and in the browser, and a first AP on 120 frames of the benchmark, each with the date, commit and machine it was measured on.

| slot | component | shared with EconomicGrasp |
|---|---|---|
| `backbone` | `SparseUNet14` | yes, with 6 input channels instead of 3 |
| `proposal` | `GraspnessFPS` | yes |
| `view` | `TemplateViewStage` | yes |
| `assigner` | `CompactLabelAssigner` | yes |
| `grouper` | `MultiScaleGroup` of four `CylinderInteractionGroup` | every branch is EconomicGrasp's grouper; the fusion is new |
| `head` | `EconomicHead` | yes |

The implementation is clean-room: written from the paper and from the behaviour of the
upstream code, with no upstream source in it.

## FineGrasp against EconomicGrasp, stage by stage

Read from the code of both references (`robo_orchard_lab` at `221ec0e`,
`iSEE-Laboratory/EconomicGrasp` at `7361bb6`). "Same component" means the same class with
the same values in both model configs (`tests/unit/test_finegrasp_detector.py` compares
the two files); everything that is the same is interchangeable between the two methods,
and with GSNet wherever [EconomicGrasp's README](../economicgrasp/README.md) says so.

| Stage | EconomicGrasp | FineGrasp | Here |
|---|---|---|---|
| Input features | three ones per point | `[x, y, z, n_x, n_y, n_z]`: coordinates and the surface normal of the point's pixel | **option on an existing component**: `model.point_features: observation` (was `ones`), fed by two **new transforms**, `DepthNormals` and `CoordinateFeatures` |
| Encoder | MinkUNet14D, 3 → 512 channels, 5 mm voxels | the same network, 6 → 512 channels | **same component** (`SparseUNet14`), `in_channels: 6`. `voxel_features: first` now matters: points that share a voxel no longer carry equal features, and the reference uses those of the first |
| Graspness definition | fraction of usable grasps per label point, spread to the cloud by nearest neighbour, min-max normalised over the frame | the same, with a min-max normalisation within every object before the spreading | **option on an existing component**: `graspness.normalization: instance` of the asset generator (was `scene`); another asset with another hash |
| Proposal | objectness + graspness head, threshold 0.1, FPS to 1024 seeds | the same | **same component** (`GraspnessFPS`) |
| View | 300 view scores, sampled in training, arg-max in inference; its hidden feature is added to the seed feature | the same | **same component** (`TemplateViewStage`). The reference adds a guard: when all 300 scores of some seed are equal it takes the arg-max for the whole frame; `TemplateViewStage` samples that seed uniformly. Neither happened in the recorded batches |
| Grouping | one cylinder among the seeds, radius 0.05 m, 16 neighbours, shared MLP, neighbour attention, second MLP, max | four such modules with weights of their own, radii 0.25, 0.5, 0.75 and 1 × 0.07 m; the four 256-d features of a seed pass two transformer encoder layers (8 heads, width 1024) and are summed with soft-max weights | **new component** `MultiScaleGroup`, whose branches are the **same component** (`CylinderInteractionGroup`, unchanged). Its encoder layers start with EconomicGrasp's attention block (`TokenInteraction`), which gained the **option** `residual_dropout` (0 there) |
| Head | four task tokens, token attention, angle / depth / quality classes and a width | the same | **same component** (`EconomicHead`) |
| Labels | compact labels: the best grasp per label point and view | the same files, made with EconomicGrasp's generator | **same component** (`CompactLabelAssigner`) and the same asset |
| Label matching | nearest label point, 5 mm, one supervised view per labelled view | the same function | **same component** |
| Losses | objectness 1, graspness 10, view 100, angle 1, depth 1, quality 1, width 10 | the same terms and weights | **same components** |
| Decoder | arg-max over all logits (at `7361bb6`), logit-weighted score, width × 1.2 | the same | **same component** (`head.decode_argmax: all_logits`) |
| Augmentation | flip of the cloud and the poses | the same; the normals are not mirrored | **same component** (`RandomFlip`); the normals are computed before it and are features, which a flip does not touch |
| Training recipe | Adam 1e-3, no weight decay, cosine per epoch, batches of 4, 10 epochs | Adam 8e-3 over 8 processes × 4 frames, weight decay 1e-5, linear warm-up for one epoch then cosine, per step; gradient norm clipped at 10; 10 epochs | a schedule config (`_base_/schedules/finegrasp_10e.yaml`) and one **new scheduler**, `WarmupCosineLR`; clipping and per-step scheduling are **options** of the trainer |
| Collision filter | optional, voxel 0.005 m | optional, voxel 0.01 m | **same component** (`ModelFreeCollisionFilter`), another value |

So a FineGrasp model differs from an EconomicGrasp model in three config entries
(`point_features`, `backbone.in_channels`, `grouper`), and each of the paper's three
ideas can be switched on alone:

```bash
# EconomicGrasp (or GSNet) on instance-normalised graspness, nothing else changed.
# The asset is the one FineGrasp trains on: same camera, same preprocessing, same hash.
grasppanda prepare graspnet1b --graspness --camera realsense \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda train economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
    --opts data.train.dataset.graspness.normalization=instance
```

`MultiScaleGroup` takes any grouper that crops among the seeds as its branch and emits
the axes of a single cylinder, so it also fits under GSNet's head. Neither of these
variants has been trained here.

What the label change acts on, measured on the first 20 training scenes (192 object
instances): the best label point of an object reaches, on the median, 54 % of the
graspness of the best point of its scene; for 89 objects it stays below half and for 37
below a quarter, down to 0.7 %. Scene normalisation keeps these ratios, instance
normalisation sets every one of them to 1. No object of these scenes was without
contrast (the case in which the published generator divides by zero). In the frames
themselves (the asset built for scene 0, eight frames): with scene normalisation one or
two of the seven to nine visible objects have no visible point above the seed threshold
of 0.1, with instance normalisation none; the share of object points above it rises from
18-28 % to 22-34 %.

## Results

GraspNet-1Billion, single view, protocol `graspnet1b-sv-v1`: the whole test split (90
scenes, 23,040 frames, all scored). AP in percent.

| Config | Camera | Collision filter | Seen | Similar | Novel | Provenance |
|---|---|---|---|---|---|---|
| [finegrasp_sunet14_g1b-realsense_10e](finegrasp_sunet14_g1b-realsense_10e.yaml) | RealSense | off | 71.67 | 62.83 | 27.40 | reported |
| | RealSense | off | 71.56 | 62.80 | 27.42 | author-ckpt ([record](records/finegrasp_sunet14_g1b-realsense_10e_author-ckpt_test_raw.json)) |
| | RealSense | on | 73.71 | 64.56 | 28.14 | reported |
| | RealSense | on | 73.69 | 64.62 | 28.18 | author-ckpt ([record](records/finegrasp_sunet14_g1b-realsense_10e_author-ckpt_test_cd.json)) |

`reported` numbers are those of Table I of the paper; the model card of the released
weights repeats them and names the weight file used here. `author-ckpt` numbers are that
file (`finegrasp_pipeline/model.safetensors`), converted with
`tools/convert_checkpoint.py` and run through `grasppanda test` at commit `4bc5333`, one
run with seed 0; the linked result records hold the per-scene values, the config hash
and the hashes of code and weights. The workspace crop uses the ground-truth
segmentation, as in the reference (`gt_assisted: workspace`).

**Measured minus reported**: −0.11 / −0.03 / +0.02 AP without the collision filter and
−0.02 / +0.06 / +0.04 AP with it. Every split is within 0.11 AP of the paper. The normal
images the weights were trained on are not published and are reconstructed here (next
section), so this is also the whole-split evidence for that reconstruction, which the
120-frame sample below could only suggest. Not investigated: how much of the remaining
tenth of an AP is the seed of the point sampling.

**No `reproduced` number exists yet**: no FineGrasp model has been trained here.

## The normals: what the released weights expect

The reference dataset reads `scenes/<scene>/<camera>/normal/<frame>.npy` and divides it
by 255. Neither these files nor the code that made them are published. The paper gives
the estimator (its equation 5, the cross product of the two tangents of the organised
cloud); the orientation and the value range had to be identified. The released weights
were run in the reference code on 20 test frames (scenes 100 to 104, every 64th frame)
with different normal inputs:

| Normal input | AP, 20 frames, no collision filter |
|---|---|
| paper's estimator, towards the camera, as an 8-bit normal image: `(n + 1) / 2` in [0, 1] | **69.4** |
| the same without rounding to 8 bits | 69.0 |
| the same with neighbours 2 pixels apart / with forward differences | 69.3 / 68.6 |
| Open3D normals of the full cloud, oriented to the camera, same encoding | 68.9 |
| paper's estimator, *away* from the camera, same encoding | 59.5 |
| paper's estimator, towards the camera, colour channels reversed | 58.7 |
| no normals (zeros) | 43.1 |
| unit vectors in [−1, 1], away from / towards the camera | 36.8 / 31.7 |
| what the released inference processor feeds: Open3D normals of the sampled cloud, unit vectors, sign as returned | **35.0** |

`DepthNormals` with `towards_camera: true` and `encoding: image8` is the first row. With
it the reference reaches the reported AP on a sample of the benchmark (next section),
which is the evidence that the reconstruction is right; the estimators of the upper
rows cannot be told apart on 20 frames, and the paper's was chosen. The last row is a
finding about the reference rather than about this port: the inference processor shipped
with the weights (`FineGraspProcessor`) feeds normals in another range and orientation
than the dataset class the weights were trained with, and on these frames that costs
half of the AP. It is not reproduced here.

## Status of the verification

Checked with the authors' weights (`finegrasp_pipeline/model.safetensors`, sha256
`30c1f335…8f79`) against the reference code, release 0.4.0 of `robo_orchard_lab` (its
FineGrasp model, loss, label-matching, dataset and metric files are those of `221ec0e`).
The reference ran in an existing environment (Python 3.10, torch 2.4.0, MinkowskiEngine
0.5.4) with its pure-Python dependencies on the path; three things it does not ship
were supplied, as `tools/oracle/finegrasp_reference.py` documents: the `knn` wrapper its
label matching imports, the `pointnet2` sources, and the normal images above. The
numbers come from `tests/integration/test_finegrasp_parity.py`; they are parity
measurements, not benchmark results.

| What | Result |
|---|---|
| Checkpoint conversion | strict: 388 of 388 tensors consumed, 396 of 396 filled (the stacked query / key / value projections of the two fusion layers become three tensors each), 19 145 441 parameters as counted in the reference |
| Normals, 4 test frames | `DepthNormals` equals the recorded normal image at every point of the frame |
| Inference, 4 frames | identical voxels, voxel representatives, candidates, seeds and views; activations of the backbone at every level, of each of the four crops at every step, of both fusion layers, of the fusion weights and of the head within 3.7e-6 of the largest value of a stage; decoded grasps identical up to 2.2e-5 (score) |
| Inference, 32 further frames | identical seeds on 32 of 32; all 32 768 grasps decoded alike (no arg-max decided by round-off) |
| Inference, the 120 frames of the sample below | identical seeds on 120 of 120; of 122 880 grasps, one arg-max was decided by round-off |
| Training inputs, 4 frames (two mirrored) | the pipeline of the runnable config builds the reference loader's points and features: mirrored coordinates, unmirrored normals |
| Training batch (4 frames) | activations within 3.3e-6, identical seeds; every loss term within 7.2e-8 relative and their sum within 2.6e-8; total gradient norm within 3.5e-7, the 19 compared gradients (among them the 6-channel stem and every part of the fusion) within 1.4e-3 of their largest entry |
| Collision filter of the `cd` variant | the same decisions as the reference detector on the 4 frames |
| Scale fusion against `nn.TransformerEncoder` | equal to 2e-6 in evaluation mode and, without dropout, in training mode (CPU unit test, weights mapped by the converter) |

The training comparison replays what the reference draws at random or leaves undefined:
the sampled view of every seed, the entries dropped by every dropout module, and the
choice among template views that share a labelled view (see EconomicGrasp's
`assigner.shared_views`; the reference changed 10 of 4 096 seeds between two runs). Two
things about it are weaker than the rest. The attention dropout inside the two fusion
layers is drawn in a compiled kernel of the reference and cannot be recorded, so it was
off on both sides; only the unit test above pins its place. And the graspness labels of
that batch are the files of the dataset's `graspness` folder, which are not
instance-normalised, under the name the reference reads: the loss consumes them as
numbers, and the instance-normalised asset is verified separately, against its
definition, on synthetic data (`tests/unit/test_finegrasp_graspness.py`).

### A sample of the benchmark

Every 64th frame of every seen test scene: 120 frames, RealSense, AP in percent, without
/ with the collision filter. This is a sample of 1.6 % of the split, taken to see whether
the weights, the reconstructed normals and the port are sound. It is **not** an
`author-ckpt` result.

| | Seen, 120 frames |
|---|---|
| Reference code, its own point sampling | 71.48 / 73.87 |
| GraspPanda fed the same points and normals | 71.48 / 73.85 |
| `grasppanda test`, its own point sampling | 71.21 / 72.99 |
| Reported, whole split, for orientation | 71.67 / 73.71 |

Fed the same input, GraspPanda scores one frame differently without the filter (by 0.1
AP, the frame of the one arg-max above). With the filter 52 frames differ, by −2.5 to
+3.5 AP and −0.02 on average: the evaluation engine hands the filter the cloud in single
precision, the reference's detector voxelises it in double, and points on a voxel face
are common in a depth cloud. Given the double-precision cloud, GraspPanda's filter
result is the reference's on all 120 frames (73.87). The similar and novel splits have
not been sampled.

Not verified: any training run (two epochs on 90 frames ran through `grasppanda train`
as a smoke test, on scene-normalised labels), the Kinect camera (the authors publish
RealSense only), the weights trained with simulation data (`finegrasp_pipeline_sim`,
which take no normals), and the behaviour on GraspClutter6D.

## Options of every slot

Every slot of this method, the options that fit it and those that do not, each with what
was measured for it (the verdict of its probe, the levels *composes*, *runs*, *learns* and
*predicts*, its class, the grasps per frame, the reason and kind of a refusal):
[docs/methods/finegrasp.md](../../docs/methods/finegrasp.md), or in a terminal:

```bash
grasppanda compose finegrasp/finegrasp_sunet14_g1b-realsense_10e --list            # every slot
grasppanda compose finegrasp/finegrasp_sunet14_g1b-realsense_10e --list grouper    # one slot (--json: everything)
```

A class counts verdicts of one run (seed 0, 20 iterations, 8 frames of one scene), not
accuracy.

## Usage

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda prepare graspnet1b --compact-labels \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda train finegrasp/finegrasp_sunet14_g1b-realsense_10e
grasppanda test  finegrasp/finegrasp_sunet14_g1b-realsense_10e --weights <checkpoint>
```

The compact labels are EconomicGrasp's asset; if that exists, nothing is generated. The
graspness asset is a new one, as large as the scene-normalised asset; so far it has been
built for one scene. To evaluate the authors' weights, convert them first; testing needs
no asset:

```bash
python -m tools.convert_checkpoint finegrasp_pipeline/model.safetensors \
    finegrasp_realsense.pth --source-format finegrasp --camera realsense
grasppanda test finegrasp/finegrasp_sunet14_g1b-realsense_10e --weights finegrasp_realsense.pth \
    --opts data.test.dataset.split=test_seen data.test.dataset.frame_step=64   # the sample above
```

## Reference behaviour kept as named options

Each of these changes numbers. The default (or the value in the shipped config)
reproduces the reference; an alternative is a controlled variant with its own experiment
name. The options FineGrasp inherits from EconomicGrasp (`view.train_selection`,
`assigner.*`, `head.*`, `grouper.branch.centre_coordinates`, `compact_labels.*`) are
listed in [its README](../economicgrasp/README.md) and have the same values here.

| Option | Value | What the reference does |
|---|---|---|
| `DepthNormals.towards_camera`, `.encoding` (data) | `true`, `image8` | See the section on normals: identified with the released weights, not read from code. `unit` keeps unit vectors. |
| order of `DepthNormals`, `RandomFlip`, `CoordinateFeatures` (data) | normals first, coordinates last | The flip mirrors the cloud and the object poses and leaves the normal of every point as it was in the sensor frame, so half of the training samples pair mirrored coordinates with unmirrored normals. |
| `backbone.voxel_features` | `first` | The input feature of a voxel is that of its first point, not a mean. |
| `grouper.radius`, `.radius_scales` | `0.07`, `[0.25, 0.5, 0.75, 1.0]` | Radii are the products in double precision (`0.07 * 0.75` is not `0.0525`); membership of a neighbour on a cylinder boundary depends on it. |
| `grouper.fusion_dropout` | `0.1` | The default of `nn.TransformerEncoderLayer`, in four places per layer. |
| `graspness.normalization` (data) | `instance` | An object whose label points all have the same graspness, in practice one without any usable grasp in the scene, keeps 0. The published generator divides by zero there, which makes every label of the frame NaN. |
| `scheduler.cosine_steps` | the whole run | The cosine starts after the warm-up but its period is the length of the run, so the last step trains at 2.4 % of the rate instead of the minimum. |
| `evaluation.variants.cd.voxel_size` | `0.01` | EconomicGrasp's test script uses 0.005. |

## Differences from the reference

- No model class of its own; one model instance trains and predicts.
- The normal images and the instance-normalised graspness files of the authors'
  training set are not published. Both are generated here, from the paper's equations;
  the normals are checked through the AP of the released weights, the graspness labels
  cannot be checked against the authors' at all.
- The published recipe uses eight processes with four frames each; the runnable config
  uses one process with 32, which keeps the learning rate and the number of steps but
  gives batch norm 32 frames instead of 4. With fewer frames per step, scale the rate
  and the two step counts of the scheduler. None of this has been run beyond a smoke
  test.
- A frame with fewer graspable candidates than seeds yields fewer grasps instead of
  repeats, and a frame with none yields an empty result (as for EconomicGrasp).
- The released inference processor additionally drops grasps as wide as the gripper,
  applies grasp NMS and sorts by score. That is deployment post-processing, not part of
  the benchmark metric, and not part of the configs here (`GraspNMS` exists as a
  post-processor).

## Citation

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
