# Cross-dataset evaluation

How a model trained on GraspNet-1Billion is tested on GraspClutter6D, what such a
comparison keeps fixed, and what has been measured so far. The facts about the two
datasets are in [Dataset preparation](datasets.md).

## One command

```bash
grasppanda test gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e --weights <checkpoint>
```

| Method | Transfer config | Weights it takes |
|---|---|---|
| GSNet | `gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e` | `gsnet_sunet14_g1b-realsense_10e` |
| EconomicGrasp | `economicgrasp/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e` | `economicgrasp_sunet14_g1b-realsense_10e` |
| EconomicGrasp | `economicgrasp/economicgrasp_sunet14_g1b-kinect-to-gc6d-d435_10e` | `economicgrasp_sunet14_g1b-kinect_10e` |
| GraspNet-baseline | `graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e` | `graspnet-baseline_pointnet2_g1b-realsense_18e` |
| Scale-Balanced-Grasp | `scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e` | `sbg_pointnet2-transformer_g1b-realsense_18e` |
| HGGD | `hggd/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e` | `hggd_r34w8_g1b-realsense_15e` |

Author checkpoints are converted first, exactly as for GraspNet-1Billion (the commands
are in each `configs/<method>/README.md`). Useful overrides:

```bash
--opts data.test.dataset.split=test_mini           # 24 scenes, 312 frames: minutes, a check only
--opts data.test.dataset.camera=zivid              # realsense-d415 | realsense-d435 | azure-kinect | zivid
--opts evaluation.num_workers=8                    # processes of the evaluator
```

The whole test split is 235 scenes × 13 views = 3 055 frames per camera. No derived
asset is needed to test.

## Why a config and not a list of overrides

A transfer config inherits the method's GraspNet-1Billion config and replaces two
nodes, `data.test` and `evaluation.evaluator`:

```yaml
_base_:
  - gsnet_sunet14_g1b-realsense_10e.yaml

data:
  test:
    dataset:
      type: GraspClutter6D
      camera: realsense-d435
      split: test
      load_labels: false
      preprocess:
        - {type: DepthToCloud}
        - {type: SegmentationRectWorkspace, margin: 0.1}
      transforms: ${data.val.dataset.transforms}
      keep_full_points: true
      seed: 0

evaluation:
  evaluator: {type: GraspClutter6DEvaluator}
```

- **What the weights depend on cannot drift.** The model, the point count, the input
  features and both post-processing variants are not restated; they are the inherited
  values, and `transforms` is an interpolation of the recipe's own evaluation
  transforms. A unit test compares every other section of each transfer config with its
  parent.
- The same change as `--opts` would be about ten overrides including two lists, because a
  changed `type` replaces a node and all its arguments must be given again. That is easy
  to get wrong and leaves no name to quote in a result table.
- The config has a name, a hash in every result record, and is validated by
  `grasppanda print-config --check` in the test suite.

A transfer config is a test config. Its evaluator cannot score the inherited
GraspNet-1Billion validation loader, so it sets `evaluation.interval: 0`; to train, use
the config it inherits from.

## What changes with the dataset, and what must not

| | In the transfer config | Why |
|---|---|---|
| Frames | **changes**: `GraspClutter6D`, `realsense-d435`, `test` | The benchmark. The D435 is the camera of the paper's table and the sensor model of GraspNet-1Billion's `realsense`. |
| Intrinsics, depth scale | **changes**, read per image | Nothing in a model or a transform holds a camera constant; `DepthToCloud` and HGGD read the calibration of the frame. |
| Workspace crop | **changes** to `SegmentationRectWorkspace` (margin 0.1) | The crop of the official tool, and of the authors' released baseline code. GraspNet-1Billion's crop (`SegmentationBoxWorkspace`) is a box in a table frame that this dataset does not have; see the measured effect below. HGGD has no crop on either dataset. |
| Evaluator | **changes** to `GraspClutter6DEvaluator` | Width clip 0.14 m, foreground filter, no table: the protocol of the benchmark. |
| Model, including every length | **fixed** | Voxel size (5 mm), ball-query and cylinder radii, depth bins and the 300 views are the units the weights were trained in. |
| Gripper of the model (0.10 m) | **fixed** | It bounds the labels the model learnt from and clamps its decoded widths. See "The gripper" below. |
| Points per cloud (15 000 / 20 000 / 25 600), input features | **fixed** | Part of the recipe; fewer or more points change the density the encoder was trained on. |
| HGGD network image (640 × 360) | **fixed** | The 1920 × 1080 image is reduced by 3 instead of 2; the focal length of the network image comes out nearly the same (465 to 472 px against 464 px). |
| Post-processing (`raw`, `cd`) | **fixed** | The collision filter of each reference, with its own voxel size, approach distance and threshold. |
| Point-sampling seed | **fixed** (0) | The same points every run. |

What shifts without being a setting, and is therefore not corrected:

- **Distance.** Objects are 0.70 m from the D435 in the median (0.36 to 1.04 m), against
  0.44 m (0.31 to 0.52 m) in GraspNet-1Billion. The image is larger, so a pixel covers
  about the same length on an object (0.50 mm against 0.47 mm); how much noisier the depth
  is at that distance was not measured. No depth range is clipped on either dataset.
- **Extent and density.** The reference crop here spans 0.91 × 0.57 × 0.81 m, about twice
  that of GraspNet-1Billion in each direction (0.48 × 0.35 × 0.26 m), and a scene holds
  14.5 object instances instead of 9.0. With the same point budget an object gets
  roughly half the points.
- **Objects.** All 68 test objects are unseen, also by a model trained on
  GraspClutter6D.

### The gripper

The benchmark gripper opens to 0.14 m and the evaluator clips widths there. A model
trained on GraspNet-1Billion was supervised with grasps up to 0.10 m and clamps what it
decodes to 0.10 m. On the 68 test objects, 38 % of the valid label grasps are wider than
0.10 m (4 % are wider than 0.14 m); for 18 objects more than half are, for 3 objects
more than 90 %. This part of the transfer gap is structural.
`--opts model.gripper.max_width=0.14` only moves the clamp of the decoder (which
multiplies the regressed width by 1.2); it is a controlled variant, measured below, not
the default.

## Subset results

!!! warning "Subset numbers"
    Everything in this section was measured on **`test_mini`**: 24 of the 235 test
    scenes, 312 frames per camera. These are not benchmark results and do not enter the
    model zoo. Per-scene AP ranges from below 1 to above 60 for one model, so the mean
    over 24 scenes has a scene-clustered standard error of about 3 AP (given per row).
    The full-split commands are at the end of this page.

AP / AP0.8 / AP0.4 in percent, RealSense D435, protocol `graspclutter6d-sv-v1`, 312 of
312 frames scored. `raw`: predictions as emitted; `cd`: with the reference's model-free
collision filter. Provenance of the weights: `author` is the authors' released
GraspNet-1Billion RealSense checkpoint, converted; `local` is a checkpoint the
maintainers trained with the upstream code on their own variant of the graspness labels
(epoch 18), used because no usable author checkpoint of GSNet is at hand.

| Method | Weights | Input crop | `raw` | `cd` | SE of AP (`raw` / `cd`) | Reported, whole test split |
|---|---|---|---|---|---|---|
| GraspNet-baseline | author | ground-truth rectangle | 8.71 / 10.48 / 7.01 | 17.39 / 20.99 / 13.03 | 1.5 / 2.7 | 17.95 / 21.54 / 13.53 |
| Scale-Balanced-Grasp | author | ground-truth rectangle | 20.92 / 25.50 / 15.89 | 23.03 / 27.86 / 17.61 | 2.8 / 3.2 | 22.69 / 27.57 / 16.95 |
| EconomicGrasp | author | ground-truth rectangle | 22.38 / 27.45 / 16.62 | 23.10 / 28.29 / 17.00 | 3.3 / 3.4 | 21.67 / 26.62 / 15.90 |
| GSNet | local | ground-truth rectangle | 27.62 / 33.67 / 21.50 | 28.74 / 35.11 / 22.12 | 3.7 / 3.9 | none published |
| HGGD | author | none (`gt_assisted: none`) | 15.40 / 19.67 / 8.84 | 16.67 / 21.30 / 9.39 | 2.0 / 2.2 | none published |

`Reported` is Table IX of the GraspClutter6D paper
([arXiv 2504.06866](https://arxiv.org/abs/2504.06866) v3): models trained on
GraspNet-1Billion, tested on the 235 test scenes with the D435, "workspace-cropped
inputs", collision filter not stated. HGGD is not like-for-like with the other rows: it
reads the whole image and no ground truth shapes its input.

### Against the reported numbers

- **GraspNet-baseline: the reported number is reproduced by `cd` and not by `raw`.**
  `cd` is 0.56 AP below the reported value (0.2 standard errors), and AP0.8 and AP0.4
  agree as closely; `raw` is 9.2 AP below (6 standard errors). The reference test script
  filters collisions by default, and the one baseline whose code the GraspClutter6D
  authors released (Contact-GraspNet) is run with the same filter values (voxel 0.01 m,
  approach 0.05 m, threshold 0.01 m). The evidence says that number was produced with the
  collision filter.
- **Scale-Balanced-Grasp and EconomicGrasp: consistent, but the subset cannot say which
  variant was reported.** `raw` and `cd` differ by 0.7 to 2.1 AP, both lie within 0.6
  standard errors of the reported value, and the subset resolves about 3 AP.
- What cannot be checked from here: which checkpoints the authors evaluated (the
  GraspNet-1Billion column of their table equals the published numbers of
  Scale-Balanced-Grasp and EconomicGrasp to two decimals, 44.85 and 51.63, which
  suggests the public RealSense weights); their point sampling; and their input code for
  three of the four methods, which is not public.
- **GSNet and HGGD have no published GraspClutter6D number.** The GSNet row uses a local
  checkpoint (on 120 frames of GraspNet-1Billion `test_seen` it scores 68 / 71 AP `raw` /
  `cd`, a subset number as well), so it says how this checkpoint transfers, not how the
  published model does.

### Controlled variants on the same 312 frames

Each row changes one thing. The last column is the change of `raw` AP against the first
row of the method, with the standard error of the per-scene differences: rows of one
method are scored on the same frames, so their difference is resolved far better than
the 3 AP of a single number.

| Method | Variant | `raw` AP | `cd` AP | Change of `raw` |
|---|---|---|---|---|
| GraspNet-baseline | as above | 8.71 | 17.39 | |
| | GraspNet-1Billion crop | 8.91 | 18.76 | +0.2 (`cd`: +1.4 ± 0.4) |
| Scale-Balanced-Grasp | as above | 20.92 | 23.03 | |
| | GraspNet-1Billion crop | 24.04 | 26.88 | +3.1 ± 0.9 |
| EconomicGrasp | as above | 22.38 | 23.10 | |
| | GraspNet-1Billion crop | 23.20 | 23.84 | +0.8 ± 0.6 |
| | camera `realsense-d415` | 25.30 | 25.70 | +2.9 ± 1.7 |
| | camera `zivid` | 29.76 | 30.81 | +7.4 ± 1.8 |
| | Kinect weights (`...g1b-kinect-to-gc6d-d435...`) on the D435 | 21.52 | 22.10 | −0.9 ± 0.9 |
| | Kinect weights on `azure-kinect` | 15.72 | 16.31 | −5.8 ± 2.0 against the row above |
| GSNet | as above | 27.62 | 28.74 | |
| | GraspNet-1Billion crop | 28.23 | 29.29 | +0.6 ± 0.4 |
| | `model.gripper.max_width=0.14` | 28.38 | 29.47 | +0.8 ± 0.2 |

"GraspNet-1Billion crop" is the crop the weights were trained with, applied in the world
frame of the rig:

```bash
--opts "data.test.dataset.preprocess.1={type: SegmentationBoxWorkspace, margin: 0.02, invalid_depth_foreground: include}"
```

- **The crop matters for some methods and not for others.** The official crop keeps a
  wide margin of table and background: on view 0 of the subset scenes 30 % of its points
  are on objects, against 46 % for the box. Scale-Balanced-Grasp gains 3.1 AP from the
  tighter box and GraspNet-baseline 1.4 AP with the collision filter; for EconomicGrasp
  and GSNet the change is within its error. A table that mixes crops would rank methods
  by their input.
- **The sensor moves the result as much as the choice of method.** The same EconomicGrasp
  weights score 22.4 AP on the D435 and 29.8 on the Zivid.
- **The sensor a model was trained on does not carry over by name.** The Kinect weights
  score 21.5 AP on the D435 and 15.7 on the Azure Kinect of this dataset, whose
  3840 × 2160 frames give a crop of 2.4 million points of which a quarter are on objects.
  Why that pairing is the weakest was not investigated.
- **The width clamp costs GSNet 0.8 AP.** Letting the decoder open to 0.14 m raises `raw`
  from 27.62 to 28.38. The larger part of the gripper mismatch is in what the model
  learnt, which no test-time setting changes.
- **The collision filter is worth 8.7 ± 1.4 AP for GraspNet-baseline** and 0.7 to 2.1 AP
  for the other four methods.
- **The subset cannot rank Scale-Balanced-Grasp against EconomicGrasp** (−1.5 ± 1.3 AP
  `raw`; the paper has Scale-Balanced-Grasp 1.0 AP ahead). The GSNet checkpoint is ahead
  of EconomicGrasp by 5.2 ± 1.4 AP, with the caveat on its provenance above.

## The whole test split

Run so far for four methods with the authors' converted checkpoints: GraspNet-baseline,
Scale-Balanced-Grasp, EconomicGrasp and HGGD, 3 055 of 3 055 frames scored each. Their
numbers are `author-ckpt` rows of the transfer configs and stand in the
[model zoo](../model_zoo.md#graspclutter6d-transfers), the reported ones beside them; the
result records are in `configs/<method>/records/`. GSNet has no row: no checkpoint of its
authors has been scored, and the model trained here enters the zoo with its
GraspNet-1Billion result.

What the whole split adds to the subset: the reported value lies 0.81 AP above `cd` and
9.26 above `raw` for GraspNet-baseline, 0.02 AP below `cd` and 0.92 above `raw` for
Scale-Balanced-Grasp, and 0.02 AP below `raw` and 0.58 below `cd` for EconomicGrasp. So
the reported table is closest to the filtered run for two methods and to the unfiltered
run for the third. The paper does not state the collision filter, and whether it was
applied to some methods and not to others was not investigated. The controlled variants
above (crops, cameras, Kinect weights, gripper width) exist for the subset only.

A run took 8 to 12 minutes per method on the development machine (one GPU, four
evaluator processes, shared with other jobs):

```bash
for CONFIG in \
    economicgrasp/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e \
    graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e \
    scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e \
    hggd/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e \
    gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e
do
    grasppanda test $CONFIG --weights <the converted checkpoint of that method> \
        --opts evaluation.num_workers=8
done
```

A result of such a run is an `author-ckpt` number of the transfer config, stamped with
the protocol, the split, the camera, the variant and `gt_assisted`. It enters the zoo
with its result records:

```bash
python -m tools.record_result <work directory of the run> --method <directory below configs/>
```

checks that every frame was scored by clean code with hashed weights, copies the records
to `configs/<method>/records/` and prints the rows for the method's `metafile.yaml`.

## Checking a run against the official tool

`grasppanda test` writes `predictions/<variant>/scene_NNNN/<camera>/<view>.npy`. The
official tool names files by image id. To score a dump with `graspclutter6dAPI`, link it
into that layout:

```python
from grasppanda.datasets.graspclutter6d import GraspClutter6D
from grasppanda.evaluation.graspclutter6d import export_official_layout
from grasppanda.utils.paths import dataset_root

keys = GraspClutter6D.load_manifest(dataset_root("graspclutter6d")).keys("test", "realsense-d435")
export_official_layout("<run>/predictions/raw", "official_dump", keys)
# then: GraspClutter6DEval(root, "realsense-d435", split="test").eval_all("official_dump")
```

The other way round, `evaluation.evaluator: {type: GraspClutter6DEvaluator, layout:
official}` scores a dump that was written for the official tool. One difference is
deliberate: the official tool scores a frame without a file as zero, GraspPanda reports
it as missing and gives no headline number for an incomplete run.

## Training on GraspClutter6D

`gsnet/gsnet_sunet14_gc6d-d435_10e` and `economicgrasp/economicgrasp_sunet14_gc6d-d435_10e`
carry the GraspNet-1Billion recipe over and change the gripper to 0.14 m. They are this
toolbox's choice, not a reproduction: no GSNet or EconomicGrasp recipe for this dataset
has been published. Each has run for 20 iterations on one scene (the loss fell from 20.1
to 1.9 and from 28.6 to 2.5); **no model has been trained**. The assets and their cost
are listed in [Dataset preparation](datasets.md#derived-assets).

```bash
grasppanda prepare graspclutter6d --labels
grasppanda prepare graspclutter6d --graspness --camera realsense-d435
grasppanda prepare graspclutter6d --compact-labels --config economicgrasp/economicgrasp_sunet14_gc6d-d435_10e
grasppanda train gsnet/gsnet_sunet14_gc6d-d435_10e
grasppanda train economicgrasp/economicgrasp_sunet14_gc6d-d435_10e
```
