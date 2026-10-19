# Cross-dataset evaluation

**Goal:** test a model trained on GraspNet-1Billion on GraspClutter6D, with nothing changed
but the frames that are scored and the evaluator.

**You need:** GraspClutter6D on disk ([Datasets](datasets.md#graspclutter6d)) and the weights
of a GraspNet-1Billion config: trained with [Train and test](train_test.md), or the
released checkpoint converted as its [method page](../methods/index.md) shows. Testing
needs no derived asset.

## 1. Run the transfer config of your method

=== "GraspNet-baseline"

    ```bash
    grasppanda test \
        graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense-to-gc6d-d435_18e \
        --weights <weights of graspnet-baseline_pointnet2_g1b-realsense_18e>
    ```

=== "GSNet"

    ```bash
    grasppanda test \
        gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e \
        --weights <weights of gsnet_sunet14_g1b-realsense_10e>
    ```

=== "Scale-Balanced-Grasp"

    ```bash
    grasppanda test \
        scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e \
        --weights <weights of sbg_pointnet2-transformer_g1b-realsense_18e>
    ```

=== "EconomicGrasp"

    ```bash
    grasppanda test \
        economicgrasp/economicgrasp_sunet14_g1b-realsense-to-gc6d-d435_10e \
        --weights <weights of economicgrasp_sunet14_g1b-realsense_10e>
    grasppanda test \
        economicgrasp/economicgrasp_sunet14_g1b-kinect-to-gc6d-d435_10e \
        --weights <weights of economicgrasp_sunet14_g1b-kinect_10e>
    ```

=== "FineGrasp"

    No transfer config ships for FineGrasp yet, and FineGrasp has not been tested on
    GraspClutter6D here. A transfer config is a short child of the method config; the fold
    *Why a config and not a list of overrides* below shows one.

=== "HGGD"

    ```bash
    grasppanda test \
        hggd/hggd_r34w8_g1b-realsense-to-gc6d-d435_15e \
        --weights <weights of hggd_r34w8_g1b-realsense_15e>
    ```

    HGGD reads the whole image, as in training: no ground truth shapes its input.

Each transfer config scores the RealSense D435 frames of the `test` split: 235 scenes ×
13 views = 3 055 frames. A run took 8 to 12 minutes per method on the development machine
(one GPU, shared with other jobs). The result records say
`protocol: graspclutter6d-sv-v1`, the camera, the variant and `gt_assisted`.

## 2. Try a subset or another camera

```bash
--opts data.test.dataset.split=test_mini     # 24 scenes, 312 frames
--opts data.test.dataset.camera=zivid        # or realsense-d415, azure-kinect
--opts evaluation.num_workers=8              # evaluator processes
```

A number measured on `test_mini` is a subset number, not a benchmark result.

## What changes with the dataset, and what must not

- **Changes:** the frames (`GraspClutter6D`, `realsense-d435`, `test`), the intrinsics and
  depth scale (read per image), the workspace crop (`SegmentationRectWorkspace`, the crop of
  the official tool; HGGD crops nothing on either dataset) and the evaluator (width clip
  0.14 m, foreground filter, no table).
- **Fixed:** the model with every length and grid it was trained in, its gripper of
  0.10 m, the points per cloud and input features, both post-processing variants and the
  point-sampling seed.

??? note "Every row of it, and why"

    | | In the transfer config | Why |
    |---|---|---|
    | Frames | **changes**: `GraspClutter6D`, `realsense-d435`, `test` | The benchmark. The D435 is the camera of the paper's table and the sensor model of GraspNet-1Billion's `realsense`. |
    | Intrinsics, depth scale | **changes**, read per image | Nothing in a model or a transform holds a camera constant; `DepthToCloud` and HGGD read the calibration of the frame. |
    | Workspace crop | **changes** to `SegmentationRectWorkspace` (margin 0.1) | The crop of the official tool, and of the authors' released baseline code. GraspNet-1Billion's crop (`SegmentationBoxWorkspace`) is a box in a table frame that this dataset does not have; see the measured effect below. HGGD has no crop on either dataset. |
    | Evaluator | **changes** to `GraspClutter6DEvaluator` | Width clip 0.14 m, foreground filter, no table: the protocol of the benchmark. |
    | Model, including every length | **fixed** | The units the weights were trained in: the ball-query radii of the PointNet++ encoders (GraspNet-baseline, Scale-Balanced-Grasp), the 5 mm voxels of the sparse U-Net (GSNet, EconomicGrasp, FineGrasp), the crop radius of every seed-based grouper, the 300 views and 4 depth bins of the five seed-based methods, HGGD's anchors and grid. |
    | Gripper of the model (0.10 m) | **fixed** | It bounds the labels the model learnt from and clamps its decoded widths. See "The gripper" below. |
    | Points per cloud, input features | **fixed** | Part of the recipe (20 000 points for GraspNet-baseline, Scale-Balanced-Grasp, EconomicGrasp and FineGrasp, 15 000 for GSNet, 25 600 for HGGD); fewer or more points change the density the encoder was trained on. |
    | HGGD network image (640 × 360) | **fixed** | The 1920 × 1080 image is reduced by 3 instead of 2; the focal length of the network image comes out nearly the same (465 to 472 px against 464 px). |
    | Post-processing (`raw`, `cd`) | **fixed** | The collision filter of each reference, with its own voxel size, approach distance and threshold. |
    | Point-sampling seed | **fixed** (0) | The same points every run. |

    What shifts without being a setting, and is therefore not corrected:

    - **Distance.** Objects are 0.70 m from the D435 in the median (0.36 to 1.04 m), against
      0.44 m (0.31 to 0.52 m) in GraspNet-1Billion. The image is larger, so a pixel covers
      about the same length on an object (0.50 mm against 0.47 mm); how much noisier the
      depth is at that distance was not measured. No depth range is clipped on either
      dataset.
    - **Extent and density.** The reference crop here spans 0.91 × 0.57 × 0.81 m, about twice
      that of GraspNet-1Billion in each direction (0.48 × 0.35 × 0.26 m), and a scene holds
      14.5 object instances instead of 9.0. With the same point budget an object gets
      roughly half the points.
    - **Objects.** All 68 test objects are unseen, also by a model trained on
      GraspClutter6D.

??? note "The gripper"

    The benchmark gripper opens to 0.14 m and the evaluator clips widths there. A model
    trained on GraspNet-1Billion was supervised with grasps up to 0.10 m and clamps what it
    decodes to 0.10 m. On the 68 test objects, 38 % of the valid label grasps are wider than
    0.10 m (4 % are wider than 0.14 m); for 18 objects more than half are, for 3 objects
    more than 90 %. This part of the transfer gap is structural.
    `--opts model.gripper.max_width=0.14` only moves the clamp of the decoder; it is a
    controlled variant, measured below for GSNet, not the default.

??? note "Why a config and not a list of overrides"

    A transfer config inherits the method's GraspNet-1Billion config and replaces two
    nodes, `data.test` and `evaluation.evaluator`. GraspNet-baseline's, without comments:

    ```yaml
    _base_:
      - graspnet-baseline_pointnet2_g1b-realsense_18e.yaml

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
      interval: 0
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

## Subset results

Measured on `test_mini` before the whole split was run, with controlled variants that
exist for the subset only: the workspace crop, the camera, Kinect weights and the gripper
width.

??? note "Subset numbers and controlled variants"

    Everything in this fold was measured on **`test_mini`**: 24 of the 235 test scenes, 312
    frames per camera. These are not benchmark results and do not enter the model zoo.
    Per-scene AP ranges from below 1 to above 60 for one model, so the mean over 24 scenes
    has a scene-clustered standard error of about 3 AP (given per row).

    AP / AP0.8 / AP0.4 in percent, RealSense D435, protocol `graspclutter6d-sv-v1`, 312 of
    312 frames scored. `raw`: predictions as emitted; `cd`: with the reference's model-free
    collision filter. Provenance of the weights: `author` is the authors' released
    GraspNet-1Billion RealSense checkpoint, converted; `local` is a checkpoint the
    maintainers trained with the upstream code on their own variant of the graspness labels
    (epoch 18), used because no usable author checkpoint of GSNet is at hand.

    | Method | Weights | Input crop | `raw` | `cd` | SE of AP (`raw` / `cd`) | Reported, whole test split |
    |---|---|---|---|---|---|---|
    | GraspNet-baseline | author | ground-truth rectangle | 8.71 / 10.48 / 7.01 | 17.39 / 20.99 / 13.03 | 1.5 / 2.7 | 17.95 / 21.54 / 13.53 |
    | GSNet | local | ground-truth rectangle | 27.62 / 33.67 / 21.50 | 28.74 / 35.11 / 22.12 | 3.7 / 3.9 | none published |
    | Scale-Balanced-Grasp | author | ground-truth rectangle | 20.92 / 25.50 / 15.89 | 23.03 / 27.86 / 17.61 | 2.8 / 3.2 | 22.69 / 27.57 / 16.95 |
    | EconomicGrasp | author | ground-truth rectangle | 22.38 / 27.45 / 16.62 | 23.10 / 28.29 / 17.00 | 3.3 / 3.4 | 21.67 / 26.62 / 15.90 |
    | HGGD | author | none (`gt_assisted: none`) | 15.40 / 19.67 / 8.84 | 16.67 / 21.30 / 9.39 | 2.0 / 2.2 | none published |

    `Reported` is Table IX of the GraspClutter6D paper
    ([arXiv 2504.06866](https://arxiv.org/abs/2504.06866) v3): models trained on
    GraspNet-1Billion, tested on the 235 test scenes with the D435, "workspace-cropped
    inputs", collision filter not stated. HGGD is not like-for-like with the other rows: it
    reads the whole image and no ground truth shapes its input.

    **Against the reported numbers.**

    - **GraspNet-baseline: the reported number is reproduced by `cd` and not by `raw`.**
      `cd` is 0.56 AP below the reported value (0.2 standard errors), and AP0.8 and AP0.4
      agree as closely; `raw` is 9.2 AP below (6 standard errors). The reference test script
      filters collisions by default, and the one baseline whose code the GraspClutter6D
      authors released (Contact-GraspNet) is run with the same filter values (voxel 0.01 m,
      approach 0.05 m, threshold 0.01 m). The evidence says that number was produced with
      the collision filter.
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

    **Controlled variants on the same 312 frames.** Each row changes one thing. The last
    column is the change of `raw` AP against the first row of the method, with the standard
    error of the per-scene differences: rows of one method are scored on the same frames,
    so their difference is resolved far better than the 3 AP of a single number.

    | Method | Variant | `raw` AP | `cd` AP | Change of `raw` |
    |---|---|---|---|---|
    | GraspNet-baseline | as above | 8.71 | 17.39 | |
    | | GraspNet-1Billion crop | 8.91 | 18.76 | +0.2 (`cd`: +1.4 ± 0.4) |
    | GSNet | as above | 27.62 | 28.74 | |
    | | GraspNet-1Billion crop | 28.23 | 29.29 | +0.6 ± 0.4 |
    | | `model.gripper.max_width=0.14` | 28.38 | 29.47 | +0.8 ± 0.2 |
    | Scale-Balanced-Grasp | as above | 20.92 | 23.03 | |
    | | GraspNet-1Billion crop | 24.04 | 26.88 | +3.1 ± 0.9 |
    | EconomicGrasp | as above | 22.38 | 23.10 | |
    | | GraspNet-1Billion crop | 23.20 | 23.84 | +0.8 ± 0.6 |
    | | camera `realsense-d415` | 25.30 | 25.70 | +2.9 ± 1.7 |
    | | camera `zivid` | 29.76 | 30.81 | +7.4 ± 1.8 |
    | | Kinect weights (`...g1b-kinect-to-gc6d-d435...`) on the D435 | 21.52 | 22.10 | −0.9 ± 0.9 |
    | | Kinect weights on `azure-kinect` | 15.72 | 16.31 | −5.8 ± 2.0 against the row above |

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
    - **The sensor moves the result as much as the choice of method.** The same
      EconomicGrasp weights score 22.4 AP on the D435 and 29.8 on the Zivid.
    - **The sensor a model was trained on does not carry over by name.** The Kinect weights
      score 21.5 AP on the D435 and 15.7 on the Azure Kinect of this dataset, whose
      3840 × 2160 frames give a crop of 2.4 million points of which a quarter are on
      objects. Why that pairing is the weakest was not investigated.
    - **The width clamp costs GSNet 0.8 AP.** Letting the decoder open to 0.14 m raises `raw`
      from 27.62 to 28.38. The larger part of the gripper mismatch is in what the model
      learnt, which no test-time setting changes.
    - **The collision filter is worth 8.7 ± 1.4 AP for GraspNet-baseline** and 0.7 to 2.1 AP
      for the other four methods of the table.
    - **The subset cannot rank Scale-Balanced-Grasp against EconomicGrasp** (−1.5 ± 1.3 AP
      `raw`; the paper has Scale-Balanced-Grasp 1.0 AP ahead). The GSNet checkpoint is ahead
      of EconomicGrasp by 5.2 ± 1.4 AP, with the caveat on its provenance above.

## The whole test split

Run so far for five methods, 3 055 of 3 055 frames scored each: GraspNet-baseline,
Scale-Balanced-Grasp, EconomicGrasp and HGGD with the authors' converted checkpoints
(`author-ckpt`), and GSNet with the model trained from scratch by this toolbox
(`reproduced`; its weights are not published yet). AP / AP0.8 / AP0.4 in percent, RealSense
D435; the rows, with the reported numbers beside them, stand in the
[model zoo](../model_zoo.md#graspclutter6d-transfers) and the result records in
`configs/<method>/records/`.

| Method | Weights | `raw` | `cd` | Reported |
|---|---|---|---|---|
| GraspNet-baseline | `author-ckpt` | 8.69 / 10.35 / 6.96 | 17.14 / 20.60 / 12.86 | 17.95 / 21.54 / 13.53 |
| GSNet | `reproduced` | 25.58 / 31.38 / 19.42 | 26.87 / 33.00 / 20.04 | none published |
| Scale-Balanced-Grasp | `author-ckpt` | 21.77 / 26.49 / 16.46 | 22.71 / 27.61 / 17.01 | 22.69 / 27.57 / 16.95 |
| EconomicGrasp | `author-ckpt` | 21.69 / 26.64 / 15.79 | 22.25 / 27.30 / 16.06 | 21.67 / 26.62 / 15.90 |
| HGGD | `author-ckpt` | 14.25 / 17.41 / 9.23 | 14.56 / 17.79 / 9.33 | none published |

The four seed-based rows read the cloud inside the ground-truth rectangle of the official
tool (`gt_assisted: [workspace]`). HGGD is not like-for-like with them: it reads the whole
image (`gt_assisted: none`), and its `cd` is its own filter followed by NMS. The GSNet row is one training run with one seed; its
weights are another model than the `local` checkpoint of the subset numbers above.

The reported value lies 0.81 AP above `cd` and 9.26 above `raw` for GraspNet-baseline,
0.02 AP below `cd` and 0.92 above `raw` for Scale-Balanced-Grasp, and 0.02 AP below `raw`
and 0.58 below `cd` for EconomicGrasp. So the reported table is closest to the filtered run
for two methods and to the unfiltered run for the third. The paper does not state the
collision filter, and whether it was applied to some methods and not to others was not
investigated.

??? note "Entering a result into the model zoo"

    A result of such a run is an `author-ckpt` number of the transfer config, stamped with
    the protocol, the split, the camera, the variant and `gt_assisted`. It enters the zoo
    with its result records:

    ```bash
    python -m tools.record_result <work directory of the run> --method <directory below configs/>
    ```

    which checks that every frame was scored by clean code with hashed weights, copies the
    records to `configs/<method>/records/` and prints the rows for the method's
    `metafile.yaml`.

## Checking a run against the official tool

`grasppanda test` writes `predictions/<variant>/scene_NNNN/<camera>/<view>.npy`; the
official tool names files by image id. GraspPanda links a dump into the official layout,
and reads one written for it.

??? note "Both directions"

    To score a dump with `graspclutter6dAPI`:

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

Two configs carry a GraspNet-1Billion recipe over and change the gripper to 0.14 m:
`gsnet/gsnet_sunet14_gc6d-d435_10e` and `economicgrasp/economicgrasp_sunet14_gc6d-d435_10e`.
They are this toolbox's choice, not a reproduction: no recipe of these methods for this
dataset has been published, and no model has been trained with them.

??? note "Commands and what has run"

    Each config has run for 20 iterations on one scene (the loss fell from 20.1 to 1.9 and
    from 28.6 to 2.5). The assets and their cost are listed under
    [Datasets › GraspClutter6D](datasets.md#graspclutter6d).

    ```bash
    grasppanda prepare graspclutter6d --labels
    grasppanda prepare graspclutter6d --graspness --camera realsense-d435
    grasppanda prepare graspclutter6d --compact-labels --config economicgrasp/economicgrasp_sunet14_gc6d-d435_10e
    grasppanda train gsnet/gsnet_sunet14_gc6d-d435_10e
    grasppanda train economicgrasp/economicgrasp_sunet14_gc6d-d435_10e
    ```

## Next

- [Read an analysis](analysis.md): where the AP of a result comes from.
- [Model zoo](../model_zoo.md#graspclutter6d-transfers): the transfer rows beside the
  reported numbers.
