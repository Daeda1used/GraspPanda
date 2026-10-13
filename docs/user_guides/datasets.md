# Dataset preparation

GraspPanda reads two benchmarks: **GraspNet-1Billion** and **GraspClutter6D**. Both are
read in place, and everything derived from them is built by one command,
`grasppanda prepare`.

- Raw datasets are never modified. Nothing is written into a dataset tree.
- Derived labels (the grasp-label store, point-wise graspness and the label assets of
  individual methods) live in a separate asset store under `$GRASPPANDA_CACHE`. Each
  asset records its generator, its parameters and the preprocessing it assumes, and is
  found by a hash of those. A config that asks for other parameters asks for another
  asset; it can never read misaligned labels.
- Splits are explicit sets of `(scene, camera, frame)`. Validation data is a scene-level
  hold-out from the training scenes, never a test split.

## Where things are looked for

| What | Resolved from, in this order |
|---|---|
| GraspNet-1Billion | `root` in the dataset config, `$GRASPPANDA_GRASPNET1B_ROOT`, `$GRASPPANDA_DATA/graspnet1b`, `./data/graspnet1b` |
| GraspClutter6D | `root` in the dataset config, `$GRASPPANDA_GRASPCLUTTER6D_ROOT`, `$GRASPPANDA_DATA/graspclutter6d`, `./data/graspclutter6d` |
| Derived assets | `$GRASPPANDA_CACHE` |

## `grasppanda prepare`

```bash
grasppanda prepare <dataset> --check                 # what the tree and the assets lack
grasppanda prepare <dataset> --labels                # dense grasp labels as a memory-mapped store
grasppanda prepare <dataset> --graspness --camera C  # point-wise graspness maps
grasppanda prepare <dataset> --compact-labels --config <config with compact_labels>
grasppanda prepare <dataset> --tolerance      --config <config with tolerance>
grasppanda prepare <dataset> --view-grasps --camera C --config <config with view_grasps>
grasppanda prepare <dataset> --scale-statistic --config <config with scale_statistic>
```

Every action reads a dataset config: the shipped base config of the dataset and camera,
or the one named with `--config`. What is built is exactly what that config later asks
for. `--split` and `--scenes 5,9,13` restrict the scenes; generation is resumable and
finished shards are skipped. Evaluation needs none of these assets: a test loader reads
depth (and, for the reference workspace crop, the segmentation) only.

| Asset | Flag | Read by | Depends on the camera |
|---|---|---|---|
| Grasp-label store | `--labels` | every training recipe, and every other generator | no |
| Point-wise graspness | `--graspness` | GSNet, EconomicGrasp | yes, and on the preprocessing |
| Compact grasp labels | `--compact-labels` | EconomicGrasp | no |
| Grasp tolerance | `--tolerance` | GraspNet-baseline, Scale-Balanced-Grasp | no |
| Per-view grasps | `--view-grasps` | HGGD | yes |
| Scale statistic | `--scale-statistic` | Scale-Balanced-Grasp | yes |

## GraspNet-1Billion

The official download, 190 scenes with 256 frames per scene and camera:

```text
scenes/scene_XXXX/object_id_list.txt
scenes/scene_XXXX/{realsense,kinect}/{rgb,depth,label}/NNNN.png
scenes/scene_XXXX/{realsense,kinect}/meta/NNNN.mat          intrinsics, depth factor, poses
scenes/scene_XXXX/{realsense,kinect}/annotations/NNNN.xml   poses, four decimals
scenes/scene_XXXX/{realsense,kinect}/{camera_poses,cam0_wrt_table}.npy
grasp_label/NNN_labels.npz
collision_label/scene_XXXX/collision_labels.npz
models/NNN/nontextured.ply,  dex_models/NNN.pkl
```

Splits (shipped in `grasppanda/datasets/splits/graspnet1b.yaml`): `train` (scenes 0 to
99), `test_seen` (100 to 129), `test_similar` (130 to 159), `test_novel` (160 to 189),
`test` (their union), `train_dev` / `val_dev` (90 and 10 training scenes) and `sample`.

```bash
grasppanda prepare graspnet1b --check
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense
```

The method-specific assets are listed with their commands in each method's
`configs/<method>/README.md`.

## GraspClutter6D

[GraspClutter6D](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D)
(RA-L 2025, [arXiv 2504.06866](https://arxiv.org/abs/2504.06866), data licence
CC BY-SA 4.0) holds 1 000 scenes in BOP layout. Its grasp labels have the format of
GraspNet-1Billion, which is why every label generator of this toolbox runs on it
unchanged. Download and unpack it as its README says; the reader needs

```text
split_info/grasp_{train,test}_scene_ids.json    the grasp splits (authoritative)
split_info/obj_ids_per_scene.json
scenes/<scene:06d>/{rgb,depth,label}/<image:06d>.png
scenes/<scene:06d>/{mask,mask_visib}/<image:06d>_<instance:06d>.png
scenes/<scene:06d>/scene_camera.json    per image: cam_K, depth_scale, cam_{R,t}_w2c
scenes/<scene:06d>/scene_gt.json        per image and instance: cam_{R,t}_m2c, obj_id
grasp_label/obj_<object:06d>_labels.npz
collision_label/<scene:06d>.npz         one array per instance, in scene_gt order
models_m/obj_<object:06d>.ply           object models in metres
dex_models/obj_<object:06d>.pkl         force-closure data of the evaluator
```

`grasppanda prepare graspclutter6d --check` counts all of this for the 648 scenes of the
grasp splits and then reports the derived assets of every camera.

### Cameras

A scene has 13 annotated views, and every view was taken by four cameras. All 52 images
of a scene share the directories; the camera is encoded in the image id,
`image = 4 * view + c`. A `SampleKey` holds the camera name and the view (0 to 12).

| Camera (`camera:`) | `c` | Image | Raw depth unit | Focal length (px) | Base config |
|---|---|---|---|---|---|
| `realsense-d415` | 1 | 1920 × 1080 | 1 mm | 1404 or 1429 | `graspclutter6d_d415.yaml` |
| `realsense-d435` | 2 | 1920 × 1080 | 1 mm | 1394 or 1415 | `graspclutter6d_d435.yaml` |
| `azure-kinect` | 3 | 3840 × 2160 | 0.1 mm | 1996 or 2112 | `graspclutter6d_kinect.yaml` |
| `zivid` | 4 | 1920 × 1200 | 0.1 mm | 2790 | `graspclutter6d_zivid.yaml` |

Intrinsics and depth scale are read per image from `scene_camera.json`. Three of the
cameras have two calibrations in the grasp splits (in the scenes inspected, view 0 has
one and views 1 to 12 the other), so one constant camera matrix per camera, as
GraspNet-1Billion pipelines often hard-code, would be wrong here. The depth scale was
the same on all 8 424 images of each camera. The paper reports its benchmark on the
RealSense D435.

### Splits

| Split | Scenes | Frames per camera | Object models | Source |
|---|---|---|---|---|
| `train` | 413 | 5 369 | 123 (ids 69 to 198) | `split_info/grasp_train_scene_ids.json` |
| `test` | 235 | 3 055 | 68 (ids 1 to 68) | `split_info/grasp_test_scene_ids.json` |
| `all` | 648 | 8 424 | 191 | union of the two |
| `sample` | 1 (scene 5) | 13 | 6 | first training scene, for smoke tests |
| `test_mini` | 24 | 312 | 66 of the 68 | every tenth test scene by rank; shipped |
| `val_dev` | 41 | 533 | 110 of the 123 | every tenth training scene by rank; shipped |
| `train_dev` | 372 | 4 836 | 123 | `train` without `val_dev` |

- **The test objects are all unseen.** No object model occurs in both splits: the
  benchmark has no "seen" part, unlike GraspNet-1Billion.
- The three subsets are fixed scene lists in `grasppanda/datasets/splits/graspclutter6d.yaml`.
  They exist only when the split files of the tree contain them. **`test_mini` is for
  quick checks; a number measured on it is a subset number, not a benchmark result.**
  Per-scene AP varies widely on this dataset (from below 1 to above 60 for one model),
  so the mean over 24 scenes carries a standard error of about 3 AP.
- All objects of `val_dev` are training objects, so it measures fit to new arrangements,
  not to new objects.
- The other 352 scenes of the dataset are in neither grasp split and are not read.
- The paper states 132 training objects. The split files give 123: object ids 112 to
  117, 191, 199 and 200 occur in no scene of either grasp split.

### What a frame holds

- Scenes are denser than in GraspNet-1Billion: 14.5 object instances per scene on
  average (5 to 20), and in 84 % of the scenes at least one object model occurs more than
  once. The `label` image stores object ids and cannot tell two copies apart, so the
  reader builds the instance map from the per-instance visibility masks (`mask_visib`),
  and grasp labels, collision labels and compact labels are keyed by **instance**
  (position in `scene_gt.json`), not by object id.
- The instance list and its order are the same for all 52 images of a scene (checked on
  all 648 scenes). The reader and the evaluator refuse a frame for which that fails.
- Lengths in the JSON files are millimetres; poses leave the reader in metres.
- `cam_R_w2c` / `cam_t_w2c` hold the camera pose in the world frame (camera to world),
  contrary to what the names say. The world frame is fixed to the rig, not to the table:
  bounds of a `TableBoxWorkspace` are world coordinates here.
- `scene_camera.json` also has entries `-4` to `0`; they have no image and are ignored.

### Derived assets

Built on three training scenes (5, 9 and 13; 11 object models, 49 instances) on the
development machine while it was heavily loaded, so the times are upper bounds:

| Asset | Command (add `--scenes 5,9,13` for the three scenes) | Three scenes | Whole `train` split, extrapolated |
|---|---|---|---|
| Grasp-label store | `grasppanda prepare graspclutter6d --labels` | 2.8 GB | about 55 GB (236 MB per object model, 62 MB per scene) |
| Graspness, one camera | `... --graspness --camera realsense-d435` | 2.2 MB, 3 min with 3 workers | about 0.3 GB, 5 h with 4 workers |
| Compact labels | `... --compact-labels --config economicgrasp/economicgrasp_sunet14_gc6d-d435_10e` | 24 MB, 11 s | about 3 GB, 25 min |
| Tolerance | `... --tolerance --config _base_/datasets/graspclutter6d_d435_tolerance.yaml` | 332 MB, 1 min with 4 workers | about 4 GB, 10 min |
| Per-view grasps, one camera | `... --view-grasps --camera realsense-d435 --config _base_/datasets/graspclutter6d_d435_hggd.yaml` | 96 kB, 1 min with 3 workers | about 15 MB, 1.5 h with 4 workers |

The whole-split figures are extrapolations from the three scenes, not measurements. The
label store is the large one; only training reads it. The evaluator keeps a cache of its
own, extracted from `dex_models/` and `models_m/` on first use: about 7 MB per object
model, 0.5 GB for the test objects.

Two facts measured on those scenes that matter for training:

- Only 9.6 % of the annotated grasp points keep a collision-free grasp of friction
  ≤ 0.4 (10 401 of 108 094); on GraspNet-1Billion 39 % do. The scenes are far more
  cluttered, so EconomicGrasp's compact labels supervise far fewer seeds per frame.
- The per-view labels of HGGD come to 17 to 67 per frame (GraspNet-1Billion scene 0:
  59), from 471 to 1 964 candidate grasps per scene.

### Dataset configs

| Config (`configs/_base_/datasets/`) | Recipe carried over from | Cloud | Assets it reads for training |
|---|---|---|---|
| `graspclutter6d_{d415,d435,kinect,zivid}.yaml` | Graspness (GSNet) | 15 000 points, official workspace crop | label store, graspness |
| `graspclutter6d_d435_tolerance.yaml` | GraspNet-baseline, Scale-Balanced-Grasp | 20 000 points, official workspace crop | label store, tolerance |
| `graspclutter6d_d435_hggd.yaml` | HGGD | colour and depth images, 25 600 points, no crop | per-view grasps |
| EconomicGrasp: `configs/economicgrasp/economicgrasp_sunet14_gc6d-d435_10e.yaml` | EconomicGrasp | 20 000 points, official workspace crop | compact labels, graspness |

These carry the GraspNet-1Billion recipe of each method family over and change only what
the dataset fixes. None reproduces a published GraspClutter6D training recipe: none
exists for the ported methods. HGGD's network image is 640 × 360, and the detector takes
any sensor image that is an integer multiple of it: the D415, D435 (×3) and Azure Kinect
(×6) images fit, the Zivid images (1920 × 1200) do not.

## What differs between the two benchmarks

Measured values are medians over 20 test frames of GraspNet-1Billion (two frames of
every ninth test scene, RealSense unless a camera is named) and over 104 test frames of
GraspClutter6D (all 13 views of every third `test_mini` scene).

| | GraspNet-1Billion | GraspClutter6D |
|---|---|---|
| Scenes, frames | 190 scenes × 256 frames per camera | 648 scenes in the grasp splits × 13 views per camera |
| Cameras | `realsense` (D435), `kinect` (Azure Kinect), one directory each | D415, D435, Azure Kinect, Zivid, interleaved by image id |
| Image | 1280 × 720 | 1920 × 1080, 3840 × 2160 (Kinect), 1920 × 1200 (Zivid) |
| Intrinsics | one calibration per camera | per image; two calibrations for three of the cameras |
| Raw depth unit | 1 mm | 1 mm (RealSense), 0.1 mm (Azure Kinect, Zivid); per image |
| Camera to objects | 0.44 m (0.31 to 0.52) | 0.70 m (0.36 to 1.04) on the D435; the paper states 0.31 to 1.37 m over all cameras |
| Size of a pixel on the objects | 0.47 mm (RealSense), 0.74 mm (Kinect) | 0.50 mm (D435), 0.47 (D415), 0.36 (Kinect), 0.26 (Zivid) |
| Object ids | 0 to 87; the segmentation stores id + 1 | 1 to 200; the segmentation stores the id |
| Instances | an object occurs once per scene | repeated objects; instances from `mask_visib` |
| Label files | `grasp_label/NNN_labels.npz`, `collision_label/scene_XXXX/collision_labels.npz` | `grasp_label/obj_NNNNNN_labels.npz`, `collision_label/NNNNNN.npz` |
| Label grid | 300 views × 12 angles × 4 depths | the same |
| Label widths | gripper of 0.10 m | up to 0.15 m; gripper of 0.14 m |
| Object poses | `meta/*.mat` or `annotations/*.xml` | `scene_gt.json` (millimetres) |
| Table frame | `cam0_wrt_table.npy`, z up, origin on the table | the rig's world frame; the table is not at z = 0 |
| Reference workspace crop | 3-D box around the ground-truth segmentation, 2 cm margin (`SegmentationBoxWorkspace`) | image rectangle around the ground-truth segmentation, 10 % margin (`SegmentationRectWorkspace`) |
| Cloud in that crop | 554 000 points, 0.48 × 0.35 × 0.26 m, 39 % on objects | 908 000 points, 0.91 × 0.57 × 0.81 m, 32 % on objects (D435) |
| Splits | train / seen / similar / novel | train / test, disjoint objects |
| Evaluator | `GraspNet1BEvaluator`, protocol `graspnet1b-sv-v1` | `GraspClutter6DEvaluator`, protocol `graspclutter6d-sv-v1` |
| Ranks and frictions | top 50; μ = 0.2, 0.4, …, 1.2 | the same |
| Width clip | 0.10 m | 0.14 m |
| Foreground filter | none | grasps 5 cm or more from every object model are dropped before NMS |
| Collision scene | object models and a synthetic table | object models only |

Both reference crops are derived from the **ground-truth** segmentation, at test time
too. Results on them are stamped `gt_assisted: [workspace]`.

The evaluator constants are those of the official tools (`graspnetAPI` 1.2.x and
`graspclutter6dAPI` at `ddb2ee7`). For GraspClutter6D the equality is checked stage by
stage on label-derived grasps for all four cameras and on the dumps that `grasppanda
test` wrote for several detectors (`tests/integration/test_eval_clutter_parity.py`).
How a model trained on one benchmark is tested on the other is the subject of
[Cross-dataset evaluation](cross_dataset.md).
