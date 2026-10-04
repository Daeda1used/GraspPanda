# Datasets and observation protocols

Select a dataset first. The browser then offers its methods, cameras, splits and compatible operations. Dataset paths are remembered independently during the session. A shared runtime does not make different observation or supervision protocols interchangeable.

| Dataset | Observations | Methods in this integration | Protocol |
|---|---|---|---|
| [GraspNet-1B](https://graspnet.net/) | Real RGB-D; single-view, fused-view and temporal adapters | See [Methods & papers](METHODS.md) | Native GraspNet splits and evaluation |
| [GraspClutter6D](https://sites.google.com/view/graspclutter6d) | 1,000 real scenes, 52,000 RGB-D images, four cameras, 9.3 billion grasps | Native Contact-GraspNet; GraspNet Baseline and Graspness transfer | Published 413-scene training / 235-scene grasp test split |
| [ZeroGrasp-11B](https://github.com/sh8/ZeroGrasp) | One million synthetic observations, 12,000 objects, 11.3 billion grasps | Native ZeroGrasp with interchangeable image encoders | Released training shards; no held-out grasp AP is implied |
| [DexGraspNet 2.0](https://pku-epic.github.io/DexGraspNet2.0/) | 1,319 objects, 8,270 synthetic scenes, approximately 427 million dexterous grasps | Native diffusion; author ISAGrasp and GraspTTA baselines | Rendered depth and 16-joint LEAP-hand prediction; simulation evaluation remains an upstream workflow |
| [Jacquard](https://jacquard.liris.cnrs.fr/database.php) | Synthetic RGB-D; 54,485 scenes and 4.97 million rectangle annotations | GR-ConvNet RGB-D and depth | Explicit train/validation split; planar IoU evaluation |
| [SuctionNet-1B](https://graspnet.net/suction) | Shared GraspNet RGB-D scenes; separate suction supervision | Native SuctionNet RGB-D | RealSense prediction, training and resume; native suction outputs |
| [TARGO](https://huggingface.co/datasets/randing2000/TARGO) | Synthetic target grasping; 2.47 million released training trials | TARGO-Net + frozen AdaPoinTr | Native prediction and training; scene-disjoint labelled validation |
| [GraspGen](https://github.com/NVlabs/GraspGen) | Object-centric rendered partial depth and grasp labels | Native GraspGen generator/discriminator | Separate stage training; simulation evaluation remains upstream |
| [ACRONYM](https://github.com/NVlabs/acronym) | 8,872 objects and 17.7 million simulated parallel-jaw grasps | GraspLDM partial-cloud VAE/diffusion | Toolbox depth rendering and all-success-label training; native networks and objectives |

GraspClutter6D is a 2025 RA-L dataset, also presented at ICRA 2026 ([paper](https://arxiv.org/pdf/2504.06866)). ZeroGrasp is CVPR 2025 ([paper](https://arxiv.org/pdf/2504.10857)). Scale describes the publishers' datasets, not the number of observations evaluated by this toolbox.

DexGraspNet 2.0 is CoRL 2024 ([paper](https://arxiv.org/pdf/2410.23004)); the pinned dataset snapshot was updated in February 2025. Its original release was in 2024.

## Components across datasets

| Reusable implementation | GraspNet-1B | GraspClutter6D | ZeroGrasp-11B | DexGraspNet 2.0 |
|---|---|---|---|---|
| PointNet | Dense and sparse adapters | Contact seeds | — | Sparse features |
| Sonata / PTv3 | Dense and sparse adapters | Contact seeds | — | Sparse features |
| Sparse U-Net 18 | Graspness features | Graspness transfer | — | Native sparse features |
| ConvNeXt V2, RepViT, MobileNetV4 | HGGD / RNG image pyramids | — | Calibrated dense RGB features | — |

Shared implementations retain dataset-specific adapters. A dense contact seed, a sparse voxel row and an image pixel are different contracts. Replacement encoders require training; `reuse_unchanged` retains only compatible unchanged checkpoint modules. A trained toolbox checkpoint then loads with `strict` and its saved module configuration.

## SuctionNet-1B

SuctionNet shares GraspNet's `scenes/` and `models/`. Its seal, wrench and suction-collision labels are separate. The integrated RGB-D model uses the **RealSense** release. [Official dataset, download links and terms](https://graspnet.net/suction).

Start with the [GraspNet RGB-D scene and object-model downloads](DOWNLOADS.md#graspnet-1b). If those files are already present, link them into a dedicated suction root to avoid duplicating the dataset:

```bash
mkdir -p /data/SuctionNet-1B
ln -s /data/GraspNet-1B/scenes /data/SuctionNet-1B/scenes
ln -s /data/GraspNet-1B/models /data/SuctionNet-1B/models
./panda data suctionnet1b --root /data/SuctionNet-1B --fetch
./panda weights suctionnet_rgbd --camera realsense
./panda init --dataset suctionnet1b --dataset-root /data/SuctionNet-1B -o suction.local.yaml
./panda check suction.local.yaml
./panda run suction.local.yaml
```

Replace `/data` with your data disk or NAS mount. The label command verifies the original archive and every extracted file. In the browser, **Download suction labels** performs the same operation; shared images and models are still required. Inference needs only the selected RGB, depth, calibration and checkpoint files.

For a bounded training run, use `./panda init --example suctionnet-train-selected -o suction-train.local.yaml`. Set `dataset_root`, then check and run it. The preset fine-tunes on scene 0, frames 0–1. For the complete training split, use `--example suctionnet-train`: it trains from scratch over scenes 0–99 excluding scene 51, matching the author's loader. Released weights can instead initialize training by setting `checkpoint`.

| Control | Meaning |
|---|---|
| `scene`, `frame`, `frames` | Contiguous selection for inference, short training and bounded epoch training |
| `train_batch_limit: 0` | Full native training split; selection fields do not limit an epoch |
| `train_batch_limit: 1` | At most one batch per epoch from the selected frames |
| `batch_size` | At least 2 and no larger than the selected set; native incomplete batches are dropped |
| `label_root` | Optional writable cache directory; default: `DATASET_ROOT/.grasppanda/suction-v1/` |
| `epochs`, `learning_rate`, `data_workers` | Epoch count, initial Adam learning rate and data-loading workers |
| `train_checkpoint_mode: resume` | Restore a toolbox epoch checkpoint; increase total `epochs` and keep data/settings unchanged |

Training automatically invokes the pinned author target generators **only for requested views**, then verifies and reuses compressed label caches. Initial access needs object PLY models, per-view annotations, seal labels and scene collision labels. Place the cache and experiment outputs on a large data volume. Pre-existing author `score_maps/` caches use a different layout and are not consumed by this adapter.

Prediction files retain the author layout `predictions/SPLIT/scene_XXXX/realsense/suction/FFFF.npz`. Array `arr_0` contains **1,024 × 7** values: `[score, nx, ny, nz, x, y, z]`, in camera coordinates with positions in metres. The native preprocessing clips depth to one metre, including during back-projection. Predictions do not use object masks, CAD geometry or target labels. The preview shows the combined heatmap and leading candidate pixels.

Training reports seal-map and center-map MSE. Suction AP and robot success are separate measurements: use the [author evaluator](https://github.com/graspnet/suctionnetAPI) and its dense evaluation clouds. This adapter does not expose a benchmark evaluation button or interchangeable network components.

## Get data from the UI or CLI

In **Experiments**, choose a dataset, enter **Dataset root on server**, and expand **Get dataset inputs**. **Download starter data** prepares ZeroGrasp shard 0, DexGraspNet scene 0 views 0–1, or the official Jacquard sample, including native training supervision. Then download the method weights and check the preset. Starter predictions use training observations; use the published evaluation protocol for scientific comparisons.

The CLI previews the destination, exact sources and download size without writing files. Add `--fetch` to download:

```bash
./panda data zerograsp11b --root /data/ZeroGrasp-11B
./panda data zerograsp11b --root /data/ZeroGrasp-11B --fetch
./panda data dexgraspnet2 --root /data/DexGraspNet2.0 --fetch
```

ZeroGrasp's starter is 249 MB. DexGraspNet streams bounded prefixes of pinned archives and verifies every selected member before installation; it keeps approximately 6 MB. SHA-256 checks reject altered existing files. Interrupted file downloads resume; interrupted gzip extraction reuses completed verified members and re-reads the bounded prefix. A per-root lock prevents overlapping downloads.

For complete GraspClutter6D or DexGraspNet archives, use `--profile archives`. The plan reports the compressed size; extraction requires additional space. Downloading archives does not extract them automatically. Optional `--include 'scenes_*.tar.gz'` filters registered archive filenames. Full ZeroGrasp downloads use the author's script linked below. GraspNet's official mirrors are in [Data & weights](DOWNLOADS.md#graspnet-1b).

## GraspClutter6D

Keep the approximately 220 GB of compressed archives, extracted scenes and generated training targets on a disk with enough space. The scene archive spans five volumes; all five are needed for extraction. Install `p7zip-full` for the commands below.

```bash
./panda data graspclutter6d --root /data/GraspClutter6D --profile archives
./panda data graspclutter6d --root /data/GraspClutter6D --profile archives --fetch

7z x /data/GraspClutter6D/archives/scenes.7z.001 -o/data/GraspClutter6D
for archive in split_info grasp_label collision_label models_m dex_models; do
  7z x "/data/GraspClutter6D/archives/$archive.7z" -o/data/GraspClutter6D
done
```

The final root contains `scenes/000002/rgb/`, `depth/`, `label/`, `scene_camera.json`, `scene_gt.json`, and the separate `split_info/`, `grasp_label/`, `collision_label/`, `models_m/` and `dex_models/` directories. The [official dataset card](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) describes the other model representations and dataset terms.

Predict with the author's D435-trained checkpoint:

```bash
./panda weights contact_graspnet_gc6d --camera realsense-d435
./panda init --dataset graspclutter6d --dataset-root /data/GraspClutter6D -o gc6d.local.yaml
./panda check gc6d.local.yaml
./panda run gc6d.local.yaml
```

The default is test scene 2, ordinal view 0. Camera names are `realsense-d415`, `realsense-d435`, `azure-kinect` and `zivid`. An ordinal view is 0–12; the provider maps it to the camera-specific image ID and depth units. The D435 checkpoint on another camera measures cross-camera transfer. Baseline/Graspness presets use GraspNet-trained RealSense weights and explicitly represent cross-dataset transfer.

`official_gt_workspace` uses the author's visible-instance bounding box with a 10% image margin. `depth_only` uses all valid depth pixels. These are different input protocols and must be reported separately.

### Train and replace an encoder

Generate the native contact targets from real training observations:

```bash
./panda prepare-contacts --dataset-root /data/GraspClutter6D \
  --camera realsense-d435 --scenes 5 --frames 0
./panda init --example gc6d-pointmlp --dataset-root /data/GraspClutter6D -o gc6d-train.local.yaml
./panda check gc6d-train.local.yaml
./panda run gc6d-train.local.yaml
```

Omit `--scenes` and `--frames` to prepare the complete published training split. Preparation uses the pinned author implementation and requires CUDA. Targets are written atomically to `scene_contacts/train/`, with source/input hashes. Reusing toolbox-generated targets checks their camera, view, seed and preprocessing revision.

The native PointNet++ contact model supports the registered shared `pointnet`, `pointmlp` and `sonata_ptv3` encoders. Replacement preserves the native contact heads, coordinate centering and grasp decoder; it changes the seed representation and requires training. Choose `reuse_unchanged` to initialize unchanged heads from author weights, then `strict` for the resulting model. Epoch training uses the author AdamW, gradient clipping and cosine schedule; `train_checkpoint_mode: resume` restores optimizer, schedule and random states.

Official evaluation requires every prediction in the selected camera's 235 × 13 test views, the matching manifest/checkpoint, and the evaluator's model assets. Partial inference runs do not receive an AP score. The YCB pose-estimation split is a different protocol.

## ZeroGrasp-11B

The public training release has 10,000 compressed WebDataset shards, about 100 observations per shard. Start with one approximately 250 MB shard; the full release is on the order of 2.5 TB. Shards are read locally without extraction or an S3 account.

```bash
./panda data zerograsp11b --root /data/ZeroGrasp-11B --fetch

./panda weights zerograsp
./panda init --dataset zerograsp11b --dataset-root /data/ZeroGrasp-11B -o zero.local.yaml
./panda check zero.local.yaml
./panda run zero.local.yaml
```

The [author's download script](https://github.com/sh8/ZeroGrasp/blob/main/download.sh) provides `train_tiny` and full `train` downloads. The toolbox consumes its `train/shard-XXXXXX.tar.gz` layout directly. Each selected shard must finish downloading before use.

In the browser, **First shard**, **First sample** and **Sample count** select observations. The corresponding configuration keys remain `scene`, `frame` and `frames` for common experiment tooling; they represent shard index, within-shard ordinal and observation count here. Each result records the actual author sample key. The public shard release exposes the `train` split; the separate ReOcS datasets are reconstruction benchmarks and are not interchangeable with held-out grasp AP.

Inference uses the left RGB image, calibrated synthetic depth, visible instance-mask planes and their 2D bounding-box metadata. It never loads target surface points, object poses or grasp labels. Masks are an explicit supplied input, not a predicted segmentation result. Overlapping mask boundary pixels remain separate instance planes. Prediction arrays use metres and the GraspNet 17-column parallel-jaw representation.

The adapter retains native reconstruction, contact/grasp decoding and per-object NMS. `collision_thresh: 0.01` enables the author's geometric collision/refinement rules; `0` disables them. The author's all-colliding fallback is retained and recorded per object, so returned predictions must not automatically be called collision-free. Compatibility fixes supply correctly laid-out masks and scene/object ranges to the pinned CUDA feature extractor.

### Train, compose and resume

```bash
./panda init --example zero-mobilenet --dataset-root /data/ZeroGrasp-11B -o zero-train.local.yaml
./panda check zero-train.local.yaml
./panda run zero-train.local.yaml
```

The example is a bounded composition experiment. Increase `frames` and `epochs` to train over more downloaded observations. Every unbounded epoch consumes the selected range once; `train_batch_limit` caps batches when requested. Short training uses `training_steps`. Surface-point and grasp annotations come from the same shard, with their row alignment preserved.

| Image encoder | Shared implementation | ZeroGrasp boundary |
|---|---|---|
| `upstream` | Author ResNeXt-50 U-Net | Original checkpoint layout |
| `convnextv2` | The same registered timm encoder used by GraspNet RGB-D workflows | Three RGB channels; calibrated dense feature projection |
| `repvit` | Shared RepViT implementation | Same 32-channel dense feature contract |
| `mobilenetv4` | Shared MobileNetV4 implementation | Same 32-channel dense feature contract |

The native octree, CVAE, reconstruction and grasp heads remain in place. Use `reuse_unchanged` for author-head initialization and `strict` with a trained composed checkpoint. The example's image encoder is newly initialized; a short run does not establish grasp quality.

Training preserves native losses, AdamW, the step schedule, depth/mask augmentation and gradient clipping at 0.5. To resume epoch training, set the saved `checkpoint.pt`, `train_checkpoint_mode: resume`, `checkpoint_policy: strict`, and a larger total `epochs`; retain the original selected data and model settings. Model, optimizer, schedule and step are checked for exact restoration before the next update. Native CUDA reductions are numerically nondeterministic, so this is not a guarantee of bitwise-identical training trajectories.

The published author's demo uses ImageNet RGB normalization; its training reader uses CLIP normalization. Author checkpoint inference preserves the demo convention. Toolbox-trained checkpoints retain training normalization when reused for inference. **Runs & results → Prepare inference from checkpoint** carries over the component configuration.

## DexGraspNet 2.0

The integration retains dexterous grasping: each prediction includes a camera-frame rotation, translation in metres, 16 LEAP joint angles in radians, native scores and explicit joint names. The diffusion model and the dataset authors' ISAGrasp/GraspTTA baselines share the same observation and hand-output contracts. They run in the shared Python environment.

Download archives from the [official dataset repository](https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0). The complete published archive collection is approximately 222 GB before extraction. The project and dataset use [CC BY-NC 4.0](https://pku-epic.github.io/DexGraspNet2.0/#license).

```bash
./panda data dexgraspnet2 --root /data/DexGraspNet2.0 --profile archives
./panda data dexgraspnet2 --root /data/DexGraspNet2.0 --profile archives --fetch

for archive in /data/DexGraspNet2.0/archives/scenes_*.tar.gz; do
  tar -xzf "$archive" -C /data/DexGraspNet2.0
done
for archive in dex_graspness_new dex_grasps_new meshdata; do
  tar -xzf "/data/DexGraspNet2.0/archives/$archive.tar.gz" -C /data/DexGraspNet2.0
done
```

The root contains `scenes/scene_0000/realsense/`, `dex_graspness_new/`, `dex_grasps_new/` and `meshdata/`. Native observations use `depth_gt/`, `label_gt/`, `meta/`, `camera_poses.npy` and `cam0_wrt_table.npy`. RGB is not required. The hand meshes are included in the pinned author implementation; object meshes are needed for upstream simulation and additional native workflows.

```bash
./panda weights dexgraspnet2
./panda init --dataset dexgraspnet2 --dataset-root /data/DexGraspNet2.0 -o dex.local.yaml
./panda check dex.local.yaml
./panda run dex.local.yaml
```

The preset predicts 1,024 hand proposals from training scene 0, view 0, using 40,000 sampled points. Choose `dexgraspnet2_isa` or `dexgraspnet2_cvae` for the author baselines. Weight downloads use verified byte ranges from the pinned tar archive, so an individual method downloads only its approximately 177–192 MB checkpoint. The archive remains available as a manual fallback.

Predictions are `.npz` files containing `rotation`, `translation`, `qpos`, `joint_names`, `score`, `graspness`, `log_probability`, `camera_to_world` and `camera_to_table`. **Runs & results** displays the highest-scoring native hand proposal and observed points in an interactive 3D plot. `hand_scene.glb` exports that geometry for external viewers. Joint angles remain unmodified. The display does not imply collision-free execution or simulated success.

### Train and compose

Use **Short training run** for bounded updates, or **Train across epochs** to visit the selected view range. The finite toolbox epochs preserve native target construction, objectives, equal-object sampling, 128 grasp proposals, 64 matched contact seeds, the 6 mm matching threshold, in-plane rotation, Adam, gradient clipping at 10 and the 50,000-update cosine schedule. Final partial batches are retained. This selected-view epoch policy differs from the author's infinite random loader.

The registered training scene IDs are `0–99` and `1000–8499`. Held-out GraspNet-derived scene groups use `test_seen` (`100–129`), `test_similar` (`130–159`) and `test_novel` (`160–189`). `scene`, `frame` and `frames` choose the view range; frames are `0–255`. Graspness targets must match the rendered-depth workspace exactly. Missing views or unmatched supervision produce an input error; they are not silently replaced by another scene.

Generate the same composition with `./panda init --example dex-pointnet --dataset-root /data/DexGraspNet2.0`. The starter contains the two views used by that example. A minimal single-view configuration is:

```yaml
dataset: dexgraspnet2
method: dexgraspnet2
action: train
dataset_root: /data/DexGraspNet2.0
checkpoint: checkpoints/dexgraspnet2/checkpoint.pth
checkpoint_policy: reuse_unchanged
camera: realsense
split: train
scene: 0
frame: 0
frames: 1
num_points: 40000
batch_size: 1
epochs: 1
learning_rate: 0.001
workspace: official_gt_workspace
collision_thresh: 0
modules:
  backbone: pointnet
```

Backbones are `upstream`, `pointnet`, `sparse_unet18` and `sonata_ptv3`. They preserve the native sparse coordinate map and 512-channel feature contract. Native pose, joint and density heads remain in place. The GraspTTA baseline retains its differentiable hand-geometry loss and compiled primitive-distance operators.

For continued epoch training, select the saved `checkpoint.pt`, keep the same data/component settings, set `checkpoint_policy: strict`, `train_checkpoint_mode: resume` and a larger final `epochs`. Resume restores and checks model, optimizer and scheduler state, then restores random state. Previously consumed training files are content-checked. CUDA kernels can still introduce numerical nondeterminism. **Prepare inference from checkpoint** retains the selected observation for hand models; explicitly choose a held-out split when testing generalization.

### Native simulation evaluation

Dexterous-hand success is not GraspNet parallel-gripper AP. The toolbox does not substitute an AP value or a simplified physics score. The [author evaluation instructions](https://github.com/PKU-EPIC/DexGraspNet2#evaluation) cover Isaac Gym 4, ACRONYM test scenes, larger-clutter suites and world-frame output preparation. These remain separate from the verified shared-runtime prediction/training workflow. The legacy simulator requires its own supported runtime; it is not needed to train, compose or inspect the hand models here.

## Data locations

`/data/...` paths above are examples. Point each configuration at your own storage. Large data, weight files, native build caches and experiment outputs can live on a mounted storage volume; keep the source checkout separate. Use `--runs-dir /storage/experiments` for CLI outputs and `GRASPPANDA_RUNS` for the browser. Dataset-specific defaults are `GRASPPANDA_GRASPCLUTTER6D_ROOT` `GRASPPANDA_ZEROGRASP11B_ROOT` and `GRASPPANDA_DEXGRASPNET2_ROOT`; GraspNet retains `GRASPPANDA_DATASET_ROOT`.

## Jacquard

[Jacquard](https://jacquard.liris.cnrs.fr/database.php) contains 54,485 rendered scenes of 11,619 objects and approximately 4.97 million planar grasp annotations. The registered [GR-ConvNet implementation](https://github.com/skumra/robotic-grasping) supplies both RGB-D and depth-only Jacquard checkpoints. These predict image-space rectangles; they do not produce calibrated 6-DoF poses.

### Download and predict

The starter downloads the official 57.9 MB ZIP, verifies it, and installs RGB, perfect-depth and target files for 48 observations of 10 objects. The archive stays under the chosen root. In the browser, select **Jacquard → Get dataset inputs → Download starter data**, then **Download registered weights → Check current form → Run current form**.

```bash
./panda data jacquard --root /data/Jacquard --fetch
./panda weights grconvnet_rgbd --camera synthetic-rgbd
./panda init --dataset jacquard --dataset-root /data/Jacquard -o jacquard.local.yaml
./panda check jacquard.local.yaml
./panda run jacquard.local.yaml
```

Use `--method grconvnet_depth` and `./panda weights grconvnet_depth --camera synthetic-rgbd` for depth-only prediction. Both use the same shared environment. The source remains pinned and its weights are checksum-verified.

For the full dataset, follow the [author's access instructions](https://jacquard.liris.cnrs.fr/), download its archives to your data volume, and extract them there. Select a root containing the extracted object folders; nested archive directories are supported. Keep one copy of each observation under the root. Do not extract the starter and the full dataset into overlapping trees.

```text
Jacquard/
  object-id/
    0_object-id_RGB.png
    0_object-id_perfect_depth.tiff
    0_object-id_grasps.txt
  ...
```

### Splits, training and evaluation

The default split assigns 90% of object identities to training, with a deterministic seed. Validation objects are disjoint from training objects. Change these controls in **Input & preprocessing → Dataset parameters**, or in the configuration:

```yaml
dataset_options:
  train_fraction: 0.9
  split_policy: object
  split_seed: 0
  iou_threshold: 0.25
```

`split_policy: ordered_image` instead splits the sorted image list, matching the author's unshuffled image-split convention; it may share objects across splits. `scene` is the zero-based sample index within the chosen split, `frame` is always `0`, and `frames` is the sample count. `native_image` preserves native image preprocessing and requires `collision_thresh: 0`.

Use `./panda init --example jacquard-train --dataset-root /data/Jacquard -o train.local.yaml`. **Train across epochs** with `train_batch_limit: 0` visits the full training split once per epoch; bounded training uses the selected sample range. **Short training run** performs the requested optimizer steps. Set `checkpoint: ''` to train from scratch. The native 300-pixel network, rotation/zoom augmentation, Adam optimizer and four Smooth L1 terms are retained. This finite loader replaces the author's repeated fixed-batch epoch convention. Validation reports native loss after each epoch; `eval_batch_limit: 0` covers the full validation split.

Epoch checkpoints support `initialize` and `resume`. For resume, retain the data, split and training configuration, set `checkpoint` to the saved `checkpoint.pt`, set `train_checkpoint_mode: resume`, and increase `epochs`. Checkpoints retain optimizer and random states, configuration and file metadata for both splits. Changed configuration, input sizes or modification times are rejected. Use **Prepare inference from checkpoint** to inspect a trained model.

For rectangle evaluation, predict **every** observation of `split: val` in one inference run (`scene: 0`, `frames` equal to its sample count, reported by **Check current form**). Then select **Evaluate predictions**, keep the same checkpoint and dataset parameters, and point **Prediction directory** to that run's `predictions/`. The verifier requires complete split coverage, matching input hashes, checkpoint and maps. `metrics.planar_iou_success` is the fraction of samples with a matching top-1 rectangle under the chosen IoU threshold and the native 30-degree angle criterion. Width maps are in pixels of the 300-pixel model input. This metric is neither physical grasp success nor GraspNet AP. Author pretrained models may have seen the selected observations; use a declared held-out training protocol for research claims.

The original TIFF reader is adapted to Pillow to preserve floating-point pixels without the legacy optional TIFF codec. Encoder replacement is not registered for GR-ConvNet; its architecture and published checkpoints load strictly.

## GraspGen

[NVIDIA's GraspGen release](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen) contains over 57 million simulated grasps across 8,515 objects and three grippers. The current adapter supports **Franka Panda** with the official **PTv3 diffusion generator and discriminator**. Inputs are single partial depth point clouds rendered from the released object geometry. Network inputs contain no grasp labels. This is an object-centric protocol, separate from GraspNet scene detection.

### Download and predict

The approximately 39.3 MB starter installs four original objects: two from the author training split and two from validation. It retrieves only the required byte ranges from a pinned grasp archive, downloads the original Objaverse GLBs, and verifies every file. Rendered observations are created locally; no full mesh is passed to the predictor.

```bash
./panda data graspgen --root /data/GraspGen --fetch
./panda weights graspgen --camera synthetic-depth
./panda init --dataset graspgen --dataset-root /data/GraspGen -o graspgen.local.yaml
./panda check graspgen.local.yaml
./panda run graspgen.local.yaml
```

In the browser, select **GraspGen → Download starter data → Download registered weights → Check current form → Run current form**. The paired author weights and architecture configuration total approximately 1.07 GB. Configure [NAS storage](INSTALL.md#choose-the-storage-volume-first) before installation and downloading; datasets use the explicit root above. Both networks run in the existing shared environment.

`scene` selects an object index within `train` or `valid`; `frames` is the object count and `frame` remains zero. The registered protocol uses 2,048 points from a noisy 256-pixel depth rendering, a 60-degree field of view, and an isolated object. The author renderer is retained with `prob_object_only=1` and seeded depth noise. Rendering requires a working NVIDIA EGL driver on the experiment host. Keep `workspace: object_partial`, `collision_thresh: 0` and `data_workers: 0`.

Each prediction NPZ contains `points`, `poses` (N×4×4), `scores` and `object_to_observation`. Poses and points share the centered observation frame, with translations in metres and the author's Franka Panda gripper-base convention. The saved transform maps the scaled object geometry into that frame. Apply its inverse to poses when returning to the object frame. The preview shows the observation and native gripper outlines. The interactive result view and `grasp_scene.glb` export contain the top three predicted grippers. Visualization uses the author's fixed gripper opening; it is not a predicted width. These are native poses, with no invented opening width or GraspNet conversion.

### Full dataset layout

Use the [pinned release](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen/tree/de99bde9c3cd9c12ff5dc448f8ed8ab09c43d2e5) for the eight `grasp_data/franka_panda/shard_*.tar` files and `uuid_index.json`. Keep the TARs intact. Copy the author `splits/franka_panda/train.txt` and `valid.txt` into the root. Download the corresponding Objaverse meshes following the [author mesh preparation instructions](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen/blob/de99bde9c3cd9c12ff5dc448f8ed8ab09c43d2e5/scripts/download_objaverse.py), retaining UUID-to-file mapping. Mesh paths must resolve inside `meshes/`; relative paths are portable. Use a separate root for the full release and starter to preserve split identity.

```text
GraspGen/
  train.txt
  valid.txt
  grasp_data/franka_panda/
    uuid_index.json
    shard_000.tar ... shard_007.tar
  meshes/
    map_uuid_to_path.json
    <uuid>.glb
```

The starter uses `labels/<uuid>.json` plus `labels/map_uuid_to_path.json` in place of TARs. TAR readers cache file offsets under `.grasppanda/graspgen-index-v1/`, avoiding a full shard scan per object. Set `label_root` to a writable cache volume when data is read-only. Both formats retain original labels; the toolbox does not follow the historical absolute mesh paths embedded in the annotation JSON. Preflight requires every object named in the selected split to be installed, and rejects train/validation overlap. Full-release data loading is supported; operational validation uses the four-object starter, not full-dataset convergence.

For **Train across epochs**, choose **Method training stages → GraspGen training stage**. See [training controls and checkpoint reuse](REFERENCE.md#graspgen-training). Inference rendering never selects views using grasp success labels; training retains native visible-grasp supervision and retries invalid views, recording the retries instead of silently omitting objects. Simulator success, on-policy label generation, other grippers and arbitrary backbone replacement remain upstream workflows.

## TARGO

[TARGO (IJCV 2026)](https://arxiv.org/pdf/2407.06168) studies target-driven grasping under occlusion. The pinned [training CSV](https://huggingface.co/datasets/randing2000/TARGO/blob/45ae479029f1a533b2f60bd7d2fbca6bb58d087c/syn_train/grasps.csv) contains 2,474,952 grasp trials. GraspPanda loads the official TARGO-Net and frozen AdaPoinTr checkpoints in the shared runtime. Inputs are **single-view depth-derived points and supplied target segmentation**; the adapter does not predict the target mask.

### Download and predict

```bash
./panda data targo --root /data/TARGO --fetch
./panda weights targonet --camera synthetic-depth
./panda init --example targo-target --dataset-root /data/TARGO -o targo.local.yaml
./panda check targo.local.yaml
./panda run targo.local.yaml
```

The **5.2 MB starter** contains four original training scenes, their 58 original grasp trials, and the author's released test scene. Label downloads retrieve exact CSV byte ranges and assemble them locally without changing values. The two registered checkpoints total **393 MB**. Choose **TARGO → Download starter data → Download registered weights** in the browser, then check and run the form. Select a [NAS storage root](INSTALL.md#choose-the-storage-volume-first) before installing and downloading.

Predictions consume the released `pc_depth_targ` and `pc_scene_no_targ` visual arrays in each scene NPZ. These were generated from depth, camera calibration and supplied masks. The target is normalized and filtered using the author workspace, then completed by frozen AdaPoinTr. Native TARGO-Net, completed-target TSDF filtering and nonmaximum suppression produce poses and opening widths. The scene includes the author's sampled support plane. No grasp labels, CAD geometry or object poses are passed to the predictor. A failed completion stops the job.

Each prediction NPZ stores `poses` (N×4×4), `widths`, `scores`, `scene_points`, `target_points` and `completed_target`. Geometry uses the author's **0.3 m workspace frame**, with translations and widths in metres and the VGN gripper-pose convention. It is not the camera frame or a GraspNet grasp array. The preview and rotatable GLB show native grippers and observed points; the PNG also shows completed target points. `force_detection: true` preserves the author fallback to the best lower-confidence candidate; the manifest records when it was used. Disabling it can legitimately produce an empty prediction.

### Full release and training

Download [the pinned dataset](https://huggingface.co/datasets/randing2000/TARGO/tree/45ae479029f1a533b2f60bd7d2fbca6bb58d087c) to a separate dataset root. Keep the numbered scene folders intact; both flat and numbered layouts are supported. For example, with the Hugging Face CLI:

```bash
hf download randing2000/TARGO --repo-type dataset \
  --revision 45ae479029f1a533b2f60bd7d2fbca6bb58d087c \
  --include "syn_train/**" "test_set_gaussian_0.005/**" --local-dir /data/TARGO-full
```

Use the author's [setup.json](https://raw.githubusercontent.com/TARGO-benchmark/TARGO/e71d00e6c081aa39a164a6a402375aa75173ff80/setup/setup.json) in `syn_train/` if the download does not include it. The full dataset is much larger than the starter; download it to your data volume.

```text
TARGO/
  syn_train/
    setup.json
    grasps.csv
    scenes/000/<scene_id>.npz
  test_set_gaussian_0.005/
    setup.json
    scenes/<scene_id>.npz
```

Training uses the author's released target TSDF points and surrounding scene points, target sampling, quaternion symmetry and grasp objective. The grasp trainer does not optimize AdaPoinTr. The toolbox removes the upstream script's default tiny random subsample, uses finite epochs, and groups **all targets and single/double/clutter variants of a base scene together** when splitting `syn_train`. The default is 90% training groups and 10% validation groups with seed 0; the starter has three training scenes and one validation scene. This is a toolbox validation protocol, not the author's random row split or simulator benchmark. The three invalid scene IDs excluded by the native loader remain excluded.

`scene` and `frames` select scene indices and counts in the chosen split. Unbounded epoch training consumes all labels in its training split, including the final partial batch; short training and bounded epochs use the selected scenes. `data_workers` remains zero. Dataset inventories are reused in the UI and invalidated when source metadata changes. See [training, resume and decoding controls](REFERENCE.md#targo-training).

Epoch validation reports held-out grasp losses and binary label accuracy. **These are not physical grasp success rates.** PyBullet evaluation requires separate object URDFs/meshes and the upstream simulation workflow; it is not exposed as a toolbox evaluator. The author-linked GIGA object asset archive was unavailable when this adapter was validated. No training-accuracy proxy is substituted for simulation results.

## ACRONYM

[ACRONYM](https://github.com/NVlabs/acronym) contains **17.7 million simulated grasps on 8,872 objects**. The integrated [GraspLDM model (IEEE Access 2024)](https://arxiv.org/pdf/2312.11243) uses its released **partial-point-cloud** weights, with a PVCNN encoder, conditional VAE and latent diffusion model. Full-surface point sampling is not used as a visual observation.

### Start with original assets

```bash
./panda data acronym --root /data/ACRONYM --fetch
./panda weights graspldm --camera synthetic-depth
./panda init --example acronym-diffusion --dataset-root /data/ACRONYM -o acronym.local.yaml
./panda check acronym.local.yaml
./panda run acronym.local.yaml
```

The **0.6 MB starter** downloads NVIDIA's original Mug and Table examples, their meshes and **4,000 original grasp trials**. Its explicit toolbox demonstration split assigns Mug to training and Table to testing; this is not the paper's split. The paired model downloads total **46 MB**. In the browser, select **ACRONYM**, download the starter and registered weights, then check and run the preset. New data, caches and experiments follow your selected [storage volume](INSTALL.md#choose-the-storage-volume-first).

Each input is **1,024 points from a single rendered depth image** of an isolated object, using the author's 640×480 camera calibration and 0.1 mm depth quantization. The toolbox samples seeded camera views on a 0.5–0.8 m shell, independent of grasp labels. Mesh geometry and scale generate the synthetic observation; only the resulting partial point cloud enters the model. The author's point-cloud centering and normalization are retained. This object-centric protocol does not include scene segmentation or clutter collision rejection.

`scene` indexes an object in the selected split, `frame` selects one of its 20 seeded views, and `frames` counts views, continuing to the next object when necessary. Derived numeric observations are verified and reused under `DATASET_ROOT/.grasppanda/acronym-depth-v1/`; set `label_root` to a writable cache directory when the dataset is read-only. A changed mesh, camera calibration or view seed selects a new cache entry.

Outputs store `poses` (N×4×4), `scores`, `points` and `object_to_camera` in NPZ files. Translations use metres in the **OpenCV camera frame**; the grasp frame is the native ACRONYM Franka gripper base. Width is not predicted. The PNG and interactive GLB use the author's gripper marker. Scores are native sigmoid outputs, not measured simulation success probabilities.

### Full dataset

Follow [NVIDIA's full ACRONYM instructions](https://github.com/NVlabs/acronym#using-the-full-acronym-dataset) for the approximately 1.6 GB annotation archive, **ShapeNetSem access**, mesh terms and watertight preprocessing. Original HDF5 metadata determines each mesh's path and scale. The starter examples do not grant access to the complete ShapeNet collection.

For the author's category split files:

```bash
hf download kuldeepbarad/GraspLDM \
  --revision 3da18c20aac385fcb1ae4843a83f2ffa43001a99 \
  --include "splits/*.json" --local-dir /data/ACRONYM-full
```

```text
ACRONYM-full/
  grasps/<category>_<mesh-id>_<scale>.h5
  meshes/<category>/<mesh-id>.obj
  splits/<category>.json
```

Every object listed in the selected split must be installed. Alternatively, create `splits.json` with explicit `train` and `test` lists of original HDF5 basenames; it takes precedence over `splits/`. Mesh identities may not overlap between those lists, including differently scaled copies. Use a separate full-dataset root so the starter's demonstration `splits.json` does not override your full split.

### Training protocol

Both native objectives are available through **Train across epochs → Method training stages → GraspLDM training stage**. The toolbox renders observations lazily and samples original **successful** grasps as supervision. It does **not** reproduce the paper's visibility-filtered targets: the public author renderer imports `GripperCollision`, which is absent from the pinned source. Its rendered training release is also not supplied by the registered weight repository. The selectable toolbox protocol is explicitly named **`all_success`**; it may supervise grasps on surfaces hidden from the selected view.

An unbounded epoch visits all 20 views of every training object once, sampling `grasps_per_view` labels per view. It is not an exhaustive pass over every grasp trial. Objects are not silently skipped; missing meshes, unusable depth or objects without successful labels stop the job. See [stage selection, initialization, resume and checkpoint reuse](REFERENCE.md#acronym-and-graspldm). The starter validates operation only; no full-dataset convergence or simulator benchmark is claimed.

## VGN simulation

[VGN (CoRL 2020)](https://github.com/ethz-asl/vgn) adds a **self-supervised simulation workflow**: generate labelled scenes, train on depth-derived TSDF volumes, then test closed-loop clutter removal. This provider generates data locally; it is not an additional downloaded large-scale benchmark. Increase `scene_count` and `grasps_per_scene` to collect a larger training set on your selected storage volume.

```bash
./panda data vgn --root /data/VGN --fetch
./panda weights vgn --camera synthetic-depth
./panda init --example vgn-generate --dataset-root /data/VGN -o generate.local.yaml
./panda run generate.local.yaml
./panda init --example vgn-train --dataset-root /data/VGN -o train.local.yaml
./panda run train.local.yaml
```

The **6.5 MB author asset archive** contains the original object and gripper models; it contains no labelled sensor observations. In the browser, select **VGN simulation**, download these assets, then choose **Generate simulation data**. Generation runs in the experiment queue and can be cancelled. Repeating the same generation configuration with the recorded PyBullet, Open3D and NumPy versions verifies and reuses completed scenes; incomplete scenes are regenerated. Use a new dataset root to change the generation plan. The default collects three scenes with 120 candidate locations each, evaluating six wrist rotations per candidate using the native simulator.

Generation uses the author's `pile/train` or `packed/train` objects, object-count distribution, view sampling, contact rules and success labels. The toolbox seeds each scene, sorts object discovery, records hashes, and assigns complete scenes to training or validation. The default validation fraction is 10%, with at least one scene in each split. Depth views are chosen independently of grasp success labels. The native 0.02–0.28 m workspace cleaning is applied before converting positions and widths into voxel coordinates. Training and validation balance successful/unsuccessful labels independently within their own scenes; all selected balanced labels are visited once per unbounded epoch, including the final partial batch.

```text
VGN/
  assets/vgn-data.zip
  generation.json
  splits.json
  raw/setup.json
  raw/grasps.csv
  raw/scenes/<scene>.npz
  raw/labels/<scene>.csv
  processed/grasps.csv
  processed/scenes/<scene>.npz
  processed/records/<scene>.json
```

The network receives a **40×40×40 TSDF** integrated from one to six rendered depth images, following the native VGN convention. Meshes are used by the simulator, not supplied to the network. Predictions store native grasp transforms, widths and confidence in the **VGN workspace frame, in metres**; the gripper convention remains the author's TCP convention. Zero predictions above the configured threshold is a valid result, displayed with the observed scene.

**Simulate clutter removal** uses the corresponding author `pile/test` or `packed/test` object set in newly generated scenes. It follows the native stopping rules and candidate order, and records per-round object counts and executed grasp outcomes. Report both the number of rounds/attempts and the simulated success and object-clearing fractions. These are physics-engine results, not real-robot success rates or a reproduction of the paper's full evaluation. A small generated training set is intended for setup and iteration, not accuracy claims. See [controls, resume and simulation](REFERENCE.md#vgn-simulation).
