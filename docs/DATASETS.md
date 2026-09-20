# Datasets and observation protocols

Select a dataset first. The browser then offers its methods, cameras, splits and compatible operations. Dataset paths are remembered independently during the session. A shared runtime does not make different observation or supervision protocols interchangeable.

| Dataset | Observations | Methods in this integration | Protocol |
|---|---|---|---|
| [GraspNet-1B](https://graspnet.net/) | Real RGB-D; single-view, fused-view and temporal adapters | See [Methods & papers](METHODS.md) | Native GraspNet splits and evaluation |
| [GraspClutter6D](https://sites.google.com/view/graspclutter6d) | 1,000 real scenes, 52,000 RGB-D images, four cameras, 9.3 billion grasps | Native Contact-GraspNet; GraspNet Baseline and Graspness transfer | Published 413-scene training / 235-scene grasp test split |
| [ZeroGrasp-11B](https://github.com/sh8/ZeroGrasp) | One million synthetic observations, 12,000 objects, 11.3 billion grasps | Native ZeroGrasp with interchangeable image encoders | Released training shards; no held-out grasp AP is implied |
| [DexGraspNet 2.0](https://pku-epic.github.io/DexGraspNet2.0/) | 1,319 objects, 8,270 synthetic scenes, approximately 427 million dexterous grasps | Native diffusion; author ISAGrasp and GraspTTA baselines | Rendered depth and 16-joint LEAP-hand prediction; simulation evaluation remains an upstream workflow |

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

## Get data from the UI or CLI

In **Experiments**, choose a dataset, enter **Dataset root on server**, and expand **Get dataset inputs**. **Download starter data** prepares ZeroGrasp shard 0 or DexGraspNet scene 0 views 0–1, including native training supervision. Then download the method weights and check the preset. Starter predictions use training observations; use the published evaluation protocol for scientific comparisons.

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
