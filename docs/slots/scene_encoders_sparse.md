# Scene encoders: sparse-voxel networks and released weights

This page is about one config line, `model.backbone`, for the encoders that run on a
voxel grid (sparse convolutions, and the point transformers that serialise voxels), and
about starting them from **published checkpoints**. Point-based encoders (PointNet++,
PointNeXt) and image encoders have their own pages.

Nothing here is a grasp-detection result. Every option below was built, loaded (where it
has weights) and probed for twenty training iterations on real GraspNet-1Billion frames; no
option has been trained to convergence in GraspPanda and **no AP is claimed for any of
them**.

## 1. The task of the slot

The scene encoder is the first learned stage of a seed-based grasp detector (GSNet,
EconomicGrasp, FineGrasp, GraspNet-baseline, Scale-Balanced-Grasp). It receives the
partial point cloud of one depth view, a cluttered table top seen by one camera, and
returns one feature vector per point. Everything after it reads those features: the
proposal stage scores every point ("is there a grasp here?") and picks about a thousand
seeds, the view stage chooses an approach direction per seed, the grouper crops a
neighbourhood per seed and the head regresses the grasp. The encoder is therefore the
only stage that sees the whole scene, and the only one that can know that a point lies
on the rim of a bowl next to a box.

```text
depth view ─► points ─► [ backbone ] ─► proposal ─► view ─► grouper ─► head ─► grasps
                          this page      seeds      approach   crops     6-DoF
```

## 2. Input and output

Read from `grasppanda/models/base.py` (`Backbone`) and
`grasppanda/models/detectors/seed_based.py`.

| | |
|---|---|
| Call | `backbone(points, feats) -> PointFeatures` |
| `points` | `PointSet`: `(N, 3)` float coordinates **in metres, camera frame**, of all clouds of the batch packed one after another, with batch offsets. `N` is 15 000 per frame under GSNet and 20 000 under the other methods. |
| `feats` | `(N, in_channels)` float. With `point_features: ones` (GSNet, EconomicGrasp, GraspNet-baseline, Scale-Balanced-Grasp) it is a **constant 1 in every channel**: the network sees geometry only, through the voxel grid. With `point_features: observation` (FineGrasp) it is the 6 real channels of the dataset, and `compose` sets `in_channels: 6`. |
| Returns | `PointFeatures`: `(N, out_channels)` bound to the point set they live on. |
| `output` | `"dense"`: one feature for **every input point** (all encoders of this page), or `"subsampled"`: features on a child set of about 1024 points (PointNet++). |
| Attributes | `in_channels`, `out_channels`: the detector builds proposal, view stage and grouper with `in_channels = backbone.out_channels`. |

What the neighbours require:

- A proposal states `requires = "dense"`, `"subsampled"` or `"any"`. GraspNet-baseline and
  Scale-Balanced-Grasp take seeds from a sub-sampling encoder; `compose` then wraps a
  dense encoder in the adapter `FurthestPointSubsample` (and tells you).
- The width is free, with one exception met in the probes: the grouper of
  Scale-Balanced-Grasp (`DepthCylinderGroup` with a gated seed residual) adds seed
  features to 256-channel crop features, so the encoder must return 256 channels there.
  Encoders with a free `out_channels` (`SparseUNet14`, `SpUNet`, `MinkUNet`) are set to the
  width of the encoder they replace; PTv3 and LitePT have a fixed width and fail to build
  under that one method, with a message that says so.

What a sparse-voxel encoder does with this input, in all classes of this page: it
quantises the coordinates to a grid it owns (`voxel_size` or `grid_size`, **0.005 m in
every preset**, the grid of GSNet), runs on one feature per occupied voxel and hands the
feature of a voxel back to every point in it. The grid is the encoder's unit of length:
a network whose weights were trained on rooms at 0.02 m sees, at 0.005 m, a quarter of
the extent per kernel. Sparse convolutions go through the facade `grasppanda.ops.sparse`
(spconv underneath; extra `spconv-cu118` and siblings).

## 3. Options

"Route" says where the code comes from: **A** = adapted from a permissive repository
(listed in `THIRD_PARTY_NOTICES.md`), **W** = released weights loaded into a class the
toolbox has, **own** = written for this code base earlier (see
[Scene encoders](../model_zoo_backbones.md)). "Measured fit" is the verdict of
`grasppanda probe` on real frames, in the vocabulary of
[Measured combinations](../advanced_guides/measured_combinations.md): *runs*, *runs
(adapter)* when `compose` inserted `FurthestPointSubsample`, *fails* with the reason.
All options are **universal** in intent: the slot contract is the same for the five
methods; the one method-specific limit is the 256-channel rule above.

### 3.1 Encoders

| Preset (`--set backbone=…`) | Class | Paper, venue | Source, licence | Route | Weights | Extra | Width |
|---|---|---|---|---|---|---|---|
| `sparse-unet14` | `SparseUNet14` | Choy et al., CVPR 2019; the encoder of GSNet | written here | own | random init | spconv | free (512) |
| `minkunet18a` | `MinkUNet` | Choy et al., CVPR 2019 ([arXiv 1904.08755](https://arxiv.org/abs/1904.08755)) | [NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) @ `02fc608`, MIT | A | random init | spconv | free |
| `minkunet34c` | `MinkUNet` | same | same | A | random init | spconv | free |
| `minkunet34c-pointcontrast` | `MinkUNet` | PointContrast, Xie et al., ECCV 2020 ([arXiv 2007.10985](https://arxiv.org/abs/2007.10985)) | weights of [facebookresearch/PointContrast](https://github.com/facebookresearch/PointContrast) (code MIT) | W | `pointcontrast-nce`, licence unstated | spconv | free |
| `minkunet18a-openscene` | `MinkUNet` | OpenScene, Peng et al., CVPR 2023 ([arXiv 2211.15654](https://arxiv.org/abs/2211.15654)) | weights of [pengsongyou/openscene](https://github.com/pengsongyou/openscene) @ `0f369bc` (code Apache-2.0) | W | `openscene-scannet-openseg`, licence unstated | spconv | free |
| `spunet` | `SpUNet` | Pointcept `SpUNet-v1m1`, in the 34C sizes of Choy et al. | written here | own | random init | spconv | free |
| `spunet-gaussiancross` | `SpUNet` | GaussianCross, Yao et al., ACM MM 2025 ([arXiv 2508.02172](https://arxiv.org/abs/2508.02172)) | weights on [RayYoh/GaussianCross](https://huggingface.co/RayYoh/GaussianCross) | W | `gaussiancross-scannet-lin`, MIT | spconv | free |
| `ptv3` | `PTv3` | Point Transformer V3, Wu et al., CVPR 2024 ([arXiv 2312.10035](https://arxiv.org/abs/2312.10035)) | written here | own | random init | spconv | 64 |
| `ptv3-scannet200` | `PTv3` | same | weights on [Pointcept/PointTransformerV3](https://huggingface.co/Pointcept/PointTransformerV3) | W | `ptv3-scannet200`, MIT | spconv | 64 |
| `ptv3-ppt` | `PTv3` | Point Prompt Training, Wu et al., CVPR 2024 ([arXiv 2308.09718](https://arxiv.org/abs/2308.09718)) | same hub repository | W | `ptv3-ppt-s3dis`, MIT | spconv | 64 |
| `ptv3-encoder` | `PTv3` with `enc_mode` | Point Transformer V3 | option of the class | own | random init (or the encoder half of any PTv3 file) | spconv | 992 |
| `ptv3-sonata` | `PTv3Sonata` | Sonata, Wu et al., CVPR 2025 | written here | own | random init | spconv | 1232 |
| `ptv3-sonata-small` | `PTv3Sonata` | Sonata ([arXiv 2503.16429](https://arxiv.org/abs/2503.16429)) | weights on [facebook/sonata](https://huggingface.co/facebook/sonata) | W | `sonata-small`, **CC BY-NC 4.0 (NC)** | spconv | 992 |
| `litept` | `LitePT` | LitePT, Yue et al., CVPR 2026 ([arXiv 2512.13689](https://arxiv.org/abs/2512.13689)) | written here | own | random init | spconv | 72 |
| `litept-structured3d` | `LitePT` | same | weights on [prs-eth/LitePT](https://huggingface.co/prs-eth/LitePT) | W | `litept-small-structured3d`, MIT | spconv | 72 |
| `oacnns` | `OACNNs` | OA-CNNs, Peng et al., CVPR 2024 | written here | own | random init (no checkpoint is released) | spconv | free |

Venues are those printed on the arXiv records linked (opened on 2026-10-13).

`MinkUNet` is one class for the whole family of MinkowskiEngine; `layers`, `planes` and
`block` select the member:

| Member | `layers` | decoder half of `planes` (the encoder half is 32, 64, 128, 256) | `block` |
|---|---|---|---|
| 14A / 14B / 14C / 14D | `[1, 1, 1, 1, 1, 1, 1, 1]` | 128-128-96-96 / 128-128-128-128 / 192-192-128-128 / 384-384-384-384 | `basic` |
| 18A / 18B / 18D | `[2, 2, 2, 2, 2, 2, 2, 2]` | as 14A / 14B / 14D | `basic` |
| 34A / 34B / 34C | `[2, 3, 4, 6, 2, 2, 2, 2]` | 256-128-64-64 / 256-128-64-32 / 256-128-96-96 | `basic` |
| 50 / 101 | 34's / `[2, 3, 4, 23, 2, 2, 2, 2]` | 256-128-96-96 | `bottleneck` |

Only 18A and 34C were probed; the others are the same code with other numbers. Bottleneck
blocks are covered by a unit test, not by a probe.

### 3.2 Released weights

Every row was **downloaded and loaded strictly** on 2026-10-13: each tensor of the class
(built with the options of the row) comes from the file with the same shape, and the
encoder part of the file holds no tensor the class lacks
(`tests/integration/test_backbone_released_weights.py`, 15 files). This proves names and
shapes. Activations were not compared with the original code for these files; the
classes `PTv3`, `PTv3Sonata` and `LitePT` carry such comparisons for other files of the
same releases ([Scene encoders](../model_zoo_backbones.md)).

`pretrained:` takes the name in the first column, or a path.

| Name | Class | From | Trained on | Licence of the weights | Size | Class options for a strict load |
|---|---|---|---|---|---|---|
| `gaussiancross-scannet-lin` | `SpUNet` | GaussianCross (ACM MM 2025), [hub](https://huggingface.co/RayYoh/GaussianCross) @ `3cbceee` | self-supervised pre-training on ScanNet; the linear-probing checkpoint of the release (`semseg-spunet-base-lin`), whose class layer lies outside the encoder | MIT | 157 MB | `out_channels: null` |
| `gaussiancross-scannet` | `SpUNet` | same | the same pre-training, fine-tuned for ScanNet semantic segmentation | MIT | 470 MB | defaults |
| `volt-teacher-scannet` | `SpUNet` | Volt (ECCV 2026, [arXiv 2604.19609](https://arxiv.org/abs/2604.19609)), [hub](https://huggingface.co/KadirYilmaz/Volt) @ `9dcb91d`, `teacher_weights/` | ScanNet semantic segmentation; a teacher network of Volt (`scannet_unet_teacher.pth`) | MIT | 470 MB | defaults |
| `pointcontrast-nce` | `MinkUNet` | PointContrast (ECCV 2020), [README](https://github.com/facebookresearch/PointContrast) link to Dropbox | self-supervised, PointInfoNCE, pairs of partial ScanNet RGB-D frames; Res16UNet34C, 3³ stem, 0.025 m, batch-norm momentum 0.05 (its `defaults.yaml`) | unstated (code MIT; ScanNet terms apply) | 303 MB | `out_channels: 32` |
| `mask3d-scannet` | `MinkUNet` | Mask3D (ICRA 2023, [arXiv 2210.03105](https://arxiv.org/abs/2210.03105)), [README](https://github.com/JonasSchult/Mask3D) link `scannet_pretrained.ckpt` | the Res16UNet34C backbone inside the instance-segmentation model, ScanNet | unstated (code MIT; ScanNet terms apply) | 159 MB | `out_channels: 20`, `stem_kernel: 5` |
| `openscene-scannet-openseg` | `MinkUNet` | OpenScene (CVPR 2023), `scannet_openseg.pth.tar` | MinkUNet18A distilled to predict per-point OpenSeg (CLIP-aligned) features on ScanNet at 0.02 m; **input: three constant ones per point** (`feature_loader.py`: "directly use color=(1, 1, 1) for all points") | unstated (code Apache-2.0; ScanNet terms apply) | 187 MB | `layers: [2,2,2,2,2,2,2,2]`, `planes: [32,64,128,256,128,128,96,96]`, `stem_kernel: 5`, `out_channels: 768`, `final_bias: false` |
| `ptv3-scannet200` | `PTv3` | Point Transformer V3 (CVPR 2024), [hub](https://huggingface.co/Pointcept/PointTransformerV3) @ `5e25cfa` | supervised ScanNet200 segmentation | MIT | 554 MB | defaults |
| `ptv3-s3dis-rpe` | `PTv3` | same | supervised S3DIS segmentation, relative position bias | MIT | 554 MB | `enable_rpe: true`, `enc_patch_size: [128,128,128,128,128]`, `dec_patch_size: [128,128,128,128]` |
| `ptv3-nuscenes` | `PTv3` | same | supervised nuScenes lidar segmentation (outdoor) | MIT | 554 MB | `in_channels: 4` |
| `ptv3-ppt-s3dis` | `PTv3` | Point Prompt Training (CVPR 2024), same hub repository, `s3dis-semseg-pt-v3m1-1-ppt-extreme` | supervised jointly on ScanNet, S3DIS and Structured3D, one set of normalisation layers per dataset | MIT | 446 MB | `enable_rpe: true`, `enc_channels: [32,64,128,256,384]`, `enc_num_head: [2,4,8,16,24]`, `enc_patch_size: [128]×5`, `dec_channels: [64,64,128,256]`, `dec_num_head: [4,4,8,16]`, `dec_patch_size: [128]×4`, `pretrained_condition: 0` (0 ScanNet, 1 S3DIS, 2 Structured3D) |
| `sonata-small` | `PTv3Sonata` | Sonata (CVPR 2025), [hub](https://huggingface.co/facebook/sonata) @ `df99897` | self-supervised; small encoder | **CC BY-NC 4.0 (NC)**, per the repository README; the card states none | 154 MB | `enc_depths: [2,2,2,6,2]`, `enc_channels: [32,64,128,256,512]`, `enc_num_head: [2,4,8,16,32]` |
| `sonata` | `PTv3Sonata` | same | self-supervised; base encoder | **CC BY-NC 4.0 (NC)** | 434 MB | defaults |
| `concerto-base-origin` | `PTv3Sonata` | Concerto (NeurIPS 2025, [arXiv 2510.23607](https://arxiv.org/abs/2510.23607)), [hub](https://huggingface.co/Pointcept/Concerto) @ `efc1307` | joint 2-D/3-D self-supervised; **the model of the paper** (the default `concerto_*.pth` files are pre-release Utonia models, says the card) | **CC BY-NC 4.0 (NC)** | 433 MB | defaults |
| `litept-small-structured3d` | `LitePT` | LitePT (CVPR 2026), [hub](https://huggingface.co/prs-eth/LitePT) @ `a8e76e9` | supervised Structured3D segmentation (synthetic rooms) | MIT | 152 MB | defaults |
| `litept-small-nuscenes` | `LitePT` | same | supervised nuScenes lidar segmentation (outdoor) | MIT | 152 MB | `in_channels: 4` |

Other files of the same releases have the same layout and are expected to load in the
same way, but were **not downloaded here**: the other GaussianCross models (ScanNet200,
S3DIS, data-efficient), the other Volt teachers (ScanNet200, ScanNet++, nuScenes,
SemanticKITTI, Waymo), `scannet-semseg-pt-v3m1-1-ppt-extreme` (larger PPT model),
`sonata_20251024.pth`, `concerto_large_origin.pth`, the larger and Waymo LitePT models,
the other Mask3D checkpoints. Pass the downloaded file as a path:
`pretrained: /data/weights/scannet200_unet_teacher.pth`, with the options the load
report asks for.

### 3.3 Measured fit

Verdicts of `grasppanda probe` (20 training iterations and a prediction on 8 real
GraspNet-1Billion frames; run `wave5-enc_sparse`, 2026-10-13, on an RTX A6000 shared with
other jobs). The same records are in
[Measured combinations](../advanced_guides/measured_combinations.md) with their numbers.

| Preset | GSNet | EconomicGrasp | FineGrasp | GraspNet-baseline | Scale-Balanced-Grasp |
|---|---|---|---|---|---|
| `minkunet18a` | runs | runs | runs | runs (adapter) | runs (adapter) |
| `minkunet34c` | runs | runs | runs | runs (adapter) | runs (adapter) |
| `minkunet34c-pointcontrast` | runs | runs | runs | runs (adapter) | runs (adapter) |
| `minkunet18a-openscene` | runs | runs | runs | runs (adapter) | runs (adapter) |
| `spunet` | runs | runs | runs | runs (adapter) | runs (adapter) |
| `spunet-gaussiancross` | runs | runs | runs | runs (adapter) | runs (adapter) |
| `ptv3` | runs | runs | runs | runs (adapter) | fails (width: needs 256) |
| `ptv3-scannet200` | runs | runs | runs | runs (adapter) | fails (width: needs 256) |
| `ptv3-ppt` | runs | runs | runs | runs (adapter) | fails (width: needs 256) |
| `ptv3-encoder` | runs-unstable | runs-unstable | runs | runs (adapter) | fails (width: needs 256) |
| `ptv3-sonata` | runs-unstable | runs-unstable | runs | runs (adapter) | fails (width: needs 256) |
| `ptv3-sonata-small` | runs-unstable | runs-unstable | runs | runs (adapter) | fails (width: needs 256) |
| `litept` | runs | runs | runs | runs (adapter) | fails (width: needs 256) |
| `litept-structured3d` | runs | runs | runs | runs (adapter) | fails (width: needs 256) |

*runs (adapter)*: `compose` wrapped the encoder in `FurthestPointSubsample`, because the
proposal of that method takes seeds from a sub-sampling encoder. *runs-unstable*: the
iterations completed and the model predicted, but a loss became non-finite or an
iteration had no real or no supervised seed (the linked page has the cell). *fails (width:
needs 256)*: the model was refused when built, with the message quoted in section 5.
`mask3d-scannet` and the other names of table 3.2 without a preset were loaded strictly
(section 3.2) but not probed under a method.

## 4. How to use each option

### 4.1 A new encoder under a method

```bash
grasppanda compose --list backbone                 # the presets, one line each
grasppanda compose --show backbone=minkunet18a     # one preset with its notes
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set backbone=minkunet18a -o gsnet_minkunet18a.yaml
grasppanda probe gsnet_minkunet18a.yaml            # a few iterations on real frames
grasppanda train gsnet_minkunet18a.yaml
```

The other four methods: `economicgrasp/economicgrasp_sunet14_g1b-realsense_10e`,
`finegrasp/finegrasp_sunet14_g1b-realsense_10e`,
`graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e`,
`scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e`. Under the last two
`compose` wraps the encoder in `FurthestPointSubsample` by itself:

```text
$ grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
      --set backbone=minkunet18a-openscene -o gnb_openscene.yaml

slot      component                           from
────────  ──────────────────────────────────  ─────────────────────────────────────────────────────────────────────
backbone  MinkUNet in FurthestPointSubsample  preset `minkunet18a-openscene`, in the adapter FurthestPointSubsample
proposal  SubsampledSeeds                     base config
view      ObjectnessViewStage                 base config
assigner  NearestLabelAssigner                base config
grouper   DepthCylinderGroup                  base config
head      OperationHead                       base config

what compose did
  • backbone: preset `minkunet18a-openscene` (MinkUNet). MinkUNet18A started from the OpenScene
    weights (distilled from CLIP features, ScanNet).
  • out_channels: 256, the output width of the encoder it replaces (PointNet2), so that the other
    stages keep the width they have in the base model and the encoder is the only stage that
    differs.
  • wrapped MinkUNet in the adapter FurthestPointSubsample (preset `furthest-point-subsample`):
    SubsampledSeeds requires 'subsampled' scene features, but MinkUNet returns 'dense' ones (one
    feature per scene point). An adapter changes the model; its notes below say how.
  ...
config check
  ✓ valid  (hash 1943062dfa2b)
```

The same as a YAML fragment, for a config written by hand (any member of the family):

```yaml
model:
  backbone:
    type: MinkUNet
    in_channels: 3            # what the method feeds: 3 ones, or 6 under FineGrasp
    out_channels: 512         # free 1x1 projection; null returns the decoder width (96)
    voxel_size: 0.005         # the encoder's unit of length, in metres
    layers: [2, 2, 2, 2, 2, 2, 2, 2]
    planes: [32, 64, 128, 256, 128, 128, 96, 96]
    stem_kernel: 5            # 5 in MinkUNet; 3 in the Res16UNet34C checkpoints
```

Settings that matter, for all sparse encoders:

- **`voxel_size` / `grid_size`**: 0.005 m keeps the grid of the reference methods, with
  about one point per voxel. Weights from room-scale data were trained at 0.02 to
  0.025 m; choosing that grid matches the weights and leaves a table-top frame with
  far fewer voxels (a surface has roughly a sixteenth of them at four times the edge). There is no published answer for grasping; it is the first thing
  to sweep (`--opts model.backbone.voxel_size=0.01`).
- **`out_channels`**: the width the later stages are built with. `compose` keeps the width
  of the encoder it replaces. For the transformer encoders it is fixed by the
  architecture.
- **depth against the recipe**: the reference recipes (Adam, 1e-3, no warm-up) were tuned
  for a 14-layer network. Deeper encoders and pretrained ones usually want a lower rate
  and a warm-up; none of this was tuned here.

### 4.2 Starting from released weights

1. Pick a row of table 3.2 and fetch the file into `$GRASPPANDA_CACHE/weights_components/`.
   You do not need to look up the link: build the model once and the error prints the
   command.

    ```text
    SpUNet: the released weights `gaussiancross-scannet-lin` (GaussianCross (Yao et al., ACM MM 2025);
    MIT; 157 MB) are not on this machine. Fetch them with
      mkdir -p $GRASPPANDA_CACHE/weights_components && curl -L -o $GRASPPANDA_CACHE/weights_components/spunet_gaussiancross_scannet_lin.pth https://huggingface.co/RayYoh/GaussianCross/resolve/3cbceee8b88113780cec6a734fea8b14e2baae4f/scannet/semseg-spunet-base-lin/model/model_best.pth
    or set `pretrained: null` to build the encoder without them (for example to load a grasp
    checkpoint that already holds the encoder).
    ```

2. Compose with the preset of that release, or add two lines to any config of the class:

    ```yaml
    model:
      backbone:
        type: SpUNet
        in_channels: 3
        out_channels: 512
        grid_size: 0.005
        pretrained: gaussiancross-scannet-lin   # a name of table 3.2, or a file path
        pretrained_mismatch: skip               # see below
    ```

3. Read the load report in the log. It names every tensor that did not come from the
   file. These are the reports of three presets, as printed:

    ```text
    SpUNet: pretrained weights from $GRASPPANDA_CACHE/weights_components/spunet_gaussiancross_scannet_lin.pth (prefix `module.backbone.`): 353 tensors loaded; 2 absent from the file (final.weight, final.bias); 1 of another shape (conv_input.0.weight (32, 5, 5, 5, 6) in the file, (32, 5, 5, 5, 3) here); 2 tensors outside the encoder ignored
    MinkUNet: pretrained weights from $GRASPPANDA_CACHE/weights_components/minkunet18a_openscene_scannet_openseg.pth.tar (prefix `net3d.`): 288 tensors loaded; 1 absent from the file (final.bias); 1 of another shape (final.weight (768, 96) in the file, (512, 96) here)
    PTv3: pretrained weights from $GRASPPANDA_CACHE/weights_components/ptv3_ppt_s3dis.pth (prefix `module.backbone.`): 507 tensors loaded; 1 of another shape (embedding.stem.conv.weight (32, 5, 5, 5, 6) in the file, (32, 5, 5, 5, 3) here); 5 tensors outside the encoder ignored
    ```

How the option behaves:

- **What is loaded.** The encoder alone. The loader finds the prefix under which the file
  keeps it (`module.backbone.`, `net3d.`, `model.backbone.`, none), ignores heads and
  optimiser state, and re-lays kernels where the original library stores them
  differently (Pointcept 1×1 kernels for `SpUNet`, MinkowskiEngine kernels for
  `MinkUNet`).
- **`pretrained_mismatch: error`** (the default) refuses when a tensor of the encoder is
  absent from the file or has another shape there. Use it with the options of table 3.2
  to be sure that everything came from the file.
- **`pretrained_mismatch: skip`** keeps the initialisation of exactly those tensors and
  lists them. A grasp model needs it almost always, because of two layers: the **stem**
  (the releases read 6 or 9 real channels, colour and normal; the grasp methods feed 3
  ones) and the **final projection** when `out_channels` differs. Everything between
  them is loaded.
- **No pickled code runs.** Training checkpoints of research code pickle schedulers and
  numpy scalars; the reader maps every such name to an inert object and reads only
  tensors.
- **Prompt-driven normalisation** (`ptv3-ppt`): the checkpoint holds three sets of
  normalisation layers, one per training dataset. `pretrained_condition: 0 | 1 | 2`
  takes one set; the model itself is a plain `PTv3`.
- **Encoder-only PTv3** (`ptv3-encoder`): `enc_mode: true` builds no decoder and returns
  the encoder stages concatenated on every voxel (`concat_stages`). With `pretrained:` a
  full checkpoint loads its encoder half; the report lists the 200 decoder tensors as
  "in the file but not in the model".

Which release for which use, as far as the sources say (none was evaluated for grasping
here):

| If you want | Take | Why |
|---|---|---|
| weights whose input is what the grasp methods feed | `minkunet18a-openscene` | trained on three constant ones per point, so stem included everything loads under GSNet, EconomicGrasp, GraspNet-baseline and Scale-Balanced-Grasp |
| language-aligned point features | the same, with `--opts model.backbone.out_channels=768 model.backbone.final_bias=false` | the released output layer predicts OpenSeg / CLIP features; the load is then strict |
| pre-training on single partial RGB-D frames | `minkunet34c-pointcontrast` | PointContrast trains on pairs of ScanNet frames, the regime of one table-top view |
| permissive licence, self-supervised | `spunet-gaussiancross` | MIT on the card |
| supervision from more than one indoor dataset | `ptv3-ppt` | trained on ScanNet, S3DIS and Structured3D jointly |
| the smallest download | `litept-structured3d`, `ptv3-sonata-small` (NC) | about 150 MB each |

## 5. Pitfalls met while building this

- **A grasp checkpoint still wants the released file.** `pretrained` loads when the
  encoder is constructed, also when you then load a grasp checkpoint with `--weights` or
  resume a run. Keep the file in place, or set `pretrained: null` for evaluation (the
  grasp checkpoint holds the whole encoder).
- **The stem is the tensor that does not load**, and it is the layer that reads the
  input. Under FineGrasp `compose` sets `in_channels: 6`, so a 6-channel stem *does*
  load there, but nothing says that the six channels of that method are the colour and
  normal, in the scaling, the release was trained on. Loading says nothing about meaning.
- **The default Concerto files are not the paper's Concerto.** The model card calls
  `concerto_{tiny,small,base,large}.pth` pre-release Utonia models; the paper's model is
  `concerto_base_origin.pth`. The card also says the default files accept inputs without
  colour and normal ("please set them to zeros"), which is closer to what a grasp model
  has.
- **Licences differ inside one family.** PTv3 and LitePT weights are MIT; Sonata and
  Concerto weights are CC BY-NC 4.0 (the Sonata hub card states no licence, the
  repository README does). PointContrast, Mask3D and OpenScene publish code under MIT or
  Apache-2.0 and say nothing about the weights, which were trained on ScanNet.
- **`SpUNet` is not `MinkUNet34C`.** Same block counts and widths; in Pointcept's network
  the stride-2 convolution changes the width, in MinkowskiEngine's the first block of the
  stage does. A checkpoint of one does not fit the other; the loader says "no tensor
  fits" or lists hundreds of mismatches rather than loading half of it.
- **Two kernel sizes may not share a spconv indice key.** A 5³ stem that shared the key of
  the 3³ blocks of its level built, ran on CPU with a 3³ stem and failed on the GPU with
  `you must provide same algo to layers share same indice key`. The stem of `MinkUNet`
  has its own key; the unit test runs a 5³ stem.
- **Fixed-width encoders under Scale-Balanced-Grasp** fail when the model is built
  (`DepthCylinderGroup: the gated seed residual adds 64-channel seed features to
  256-channel crop features`). Use an encoder with a free `out_channels`, or change the
  grouper's residual.
- **Old checkpoints in the legacy (non-zip) PyTorch format that pickle foreign objects**
  cannot be read without executing them and are refused (met with the Contrastive Scene
  Contexts files). Re-save the state dict in the environment that wrote it.
- **Probes on a shared GPU.** Several cells first "failed" with `CUDA out of memory`
  because other jobs held the card; such records say nothing about the config and were
  probed again.

## 6. Not wrapped

| Candidate | Opened | Why not |
|---|---|---|
| Contrastive Scene Contexts, Res16UNet34C pre-training ([repository](https://github.com/facebookresearch/ContrastiveSceneContexts), MIT, archived) | `partition8_4096_100k.pth` was downloaded | The file is a legacy-format pickle with objects of its code base; the reader refuses to execute it. The network is the one `pointcontrast-nce` loads into. |
| FCAF3D / TR3D sparse backbones (single-view RGB-D detection; `MinkResNet` in [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) @ `fe25f7a`, Apache-2.0) | `mmdet3d/models/backbones/mink_resnet.py` | Kernel-3 stride-2 convolutions, max pooling and strided 1×1 shortcuts in the MinkowskiEngine rule: layers the facade does not have, and not what spconv's strided layer computes. It returns a pyramid of coarse levels, not per-point features. |
| SPVCNN / SPVNAS ([mit-han-lab/spvnas](https://github.com/mit-han-lab/spvnas), MIT, archived) | repository metadata only | It runs on TorchSparse ([mit-han-lab/torchsparse](https://github.com/mit-han-lab/torchsparse), MIT), a second sparse library the facade has no backend for. The research report describes a point branch with trilinear devoxelisation and weights for driving LiDAR; neither was read here. |
| PonderV2 `SpUNet-v1m3` ([hub](https://huggingface.co/HaoyiZhu/PonderV2), MIT; six files of about 350 MB) | file list, Pointcept `spconv_unet_v1m3_pdnorm.py` | Another module tree than `SpUNet-v1m1` (named `conv`/`bn` members, `bns.<i>` per dataset); needs a key map in addition to the condition choice. Small follow-up. |
| Point-MoE ([hub](https://huggingface.co/uva-cv-lab/Point-MoE), MIT) | hub metadata | According to the key comparison of the research report its attention projections are mixtures of experts (`attn.proj.experts.*`): a change of the block, not weights for `PTv3` as it is. Not downloaded. |
| D-DITR distilled PTv3 ([hub](https://huggingface.co/rwthvision/ditr)) | hub metadata | The weights are **AGPL-3.0**. They have the plain PTv3 layout, so `pretrained: <path>` is expected to load them; not downloaded, not listed. |
| LAM3C ([hub](https://huggingface.co/aist-cvrt/lam3c-roomtours), CC BY-NC 4.0) | hub metadata | Non-commercial; same layout as Sonata, so `pretrained: <path>` on `PTv3Sonata` is expected to load it. Not downloaded. |
| GraspGen object encoder ([NVlabs/GraspGen](https://github.com/NVlabs/GraspGen), "NVIDIA License") | licence file, hub file list | Non-commercial code and weights, and an object-level encoder (one segmented object, not a scene). `PTv3(in_channels: 3, enc_mode: true)` is the architecture; nothing was downloaded. |
| Mosaic3D SpUNet ([hub](https://huggingface.co/junhalee/Mosaic3D)) | hub metadata | No licence on the card. |
| OA-CNNs weights | Pointcept hub organisation | None released; the class stays random-init. |
