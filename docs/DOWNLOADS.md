# Data and checkpoint downloads

Choose the inputs for your first experiment; you can add training archives later.

| Start with | What to prepare |
|---|---|
| An author sample, without GraspNet | Select ASGrasp in the UI, load its preset and download its registered weights. |
| A GraspNet frame | Download `test_seen.zip`, set the dataset root and download weights for your method and camera. |
| Training or evaluation | Add the models, labels and method-specific targets described below. |
| GtG2 candidate graphs | Follow [Candidate graph experiments](REFERENCE.md#candidate-graph-experiments) to prepare graphs and train an ensemble. |
| SPGrasp planar sequences | Training needs RGB, instance labels and `rect_labels.zip`; prediction needs RGB, first-frame prompts and your trained checkpoint. |

## Start without GraspNet

After [installation](INSTALL.md), run the ASGrasp stereo sample without downloading GraspNet:

```bash
./panda weights asgrasp
./panda init --method asgrasp -o sample.local.yaml
./panda run sample.local.yaml
```

The pinned implementation includes the sample inputs; the first command downloads the two required networks. Leave the generated input settings unchanged. In the browser, select **ASGrasp → Load preset → Download registered weights → Run current form**, then open **Runs & results** to inspect the prediction. A supported GPU and the shared runtime are still required.

ZeroGrasp and SpaHybGen also provide author samples. These fixed recipes have their own observation protocols; see [Methods & papers](METHODS.md) before substituting your own inputs.

## GraspNet-1B

Use the [official dataset download page](https://graspnet.net/datasets.html), which lists Google Drive, Baidu and SJTU Jbox mirrors. Its archives contain 190 scenes across two cameras. For the real-frame examples, begin with **test_seen.zip** (approximately 20 GB compressed), containing scene 0100. Training images are separate archives. Evaluation additionally needs the object models and grasp/collision labels; the optional Dex-Net cache accelerates evaluation. See the [official API documentation](https://graspnetapi.readthedocs.io/en/latest/) for data and evaluation formats.

Download a listed archive through your browser. On a server, use the exact Google Drive link with the installed `gdown` CLI:

```bash
.venv/bin/gdown 'https://drive.google.com/file/d/1_nxiCmHhtsjCgA1IKJn3AuMq4SH_fseW/view' -O test_seen.zip
mkdir -p /data/GraspNet-1B
unzip -l test_seen.zip | head
```

Use a writable dataset location of your choice; `/data/GraspNet-1B` is an example. Choose the extraction command that matches the archive listing:

| First directory in the archive | Extract with |
|---|---|
| `scene_0100/` | `unzip -n test_seen.zip -d /data/GraspNet-1B/scenes` |
| `scenes/scene_0100/` | `unzip -n test_seen.zip -d /data/GraspNet-1B` |

If your mirror adds another enclosing folder, place its `scene_*` directories under the final `scenes/` directory. Avoid `scenes/scenes/`. The toolkit never downloads the full dataset automatically.

```text
/data/GraspNet-1B/
├── scenes/
│   └── scene_0100/
│       ├── kinect/
│       └── realsense/
│           ├── rgb/0000.png
│           ├── depth/0000.png
│           ├── label/0000.png
│           ├── meta/0000.mat
│           ├── camera_poses.npy
│           └── cam0_wrt_table.npy
├── models/             # official evaluation
├── grasp_label/        # training/evaluation
└── collision_label/    # training/evaluation
```

The supplied frame presets use scene 0100/frame 0000. Training-step examples use a training scene. The dataset and models retain the [publisher's terms](https://graspnet.net/datasets.html#license); this repository does not redistribute them.

<details>
<summary>All official archive links and mirrors</summary>

Links were collected from the official page for this release. If a mirror changes or reports a quota, return to that page and select another mirror.

| Archive | Mirrors |
|---|---|
| train_1.zip | [Google](https://drive.google.com/file/d/1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1hchwl8Sk9kk_i8t1JbcLXw) · [Jbox](https://jbox.sjtu.edu.cn/l/71Kb9K) |
| train_2.zip | [Google](https://drive.google.com/file/d/1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/158mOzU6bx4cvexQx5FXn_A) · [Jbox](https://jbox.sjtu.edu.cn/l/G57uyS) |
| train_3.zip | [Google](https://drive.google.com/file/d/1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1D9Mq6VsIHE7Pra8QplEWkQ) · [Jbox](https://jbox.sjtu.edu.cn/l/wJorXZ) |
| train_4.zip | [Google](https://drive.google.com/file/d/1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1A3Tyc7l_u9UwgKqhVJSrNg) · [Jbox](https://jbox.sjtu.edu.cn/l/SHwJVL) |
| grasp_label.zip | [Google](https://drive.google.com/file/d/1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/18yLPWIwM9uJBih6GMoQRNg) · [Jbox](https://jbox.sjtu.edu.cn/l/noXqUa) |
| collision_label.zip | [Google](https://drive.google.com/file/d/1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1cj3Wea0RtgHrLb4iGUyA1g) · [Jbox](https://jbox.sjtu.edu.cn/l/DuUptQ) |
| test_seen.zip | [Google](https://drive.google.com/file/d/1_nxiCmHhtsjCgA1IKJn3AuMq4SH_fseW/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/11_mTb5p0g6PqE8ZrOw2T8A) · [Jbox](https://jbox.sjtu.edu.cn/l/XH2KQl) |
| test_similar.zip | [Google](https://drive.google.com/file/d/1njgthC-uUvTXgG99qq1fjS-fzofttFms/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1gVkvNS2Q9P0SS_N9Lm7hFQ) · [Jbox](https://jbox.sjtu.edu.cn/l/k03rDE) |
| test_novel.zip | [Google](https://drive.google.com/file/d/1xixvgY0yK7TEALq3k7JcJk2_SP_6r8nk/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1KNcLMwJCTxKGMSt7WPyZOw) · [Jbox](https://jbox.sjtu.edu.cn/l/iJFKg4) |
| rect_labels.zip | [Google](https://drive.google.com/file/d/1lR6ZSgtgV1KlqzM14mKlQ8oKhE3UCltO/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/15k9ko5iCoLTgufbn_6YXaA?pwd=nhp2) |
| dex_models.zip | [Google](https://drive.google.com/file/d/1RElNqUHNoA9l_muTGNu7yAc3ql_e7pL3/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1KTPJMAayVQkgx2uwUCNMOQ) |
| models.zip | [Google](https://drive.google.com/file/d/1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5/view?usp=sharing) · [Baidu](https://pan.baidu.com/s/1SoaE_7AqfR5R6w8dO79rsg) · [Jbox](https://jbox.sjtu.edu.cn/l/jFF3no) |

</details>

### Run your first GraspNet frame

After extracting the images, run these commands from the installed repository:

```bash
./panda weights graspness --camera realsense
./panda init --method graspness \
  --dataset-root /data/GraspNet-1B -o graspness.local.yaml
./panda run graspness.local.yaml
./panda ui
```

This preset predicts scene 0100, frame 0000 with the registered RealSense checkpoint. Open **Runs & results** to view the prediction. The dataset root must contain `scenes/scene_0100/realsense/`; a scene directory itself is not the dataset root. Configuration, downloaded weights and prediction files are created locally. To change the method or camera, load its preset and download its matching weights; training requires the additional targets below.

## Method-specific preprocessing

Basic frame inference does not require downloading all training targets. Additional processing is method-specific:

| Workflow | Additional requirements | Instructions |
|---|---|---|
| Baseline / SBG training | `tolerance/` plus original grasp and collision labels | [Baseline](https://github.com/graspnet/graspnet-baseline), [SBG](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) |
| Graspness training | `graspness/` and `grasp_label_simplified/`, plus original labels | [Graspness implementation](https://github.com/rhett-chen/graspness_implementation) |
| HGGD training | Author-preprocessed 2D/local targets under `HGGD_Preprocessed/6dto2drefine_CAMERA/6d_dataset/scene_0/grasp_labels/0_view.npz`; override `label_root` in JSON when using a different location | [HGGD preparation and downloads](https://github.com/THU-VCLab/HGGD#data-preparation) |
| RNG short training | HGGD preprocessed labels; native proposals and local targets are prepared in the experiment directory. The unreleased training schedule is not reproduced | [RNG](https://github.com/THU-VCLab/RegionNormalizedGrasp) |
| FGC short training | Original grasp/collision labels and author `FGC_label/` scores | [FGC](https://github.com/luyh20/FGC-GraspNet) |
| EconomicGrasp training | `economic_grasp_label_300views/` and `graspness/` | [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp) |
| DOGraspNet short training | Simplified grasp/collision labels and `graspness_label/`; the pinned equivalent `graspness/` targets are accepted | [DOGraspNet](https://github.com/huamo555/DOGraspNet) |
| Generalizing-Grasp inference / short training | Fused XYZ/normals for inference; training additionally needs matched segmentation, full grasp/collision labels, `tolerance/` and object SDF grids | [Original method repository](https://github.com/mahaoxiang822/Generalizing-Grasp) |
| ASGrasp fixed recipe | Bundled RGB and left/right IR sample; these IR inputs are not supplied by standard RGB-D alone | [Original method repository](https://github.com/jun7-shi/ASGrasp) |

Short training for ContactGraspNet, GraNet and RGB Matters generate the selected frame's native targets inside its experiment directory. CenterGrasp generates object 000 SGDF targets from the official mesh/grasp labels and Kinect RGB targets from segmentation/poses; download both its RGB and SGDF weights first. Its native mesh sampler requires the geometry dependencies installed by `./panda install`. A scene-wise folder named `SGDF` is not the CenterGrasp object-level format.

GraspBalance requires original grasp/collision labels and tolerance. Graspness modern requires the same simplified labels and graspness maps as its ResUNet implementation. All preprocessing remains method-specific; an identical folder name does not guarantee compatible supervision.

### Generalizing-Grasp SDF grids

Download the official `models.zip` and the author's [fused data](https://drive.google.com/file/d/12YODD0ZUu6XTudU1fZBhVtAmIpMZk8xQ/view?usp=sharing). Follow the [native fusion preparation](https://github.com/mahaoxiang822/Generalizing-Grasp) for matched point/segmentation arrays. The toolbox rejects mismatched row counts instead of guessing correspondence.

Generate the 88 SDF grids in a writable cache, keeping the source dataset read-only:

```bash
./panda prepare-sdf \
  --dataset-root /data/GraspNet-1B \
  --output-root /data/GraspPanda-cache/generalizing-sdf
```

Set `sdf_root: /data/GraspPanda-cache/generalizing-sdf` in experiment JSON/YAML. The generated layout is `models/000/grid_sampled_sdf.npz` through `models/087/grid_sampled_sdf.npz`; omit `sdf_root` if those files already exist under the dataset root. `--objects 0` checks one object's preprocessing; the native training loss initializes all 88 objects and therefore requires the complete set. Existing files are not overwritten. Preparation records include source/output hashes and sampler settings.

Check each original README for the full training-data preparation. Keep large derived files in a dataset/cache location or ignored experiment directory.

## Registered checkpoints

<details>
<summary>Download links and local paths for each method</summary>

```bash
./panda weights hggd --camera realsense
./panda weights region_normalized_grasp --camera kinect
./panda weights economicgrasp --camera kinect
./panda weights asgrasp
```

The UI's **Download registered weights** button invokes the same downloader. Every required role is downloaded, extracted when necessary, and verified against `grasppanda/resources/checkpoints.json`. A mismatched file is not installed. For manual downloads, place the exact file at the registered path. Camera-agnostic weights are marked `any`; the recipe camera still defines the input format.

Google Drive downloads support resume. Other HTTP downloads restart cleanly because some academic mirrors ignore range requests. If a host is unavailable, use the author link and retry later. Model availability and licensing are controlled by the authors.

<details>
<summary>All registered checkpoint files and author download links</summary>

| Method | Camera | Role | Local target | Author download |
|---|---|---|---|---|
| graspness | realsense | primary | `checkpoints/graspness/checkpoint-rs.tar` | [Download](https://drive.google.com/file/d/1RfdpEM2y0x98rV28d7B2Dg8LLFKnBkfL/view) |
| graspnet_baseline | realsense | primary | `checkpoints/graspnet_baseline/checkpoint-rs.tar` | [Download](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view) |
| fgc_graspnet | realsense | primary | `checkpoints/fgc_graspnet/checkpoint-rs.tar` | [Download](https://drive.google.com/file/d/1Y-CWHr_eZDoZm3XJocrUJq1SA5tfrONX/view) |
| scale_balanced_grasp | realsense | primary | `checkpoints/scale_balanced_grasp/checkpoint-rs.tar` | [Download](https://drive.google.com/file/d/1Pp-xFL0QrWcEK1tpVXM3c0RnpMcRwzIX/view) |
| hggd | realsense | primary | `checkpoints/hggd/checkpoint-realsense.tar` | [Download](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/files/?p=%2FHGGD_realsense_checkpoint&dl=1) |
| hggd | kinect | primary | `checkpoints/hggd/checkpoint-kinect.tar` | [Download](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/files/?p=%2FHGGD_kinect_checkpoint&dl=1) |
| region_normalized_grasp | realsense | primary | `checkpoints/region_normalized_grasp/checkpoint-realsense.tar` | [Download](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/files/?p=%2FRNGNet_realsense_checkpoint&dl=1) |
| region_normalized_grasp | kinect | primary | `checkpoints/region_normalized_grasp/checkpoint-kinect.tar` | [Download](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/files/?p=%2FRNGNet_kinect_checkpoint&dl=1) |
| dograspnet | realsense | primary | `checkpoints/dograspnet/checkpoint-realsense.tar` | [Download](https://drive.google.com/file/d/1ykH4W5KZEG5u-ERXyFqf5PTzXuvhKm3z/view) |
| contact_graspnet_g1b | realsense | primary | `checkpoints/contact_graspnet_g1b/checkpoint-realsense.tar` | [Download](https://drive.google.com/file/d/1biNXRIZ6V--ivLIYGgjojXIzrCiJeZXl/view) |
| active_ngf | realsense | primary | `checkpoints/active_ngf/checkpoint-realsense.tar` | [Download](https://drive.google.com/file/d/1OswUcXVJv_LAgyyNt_KjfOPhIE4LEyk7/view) |
| motiongrasp | realsense | primary | `checkpoints/motiongrasp/checkpoint-realsense.tar` | [Download](https://drive.google.com/file/d/1EjVGnWOMxLAfnebaC_b6Z-IkHTcIg5f7/view) |
| zerograsp | any | primary | `checkpoints/zerograsp/checkpoint-realsense.tar` | [Download](https://drive.google.com/file/d/1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ/view) |
| rgb_matters | realsense | primary | `checkpoints/rgb_matters/checkpoint-realsense.tar` | [Download](https://drive.google.com/file/d/1H4JF3saXgbP5FfYlaXctbsF-6onx_MUz/view) |
| economicgrasp | kinect | primary | `checkpoints/economicgrasp/checkpoint-kinect.tar` | [Download](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v1/economicgrasp_kinect.tar) |
| generalizing_grasp | realsense | primary | `checkpoints/generalizing_grasp/model.tar` | [Download](https://drive.google.com/file/d/1WJj54l7MxFO1kgXoXA9tF6FCfB2okKr3/view) |
| gfla | realsense | primary | `checkpoints/gfla/checkpoint-realsense.tar` | [Download](https://github.com/Nx1021/GFLA-release/releases/download/v1.0.0/realsense_best.pth) |
| gfla | realsense | segmentation | `checkpoints/gfla/FastSAM-s.pt` | [Download](https://github.com/Nx1021/GFLA-release/releases/download/v1.0.0/FastSAM-s.pt) |
| centergrasp | kinect | primary | `checkpoints/centergrasp/ckpt_rgb/el6oa23g/epoch=704-step=564000.ckpt` | [Download](https://centergrasp.cs.uni-freiburg.de/download/ckpt_rgb/el6oa23g.zip) |
| centergrasp | kinect | shape | `checkpoints/centergrasp/ckpt_sgdf/6953cfxt/epoch=199-step=35200.ckpt` | [Download](https://centergrasp.cs.uni-freiburg.de/download/ckpt_sgdf/6953cfxt.zip) |
| graspness_modern | realsense | primary | `checkpoints/graspness_modern/checkpoint-realsense.tar` | [Download](https://github.com/SimonHanrath/graspness_modern/releases/download/v1.0.0/gsnet_resunet14_epoch10.tar) |
| asgrasp | realsense | primary | `checkpoints/asgrasp/graspness.tar` | [Download](https://drive.google.com/file/d/1T8-4UaH1MBWqKt_YacDE0jM2-rHqpgJR/view) |
| asgrasp | realsense | stereo | `checkpoints/asgrasp/raftmvs.pth` | [Download](https://drive.google.com/file/d/1EDfJEVfumWRKhvGv0e9WQ0lHPBr8Y1dO/view) |
| dreds | any | primary | `checkpoints/dreds/model.pth` | [Download](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/checkpoint/SwinDRNet/models/model.pth) |
| spahybgen | any | primary | `upstream/related/multi_hand/spahybgen/assets/trained_models/spahybgen_unet_64_voxel.pt` | [Download](https://raw.githubusercontent.com/wangzivector/SpaHybGen/39e5794506bbbf2fd41804f73c93b7b393be550e/assets/trained_models/spahybgen_unet_64_voxel.pt) |
| graspfast | realsense | primary | `checkpoints/graspfast/graspfast_checkpoint.tar` | [Download](https://media.githubusercontent.com/media/YZ-331/GraspFast/fa028134b3a1b0271da7acb8bac7463e6585985e/logs/trained_model_weight/graspfast_checkpoint.tar) |

</details>

MotionGrasp also requires the baseline checkpoint; the downloader includes it. The PointNet2 compatibility port reuses the baseline weights. RNGNet SDK and SpaHybGen sample weights are included by their upstream repositories. GraNet's organized source and separately distributed legacy checkpoints are not interchangeable. Methods without registered, compatible weights are explicitly identified in [Methods & papers](METHODS.md).

</details>

## Additional inputs

**SPGrasp:** `./panda weights spgrasp` downloads the [SAM2.1 Hiera Base+ initializer](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) to `checkpoints/spgrasp/sam2.1_hiera_base_plus.pt`. This initializes training; it is not a trained grasp detector and inference rejects it. No author fine-tuned SPGrasp weights are registered. Use your training run's `checkpoint.pt` for prediction. Training requires consecutive RGB and instance-label frames, with rectangle labels in `rect_labels/scene_XXXX/CAMERA/FFFF.npy` or `scenes/scene_XXXX/CAMERA/rect/FFFF.npy`. Set `label_root` to select another rectangle root containing `scene_XXXX/CAMERA/`. The [SAM2 author instructions](https://github.com/facebookresearch/sam2#download-checkpoints) describe the initializer, and the [SPGrasp guide](REFERENCE.md#prompted-planar-sequences) explains prompts and width units.

ZeroGrasp's inference recipe uses the author's RGB-D and instance-mask sample. Its separate reconstruction training dataset is available through the [author download script](https://github.com/sh8/ZeroGrasp/blob/main/download.sh). Standard GraspNet frames do not provide that supervision.

ActiveNGF performs online field optimization from multiple views through its native recipe. SpaHybGen uses an author voxel sample for its contact optimization recipe. These input protocols are listed in [Methods & papers](METHODS.md).

GFLA short training prepares native contact labels and surface/visibility targets inside the run directory. First-time preparation may take several minutes. Set `data_workers: 1` to limit preparation concurrency; `label_root` can reuse a previous run's `prepared/ctt_grasp_label` cache. Its contact/SDF objective uses `learning_rate: 0.000002`.

GraspFast short training generates native graspability targets in the run directory. It requires `grasp_label/`, `grasp_label_simplified/` and `collision_label/` and uses the released unweighted five-part objective.

## FineGrasp

<details>
<summary>FineGrasp data and initialization</summary>

`./panda weights finegrasp --camera realsense` downloads the author's pinned `model.safetensors` and companion `model.config.json`. Keep both files together; inference loads the exact selected checkpoint with its architecture configuration.

[Author weights](https://huggingface.co/HorizonRobotics/FineGrasp) · [Original implementation](https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/projects/finegrasp_graspnet1b)

The preset accepts RGB-D and camera metadata. The detector uses depth-derived XYZ and estimated normals; RGB is retained by the native input wrapper and preview. Native camera-space limits are x/y in [-1, 1] metres and z in [0, 2] metres, with no GT segmentation mask. Point count, voxel size, collision threshold and random seed are configurable.

For training, prepare `economic_grasp_label_300views/` using the [EconomicGrasp author instructions](https://github.com/iSEE-Laboratory/EconomicGrasp), plus either `instance_norm_graspness/` or the workspace-ordered `graspness/` maps. The reader also needs RGB/depth, segmentation, camera metadata and poses for the training scenes. No training labels are needed for the FineGrasp inference preset.

Existing `scenes/scene_XXXX/CAMERA/normal/FFFF.npy` and `instance_norm_graspness/scene_XXXX/CAMERA/FFFF.npy` are used directly. When absent, GraspPanda creates derivatives on demand under the run's `prepared/finegrasp/`, or a reusable `label_root` cache. It checks source and generated-file hashes before reuse; original dataset files are never modified.

Generated graspness follows the paper's per-instance min-max normalization and subsequent scene normalization. Applying this to an already scene-normalized map gives the same values for nonconstant objects; constant instances and background receive zero. Normal generation uses the author's Open3D estimator (0.1-metre radius, 30 neighbors) on the full valid workspace cloud. Signed float32 maps are stored scaled by 255 to match the native dataset reader. The author offline normal-generation script is not released, so this is a documented preprocessing adaptation, not a claim of identical author training data. Full normal maps can occupy substantial disk space; choose a writable cache with adequate capacity.

The toolbox also corrects native flip augmentation to transform normals together with points and poses. See [FineGrasp composition](REFERENCE.md#finegrasp-training-and-composition) for training, loss and resume settings.

</details>

## Pretrained image components

<details>
<summary>DINO image encoder weights</summary>

The image encoders have separate initialization weights from the method's grasp checkpoint. Downloads stay under `checkpoints/components/`; the source repository and its archives do not contain these files. The registry pins each source revision, byte size, SHA256 and RGB normalization in `grasppanda/resources/component_weights.json`.

```bash
./panda component-weights dinov3_small
./panda weights hggd --camera realsense
```

| ID | Weight conversion and source | Size | Terms |
|---|---|---|---|
| `dinov2_small` | [timm DINOv2 Small](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) | 88 MB | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| `dinov2_base` | [timm DINOv2 Base](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m) | 346 MB | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| `dinov3_small` | [timm DINOv3 Small](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) | 86 MB | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| `dinov3_base` | [timm DINOv3 Base](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m) | 343 MB | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |

These registered conversions can be fetched without account credentials. An unavailable or changed download is reported explicitly; incompatible or corrupted files are not installed. Configure `pretrained: false` only when you intend random initialization. See [DINO composition](REFERENCE.md#pretrained-dino-image-features) for freezing and fine-tuning settings.

</details>

## Pretrained point components

<details>
<summary>Concerto and Utonia weights</summary>

Point initialization weights are separate from grasp checkpoints and are downloaded locally under `checkpoints/components/`. Select the encoder in **Compose modules** and use its pretrained-encoder download button, or run:

```bash
./panda component-weights concerto_tiny
./panda component-weights utonia
```

| ID | Author weights | Size |
|---|---|---|
| `concerto_tiny` | [Concerto](https://huggingface.co/Pointcept/Concerto) | 19 MB |
| `concerto_small` | [Concerto](https://huggingface.co/Pointcept/Concerto) | 155 MB |
| `concerto_base` | [Concerto](https://huggingface.co/Pointcept/Concerto) | 434 MB |
| `concerto_large` | [Concerto](https://huggingface.co/Pointcept/Concerto) | 831 MB |
| `utonia` | [Utonia](https://huggingface.co/Pointcept/Utonia) | 549 MB |

The registry fixes each weight revision, SHA256, byte size and encoder configuration. Files are loaded with PyTorch's restricted weights-only loader. Initial training prepares a missing registered file; inference or resume from a complete grasp checkpoint does not require the initialization download. No weight files are included in the repository.

Author code is Apache-2.0; pretrained weights are CC-BY-NC-4.0. Those weight terms also matter when using or sharing a trained model initialized from them. See the [Utonia terms](https://github.com/Pointcept/Utonia#license) and [Concerto terms](https://github.com/Pointcept/Concerto#license), and [component controls](REFERENCE.md#pretrained-point-encoders) for input and fine-tuning settings.

</details>

## MambaVision initialization

<details>
<summary>MambaVision encoder weights</summary>

MambaVision uses author ImageNet-1K **Safetensors** weights, separate from the grasp checkpoint. Select `mambavision` under **Compose modules** and prepare its encoder weights, or use:

```bash
./panda component-weights mambavision_tiny
```

Available IDs are `mambavision_tiny`, `mambavision_tiny2`, `mambavision_small`, `mambavision_base`, `mambavision_large` and `mambavision_large2`. Sources, exact revisions, sizes and hashes are registered in `grasppanda/resources/component_weights.json`; files are generated locally under `checkpoints/components/`. The [author repository](https://github.com/NVlabs/MambaVision) links each weight release. Only the selected encoder weights are downloaded.

Source and weights use NVIDIA non-commercial research terms. See [component configuration](REFERENCE.md#mambavision-hybrid-image-hierarchy) for RGB-D fusion, freezing and structural edits. The larger author pickle training archives are not required.

</details>

## Scale-Balanced-Grasp clean scenes

<details>
<summary>Prepare noisy-clean training and object-balanced sampling</summary>

NcM training needs CAD-aligned observations in addition to the original grasp, collision and tolerance labels. Download the official `models.zip` and training scenes, then prepare a reusable local cache:

```bash
./panda prepare-clean-scenes --dataset-root /data/GraspNet-1B \
  --output-root outputs/prepared/sbg-clean --camera realsense
./panda weights scale_balanced_grasp --camera realsense
./panda init --example train-sbg-ncm -o ncm.local.yaml
```

The command defaults to all training scenes and frames. Add `--workers 4` to prepare frames in parallel in the same environment; the default is one process. Reduce the worker count if CPU memory or storage bandwidth is limited. For a small initial experiment, append `--scenes 0 --frames 0 1` and set `train_batch_limit: 1`, `batch_size: 2` in the training configuration. Set `dataset_root`, `label_root` and checkpoint paths before running. Validation uses ordinary observed frames and needs no clean validation cache.

The generator uses the selected camera's XML poses, a 5 mm CAD voxel grid and an 8 mm agreement threshold against observed workspace depth. `--voxel-size` and `--distance` change these geometric choices in metres. Points and instance IDs are saved with source hashes under the output directory. Camera-specific manifests prevent cross-camera reuse. Existing matching frames are reused; changed or incomplete caches require a new output directory. Cache generation is locked per frame. The dataset is read only. This corrects the fixed Kinect XML selection in the author's generation script.

OBS inference has a separate segmentation checkpoint and needs no clean-scene cache:

```bash
./panda component-weights scale_balanced_dsn
./panda init --example infer-sbg-obs -o obs.local.yaml
```

[Author segmentation checkpoint](https://drive.google.com/file/d/1Fe6RPN9cwEk6SsvGix9huspZf9Qz2yju/view) · [Original implementation](https://github.com/mahaoxiang822/Scale-Balanced-Grasp). The registered checkpoint is for RealSense; it is verified by the component-weight registry. See [sampling and training parameters](REFERENCE.md#scale-balanced-grasp-components).

</details>

## Contact-score refinement

Download the auxiliary networks once in the shared environment:

```bash
./panda component-weights generalizing_contactnet
./panda component-weights generalizing_scorenet
./panda component-weights scale_balanced_dsn
./panda init --example refine-hggd -o refinement.local.yaml
```

In the browser, expand **Refine grasps** and use **Prepare refinement networks**. These weights are separate from the selected method's checkpoint.

| Network | Author download |
|---|---|
| ContactNet | [ZIP](https://drive.google.com/file/d/1yMZ5rgloo0xbYvuR46t3sSKMvaVpOavx/view) |
| ScoreNet | [ZIP](https://drive.google.com/file/d/1didqsuweIbWb6UhL15IMhs2HrDhvC3EQ/view) |
| RealSense DSN | [Checkpoint](https://drive.google.com/file/d/1Fe6RPN9cwEk6SsvGix9huspZf9Qz2yju/view) |

The registry verifies the extracted files by size and SHA256. Kinect requires your own matching DSN checkpoint. Default table-frame refinement needs `cam0_wrt_table.npy` and, for single-view frames, `camera_poses.npy`. It uses observed depth/fused XYZ and camera calibration; object models and GT instance masks are not refinement inputs.

For Generalizing-Grasp, place the [author fused data](https://drive.google.com/file/d/12YODD0ZUu6XTudU1fZBhVtAmIpMZk8xQ/view?usp=sharing) under `fusion_scenes/scene_XXXX/CAMERA/points.npy`. A numeric `points.npz` with aligned floating-point `xyz`, `normal` and `color` arrays of shape `[N,3]` is also supported and takes precedence when present. XYZ is in table coordinates and metres. The legacy NumPy dictionary is parsed as data without executing pickle callables; unsupported object layouts are rejected. Native training still requires aligned `seg.npy` supervision.

```bash
./panda weights generalizing_grasp --camera realsense
./panda init --example refine-fused -o fused.local.yaml
```

Set your dataset root and checkpoint, then run the generated configuration. [Refinement parameters and observation protocols](REFERENCE.md#contact-score-refinement) explain single-view transfer and fused-scene output coordinates.
