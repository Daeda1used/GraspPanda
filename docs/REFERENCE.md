# Component reference

Detailed parameters, input contracts and method-specific training behavior. Start with [Compose modules](MODULES.md); expand only the component you need.

| Configure | Reference |
|---|---|
| Select compatible parts | [Slots and parameters](MODULES.md#component-selection-and-parameters) · [Scale-Balanced-Grasp](#scale-balanced-grasp-components) · [EconomicGrasp](#economicgrasp-components) |
| Point encoders | [SP2T](#sp2t-sparse-proxy-hierarchy) · [Swin3D](#swin3d-sparse-window-hierarchy) · [PointRWKV](#pointrwkv-released-code-hierarchy) · [PointHR](#pointhr-multi-resolution-point-features) · [PointCNN++](#pointcnn-native-point-convolution) · [Flash3D](#flash3d-native-hierarchy) · [OA-CNNs](#oa-cnns-adaptive-sparse-hierarchy) · [KPConvX](#kpconvx-kernel-point-hierarchy) · [PointVector](#pointvector-encoder) · [PointMetaBase](#pointmetabase-encoder) · [PointMamba](#pointmamba-encoder) · [PCM](#point-cloud-mamba-hierarchy) · [OctFormer](#octformer-hierarchy) · [PTv2](#point-transformer-v2) · [LitePT](#litept-encoder) · [PTv3](#point-transformer-encoder) |
| Local grouping and interaction | [Cylindrical ResLFE](#residual-local-aggregation-in-cylinders) · [Kernel point cylinders](#kernel-point-cylinder-aggregation) · [Seed interaction](#grouped-seed-interaction) · [FineGrasp](#finegrasp-training-and-composition) |
| Quality prediction | [Residual score heads, BCE, Varifocal and MAL](#quality-score-heads) |
| Sampling | [Network seeds and hierarchy stages](#network-sampling-policies) · [Observation sampling](#training-controls) |
| Pretrained point encoders | [Utonia and Concerto](#pretrained-point-encoders) |
| Dynamic point adapters | [PointTPA](#pointtpa-adaptation) |
| Image encoders | [RGB-D encoders](#rgb-d-image-encoders) · [MambaVision](#mambavision-hybrid-image-hierarchy) · [RALA](#rala-image-hierarchy) · [VMamba](#vmamba-state-space-image-features) · [DINO](#pretrained-dino-image-features) |
| Training | [Losses and augmentation](#training-controls) · [LogitNorm / MbLS / LogitClip](#logit-normalization-margin-penalties-and-clipping) · [Optimization](#optimizers-and-schedules) · [Checkpoints](#checkpoint-policies) |
| Run experiments | [Short training](#short-training) · [Epoch training](#train-a-composed-model-across-epochs) · [HGGD](#hggd-epoch-training) · [GtG2](REFERENCE.md#candidate-graph-experiments) |
| Add a method, component or dataset | [Extension guide](#extending-grasppanda) |
| Temporal RGB | [SPGrasp prompts, Hiera and memory](#prompted-planar-sequences) |

<details>
<summary>Scale-Balanced-Grasp encoders, MSCQ branches and training</summary>

## Scale-Balanced-Grasp components

Generate `./panda init --example compose-sbg` for a composed short training run. Select **Scale-Balanced-Grasp → Compose modules** in the browser to configure the same slots. Its [native architecture](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) uses a point transformer encoder, four cylindrical branches, a learned scale fusion and a gated seed-feature connection. The [CoRL paper](https://proceedings.mlr.press/v205/ma23a.html) describes the complete method.

`backbone` accepts the [dense point encoders](MODULES.md#component-selection-and-parameters), their layer controls and [network sampling](#network-sampling-policies). Replacements return 256-channel features and original-input seed indices. Their outputs feed the native approach predictor and MSCQ stage.

Generate the editable configuration with `./panda init --example compose-sbg`.

`branches` contains exactly four policies in native branch order (base radii 0.02, 0.04, 0.06 and 0.08 metres). Each is a name or `{type, ...}` mapping. Available types are `upstream`, `multiscale`, `cylinder`, `reslfe_cylinder` and `kpconvx_cylinder`. Every branch accepts `radius_scale` from 0.1 to 4, multiplying its own native radius. Upstream branches accept `nsample` from 4 to 128 (default 64); replacements accept the corresponding Baseline crop parameters. Branches also accept the registered `seed_interaction` settings. An omitted branch list preserves all four native branches.

Each branch retains the four depth bins and 256 output channels. Scale fusion, the seed gate, operation head and tolerance head stay native. `reuse_unchanged` initializes only replaced parameter groups; changing only an upstream branch's geometry retains its weights. Save the configuration alongside the checkpoint because geometry and sampling policies are not learnable state.

Training accepts loss coefficients and formulations for `graspable`, `view`, `score`, `angle`, `width` and `tolerance`. `graspable` is the author's robust binary target, rather than raw foreground objectness. View and grasp terms retain the scale-distribution prior and weighted denominators. The score term retains its native mask shared across depths; width and tolerance retain their physical normalization. Equivalent CE/MSE/Huber settings preserve the native objective. [Loss parameters](#training-controls), point augmentation, Adam/AdamW/SGD/Lion and update-based schedules are available. Epoch training retains the native optimizer and schedule by default; resume retains the original final-epoch horizon.

The default preset retains the ordinary point loader and native seed sampling. The following policies are independent of encoder and MSCQ choices.

### Object-balanced inference sampling

```yaml
modules:
  sampling:
    type: object_balanced
    seed_count: 1024
    empty_policy: scene_fps
```

Prepare the independent segmentation network with `./panda component-weights scale_balanced_dsn`, or expand **Object-balanced sampling inputs** in the browser. Generate `./panda init --example infer-sbg-obs` for the complete experiment. The released DSN checkpoint uses RealSense; Kinect requires a matching checkpoint in `modules.sampling.segmentation_checkpoint`. Custom checkpoints must match the native DSN architecture and load strictly.

The DSN predicts foreground and object centers from the same observed XYZ. Gaussian mean shift forms instances, then native FPS allocates seeds equally across predicted objects; the final object receives the integer remainder. No dataset instance labels enter this selection. Interpolation supplies 256-channel features at the selected original-input indices. All learnable grasp-model keys remain unchanged, so the matching original grasp checkpoint loads strictly. OBS is available for inference/evaluation; training rejects this inference-only policy.

| Parameter | Default | Meaning |
|---|---|---|
| `seed_count` | `1024` | Total grasp seeds, 32–4096; must cover every predicted object |
| `iterations` | `10` | Mean-shift updates, 1–100 |
| `epsilon` / `sigma` | `0.05` / `0.02` | Cluster merge radius / Gaussian bandwidth, in metres |
| `cluster_seeds` | `50` | Mean-shift initialization seeds, 1–200 |
| `subsample_factor` | `5` | Every Nth foreground center used for mean-shift fitting |
| `min_cluster_points` | `10` | Minimum retained object size in input points |
| `foreground_threshold` | `0.5` | Foreground softmax probability threshold |
| `empty_policy` | `scene_fps` | Use scene-wide FPS if no object survives; `error` stops instead |
| `segmentation_checkpoint` | registered weights | Optional path to a camera-matched DSN checkpoint |

The adapter handles all-foreground inputs without assuming a background label exists. Tiny or coincident foreground sets cap initialization seeds at the available distinct centers, avoiding uninitialized seeds in the original routine. Empty-scene fallback and per-object allocations are recorded with predictions. Evaluation verifies the segmentation checkpoint hash as well as the grasp configuration.

### Noisy-clean mixed training

```yaml
trainer:
  noisy_clean: true
  clean_probability: 0.25
label_root: outputs/prepared/sbg-clean
```

Generate `./panda init --example train-sbg-ncm`, prepare the cache using [Data & weights](DOWNLOADS.md#scale-balanced-grasp-clean-scenes), then select short or epoch training. `clean_probability` ranges from 0 to 1 and selects the CAD-derived source independently for each observed instance, including background, following the author's NcM routine. Each source is sampled first; selected object segments are concatenated and resampled to `num_points`. Missing clean instances contribute no rows, matching the native mixing rule; an entirely empty mixture stops with an input error. Probability 0 uses observed segments, while 1 uses the clean segments for observed instances.

The native NcM loader retains collision masks, visible-grasp filtering and grasp targets for the mixed observations. The extra noisy/clean arrays are diagnostic loader outputs; the author grasp objective does not add a consistency term between them. This remains a single-view inference protocol: CAD geometry and object annotations are used only to prepare training inputs.

Epoch training uses the native NcM flips and rotations by default. `augmentation.mode: native` applies the same transform to mixed, noisy and clean clouds in short training; `mode: none` disables augmentation. Custom rigid transforms update all three clouds and object poses together, and observation sampling preserves mixed-point objectness/instance correspondence. `aug_trans` records inverse rotation; translations remain in transformed poses. Resume verifies the clean-cache inventory, mixing settings and original final-epoch horizon. Prepared caches are checked against their source frame and CAD files and are never downloaded with the toolbox.

</details>

<details>
<summary>Flash3D: native hash, bucket attention and multilevel encoder/decoder</summary>

## Flash3D native hierarchy

`flash3d` uses the [original Flash3D implementation](https://github.com/cruise-automation/Flash3D) ([paper](https://arxiv.org/pdf/2412.16481)): native PSH hashing, bucket attention with backward kernels, recursive pooling/unpooling and Transformer Engine layers. It is available for Baseline, the PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp and FineGrasp.

Generate `./panda init --example compose-flash3d` to initialize a Graspness composition, or select `flash3d` under **Compose modules → Point encoder**. Train the replacement before using its checkpoint. Original grasp checkpoints initialize unchanged heads; they do not contain pretrained Flash3D grasp features.

Generate the editable configuration with `./panda init --example compose-flash3d`.

| Control | Behavior |
|---|---|
| `channels`, `enc_depths`, `dec_depths` | Matching lists for 1–5 hierarchy levels, in fine-to-coarse order; 1–12 blocks per level. |
| `enc_heads`, `dec_heads` | Scalar or one value per block; each head has 16, 32, 64 or 128 channels. |
| `enc_mlp_ratio`, `dec_mlp_ratio` | Scalar or per-block expansion; hidden widths must be integral multiples of eight. |
| `enc_qkv_bias`, `dec_qkv_bias` | Scalar or per-block query/key/value bias switches. |
| `enc_scope_plan`, `dec_scope_plan` | Cycles of `contiguous`, `shifted` and `strided` bucket windows. |
| `enc_scope_size`, `dec_scope_size` | Scalar or per-block window size, in buckets. |
| `enc_scope_shift`, `dec_scope_shift` | Shifted windows only; positive shift smaller than that window. |
| `enc_scope_stride`, `dec_scope_stride` | Strided windows only; positive stride in buckets. |
| `enc_mlp_activation`, `dec_mlp_activation` | Cycles of `swiglu`, `gelu` or `relu`. |
| `enc_residual_dropout`, `dec_residual_dropout` | Scalar or per-block dropout, from 0 to 0.5. |
| `bucket_size`, `hash_type` | Buckets of 128, 256 or 512 points; native hash policy 1–4. |
| `pooling` | `mean`, `sum`, `min` or `max`; one name for all transitions, or one per adjacent level pair. |
| `normalization`, `norm_eps` | `layer` or `rms` normalization and its epsilon. |

Per-block lists enumerate each level's blocks in **fine-to-coarse order**, independently for encoder and decoder. Pattern lists repeat through that order. The configuration validator checks heads, widths and window alignment together; changing a hierarchy may require changing its per-block lists. Dense methods also accept [seed sampling](#network-sampling-policies).

Scenes are processed independently. Coordinates are hashed in FP16 and attention/MLPs use BF16; projected features and the caller's original coordinates retain FP32. The adapter preserves input feature channels, including FineGrasp normals, and restores sparse coordinate correspondence or original dense seed indices. Native alignment may require repeating valid input rows; those extra outputs are discarded after restoring correspondence. FP8 calibration/checkpoints are not exposed. GraspPanda checkpoints keep tensor-only precision metadata for standard `weights_only` loading.

The [installer](INSTALL.md#files-created-locally) builds the pinned kernels in a private native directory using the shared Python interpreter. Compatibility patches correct native attention gradients, RNG initialization, min/max pooling gradients and gradient packing. CUDA compatibility changes affect the build; the native hierarchy and losses remain in use. The author repository has newer compiler requirements, so its standalone installation commands do not replace the GraspPanda installer.

</details>

<details>
<summary>Network sampling: output seeds and hierarchy stages</summary>

## Network sampling policies

Baseline, its PointNet2 port and Scale-Balanced-Grasp accept `seed_sampling` on every listed replacement point encoder. It selects original-input grasp seed rows and gathers or reconstructs their features. `pointnext`, `pointvector` and `pointmeta` additionally accept four `stage_sampling` policies, in fine-to-coarse order. These change native downsampling centers while retaining grouping, feature propagation and supervision. The original `upstream` backbone and sparse-method encoders do not expose these controls.

Generate the editable configuration with `./panda init --example sample-network`.

A policy is a name, a `{type, ...}` mapping, or a pair of `train` and `eval` policies. The pair follows the model's training/evaluation mode, including native validation. Both entries are required; nested pairs are rejected. A bare random policy remains random in evaluation; use an explicit eval policy when deterministic sampling is required. Omitting a policy preserves the original sampler. Generate `./panda init --example sample-network`; edit these settings under **Compose modules → Component parameters by slot**. The parameter guide lists accepted fields, and sweeps can vary paths such as `modules.backbone.seed_sampling.eval.keep_ratio`.

| Policy / field | Behavior |
|---|---|
| `upstream` | Call the original sampling function unchanged. |
| `uniform` | Uniform random selection without replacement. |
| `fps` | Farthest-point selection over all input rows using the toolbox CUDA operator. |
| `pointsp_wrs` | Density-weighted random selection without replacement, using the [PointSP (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/48) rule. |
| `pointsp_ffps` | Rank rows by PointSP density weight, retain an eligible subset, then apply masked FPS. [Paper](https://arxiv.org/pdf/2408.12062) · [author source](https://github.com/tangsankou/PointSP/tree/8206043f27e8b849fecde48841ff4b2513e88439). |
| `neighbors`, `density_quantile` | Density policies only: default `20` and `0.5`; ranges 1–128 and 0–1. Neighbor count is capped at each stage's input size. Self/repeated points participate; neighborhoods and the threshold are computed independently for each scene. |
| `keep_ratio` | FFPS only: default `0.95`, range 0.1–1. Keep `max(requested_count, floor(input_count * keep_ratio))` rows so the sampler can return the requested count without repeating row indices. |
| `start` | FPS/FFPS only: `first` chooses the first eligible input row; `random` chooses uniformly among eligible rows using the experiment's Torch RNG. |

The GPU density calculation uses the shared exact KNN operator and direct squared coordinate differences, without a dense pairwise matrix. Density ties at the filtering boundary favor lower input indices. Masked FPS uses one common eligible starting point for each scene, excludes already selected rows, accepts points near the origin and resolves equal distances by lower row index. It respects the active CUDA device and stream. The installer builds `_grasppanda_sampling_cuda` into the shared environment; rerun `./panda install` after upgrading. There is no separate environment or JIT fallback.

These are explicit grasp adaptations of the sampling rules, not the complete PointSP classification protocol. The CUDA implementation uses deterministic selection and checked inputs; it does not reproduce the author's time-seeded kernel behavior. Different row indices can still contain identical XYZ after observation padding. Stage masks restrict center selection; they do not remove those points from native grouping or reconstruct new geometry. No tangent-plane interpolation is performed. Policies add no trainable parameters, but changing them changes predictions. Retain the policy configuration with checkpoints; **Prepare inference from checkpoint** preserves network sampling while clearing training-only observation augmentation.

</details>

<details>
<summary>Parameter reference and configuration example</summary>

| Component | Parameters |
|---|---|
| Baseline `pointnet` | `local_channels`, `global_channels`, `fusion_channels` (layer widths); `activation`: relu/gelu/silu; `normalization`: batch/group/none; `dropout` |
| `pointnext` | `width`, `blocks` (five stage depths), `nsample`, `radius` (metres), `radius_scaling`, `expansion`, `activation`, `reduction`: max/mean/sum, `decoder_layers` |
| `pointvector` | `width`, `blocks` (five stages), `nsample`, `local_nsample`, `radius`, `radius_scaling`, `normalize_dp`, `sa_layers`, `sa_use_res`, `decoder_layers` |
| `pointmeta` | `width`, `blocks`, `nsample`, `radius`, `radius_scaling`, `expansion`, `normalize_dp`, `local_reduction`, `activation`, `use_res`, `sa_layers`, `sa_use_res`, `decoder_layers` |
| `pointmlp` | `embed_dim`, `dim_expansion`, `pre_blocks`, `pos_blocks`, `stage_points`, `k_neighbors`, `decoder_channels`, `decoder_blocks`, `res_expansion`, `activation`, `normalize`: anchor/center |
| `multiscale` | `radius_factors` relative to the method's native cylinder radius |
| `finegrasp` / `native_cylinder` | `nsample`, `radius_factors`; FineGrasp also accepts `radius`. Cross-radius attention: `fusion_layers`, `fusion_heads`, `fusion_ffn_dim`, `fusion_dropout`, `fusion_activation`, `fusion_pre_norm` |
| `cylinder` | `hidden_channels`, `radius_factors`, `nsample`, `pooling`: max/mean/attention, `activation`, `normalization` |
| `reslfe_cylinder` | `width`, `depth`, `local_neighbors`, `nsample`, `radius_factors`, `mlp_ratio`, `drop_path`, `bn_momentum`, `pooling`, `activation`, `normalization` |
| Image `dinov2` / `dinov3` | `variant`: small/base; `pretrained`, `out_indices`, `trainable_blocks`, `drop_path`, `gradient_checkpointing`, `projection_norm` |
| Image `vmamba` | `stage_channels`, `stage_depths`, `state_dim`, `ssm_ratio`, `dt_rank`, `scan`, `ssm_conv`, `ssm_conv_bias`, `ssm_activation`, `ssm_dropout`, `mlp_ratio`, `mlp_activation`, `mlp_dropout`, `drop_path`, `gradient_checkpointing`, `projection_norm` |
| Image `native_resnet` | `variant`: 18/34/50 (string); optional four-element `stage_depths` |
| Image `convnextv2` | `variant`: atto/tiny; optional four-element `stage_channels`, `stage_depths`; `drop_path`, `projection_norm` |
| Image `repvit` | `variant`: m0_9/m1_1; optional four-element `stage_channels` (multiples of 8), `stage_depths`; `projection_norm` |
| Image `mobilenetv4` | `variant`: small/medium (convolutional models); `drop_path`, `projection_norm` |

`pointnext` uses the [official PointNeXt encoder and decoder](https://github.com/guochengqian/PointNeXt) ([NeurIPS 2022 paper](https://arxiv.org/pdf/2206.04670)), followed by a feature projection and original-input seed sampling. Its OpenPoints operator has a separate compiled namespace to coexist with legacy grasp operators. Architecture settings do not automatically provide pretrained weights.

`pointmlp` adapts the [official PointMLP segmentation blocks](https://github.com/ma-xu/pointMLP-pytorch/blob/main/part_segmentation/model/pointMLP.py) ([ICLR 2022 paper](https://openreview.net/pdf?id=3Pbra-_u76D)): geometric affine normalization, residual MLP extraction and native feature propagation. An XYZ embedding replaces the normals-dependent input; the part-category token and segmentation classifier are omitted. Dense features are projected to 256 channels and sampled at original-input seed indices. This is a grasp backbone adaptation, without pretrained grasp weights. Its native propagation fusion retains ReLU; `activation` controls embedding and residual blocks.

PointMLP stage parameters are four-element lists. `stage_points` defaults to `[1024, 256, 64, 16]` and must decrease; each `k_neighbors` value must fit that stage's input. Fixed stage sizes bound neighborhood memory as the input cloud grows. `dim_expansion` multiplies the preceding channel width; `decoder_channels` specifies output widths from coarse to fine. Train any changed architecture before evaluating it.

`finegrasp` uses the author's [CylinderGroup and GroupTransformerFusion](https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/robo_orchard_lab/models/finegrasp) from [FineGrasp (2025)](https://arxiv.org/pdf/2507.05978). This replacement retains Graspness seed features and decoder; it does not replace the complete method with FineGrasp. Start with `radius_factors: [0.25, 0.5, 0.75, 1.0]` and `nsample: 16`, then train the new grouping layers.

`pointnet` is a toolbox shared-MLP/global-pooling adapter. `cylinder` exposes native oriented queries with configurable feature extraction and pooling; it is not a complete reproduction of a named attention or scale-balanced method. `sparse_unet18` uses MinkUNet18D. These choices are available only in their registered semantic slots.

```yaml
modules:
  backbone:
    type: pointnext
    width: 32
    blocks: [1, 2, 2, 2, 1]
    radius: 0.05
    nsample: 32
  crop:
    type: cylinder
    hidden_channels: [64, 128]
    radius_factors: [0.5, 1.0, 1.5]
    pooling: attention
checkpoint_policy: reuse_unchanged
```

In the UI, select the component names under **Compose modules**, then enter parameters keyed by slot in **Component parameters by slot**. Do not repeat `type` in this parameter editor; the selector supplies it. Full YAML/JSON configurations include `type` in each slot mapping.

</details>

<details>
<summary>EconomicGrasp encoder, grouping and prediction head</summary>

## EconomicGrasp components

EconomicGrasp retains its native sparse quantization, graspable-point selection, economic labels, interactive grasp head and decoder. Its [author implementation](https://github.com/iSEE-Laboratory/EconomicGrasp) is described in the [ECCV 2024 paper](https://arxiv.org/pdf/2407.08366). Generate an editable composition with `./panda init --example compose-economicgrasp`.

The backbone receives constant input features on the quantized camera lattice and returns 512-channel features in the same sparse row order. `pointnet` uses the toolbox's sparse PointNet-style encoder; `sonata_ptv3` uses the [PTv3 adapter](#point-transformer-encoder), and `point_transformer_v2` uses [PTv2 grouped vector attention](#point-transformer-v2). Each reconstructs camera positions from lattice coordinates and voxel size and preserves the map used by `quantize2original`.

`native_tdunet` configures the author's eight-stage encoder/decoder without changing its down/up-sampling strides or skip connections:

| Parameter | Default | Meaning |
|---|---|---|
| `channels` | `[32,64,128,256,192,192,192,192]` | Eight stage plane widths, each 8–512; Bottleneck expands residual output widths by four |
| `blocks` | `[1,1,1,1,1,1,1,1]` | Residual blocks per stage, each 1–8 |
| `dilations` | `[1,1,1,1,1,1,1,1]` | Residual convolution dilation per stage in lattice cells, each 1–8 |
| `stem_channels` | `32` | Stem width, 8–128; the native 5-cell kernel is retained |
| `block` | `basic` | Native MinkowskiEngine `basic` or `bottleneck` residual blocks |
| `bn_momentum` | `0.1` | Momentum of all sparse batch-normalization layers, 0.001–1 |

`native_cylinder` retains the native two MLP stacks and cylinder query while exposing the following controls:

| Parameter | Default | Meaning |
|---|---|---|
| `nsample` | `16` | Samples per oriented cylinder, 4–128 |
| `radius` | `0.05` | Cylinder radius in metres, 0.005–0.5 |
| `hmin`, `hmax` | `-0.02`, `0.04` | Axial bounds in metres; -0.2–0 and 0–0.2, with `hmin < hmax` |
| `attention_heads` | `1` | Within-cylinder attention heads: 1, 7 or 37, dividing the native 259-dimensional feature-plus-XYZ vector |
| `attention_dropout` | `0.05` | Attention-probability dropout, 0–0.8 |
| `local_attention` | `true` | Set false to bypass the within-cylinder attention and its LayerNorm; omit attention parameters in this ablation |

The `cylinder` and `reslfe_cylinder` choices replace the native within-cylinder processing with the toolbox's configurable aggregation or [ResLFE](#residual-local-aggregation-in-cylinders). All grouping choices can add [seed interaction](#grouped-seed-interaction) afterward. Its `interaction_heads` divides 256 and is independent of `native_cylinder.attention_heads`.

Default `native_tdunet` and `native_cylinder` retain native state names and shapes, so author weights can load with `strict`. Geometry, dropout, dilation and head count are configuration values rather than learned tensors: matching weight shapes alone does not establish matching behavior. Architecture changes use `reuse_unchanged`, which initializes each selected replacement slot and retains the rest of the checkpoint. Train the replacement, then use `strict` with the saved composition for inference.

EconomicGrasp training also supports [loss formulations and coefficients](#loss-formulations), [aligned point augmentation](#point-augmentation), Adam/AdamW/SGD/Lion and update-based schedules. Score remains a six-class objective; angle/depth retain their native invalid classes and validity masks. Width targets remain scaled by ten. Augmentation transforms object poses with observations, updates per-point labels through one sampling map and regenerates `coordinates_for_voxel`; native flips retain float32 observations. Use `action: train` for native epoch training and checkpoint resume; see [EconomicGrasp epoch training](#economicgrasp-epoch-training).

### Initialize replacement seed features

New encoders may need seed initialization before full grasp training. See [Seed-prediction warmup](#seed-prediction-warmup) for the shared control and stage schedules. Pretrained point encoders also accept `projection_norm: layer`; select a learning rate appropriate for their new projection and unfrozen blocks.

## EconomicGrasp interactive head

Select `head: native_interactive` to configure the [author's interactive grasp head](https://github.com/iSEE-Laboratory/EconomicGrasp/blob/main/models/modules_economicgrasp.py). It consumes `[B,256,N]` grouped features. For each seed independently, four task tokens interact in **angle, depth, width, score** order. This is separate from attention over cylinder neighbors or over spatial seeds.

| Parameter | Default | Meaning |
|---|---|---|
| `feature_channels` | `64` | Shared task-token width, 16–512; every attention head count must divide it. |
| `branch_depths` | `[1,1,1,1]` | Convolution count per angle/depth/width/score feature branch, each 1–4. The first maps 256 input channels to the task width; subsequent layers retain it. |
| `activation` | `relu` | `relu`, `gelu` or `silu` between additional branch layers. Only accepted when a branch has depth greater than one. |
| `interaction` | `attention` | Native attention with residual addition and LayerNorm, or `none` for independent task branches without that normalization. |
| `attention_layers` | `1` | Number of native attention blocks, 1–8. |
| `attention_heads` | `1` | Head count, 1–64; scalar shared across blocks or a list of exactly one value per block. |
| `attention_dropout` | `0.05` | Attention-probability dropout, 0–0.8; scalar or one value per block. |

Generate `./panda init --example compose-economic-head`. Its three interaction layers use different head counts and dropout settings; branch depths are independently configured. In the browser, **Compose modules → Grasp prediction head** appears only for methods with a registered head slot. Choose `native_interactive` and enter fields under `head` in **Component parameters by slot**. The parameter reference and configuration preview include the new slot. Sweep `modules.head.feature_channels`, layer settings, or complete head mappings; an incompatible combination is rejected before execution.

The default configuration preserves native parameter names, initialization and forward behavior and can load author weights with `strict`. Width, branch depth, interaction removal or stacked blocks change parameter structure: initialize the replaced head using `reuse_unchanged`, train it and reuse the saved configuration for strict inference or epoch resume. Changing only head counts/dropout can retain compatible weight shapes while changing behavior; those settings are still part of the resume contract. Attention fields are rejected with `interaction: none`.

Angle and depth outputs retain their additional invalid classes, score retains six classes, and width remains one regression output with the native target scaling. Loss masks, selected-view supervision, seed ordering and decoding remain unchanged. Width specialization uses the pinned native forward; additional branch layers and stacked attention are toolbox architecture options, not separate pretrained methods. This head is registered for EconomicGrasp only: other detectors need their own supervision and decoder contracts.

</details>

<details>
<summary>Grouped seed interaction</summary>

## Grouped seed interaction

Baseline, its PointNet2 port, Graspness and EconomicGrasp support `seed_interaction: gaussian` inside `modules.crop`. It adds distance-biased attention **after the selected grouping**. You can retain `type: upstream` or combine it with a registered grouping replacement. Native FineGrasp, HGGD/RNG and GtG2 have different feature contracts and do not expose this option.

```yaml
modules:
  crop:
    type: upstream
    seed_interaction: gaussian
    interaction_heads: 4
    interaction_sigma: 0.05
    interaction_layers: 1
    interaction_dropout: 0.1
checkpoint_policy: reuse_unchanged
```

Generate a complete EconomicGrasp example with `./panda init --example compose-seed-interaction`. In the browser, expand **Compose modules**, retain the desired grouping and enter these fields under `crop` in **Component parameters by slot**. The parameter reference lists the accepted values. Sweep paths such as `modules.crop.interaction_sigma` and `modules.crop.interaction_heads` to compare settings.

| Parameter | Default | Meaning |
|---|---|---|
| `seed_interaction` | `none` | `gaussian` enables the interaction; omit interaction parameters when disabled |
| `interaction_heads` | `4` | Attention heads; 1–32 and must divide 256 |
| `interaction_sigma` | `0.05` | Gaussian distance scale in camera-frame metres; 0.001–1 |
| `interaction_layers` | `1` | Independently parameterized interaction blocks; 1–4 |
| `interaction_dropout` | `0.1` | Attention-probability dropout during training; 0–0.8 |

The implementation loads the pinned `GraspGNN` attention classes from [GCF-GraphGrasp](https://github.com/qzsrh/GCF-GraphGrasp/blob/8ab374104f1559005f292f163bcf3b17e8211575/models/modules_economicgrasp.py). It retains the native Q/K/V/output projections, two residual LayerNorm operations and learned distance-bias scale. The bias is `-distance_squared / (2 * sigma_squared) * bias_scale`; each block initializes its scale to one and leaves its sign unconstrained, as in the source. This is dense attention over grasp seeds, not a k-nearest graph or a new cylinder query.

The adapter preserves seed order and 256-channel output features. It processes scenes independently and, for Baseline's `[B,256,N,4]` output, processes each depth bin independently. Inputs and computation use float32. Attention memory grows quadratically with the number of seeds and linearly with batch, depth bins, heads and layers; changing the scene point count does not necessarily change the method's seed count.

When retaining the original grouping, `reuse_unchanged` loads its existing weights and initializes only the added interaction layers. Train the composition before inference, then load its checkpoint with `strict` and the same module settings. If you also replace the grouping or encoder, those replacements follow their usual initialization policy. Baseline, Graspness and EconomicGrasp support this composition in epoch training and resume; the PointNet2 port uses its registered short-training operation.

This component uses each selected method's own data, supervision, seed selection and decoder. It does not reproduce the GCF fork's SAM/patch-feature preprocessing or its changed angle/depth decoding. GCF's README cites EconomicGrasp; a separate GCF publication and pretrained checkpoint have not been verified. The full-method entry remains in [Methods & papers](METHODS.md).

</details>

<details>
<summary>Point encoders: OctFormer, PointVector, PointMetaBase and Mamba</summary>

## OctFormer hierarchy

`octformer` adapts the [author implementation](https://github.com/octree-nn/octformer) of [OctFormer (SIGGRAPH / TOG 2023 PDF)](https://arxiv.org/pdf/2305.03045) to Baseline and its PointNet2 port. It retains the native octree convolution stem, alternating regular/dilated window attention, positional convolution, MLP blocks and multi-scale segmentation decoder. Camera XYZ replaces the ScanNet input features; the decoder produces 256-channel features sampled at the original grasp seed indices. Start with [`compose-octformer`](USAGE.md#configuration-examples) (`./panda init --example compose-octformer`).

| Configure | Parameters and defaults |
|---|---|
| Octree geometry | `depth: 11`, `full_depth: 2`, `nempty: true` |
| Encoder stages | `channels: [96,192,384,384]`, `num_blocks: [2,2,18,2]`, `num_heads: [6,12,24,24]` |
| Window grouping | `patch_size: 32`, `dilation: 4`; regular and dilated attention alternate within each stage |
| Stage behavior | `mlp_ratio: 4`, `qkv_bias: true`, `use_rpe: true`, `attn_drop: 0`, `proj_drop: 0`, `use_dwconv: true`, `use_checkpoint: true` |
| Residual regularization | `drop_path: 0.5`, increasing across the encoder blocks |
| Stem and decoder | `stem_down: 2`, `head_up: 2`, `fpn_channel: 168`, `head_drop: [0,0]` |

The **stage behavior** fields accept a scalar for all stages or a list with one entry per stage. `use_dwconv: false` selects the author's eight-group positional convolution. `use_rpe: false` removes relative-position tables; the released author model enables them, although the original paper experiments did not. Native BatchNorm and GELU are retained. Activation checkpointing uses temporary BatchNorm buffers during recomputation so statistics update only once.

Stage lists must have equal lengths (one to six stages); widths must be divisible by their head counts, and grouped positional convolution requires widths divisible by eight. Zero `num_blocks` retains the convolutional hierarchy at that stage. The coarsest stage depth, `depth - stem_down - stage_count + 1`, must be at least `full_depth`, which is at least two. `head_up` cannot exceed `stem_down`. Larger depth creates finer voxels; it does not add encoder stages. More input points, channels or larger attention windows increase memory use. Degenerate training inputs with fewer than two nodes at a used resolution require a larger batch or a non-degenerate point cloud.

Each scene is normalized isotropically into the octree cube while its feature channels retain camera XYZ in meters. Querying the decoder preserves original input rows, including duplicate voxel assignments. Windows are padded independently per scene: changing a preceding scene's node count cannot shift another scene's window origin. Native batch normalization still shares training statistics across the batch. This is a grasp adaptation with no pretrained OctFormer grasp checkpoint; initialize compatible unchanged grasp layers, then train the new backbone. Use `train` for epoch training and `strict` checkpoint loading for resume/inference.

## PointVector encoder

`pointvector` uses the native segmentation encoder and decoder from [PointVector (CVPR 2023 PDF)](https://openaccess.thecvf.com/content/CVPR2023/papers/Deng_PointVector_A_Vector_Representation_in_Point_Cloud_Analysis_CVPR_2023_paper.pdf), implemented in the [pinned OpenPoints source](https://github.com/guochengqian/openpoints/blob/db31e0d94ede35bc672ef06d1b6e4073a9c269d5/models/backbone/pointvector.py). It reuses the shared OpenPoints CUDA operators. No additional environment or component download is required after installation.

The adapter supplies XYZ features, decodes back to every original input point, projects features to 256 channels and samples 1,024 grasp seeds at original-input indices. Camera coordinates and label indexing remain unchanged. It omits the segmentation classifier and does not load segmentation weights. Use `reuse_unchanged` with the grasp checkpoint, train the replacement and reload its checkpoint with `strict`.

| Setting | Default / meaning |
|---|---|
| `width`, `blocks` | `32`, `[1,3,5,3,3]`; five stage depths, each including its initial abstraction/stem block |
| `nsample` | `32`; neighbors for downsampling abstraction |
| `local_nsample` | `8`; neighbors for the native vector aggregation blocks |
| `radius`, `radius_scaling` | `0.05` metres, `2.0`; initial neighborhood radius and native stage scaling |
| `normalize_dp` | `true`; divide relative query offsets by the neighborhood radius; absolute points stay in metres |
| `sa_layers`, `sa_use_res` | `1`, `false`; abstraction MLP depth and residual connection |
| `decoder_layers` | `2`; MLP depth of native feature propagation |

Vector blocks retain the author's angle-based scalar-to-vector transforms, channel-grouped projection, ReLU/batch normalization and sum reduction. Increasing `local_nsample` changes both support and the scale of that sum. A stage depth of one omits its additional vector blocks; choosing one for every stage is an explicit no-vector ablation. Start with [`compose-pointvector`](USAGE.md#configuration-examples) (`./panda init --example compose-pointvector`). Baseline also supports this encoder in epoch training; the PointNet2 port uses its registered short-training and inference operations.

## PointMetaBase encoder

`pointmeta` uses the native PointMetaBase encoder and its PointNext decoder from [Meta Architecture for Point Cloud Analysis (CVPR 2023 PDF)](https://arxiv.org/pdf/2211.14462) ([pinned implementation](https://github.com/linhaojia13/PointMetaBase/blob/c364a671ee255453c0a03b2474af555664604db9/openpoints/models/backbone/pointmetabase.py)). It is available for Baseline and its PointNet2 port. Run `./panda install` after upgrading to fetch the pinned source; the installer verifies that its CUDA sources are identical to the shared OpenPoints operators before reusing them. The author's Python layers load in a separate namespace from PointNeXt and PointVector.

The encoder updates point features before grouping and adds explicit positional features shared by the local blocks in each stage. The adapter supplies XYZ input features, decodes back to every original input point, projects to 256 channels and gathers grasp seeds at original-input FPS indices. Coordinates and labels retain their camera-frame correspondence. It does not use the segmentation classifier, height features or pretrained segmentation weights.

| Setting | Default / meaning |
|---|---|
| `width`, `blocks` | `32`, `[1,3,5,3,3]`; five stage depths, including each stage's abstraction/stem block. `blocks[0]` must be one because the native stem has no local positional embedding. |
| `nsample` | `32`; neighbors per query. Local features and shared positional features use matching queries. |
| `radius`, `radius_scaling` | `0.05` metres, `2.0`; starting radius and native stage scaling. |
| `expansion` | `1`; hidden expansion in local point-update MLPs. |
| `normalize_dp` | `true`; radius-normalize relative position offsets. Absolute XYZ remains in metres. |
| `local_reduction` | `max`; also `mean` or `sum`, for local feature aggregation. Downsampling abstraction retains native max pooling. |
| `activation`, `use_res` | `relu`, `true`; encoder activation (also `gelu` or `silu`) and local residual connections. |
| `sa_layers`, `sa_use_res` | `1`, `false`; abstraction feature-MLP depth and residual connection. Its positional branch retains the native depth rule. |
| `decoder_layers` | `2`; feature-propagation MLP depth. The native decoder retains ReLU and BatchNorm. |

A stage depth of one omits its extra local blocks, so local reduction, expansion and residual settings have no effect in that stage. Selecting one in all stages gives an abstraction-only ablation. Use `reuse_unchanged` with the original grasp checkpoint, train the replacement, then use `strict` for the resulting checkpoint. Baseline supports epoch training and resume; the PointNet2 port supports its registered short-training and inference operations. Start with [`compose-pointmeta`](USAGE.md#configuration-examples) (`./panda init --example compose-pointmeta`), which also replaces cylinder processing with ResLFE.

## PointMamba encoder

`pointmamba` adapts [PointMamba (NeurIPS 2024 PDF)](https://arxiv.org/pdf/2402.10739) from the [author implementation](https://github.com/LMD0311/PointMamba). It retains the native local point encoder, two Hilbert sequences, order-specific scales and Mamba residual blocks. The pinned main branch provides classification and pretraining models; GraspPanda adds a grasp feature decoder. It is available for Baseline and its PointNet2 port.

FPS centers and Euclidean nearest neighbors form centered local patches. Shared PointNet2/PyTorch3D operators replace the author's separate grouping packages; equal-distance neighbor ties can choose different points. Tokens follow `hilbert` and `hilbert-trans` orders. Each output sequence is restored to the original center order before mean or concatenation fusion. Three-neighbor interpolation and a learned projection produce 256-channel features at 1,024 original-input grasp seeds. Camera XYZ stays in metres; the dataset's seed/label correspondence is preserved.

| Setting | Default / meaning |
|---|---|
| `dim`, `depth` | `384`, `12`; token width (multiple of eight) and native residual block count |
| `num_group`, `group_size` | `128`, `32`; sampled patch centers and nearest neighbors per patch |
| `grid_size` | `0.02` metres; spatial quantization for ordering, without averaging or removing input points |
| `d_state`, `d_conv`, `expand` | `16`, `4`, `2`; state size, causal convolution width (2–4) and inner-width expansion |
| `dt_rank` | Omit for the native automatic rank; otherwise a positive integer |
| `rms_norm` | `false`; choose native RMSNorm instead of LayerNorm |
| `drop_path`, `dropout` | `0.5`, `0.0`; linearly increasing block drop probability and dropout after each block |
| `order_fusion` | `mean`; alternatively `concat`, doubling the projection input width |
| `gradient_checkpointing` | `false`; recompute token blocks during training while preserving dropout RNG |

The adapter keeps the native block constructor initialization, including the learned timestep bias initialization and depth-scaled output projections. It does not apply the classification trainer's outer scratch initializer, load classification weights or provide pretrained grasp weights. Start with [`compose-pointmamba`](USAGE.md#configuration-examples) (`./panda init --example compose-pointmamba`), use `reuse_unchanged` to retain the baseline's other modules, train the replacement, and use `strict` to reload the resulting grasp checkpoint. Width, depth, state size, grouping and fusion are saved with the experiment and can be swept through dotted configuration paths.

Run `./panda install` to fetch the pinned sources and build both native CUDA extensions in the shared runtime. Fast Mamba convolution and selective scan use isolated extension names; no global `mamba_ssm` package is installed. Coordinates must be CUDA float32; extremely small grids that exceed the native 16-bit spatial encoding are rejected. Single-cell clouds retain a valid ordering. This encoder has one token resolution; it does not provide a hierarchical point decoder or make multi-view inputs interchangeable with single-view observations.

## Point Cloud Mamba hierarchy

`pointcloud_mamba` adapts [Point Cloud Mamba (2024 preprint, PDF)](https://arxiv.org/pdf/2403.00762) from [SkyworkAI's implementation](https://github.com/SkyworkAI/PointCloudMamba). It is a separate component from LMD's `pointmamba`. Baseline and its PointNet2 port can use its four-stage point hierarchy, bidirectional Mamba blocks, order prompts, positional projections, native feature propagation and global context.

The grasp adapter supplies camera XYZ in metres, replaces the S3DIS input channels with XYZ, and projects dense decoder features to 256 channels at 1,024 original-input FPS seeds. It retains native `bimamba_type: v2`, local grouping and window processing. Classification/category tokens, RGB features and pretrained grasp weights are not supplied.

| Setting | Default / meaning |
|---|---|
| `embed_dim`, `dim_expansion` | `96`, `[1,2,2,2]`; stem width and four successive multipliers |
| `pre_blocks`, `pos_blocks` | `[1,1,1,1]`, `[0,0,0,0]`; local extraction and post-extraction residual depths |
| `mamba_blocks` | `[1,2,2,4]`; scan-block counts, one entry per encoder stage; `0` keeps that stage local |
| `k_neighbors`, `k_strides`, `reducers` | `[12,12,12,12]`, `[1,1,1,1]`, `[4,4,2,2]`; KNN query, neighbor subsampling stride and point-count reduction |
| `orders` | One order per encoder Mamba block. Omit to cycle `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx`, `hilbert`, `z`, `z-trans`; `hilbert-trans` is also accepted |
| `use_order_prompt`, `prompt_num_per_order` | `true`, `6`; prepend and append learned order prompts |
| `mamba_pos`, `pos_type`, `pos_proj_type` | `true`, `share`, `linear`; alternatively `per_layer` projections or `mlp` |
| `rms_norm`, `fused_add_norm`, `residual_in_fp32`, `block_residual` | All `true`; native encoder normalization and residual controls |
| `drop_path` | `0.1`; native encoder stochastic-depth schedule |
| `use_windows`, `window_sizes`, `grid_size` | `true`, `[1024,512,256,128]`, `0.04` metres; stage windows and spatial ordering resolution |
| `use_xyz`, `normalize`, `res_expansion`, `activation` | `true`, `anchor`, `1.0`, `relu`; local extraction also accepts `center`, `gelu` or `silu` |
| `decoder_channels`, `decoder_blocks` | Widths derived from the encoder, `[384,192,96,96]` by default; `[1,1,1,1]` propagation depths. Entries run from coarse to fine |
| `decoder_mamba_blocks`, `decoder_orders` | `[0,0,0,0]`, `[]`; optional decoder scans. Omitted orders cycle as above; a final-resolution scan must end with the string `"null"` |
| `decoder_rms_norm`, `decoder_fused_add_norm`, `decoder_residual_in_fp32` | `true`, `false`, `false`; normalization controls for enabled decoder scans |
| `gmp_dim` | `64`; native pooled global-context width, concatenated with dense features |
| `d_state`, `d_conv`, `expand` | `16`, `4`, `2`; native state size (1–256), causal kernel width (2–4) and inner-width multiplier (1–4) |
| `dt_rank` | Omit for native automatic rank; otherwise 1–128 |
| `ssm_bias`, `ssm_conv_bias` | `false`, `true`; native input/output projection bias and depthwise convolution bias |
| `decoder_d_state`, `decoder_d_conv`, `decoder_expand`, `decoder_dt_rank`, `decoder_ssm_bias`, `decoder_ssm_conv_bias` | Corresponding settings for enabled decoder scans; defaults match the encoder's native defaults independently |

SSM settings accept a scalar for all active blocks or a list with one value per active block, ordered across stages just like `orders`. For `mamba_blocks: [0,1,0,2]`, `d_state: [16,32,64]` configures the single stage-2 block followed by both stage-4 blocks. Encoder and decoder settings are independent. The adapter forwards them to the native Mamba constructor and retains both scan directions and native initialization; it does not switch to a unidirectional fallback.

A zero encoder scan count retains the stage's native grouping and local extraction, skips its scan windows and positional/prompt additions, and removes projections that cannot reach an active scan. Residuals pass through intermediate local stages only when a later scan consumes them. With `mamba_blocks: [0,0,0,0]`, omitted `use_order_prompt`, `mamba_pos` and `use_windows` resolve to `false`; explicit scan-only overrides are rejected. Local hierarchy, dense decoding and global context remain trainable. `0` in `pre_blocks` selects native pure max pooling, which requires `use_xyz: false` and `dim_expansion: 2` at that stage to match the concatenated neighbor/center width. `0` in `decoder_blocks` removes residual extraction while retaining native feature fusion and interpolation. These settings support local-only and mixed local/scan ablations without adding substitute layers.

Stage widths must be multiples of eight and at most 2,048. Each neighbor count must divide evenly by its stride and fit the incoming stage. Every coarse level must retain at least three points for native interpolation. Windows preserve the author's sorted-FPS truncation to a complete number of windows and within-window coordinate normalization; dense decoding returns to the full original input. Configuration validation accounts for truncation only in stages with active scans. Finite coordinates and a maximum 16-bit voxel extent are required.

GraspPanda applies three correspondence fixes to the pinned source: CTS uses per-point coordinates with a zero-based serpentine endpoint convention and separate batch ranges; prompt IDs follow first occurrence instead of unordered set iteration; each decoder scan stage restores feature rows before the next spatial interpolation. These fixes affect serialized ordering and optional decoder scans, so the adapter does not claim numerical equivalence to the unmodified upstream model. The scoped loader also removes an unsupported, redundant keyword from the native standalone RMSNorm wrapper, allowing non-fused RMSNorm blocks. Native grouping, normalization mathematics, Mamba layers, global context and propagation remain in use.

Use [`compose-pcm`](USAGE.md#configuration-examples) (`./panda init --example compose-pcm`), or select `pointcloud_mamba` and enter parameters in the UI. `./panda install` downloads the pinned source and builds its two CUDA extensions with isolated names in the shared runtime; its older causal-convolution ABI coexists with `pointmamba`. Source terms are described in [Third-party notices](THIRD_PARTY.md).

Training requires **`batch_size >= 2`** because native global-context BatchNorm operates on one pooled feature per sample. Short training uses a fixed batch of consecutive labelled frames within the selected scene; epoch training drops an incomplete final batch. Inference accepts one frame. Keep the exact component configuration when reloading weights; use `reuse_unchanged` for initial component replacement, train it, then use `strict` with the resulting checkpoint. Epoch `resume` restores the saved composition and optimizer contract. Run full training and held-out evaluation before interpreting grasp quality.

</details>

<details>
<summary>OA-CNNs: adaptive sparse receptive fields</summary>

## OA-CNNs adaptive sparse hierarchy

`oacnns` uses the official [Pointcept implementation](https://github.com/Pointcept/Pointcept/blob/9f37497e4f3005c90bbbe7221b86439c29d60611/pointcept/models/oacnns/oacnns_v1m1_base.py) of [OA-CNNs (CVPR 2024, PDF)](https://arxiv.org/pdf/2403.14418). Its encoder learns point relations within several grid scales, then learns how to combine their receptive fields. Sparse convolutions, inverse convolutions and decoder skip fusion remain native. It shares the installed spconv and PyG operators; no additional environment or segmentation weights are needed.

Baseline, its PointNet2 port, Graspness, EconomicGrasp and FineGrasp accept this encoder. Baseline retains original-input FPS indices and 256-channel seed features. Sparse methods retain their Minkowski coordinate map and 512-channel row correspondence. The native final 1×1 sparse projection supplies grasp features. Each method retains its own seeds, labels, grouping and grasp decoder.

Generate `./panda init --example compose-oacnns`, then set your paths. In the UI, select **oacnns** under **Compose modules** and edit **Component parameters by slot**. All stage lists have the same length, from 1 to 6, ordered from fine to coarse. Downsampling is fixed at stride 2. Changing the number of stages requires supplying every stage list.

| Parameter | Default | Control |
|---|---|---|
| `embed_channels`, `stem_depth` | `64`, `3` | Stem width (8–512) and actual convolution layers (1–6) |
| `enc_channels`, `enc_depth` | `[64,64,128,256]`, `[2,3,6,4]` | Stage widths (8–512) and adaptive blocks (0–24); zero retains downsampling |
| `dec_channels`, `decoder_layers` | `[96,96,128,256]`, `[2,2,2,2]` | Decoder widths (8–512) and actual linear/normalization/activation layers (1–6) |
| `point_grid_size` | `[[16,32,64],[8,16,24],[4,8,12],[2,4,6]]` | One list per stage; 1–8 scales, each 1–512 stage lattice cells; repeated scales are allowed |
| `activation` | `relu` | ReLU, GELU or SiLU throughout the native blocks and fusion |
| `bn_eps`, `bn_momentum` | `0.001`, `0.01` | BatchNorm epsilon (0.000001–0.1) and momentum (0.001–1) |
| `relation_normalization` | `cluster` | Per-cluster/channel stable softmax; `native` retains the author's global maximum and epsilon denominator |
| `grid_origin` | `zero` | Fixed origin on the shifted lattice; `native` uses the batch-wide minimum at each stage |
| `sparse_padding` | `stride` | Pad spatial extents to multiples of the total stride; `native` uses maximum coordinate plus one |

A scale of `g` at encoder stage `i` (zero-based) spans `g × 2**(i+1) × voxel_size` metres on each axis. Lattice origins shift per scene; camera coordinates and supervision remain unchanged. Mean voxel coalescing retains an inverse mapping to every original point. Padding adds spatial extent, not synthetic occupied voxels, and keeps narrow or odd-sized clouds valid through the full hierarchy.

The default relation normalization avoids underflow and cross-scene numerical coupling from the native global shift. `grid_origin: zero` fixes cluster partitions when another scene joins the batch. For comparison with the original forward computation, select `native` for all three numerical/geometry options and use sufficiently large extents. Native normalization can underflow for large logits. Training BatchNorm still mixes statistics across the batch in either mode.

`stem_depth`, `decoder_layers` and the numerical/geometry policies are toolbox controls. The source's `groups`, `enc_num_ref`, `down_ratio` and `dec_depth` do not control the advertised computations; they are not accepted as experimental parameters. Native default widths and block counts are preserved.

Train a changed encoder with `checkpoint_policy: reuse_unchanged`, then infer with the saved checkpoint and `strict`. No GraspNet-pretrained OA-CNNs checkpoint is registered. Short training uses consecutive frames and requires `batch_size >= 2`; epoch training drops incomplete batches. Loss, point augmentation, Adam/AdamW/SGD/Lion and update schedules follow the selected method's training contract. Muon is excluded because sparse convolution kernels have no registered routing. Segmentation results from the paper do not establish grasp accuracy.

</details>

<details>
<summary>KPConvX: kernel point attention</summary>

## KPConvX kernel point hierarchy

`kpconvx` loads the official [KPConvX Pointcept wrapper](https://github.com/apple/ml-kpconvx/tree/54e644a9f3bddd4c344a58193897a44582b0fea4/Pointcept-wrapper/models/kpconvx) from [KPConvX (CVPR 2024, PDF)](https://openaccess.thecvf.com/content/CVPR2024/papers/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf). It retains the native kernel-point stem, depthwise/kernel-attention blocks, multilevel grid pooling, decoder skips and feature head. This is the author's KPConvX hierarchy; the separate KPNeXt/PTv2 hybrid is not substituted.

Baseline, its PointNet2 port, Graspness, EconomicGrasp and FineGrasp accept this encoder. Dense adapters preserve original FPS seed indices and 256-channel features. Sparse adapters preserve the Minkowski coordinate manager, map and feature-row order, including FineGrasp's normal inputs. The final native projection supplies the grasp feature width. Grasp labels, sampling and downstream grouping remain method-specific.

Generate `./panda init --example compose-kpconvx`, set your paths, and run the configuration or select **kpconvx** in the UI. Upgrade with `./panda install` to fetch the pinned source. It reuses the shared Pointcept pointops, PyG and torch-scatter installation. No separate environment or segmentation checkpoint is required.

| Parameter | Default | Control |
|---|---|---|
| `layer_blocks` | `[3,3,9,12,3]` | Encoder block counts, fine to coarse; 1–6 stages with 1–24 blocks each |
| `init_channels`, `channel_scaling` | `64`, `sqrt(2)` | Stage width is `ceil((init_channels * channel_scaling**i - 0.1) / 16) * 16` |
| `stage_channels` | Derived above | Optional explicit widths, one per stage; multiples of 16, up to 1024; overrides the width formula |
| `neighbor_limits` | `[12,16,20,20,20]` | Per-stage KNN limits, 1–128; missing neighbors use the native zero-feature shadow row |
| `subsample_size`, `radius_scaling` | `0.02`, `2.2` | Metric base size and stage scale; the first pooling grid is their product |
| `shell_sizes` | `[1,14,28]` | Kernel points per shell; 2–4 shells starting with one center point, at most 128 points total |
| `kp_radius`, `kp_sigma` | `2.3`, `2.3` | Kernel support and influence scales, relative to the base size and stage scale |
| `kp_influence` | `linear` | `constant`, `linear` or `gaussian` kernel influence |
| `kp_aggregation` | `nearest` | Stem aggregation: `nearest` or `sum`; modern depthwise blocks always use nearest-kernel assignment |
| `kp_mode`, `first_inv_layer` | `kpconvx`, `1` | Kernel attention or `kpconvd`; the first indicated encoder stages use depthwise convolution without attention |
| `inv_groups`, `inv_act` | `8`, `sigmoid` | Attention groups; negative means channels per group, zero selects depthwise blocks; `sigmoid`, `tanh`, `softmax` or `none` activation |
| `inv_grp_norm`, `modulation_scope` | `true`, `native` | Kernel-modulation GroupNorm; `native` includes all packed points, `point` normalizes each point separately |
| `share_kp`, `kpx_upcut` | `false`, `false` | Share stage kernel geometry/influences; enable the expanded-feature shortcut between encoder blocks |
| `decoder_layer` | `true` | Add a native residual block after each decoder skip fusion |
| `drop_path_rate` | `0` | Maximum encoder stochastic-depth probability; native linear block schedule |
| `norm`, `bn_momentum` | `batch`, `0.1` | `batch`, `group`, `layer` or `none`; native `none` retains learned biases |
| `activation` | `leaky_relu` | Native LeakyReLU(0.1), or ReLU/GELU/SiLU in feature blocks |

Defaults follow the native constructor. The author's ScanNet recipe instead selects constant influence and a 0.3 drop-path rate. Native packed GroupNorm includes points from other scenes even during inference; hold batching fixed when comparing it. `modulation_scope: point` is an explicit GraspPanda variation that removes this coupling from kernel attention. With `norm: group`, feature GroupNorm still uses packed-point statistics. BatchNorm uses the usual running statistics in evaluation.

The published wrapper uses **KNN**, not a hard radius cutoff, and **grid pooling with exact inverse upsampling**. The adapter keeps every input feature row; `subsample_size` defines the hierarchy scale rather than an extra initial downsampling pass. `kp_radius` changes the kernel geometry; it does not limit KNN distance. The unused `upsample_n`, old `conv_groups`, classification loss options and incomplete non-grid pooling branch are not exposed. A single stage omits pooling while retaining its encoder and head.

Kernel dispositions are cached locally by the complete shell tuple, avoiding collisions between layouts with the same point count. The default uses the author's supplied disposition. Other layouts use the author optimizer with an isolated fixed initialization seed; the first use can take longer. Native per-layer rotations/noise follow the experiment seed, and kernel buffers are stored in checkpoints. Downloaded source stays unchanged.

There are no registered GraspNet-pretrained KPConvX weights. Start with `reuse_unchanged`, train the replacement, then use `strict`; epoch resume retains the complete configuration and optimizer state. Training uses at least two frame samples per batch, consecutive frames for short runs, and drops incomplete epoch batches. Existing loss, augmentation and optimizer settings follow the selected grasp method. Paper segmentation scores do not establish grasp accuracy.

</details>

<details>
<summary>Kernel point aggregation inside cylinders</summary>

## Kernel point cylinder aggregation

`kpconvx_cylinder` uses the [official KPConvX blocks](https://github.com/apple/ml-kpconvx/blob/54e644a9f3bddd4c344a58193897a44582b0fea4/Pointcept-wrapper/models/kpconvx/utils/kpnext_blocks.py) ([CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf)) as a configurable **local aggregation component** for Baseline, its PointNet2 port, Graspness, EconomicGrasp and FineGrasp. This is a GraspPanda adaptation of the native blocks; it does not reproduce a published KPConvX grasp detector or add the scene segmentation hierarchy.

Each method retains its oriented cylinder queries, native padding/repeated samples, seed order and depth semantics. Within each cylinder, the adapter projects grouped features, constructs an independent local KNN graph and applies native kernel-attention or depthwise residual blocks. The output projects to 256 channels before max, mean or learned attention pooling. No neighbors are drawn from another cylinder. Explicit seed interaction, when selected separately, still operates after local aggregation.

Aligned local XYZ is expressed in units of that cylinder's radius. Baseline query coordinates are divided by radius; Graspness-style queries already perform this normalization. Thus `kernel_radius: 1` means one query radius, including its scale factor. Feature channels are retained. Kernel radius controls geometry, not a hard neighbor-distance cutoff. Native frame coordinates and supervision remain unchanged.

Generate `./panda init --example compose-kernel-cylinders` or choose **kpconvx_cylinder** for **Local cylindrical grouping** in the UI. Replace the `crop` slot independently of the backbone, losses and optimizer. Run `./panda install` after upgrading to ensure the pinned KPConvX source is available in the shared runtime.

| Parameter | Default | Control |
|---|---|---|
| `channels` | `[64,64,64]` | Embedding width followed by one output width per block; 1–8 blocks, widths 16–256 in multiples of 8 |
| `local_neighbors` | `8` | KNN count, at most `nsample` |
| `expansion` | `4` | Native inverted-block expansion, 1–8 |
| `attention_groups` | `8` | Must divide the input width; negative means channels per group, zero selects depthwise blocks |
| `kernel_radius`, `kernel_sigma` | `1`, `0.5` | Kernel geometry and influence scales, in query-radius units |
| `layer_scale`, `drop_path` | `0`, `0` | Optional native residual scaling and stochastic-depth probability |
| `block` | `kpconvx` | Kernel attention or `kpconvd` depthwise convolution |
| `shell_sizes`, `influence` | `[1,14,28]`, `linear` | Centered kernel shells and `constant`, `linear` or `gaussian` influence |
| `attention_activation` | `sigmoid` | `sigmoid`, `tanh`, `softmax` or `none` |
| `modulation_norm`, `modulation_scope` | `true`, `cylinder` | Kernel-modulation GroupNorm per complete cylinder, or `point` for per-point statistics |
| `normalization`, `bn_momentum` | `layer`, `0.1` | Feature normalization: `layer`, per-cylinder `group`, full-batch `batch`, or bias-only `none` |
| `activation` | `gelu` | GELU, ReLU, SiLU or native LeakyReLU(0.1) |
| `use_upcut` | `false` | Expanded-feature shortcut; consecutive expanded input widths must match |
| `nsample`, `radius_factors` | `16`, `[1]` | Native query sample count and radius multipliers; FineGrasp defaults to its original radius groups |
| `pooling` | `max` | `max`, `mean` or learned `attention` within each cylinder |
| `chunk_size`, `checkpoint` | `128`, `true` | Complete cylinders per memory chunk; `0` processes all. Activation recomputation reduces training memory |

`local_neighbors`, `expansion`, `attention_groups`, `kernel_radius`, `kernel_sigma`, `layer_scale` and `drop_path` accept a scalar or a list with one entry per block. For example, `channels: [64,96,128]` has two blocks and accepts `local_neighbors: [8,12]`. Attention settings are inactive in depthwise-only blocks.

LayerNorm and per-cylinder GroupNorm keep different candidates' statistics separate. BatchNorm deliberately uses all grouped samples in a query call and requires `chunk_size: 0`; activation recomputation preserves its running statistics. Chunk size leaves evaluation and deterministic (`drop_path: 0`) training mathematics unchanged up to floating-point roundoff. With stochastic depth enabled, keep chunk size fixed for identical random-mask assignment. Scale fusion is pointwise linear/LayerNorm/activation, without statistics shared across candidates or depth bins.

Baseline retains four depth outputs at each radius. Graspness and EconomicGrasp fuse radius outputs into one seed feature. FineGrasp retains its native cross-radius attention and receives one new local encoder per radius; `radius` and the existing `fusion_*` options control its base radius and fusion Transformer. This replacement includes local interaction, so FineGrasp's original local-attention block is not appended a second time.

No pretrained grasp weights are provided for this local component. Initialize with `reuse_unchanged`, train with the selected method's labels, then load the saved composition with `strict`. Native epoch resume restores its full configuration and optimizer state. The source and kernel disposition cache are shared with the [KPConvX backbone](#kpconvx-kernel-point-hierarchy).

</details>

<details>
<summary>Cylinder aggregation and Point Transformers</summary>

## Residual local aggregation in cylinders

`reslfe_cylinder` adapts the native ResLFE block from [DeepLA-Net (CVPR 2025 PDF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_DeepLA-Net_Very_Deep_Local_Aggregation_Networks_for_Point_Cloud_Analysis_CVPR_2025_paper.pdf) ([pinned implementation](https://github.com/zeng-ziyin/DeepLA-Net/blob/7f572899de7db26d2c5eac538395d9932faafb89/S3DIS/deepla_semseg.py)) to the `crop` slot of Baseline, its PointNet2 port and Graspness. Run `./panda install` after upgrading to fetch the source and build its CUDA operators in the shared runtime.

Each method retains its native oriented cylinder query. Within each cylinder, GraspPanda embeds the grouped input features and center-relative, approach-aligned XYZ, then constructs a local KNN graph among the grouped samples. Native ResLFE layers alternate feature-difference max pooling and residual feed-forward updates, with positional features added at each layer. The resulting features are projected to 256 channels, pooled within the cylinder and fused across radii. Baseline retains all four native depth bins; Graspness retains its seed order and native depth decoder. Relative-coordinate normalization follows the native query: Graspness divides offsets by the cylinder radius, while Baseline uses metres. Native padded query samples, including repeated points, are preserved.

This is a local-block adaptation. It does not introduce DeepLA's scene segmentation hierarchy, segmentation labels or hybrid deep-supervision losses. It initializes without pretrained DeepLA weights. Use `reuse_unchanged` with the original grasp checkpoint, train the new component, then select the resulting checkpoint with `strict`.

| Setting | Default / meaning |
|---|---|
| `width`, `depth` | `64`, `4`; feature width and number of native ResLFE iterations. Width must be a multiple of eight. |
| `nsample`, `local_neighbors` | `16`, `8`; native cylinder samples and neighbors within that cylinder. Local neighbors must not exceed the sample count. |
| `radius_factors` | `[1.0]`; multipliers of the method's native cylinder radius. |
| `mlp_ratio` | `1.0`; hidden expansion of native feed-forward layers. |
| `drop_path` | `0.1`; linearly increasing stochastic depth within the block, sampled independently per cylinder during training. A single iteration uses zero stochastic depth. |
| `bn_momentum` | `0.02`; native ResLFE BatchNorm momentum. Native zero-initialized residual scales are retained. |
| `pooling` | `max`; final cylinder pooling, also `mean` or `attention`. This does not change the native internal max-difference operator. |
| `activation` | `gelu`; also `relu` or `silu`, used in embeddings and feed-forward layers. |
| `normalization` | `batch`; also `group` or `none`, for the input/position embeddings, output projection and radius fusion. Native ResLFE layers retain BatchNorm. |

The native operator supports float32 and float16; bfloat16 is rejected. CUDA launches use the current PyTorch stream and tensor device. More samples increase the within-cylinder distance matrix quadratically; start with [`compose-reslfe`](USAGE.md#configuration-examples) (`./panda init --example compose-reslfe`). Source terms are described in [Third-party notices](THIRD_PARTY.md).

## Point Transformer V2

`point_transformer_v2` loads the original **mode 1** implementation of [Point Transformer V2 (NeurIPS 2022 PDF)](https://arxiv.org/pdf/2210.05666) from [Pointcept's pinned source](https://github.com/Pointcept/Pointcept/blob/9f37497e4f3005c90bbbe7221b86439c29d60611/pointcept/models/point_transformer_v2/point_transformer_v2m1_origin.py). It retains grouped linear weight encoding, grouped vector attention, metric grid pooling and the native skip decoder. Baseline, its PointNet2 port, Graspness, EconomicGrasp and FineGrasp expose it in their `backbone` slot. Run `./panda install` after upgrading to build its isolated pointops extension in the shared runtime.

Start with `./panda init --example compose-ptv2`. The example uses a smaller hierarchy and metre-scale grasp neighborhoods; omitted settings follow the original constructor. Each encoder/decoder list describes stages **from fine to coarse**, although decoding executes in reverse. All stage lists must have the same length, from one to six stages.

| Configure | Parameters and native defaults |
|---|---|
| Patch embedding | `patch_embed_depth: 1`, `patch_embed_channels: 48`, `patch_embed_groups: 6`, `patch_embed_neighbours: 8` |
| Encoder stages | `enc_depths: [2,2,6,2]`, `enc_channels: [96,192,384,512]`, `enc_groups: [12,24,48,64]`, `enc_neighbours: [16,16,16,16]` |
| Decoder stages | `dec_depths: [1,1,1,1]`, `dec_channels: [48,96,192,384]`, `dec_groups: [6,12,24,48]`, `dec_neighbours: [16,16,16,16]` |
| Partition pooling | `grid_sizes: [0.06,0.12,0.24,0.48]` in metres; coordinates stay in the method's camera frame |
| Attention | `attn_qkv_bias: true`, `pe_multiplier: false`, `pe_bias: true`, `attn_drop_rate: 0` |
| Regularization and memory | `drop_path_rate: 0`, `enable_checkpoint: false` |
| Upsampling | `unpool_backend: map` uses saved grid membership; `interp` uses native inverse-distance interpolation with missing neighbors masked |

Each channel width must be divisible by its attention group count. Neighborhood sizes are at most 128, matching the native CUDA operator. `pe_multiplier: true` enables the paper's additional multiplicative position encoding; the original constructor defaults to false. The example enables it explicitly. Drop-path rates follow the native linear encoder/decoder schedules; attention dropout applies to every attention block. Training requires `batch_size >= 2`, since pooling can leave one coarse point per scene. Short runs use consecutive labelled frames; epoch loaders drop incomplete final batches. Inference supports a single frame.

The dense adapter projects decoded features to 256 channels and samples original-input seed indices. Sparse adapters reconstruct camera XYZ from the lattice, combine it with the original sparse features, and restore the identical coordinate map and row order. The method retains its own seed prediction, crop, losses and decoder. FineGrasp retains its XYZ/normal features. Changing the backbone requires grasp training; these adapters do not supply pretrained grasp weights. Use `reuse_unchanged` for initialization and `strict` when reloading the resulting composed checkpoint.

Compatibility changes fix the original grouped-linear divisibility assertion, keep native CUDA work on the selected device/current stream, mask absent interpolation neighbors to prevent cross-scene feature leakage, and avoid counting BatchNorm updates twice during checkpoint recomputation. Native attention's neighbor masking and pooling reductions are retained. These are grasp feature adapters; their training uses the selected grasp method.

## LitePT encoder

`litept` uses the standalone [LitePT implementation](https://github.com/prs-eth/LitePT) ([CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yue_LitePT_Lighter_Yet_Stronger_Point_Transformer_CVPR_2026_paper.pdf)): sparse convolutions in early stages, serialized attention with three-axis PointROPE in later stages, and a lightweight skip decoder. It is available for Baseline, its PointNet2 port, Graspness, EconomicGrasp and FineGrasp. It retains each detector's own supervision and grasp decoder.

Generate `./panda init --example compose-litept`. In the browser, select **Point encoder → litept** and enter overrides under `backbone` in **Component parameters by slot**. Omitted parameters use the native small model. The example enables additional decoder attention to illustrate independent encoder/decoder settings.

| Parameter | Native default / meaning |
|---|---|
| `enc_depths`, `enc_channels` | `[2,2,2,6,2]`, `[36,72,144,252,504]`; define 1–6 encoder stages |
| `dec_depths`, `dec_channels` | `[0,0,0,0]`, `[72,72,144,252]`; one fewer stage, ordered from fine to coarse. Zero depth retains native projection and skip unpooling without extra blocks |
| `stride` | `[2,2,2,2]`; integer grid-pooling strides, 1–8, one between each encoder stage |
| `enc_conv`, `enc_attn` | `[true,true,true,false,false]`, `[false,false,false,true,true]`; boolean or one boolean per encoder stage |
| `dec_conv`, `dec_attn` | Both false by default; boolean or one per decoder stage. Attention may be enabled independently of the corresponding encoder stage |
| `enc_num_head`, `dec_num_head` | `[2,4,8,14,28]`, `[4,4,8,14]`; head counts for active attention |
| `enc_patch_size`, `dec_patch_size` | `1024` at every stage; lists of window sizes, 1–4096. Padding remains within each scene |
| `enc_rope_freq`, `dec_rope_freq` | `100`; scalar or one frequency base per stage, 1–10000 |
| `order`, `shuffle_orders` | `['z','z-trans','hilbert','hilbert-trans']`, true; serialization orders and training/evaluation order shuffling |
| `pooling` | `max`; native feature reduction can also be `mean`, `min` or `sum`. Coordinates retain mean pooling |
| `mlp_ratio`, `qkv_bias`, `qk_scale` | `4`, true, native inverse-square-root head scaling; set a positive `qk_scale` to override the scale |
| `attn_drop`, `proj_drop`, `drop_path`, `pre_norm` | `0`, `0`, `0.3`, true; native attention/MLP dropout, stochastic depth and normalization placement |
| `rope_backend` | `cuda` for the native arithmetic with safe stream/gradient handling, or `torch` for the author's PyTorch implementation. Both use FlashAttention; small floating-point differences are expected |

Every stage list must match the chosen hierarchy. Active attention requires channels divisible by heads, with head width divisible by six and at most 252. Decoder controls act only where `dec_depths` is positive. Input lattice resolution comes from the experiment's `voxel_size`, in metres. Points sharing a cell are mean-coalesced; outputs map back to every original point. Sparse adapters preserve the original coordinate map, and dense adapters preserve the original FPS seed indices. Decoder serialization and attention caches are refreshed when required by a configured stage.

Run `./panda install` after upgrading: the lock installs the author's compatible FlashAttention wheel, and the installer builds the small PointROPE extension in the shared runtime. LitePT attention requires an Ampere or newer GPU supported by that runtime. Training requires `batch_size >= 2` for coarse-stage batch normalization; short runs use consecutive frames in one scene, and epoch loaders omit incomplete final batches. The architecture starts with random weights: use `reuse_unchanged` with the method checkpoint, train the composition, then use `strict` for inference or native epoch resume. No pretrained LitePT grasp detector or paper-level grasp accuracy is implied.

## Point Transformer encoder

`sonata_ptv3` uses the native encoder and decoder from [Sonata (CVPR 2025 PDF)](https://arxiv.org/pdf/2503.16429) ([source](https://github.com/facebookresearch/sonata)), based on [Point Transformer V3 (CVPR 2024 PDF)](https://arxiv.org/pdf/2312.10035) ([source](https://github.com/Pointcept/PointTransformerV3)). The installer fetches its pinned author source into the shared environment. Attention uses the native non-Flash path; no extra environment or FlashAttention build is needed.

Baseline and its PointNet2 port use XYZ features. Points in the same voxel are averaged separately per batch, then decoded features are mapped back to every original row. FPS selects the original camera-space points, preserving the grasp labels' indices. Graspness uses its existing integer lattice and concatenates camera-space lattice XYZ with the native three input features; output retains the exact Minkowski coordinate map and row order. The experiment's `voxel_size` controls both adapters. Coordinates remain in metres.

This is a trainable architecture adaptation with random initialization. It does not load Sonata's self-supervised checkpoint or synthesize absent color/normal channels. Use `reuse_unchanged` to retain the grasp heads when replacing a native backbone, then `strict` with the resulting composition checkpoint.

| Setting | Meaning / default |
|---|---|
| `enc_depths`, `enc_channels`, `enc_num_head` | Five fine-to-coarse stages; defaults `[3,3,3,12,3]`, `[48,96,192,384,512]`, `[3,6,12,24,32]` |
| `dec_depths`, `dec_channels`, `dec_num_head` | Four fine-to-coarse stages; defaults `[3,3,3,3]`, `[96,96,192,384]`, `[6,6,12,32]` |
| `enc_patch_size`, `dec_patch_size` | Five/four attention-window caps; 128 at each stage by default; native attention reduces the cap to the smallest batch member's point count |
| `stride` | Four pooling strides, each 1/2/4/8; default `[2,2,2,2]` |
| `order` | `z`, `z-trans`, `hilbert`, `hilbert-trans`, paired orders or all four joined by `+`; default `z+z-trans` |
| `pooling` | Native hierarchy reduction: max/mean/sum/min; default max |
| `mlp_ratio`, `drop_path`, `attn_drop`, `proj_drop` | Defaults 4, 0.3, 0, 0 |
| `qkv_bias`, `pre_norm`, `shuffle_orders` | Boolean controls; default true. Order shuffling applies at every pooling stage, including inference |
| `enable_rpe`, `upcast_attention`, `upcast_softmax` | Boolean controls; default false |
| `layer_scale` | Optional positive residual scale, omitted by default |

Stage widths must be divisible by their head counts and by eight. Encoder and decoder windows can differ: the adapter refreshes native padding/relative-position caches when the window changes. Larger widths, depths, point counts and windows increase memory use. Adam, AdamW, SGD and Lion support this encoder; Muon is excluded because its current routing assumes dense convolution layouts. See the [`compose-ptv3`](USAGE.md#configuration-examples) (`./panda init --example compose-ptv3`) for a smaller trainable configuration.

</details>

<details>
<summary>Pretrained point encoders: Utonia and Concerto</summary>

## Pretrained point encoders

[Utonia](https://github.com/Pointcept/Utonia) ([paper PDF](https://arxiv.org/pdf/2603.03283), ICML 2026) and [Concerto](https://github.com/Pointcept/Concerto) ([paper PDF](https://arxiv.org/pdf/2510.23607), NeurIPS 2025) provide pretrained point features. Select `utonia` or `concerto` as the backbone for Baseline, its PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp or FineGrasp. Their grasp proposal, supervision and decoding remain method-specific.

```yaml
modules:
  backbone:
    type: concerto
    variant: tiny
    pretrained: true
    trainable_blocks: [0, 0, 0, 0, 1]
    feature_levels: [0, 1, 2, 3, 4]
checkpoint_policy: reuse_unchanged
```

`concerto` supports `tiny`, `small`, `base` (default) and `large`; `utonia` uses its released base encoder. Architecture widths and depths match the selected weights. Use the separate configurable PTv3 encoder for experiments that change those dimensions. The pretrained encoders expose:

| Parameter | Behavior |
|---|---|
| `pretrained` | Load the registered author initialization when replacing a backbone; defaults to `true`. |
| `trainable_blocks` | Five counts, fine to coarse; unfreeze the last N blocks and the downsampling projection in each selected stage. Defaults to `[0,0,0,0,0]`, training only the new grasp projection. Counts must fit the variant's depths. |
| `projection_norm` | `none` (default) or `layer` after the grasp feature projection; useful when connecting an encoder with a different feature distribution to an existing head. |
| `train_embedding` | Unfreeze the input embedding; defaults to `false`. The unused pretraining mask token remains frozen. |
| `feature_levels` | Increasing, distinct stage indices from 0 to 4. Lift and concatenate these features onto the input voxel rows, then project to the native grasp feature width. Defaults to all stages. |
| `input_scale` | Scale only the encoder's point coordinates; default 4 for Utonia and 1 for Concerto. |
| `grid_size` | Encoder voxel size after scaling; default 0.01 for Utonia and 0.02 for Concerto. Separate from the grasp method's `voxel_size`. |
| `enc_patch_size` | Five attention window sizes; default 1,024 each. |
| `drop_path`, `attn_drop`, `proj_drop` | Training regularization in unfrozen blocks; omitted values retain the registered architecture defaults. |
| `shuffle_orders` | Shuffle serialization orders while fine-tuning the backbone or PointTPA branches; default `true`. Evaluation and an encoder without trainable blocks or adapters use fixed orders. |

Dense adapters preserve original camera XYZ and seed row indices, and support [seed sampling](#network-sampling-policies). Sparse adapters encode voxel-coordinate XYZ and preserve the incoming sparse coordinate map and row order. These adapters encode XYZ only, supplying zero color and normal channels under the author's missing-input convention. Each scene is centered and voxelized independently using the native transform. Evaluation uses a fixed per-scene voxel representative seed without consuming the augmentation RNG. Training uses the run's seeded sampling state.

Each scene also passes independently through the encoder so another scene's spatial extent cannot alter its serialization windows. This uses sequential encoder forwards within a batch; the loss and optimizer still operate on the complete grasp batch. Native FlashAttention requires a supported Ampere-or-newer GPU in the [shared runtime](INSTALL.md). Frozen weights and their dropout behavior stay fixed while gradients can pass through them to selected trainable layers. The feature projection is always trainable.

Download initialization weights from the browser's pretrained-encoder panel or with `./panda component-weights concerto_tiny` / `./panda component-weights utonia`. `reuse_unchanged` strictly retains the unchanged grasp layers and initializes the replacement encoder from its registered source. Train the new projection before inference. `strict` loading restores a complete grasp checkpoint without downloading or reapplying pretraining. Code and pretrained weights have separate terms; see [downloads](DOWNLOADS.md#pretrained-point-components).

Start with `./panda init --example compose-pretrained-points`. After updating an existing installation, run `./panda install` to prepare the new pinned sources in the shared runtime. These integrations provide trainable features, not pretrained grasp detectors. Short training can still produce empty grasp sets; use epoch training with sufficient data coverage before assessing detection performance.

</details>

<details>
<summary>PointTPA: dynamic parameter adaptation inside PTv3 blocks</summary>

## PointTPA adaptation

[PointTPA](https://github.com/H-EmbodVis/PointTPA) ([paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_PointTPA_Dynamic_Network_Parameter_Adaptation_for_3D_Scene_Understanding_CVPR_2026_paper.pdf), CVPR 2026) adds serialization-based neighborhood grouping and input-dependent parameter projections. Configure `adaptation` inside `sonata_ptv3`, `concerto` or `utonia`; the selected grasp method keeps its own heads, labels and decoder. The toolbox uses the author's adaptation modules and residual placement inside the encoder blocks.

Generate the editable configuration with `./panda init --example compose-pointtpa`.

| Setting | Meaning |
|---|---|
| `blocks` | Optional increasing list of flat encoder block indices, numbered from zero in fine-to-coarse stage order. Defaults to every block contributing to a selected feature level. |
| `bottleneck_channels` | Adapter bottleneck width, 1–512; default 64. |
| `experts` | Number of parameter experts in each dynamic projection, 1–16; default 4. |
| `group_size` | Positive grouping parameter, 1–4096; default 100. With `num`, it is the number of groups; with `length`, it is the point count per group. |
| `group_mode` | `num` (default) or `length`; grouping follows the native serialization order. |
| `dynamic_down`, `dynamic_up` | Enable dynamic down/up projections; defaults `true` / `false`. A disabled projection is the author's ordinary linear projection. |
| `scale` | Positive residual branch scale, 0.001–10; default 1. |

Every setting except `type` and `blocks` accepts a scalar or one value per selected block, in the order of `blocks`. For example, `blocks: [0, 2]`, `experts: [4, 2]` and `dynamic_up: [false, true]` configure those two blocks independently. Blocks beyond the highest selected feature level are rejected because they cannot affect the grasp features.

For Concerto/Utonia, the existing `trainable_blocks` and `train_embedding` settings control original pretrained parameters; adaptation parameters and the grasp projection remain trainable. Initialization requires all original pretrained tensors to match and adds only the new adapter state. The native up projection starts at zero, so early updates can have zero gradients in the preceding adapter layers. Training activates those paths without changing the original initialization rule. Group padding and its contribution to patch attention also follow the author implementation.

`sonata_ptv3` uses its configured trainable backbone and does not automatically download pretrained weights. Choose Concerto or Utonia for pretrained parameter-efficient adaptation. Preserve the complete `modules` configuration when loading or resuming a saved grasp checkpoint; `strict` loading restores the trained adapter state without reinitializing it. `reuse_unchanged` starts a new replacement component from its declared initialization, including new PointTPA branches.

Generate `./panda init --example compose-pointtpa` to start. In the browser, select a supported backbone and add the `adaptation` mapping to **Component parameters by slot**; its parameter guide describes the accepted values. These are configurable grasp integrations, not a reproduction of the author's segmentation benchmark or a pretrained grasp detector.

</details>

<details>
<summary>RGB-D encoders, VMamba and pretrained DINO</summary>

## RGB-D image encoders

HGGD and RNG use native `D,R,G,B` tensors shaped `[B,4,640,360]`, with the author's transposed spatial axes. Image adapters retain that convention and provide five feature maps with strides 2/4/8/16/32 and channels 8/16/32/64/128. Their anchor heads, losses, coordinate conversion, local refinement and collision filtering stay native. HGGD's detached features remain detached at the original point-fusion boundary.

The modern encoders use the implementation in the locked [timm library](https://github.com/huggingface/pytorch-image-models). Paper and author-code references:

| Family | Paper | Author implementation |
|---|---|---|
| ConvNeXt V2 | [CVPR 2023 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf) | [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) |
| RepViT | [CVPR 2024 PDF](https://arxiv.org/pdf/2307.09283) | [RepViT](https://github.com/THU-MIG/RepViT) |
| MobileNetV4 | [ECCV 2024 PDF](https://arxiv.org/pdf/2404.10518) | [TensorFlow Models](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py) |

ConvNeXt/RepViT/MobileNet inputs are zero-padded only at the high ends of the spatial axes. ConvNeXt patch-center offsets are resampled onto the native lattice with bilinear interpolation and border extension; RepViT/MobileNet use their centered, odd-kernel lattices. Encoders without stride-2 outputs gain a native-sized stem. Learned projections supply the expected channels; `projection_norm` accepts batch/group/none. These are explicit grasp adaptations, not reproductions of image-classification results.

Selecting a new encoder initializes it and its projections from scratch. `reuse_unchanged` retains only the original anchor heads and complete local network; use `strict` for subsequent checkpoint inference. No ImageNet weights are downloaded implicitly. See [`compose-hggd`](USAGE.md#configuration-examples) (`./panda init --example compose-hggd`).

### VMamba state-space image features

`vmamba` uses the native visual state-space backbone from [VMamba (NeurIPS 2024 PDF)](https://proceedings.neurips.cc/paper_files/paper/2024/file/baa2da9ae4bfed26520bb61d259a3653-Paper-Conference.pdf) and its [author implementation](https://github.com/MzeroMiko/VMamba). It replaces the HGGD/RNG image encoder while preserving their anchor and local branches. The installer builds the pinned `selective_scan_cuda_oflex` operator in the shared environment; native Triton kernels perform cross-scan/merge. There is no additional virtual environment or implicit slow scan fallback.

The native convolutional patch embedding and downsampling retain pixel-zero lattice centers. This adapter processes the original `[B,4,640,360]` tensor without extra image padding, adds the required stride-2 stem and projects the four native stages into the grasp decoder's channels. Its input remains the method's native depth/RGB representation. ImageNet checkpoints are not loaded; use `reuse_unchanged` with author grasp weights, train the replacement and reload the resulting grasp checkpoint with `strict`.

| Setting | Default / choices |
|---|---|
| `stage_channels`, `stage_depths` | `[96,192,384,768]`, `[2,2,5,2]`; four stages, widths divisible by 8 |
| `state_dim`, `ssm_ratio` | `1`, `2.0`; state size and inner-channel expansion |
| `dt_rank` | Native automatic rank; supply an integer to override |
| `scan` | `cross2d`; native alternatives `unidirectional`, `bidirectional`, `cascade2d` |
| `ssm_conv`, `ssm_conv_bias` | `3`, `false`; odd local convolution kernel from 1 to 9 |
| `ssm_activation`, `mlp_activation` | `silu`, `gelu`; each also accepts relu/gelu/silu |
| `ssm_dropout`, `mlp_dropout`, `mlp_ratio` | `0`, `0`, `4.0` |
| `drop_path` | `0.2` |
| `gradient_checkpointing` | `false`; recompute native blocks during training using non-reentrant checkpointing |
| `projection_norm` | `batch`; alternatives `group`, `none` |

Start with [`compose-vmamba`](USAGE.md#configuration-examples) (`./panda init --example compose-vmamba`). Smaller stage widths/depths make configuration sweeps less expensive. HGGD also supports these components in [epoch training](#hggd-epoch-training). A larger state, stage or point count increases memory use; scan choices are architectural experiments, not an accuracy ranking.

### Pretrained DINO image features

HGGD and RNG accept `dinov2` and `dinov3` as image encoders. GraspPanda uses the locked timm implementations and verified timm conversions of the released weights. These are RGB feature encoders adapted to a grasp network; they are not independently trained GraspNet detectors.

| Component | Paper | Author implementation | Available variants |
|---|---|---|---|
| DINOv2 | [Oquab et al., 2023](https://arxiv.org/pdf/2304.07193) | [DINOv2](https://github.com/facebookresearch/dinov2) | Small / Base, 14-pixel patches |
| DINOv3 | [Simeoni et al., 2025](https://arxiv.org/pdf/2508.10104) | [DINOv3](https://github.com/facebookresearch/dinov3) | Small / Base, 16-pixel patches |

The adapter restores conventional RGB image axes, applies the registered RGB normalization and pads the right/bottom boundary by replication to complete patches. Four intermediate block outputs are projected onto native pixel centers at strides 4/8/16/32. This resampling accounts for patch-center offsets and uses border clamping. Independent depth projections retain the method's centered depth values; a local RGB-D stem supplies stride 2. The original grasp heads, local refinement and five output lattices remain in place. This feature fusion is a toolbox adaptation, not an author-provided DINO grasp architecture.

| Parameter | Default and behavior |
|---|---|
| `variant` | `small`; `base` uses a wider encoder. |
| `pretrained` | `true`; initialize a replaced encoder from the registered weights. `false` gives random encoder weights. Projection, depth and stride-2 layers always initialize locally when replacing the backbone. |
| `out_indices` | `[2, 5, 8, 11]`; four strictly increasing zero-based block indices ending at 11, assigned in order to strides 4/8/16/32. |
| `trainable_blocks` | `12` trains the whole encoder. `1`–`11` train that many final blocks and the final normalization; earlier blocks, patch embedding and positional parameters are frozen. `0` freezes the entire encoder. Adapter layers stay trainable. |
| `drop_path` | `0`; stochastic-depth rate up to 0.5. Frozen blocks stay in evaluation mode. |
| `gradient_checkpointing` | `false`; enable to reduce intermediate activation memory at the cost of recomputation. |
| `projection_norm` | `batch`; alternatives `group` and `none` apply to the RGB/depth projections. The stride-2 stem retains batch normalization. |

Start with [`compose-dino`](USAGE.md#configuration-examples) (`./panda init --example compose-dino`). Use `checkpoint_policy: reuse_unchanged` with the method's grasp checkpoint to initialize its unchanged heads and local branch. Missing encoder weights are downloaded and checksum-verified during initialization; the UI's **Pretrained image encoders** panel or `./panda component-weights dinov3_small` can prepare them ahead of time. This uses the same shared runtime.

After training, retain the module configuration and load the resulting grasp checkpoint with `strict`. Strict loading does not download or reapply DINO initialization, so it preserves the trained encoder and works without the original pretrained-weight file. Initialization provenance records the selected weight ID, immutable source URL and SHA256. Changing the source registry while a job waits causes the job to stop rather than use different initialization.

Image pretraining does not train the new grasp feature projections. Run grasp training before interpreting predictions; use the existing RNG anchor warmup when a new feature adapter produces no labeled local proposals. Weight sources and licenses are in [Data & weights](DOWNLOADS.md#pretrained-image-components).

</details>

<details>
<summary>Losses and data augmentation</summary>

## Training controls

Baseline, its PointNet2 port, Graspness, FineGrasp and EconomicGrasp accept `loss` and `augmentation` overrides in supported `train_check` or `train` actions. HGGD exposes these controls in `train_check` and `train`; RNG supports `train_check`; see [RGB-D training controls](#rgb-d-training-controls). Other methods retain their own supervision contracts. Start with [`train-controls`](USAGE.md#configuration-examples) (`./panda init --example train-controls`).

In the browser, expand **Training settings → Choose loss formulations**, select classification and regression families, then **Apply loss choices**. This writes the per-term formulations into **Loss configuration**, preserving your coefficients. Edit each term there to use different parameters. The configuration editor and sweeps use the same schema.

Generate the editable configuration with `./panda init --example train-controls`.

### Loss formulations

Weights are absolute coefficients. Unspecified terms retain their native coefficients and formulations. Omit `functions`, or select `upstream`, to retain the native objective; identical coefficient overrides preserve its values and gradients.

| Method | Classification terms | Regression terms | Native coefficients |
|---|---|---|---|
| Baseline / PointNet2 port | `objectness`, `angle` | `view`, `score`, `width`, `tolerance` | Objectness/view: 1; score/angle/width/tolerance: 0.2 |
| Graspness | `objectness` | `graspness`, `view`, `score`, `width` | Objectness: 1; graspness: 10; view: 100; score: 15; width: 10 |
| FineGrasp | `objectness`, `angle`, `depth`, `score` | `graspness`, `view`, `width` | See the FineGrasp training section below |
| EconomicGrasp | `objectness`, `angle`, `depth`, `score` | `graspness`, `view`, `width` | Objectness/angle/depth/score: 1; graspness/width: 10; view: 100 |

Each `functions` value accepts a name or `{type: NAME, ...}`. Numeric parameters use bounded, finite values; symbolic choices are listed explicitly. Omitted parameters use the defaults shown.

| Formulation | Parameters: default [range] | Behavior |
|---|---|---|
| `cross_entropy` | `label_smoothing`: 0 [0, 0.5] | Softmax cross entropy with optional uniform smoothing |
| `focal` | `gamma`: 2 [0, 8]; optional `alpha` [0, 1] | Softmax focal loss; alpha weights foreground versus background and is accepted only for binary objectness |
| `poly1` | `epsilon`: 1 [-1, 10] | Cross entropy + epsilon × (1 − target-class probability); epsilon 0 recovers cross entropy |
| `asl` | `gamma_pos`: 0 [0, 8]; `gamma_neg`: 4 [0, 8]; `label_smoothing`: 0.1 [0, 0.5] | Single-label softmax asymmetric loss, with the author's defaults |
| `logit_norm` | `temperature`: 1 [0.001, 10] | Cross entropy after per-item L2 logit normalization and temperature scaling |
| `mbls` | `margin`: 10 [0, 100]; `alpha`: 0.1 [0, 100] | Cross entropy plus a class-mean penalty on logit gaps above the margin |
| `logit_clip` | `threshold`: 1 [0.01, 100]; `scale`: 1 / threshold [0.01, 100]; `norm_order`: 2 [1, 8] or `"inf"`; `base`: `cross_entropy` | Clip each item's logit vector before the selected classification objective; see below |
| `l1` / `mse` | None | Absolute / squared normalized error |
| `smooth_l1` | `beta`: 1 [0, 10] | Quadratic-to-linear transition at beta; beta 0 is L1 |
| `huber` | `delta`: 1 [0.000001, 10] | Huber transition at delta; its scale differs from Smooth L1 when delta is not 1 |
| `charbonnier` | `epsilon`: 0.001 [0.000001, 1] | sqrt(error² + epsilon²) − epsilon |

Softmax focal loss adapts [Focal Loss (ICCV 2017)](https://arxiv.org/pdf/1708.02002) to the existing class heads. Regression transitions follow the [PyTorch Smooth L1](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html) and [Huber](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html) definitions.

The point adapters retain native positive masks, angle-label argmax/gather and target units. Baseline width and tolerance errors are divided by the native maximum width/tolerance; its grasp terms divide by the float32 valid count plus 1e-6. Graspness width targets are multiplied by 10; width loss uses positive quality labels only. Other substituted point terms retain native valid-item means. Empty masks produce a gradient-connected zero for **substituted point** terms; unmodified upstream terms keep their original behavior. A non-finite training objective stops the run.

[PolyLoss (ICLR 2022)](https://arxiv.org/pdf/2204.12511) uses the [author's Poly-1 formulation](https://waymo.com/research/polyloss-a-polynomial-expansion-perspective-of-classification-loss-functions/). [ASL (ICCV 2021)](https://arxiv.org/pdf/2009.14119) follows the [author's single-label softmax variant](https://github.com/Alibaba-MIIL/ASL), checked against the locked timm implementation. Probability complements and fractional powers use numerically stable evaluation at saturated logits. These are classification-head adaptations; no grasp accuracy improvement is implied.

[Varifocal Loss](https://github.com/hyz-xmaster/VarifocalNet) requires sigmoid quality logits. It is available through the explicitly paired [quality score head](#quality-score-heads), not as a direct substitution on an unchanged raw-score head.

### Logit normalization, margin penalties and clipping

These options work with the softmax classification terms of Baseline, its PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp and FineGrasp. They are rejected for HGGD/RNG's independent sigmoid labels and for regression terms. They change training objectives only: the native decoder, grasp-score ranking and inference thresholds remain unchanged. They are not post-hoc probability calibration, and their classification results do not establish better grasp AP or calibrated grasp confidence.

| Formulation | Paper | Reviewed implementation |
|---|---|---|
| LogitNorm | [ICML 2022 PDF](https://proceedings.mlr.press/v162/wei22d/wei22d.pdf) | [Author source](https://github.com/hongxin001/logitnorm_ood/blob/0a60eeffb7dfc970fe68e07c5649ea1c9c8244c6/common/loss_function.py) |
| MbLS | [CVPR 2022 PDF](https://arxiv.org/pdf/2111.15430) | [Author source](https://github.com/by-liu/MbLS/blob/dc86503691d1564dc29e2eed66cf7698fbbe4a2c/calibrate/losses/logit_margin_l1.py) |
| LogitClip | [ICML 2023 PDF](https://proceedings.mlr.press/v202/wei23e/wei23e.pdf) | [Author source](https://github.com/hongxin001/LogitClip/blob/7e45730b0073b6ba9af2d2af574300d42d50fcee/algorithms/clip.py) |

For a logit row `z`, LogitNorm evaluates CE on `z / (L2_norm(z) + 1e-7) / temperature`. The default temperature is the author's constructor default, 1; the author's CIFAR example uses 0.01. Choose this parameter with training/validation data rather than assuming it transfers to grasp heads.

MbLS adds `alpha * mean_classes(relu(max(z) - z - margin))` to CE. The penalty uses the same selected items and outer reduction as the classification term. `alpha: 0` recovers CE. Alpha is constant; the optional scheduling extensions in the author repository are not enabled.

LogitClip computes `n = p_norm(z) + 1e-7`, retains `z` when `n <= threshold`, and otherwise substitutes `scale * z / n`. Its default `scale = 1 / threshold` follows the released trainer and the paper's two-parameter form (Equation 4). For the single-bound norm clipping in Equation 3, explicitly set `scale` equal to `threshold`; the same small numerical epsilon is retained. Changing these parameters can introduce a discontinuity at the threshold when their values differ. `norm_order: "inf"` uses the maximum absolute logit; other orders are finite numbers from 1 to 8.

The optional `base` is a name or `{type: NAME, ...}` for `cross_entropy`, `focal`, `poly1`, `asl` or `mbls`, with the corresponding parameters above. Nested clipping and LogitNorm bases are rejected. Focal `alpha` remains limited to binary objectness/graspability even inside a clipped objective. MbLS `alpha` is the margin coefficient and is available on every supported classification term.

```yaml
loss:
  functions:
    objectness: {type: logit_norm, temperature: 1.0}
    angle:
      type: logit_clip
      threshold: 2.0
      scale: 2.0
      base: {type: mbls, margin: 1.0, alpha: 0.1}
```

Generate a complete local example with `./panda init --example train-calibration`. In the browser, choose a formulation under **Training settings → Choose loss formulations**, then edit individual terms in **Loss configuration**. Sweeps can vary paths such as `loss.functions.angle.threshold` or `loss.functions.angle.base.margin`; choose an explicit mapping in the base configuration first. These stateless objectives use the shared PyTorch runtime without additional source or weight downloads. Loss settings are saved with the experiment and checkpoint; epoch resume checks them against the saved training configuration.

### Quality score heads

`modules.head: quality_residual` adds a configurable scoring branch to Baseline, its PointNet2 port, Scale-Balanced-Grasp or Graspness. Native angle, width and tolerance predictions, seed indices and grasp geometry are retained. The branch adds a learned correction to the native score output and exposes separate `grasp_score_logits` for sigmoid objectives. This is a GraspPanda adaptation of quality-aware learning, not a DETR matcher or a reproduction of the full DEIM detector.

| Head parameter | Default | Accepted values |
|---|---|---|
| `hidden_channels` | `[128, 128]` | 1–8 pointwise layer widths, each 8–2048 |
| `activation` | `relu` | `relu`, `gelu`, `silu` |
| `normalization` | `batch` | `batch`, `group`, `none` |
| `initial_probability` | 0.1 | 0.001–0.999; initializes the residual bias to logit(p), with zero output weights |

The initial probability sets only the residual bias; the native score also contributes to the final logit. With `reuse_unchanged`, all original head weights are retained and only the residual branch is initialized. Train the changed scoring model before using its predictions. A completed grasp checkpoint reloads with `strict` and the same head settings. Both the existing score path and the residual branch participate in learning; the hidden residual layers begin receiving gradients after the zero-initialized output weights update.

#### Score mappings and supervision

| Method | Native label `s` | Sigmoid target `q` | Score passed to the native decoder |
|---|---|---|---|
| Graspness | Batch-normalized grasp quality in [0, 1] | `q = s` | `sigmoid(z)` |
| Baseline / PointNet2 port / Scale-Balanced-Grasp | Nonnegative log friction-ratio quality | `q = 1 - exp(-s)` | `softplus(z)`, the stable inverse mapping of `sigmoid(z)` |

The dense-method mapping avoids assuming that log-quality labels lie below 1 or silently clipping them. It retains nonnegative log-score semantics; the native depth selection, tolerance multiplication and geometric decoder still consume the mapped score. These scores are grasp-ranking quantities, not calibrated physical success probabilities. FineGrasp/EconomicGrasp's categorical quality heads and HGGD/RNG's heatmap targets do not share this contract and reject these score objectives.

Select one of the following under `loss.functions.score`; all require the paired `quality_residual` head. Other loss terms and coefficients remain independently configurable. Selecting the head with an upstream or regression objective instead is a separate ablation: that objective acts on the decoded score and retains its original supervision.

| Objective | Parameters | Per-item definition |
|---|---|---|
| `quality_bce` | None beyond supervision/reduction | BCE-with-logits against `q` |
| `varifocal` | `alpha`: 0.2 [0, 10]; `gamma`: 2 [0, 8] | BCE against `q`, weighted by `q` for positive quality and `alpha * p^gamma` for zero quality |
| `mal` | `alpha`: 1 [0, 10]; `gamma`: 2 [0.000001, 8] | BCE against `q^gamma`, weighted by 1 for positive quality and `alpha * p^gamma` for zero quality |

Here `p = sigmoid(z)` is detached when forming the focusing weights. Positive quality means `q > 0`; zero-quality bins form the negative set. These definitions and defaults follow the [reviewed DEIM implementation](https://github.com/Intellindust-AI-Lab/DEIM/blob/09d35d53d39ee3145a1e61e3a989b28b9468d1dd/engine/deim/deim_criterion.py). MAL is from [DEIM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf); Varifocal originates in [VarifocalNet, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf). Grasp quality replaces detection IoU, with the following explicit grasp reductions. MAL's powered target changes the learned score emphasis; it is not equivalent to BCE or a claim of better grasp calibration.

All three objectives also accept:

- `supervision: native` (default): Graspness uses all native score bins. Dense Baseline methods retain GT best-angle selection for training and the native object/quality mask. SBG retains its depth-shared score mask and scale prior. Ordinary Baseline's native score mask contains positive quality only, so the negative focusing branch is inactive in this mode.
- `supervision: all_angles`: Dense methods supervise every angle/depth bin on object seeds, including zero-quality bins; this changes the score-training protocol. SBG also retains its scale prior. Graspness already supervises all bins, so the two modes coincide there. Inference never selects angles using labels.
- `normalization: native` (default): Divide by the selected-item count, or selected scale-weight sum for SBG, retaining the dense methods' 1e-6 denominator offset.
- `normalization: positive`: Divide by the positive-quality count (scale-weighted for SBG), clamped to at least 1. This can increase the contribution of abundant zero-quality bins; set the score coefficient deliberately when comparing it with a native mean.

Invalid targets are rejected. Empty selected sets return a gradient-connected zero. Regression, angle, width and tolerance supervision are unaffected by these score-only settings. No improved AP or calibration is implied by selecting an objective.

```bash
./panda init --example train-quality-head
```

```yaml
modules:
  head: {type: quality_residual, hidden_channels: [128, 128]}
checkpoint_policy: reuse_unchanged
loss:
  functions:
    score: {type: mal, gamma: 2.0, supervision: all_angles, normalization: native}
```

In the browser, choose the head under **Compose modules**, then select **Quality score objective** under **Training settings → Choose loss formulations** and click **Apply loss choices**. Edit score parameters in **Loss configuration**. The form, configuration editor, sweeps and epoch checkpoints retain both the head and objective settings. Sweep paths include `modules.head.hidden_channels`, `loss.functions.score.gamma` and `loss.functions.score.supervision`.

### RGB-D training controls

HGGD and RNG use independent sigmoid classification targets. The UI's classification selector applies binary cross-entropy, sigmoid focal, binary Poly-1 or multi-label ASL to these methods. The same `loss.functions` syntax applies; regression choices and their parameters are unchanged.

| Term | Methods | Default coefficient | Supervision |
|---|---|---|---|
| `anchor_location` | HGGD / RNG | 1 | Location heatmap; positive threshold 0.99; negative suppression `(1-target)^4` |
| `anchor_classification` | HGGD / RNG | 1 | In-plane anchor classes; positive threshold 0.5 |
| `anchor_theta`, `anchor_depth`, `anchor_width` | HGGD / RNG | 5/3 each | Three normalized anchor offsets, with native prediction clipping to [-0.5, 0.5] and the same positive-anchor mask |
| `local_orientation` | HGGD / RNG | 1 | Native gripper-symmetric orientation similarity targets; positive threshold 0.99 |
| `local_offset` | HGGD / RNG | 1 | XYZ offsets normalized by 0.02 metres; strict orientation-positive mask |
| `local_theta_classification` | RNG | 1 | Native local angle histogram targets; positive threshold 0.4 |
| `local_theta` | RNG | 5 | Normalized within-bin angle offsets |
| `local_width` | RNG | 1 | Log width relative to the 60-mm local width anchor |

Coefficients follow the registered training presets, including HGGD's published joint-training shell configuration. The anchor regression coefficient 5 is divided equally across its three terms. Coefficients are absolute: setting one to zero removes that objective; it does not freeze shared parameters or disable optimizer weight decay. At least one objective must remain positive, and RNG warmup requires a positive anchor objective.

Classification keeps the native positive-count denominator (or an unnormalized sum when there are no positives), negative heatmap suppression and class balancing: positive alpha 0.25 for anchor/local-theta classes and 0.5 for location/orientation. `focal.alpha` can override alpha per classification term. Probability clipping remains 1e-6 for HGGD and 1e-4 for RNG. `cross_entropy.label_smoothing` mixes binary labels with 0.5. `poly1.epsilon` adds the binary Poly-1 term before the same weighting and reduction.

For these sigmoid heads, `asl` follows the [multi-label ASL formulation](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py): `gamma_pos: 0`, `gamma_neg: 4`, and `clip: 0.05` (range 0 to 0.5), with gradients through the focusing factor. It retains the grasp method's balancing, suppression and reduction. It accepts `clip` instead of the softmax variant's `label_smoothing`. This is an adaptation of the classification objective to grasp supervision.

All loss choices use the pinned native target builders. Regression retains native masks, normalized targets and reductions: anchor terms use the valid-anchor count plus the method's epsilon; RNG local theta/width use the valid-bin count; local XYZ sums three coordinates per valid orientation. Original empty-label behavior is retained and non-finite objectives stop the run. Inference and decoding are unchanged; the UI clears training-only overrides when reusing a checkpoint.

#### RGB-D observation augmentation

Use `augmentation.mode: custom` or omit `mode` when providing the following parameters. An empty mapping, `native` or `none` retains the unaugmented short-training preset. HGGD epoch training uses the stage-dependent behavior described [below](#hggd-epoch-training). These settings are separate from the point augmentation schema.

| Parameter | Default · accepted values | Meaning |
|---|---|---|
| `brightness`, `contrast`, `saturation` | 0 · [0, 1] | torchvision ColorJitter factors sampled within 1 ± the value |
| `hue` | 0 · [0, 0.5] | Hue shift sampled within ± the value |
| `grayscale_probability` | 0 · [0, 1] | Convert RGB to three-channel grayscale |
| `blur_probability` | 0 · [0, 1] | Apply Gaussian blur after color jitter and grayscale |
| `blur_kernel` | 5 · odd integers [3, 31] | Full-resolution blur kernel |
| `blur_sigma_min`, `blur_sigma_max` | 0.1 / 2 · [0.001, 10] | Ordered blur sigma bounds |
| `depth_noise_std`, `depth_noise_clip` | 0 / 0.01 · [0, 0.01] / [0, 0.1] | Noise standard deviation is `depth_noise_std * z^2` metres; clip is an absolute metre bound |
| `depth_dropout` | 0 · [0, 0.3] | Independent missing-depth probability at valid pixels |

RGB and depth perturbations are applied at full resolution before the native anchor resize. The same modified observations feed the local point cloud (HGGD) or local image patches (RNG). Invalid depth stays zero, valid noisy depth stays positive, and absolute grasp poses remain fixed. Native relative-depth targets are generated from the modified depth observation. Geometric image rotation/cropping and point-only perturbations are rejected because they require additional camera and local-label transforms.

HGGD samples augmentation for each dataset item. RNG samples once before creating its fixed training frame, optional anchor warmup and local patches; it retains its bounded native-objective training protocol. Start with the loss and augmentation section of [`compose-hggd`](USAGE.md#configuration-examples) (`./panda init --example compose-hggd`). Sweeps can vary paths such as `loss.functions.local_orientation`, `loss.weights.local_offset` and `augmentation.depth_noise_std`.

##### PRIME photometric augmentation

For HGGD and RegionNormalizedGrasp training, add `prime` to the RGB-D augmentation mapping. This adapts the smooth-color and random-filter primitives from [PRIME (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850615.pdf), using its [pinned author code](https://github.com/amodas/PRIME-augmentations). It augments RGB appearance while retaining the image lattice, camera calibration, depth and grasp labels. Spatial diffeomorphisms and classification JSD consistency are excluded; this is the photometric adaptation, not a reproduction of PRIME classification training.

```yaml
augmentation:
  mode: custom
  prime:
    primitives: [color, filter]
    mixture_width: 3
    mixture_depth: -1
    max_depth: 3
    probability: 1.0
```

Generate a complete experiment with `./panda init --example train-prime-rgbd`. In the UI, select HGGD/RNG training, expand **Choose RGB photometric policy**, select the primitives and **Apply photometric policy**. This preserves other augmentation entries; all parameters remain editable in **Augmentation configuration**. Removing the PRIME override retains the other entries and selected augmentation mode.

| `prime` parameter | Default · accepted values | Meaning |
|---|---|---|
| `primitives` | `[color, filter]` | A nonempty unique list; each chain step selects one primitive uniformly |
| `probability` | 1 · [0, 1] | Probability of applying the mixture to an RGB observation |
| `mixture_width` | 3 · integers [1, 8] | Number of independently transformed chains mixed with Dirichlet(1) weights |
| `mixture_depth` | -1 · -1 or integers [1, `max_depth`] | -1 samples each chain's depth uniformly; positive values give an exact fixed depth |
| `max_depth` | 3 · integers [1, 8] | Maximum random depth and upper bound for fixed depth |
| `stochastic` | true · boolean | Sample primitive strength for each application; false uses the configured strength and kernel |
| `color_cut` | 500 · integers [1, 500] | Largest smooth-color sine frequency; stochastic mode first samples the cutoff uniformly |
| `color_bandwidth` | 20 · integers [1, 500] | Maximum number of consecutive sine frequencies, with a random start in the current cutoff |
| `color_temperature` | 0.05 · [0, 0.1] | Variance of Gaussian sine coefficients; stochastic mode samples temperature uniformly from zero |
| `filter_kernel` | 3 · odd integers [3, 15] | Odd spatial-filter size, retained in stochastic mode; the released kernel candidate range contains only this size |
| `filter_sigma` | 4 · [0, 4] | Standard deviation of the additive Gaussian filter coefficients; stochastic mode samples it uniformly from zero |

The mixture blends the original RGB with its augmented chains using Beta(1,1). Existing color jitter, grayscale and blur run first. PRIME then runs once at full resolution before native resizing and local-feature extraction; the two grasp branches receive the same augmented observation. Depth settings, if provided separately, retain their existing behavior. Evaluation and inference do not apply PRIME.

Two source corrections are explicit: the identity impulse uses the center of the odd filter, so zero noise preserves pixel locations; a fixed chain depth no longer receives the author's random early-stop mask. Smooth-color frequencies are evaluated in chunks with the same sampled coefficients to bound memory, and the final RGB is rounded to 8-bit for the native PIL loader. CPU dataset workers use the experiment's seeded PyTorch RNG; changing worker count changes the sampled augmentation stream. Larger widths, depths and frequency bandwidths increase preprocessing cost. No additional package environment or model weights are required; `./panda install` fetches the pinned sources.

Use sweep paths such as `augmentation.prime.color_temperature`, `augmentation.prime.filter_sigma`, or `augmentation.prime.mixture_width`. Keep dataset and observation protocols fixed when comparing robustness; these transforms do not imply improved grasp AP.

### Point augmentation

An empty mapping preserves the action's original behavior. `mode: none` disables augmentation; `mode: native` selects the author's transform. The following parameters require `mode: custom` (the default for a nonempty mapping).

| Parameter | Default · accepted values | Meaning |
|---|---|---|
| `rotation_axis` | x · x/y/z | Rotation axis in camera coordinates |
| `rotation_degrees` | 0 · [0, 180] | Uniform rotation within ± the selected angle |
| `translation` | 0 · [0, 0.5] metres | Independent uniform translation per axis |
| `jitter_std` / `jitter_clip` | 0 / 0.01 · [0, 0.02] / [0, 0.1] metres | Gaussian XYZ observation noise and absolute clipping bound |
| `point_dropout` | 0 · [0, 0.8] | Fraction of remaining sampled rows discarded randomly |
| `cutout_fraction` | 0 · [0, 0.5] | Fraction of sampled rows removed nearest a randomly chosen 3D point |
| `depth_noise_std` | 0 · [0, 0.01] | Camera-depth noise coefficient: standard deviation in metres = coefficient × z², with z in metres |
| `depth_noise_clip` | 0.01 · [0, 0.1] metres | Absolute depth perturbation bound |

Depth noise follows each original camera ray before the rigid transform. Points and object poses then transform together; XYZ jitter affects observations only. Local cutout precedes random dropout. Removed rows are replaced by sampled retained rows, keeping the configured point count and at least min(1024, input count) retained source rows. This bound does not guarantee distinct geometric points.

One row map updates points, colors/features, objectness and per-point graspness labels together. Sparse coordinates are recomputed afterward. Object-frame grasp annotations remain clean; their object poses carry the rigid transformation. Epoch augmentation applies to training samples only, including when using loader workers. Short training resamples the selected labelled sample each update.

#### Density and local/global sampling

Baseline, its PointNet2 port, Graspness, EconomicGrasp and FineGrasp accept `augmentation.resampling` in custom mode, with any registered backbone or crop. The controls use sampling rules from [PointSP (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/48), [paper PDF](https://arxiv.org/pdf/2408.12062) and [pinned author implementation](https://github.com/tangsankou/PointSP/blob/8206043f27e8b849fecde48841ff4b2513e88439/PCT_Pytorch/sampling.py).

```yaml
augmentation:
  mode: custom
  rotation_degrees: 10
  resampling:
    type: pointsp_wrs
    keep_ratio: [0.5, 1.0]
    neighbors: 20
    density_quantile: 0.5
```

| Parameter | Default / meaning |
|---|---|
| `type` | `uniform`: uniformly choose retained rows; `pointsp_wrs`: choose without replacement using density weights; `pointsp_lgd`: remove points from a random local/global neighborhood. |
| `keep_ratio` | `[0.5, 1.0]`: draw a fresh log-uniform retention ratio per sample. A scalar fixes the ratio; values must be in `[0.1, 1]`. At least 1,024 rows are retained. |
| `probability` | `1.0`: probability of applying this sampling step, from 0 to 1. |
| `neighbors` | WRS only; `20`, from 1 to 128, including self and repeated observations. |
| `density_quantile` | WRS only; `0.5`, from 0 to 1. Compute the selected quantile of per-point mean squared neighbor distances, count each point's neighbors below that threshold, then normalize counts into sampling probabilities. |
| `global_fraction` | LGD only; `random` draws uniformly from `[0, 1)`. `0` removes the nearest points to a randomly selected center; `1` permits removal anywhere. Intermediate values expand the eligible neighborhood. |

Sampling runs after the native loader, configured rigid/noise transforms, cutout and dropout, before collation. LGD follows the author's removal count and neighborhood formula; WRS uses an exact CPU KD-tree with double-precision distances instead of the author's dense pairwise tensor. Exact self-distances are zero, avoiding the dense formula's floating-point cancellation when neighborhoods collapse to a point. It requires no new environment or CUDA build. NumPy follows the experiment/loader seeds; its random draws do not reproduce the author's Torch RNG sequence.

These are **training observation adaptations**: retained rows keep their original positions, and removed positions are filled with randomly repeated retained rows to maintain `num_points`. Colors, normals and per-point labels use the identical index map; voxel coordinates are rebuilt. Object grasp labels and poses retain their native supervision. Padding does not create new geometry, and duplicate rows are still repeated observations, not additional coverage. The full PointSP classification protocol, filtered FPS and tangent-plane interpolation are not enabled by this setting. Validation and inference keep their original sampling.

Generate `./panda init --example augment-pointsp`. In the UI, open **Training settings → Choose observation sampling**, select a rule and retention limits, then **Apply sampling choices**. This switches to custom augmentation and writes **Augmentation configuration**, preserving other transforms. Advanced options remain editable there; switching rules retains shared probability but removes parameters belonging to the previous rule. **Remove sampling override** leaves the augmentation mode and other transforms unchanged. Compare sampling rules with `augmentation.resampling.type` in a sweep, removing type-specific fields from the base when comparing different rules. Keep the same observation count, seed and training budget when assessing their effect. See [third-party notices](THIRD_PARTY.md) for provenance.

These are point-observation controls. Image crops, nonrigid warps, scaling and scene mixing need their own camera, grasp-pose, width and collision-label transformations; matching tensor sizes does not make their supervision interchangeable.

Masked objectives can give a connected component zero gradient for a batch. Point trainers retain the native update and record `zero_gradient_components` in run details; disconnected components, non-finite gradients and missing updates from active components remain errors.

Loss and augmentation settings are training-only. Resume requires saved configuration metadata for training overrides and the same objective, augmentation, sampled frame range, seed and loader worker count. **Prepare inference from checkpoint** retains architecture settings and clears training-only options automatically. To sweep a loss parameter, use a structured formulation in the base configuration and vary, for example, `loss.functions.objectness.epsilon`; augmentation uses paths such as `augmentation.point_dropout`.

</details>

<details>
<summary>Optimizers and learning-rate schedules</summary>

## Optimizers and schedules

Baseline, its PointNet2 port, Graspness, FineGrasp, EconomicGrasp, SBG, HGGD and RNG accept optimization overrides in their registered training actions. Empty mappings retain the method's optimizer and schedule. Select the optimizer/schedule under **Training settings** in the UI, then enter parameters without `type`; full experiment files include `type` as below.

```yaml
learning_rate: 0.00003
optimizer:
  type: lion
  weight_decay: 0.01
scheduler:
  type: cosine
  warmup_steps: 1
  min_lr_ratio: 0.01
```

| Optimizer | Configurable parameters |
|---|---|
| `adam`, `adamw` | `weight_decay`, two-element `betas`, `eps` |
| `sgd` | `weight_decay`, `momentum` (default 0.9), `nesterov` |
| `lion` | `weight_decay`, two-element `betas` |
| `muon` | `weight_decay`, `momentum`, `nesterov` (default true), `ns_steps`, `fallback_lr_scale`, two-element fallback `betas`, `eps` |

The base learning rate always comes from `learning_rate`. Explicit optimizer overrides default to zero weight decay. Adam/AdamW/SGD use PyTorch; Lion and Muon use the locked timm implementations. [Lion](https://github.com/google/automl/tree/master/lion) ([NeurIPS 2023 paper](https://arxiv.org/pdf/2302.06675)) typically needs a smaller learning rate than AdamW. [Muon](https://github.com/KellerJordan/Muon) uses timm's matrix/convolution routing and AdamW fallback, with flattened convolution kernels. It is registered only for dense baseline/PointNet2 compositions and HGGD/RNG models; PTv3, LitePT and OA-CNNs use spconv and are excluded; sparse-kernel parameter layouts need a separate routing contract. These choices do not imply improved grasp accuracy.

| Schedule | Parameters and behavior |
|---|---|
| `constant` | Optional `warmup_steps`, then the base learning rate |
| `cosine` | Optional `warmup_steps` and `min_lr_ratio` (default 0), decaying over the configured update horizon |
| `multistep` | Increasing, unique `milestones` and `gamma` (default 0.1) |

Schedule units are **completed optimizer updates**, including for epoch training. A milestone of 2 changes the third update's rate. Warmup starts at `base_lr / warmup_steps`; cosine reaches its floor after the final update. A short run's horizon is `training_steps`, plus optional `proposal_warmup_steps`; epoch training uses updates per epoch times `epochs`, including configured batch limits. HGGD divides the loader length by `trainer.accumulation_steps`, rounding up for the final partial group. Changing the loader or final epoch horizon on resume is rejected. Model, optimizer and schedule states are restored and checked before the next update.

Optimizer and scheduler settings are training-only; clear them for manually authored inference configurations. UI checkpoint reuse does this automatically. Sweep paths such as `optimizer.type`, `optimizer.weight_decay` and `scheduler.min_lr_ratio` are supported, subject to each selected implementation's validation.

</details>

## Checkpoint policies

- `strict`: every checkpoint key and parameter shape must match. Use this for original models and a saved checkpoint of the same composition.
- `reuse_unchanged`: discard checkpoint parameters only inside explicitly replaced modules, retain their constructor initialization, and load all remaining parameters strictly. The result records exactly which keys were initialized or discarded.

Selecting a different encoder and silently accepting all missing keys would conceal implementation mistakes. GraspPanda rejects mismatches outside the chosen slots.

<details>
<summary>Seed-prediction warmup for new Graspness, EconomicGrasp and FineGrasp encoders</summary>

## Seed-prediction warmup

These methods choose graspable seeds from predicted foreground and graspness. A newly initialized backbone can produce an empty candidate set. For `action: train_check`, set `proposal_warmup_steps` to first optimize the native foreground and graspness objectives, then perform `training_steps` complete grasp updates. The UI exposes **Proposal warmup updates** under training settings; `0` retains the normal path.

Warmup uses the same optimizer, augmentation, seed-loss formulations and coefficients. Candidate-dependent heads are inactive during warmup. Full training uses predicted seeds and the unchanged native threshold; an empty set still stops the run. Schedules count both phases. The transition to all grasp objectives can require a lower learning rate, for example:

```yaml
method: economicgrasp
action: train_check
proposal_warmup_steps: 300
training_steps: 20
learning_rate: 0.001
scheduler: {type: multistep, milestones: [300], gamma: 0.01}
```

Generate `./panda init --example compose-flash3d-economic` for the complete configuration with its Kinect checkpoint and input settings. Set the warmup budget and learning-rate schedule for your experiment; these example values are initialization settings, not a convergence guarantee.

To continue across epochs, use the saved `checkpoint.pt`, keep the same `modules`, set `checkpoint_policy: strict`, `train_checkpoint_mode: initialize`, `action: train` and `proposal_warmup_steps: 0`. Choose an epoch-appropriate schedule or clear `scheduler` to use the native one. This starts a new optimizer from the initialized model; later epoch checkpoints support `resume`. For inference, use **Prepare inference from checkpoint**, which retains the composition and clears training controls.

</details>

<details>
<summary>Short training and checkpoint reuse</summary>

## Short training

Download the baseline weights and prepare its training labels as described in [downloads](DOWNLOADS.md). Generate a local configuration and set your dataset path:

Generate the editable configuration with `./panda init --example compose-baseline`.

```bash
./panda run experiment.local.yaml
```

By default, short training repeats one labelled frame without augmentation and computes the native loss. Registered overrides apply the configured objective and augmentation. HGGD, GraNet and fusion use batch size 2; FineGrasp, PCM, PTv2, LitePT and OA-CNNs compositions use the configured `batch_size`; PCM, PTv2, LitePT and OA-CNNs require at least 2 consecutive frames. Other point methods use batch size 1. RNG uses anchor batch 2 and up to 48 local patches. CenterGrasp checks its SGDF and RGB objectives separately. Outside FineGrasp, PCM, PTv2, LitePT and OA-CNNs compositions, the general `batch_size` field applies to native epoch training. `epochs` always applies to the full `train` action.

View loss curves and the saved checkpoint in **Runs & results**; [output handling](USAGE.md#experiment-artifacts) is shared across methods.

## Use the saved model

Follow [Train, resume and reuse](USAGE.md#train-resume-and-reuse). Keep the trained module configuration and use `strict` checkpoint loading for prediction.

</details>

<details>
<summary>Epoch training: Baseline, Graspness, EconomicGrasp, HGGD and FineGrasp</summary>

## Train a composed model across epochs

Baseline, Graspness, EconomicGrasp, FineGrasp and HGGD accept their registered module choices in `action: train`. SBG also exposes its native epoch trainer. Native dataset loops remain in use; omitted controls retain the author's augmentation, objective, optimizer and schedule. Object/collision labels load through bounded caches instead of eagerly occupying memory for every scene.

Start an epoch run by changing the example above. Use a small batch limit for an initial run; set both limits to `0` for the complete native ranges:

```yaml
action: train
train_checkpoint_mode: initialize
epochs: 1
batch_size: 2
train_batch_limit: 3
eval_batch_limit: 1
data_workers: 0
```

`initialize` loads model weights with the selected transfer policy and starts a fresh optimizer at epoch 0. `resume` requires `strict`, restores the optimizer and epoch, and requires the same composition and data settings. Restored model and optimizer tensors are checked before the next update. Native CUDA point operators use atomic gradient accumulation, so later loss trajectories are not guaranteed to be bitwise identical. Set `epochs` to the final epoch number, greater than the saved epoch. SBG's OneCycle schedule requires the original final-epoch horizon when resuming; initialize a new run to change that horizon.

A nonzero training batch limit selects consecutive frames from `scene`/`frame` before native shuffling and augmentation. The native validation loop uses a prefix of test_seen for Baseline/SBG; Graspness has no validation loop in its released trainer. These validation losses are diagnostics, not benchmark AP or a recommended model-selection protocol. Set both limits to **0** for the complete native training/validation ranges and increase `timeout_minutes` for a long run. Epoch-boundary seeds are controlled by the configured seed plus epoch.

Each run saves native epoch checkpoints under `training/` and the last checkpoint as `checkpoint.pt`. The UI exposes initialization/resume, batch limits, worker count, losses and checkpoint inference.

## EconomicGrasp epoch training

Generate `./panda init --example train-economicgrasp`, set the dataset path, then run the generated file. The template starts without pretrained weights and uses the author's ten-epoch Kinect recipe with Adam, batch size 4, 20,000 points and learning rate 0.001. Prepare `economic_grasp_label_300views/` and `graspness/` for training scenes 0000–0099 using the [author instructions](https://github.com/iSEE-Laboratory/EconomicGrasp#training). These scene labels are opened by the native dataset reader as needed; full grasp/collision archives are not consumed by this training loop.

The native driver retains its sampler, collator, selected-view supervision, optimizer updates and epoch checkpoints. Empty `augmentation`, `loss`, `optimizer` and `scheduler` mappings preserve native settings, including YZ reflection and epoch-based cosine learning rate. Registered [components](#economicgrasp-components), objectives, point augmentation and optimization controls can be combined in the same epoch run. A configured scheduler uses optimizer updates instead of epochs.

For a bounded first run, set `train_batch_limit: 1`; the selected range begins at `scene`/`frame` and contains up to `batch_size` frames per batch. Set it to `0` for all 25,600 training views. There is no native validation loop: keep `eval_batch_limit: 0` and evaluate complete split predictions separately.

To continue an interrupted run, select its last completed `training/checkpoints/epoch_XXXX.tar`, set `train_checkpoint_mode: resume` and `checkpoint_policy: strict`, and retain its module, data, loss, augmentation and optimizer settings. Keep `epochs` at the **original final epoch**, greater than the saved epoch: the native cosine schedule depends on that horizon. A configured schedule also requires its saved state and original update horizon. Author checkpoints without toolbox metadata can restore native model/optimizer/epoch state with no training overrides; supply the original horizon yourself because those files do not record it. Use `initialize` to change the experiment or fine-tune a completed checkpoint.

Model and optimizer state are checked exactly before the resumed update. Subsequent CUDA reductions can vary numerically; epoch-boundary seeds do not guarantee bitwise-identical trajectories. The last completed epoch is exported as `checkpoint.pt` for strict inference with the same module settings.

## RNG proposal initialization

A newly initialized image encoder may produce proposals with no local grasp labels. For RNG short training, set `proposal_warmup_steps` to train the anchor on its native heatmap targets before preparing its own local patches. These updates are additional to `training_steps`; a custom learning-rate schedule spans both phases. The default is `0`. Warmup uses the selected optimizer and real targets, with no teacher model or replacement labels. Its loss is shown separately as **Anchor warmup**. The required duration depends on initialization, learning rate and scene; a positive proposal set is checked before local training.

The browser exposes this setting under **Training settings** for RNG. Checkpoint inference clears this training-only setting. This initialization procedure is a toolbox option; the unreleased RNG full training schedule is not reproduced.

## HGGD epoch training

Generate the [`train-hggd` example](USAGE.md#configuration-examples) with `./panda init --example train-hggd`, set your dataset/checkpoint paths in `experiment.local.yaml`, then use `./panda run experiment.local.yaml`. The example fine-tunes author weights at a conservative learning rate; it is not a reproduction of the paper's training hyperparameters. Prepare the camera-specific [HGGD targets](DOWNLOADS.md#method-specific-preprocessing) for training scenes 0000-0099 and validation scene 0100. `workspace: native_demo` retains the native RGB-D geometry and target generation. Batch size must be at least 2.

Set `trainer` in YAML/JSON or expand **Training settings → Method training stages** in the UI:

| Parameter | Default | Meaning |
|---|---|---|
| `joint_training` | `true` | Keep the anchor network trainable after the local stage starts; `false` freezes it, including batch-normalization state |
| `pre_epochs` | `0` | Initial anchor-only epochs; local training starts afterward |
| `shift_epochs` | `5` | Refine gamma/beta anchors during these initial epochs when local labels are available; `0` disables refinement |
| `accumulation_steps` | `2` | Average gradients across this many full mini-batches per optimizer update |
| `center_num` | `128` | Proposed local training centers per image |
| `group_num` | `512` | Points per local training group |
| `local_grasp_num` | `500` | Maximum local grasp labels per group |
| `shift_min_labels` | `1000000` | Collected local labels needed to trigger native anchor refinement; collection resets each epoch |

`initialize` loads both available network branches and gamma/beta anchors, then creates fresh optimizer state. Use `checkpoint_policy: reuse_unchanged` when replacing the image encoder. An empty checkpoint starts the grasp networks from their constructors; select positive `pre_epochs` to establish anchor proposals before local training. Configured DINO pretraining still applies. A batch with no local proposals stops with an actionable error.

Empty optimization settings use AdamW with weight decay `0.01`, epoch StepLR (factor `0.1` every 5 epochs) and the author's anchor batch-normalization momentum schedule. Each accumulated update uses mean gradients and native value clipping at 1; the final partial accumulation group is also stepped. This corrects the native loop's first-group/tail handling, so its original learning rate may need adjustment. Registered [optimizers and update schedules](#optimizers-and-schedules) can replace these defaults.

Empty or `native` augmentation preserves the author's extra anchor-pretraining augmentation when `joint_training: false`. `none` disables it; `custom` applies only the configured RGB-D observation perturbations to each training item. Validation stays unaugmented. Rotation and zoom remain disabled to preserve camera and label correspondence.

`train_batch_limit: 0` selects all training frames, with incomplete mini-batches dropped. A positive limit selects consecutive frames from `scene`/`frame` before shuffling. Native validation always uses scene 0100: `eval_batch_limit: 0` selects its 256 frames, otherwise its first N frames. Its outputs are geometric matching, coverage and 2D overlap, not official AP. Generate full-split predictions and use `evaluate` for the benchmark metric.

Each completed training epoch is saved before validation to `training/checkpoints/epoch_NNNN.tar` and `checkpoint.pt`. Resume requires a complete epoch checkpoint, `checkpoint_policy: strict` and unchanged data, components, objectives, augmentation, trainer and optimization settings. It restores and checks both networks, optimizer, schedule, batch-normalization momentum and anchors. Frame inventory checking uses paths, sizes and modification times, not image-content hashes. Native StepLR permits increasing the final `epochs`; configured update schedules require the original horizon. Workers restart with explicit epoch seeds; later GPU training trajectories are not guaranteed to be bitwise identical after process restart.

## FineGrasp training and composition

FineGrasp exposes its own native training adapter, alongside the FineGrasp grouping replacement available to Graspness. Use [`train-finegrasp`](USAGE.md#configuration-examples) (`./panda init --example train-finegrasp`) with the economic labels described in [Data & weights](DOWNLOADS.md#finegrasp).

| Part | Configuration and contract |
|---|---|
| Point encoder | `modules.backbone: upstream` keeps the released MinkUNet. `sonata_ptv3`, `point_transformer_v2` and `litept` expose their documented stage settings; each adapter concatenates lattice XYZ with the six native XYZ/normal features and projects to 512 channels while preserving the sparse row map. |
| Cylinder grouping | `modules.crop.type: native_cylinder` keeps the author's oriented queries and local interaction. `nsample` defaults to 16; `radius` to 0.07 metres; `radius_factors` to `[0.25, 0.5, 0.75, 1.0]`. |
| Cross-radius attention | Optional `fusion_layers` (2), `fusion_heads` (8), `fusion_ffn_dim` (1024), `fusion_dropout` (0.1), `fusion_activation` (`relu`) and `fusion_pre_norm` (`false`) configure the native Transformer. Heads must divide 256. The learned attention reduction across groups stays native. These settings also apply to Graspness's `crop: finegrasp`. |
| Classification objectives | `objectness`, `angle`, `depth` and `score`; retain the native target classes and validity masks. Unlike Graspness, FineGrasp's depth and quality scores are classification tasks. |
| Regression objectives | `graspness`, `view`, `width`; retain the native masks and target units. Width targets are multiplied by 10 by the author objective. |
| Loss coefficients | Defaults: objectness 1, graspness 10, view 100, angle 1, depth 1, score 1, width 10. Override using `loss.weights`; choose registered formulations using `loss.functions`. |
| Augmentation | Native mode flips points, normals and object poses together. Custom rigid transforms, point dropout, cutout, jitter and depth-ray noise retain aligned labels. Jitter/depth noise re-estimate normals on the perturbed observation; latent object grasp annotations stay unchanged. |

Changing cross-radius attention parameters initializes the Transformer under `reuse_unchanged`, while retaining the native attention-reduction weights. Replacing grouping initializes its cylinder layers. The resulting checkpoint must be loaded with the same module configuration and `strict` for inference or resume. Author safetensors and toolbox training checkpoints both require their companion `model.config.json`; keep it beside the selected checkpoint.

An empty checkpoint starts FineGrasp from random weights; a supplied author or toolbox checkpoint initializes the selected architecture. Epoch training defaults to the author's Adam optimizer (weight decay `1e-5`), one-epoch linear warmup, native cosine formula and gradient norm clipping at 10. The cosine denominator uses the full planned update count, matching the pinned source. Registered optimizers and update schedules may replace these defaults. Short training uses a constant learning rate unless a schedule is selected. Empty augmentation preserves native flips for epoch training and disables augmentation for short training.

Set `train_batch_limit: 0` for the complete training split. A positive limit selects consecutive frames from `scene` / `frame`, then shuffles them. Data-loader workers restart at epoch boundaries and receive explicit seeds, so a resumed epoch has the same sampling setup. This differs from the author's persistent-worker lifecycle. Resume restores model, optimizer and schedule state at an epoch boundary; keep the original final `epochs` horizon and data/optimization configuration. The checkpoint's update count must match its completed epochs. GPU operator results are not promised to be bitwise identical after restarting a process.

FineGrasp has no automatic validation loop in this adapter: leave `eval_batch_limit: 0`, generate complete split predictions and run `evaluate` separately. A batch item with no predicted graspable seeds stops with an explicit error; do not use ground-truth seeds to conceal an unusable initialization.

For new adapters and datasets, see [Extending GraspPanda](#extending-grasppanda).

</details>

<details>
<summary>SPGrasp: prompted RGB sequences, Hiera and temporal memory</summary>

## Prompted planar sequences

SPGrasp uses consecutive RGB frames and explicit first-frame point or box prompts. It outputs **planar** grasp centers, opening angles and widths in original image coordinates. Its temporal `memory` slot is separate from point-cloud grouping. It does not produce the 6-DoF GraspNet representation or expose the official AP evaluator.

```bash
./panda weights spgrasp
./panda init --example train-spgrasp -o planar.local.yaml
# Set dataset_root and prepare rectangle and instance labels.
./panda run planar.local.yaml
```

The preset performs short training on consecutive labelled clips, reapplying photometric augmentation each update. `batch_size` selects 1–4 adjacent clips; each contains `frames` images and must stay within the selected scene's 256 frames. The default uses eight frames and one clip. Longer sequences, more objects and larger resolutions increase memory use. Epoch training and optimizer-state resume are not exposed by this adapter. A trained checkpoint can initialize another short run with a fresh optimizer.

| Configuration | Values and meaning |
|---|---|
| `modules.backbone.type` | `upstream` retains defaults; `hiera` exposes the native Base+ encoder parameters below. |
| `backbone.resolution` | 256, 512 (default), 768 or 1024; square letterbox with original aspect ratio. |
| `backbone.drop_path` | 0–0.5; default 0.1. |
| `backbone.trainable_blocks` | -1 trains the complete encoder; 0 freezes it; 1–24 trains only the last N Hiera blocks, keeping the embedding and neck frozen. Frozen stochastic layers remain in evaluation mode. |
| `modules.memory.type` | `upstream` retains defaults; `temporal` exposes native memory parameters. |
| `memory.frames` / `layers` | Spatial memory length 2–16 (default 7); attention layers 1–8 (default 4). The separate native object-pointer history is retained. |
| `memory.heads` / `dropout` | Heads: 1, 2, 4 or 8 (default 1); dropout: 0–0.5 (default 0.1). |
| `trainer.objects` | Maximum objects per clip, 1–8 (default 3), sampled from first-frame instances with positive rectangle targets. |
| `trainer.box_probability` | Probability of a box instead of point initialization, 0–1 (default 0.5). |
| `trainer.correction_clicks` | Iterative correction clicks per selected frame, 0–7 (default 7). |
| `trainer.conditioning_frames` / `correction_frames` | Upper bounds used by native frame sampling, 1–4 (both default 2); conditioning <= correction <= clip length. |
| `loss.weights` | `position` (default 2), `angle`, `width`, `semantic` (default 1 each); nonnegative coefficients on the released loss terms. Custom loss formulations are not registered. |
| `augmentation.mode` | `native`, `none`, or `custom`. Custom accepts `brightness`, `contrast`, `saturation`, `grayscale` in [0,1], and boolean `consistent` for transforms shared across a clip. Geometry and target correspondence remain fixed. |
| `optimizer` / `scheduler` | Native AdamW, image-encoder layer decay 0.9, no bias/LayerNorm weight decay and cosine LR by default. Shared Adam, AdamW, SGD, Lion and update schedules are configurable. An optimizer override replaces native parameter grouping and defaults to cosine with a 0.1 LR floor. A scheduler-only override retains native groups. |
| `planar.width_scale_pixels` | Explicit training width normalization, default 1280; recorded in the checkpoint and reused during prediction. |
| `planar.score_threshold` / `semantic_threshold` | Output thresholds in [0,1], both default 0.5. |
| `planar.max_grasps` / `min_distance` | Up to 1–100 grasps per object (default 10); minimum center spacing in original pixels (default 20). |

The learning-rate field sets the base rate (preset 0.000005); the native encoder rate is 0.6 times that rate before layer decay. Gradient clipping retains the native norm limit 0.1 and computation uses CUDA bfloat16 autocast.

**Prediction:** select your completed run's `checkpoint.pt`, choose **Predict grasps**, and open **Planar sequence**. Load the first RGB frame and click foreground/background points for each object ID, or edit the prompt JSON. IDs identify independently tracked objects; they do not need to match dataset instance labels. Changing the first frame clears the displayed prompts. Boxes and points can be combined:

```json
[{"id": 1, "box": [430, 260, 590, 410]},
 {"id": 2, "points": [[640, 360, 1], [730, 360, 0]]}]
```

Coordinates must lie inside the selected original RGB image. Replace these illustrative coordinates with your own object prompts. The CLI uses the same list in `prompts`. Training generates prompts from its labels and requires `prompts: []`; prediction never reads depth, segmentation or rectangle labels. **Prepare inference from checkpoint** retains the architecture and sequence length in the JSON editor; add your first-frame prompts before running that JSON. Omitted component settings during prediction use the checkpoint architecture; explicit settings must match it.

Outputs are `planar-grasps.json` and a last-frame preview. Each grasp includes object ID, center `[x,y]`, opening angle in radians modulo pi, opening width in original pixels, grasp score and semantic score. The preview draws opening segments; its finger ticks are decorative, since this decoder predicts no rectangle height. Rectangle IoU evaluation and 3D collision checks are not supplied.

**Adaptation details:** the released reader casts continuous angle/width maps to uint8, and its error-point sampler overwrites predictions with targets. The scoped source patch preserves float32 targets and samples corrections from semantic prediction errors. Training always uses point/box prompts instead of the original full grasp-map mask shortcut. RGB and continuous labels share the explicit letterbox transform.

The author width divisor of 100 produces values above one for some GraspNet rectangles, outside the native BCE target range. This adapter uses the explicit pixel scale and rejects widths exceeding it instead of clipping. Every training batch must reach the native width positive-weight cap of ten. Decoding uses `sigmoid(width_logit - log(10)) * width_scale_pixels` to undo weighted BCE's continuous-target shift. The checkpoint stores that calibration; prediction does not estimate it from annotations. The native height-edge angle is converted to the opening-edge convention. Setting correction clicks to zero bypasses the original empty-loop return while retaining the initial prediction. These are documented adaptations, not an identical reproduction of the original training pipeline or paper metrics.

</details>

## PointCNN++ native point convolution

<details>
<summary>Configure encoder stages and residual neighborhoods</summary>

`pointcnnpp` selects the author's native ResUNet point convolution for Baseline, the PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp and FineGrasp. The dense adapter restores every input point before selecting the native 1,024 seeds. Sparse adapters preserve their coordinate map and all input feature channels. `grid_size` controls the point network's geometric scale; it does not replace the method's input voxel size.

Generate the editable configuration with `./panda init --example compose-pointcnnpp`.

`channels` and `depths` have eight entries: four encoder stages from fine to coarse, followed by four decoder stages from coarse to fine. `base_channels` controls the stem width. The three encoder downsampling strides remain 2, and the stem and output kernels remain 5 and 1.

`block_kernel_sizes` accepts 1, 3 or 5, either one value for every residual block or a list whose length equals `sum(depths)`. `block_radius_scalers` has the same scalar/list convention and accepts 0.1–8. It multiplies the neighborhood sphere **volume**, following the author operator; it is not a radius in metres. Both convolutions within each residual block share these settings. Neighborhood indices are rebuilt when their kernel or geometric scale changes. The adapter also clears the author decoder's cached cross-resolution upsampling indices before residual convolutions on the restored point set. This corrects a stale-rulebook issue in the pinned source, so outputs are not numerically identical to that unmodified decoder. Native decoder upsampling keeps its kernel 3 and volume scaler 2.5.

`activation` selects `relu`, `gelu` or `silu`; optional `block_activations` overrides it with one value per residual block. Batch normalization remains native, with configurable momentum and epsilon. `normalize_features: true` applies the author's final L2 feature normalization. Dense methods also support the [seed sampling policies](#network-sampling-policies).

Use `./panda init --example compose-pointcnnpp` for a smaller training configuration. An upstream grasp checkpoint can initialize unchanged heads with `checkpoint_policy: reuse_unchanged`; the replacement backbone requires training. Predict with that run's checkpoint and `checkpoint_policy: strict`.

The installer compiles the pinned author CUDA/CUTLASS operators in the shared Python environment. Native binaries, downloaded source and build intermediates are generated locally. This component adapts the segmentation/registration encoder to grasp detection; it does not provide author-trained GraspNet weights.

[Paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_PointCNN_Performant_Convolution_on_Native_Points_CVPR_2026_paper.pdf) · [Author implementation](https://github.com/robbyant-research/pointelligence)

</details>

## RALA image hierarchy

<details>
<summary>Configure rank-augmented attention and RGB-D stages</summary>

`rala` replaces the HGGD or RegionNormalizedGrasp image encoder with the author's RAVLT feature hierarchy. It retains rank-augmented linear attention, 2D rotary positions, local positional convolutions and native feature merging. The four-channel input remains D,R,G,B with the method's preprocessing and axis convention. A stride-2 stem and channel projections produce the five native anchor/refinement feature lattices; the original grasp heads and local point branch remain in place.

Generate the editable configuration with `./panda init --example compose-rala`.

Stage lists have four entries in fine-to-coarse order. Each stage width must be divisible by four times its head count, preserving 2D rotary position dimensions. `mlp_ratios` (1–8), `layer_scale` and `layer_scale_init` (1e-8–1) accept a scalar or four stage values. Widths range from 16 to 1,024, depths from 1 to 24 and head counts from 1 to 64. `drop_path` sets the maximum of the native linearly increasing block schedule.

Optional `block_attention` supplies exactly `sum(stage_depths)` entries, each `rala` or `softmax`, in stage/block order. Omitted values use RALA in stages 1–2 and softmax in stages 3–4, matching the native default attention layout. For depths `[1,1,2,1]`, `[rala,rala,rala,softmax,softmax]` changes the first block of stage 3 to RALA. Softmax at early, high-resolution stages uses a quadratic attention matrix and can require substantial GPU memory.

`gradient_checkpointing: true` recomputes native residual blocks during backward while preserving stochastic-depth randomness. `freeze_norm_stats: true` freezes running statistics for both BatchNorm and SyncBatchNorm, including the projection/stem; affine parameters remain trainable. This is an explicit toolbox control: the original `norm_eval` checks only BatchNorm2d and does not freeze its SyncBatchNorm layers. `projection_norm` chooses `batch`, `group` or `none` for the output projections; the native encoder normalizations are retained.

Use `./panda init --example compose-rala` for a smaller hybrid configuration. The image encoder initializes randomly; `checkpoint_policy: reuse_unchanged` transfers unchanged parts of the selected grasp method's checkpoint. Train the replacement, then use its saved checkpoint with strict loading. Author image-classification and segmentation checkpoints are not registered as grasp checkpoints.

The installer fetches pinned source in the shared environment. The wrapper loads the segmentation backbone directly, omitting only framework registration and the framework-specific weight loader; a separate MMSegmentation environment is unnecessary.

[Paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_Breaking_the_Low-Rank_Dilemma_of_Linear_Attention_CVPR_2025_paper.pdf) · [Author implementation](https://github.com/qhfan/RALA)

</details>

## PointHR multi-resolution point features

<details>
<summary>Configure parallel resolutions, grouped attention and the decoder</summary>

`pointhr` adapts the author's 2023 PointHR semantic-segmentation hierarchy for Baseline, the PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp and FineGrasp. Four encoder stages retain one, two, three and four parallel resolutions. Every multi-resolution module processes each stream with native grouped-vector attention, then fuses features through learned pooling and unpooling paths. The network restores all input rows before the selected method performs its own seed selection and grasp decoding. Sparse adapters preserve coordinate maps, row correspondence and input feature channels.

Generate the editable configuration with `./panda init --example compose-pointhr`.

`enc_channels` and `enc_groups` specify each stage's finest stream; coarser streams multiply both by powers of two. `enc_depths` counts multi-resolution modules per stage, and `enc_blocks` counts attention blocks per stream within each module. The native stream counts `[1,2,3,4]` remain fixed. `grid_sizes` specifies four strictly increasing pooling sizes in metres; these grasp-scale defaults differ from the author's room-scale segmentation configuration. The method's input voxel size remains separate.

Decoder lists run from the finest original-point resolution to the coarsest decoded resolution. Optional `dec_channels` has four widths; omitted values derive from the stem and final encoder stage, giving `[32,32,64,128]` in the generated template. Unlike the pinned constructor, which overwrites this argument, GraspPanda rebuilds explicitly resized native decoder blocks with the actual encoder skip widths. For example, `[32,48,96,192]` changes the decoder without changing encoder streams. Every width must be divisible by its attention group count. `unpool_backend` selects native cluster maps or inverse-distance 3-neighbor interpolation. Missing interpolation neighbors are masked rather than indexing another scene's final point.

`fusion: sum` retains native fusion; `mean` divides each fused stream by the number of incoming streams. `attn_qkv_bias`, `pe_multiplier` and `pe_bias` control native attention projections and relative-position encoding. `attn_drop_rate` and `drop_path_rate` range from 0 to 0.8. Native BatchNorm uses configurable `bn_momentum` (default 0.1) and `bn_eps` (default 1e-5). `gradient_checkpointing` recomputes attention during backward while preserving running statistics. Dense methods also support [seed sampling policies](#network-sampling-policies).

Start with `./panda init --example compose-pointhr`. The encoder initializes randomly; transfer unchanged grasp heads with `checkpoint_policy: reuse_unchanged`, train, then use the saved checkpoint with strict loading. The installer verifies that the author's native pointops sources match the shared Pointcept operators before reuse; no additional environment is created.

[Paper PDF](https://arxiv.org/pdf/2310.07743) · [Author source](https://github.com/haibo-qiu/PointHR) · [Source terms](THIRD_PARTY.md)

</details>

<details>
<summary>GtG2: candidate preparation, graph components and ensemble training</summary>

## Candidate graph experiments

GtG2 generates grasp candidates with GPG, constructs a local graph for each candidate and averages independently trained graph regressors. Its graph slots are separate from scene-point encoders and cylinder decoders. See [Methods & papers](METHODS.md) for the paper, original implementation and availability.

### Prepare, train and predict

Install the shared runtime using [Installation](INSTALL.md). Download the selected GraspNet training scenes and object models; `dex_models` avoids rebuilding the object geometry cache. Dataset links are in [Data & weights](DOWNLOADS.md). This workflow creates candidate labels from official collision and force-closure primitives; it does not require Graspness maps.

```bash
./panda init --example train-gtg2 -o gtg2.local.yaml
# Set dataset_root and label_root in gtg2.local.yaml.
./panda prepare-gtg2 --config gtg2.local.yaml
./panda run gtg2.local.yaml
```

Preparation inherits the configuration's camera, frame, seed, training scenes and crop geometry. Explicit preparation CLI arguments override those values; keep any geometry or seed override in the training configuration too. `--scoring-batch` controls temporary scoring memory without reducing the candidate set. Preparation can take substantial time and storage; completed frames are reused on subsequent invocations.

For an initial bounded experiment, set `trainer.scenes: [0, 1]`, `trainer.folds: [0, 1]`, `modules.crop.candidate_limit: 256`, `epochs: 2`, `batch_size: 16`, `train_batch_limit: 1` and `eval_batch_limit: 1` **before preparation**. Each model trains on one scene and validates on the other. This is a functional experiment, not the full training protocol. Set candidate and batch limits to `0` for complete data, and omit `trainer.scenes` to select scenes 0–99.

In the browser, select **GtG2 reconstruction → Load preset**, choose the prepared graph root under **Training settings**, and configure the graph under **Compose modules**. Select explicit `gtg_sage` / `gtg_gatv2` and `grasp_graph` types before entering component parameters. **Method training stages** controls scene folds and sampling. Preparation currently uses the CLI command above.

After training, choose **Prepare inference from checkpoint** in **Runs & results** and run the generated JSON. It keeps the graph configuration and selects every trained ensemble member's best validation checkpoint. For CLI use, copy the training configuration, set `action: infer`, `split: test_seen`, `scene: 100`, the chosen `frame`, and `checkpoint` to the generated `checkpoint.pt`; remove `trainer`, `loss`, `augmentation`, `optimizer` and `scheduler` overrides.

Inference permits a different `candidate_limit` and `gpg_threads`; other graph and encoder settings must match the checkpoint. Set `candidate_limit: 0` to score all eligible candidates even when the checkpoint used bounded training preparation. `batch_size` controls graph inference batches. Predictions use the standard GraspNet format and can enter the shared [official evaluation workflow](USAGE.md#experiment-artifacts).

### Configure the graph

| Slot / setting | Choices and defaults |
|---|---|
| `backbone: gtg_sage` | `hidden: 64`, `layers: 3`, `aggregation: max` (`mean` / `sum` supported) |
| `backbone: gtg_gatv2` | `hidden: 64`, `layers: 3`, `heads: 4`, `dropout: 0`, `add_self_loops: true`, `share_weights: false`; hidden width must divide evenly across heads |
| `crop: grasp_graph` | `k: 5`, `max_points: 70` per region, `include_outside: true`, `encoding: binary` (XYZ + one flag); `onehot` uses XYZ + two type flags |
| Eligibility | `min_inside_train: 50`, `min_inside_infer: 70`; minimum counts and point cap must exceed `k` |
| Local geometry | `bound_size: 0.03`, `gripper_depth: 0.06`, `gripper_height: 0.03`, `voxel_size: 0.005`, `plane_threshold: 0.002`; distances in metres |
| Candidate generation | `dual_cloud: true`, `gpg_samples: 1000000`, `gpg_threads: 4`, `max_width: 0.1`, `candidate_limit: 0` |
| Candidate NMS | `nms_translation: 0.005` metres, `nms_rotation_degrees: 1`; either threshold at zero disables suppression |
| Training pose variants | `train_depths: [0.03, 0.04, 0.05]`, `train_width_offsets: [0.02, 0.04]`; zero width offset is always included |

The global scene-point `num_points` field is unused; retain its default. The graph's `crop.voxel_size` governs candidate preprocessing, while global `voxel_size` controls optional final collision filtering. `collision_thresh: 0` disables that final filter. All candidate generation uses `official_gt_workspace`, including dataset segmentation and camera poses.

Raw inside/outside points, candidate identities and scores are stored as numeric arrays. Cache keys include input content hashes, preparation geometry, seed and source versions. Changes to graph width, depth, feature encoding, `k`, sampling cap or outside-feature use reuse the same raw cache. Geometry changes require preparation again. Modified or corrupted cached arrays are rejected.

GATv2 uses the [PyG 2.6.1 implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html) of [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491), ICLR 2022. It is an experimental encoder replacement requiring training.

### Objectives, augmentation and resume

`loss.functions.score` supports `upstream` / `mse`, `l1`, `smooth_l1`, `huber` and `charbonnier`, with the shared [regression parameters](REFERENCE.md#loss-formulations). `loss.weights.score` defaults to `1` and must remain positive. Targets are `1.2 - friction` for valid grasps, `-1` for collisions and `-0.5` for other invalid candidates.

Empty augmentation settings preserve the released model's batch-wide random half-turn about the gripper approach axis. `mode: none` disables it. Custom augmentation runs before graph construction, preserves the quality target and never changes validation inputs:

```yaml
augmentation:
  mode: custom
  half_turn_probability: 0.5
  point_dropout: 0.1
loss:
  functions:
    score: {type: huber, delta: 0.2}
optimizer: {type: adamw, weight_decay: 0.01}
scheduler: {type: cosine, min_lr_ratio: 0.1}
```

The default optimizer is Adam with the configuration's learning rate. AdamW, SGD and Lion, plus constant/cosine/multistep schedules, use the shared controls. Schedules count updates separately for each ensemble member. A singleton final training batch joins the preceding batch so batch normalization remains valid without dropping a graph.

`trainer.folds` selects distinct residues 0–9. A fold holds out scenes where `scene % 10 == fold`. Defaults select five folds and all 100 training scenes. `sampling: balanced` keeps all nonnegative samples and samples up to `negative_per_class` from each negative class every `refresh_epochs`; `sampling: all` uses every prepared training graph. Validation uses all eligible held-out graphs unless explicitly batch-limited.

Epoch checkpoints include every member's model, best validation state, optimizer, schedule and random states. Set `train_checkpoint_mode: resume` and select a checkpoint under `training/`. Keep data, graph, fold, loss, augmentation and optimization settings unchanged. With a configured schedule, retain the original final epoch horizon. A checkpoint saved before validation resumes that pending validation before further training. CUDA runs are not promised to reproduce identical subsequent parameter trajectories.

### Reconstruction choices

The adapter retains the released GNN's position encoder and transformation block. It resolves inconsistent graph/model feature dimensions with explicit four-feature binary encoding, includes the surrounding points, corrects edge conversion and aligns each regression prediction with one scalar target. Missing preparation/inference helpers are reconstructed around the available geometry code and official evaluator primitives.

The five-member, 500-epoch preset and refreshed negative sampling follow the paper's described protocol; they are not implemented by the released short training script. Raw and inpainted proposals use one declared geometry configuration, frame-specific camera poses and widths capped at 0.1 m. Deterministic score ties replace the source's zero-threshold quadratic NMS. These choices and configurable alternatives constitute a toolbox reconstruction; they do not establish the paper's reported accuracy.

</details>

<details>
<summary>PointRWKV: released-code recurrence, local graph and point decoder</summary>

## PointRWKV released-code hierarchy

`pointrwkv_released` uses the [PointRWKV author implementation](https://github.com/hithqd/PointRWKV), associated with [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32353). It combines bidirectional matrix-state recurrence with a local graph branch. Three native feature-propagation stages restore the original input points. The category-conditioned segmentation head is replaced by a grasp-feature projection; grasp inference does not require shape-category labels.

Select it for Baseline, the PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp or FineGrasp. Dense adapters return original-input seed indices and 256-channel features. Sparse adapters preserve the coordinate map and all input rows, and return the method's native feature width. Camera XYZ remains separate from input attributes; Graspness/EconomicGrasp attributes and FineGrasp normals enter both patch encoding and the original-point decoder skip.

| Control | Default and meaning |
|---|---|
| `stage_channels`, `depths`, `stage_heads` | `[384,384,384]`, `[4,4,4]`, `[8,8,8]`; encoder widths, block counts and heads, fine to coarse |
| `num_points`, `group_sizes` | `[2048,1024,512]`, `[32,32,32]`; independently sampled centers and KNN patch sizes at each stage |
| `k_neighbors`, `graph_iterations` | `[16,8,8]`, `[3,3,3]`; local graph size and learned stabilization iterations |
| `ffn_ratios` | `[4,4,4]`; channel-mixing expansion at each stage |
| `patch_channels` | `[128,256,512]`; widths of the native patch-encoding CNN |
| `decoder_channels`, `decoder_depths` | `[384,384,384]`, `[2,2,2]`; coarse-to-fine propagation output widths and CNN depths, ending at original points |
| `drop`, `drop_path_rate` | `0`, `0.1`; channel-mixing dropout and maximum stochastic-depth probability |
| `recurrence_backend`, `chunk_size` | `parallel`, `32`; exact chunk evaluation of the released recurrence, or the original token loop with `native` |
| `gradient_checkpointing` | `false`; recompute encoder blocks during backward to reduce retained activations |

Stage widths must divide into four BQE channel quarters and complete attention heads. Center counts must be nonincreasing, local graph neighbors must fit their stage, and patch neighborhoods must fit every input scene. Decoder controls are ordered coarse to original-point resolution. The default architecture can require substantial GPU memory; configure widths, center counts and checkpointing for your experiments.

For independent block settings, use `block_heads`, `block_neighbors`, `block_graph_iterations`, `block_ffn_ratios`, `block_drop` and `block_drop_path`. Each supplied list must contain exactly `sum(depths)` entries, ordered by stage from fine to coarse, then by block within the stage. It overrides the corresponding stage or shared default; omitted lists preserve existing behavior, including the original linear stochastic-depth schedule. Head divisibility and neighbor limits are checked against each block's effective settings.

For example, `depths: [2, 1, 2]` with `block_neighbors: [8, 16, 8, 4, 8]` uses different local graphs within the first and last stages. `block_heads: [4, 6, 8, 4, 8]` is compatible with `stage_channels: [48, 64, 96]`. The [generated composition](USAGE.md#configuration-examples) (`./panda init --example compose-pointrwkv`) includes a complete example. Changes to per-block settings belong in the saved experiment configuration; use the same settings when strictly reloading its checkpoint.

The `parallel` recurrence evaluates the same state updates and gradients with direct products inside each chunk, including exact zero decay. Smaller chunks reduce intermediate memory; larger chunks expose more parallel work. This acceleration does not replace the original FPS or change KNN neighborhoods. The complete released implementation uses dense neighbor search; no linear-complexity claim is made for the whole network.

**Released-code semantics:** source stages independently sample the original cloud. BQE shifts tokens; attention uses sigmoid gates and GroupNorm; channel mixing uses SiLU; the local graph uses KNN. These differ from parts of the paper's radius-graph, hierarchical and gating descriptions. The selector deliberately identifies the code variant. FPS begins at a random point and token order affects recurrence, so changing the input ordering is not guaranteed to preserve predictions.

No compatible pretrained grasp checkpoint is registered for this replacement. Use `checkpoint_policy: reuse_unchanged` to initialize the replacement while retaining compatible grasp-head weights, then train it. Use `strict` when loading the resulting checkpoint. The installer fetches pinned source into the shared runtime; this adapter does not redistribute the author repository or grant it an additional license.

</details>

<details>
<summary>Swin3D: sparse shifted windows, stage controls and decoder</summary>

## Swin3D sparse window hierarchy

`swin3d` adapts the native [2023 Swin3D hierarchy](https://arxiv.org/pdf/2304.06906) for Baseline, the PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp and FineGrasp. It combines sparse convolutions, alternating regular/shifted window attention with contextual relative position tables, and an interpolating decoder. The final pointwise projection supplies grasp features to the selected method's seed selection and grasp heads.

Generate an editable composition with `./panda init --example compose-swin3d`. The backbone initializes randomly. Use `checkpoint_policy: reuse_unchanged` to transfer unchanged grasp heads, train the composition, then reload its saved configuration and checkpoint with strict loading. The author's Structured3D RGB segmentation weights are not registered for these XYZ/normal grasp inputs.

| Controls | Meaning and defaults |
|---|---|
| `channels`, `depths`, `heads` | One to five stages. Defaults: `[48,96,192,384,384]`, `[2,4,9,4,4]`, `[6,6,12,24,24]`. Each active stage has at least one block. |
| `window_sizes`, `quant_sizes` | Per-stage lattice window widths and relative-position quantization. Defaults: `[5,7,7,7,7]` and `[4,4,4,4,4]`. |
| `grid_size` | Input voxel width in metres; default `0.005`. Multiple input rows in one voxel are mean-aggregated, and outputs return to every original row. |
| `stem_transformer` | Default `true`. A residual stem needs at least two stages and `depths[0]: 0`. |
| `strides`, `downsample`, `knn_neighbors` | One entry per fine-to-coarse transition. Default strides `[3,2,2,2]`, native `knn` pooling with 16 neighbors. `grid` selects native grid max pooling; keep the unused neighbor entry at 16. |
| `decoder_depths`, `up_neighbors` | One entry per decoder target level, listed **fine to coarse**. Defaults: one attention block and three interpolation neighbors. Depth zero retains interpolation and removes attention at that target level. |
| `mlp_ratios`, `qkv_bias` | Scalar or one value per stage; defaults `4.0` and `true`. Applies to encoder and decoder blocks at that resolution. |
| `projection_dropout`, `mlp_dropout`, `activation` | Scalar/stage dropout defaults to zero; MLP activation is `gelu`, `relu` or `silu`. |
| `drop_path_rate`, `decoder_drop_path` | Default encoder schedule increases from zero to `0.3`; default decoder rate is `0.1`. |
| `block_heads`, `block_mlp_ratios`, `block_drop_path`, `block_projection_dropout`, `block_mlp_dropout` | Optional full vectors overriding the corresponding stage/shared controls. Order: encoder fine to coarse, then decoder coarse to fine; blocks within each stage follow execution order. |
| `gradient_checkpointing` | Recompute attention/MLP blocks during backward to reduce activation memory; default `false`. |
| `bn_eps`, `bn_momentum`, `norm_eps` | BatchNorm epsilon/momentum and LayerNorm epsilon; defaults `1e-5`, `0.1`, `1e-5`. |
| `rpe_features` | Default `xyz`. FineGrasp with native normals enabled additionally supports `xyz_normals`. |
| `seed_sampling` | Shared seed policy for the three dense adapters; sparse methods retain their own learned seed selection. |

Stage lists must match the selected number of stages; omitted stage defaults are truncated to that length. Transition lists have one fewer entry. Full block vectors contain `sum(depths) + sum(decoder_depths)` entries. Stage and block heads must be even and divide their stage width into 8, 16 or 32 channels per head. MLP ratios must yield integral hidden widths. Unknown parameters and incompatible dimensions are rejected; the author's unused attention-dropout field is not exposed.

KNN and interpolation neighborhoods accept 1–64 points, and each source scene at that resolution must contain at least that many voxels. A smaller cloud requires a suitable hierarchy or neighborhood configuration. Missing neighbors are not synthesized. The native feature network runs in float32; whole-network mixed precision is not enabled by this adapter.

Dense inputs supply camera XYZ. Sparse inputs preserve the method's lattice and feature channels; fine-grained normals come from FineGrasp's actual normal features. `xyz_normals` retains unused RGB storage columns required by the author's layout, but neither reads those columns for relative encoding nor supplies invented RGB features. Native representative selection uses all supplied coordinate attributes, as in the original implementation.

The installer builds the [pinned author source](https://github.com/microsoft/Swin3D) in the shared environment. Compatibility changes honor the active CUDA stream/device, validate native KNN inputs, bound relative-table indices, cover shared-memory indexing, unpack saved attention tensors once for gradient checkpointing, and resolve equal-distance representative ties by coordinate order. Voxel means accumulate at higher precision before returning float32 features. Sparse row ordering can change stochastic dropout assignments across independently constructed coordinate maps; a seed does not promise bitwise training replay.

[Source terms](THIRD_PARTY.md) · [Installation](INSTALL.md)

</details>


<details>
<summary>SP2T: local windows and spatial proxy attention</summary>

## SP2T sparse proxy hierarchy

`sp2t` uses the [SP²T author implementation](https://github.com/WallelWan/SP2T) ([ICCV 2025 paper](https://arxiv.org/pdf/2412.11540)) as a configurable point encoder for Baseline, the PointNet2 port, Scale-Balanced-Grasp, Graspness, EconomicGrasp and FineGrasp. Its native encoder/decoder combines sparse convolutions, serialized local attention and a spatial proxy stream with vertex associations. `./panda init --example compose-sp2t` generates a compact starting configuration.

The replacement initializes randomly. Select `reuse_unchanged` to transfer unchanged grasp heads, train the composition, then use its saved configuration and checkpoint with `strict` loading. Author segmentation weights linked in the upstream model zoo are not registered grasp checkpoints. Runtime checks do not establish GraspNet benchmark accuracy.

| Controls | Meaning and defaults |
|---|---|
| `enc_channels`, `enc_depths`, `enc_num_head`, `enc_patch_size` | One to five stages; defaults `[32,64,128,256,512]`, `[2,2,2,6,2]`, `[2,4,8,16,32]`, and windows of 1024 points. |
| `dec_channels`, `dec_depths`, `dec_num_head`, `dec_patch_size` | One fewer stage, listed fine to coarse; defaults `[64,64,128,256]`, `[2,2,2,2]`, `[4,4,8,16]`, and windows of 1024. Every configured stage contains at least one block. |
| `stride`, `pooling` | Per-transition strides of 2, 4 or 8; default all 2. Native hierarchy reduction: `max` (default), `mean`, `min` or `sum`. |
| `grid_size`, `serialization_depth`, `order` | Voxel width in metres (`0.005`), fixed lattice bit depth (`16`), and distinct orders from `z`, `z-trans`, `hilbert`, `hilbert-trans` (default all four). |
| `mlp_ratio`, `qkv_bias`, `pre_norm`, `shuffle_orders` | Defaults `4`, `true`, `true`, `true`. Disable order shuffling for deterministic evaluation comparisons. |
| `drop_path`, `attn_drop`, `proj_drop`, `enable_checkpoint` | Native scheduled drop path defaults to `0.3`, dropout to zero and activation checkpointing to `true`. |
| `block_mlp_ratios`, `block_patch_sizes`, `block_drop_path`, `block_attn_drop`, `block_proj_drop`, `block_checkpoint` | Optional vectors in encoder execution order followed by decoder execution order. Length is `sum(enc_depths) + sum(dec_depths)`. Checkpointing also accepts a scalar. |
| `enable_flash`, `enable_rpe`, `upcast_attention`, `upcast_softmax` | Flash local attention defaults to `true`; the other flags default to `false`. Local table RPE and upcasting require Flash to be disabled. Flash head widths must be divisible by eight and at most 256. |
| `proxy_start_stage`, `proxy_end_stage`, `proxy_local_attention` | Inclusive proxy stage interval, default stage 1 through the coarsest stage (stage 0 for a one-stage model). Local attention runs alongside proxies by default; `false` ablates it in active proxy stages. |
| `proxy_operator` | Native `attention` (default), `pooling` or `trb_conv`. Plain pooling does not consume relative bias; its unused bias table is omitted. |
| `proxy_initializer` | `square` searches a spatial grid with `proxy_target: 160`, `proxy_search_range: [0,1]` and `proxy_search_iterations: 10`. The target is approximate. `fixed_grid` uses `proxy_grid_shape: [4,4,4]`; `fixed_size` uses `proxy_cell_size: 0.1` metres. Supply only parameters for the selected initializer. |
| `proxy_query_reduction`, `proxy_projection`, `proxy_same_kv`, `proxy_self_attention`, `proxy_projection_norm`, `proxy_similarity_scale` | Attention query aggregation (`none`, `mean`, `min`, `max`), projection and shared K/V controls. Defaults: `none`, `true`, `true`, `false`, `false`, `1`. Self attention and projection normalization require projections. |
| `proxy_norm`, `proxy_pe_layers`, `proxy_pe_temperature` | Cross-attention norm placement (`pre` or `post`), positional embedding layers (`2`) and temperature (`10`). |
| `proxy_bias`, `proxy_bias_scale`, `proxy_bias_table_size`, `proxy_bias_split` | Native table bias defaults: enabled, scale `2.5`, size `16`, separate map/reduce biases. |
| `proxy_reuse_features`, `proxy_reuse_bias`, `proxy_decoder_reuse`, `proxy_mask_empty` | Feature/bias reuse defaults to `true`; decoder stage reuse and empty-proxy masking default to `false`. Decoder association reuse requires shared proxy geometry (`proxy_reuse_features: true`); bias reuse also requires matching encoder/decoder heads at the same stage. Empty-proxy masking requires the attention fuser. |
| `proxy_fuser` | `attention` uses `proxy_fuser_ffn: true`, `proxy_fuser_rpe: true`, RPE scale `0.4` and table size `8`. `se` instead uses `proxy_se_pool: mean` (`min`/`max` available) and `proxy_se_layers: 2`. |
| `seed_sampling` | Shared seed policy for dense adapters. Sparse methods retain their own learned seed selection. |

Voxel means accumulate in float64 and return float32 features to native operators. Outputs map back to every input row. Dense adapters retain original-input seed indices; sparse adapters retain their exact coordinate manager/map and actual feature channels, including FineGrasp normals. Fixed-grid proxy layouts require nonzero spatial extent on every axis wherever proxy geometry is initialized; use `square` or `fixed_size` for degenerate geometry.

The shared installer prepares an isolated, hash-verified namespace from pinned source. Runtime corrections cover scene slicing, single-scene proxy fusion/masking, self-attention logits, query residuals, checkpoint BatchNorm buffers, window-specific caches and unused decoder bias parameters. Forward-scoped caches prevent state leaking between models. A declared serialization depth and per-scene PyTorch attention windows keep evaluation geometry independent of neighboring scenes; training BatchNorm and stochastic layers still have their normal batch behavior. Native sparse convolutions supply weight gradients in training mode. Optional upstream MMCV FPS/KNN and Warp/Taichi paths are not selectable. Product query aggregation is omitted because large association groups can overflow.

[Source terms](THIRD_PARTY.md) · [Installation](INSTALL.md)

</details>


<details>
<summary>MambaVision: convolution, selective scans and window attention</summary>

## MambaVision hybrid image hierarchy

[MambaVision (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf) combines convolutional stages with Mamba and self-attention blocks. HGGD and RegionNormalizedGrasp accept `backbone: mambavision`. The [pinned author implementation](https://github.com/NVlabs/MambaVision/tree/7860a506b2eb844eaaae676f08461ce8c3c26f43) supplies the blocks; the toolbox adds the RGB-D grasp interface. Source and author weights use **NVIDIA non-commercial research terms**, described in [Third-party terms](THIRD_PARTY.md).

```bash
./panda init --example compose-mambavision -o mambavision.local.yaml
# Set dataset_root and prepare the selected method's training labels.
./panda component-weights mambavision_tiny
./panda run mambavision.local.yaml
```

The template trains the replacement from ImageNet initialization while retaining unchanged native grasp modules. In the UI, choose **mambavision** in **Compose modules**, use **reuse_unchanged**, and select a training operation. The pretrained-encoder button uses the same verified downloader. After training, load the saved grasp checkpoint with **strict** and the same module configuration. Strict inference and resume do not fetch or reapply ImageNet initialization.

### Inputs and features

The native `[B,4,640,360]` input stores D,R,G,B with width before height. The adapter restores RGB to height/width order, applies author ImageNet normalization, and obtains pre-downsample features from all four stages. Features return to native axis order and fuse with learned depth projections. Symmetrically padded odd image convolutions keep the pixel-zero origin; the stride-2 stem and stride-4/8/16/32 projections yield the exact native lattice, including the 23- and 12-cell boundary dimensions. The grasp decoder, local point network, camera model and target construction retain their native contracts.

This is an RGB encoder with learned depth fusion. ImageNet weights do not initialize the depth projections, stride-2 stem or grasp-specific projections. Output features are taken before pooled classification normalization; that normalization and the classifier are omitted. The original image is not resized to a square classification crop.

### Variants and configuration

`variant` selects `tiny` (T, default), `tiny2` (T2), `small` (S), `base` (B), `large` (L), or `large2` (L2). The author 1K Safetensors weights are pinned by revision, size and SHA256. Larger variants require more GPU memory and download space. The full parameter schema appears under **Available component parameters**.

| Parameters | Meaning |
|---|---|
| `embed_dim`, `stem_dim`, `stage_depths` | First-stage width, stem width and four stage depths; later widths double at each stage. Native depths and widths follow the selected variant. |
| `stage_heads`, `window_sizes` | Two values for stages 3 and 4; stages 1 and 2 contain convolutional blocks. Native window sizes are `[14,7]`. |
| `block_mixers` | One `mamba` or `attention` per block, stage 3 then stage 4. Each native stage puts `ceil(depth/2)` Mamba blocks before its attention blocks. |
| `block_heads`, `block_qkv_bias`, `block_qk_norm` | Attention configuration; heads must divide the block width. QKV bias defaults to true and QK normalization to false. |
| `block_state_dims`, `block_conv_sizes`, `block_expansions`, `block_dt_ranks` | Mamba state size, sequence convolution size, expansion and time-step rank. Defaults: 8, 3, 1 and `ceil(block_width/16)`. |
| `block_mlp_ratios` | Per-block MLP expansion, default 4. |
| `layer_scale`, `conv_layer_scale` | Optional residual scales in hybrid and convolutional blocks. B/L/L2 use native hybrid layer scale `1e-5`; other variants omit it. Convolutional layer scale is omitted by default. |
| `attention_dropout`, `mlp_dropout`, `drop_path` | Attention dropout, MLP/projection dropout and maximum linearly scheduled stochastic depth. Defaults: 0, 0 and 0.2 for T/T2/S or 0.3 for B/L/L2. |
| `trainable_stages` | Train the last N RGB stages (0–4, default 4). Zero freezes the RGB encoder; four also trains its patch embedding. Depth and grasp projections remain trainable. |
| `gradient_checkpointing`, `freeze_norm_stats` | Recompute hybrid blocks during backward; optionally hold all adapter BatchNorm statistics fixed. Both default to false. |
| `projection_norm` | `batch` (default), `group` or `none` for RGB and depth projections; the native-format stride-2 stem retains BatchNorm. |

Every `block_*` vector follows stages 3 and 4, with length equal to their combined depths. Numeric and boolean block controls also accept a scalar for all blocks. State parameters affect Mamba blocks; head/QKV parameters affect attention blocks. Block heads override stage heads. Unknown settings, invalid vector lengths and incompatible dimensions are rejected before launch.

`pretrained: true` is the default. Structural edits to widths, depths, mixer types, MLPs or state parameters require **`pretrained: false`**. Window sizes, shape-compatible head counts, dropout, freezing and checkpointing can vary with pretrained weights. These are experimental settings, not guarantees of grasp accuracy.

For a fully configurable architecture, start with `./panda init --example compose-mambavision-custom`. For example:

```yaml
modules:
  backbone:
    type: mambavision
    pretrained: false
    embed_dim: 24
    stem_dim: 16
    stage_depths: [2, 1, 3, 2]
    stage_heads: [4, 8]
    window_sizes: [5, 3]
    block_mixers: [mamba, attention, mamba, attention, mamba]
    block_state_dims: [4, 8, 16, 8, 12]
    block_mlp_ratios: [2, 3, 2, 4, 3]
    gradient_checkpointing: true
checkpoint_policy: reuse_unchanged
```

### Runtime and author compatibility

The adapter uses the existing shared Mamba CUDA scan operator through its pinned native autograd interface; it does not install a competing `mamba_ssm` package. Native block definitions are loaded without the image-classification model registry or remote model execution. Pretrained files use Safetensors; the author's larger pickle training archives are not needed.

The HF export names layer-scale parameters `g_1`/`g_2`; these map explicitly to the GitHub implementation's `gamma_1`/`gamma_2` before strict loading. The native scan and convolution branches, including the released time-step bias behavior, are preserved. The adapter corrects one attention behavior: SDPA dropout is zero in evaluation and follows the configured probability in training. Feature geometry and all remaining block operations retain the pinned source behavior.

</details>

<details>
<summary>Extend the toolbox: methods, components and datasets</summary>

## Extending GraspPanda

### Structure

```text
grasppanda/              Configuration, UI and experiment queue
  methods/               Method-specific inference, data and training adapters
  integrations/          Dataset contracts, readiness and evaluation
  modules/               Interchangeable components and architecture options
  training/              Shared trainers, objectives, augmentation and optimization
  resources/             Built-in templates, source pins, weight records and patches
  runtime/               Installation and isolated worker entry points
docs/                    User and extension guides
```

GraspPanda resolves one Python/PyTorch/CUDA environment. Each job runs in its own process to isolate upstream module names and import-time arguments. A SQLite queue owns scheduling, cancellation, timeouts and run records. Original repositories stay pinned in the local `upstream/` directory; adapters and scoped overlays supply compatibility changes.

### Add a method

1. Add its source URL, pinned revision, input protocol and license information to `grasppanda/resources/`.
2. Register checkpoint URLs, exact sizes, hashes and network roles in `checkpoints.json`.
3. Place method-specific adapters and trainers under `grasppanda/methods/`, or add a native recipe. Register actual operations in `grasppanda/config.py`; reference-only papers belong only in [Methods & papers](METHODS.md).
4. Define input readiness in the dataset provider. Write outputs using the official grasp representation and save the experiment manifest.
5. Exercise the relevant inference, loss, gradient, checkpoint reload and cancellation paths locally. Describe the supported operation in [Methods & papers](METHODS.md), with the paper PDF and original implementation.

Keep local data, weights, credentials, logs, development reports and generated configurations out of commits. Add reusable configuration templates to `grasppanda/resources/examples.json`; `./panda init` writes the selected experiment locally. The `.gitignore` resource allowlist must include any new non-Python installation input. Include relevant validation in the pull request description. Source availability alone is insufficient to enable an operation.

### Add a component

Register a choice in `grasppanda/components.py` and implement it under `grasppanda/modules/`. Match coordinates, units, sample indices, feature dimensions, neighborhoods, supervision, loss and decoder semantics. A matching tensor shape alone does not establish interchangeability. See [Compose modules](MODULES.md) for existing contracts and strict checkpoint transfer.

Declare accepted parameters and cross-stage constraints in `grasppanda/module_options.py`; the UI uses this schema for its parameter reference, and sweeps validate each expanded experiment against it. Pin external component sources in `component_sources.lock.json` and add any native build steps to `grasppanda/runtime/build_components.py`. Keep adaptation differences explicit in the component documentation.

Graspness, SBG and HGGD have different proposal, grouping and refinement structures. Replacing a branch may require its labels, losses and decoder as well as its forward method. Verify gradient flow into every replaced component and strict reload of the resulting checkpoint.

### Add a dataset

Register metadata in `grasppanda/datasets.py` and an executable provider under `grasppanda/integrations/`. Supply readers, camera conventions, target readiness, evaluation and compatible method adapters. Separate single-view, fused-view, active and temporal observation protocols. A metadata-only registration cannot run experiments.

### Compatibility

Build native operators against the locked runtime. Version source patches under `grasppanda/resources/patches/`; generate overlays and build products locally under `environments/`. Do not edit downloaded checkouts or silently discard learned parameters to make weights load.

| Component | Compatibility behavior |
|---|---|
| Legacy PyTorch helpers | Use standard-library equivalents for removed `torch._six` aliases. |
| Point/sparse operators | Compile PointNet2, KNN, MinkowskiEngine and PyTorch3D against the shared ABI. |
| HGGD / RNG | Select camera before import; retain native preprocessing and decoding. Remove profiling counters only. |
| GFLA | Adapt launcher imports and weight paths; compile the original C++ NMS. |
| CenterGrasp | Python 3.11 dataclass factories and the MPlib FCL bridge; retain native ICP and collision filtering. |
| ZeroGrasp | Adapt octree object-range arguments for the single-image sample. |
| MotionGrasp | Retain checkpoint tracker dimensions and explicit attention masks. |
| Graspness modern | Restore the deterministic view-lattice buffer; keep learned parameter matching strict. |
| GraspFast | Scope native KNN imports and map checkpoint keys without changing learned tensors. |
| ASGrasp | Fetch the pinned GSNet submodule and retain the author's RGB/stereo input path. |

Source pins and the Python lock make dependencies traceable. System compilers and hardware still affect native builds; a port does not by itself establish numerical equivalence to the historical author environment.

</details>
