# Compose modules

Module replacement is an explicit contract, not a shape-only switch. Supported choices preserve coordinates, units, sample indices, label assignment and decoder semantics. The rest of each method remains authoritative.

| Configure | Reference |
|---|---|
| Select compatible parts | [Slots and parameters](#component-selection-and-parameters) · [EconomicGrasp](#economicgrasp-components) |
| Point encoders | [PointVector](#pointvector-encoder) · [PointMetaBase](#pointmetabase-encoder) · [PointMamba](#pointmamba-encoder) · [PCM](#point-cloud-mamba-hierarchy) · [OctFormer](#octformer-hierarchy) · [Point Transformer](#point-transformer-encoder) |
| Local grouping and interaction | [Cylindrical ResLFE](#residual-local-aggregation-in-cylinders) · [Seed interaction](#grouped-seed-interaction) · [FineGrasp](#finegrasp-training-and-composition) |
| Image encoders | [RGB-D encoders](#rgb-d-image-encoders) · [VMamba](#vmamba-state-space-image-features) · [DINO](#pretrained-dino-image-features) |
| Training | [Losses and augmentation](#training-controls) · [Optimization](#optimizers-and-schedules) · [Checkpoints](#checkpoint-policies) |
| Run experiments | [Short training](#short-training) · [Epoch training](#train-a-composed-model-across-epochs) · [HGGD](#hggd-epoch-training) · [GtG2](GTG2.md) |

## Component selection and parameters

A component can be a name (`backbone: pointnet`) or a mapping containing `type` and its parameters. Both forms serialize into the experiment and checkpoint. Unknown parameters are rejected.

| Method | Slot | Choices |
|---|---|---|
| Baseline / PointNet2 port | `backbone` | `upstream`, `pointnet`, `pointnext`, `pointvector`, `pointmeta`, `pointmlp`, `pointmamba`, `pointcloud_mamba`, `octformer`, `sonata_ptv3` |
| Baseline / PointNet2 port | `crop` | `upstream`, `multiscale`, `cylinder`, `reslfe_cylinder` |
| Graspness | `backbone` | `upstream`, `pointnet`, `sparse_unet18`, `sonata_ptv3` |
| Graspness | `crop` | `upstream`, `cylinder`, `finegrasp`, `reslfe_cylinder` |
| EconomicGrasp | `backbone` | `upstream`, `native_tdunet`, `pointnet`, `sonata_ptv3` |
| EconomicGrasp | `crop` | `upstream`, `native_cylinder`, `cylinder`, `reslfe_cylinder`; optional seed interaction |
| FineGrasp | `backbone` | `upstream`, `sonata_ptv3` |
| FineGrasp | `crop` | `upstream`, `native_cylinder` |
| HGGD / RegionNormalizedGrasp | `backbone` | `upstream`, `native_resnet`, `convnextv2`, `repvit`, `mobilenetv4`, `dinov2`, `dinov3`, `vmamba` |
| GtG2 | `backbone` / `crop` | `upstream`, `gtg_sage`, `gtg_gatv2` / `upstream`, `grasp_graph`; [graph settings and training](GTG2.md) |

The baseline encoder returns original-input seed indices and 256-channel features. Graspness encoders retain sparse coordinate correspondence and 512-channel features. Crop adapters retain the native decoder's depth/view semantics.

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

In the UI, select the component names under **Compose modules**, then enter parameters keyed by slot in **Component parameters by slot**. Do not repeat `type` in this parameter editor; the selector supplies it. Full YAML/JSON configurations use the mapping form above.

## EconomicGrasp components

EconomicGrasp retains its native sparse quantization, graspable-point selection, economic labels, interactive grasp head and decoder. Its [author implementation](https://github.com/iSEE-Laboratory/EconomicGrasp) is described in the [ECCV 2024 paper](https://arxiv.org/pdf/2407.08366). Generate an editable composition with `./panda init --example compose-economicgrasp`.

The backbone receives constant input features on the quantized camera lattice and returns 512-channel features in the same sparse row order. `pointnet` uses the toolbox's sparse PointNet-style encoder; `sonata_ptv3` uses the [PTv3 adapter and its stage settings](#point-transformer-encoder). Both reconstruct camera positions from lattice coordinates and voxel size and preserve the map used by `quantize2original`.

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

EconomicGrasp short training also supports [loss formulations and coefficients](#loss-formulations), [aligned point augmentation](#point-augmentation), Adam/AdamW/SGD/Lion and update-based schedules. Score remains a six-class objective; angle/depth retain their native invalid classes and validity masks. Width targets remain scaled by ten. Augmentation transforms object poses with observations, updates per-point labels through one sampling map and regenerates `coordinates_for_voxel`; native flips retain float32 observations. Full epoch training/resume is not exposed by this adapter.

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

Generate a complete EconomicGrasp example with `./panda init --example compose-seed-interaction`. In the browser, expand **Compose modules**, retain the desired grouping and enter these fields under `crop` in **Component parameters**. The parameter reference lists the accepted values. Sweep paths such as `modules.crop.interaction_sigma` and `modules.crop.interaction_heads` to compare settings.

| Parameter | Default | Meaning |
|---|---|---|
| `seed_interaction` | `none` | `gaussian` enables the interaction; omit interaction parameters when disabled |
| `interaction_heads` | `4` | Attention heads; 1–32 and must divide 256 |
| `interaction_sigma` | `0.05` | Gaussian distance scale in camera-frame metres; 0.001–1 |
| `interaction_layers` | `1` | Independently parameterized interaction blocks; 1–4 |
| `interaction_dropout` | `0.1` | Attention-probability dropout during training; 0–0.8 |

The implementation loads the pinned `GraspGNN` attention classes from [GCF-GraphGrasp](https://github.com/qzsrh/GCF-GraphGrasp/blob/8ab374104f1559005f292f163bcf3b17e8211575/models/modules_economicgrasp.py). It retains the native Q/K/V/output projections, two residual LayerNorm operations and learned distance-bias scale. The bias is `-distance_squared / (2 * sigma_squared) * bias_scale`; each block initializes its scale to one and leaves its sign unconstrained, as in the source. This is dense attention over grasp seeds, not a k-nearest graph or a new cylinder query.

The adapter preserves seed order and 256-channel output features. It processes scenes independently and, for Baseline's `[B,256,N,4]` output, processes each depth bin independently. Inputs and computation use float32. Attention memory grows quadratically with the number of seeds and linearly with batch, depth bins, heads and layers; changing the scene point count does not necessarily change the method's seed count.

When retaining the original grouping, `reuse_unchanged` loads its existing weights and initializes only the added interaction layers. Train the composition before inference, then load its checkpoint with `strict` and the same module settings. If you also replace the grouping or encoder, those replacements follow their usual initialization policy. Baseline and Graspness support this composition in epoch training and resume; EconomicGrasp and the PointNet2 port use their registered short-training operations.

This component uses each selected method's own data, supervision, seed selection and decoder. It does not reproduce the GCF fork's SAM/patch-feature preprocessing or its changed angle/depth decoding. GCF's README cites EconomicGrasp; a separate GCF publication and pretrained checkpoint have not been verified. The full-method entry remains in [Methods & papers](METHODS.md).

## OctFormer hierarchy

`octformer` adapts the [author implementation](https://github.com/octree-nn/octformer) of [OctFormer (SIGGRAPH / TOG 2023 PDF)](https://arxiv.org/pdf/2305.03045) to Baseline and its PointNet2 port. It retains the native octree convolution stem, alternating regular/dilated window attention, positional convolution, MLP blocks and multi-scale segmentation decoder. Camera XYZ replaces the ScanNet input features; the decoder produces 256-channel features sampled at the original grasp seed indices. Start with [`compose-octformer`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-octformer`).

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

Each scene is normalized isotropically into the octree cube while its feature channels retain camera XYZ in meters. Querying the decoder preserves original input rows, including duplicate voxel assignments. Windows are padded independently per scene: changing a preceding scene's node count cannot shift another scene's window origin. Native batch normalization still shares training statistics across the batch. This is a grasp adaptation with no pretrained OctFormer grasp checkpoint; initialize compatible unchanged grasp layers, then train the new backbone. Use `train` for epoch training and `strict` checkpoint loading for resume/inference. No benchmark AP or convergence is implied by short training.

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

Vector blocks retain the author's angle-based scalar-to-vector transforms, channel-grouped projection, ReLU/batch normalization and sum reduction. Increasing `local_nsample` changes both support and the scale of that sum. A stage depth of one omits its additional vector blocks; choosing one for every stage is an explicit no-vector ablation. Start with [`compose-pointvector`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-pointvector`). Baseline also supports this encoder in epoch training; the PointNet2 port uses its registered short-training and inference operations.

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

A stage depth of one omits its extra local blocks, so local reduction, expansion and residual settings have no effect in that stage. Selecting one in all stages gives an abstraction-only ablation. Use `reuse_unchanged` with the original grasp checkpoint, train the replacement, then use `strict` for the resulting checkpoint. Baseline supports epoch training and resume; the PointNet2 port supports its registered short-training and inference operations. Start with [`compose-pointmeta`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-pointmeta`), which also replaces cylinder processing with ResLFE.

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

The adapter keeps the native block constructor initialization, including the learned timestep bias initialization and depth-scaled output projections. It does not apply the classification trainer's outer scratch initializer, load classification weights or provide pretrained grasp weights. Start with [`compose-pointmamba`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-pointmamba`), use `reuse_unchanged` to retain the baseline's other modules, train the replacement, and use `strict` to reload the resulting grasp checkpoint. Width, depth, state size, grouping and fusion are saved with the experiment and can be swept through dotted configuration paths.

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

Use [`compose-pcm`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-pcm`), or select `pointcloud_mamba` and enter parameters in the UI. `./panda install` downloads the pinned source and builds its two CUDA extensions with isolated names in the shared runtime; its older causal-convolution ABI coexists with `pointmamba`. Source terms are described in [Third-party notices](THIRD_PARTY.md).

Training requires **`batch_size >= 2`** because native global-context BatchNorm operates on one pooled feature per sample. Short training uses a fixed batch of consecutive labelled frames within the selected scene; epoch training drops an incomplete final batch. Inference accepts one frame. Keep the exact component configuration when reloading weights; use `reuse_unchanged` for initial component replacement, train it, then use `strict` with the resulting checkpoint. Epoch `resume` restores the saved composition and optimizer contract. Run full training and held-out evaluation before interpreting grasp quality.

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

The native operator supports float32 and float16; bfloat16 is rejected. CUDA launches use the current PyTorch stream and tensor device. More samples increase the within-cylinder distance matrix quadratically; start with [`compose-reslfe`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-reslfe`). Source terms are described in [Third-party notices](THIRD_PARTY.md).

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

Stage widths must be divisible by their head counts and by eight. Encoder and decoder windows can differ: the adapter refreshes native padding/relative-position caches when the window changes. Larger widths, depths, point counts and windows increase memory use. Adam, AdamW, SGD and Lion support this encoder; Muon is excluded because its current routing assumes dense convolution layouts. See the [`compose-ptv3`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-ptv3`) for a smaller trainable configuration.

## RGB-D image encoders

HGGD and RNG use native `D,R,G,B` tensors shaped `[B,4,640,360]`, with the author's transposed spatial axes. Image adapters retain that convention and provide five feature maps with strides 2/4/8/16/32 and channels 8/16/32/64/128. Their anchor heads, losses, coordinate conversion, local refinement and collision filtering stay native. HGGD's detached features remain detached at the original point-fusion boundary.

The modern encoders use the implementation in the locked [timm library](https://github.com/huggingface/pytorch-image-models). Paper and author-code references:

| Family | Paper | Author implementation |
|---|---|---|
| ConvNeXt V2 | [CVPR 2023 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf) | [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) |
| RepViT | [CVPR 2024 PDF](https://arxiv.org/pdf/2307.09283) | [RepViT](https://github.com/THU-MIG/RepViT) |
| MobileNetV4 | [ECCV 2024 PDF](https://arxiv.org/pdf/2404.10518) | [TensorFlow Models](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py) |

ConvNeXt/RepViT/MobileNet inputs are zero-padded only at the high ends of the spatial axes. ConvNeXt patch-center offsets are resampled onto the native lattice with bilinear interpolation and border extension; RepViT/MobileNet use their centered, odd-kernel lattices. Encoders without stride-2 outputs gain a native-sized stem. Learned projections supply the expected channels; `projection_norm` accepts batch/group/none. These are explicit grasp adaptations, not reproductions of image-classification results.

Selecting a new encoder initializes it and its projections from scratch. `reuse_unchanged` retains only the original anchor heads and complete local network; use `strict` for subsequent checkpoint inference. No ImageNet weights are downloaded implicitly. See [`compose-hggd`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-hggd`).

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

Start with [`compose-vmamba`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-vmamba`). Smaller stage widths/depths make configuration sweeps less expensive. HGGD also supports these components in [epoch training](#hggd-epoch-training). A larger state, stage or point count increases memory use; scan choices are architectural experiments, not an accuracy ranking.

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

Start with [`compose-dino`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-dino`). Use `checkpoint_policy: reuse_unchanged` with the method's grasp checkpoint to initialize its unchanged heads and local branch. Missing encoder weights are downloaded and checksum-verified during initialization; the UI's **Pretrained image encoders** panel or `./panda component-weights dinov3_small` can prepare them ahead of time. This uses the same shared runtime.

After training, retain the module configuration and load the resulting grasp checkpoint with `strict`. Strict loading does not download or reapply DINO initialization, so it preserves the trained encoder and works without the original pretrained-weight file. Initialization provenance records the selected weight ID, immutable source URL and SHA256. Changing the source registry while a job waits causes the job to stop rather than use different initialization.

Image pretraining does not train the new grasp feature projections. Run grasp training before interpreting predictions; use the existing RNG anchor warmup when a new feature adapter produces no labeled local proposals. Weight sources and licenses are in [Data & weights](DOWNLOADS.md#pretrained-image-components).

## Training controls

Baseline, its PointNet2 port, Graspness, FineGrasp and EconomicGrasp accept `loss` and `augmentation` overrides in supported `train_check` or `train` actions. HGGD exposes these controls in `train_check` and `train`; RNG supports `train_check`; see [RGB-D training controls](#rgb-d-training-controls). Other methods retain their own supervision contracts. Start with [`train-controls`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example train-controls`).

In the browser, expand **Training & evaluation settings → Choose loss formulations**, select classification and regression families, then **Apply loss choices**. This writes the per-term formulations into **Loss configuration**, preserving your coefficients. Edit each term there to use different parameters. The configuration editor and sweeps use the same schema.

```yaml
loss:
  weights: {width: 0.4}
  functions:
    objectness: {type: asl, gamma_pos: 0, gamma_neg: 4, label_smoothing: 0.1}
    width: {type: huber, delta: 0.5}
augmentation:
  mode: custom
  rotation_axis: x
  rotation_degrees: 15
  translation: 0.01
  point_dropout: 0.2
  cutout_fraction: 0.1
  depth_noise_std: 0.001
  depth_noise_clip: 0.003
```

### Loss formulations

Weights are absolute coefficients. Unspecified terms retain their native coefficients and formulations. Omit `functions`, or select `upstream`, to retain the native objective; identical coefficient overrides preserve its values and gradients.

| Method | Classification terms | Regression terms | Native coefficients |
|---|---|---|---|
| Baseline / PointNet2 port | `objectness`, `angle` | `view`, `score`, `width`, `tolerance` | Objectness/view: 1; score/angle/width/tolerance: 0.2 |
| Graspness | `objectness` | `graspness`, `view`, `score`, `width` | Objectness: 1; graspness: 10; view: 100; score: 15; width: 10 |
| FineGrasp | `objectness`, `angle`, `depth`, `score` | `graspness`, `view`, `width` | See the FineGrasp training section below |
| EconomicGrasp | `objectness`, `angle`, `depth`, `score` | `graspness`, `view`, `width` | Objectness/angle/depth/score: 1; graspness/width: 10; view: 100 |

Each `functions` value accepts a name or `{type: NAME, ...}`. The following parameters have bounded, finite values; omitted parameters use the defaults shown.

| Formulation | Parameters: default [range] | Behavior |
|---|---|---|
| `cross_entropy` | `label_smoothing`: 0 [0, 0.5] | Softmax cross entropy with optional uniform smoothing |
| `focal` | `gamma`: 2 [0, 8]; optional `alpha` [0, 1] | Softmax focal loss; alpha weights foreground versus background and is accepted only for binary objectness |
| `poly1` | `epsilon`: 1 [-1, 10] | Cross entropy + epsilon × (1 − target-class probability); epsilon 0 recovers cross entropy |
| `asl` | `gamma_pos`: 0 [0, 8]; `gamma_neg`: 4 [0, 8]; `label_smoothing`: 0.1 [0, 0.5] | Single-label softmax asymmetric loss, with the author's defaults |
| `l1` / `mse` | None | Absolute / squared normalized error |
| `smooth_l1` | `beta`: 1 [0, 10] | Quadratic-to-linear transition at beta; beta 0 is L1 |
| `huber` | `delta`: 1 [0.000001, 10] | Huber transition at delta; its scale differs from Smooth L1 when delta is not 1 |
| `charbonnier` | `epsilon`: 0.001 [0.000001, 1] | sqrt(error² + epsilon²) − epsilon |

Softmax focal loss adapts [Focal Loss (ICCV 2017)](https://arxiv.org/pdf/1708.02002) to the existing class heads. Regression transitions follow the [PyTorch Smooth L1](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html) and [Huber](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html) definitions.

The point adapters retain native positive masks, angle-label argmax/gather and target units. Baseline width and tolerance errors are divided by the native maximum width/tolerance; its grasp terms divide by the float32 valid count plus 1e-6. Graspness width targets are multiplied by 10; width loss uses positive quality labels only. Other substituted point terms retain native valid-item means. Empty masks produce a gradient-connected zero for **substituted point** terms; unmodified upstream terms keep their original behavior. A non-finite training objective stops the run.

[PolyLoss (ICLR 2022)](https://arxiv.org/pdf/2204.12511) uses the [author's Poly-1 formulation](https://waymo.com/research/polyloss-a-polynomial-expansion-perspective-of-classification-loss-functions/). [ASL (ICCV 2021)](https://arxiv.org/pdf/2009.14119) follows the [author's single-label softmax variant](https://github.com/Alibaba-MIIL/ASL), checked against the locked timm implementation. Probability complements and fractional powers use numerically stable evaluation at saturated logits. These are classification-head adaptations; no grasp accuracy improvement is implied.

[Varifocal Loss](https://github.com/hyz-xmaster/VarifocalNet) assumes quality logits decoded through sigmoid. The current grasp-quality heads emit raw regression scores, so it requires an explicit head/decoder adaptation before becoming a selectable loss.

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

HGGD samples augmentation for each dataset item. RNG samples once before creating its fixed training frame, optional anchor warmup and local patches; it retains its bounded native-objective training protocol. Start with the loss and augmentation section of [`compose-hggd`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example compose-hggd`). Sweeps can vary paths such as `loss.functions.local_orientation`, `loss.weights.local_offset` and `augmentation.depth_noise_std`.

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

These are point-observation controls. Image crops, nonrigid warps, scaling and scene mixing need their own camera, grasp-pose, width and collision-label transformations; matching tensor sizes does not make their supervision interchangeable.

Loss and augmentation settings are training-only. Resume requires saved configuration metadata for training overrides and the same objective, augmentation, sampled frame range, seed and loader worker count. **Prepare inference from checkpoint** retains architecture settings and clears training-only options automatically. To sweep a loss parameter, use a structured formulation in the base configuration and vary, for example, `loss.functions.objectness.epsilon`; augmentation uses paths such as `augmentation.point_dropout`.

## Optimizers and schedules

Baseline, its PointNet2 port, Graspness, FineGrasp, EconomicGrasp, SBG, HGGD and RNG accept optimization overrides in their registered training actions. Empty mappings retain the method's optimizer and schedule. Select the optimizer/schedule under **Training & evaluation settings** in the UI, then enter parameters without `type`; full experiment files include `type` as below.

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

The base learning rate always comes from `learning_rate`. Explicit optimizer overrides default to zero weight decay. Adam/AdamW/SGD use PyTorch; Lion and Muon use the locked timm implementations. [Lion](https://github.com/google/automl/tree/master/lion) ([NeurIPS 2023 paper](https://arxiv.org/pdf/2302.06675)) typically needs a smaller learning rate than AdamW. [Muon](https://github.com/KellerJordan/Muon) uses timm's matrix/convolution routing and AdamW fallback, with flattened convolution kernels. It is registered only for dense baseline/PointNet2 compositions and HGGD/RNG models; PTv3 uses spconv and is excluded; sparse-kernel parameter layouts need a separate routing contract. These choices do not imply improved grasp accuracy.

| Schedule | Parameters and behavior |
|---|---|
| `constant` | Optional `warmup_steps`, then the base learning rate |
| `cosine` | Optional `warmup_steps` and `min_lr_ratio` (default 0), decaying over the configured update horizon |
| `multistep` | Increasing, unique `milestones` and `gamma` (default 0.1) |

Schedule units are **completed optimizer updates**, including for epoch training. A milestone of 2 changes the third update's rate. Warmup starts at `base_lr / warmup_steps`; cosine reaches its floor after the final update. A short run's horizon is `training_steps`, plus RNG's optional `proposal_warmup_steps`; epoch training uses updates per epoch times `epochs`, including configured batch limits. HGGD divides the loader length by `trainer.accumulation_steps`, rounding up for the final partial group. Changing the loader or final epoch horizon on resume is rejected. Model, optimizer and schedule states are restored and checked before the next update.

Optimizer and scheduler settings are training-only; clear them for manually authored inference configurations. UI checkpoint reuse does this automatically. Sweep paths such as `optimizer.type`, `optimizer.weight_decay` and `scheduler.min_lr_ratio` are supported, subject to each selected implementation's validation.

## Checkpoint policies

- `strict`: every checkpoint key and parameter shape must match. Use this for original models and a saved checkpoint of the same composition.
- `reuse_unchanged`: discard checkpoint parameters only inside explicitly replaced modules, retain their constructor initialization, and load all remaining parameters strictly. The result records exactly which keys were initialized or discarded.

Selecting a different encoder and silently accepting all missing keys would conceal implementation mistakes. GraspPanda rejects mismatches outside the chosen slots.

## Short training

Download the baseline weights and prepare its training labels as described in [downloads](DOWNLOADS.md). Copy this configuration to `compose.local.yaml` and set your dataset path:

```yaml
dataset: graspnet1b
method: graspnet_baseline
action: train_check
dataset_root: /data/GraspNet-1B
checkpoint: checkpoints/graspnet_baseline/checkpoint-rs.tar
camera: realsense
split: train
scene: 0
frame: 0
num_points: 4096
workspace: official_gt_workspace
modules:
  backbone: pointnet
  crop: multiscale
checkpoint_policy: reuse_unchanged
training_steps: 3
learning_rate: 0.0001
```

```bash
./panda run compose.local.yaml --runs-dir outputs/cli-runs
```

By default, the check repeats one labelled frame without augmentation and computes the native loss. Registered overrides apply the configured objective and augmentation. It requires finite losses/gradients and nonzero parameter updates. It also verifies updates in every replaced component. This is a bounded optimization diagnostic, not a multi-epoch training schedule or accuracy result. HGGD, GraNet and fusion use batch size 2; FineGrasp and PCM compositions use the configured `batch_size`; PCM requires at least 2 distinct consecutive frames. Other point methods use batch size 1. RNG uses anchor batch 2 and up to 48 local patches. CenterGrasp checks its SGDF and RGB objectives separately. Outside FineGrasp and PCM compositions, the general `batch_size` field applies to native epoch training. `epochs` always applies to the full `train` action.

The output directory contains `checkpoint.pt`, `result.json`, the configuration, provenance and logs. Results include loss components, input-label hashes, transfer details and updates. The UI plots total loss and can export the run.

## Use the saved model

Choose a completed training check in **Runs & results**, click **Prepare inference from checkpoint**, review the generated JSON in **Experiments**, then click **Run edited JSON**. This retains the same module choices and switches to `strict` checkpoint loading for a test frame.

For CLI use, change `action` to `infer`, `checkpoint` to the saved file, `checkpoint_policy` to `strict`, `split` to `test_seen` and `scene` to `100`. Keep `modules` unchanged. Short training does not establish quality, especially when an encoder was newly initialized; an empty graspable-point set is reported explicitly.

## Train a composed model across epochs

Baseline, Graspness, FineGrasp and HGGD accept their registered module choices in `action: train`. SBG also exposes its native epoch trainer. Native dataset loops remain in use; omitted controls retain the author's augmentation, objective, optimizer and schedule. Object/collision labels load through bounded caches instead of eagerly occupying memory for every scene.

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

## RNG proposal initialization

A newly initialized image encoder may produce proposals with no local grasp labels. For RNG short training, set `proposal_warmup_steps` to train the anchor on its native heatmap targets before preparing its own local patches. These updates are additional to `training_steps`; a custom learning-rate schedule spans both phases. The default is `0`. Warmup uses the selected optimizer and real targets, with no teacher model or replacement labels. Its loss is shown separately as **Anchor warmup**. The required duration depends on initialization, learning rate and scene; a positive proposal set is checked before local training.

The browser exposes this setting under **Training & evaluation settings** for RNG. Checkpoint inference clears this training-only setting. This initialization procedure is a toolbox option; the unreleased RNG full training schedule is not reproduced.

## HGGD epoch training

Generate the [`train-hggd` example](../GraspNet-1B/README.md#configuration-examples) with `./panda init --example train-hggd`, set your dataset/checkpoint paths in `experiment.local.yaml`, then use `./panda run experiment.local.yaml`. The example fine-tunes author weights at a conservative learning rate; it is not a reproduction of the paper's training hyperparameters. Prepare the camera-specific [HGGD targets](DOWNLOADS.md#method-specific-preprocessing) for training scenes 0000-0099 and validation scene 0100. `workspace: native_demo` retains the native RGB-D geometry and target generation. Batch size must be at least 2.

Set `trainer` in YAML/JSON or expand **Training & evaluation settings → Method training stages** in the UI:

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

FineGrasp exposes its own native training adapter, alongside the FineGrasp grouping replacement available to Graspness. Use [`train-finegrasp`](../GraspNet-1B/README.md#configuration-examples) (`./panda init --example train-finegrasp`) with the economic labels described in [Data & weights](DOWNLOADS.md#finegrasp).

| Part | Configuration and contract |
|---|---|
| Point encoder | `modules.backbone: upstream` keeps the released MinkUNet. `sonata_ptv3` exposes the same per-stage PTv3 settings documented above; its adapter concatenates lattice XYZ with the six native XYZ/normal features and restores the sparse row map before the 512-channel projection. |
| Cylinder grouping | `modules.crop.type: native_cylinder` keeps the author's oriented queries and local interaction. `nsample` defaults to 16; `radius` to 0.07 metres; `radius_factors` to `[0.25, 0.5, 0.75, 1.0]`. |
| Cross-radius attention | Optional `fusion_layers` (2), `fusion_heads` (8), `fusion_ffn_dim` (1024), `fusion_dropout` (0.1), `fusion_activation` (`relu`) and `fusion_pre_norm` (`false`) configure the native Transformer. Heads must divide 256. The learned attention reduction across groups stays native. These settings also apply to Graspness's `crop: finegrasp`. |
| Classification objectives | `objectness`, `angle`, `depth` and `score`; retain the native target classes and validity masks. Unlike Graspness, FineGrasp's depth and quality scores are classification tasks. |
| Regression objectives | `graspness`, `view`, `width`; retain the native masks and target units. Width targets are multiplied by 10 by the author objective. |
| Loss coefficients | Defaults: objectness 1, graspness 10, view 100, angle 1, depth 1, score 1, width 10. Override using `loss.weights`; choose registered formulations using `loss.functions`. |
| Augmentation | Native mode flips points, normals and object poses together. Custom rigid transforms, point dropout, cutout, jitter and depth-ray noise retain aligned labels. Jitter/depth noise re-estimate normals on the perturbed observation; latent object grasp annotations stay unchanged. |

Changing cross-radius attention parameters initializes the Transformer under `reuse_unchanged`, while retaining the native attention-reduction weights. Replacing grouping initializes its cylinder layers. The resulting checkpoint must be loaded with the same module configuration and `strict` for inference or resume. Author safetensors and toolbox training checkpoints both require their companion `model.config.json`; keep it beside the selected checkpoint.

An empty checkpoint starts FineGrasp from random weights; a supplied author or toolbox checkpoint initializes the selected architecture. Epoch training defaults to the author's Adam optimizer (weight decay `1e-5`), one-epoch linear warmup, native cosine formula and gradient norm clipping at 10. The cosine denominator uses the full planned update count, matching the pinned source. Registered optimizers and update schedules may replace these defaults. Short training uses a constant learning rate unless a schedule is selected. Empty augmentation preserves native flips for epoch training and disables augmentation for short training.

Set `train_batch_limit: 0` for the complete training split. A positive limit selects consecutive frames from `scene` / `frame`, then shuffles them. Data-loader workers restart at epoch boundaries and receive explicit seeds, so a resumed epoch has the same sampling setup. This differs from the author's persistent-worker lifecycle. Resume restores model, optimizer and schedule state at an epoch boundary; keep the original final `epochs` horizon and data/optimization configuration. The checkpoint's update count must match its completed epochs. GPU operator results are not promised to be bitwise identical after restarting a process.

FineGrasp has no automatic validation loop in this adapter: leave `eval_batch_limit: 0`, generate complete split predictions and run `evaluate` separately. Short runs and bounded epoch checks do not establish full-training convergence or benchmark AP. A batch item with no predicted graspable seeds stops with an explicit error; do not use ground-truth seeds to conceal an unusable initialization.

For new adapters and datasets, see [Extending GraspPanda](EXTENDING.md).
