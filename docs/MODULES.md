# Compose modules

Start from a method preset, replace a compatible part, then train and compare the resulting model. GraspPanda checks coordinates, units, sample indices and supervision as well as tensor shapes. Single-view, fused-view and sequence methods retain their own input protocols.

## First composition

```bash
./panda init --example compose-baseline -o composition.local.yaml
# Set dataset_root and checkpoint, and prepare the method's training labels.
./panda run composition.local.yaml
```

The template selects a point encoder and local grouping. In the browser, **Load preset**, expand **Compose modules**, and choose the same parts. Enter additional settings in **Component parameters by slot**; **Available component parameters** lists accepted values for your selection.

In YAML/JSON, each slot accepts a name or a mapping:

```yaml
modules:
  backbone: {type: pointnet, local_channels: [64, 128]}
  crop: {type: cylinder, hidden_channels: [64, 128], pooling: mean}
checkpoint_policy: reuse_unchanged
```

`reuse_unchanged` retains the original checkpoint's unchanged layers and initializes replacements. Train those replacements before prediction; use `strict` to reload the resulting checkpoint with the same module settings. In the browser's parameter editor, omit `type` because the selectors supply it.

## Component selection and parameters

A component can be a name (`backbone: pointnet`) or a mapping containing `type` and its parameters. Both forms serialize into the experiment and checkpoint. Unknown parameters are rejected.

<details>
<summary>Compatible components by method</summary>

| Method | Slot | Choices |
|---|---|---|
| Baseline / PointNet2 port | `backbone` | `upstream`, `pointnet`, `pointnext`, `pointvector`, `pointmeta`, `pointmlp`, `pointmamba`, `pointcloud_mamba`, `octformer`, `sonata_ptv3`, `point_transformer_v2`, `litept`, `pointcnnpp`, `pointhr`, `sp2t`, `swin3d`, `pointrwkv_released`, `flash3d`, `oacnns`, `kpconvx`, `utonia`, `concerto` |
| Scale-Balanced-Grasp | `backbone` | Same replacement point encoders as Baseline, with network sampling controls |
| Scale-Balanced-Grasp | `sampling` | `upstream`, `object_balanced`; independent DSN-based inference sampling |
| Scale-Balanced-Grasp | `crop` | `upstream`, `native_mscq`; [independent branch configuration](REFERENCE.md#scale-balanced-grasp-components) |
| Baseline / PointNet2 port | `crop` | `upstream`, `multiscale`, `cylinder`, `reslfe_cylinder`, `kpconvx_cylinder` |
| Baseline / PointNet2 port / Scale-Balanced-Grasp / Graspness | `head` | `upstream`, `quality_residual`; [quality score mappings and objectives](REFERENCE.md#quality-score-heads) |
| Graspness | `backbone` | `upstream`, `pointnet`, `sparse_unet18`, `sonata_ptv3`, `point_transformer_v2`, `litept`, `pointcnnpp`, `pointhr`, `sp2t`, `swin3d`, `pointrwkv_released`, `flash3d`, `oacnns`, `kpconvx`, `utonia`, `concerto` |
| Graspness | `crop` | `upstream`, `cylinder`, `finegrasp`, `reslfe_cylinder`, `kpconvx_cylinder` |
| EconomicGrasp | `backbone` | `upstream`, `native_tdunet`, `pointnet`, `sonata_ptv3`, `point_transformer_v2`, `litept`, `pointcnnpp`, `pointhr`, `sp2t`, `swin3d`, `pointrwkv_released`, `flash3d`, `oacnns`, `kpconvx`, `utonia`, `concerto` |
| EconomicGrasp | `crop` | `upstream`, `native_cylinder`, `cylinder`, `reslfe_cylinder`, `kpconvx_cylinder`; optional seed interaction |
| EconomicGrasp | `head` | `upstream`, `native_interactive` |
| FineGrasp | `backbone` | `upstream`, `sonata_ptv3`, `point_transformer_v2`, `litept`, `pointcnnpp`, `pointhr`, `sp2t`, `swin3d`, `pointrwkv_released`, `flash3d`, `oacnns`, `kpconvx`, `utonia`, `concerto` |
| FineGrasp | `crop` | `upstream`, `native_cylinder`, `kpconvx_cylinder` |
| HGGD / RegionNormalizedGrasp | `backbone` | `upstream`, `native_resnet`, `convnextv2`, `repvit`, `mobilenetv4`, `dinov2`, `dinov3`, `vmamba`, `rala`, `mambavision` |
| GtG2 | `backbone` / `crop` | `upstream`, `gtg_sage`, `gtg_gatv2` / `upstream`, `grasp_graph`; [graph settings and training](REFERENCE.md#candidate-graph-experiments) |
| SPGrasp | `backbone` / `memory` | `upstream`, `hiera` / `upstream`, `temporal`; [planar sequence settings](REFERENCE.md#prompted-planar-sequences) |

</details>

The baseline encoder returns original-input seed indices and 256-channel features. Graspness encoders retain sparse coordinate correspondence and 512-channel features. Crop adapters retain the native decoder's depth/view semantics.


## Parameters and experiment controls

| Configure | Guide |
|---|---|
| Encoder stages, attention, grouping and sampling | [Component reference](REFERENCE.md) |
| Loss terms, LogitNorm / MbLS / LogitClip and observation augmentation | [Training controls](REFERENCE.md#training-controls) |
| RGB-D smooth-color and random-filter augmentation | [PRIME photometric policy](REFERENCE.md#prime-photometric-augmentation) |
| Optimizers, schedules and checkpoint policy | [Optimization](REFERENCE.md#optimizers-and-schedules) |
| Epoch training and method-specific preparation | [Training reference](REFERENCE.md#train-a-composed-model-across-epochs) · [Data & weights](DOWNLOADS.md) |
| Compare configurations | [Sweeps](USAGE.md#configuration-sweeps) |

Use `./panda init --list` to discover complete templates, then `./panda init --example NAME` to generate one locally. Omitted parameters use component defaults. The reference describes cross-stage constraints, physical units and changes to author implementations; no component choice implies improved accuracy or compatible pretrained grasp weights.
