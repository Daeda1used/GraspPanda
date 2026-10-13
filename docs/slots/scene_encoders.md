# Scene encoders: overview

*Words such as seed, contract, preset, adapter, label set and point painting are explained in the [slot index](index.md#words-used-on-these-pages).*

The slot `model.backbone` of the seed-based methods (GSNet, EconomicGrasp, FineGrasp,
GraspNet-baseline, Scale-Balanced-Grasp). It has more options than any other slot, so its
documentation is split by family. This page says what all of them share and where each
option is explained; the [slot index](index.md) has the class of every option.

**No grasp accuracy is claimed for any encoder that is not the one a ported method ships
with.** Each option was composed and probed on real frames under all five methods; that
shows it builds, trains for 20 iterations and predicts.

## The task, and input and output

The scene encoder is the first learned stage: it turns one depth-camera view of a cluttered
table (15 000 to 20 000 points, about 5 mm apart) into one feature vector per point.
Everything after it reads those features. Its receptive field and its resolution decide
what a seed knows about its surroundings before the grouper takes a crop.

| | |
|---|---|
| input | a `PointSet` (`coords (N, 3)` in metres in the camera frame, all clouds of the batch concatenated with batch offsets) and per-point features `(N, in_channels)`: constant ones, coordinates and normals, or nothing, as the method feeds |
| output `dense` | `PointFeatures` with one row per input point |
| output `subsampled` | `PointFeatures` on a sub-sample of the points, the same number for every cloud of the batch; those points become the seeds |
| `out_channels` | the width of the features, injected into proposal, view stage and grouper |

Three contracts tie an encoder to its neighbours, and `grasppanda compose` resolves each
of them or refuses with the fix:

1. **Where the features live.** A proposal requires `dense` or `subsampled` features.
   When the encoder returns the other kind, compose wraps it in the adapter
   `FeaturePropagation` (sub-sampled to dense) or `FurthestPointSubsample` (dense to
   sub-sampled) and says so. On the measured page such a cell reads *runs in* that adapter.
2. **The point features.** Compose sets the encoder's `in_channels` to what the method feeds.
3. **The width.** An encoder declares the width it returns for its options
   (`Backbone.output_channels`) and a grouper the width it needs; the config check compares
   them without building anything. Compose sets `out_channels` where the encoder has that
   option and otherwise, under Scale-Balanced-Grasp, puts the projected grouper in place
   ([Grouper](grouper.md), section 5).

No contract checks the unit of length. Most of these networks were published for rooms or
street scenes; the presets restate radii and grid sizes for a table top and say in their
header what was derived from what. They are starting points, not tuned values.

## The three pages

| Page | Families | Options (`--set backbone=`) |
|---|---|---|
| [Point-based networks](scene_encoders_point.md) | set abstraction, residual MLPs, graph convolutions, random sampling; released weights for PointNet++ and PointNeXt | `pointnet2-msg`, `pointnet2-msg-dense`, `pointnext`, `pointnext-b`, `pointnext-l`, `pointnext-xl`, `pointmetabase`, `pointmetabase-xl`, `pointvector`, `pointvector-xl`, `assanet`, `pointmlp`, `dela`, `dgcnn`, `dgcnn-static`, `randlanet`, `pyg-pointnet`, `pyg-edgeconv`, `pyg-point-transformer`, `pyg-pointgnn` |
| [Sparse-voxel networks and released weights](scene_encoders_sparse.md) | sparse U-Nets (the MinkUNet family, SpUNet), and released weights for them and for the serialised transformers PTv3 and LitePT | `minkunet18a`, `minkunet34c`, `minkunet34c-pointcontrast`, `minkunet18a-openscene`, `spunet`, `spunet-gaussiancross`, `ptv3`, `ptv3-scannet200`, `ptv3-ppt`, `ptv3-encoder`, `ptv3-sonata`, `ptv3-sonata-small`, `litept`, `litept-structured3d` |
| [Transformers, state-space models, pretrained 3-D encoders](scene_encoders_transformer.md) | point transformers, a plain volume transformer, octree attention, point-patch tokens (ViT, Mamba, Uni3D) | `point-transformer-v1`, `point-transformer-v2`, `volt`, `octformer`, `point-vit`, `point-mamba`, `uni3d` |
| [Scene encoders of the model zoo](../model_zoo_backbones.md) | the encoders the ported methods ship with and those ported earlier, with the evidence against their reference implementations | `sparse-unet14`, `pointnet2`, `pointnet2-dense`, `pointnet2-transformer`, `oacnns`, `ptv3-utonia` |

`grasppanda compose --list backbone` prints all of them with one line each;
`grasppanda components --slot backbones` adds the contract, the source, the status and the
optional library of every class.

## How to choose

- **To change as little as possible**, take an encoder of the kind the method already has:
  `dense` for GSNet, EconomicGrasp and FineGrasp, `subsampled` for GraspNet-baseline and
  Scale-Balanced-Grasp. No adapter is inserted then.
- **To start from released weights**, read section 3.2 of the sparse page and section 5 of
  the point and transformer pages: they list which files load into which class, their
  licences (several are non-commercial) and what does not load (usually the first layer,
  which reads the input).
- **On a small GPU**, the tables of the pages give the peak memory of the probes; the
  point-based encoders with graph convolutions over the whole scene are the most
  expensive.
- **Optional libraries**: the sparse networks need a `spconv` extra, the PyTorch Geometric
  bridge `torch_geometric`, OctFormer `ocnn`, PointMamba `mamba-ssm` (CUDA only), Uni3D
  `timm`. A config that names a missing library passes the config check and stops when
  the model is built, with the `pip install` line.

## Try one

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=pointmetabase \
    -o gsnet_pointmetabase.yaml
grasppanda probe gsnet_pointmetabase.yaml
```

Under Scale-Balanced-Grasp a probe ends as `wired` (a synthetic scene, exit status 3) unless
it is given the reference's width histogram, which cannot be built from the eight frames of a
probe; the `runs` of that method in the tables of these pages were measured with

```bash
grasppanda probe <config> --opts data.train.dataset.scale_statistic.source=legacy_import \
    data.train.dataset.scale_statistic.reference_sha256=<SHA-256 of the reference's file>
```

(`configs/scale_balanced_grasp/README.md` says how to import the file once.)

The measured verdict of every encoder under every method, with the adapter compose
inserted and the numbers of the probe, is in
[Measured combinations](../advanced_guides/measured_combinations.md#scene-encoders); the
[explorer](../explorer/index.md) shows the same per selection.
