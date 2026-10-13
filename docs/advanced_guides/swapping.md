# Swapping components

Replacing one stage of a method is one line in a config. This page says which
replacements the toolbox accepts, what each one changes besides that line, and which
swaps look harmless and are not. The tables are generated from the contract rules that
the config checker runs (`python -m tools.make_component_docs`), and every cell is
exercised by `tests/unit/test_swap_matrix.py`, so they describe the code as it is.

Nothing here is a result. A swapped model has to be trained and evaluated before it
means anything; the configs under `configs/experimental/` are starting points with no
AP attached.

## The one-line swap

```bash
grasppanda components --slot backbones          # what exists, with its contract
grasppanda components PTv3                      # options, defaults, what it fits
grasppanda print-config gsnet/gsnet_sunet14_g1b-realsense_10e --check \
    --opts "model.backbone={type: PTv3, in_channels: 3, grid_size: 0.005}"
```

or, in a file,

```yaml
_base_: [../gsnet/gsnet_sunet14_g1b-realsense_10e.yaml]
model:
  backbone: {type: PTv3, in_channels: 3, grid_size: 0.005}   # the options of the old type are dropped
```

A swap the contracts forbid stops at the config check, before a dataset is opened, and
says what to change:

```text
$ grasppanda print-config gsnet/gsnet_sunet14_g1b-realsense_10e --check \
      --opts "model.backbone={type: PointNet2}"
✗ 1 config problem in configs/gsnet/gsnet_sunet14_g1b-realsense_10e.yaml

  model
    GraspnessFPS requires 'dense' scene features, but PointNet2 returns 'subsampled' ones
    (features on a sub-sampled point set). Wrap the backbone in the explicit adapter
    `{type: FeaturePropagation, backbone: {...}}` or choose a proposal that works on
    sub-sampled features.
```

```bash
grasppanda print-config gsnet/gsnet_sunet14_g1b-realsense_10e --check \
    --opts "model.backbone={type: FeaturePropagation, backbone: {type: PointNet2}}"   # ✓ valid
```

## What the checker compares

A seed-based detector is `backbone → proposal → view → (assigner) → grouper → head`.
Each stage declares a contract as class attributes; the rules below compare them. They
belong to the skeleton, so the verdict is the same for `GSNet`, `EconomicGrasp`,
`GraspNetBaseline` and `ScaleBalancedGrasp`, and the same whether the model comes from a
YAML file or from `DETECTORS.build(...)` in Python.

| Rule | Compares | Why it exists |
|---|---|---|
| Where features live | `backbone.output` with `proposal.requires` | Features on every scene point and features on a subsample are different objects. Converting is a learned or a sampling step that changes the model, so it is a component you write down, never an interpolation behind your back. |
| Feature axes | `grouper.out_axes` with `head.in_axes` | A crop with a depth axis and one without feed different heads. An axis is never added or squeezed implicitly. |
| Label set | `assigner.label_format` with `head.label_format` and `view.label_formats` | Dense grids (48 grasps per point and view) and compact labels (one) are different supervision. |
| Fields of the matched labels | `view.needs_matched`, `head.needs_matched` with `assigner.optional_outputs` | A stage that reads an optional field needs the assigner option that produces it. The error names the option. |
| Seed weights | `assigner.optional_outputs["weight"]` with `applies_seed_weights` of view stage and head | An assigner that returns a weight per seed next to a stage that ignores it would train an unweighted model under a weighted name. |
| Seed scores | `seed_scores` of proposal and view stage; `head.required_seed_scores(options)` | One owner per score (`objectness` cannot be predicted and supervised twice), and a score the decoder multiplies in must be predicted by some stage. |
| Depth bins | `len(grouper.hmax)` with `head.num_depth` | One crop per depth bin of the head. |
| One score definition | `score_target`, `max_friction` of view stage and head | Only when both use the definition of `grasppanda.models.heads.friction`: the view target is the best grasp score of a view. |

## Encoders

<!-- generated:encoders:begin -->
| Encoder | `output` | Width | Status | Needs | `EconomicGrasp` | `GraspNetBaseline` | `GSNet` | `ScaleBalancedGrasp` |
|---|---|---|---|---|---|---|---|---|
| `ASSANet` | `dense` | 64 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `DGCNN` | `dense` | 256 (`out_channels`) | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `DeLA` | `dense` | 256 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `LitePT` | `dense` | 72 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `MinkUNet` | `dense` | 96 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `OACNNs` | `dense` | 20 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `OctFormer` | `dense` | 168 (`out_channels`) | `bridge` | `ocnn` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PTv3` | `dense` | 64 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PTv3Sonata` | `dense` | 1232 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PTv3Utonia` | `dense` | 1386 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointMLP` | `dense` | 128 (`out_channels`) | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointMamba` | `subsampled` | 384 (`out_channels`) | `bridge` | `mamba-ssm` | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointMetaBase` | `dense` | 32 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNeXt` | `dense` | 32 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNet2` | `subsampled` | 256 | `weights-verified` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointNet2Dense` | `dense` | 128 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNet2MSG` | `subsampled` | 256 | `tested` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointNet2MSGDense` | `dense` | 128 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNet2Transformer` | `subsampled` | 256 | `weights-verified` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointTransformerV1` | `dense` | 32 (`out_channels`) | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointTransformerV2` | `dense` | 48 (`out_channels`) | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointVector` | `dense` | 32 | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointViT` | `subsampled` | 384 (`out_channels`) | `tested` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PyGPointEncoder` | `dense` | 256 (`out_channels`) | `bridge` | `torch_geometric` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `RandLANet` | `dense` | 32 (`out_channels`) | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `SpUNet` | `dense` | 20 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `SparseUNet14` | `dense` | 512 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `Uni3D` | `subsampled` | once built (`out_channels`) | `bridge` | `timm` | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `Volt` | `dense` | 256 (`out_channels`) | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |

✓: the features of `model.backbone: {type: <Encoder>}` live where the proposal of the method needs them. *in `Adapter`*: the bare swap is refused with a message that names the adapter; `model.backbone: {type: <Adapter>, backbone: {type: <Encoder>}}` passes that rule.

Adapters: `FeaturePropagation` turns any encoder into a `dense` one; `FurthestPointSubsample` turns any encoder into a `subsampled` one.

*Width* is the number of channels the encoder returns with its constructor defaults, as it declares it (`Backbone.output_channels`); `(out_channels)` marks an encoder that has an option of that name to set it. The other stages take whatever width arrives, with these exceptions among the reference models: `ScaleBalancedGrasp` needs **256** channels (`DepthCylinderGroup` adds the seed features to its crop features). An encoder of another width is refused there by the config check, with both numbers and the ways out; `grasppanda compose` takes one of them by itself (the encoder's `out_channels`, or the projected variant of the grouper). *once built*: the width follows from a library model or a layout that only the constructor resolves, so the check builds the encoder once, without released weights.
<!-- generated:encoders:end -->

Every encoder can serve every method, but never silently. The two adapters are the only
bridges between the two kinds of output, and each changes the model in a way worth
knowing:

- `FeaturePropagation` (subsampled → dense) invents features for points the encoder never
  looked at: inverse-distance interpolation over the three nearest kept points, then a
  shared MLP. A graspness head then scores every scene point from features that carry
  the resolution of the subsample (1024 points with the GraspNet-baseline layout).
- `FurthestPointSubsample` (dense → subsampled) decides which points can become grasps.
  It samples the scene coordinates, so the seeds are spread evenly over the cloud, table
  included, where `GraspnessFPS` spreads them over the points predicted graspable. With
  the same sampling rules it keeps exactly the points `PointNet2` ends on, in the same
  order (`tests/unit/test_backbone_subsample.py`): swapping `PointNet2` for a wrapped
  dense encoder changes the features of the seeds and leaves the seeds where they were.

### What an encoder swap changes

The contract says where features live. It does not say what a feature has seen. These
are the constructor defaults, which is what a bare `{type: ...}` gives you:

| Encoder | Unit of length | What sets the receptive field | Seeds under its method | Re-tune first |
|---|---|---|---|---|
| `SparseUNet14` | `voxel_size`, 5 mm | Four stride-2 stages: the coarsest voxel is 16 × the grid, 8 cm at 5 mm | 1024 by `GraspnessFPS` from predicted graspness | — (reference of GSNet and EconomicGrasp) |
| `SpUNet` | `grid_size`, default **2 cm** (ScanNet rooms) | The same four stride-2 stages, with 2-3-4-6 residual blocks where `SparseUNet14` has one: more 3³ convolutions per level, so a feature reaches further in voxels | as above | `grid_size`; `out_channels` (default 20, class scores); `in_channels` (default 6); learning rate (the training check lost most seeds in two iterations at 1e-3) |
| `PTv3` | `grid_size`, default **2 cm** (ScanNet rooms) | Four stride-2 poolings, and attention over patches of 1024 serialised voxels: counted in points, not in metres | as above | `grid_size` (0.005 to keep GSNet's unit); `enc_patch_size` / `dec_patch_size` if clouds are small; learning rate and schedule |
| `PTv3Sonata`, `PTv3Utonia` | `grid_size`, default 2 cm / 1 cm | As `PTv3`, encoder only: dense features are the pooled stages copied back, no learned decoder | as above | everything under *Traps*, items 2 and 3 |
| `LitePT` | `grid_size`, default **2 cm** (ScanNet rooms) | 3³ sparse convolutions in the three finest stages, attention only in the two coarsest (8 × and 16 × the grid). At 5 mm a tabletop frame is a few hundred voxels there, fewer than one patch of 1024: attention is global | as above | `grid_size`; `in_channels` (default 6); `drop_path` (0.3, for long schedules); the 72-channel output |
| `OACNNs` | `grid_size`, default **2 cm** (ScanNet rooms) | Four stride-2 stages of 3³ sparse convolutions, plus adaptive pooling grids counted in cells (`point_grid_size`): the 8 to 16 cells of the first stage are 0.32 to 0.64 m at 2 cm and 0.08 to 0.16 m at 5 mm | as above | `grid_size`; `point_grid_size`; `out_channels` (default 20, class scores); `in_channels` (default 9); learning rate (the training check lost its seeds for three iterations at 1e-3) |
| `PointNeXt` | `radius`, default **0.1 m** doubling per stage (S3DIS rooms) | Ball radii per stage; stride 4 per stage | as above | `radius` (the GraspNet-baseline layout starts at 0.04 m); `in_channels` (default 4) |
| `PointNet2Dense` | `radii`, default 0.1 / 0.2 / 0.4 / 0.8 m (rooms) | Ball radii and the points kept per level (1024 / 256 / 64 / 16) | as above | `npoints`, `radii` |
| `PointNet2` | `radii`, 0.04 / 0.1 / 0.2 / 0.3 m | Ball radii; 2048 / 1024 / 512 / 256 points | the 1024 points of its second level, by furthest point sampling of the scene | — (reference of GraspNet-baseline) |
| `PointNet2Transformer` | `radii`, 0.04 / 0.1 m | Two levels, each with attention over 16 nearest points | as above | — (reference of Scale-Balanced-Grasp) |

What follows from a different receptive field or seed set, whichever encoder causes it:

- **Label statistics move.** The head is trained on the seeds that reach it. GSNet
  normalises score targets with the extrema of the batch (`head.score_target:
  log_friction_batch_norm`); EconomicGrasp supervises only seeds within 5 mm of a label
  point (`assigner.max_match_distance`), so an encoder that shifts where seeds land
  changes how many seeds are supervised at all (`metric/supervised_seeds` in the log);
  Scale-Balanced-Grasp weights seeds by a width histogram of the training labels.
- **The recipe was tuned for the reference encoder.** Learning rate, schedule, batch
  size and point budget in `configs/_base_/schedules/` are the authors' values for their
  encoder. They are a starting point for another one, not a claim.
- **Channel widths are injected, and checked when the model is built.** The proposal,
  view stage and grouper take the encoder's `out_channels`. One stage needs two widths
  to be equal: `DepthCylinderGroup` with `seed_residual: gated` (Scale-Balanced-Grasp)
  adds seed features to crop features. The error names both numbers and the three ways
  out; `grasppanda train` reaches it within seconds, `print-config --check` does not
  build the model.

## Stage by stage

`✓` means the pair passes with default options; a number points to the reason below the
table, which is the first sentence of the error the config check prints.

<!-- generated:pairs:begin -->
#### Encoder output and proposal

| backbone ↓ · proposal → | `GraspnessFPS` | `SampledSeeds` | `ScoredSeeds` | `SubsampledSeeds` |
|---|---|---|---|---|
| `dense` | ✓ | ✓ | ✓ | ✗ (1) |
| `subsampled` | ✗ (2) | ✗ (3) | ✗ (4) | ✓ |

1. SubsampledSeeds requires 'subsampled' scene features, but a dense encoder returns 'dense' ones (one feature per scene point).
2. GraspnessFPS requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).
3. SampledSeeds requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).
4. ScoredSeeds requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).

#### Proposal and view stage

| proposal ↓ · view → | `DirectionViewStage` | `HealpixViewStage` | `IcosphereViewStage` | `LatticeViewStage` | `ObjectnessViewStage` | `RomaRotationViewStage` | `TemplateViewStage` | `TwoVectorViewStage` |
|---|---|---|---|---|---|---|---|---|
| `GraspnessFPS` | ✓ | ✓ | ✓ | ✓ | ✗ (1) | ✓ | ✓ | ✓ |
| `SampledSeeds` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ScoredSeeds` | ✓ | ✓ | ✓ | ✓ | ✗ (2) | ✓ | ✓ | ✓ |
| `SubsampledSeeds` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

1. GraspnessFPS and ObjectnessViewStage both predict and supervise the 'objectness' of a seed; a detector needs exactly one owner for it.
2. ScoredSeeds and ObjectnessViewStage both predict and supervise the 'objectness' of a seed; a detector needs exactly one owner for it.

#### View stage and assigner

| view ↓ · assigner → | `CompactLabelAssigner` | `NearestLabelAssigner` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `ScaleBalancedAssigner` |
|---|---|---|---|---|
| `DirectionViewStage` | ✓ | ✓ | ✓ | ✓ |
| `HealpixViewStage` | ✓ | ✓ | ✓ | ✓ |
| `IcosphereViewStage` | ✓ | ✓ | ✓ | ✓ |
| `LatticeViewStage` | ✓ | ✓ | ✓ | ✓ |
| `ObjectnessViewStage` | ✗ (1) | ✗ (2) | ✓ | ✓ |
| `RomaRotationViewStage` | ✓ | ✓ | ✓ | ✓ |
| `TemplateViewStage` | ✓ | ✓ | ✓ | ✗ (3) |
| `TwoVectorViewStage` | ✓ | ✓ | ✓ | ✓ |

1. ObjectnessViewStage computes its targets from 'dense' matched labels, but CompactLabelAssigner returns 'compact' ones.
2. ObjectnessViewStage reads 'on_object' and 'view_width' of the matched labels, which NearestLabelAssigner returns only with `supervise: object_seeds` and `view_widths: true`.
3. ScaleBalancedAssigner returns a loss weight per seed, but TemplateViewStage does not apply seed weights, so the weighting would silently do nothing there.

#### Assigner and head bundle

| assigner ↓ · head → | `DecoupledGridHead` | `EconomicHead` | `OperationHead` | `SWADHead` |
|---|---|---|---|---|
| `CompactLabelAssigner` | ✗ (1) | ✓ | ✗ (2) | ✗ (3) |
| `NearestLabelAssigner` | ✓ | ✗ (4) | ✓ | ✓ |
| `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | ✓ | ✗ (4) | ✓ | ✓ |
| `ScaleBalancedAssigner` | ✗ (5) | ✗ (6) | ✓ | ✗ (7) |

1. CompactLabelAssigner returns 'compact' matched labels, but DecoupledGridHead is supervised with 'dense' ones.
2. CompactLabelAssigner returns 'compact' matched labels, but OperationHead is supervised with 'dense' ones.
3. CompactLabelAssigner returns 'compact' matched labels, but SWADHead is supervised with 'dense' ones.
4. NearestLabelAssigner returns 'dense' matched labels, but EconomicHead is supervised with 'compact' ones.
5. ScaleBalancedAssigner returns a loss weight per seed, but DecoupledGridHead does not apply seed weights, so the weighting would silently do nothing there.
6. ScaleBalancedAssigner returns 'dense' matched labels, but EconomicHead is supervised with 'compact' ones. ScaleBalancedAssigner returns a loss weight per seed, but EconomicHead does not apply seed weights, so the weighting would silently do nothing there.
7. ScaleBalancedAssigner returns a loss weight per seed, but SWADHead does not apply seed weights, so the weighting would silently do nothing there.

#### Grouper and head bundle

| grouper ↓ · head → | `DecoupledGridHead` | `EconomicHead` | `OperationHead` | `SWADHead` |
|---|---|---|---|---|
| `CylinderGroup` | ✓ | ✓ | ✗ (1) | ✓ |
| `CylinderInteractionGroup` | ✓ | ✓ | ✗ (2) | ✓ |
| `DepthCylinderGroup` | ✗ (3) | ✗ (4) | ✓ | ✗ (5) |
| `DepthCylinderProjectionGroup` | ✗ (6) | ✗ (7) | ✓ | ✗ (8) |
| `DepthEmbedGroup` | ✗ (9) | ✗ (10) | ✓ | ✗ (11) |
| `DepthLocalGroup` | ✗ (12) | ✗ (13) | ✓ | ✗ (14) |
| `DepthPoolGroup` | ✓ | ✓ | ✗ (15) | ✓ |
| `LocalGroup` | ✓ | ✓ | ✗ (16) | ✓ |
| `MultiRadiusGroup` | ✓ | ✓ | ✗ (17) | ✓ |
| `MultiScaleGroup` | ✓ | ✓ | ✗ (18) | ✓ |
| `PyGConvGroup` | ✓ | ✓ | ✗ (19) | ✓ |
| `RoIAwareGroup` | ✓ | ✓ | ✗ (20) | ✓ |
| `RoIGridGroup` | ✓ | ✓ | ✗ (21) | ✓ |
| `VoxelSetAbstractionGroup` | ✓ | ✓ | ✗ (22) | ✓ |

1. CylinderGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
2. CylinderInteractionGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
3. DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
4. DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
5. DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
6. DepthCylinderProjectionGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
7. DepthCylinderProjectionGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
8. DepthCylinderProjectionGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
9. DepthEmbedGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
10. DepthEmbedGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
11. DepthEmbedGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
12. DepthLocalGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
13. DepthLocalGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
14. DepthLocalGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
15. DepthPoolGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
16. LocalGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
17. MultiRadiusGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
18. MultiScaleGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
19. PyGConvGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
20. RoIAwareGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
21. RoIGridGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
22. VoxelSetAbstractionGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').

#### View stage and head bundle

| view ↓ · head → | `DecoupledGridHead` | `EconomicHead` | `OperationHead` | `SWADHead` |
|---|---|---|---|---|
| `DirectionViewStage` | ✓ | ✓ | ✓ | ✓ |
| `HealpixViewStage` | ✓ | ✓ | ✓ | ✓ |
| `IcosphereViewStage` | ✓ | ✓ | ✓ | ✓ |
| `LatticeViewStage` | ✓ | ✓ | ✓ | ✓ |
| `ObjectnessViewStage` | ✓ | ✓ | ✓ | ✓ |
| `RomaRotationViewStage` | ✓ | ✓ | ✓ | ✓ |
| `TemplateViewStage` | ✓ | ✓ | ✓ | ✓ |
| `TwoVectorViewStage` | ✓ | ✓ | ✓ | ✓ |
<!-- generated:pairs:end -->

Reading the tables:

- **Proposal.** `GraspnessFPS` scores every scene point and needs the graspness asset of
  the dataset (`grasppanda prepare graspnet1b --graspness`). `SubsampledSeeds` scores
  nothing: every point the encoder kept is a seed. Without `ObjectnessViewStage` behind
  it, nothing in the model separates object seeds from table seeds, and a grasp is
  decoded for each of them.
- **View stage.** `TemplateViewStage` samples the training view from its own scores
  (`train_selection: multinomial`) and reads either label set. `ObjectnessViewStage`
  takes the arg-max in both modes, predicts objectness with the views and reads two
  optional fields of dense labels.
- **Grouper.** A `seeds`-support crop gathers at most the other seeds of the frame, with
  their learned features. A `scene`-support crop gathers raw coordinates of the whole
  cloud and no learned feature. They are not one crop at two resolutions, and only the
  second has a depth axis.
- **Head bundle.** Prediction layers, target encoder, loss and decoder share one grasp
  parameterisation and move together. `SWADHead` and `OperationHead` read dense label
  grids, `EconomicHead` compact labels, which the dataset must serve
  (`compact_labels:` in the dataset config, `grasppanda prepare graspnet1b
  --compact-labels`); `OperationHead` also needs the tolerance asset.

## Every legal set of stages

<!-- generated:stage-sets:begin -->
The contracts accept **1776 of 14336** combinations of the registered stages (an assigner counts twice when it has options that switch optional fields on). The last column names the reference model that is this set.

| Encoder output | Proposal | View stage | Assigner | Grouper | Head bundle | Reference model of |
|---|---|---|---|---|---|---|
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | `EconomicGrasp` |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | `GSNet` |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `SampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `dense` | `ScoredSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `DirectionViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `HealpixViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `IcosphereViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `LatticeViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | `GraspNetBaseline` |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | `ScaleBalancedGrasp` |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `RomaRotationViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `DepthPoolGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `LocalGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiRadiusGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `PyGConvGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIAwareGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `RoIGridGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `CompactLabelAssigner` | `VoxelSetAbstractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthLocalGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthPoolGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `LocalGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiRadiusGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `PyGConvGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIAwareGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `RoIGridGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `DecoupledGridHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `VoxelSetAbstractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthCylinderProjectionGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthEmbedGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TwoVectorViewStage` | `ScaleBalancedAssigner` | `DepthLocalGroup` | `OperationHead` | — |
<!-- generated:stage-sets:end -->

The detector `type` does not change which sets are legal. It names the method whose
reference behaviour the config claims to follow; a set that is no reference model is a
new model, whatever it is called.

`HGGD` is not in these tables because it is not in this family. It proposes partial
grasps from an image heat-map, has no view stage and no approach-aligned crop, and none
of its stages fills a slot above; putting a seed-based stage into it, or its heat-map
proposal into a seed-based detector, is not a swap the registries can express. Its
parts are options of the detector (`grasppanda components HGGD`). Its "ResNet-34" is the
block layout only, with a base width of 8 and four input channels, trained from scratch:
a pretrained three-channel ResNet-34 is a different network with a different input
(trap 8 of the survey).

## Traps

Swaps that pass every check above, or look as if they should, and are not what they
seem. The numbers are stable identifiers: code comments and error messages refer to a trap by its number.

1. **An object-level pretrained encoder as scene encoder** (Point-BERT, Point-MAE, ReCon,
   Uni3D; trap 1). They turn one normalised object of a few thousand points into patch
   tokens or a single vector. A scene of 15 000 metric points is neither their input nor
   their output: there is no per-point feature for a graspness head, and normalising the
   scene destroys the metres the cylinder radii and depth bins are written in. None is
   registered as a backbone for that reason.
2. **"PTv3 with Sonata, Concerto or Utonia weights" as "PTv3"** (trap 2). The released
   self-supervised checkpoints are encoder-only, have other depths and widths, expect
   nine input channels (coordinates, colour, normal; zeros where missing) and were
   trained on a 2 cm or rescaled grid. They load into `PTv3Sonata` / `PTv3Utonia`, not
   into `PTv3`; loading one into the wrong type fails on the key report. Changing only a
   weight path changes all of the above.
3. **Keeping the grid when changing the encoder, or changing it freely** (trap 3). The
   grid is the encoder's unit of length. The same layer sees a fourfold different piece
   of the scene at 5 mm and at 2 cm. Pretrained weights are tied to the grid they were
   trained on; a model trained from scratch is tied to the grid it was trained at. No
   contract checks this: `grid_size`, `voxel_size` and `radii` are yours to set, and the
   defaults of room-scale encoders are wrong for a tabletop (table above).
4. **One PointNet++ for another** (trap 4). Seeds come from furthest point sampling and
   neighbours from ball queries; the start index, the tie rule and the fill order differ
   between implementations, so two "identical" networks hand different seeds to the head.
   Here they are options with names (`fps_tie_break`, `fps_skip_near_origin`); an author
   checkpoint reproduces only with the author's values.
5. **A head trained under one proposal, reused under another** (trap 5). Targets depend
   on which seeds reach the head (see *Label statistics move*). Retrain the head, and
   recompute any prior such as the scale-balance histogram.
6. **A seed-support grouper for a scene-support grouper "at another resolution"**
   (trap 6). See *Grouper* above; the contracts refuse the pair with the wrong head.
7. **View classification replaced by "the same thing with regression"** (trap 7). The
   label files are indexed by the 300 template views; a regressed direction needs one
   target per seed and a different assigner. There is no such view stage here.
8. **Scale-balanced weights on a model that does not read them.** Not in the survey: found
   by the swap matrix. `ScaleBalancedAssigner` with `SWADHead` or `TemplateViewStage`
   used to build, train and report losses while the weights were dropped. It is now
   refused; weighting GSNet needs a head that applies `MatchedLabels.weight`.
9. **A collision filter "on" compared across methods, and scores treated as one
   quantity** (traps 9 and 14). Post-processors are swappable in `evaluation.variants`,
   and every result is reported with and without them. But GSNet's score is an
   unbounded regressed value, EconomicGrasp's a weighted sum of raw logits and
   `OperationHead`'s a product with a tolerance (`score_kind` in the
   [catalogue](../components.md)). Anything that thresholds or fuses scores must be
   fitted per head bundle.
10. **A checkpoint carried across a swap** (trap 12 and ARCHITECTURE §8.3). Nothing claims
    a checkpoint is portable across components: `runtime.load_from` fails on missing,
    unexpected or mis-shaped keys and prints the report. Stages you did not swap keep
    their weights only if you load them deliberately (`allow_weight_mismatch`), and they
    were trained for the features of the old encoder.
11. **A weight file's licence inferred from its code** (trap 15). Sonata-family weights
    are CC BY-NC 4.0 under Apache-2.0 code. Using them makes the trained model
    non-commercial; the [scene encoder page](../model_zoo_backbones.md) lists each file.

## What is checked when

| When | What | How it fails |
|---|---|---|
| `grasppanda print-config <cfg> --check`, and first thing in `train` / `test` | Unknown keys and types; every rule of *What the checker compares*, with defaults resolved | `ConfigError` listing all problems, each with the config path and the file that set it |
| Model construction (`train` / `test`, seconds later) | The same rules on the built stages; channel widths; values a constructor validates | `ContractError`, `ValueError` or `BuildError` naming the component and the numbers |
| First batch | Shapes that depend on data: seeds per frame against the encoder's subsample, label views against the template, missing label assets | `ContractError` naming both sides |
| Never | Whether `grid_size`, radii and the recipe suit your scenes; whether the swap helps | — |

## Ready-made swap configs

`grasppanda compose` writes such a config for any combination and `grasppanda probe`
trains it for a few iterations on real frames: [Try a combination](../user_guides/combinations.md).
What that gave for every encoder and stage under every method is on the generated page
[Measured combinations](measured_combinations.md).

`configs/experimental/` holds a small set of encoder swaps that pass the check and were
trained for a few iterations on GraspNet-1Billion to prove that data, labels and model
fit together. Each file states what must be re-tuned and why. See
[`configs/experimental/README.md`](https://github.com/Daeda1used/GraspPanda/blob/main/configs/experimental/README.md).

## Adding a component to the matrix

A new stage is interchangeable when it says so in its contract and proves it in the
matrix:

1. declare the contract attributes of its slot ([catalogue](../components.md),
   [Add a component](add_component.md)) plus `status`, `source` and, if needed, `extras`;
2. add one tiny configuration to the tables of `tests/unit/test_swap_matrix.py` (the test
   fails until you do);
3. run `python -m tools.make_component_docs` to regenerate the catalogue and this page.
