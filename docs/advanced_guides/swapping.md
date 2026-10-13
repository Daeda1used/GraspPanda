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
| Encoder | `output` | Status | Needs | `EconomicGrasp` | `GraspNetBaseline` | `GSNet` | `ScaleBalancedGrasp` |
|---|---|---|---|---|---|---|---|
| `ASSANet` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `DGCNN` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `DeLA` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `LitePT` | `dense` | `weights-verified` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `MinkUNet` | `dense` | `tested` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `OACNNs` | `dense` | `tested` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `OctFormer` | `dense` | `bridge` | `ocnn` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PTv3` | `dense` | `weights-verified` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PTv3Sonata` | `dense` | `weights-verified` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PTv3Utonia` | `dense` | `weights-verified` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointMLP` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointMamba` | `subsampled` | `bridge` | `mamba-ssm` | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointMetaBase` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNeXt` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNet2` | `subsampled` | `weights-verified` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointNet2Dense` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNet2MSG` | `subsampled` | `tested` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointNet2MSGDense` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointNet2Transformer` | `subsampled` | `weights-verified` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PointTransformerV1` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointTransformerV2` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointVector` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `PointViT` | `subsampled` | `tested` | — | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `PyGPointEncoder` | `dense` | `bridge` | `torch_geometric` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `RandLANet` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `SpUNet` | `dense` | `tested` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `SparseUNet14` | `dense` | `tested` | `spconv` | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |
| `Uni3D` | `subsampled` | `bridge` | `timm` | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | ✓ |
| `Volt` | `dense` | `tested` | — | ✓ | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` |

✓: `model.backbone: {type: <Encoder>}` passes the config check as it stands. *in `Adapter`*: the bare swap is refused with a message that names the adapter; `model.backbone: {type: <Adapter>, backbone: {type: <Encoder>}}` passes.

Adapters: `FeaturePropagation` turns any encoder into a `dense` one; `FurthestPointSubsample` turns any encoder into a `subsampled` one.
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

| backbone ↓ · proposal → | `GraspnessFPS` | `SubsampledSeeds` |
|---|---|---|
| `dense` | ✓ | ✗ (1) |
| `subsampled` | ✗ (2) | ✓ |

1. SubsampledSeeds requires 'subsampled' scene features, but a dense encoder returns 'dense' ones (one feature per scene point).
2. GraspnessFPS requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).

#### Proposal and view stage

| proposal ↓ · view → | `ObjectnessViewStage` | `TemplateViewStage` |
|---|---|---|
| `GraspnessFPS` | ✗ (1) | ✓ |
| `SubsampledSeeds` | ✓ | ✓ |

1. GraspnessFPS and ObjectnessViewStage both predict and supervise the 'objectness' of a seed; a detector needs exactly one owner for it.

#### View stage and assigner

| view ↓ · assigner → | `CompactLabelAssigner` | `NearestLabelAssigner` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `ScaleBalancedAssigner` |
|---|---|---|---|---|
| `ObjectnessViewStage` | ✗ (1) | ✗ (2) | ✓ | ✓ |
| `TemplateViewStage` | ✓ | ✓ | ✓ | ✗ (3) |

1. ObjectnessViewStage computes its targets from 'dense' matched labels, but CompactLabelAssigner returns 'compact' ones.
2. ObjectnessViewStage reads 'on_object' and 'view_width' of the matched labels, which NearestLabelAssigner returns only with `supervise: object_seeds` and `view_widths: true`.
3. ScaleBalancedAssigner returns a loss weight per seed, but TemplateViewStage does not apply seed weights, so the weighting would silently do nothing there.

#### Assigner and head bundle

| assigner ↓ · head → | `EconomicHead` | `OperationHead` | `SWADHead` |
|---|---|---|---|
| `CompactLabelAssigner` | ✓ | ✗ (1) | ✗ (2) |
| `NearestLabelAssigner` | ✗ (3) | ✓ | ✓ |
| `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | ✗ (3) | ✓ | ✓ |
| `ScaleBalancedAssigner` | ✗ (4) | ✓ | ✗ (5) |

1. CompactLabelAssigner returns 'compact' matched labels, but OperationHead is supervised with 'dense' ones.
2. CompactLabelAssigner returns 'compact' matched labels, but SWADHead is supervised with 'dense' ones.
3. NearestLabelAssigner returns 'dense' matched labels, but EconomicHead is supervised with 'compact' ones.
4. ScaleBalancedAssigner returns 'dense' matched labels, but EconomicHead is supervised with 'compact' ones. ScaleBalancedAssigner returns a loss weight per seed, but EconomicHead does not apply seed weights, so the weighting would silently do nothing there.
5. ScaleBalancedAssigner returns a loss weight per seed, but SWADHead does not apply seed weights, so the weighting would silently do nothing there.

#### Grouper and head bundle

| grouper ↓ · head → | `EconomicHead` | `OperationHead` | `SWADHead` |
|---|---|---|---|
| `CylinderGroup` | ✓ | ✗ (1) | ✓ |
| `CylinderInteractionGroup` | ✓ | ✗ (2) | ✓ |
| `DepthCylinderGroup` | ✗ (3) | ✓ | ✗ (4) |
| `MultiScaleGroup` | ✓ | ✗ (5) | ✓ |

1. CylinderGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
2. CylinderInteractionGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
3. DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
4. DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
5. MultiScaleGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').

#### View stage and head bundle

| view ↓ · head → | `EconomicHead` | `OperationHead` | `SWADHead` |
|---|---|---|---|
| `ObjectnessViewStage` | ✓ | ✓ | ✓ |
| `TemplateViewStage` | ✓ | ✓ | ✓ |
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
The contracts accept **27 of 384** combinations of the registered stages (an assigner counts twice when it has options that switch optional fields on). The last column names the reference model that is this set.

| Encoder output | Proposal | View stage | Assigner | Grouper | Head bundle | Reference model of |
|---|---|---|---|---|---|---|
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | `EconomicGrasp` |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | `GSNet` |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `dense` | `GraspnessFPS` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | `GraspNetBaseline` |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `ObjectnessViewStage` | `ScaleBalancedAssigner` | `DepthCylinderGroup` | `OperationHead` | `ScaleBalancedGrasp` |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `CylinderInteractionGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `CompactLabelAssigner` | `MultiScaleGroup` | `EconomicHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner` | `MultiScaleGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `CylinderInteractionGroup` | `SWADHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `DepthCylinderGroup` | `OperationHead` | — |
| `subsampled` | `SubsampledSeeds` | `TemplateViewStage` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `MultiScaleGroup` | `SWADHead` | — |
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
