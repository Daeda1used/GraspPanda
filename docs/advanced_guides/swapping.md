# Swap rules

<span class="gp-chip">Seed-based · 5 methods</span>

This page covers the **seed-based skeleton**: GraspNet-baseline, GSNet, Scale-Balanced-Grasp,
EconomicGrasp and FineGrasp, which share the stages scene encoder → seed proposal → view →
assigner → grouper → head bundle. HGGD is a family of its own, whose own slot is the
[image encoder](../slots/image_encoders.md); [post-processing](../slots/post.md) and the
[training recipe](../slots/recipe.md) swap under every method.

Replacing one stage is one line. This page says which replacements the config check accepts,
why it refuses the others, and which look harmless and are not; whether an accepted swap trains
on real frames is on [Does it run?](measured_combinations.md). The tables are generated from the
contract rules and every cell is exercised by `tests/unit/test_swap_matrix.py`.

## The one-line swap

`grasppanda compose` writes a checked config and inserts an adapter where the rules ask for one:

```bash
grasppanda compose \
    economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
    --set backbone=ptv3
grasppanda compose \
    graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --set backbone=ptv3
```

The same encoder fits EconomicGrasp as it is. Under GraspNet-baseline, compose wraps it in
`FurthestPointSubsample` and says so, because that method's proposal reads features on a
sub-sample. A config written by hand without the adapter stops at the config check, before a
dataset is opened, with the fix.

??? note "The refusal, and the same swap written by hand"

    ```text
    $ grasppanda print-config \
          graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e --check \
          --opts "model.backbone={type: PTv3, in_channels: 3, grid_size: 0.005}"
    ✗ 1 config problem in configs/graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e.yaml

      model
        SubsampledSeeds requires 'subsampled' scene features, but PTv3 returns 'dense' ones (one
        feature per scene point). Choose a proposal that selects seeds from dense features
        (GraspnessFPS, SampledSeeds, ScoredSeeds), or wrap the backbone in the explicit adapter
        `{type: FurthestPointSubsample, backbone: {...}}`, which keeps the features of a
        furthest-point subsample.
    ```

    Under EconomicGrasp the same line passes. In a file:

    ```yaml
    _base_: [../economicgrasp/economicgrasp_sunet14_g1b-realsense_10e.yaml]
    model:
      backbone: {type: PTv3, in_channels: 3, grid_size: 0.005}   # the options of the old type are dropped
    ```

## What the checker compares

Each stage declares a contract as class attributes, and the rules compare them before anything
is built. They belong to the skeleton: the verdict is the same under every seed-based detector
type (`GraspNetBaseline`, `GSNet`, `ScaleBalancedGrasp`, `EconomicGrasp`; FineGrasp is built as
an `EconomicGrasp` detector), and the same for a YAML file or `DETECTORS.build(...)` in Python.

??? note "The rules: what each compares, and why it exists"

    | Rule | Compares | Why it exists | Kind of refusal |
    |---|---|---|---|
    | Where features live | `backbone.output` with `proposal.requires` | Features on every scene point and features on a subsample are different objects. Converting is a learned or a sampling step that changes the model, so it is a component you write down, never an interpolation behind your back. | `fixed-by-compose`: the encoder adapter |
    | Feature axes | `grouper.out_axes` with `head.in_axes` | A crop with a depth axis and one without feed different heads. An axis is never added or squeezed implicitly: an axis adapter is a component you write down (`DepthEmbedGroup` repeats a crop over the depth bins with a learned embedding of the bin, as Scale-Balanced-Grasp repeats its seed feature; `DepthPoolGroup` takes the maximum over the bins). | `fixed-by-compose`: the axis adapter. `structural` for an axis adapter under the head of the other family; `not-implemented` for a scene-support crop without a depth axis under a head with one (no adapter repeats a scene crop) |
    | Label set | `assigner.label_format` with `head.label_format` and `view.label_formats` | Dense grids (48 grasps per point and view) and compact labels (one) are different supervision; a grid cannot be recovered from one grasp. | `structural`. Compose writes the label set the head reads into the assigner and the data recipe, so the rule is met when a request sets an assigner of the other set, or a view stage that reads dense grids only under compact labels |
    | Fields of the matched labels | `view.needs_matched`, `head.needs_matched` with `assigner.optional_outputs` | A stage that reads an optional field needs the assigner option that produces it. The error names the option. | `fixed-by-compose`: the option; `structural` when the assigner never returns the field |
    | Seed weights | `assigner.optional_outputs["weight"]` with `applies_seed_weights` of view stage and head | Every shipped view stage and head bundle applies a weight per seed: each of its losses is the weighted mean over its supervised entries, every entry weighted by its seed. A stage that does not would train unweighted next to a weighted assigner. | not a refusal: a warning in the report of `grasppanda compose` and when the model is built |
    | Seed scores | `seed_scores` of proposal and view stage; `head.required_seed_scores(options)` | One owner per score (`objectness` cannot be predicted and supervised twice), and a score the decoder multiplies in must be predicted by some stage. | `fixed-by-compose` where a stage has the option: a scoring proposal predicts graspness only (`ScoredSeeds` with `select_by: graspness_only`), the head decodes without the score (`OperationHead` with `score_decode: score_x_tolerance`). `structural` where it has none: a proposal that selects its seeds by objectness |
    | Depth bins | `len(grouper.hmax)` with `head.num_depth` | One crop per depth bin of the head. | `not-a-combination` |
    | One score definition | `score_target`, `max_friction` of view stage and head | Only when both use the definition of `grasppanda.models.heads.friction`: the view target is the best grasp score of a view. | `not-a-combination` |
    | Loss terms | `loss_terms` of a stage with `accepts` of the loss in it | A classification loss on a regression term, or a loss that scores a set as a whole (`DiceLoss`) where a weight per element arrives, is not the same supervision under another name. | `structural` |

## Why a pair is refused

Every refusal carries a kind: a property of the components, or of what the toolbox implements
today. `grasppanda compose` and the explorer use the same words:

- **incompatible by design** (`structural`): the components cannot work together whatever their
  options (two label sets, two owners of one score, a classification loss on a regression term).
- **not supported yet** (`not-implemented`): the combination makes sense, but the code that joins
  the two parts does not exist yet; the message names what would lift it.
- **needs a setting compose adds** (`fixed-by-compose`): an adapter or an option lifts it, and
  `grasppanda compose` inserts or sets it and says so.
- **needs a preset's settings** (`not-a-combination`): the components fit, but the config does not
  state them fully (a missing option, or two options that disagree).
- **needs your consent** (`policy`): model code from a hub repository runs only with
  `accept_remote_code: true`.

??? note "What compose lifts by itself"

    Besides the encoder adapters and the projected grouper: the axis adapter between grouper and
    head; the assigner options a stage reads; the graspness-only option of a scoring proposal next
    to `ObjectnessViewStage`; the decode option of a head whose score no stage predicts; and the
    label set of a head bundle, written into the assigner and the data recipe (`--set
    labels=dense|compact` asks for it in so many words). Each is one sentence in its report and in
    the header of the composed file. A data recipe it does not rewrite gets the exact `--set`,
    `--unset` and `--opts` that would, in the report
    ([Compose and probe](../user_guides/combinations.md)). Its `--json` report lists the kinds in
    `refusal_kinds`, in the order of `refusal`.

## Encoders

<!-- generated:encoders:begin -->
29 registered scene encoders: 23 return `dense` features, 6 return `subsampled` features. Every one passes the contract under every seed-based method: as it is where its output is what the method's proposal needs, otherwise inside the adapter that `grasppanda compose` inserts (`FeaturePropagation` or `FurthestPointSubsample`).

??? note "Every encoder under every method"

    <div data-search-exclude markdown>

    | Encoder | `output` | Width | Status | Needs | GraspNet-baseline | GSNet | Scale-Balanced-Grasp | EconomicGrasp | FineGrasp |
    |---|---|---|---|---|---|---|---|---|---|
    | `ASSANet` | `dense` | 64 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `DGCNN` | `dense` | 256 (`out_channels`) | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `DeLA` | `dense` | 256 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `LitePT` | `dense` | 72 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `MinkUNet` | `dense` | 96 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `OACNNs` | `dense` | 20 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `OctFormer` | `dense` | 168 (`out_channels`) | `bridge` | `ocnn` | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PTv3` | `dense` | 64 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PTv3Sonata` | `dense` | 1232 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PTv3Utonia` | `dense` | 1386 | `weights-verified` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointMLP` | `dense` | 128 (`out_channels`) | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointMamba` | `subsampled` | 384 (`out_channels`) | `bridge` | `mamba-ssm` | ✓ | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | in `FeaturePropagation` |
    | `PointMetaBase` | `dense` | 32 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointNeXt` | `dense` | 32 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointNet2` | `subsampled` | 256 | `weights-verified` | — | ✓ | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | in `FeaturePropagation` |
    | `PointNet2Dense` | `dense` | 128 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointNet2MSG` | `subsampled` | 256 | `tested` | — | ✓ | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | in `FeaturePropagation` |
    | `PointNet2MSGDense` | `dense` | 128 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointNet2Transformer` | `subsampled` | 256 | `weights-verified` | — | ✓ | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | in `FeaturePropagation` |
    | `PointTransformerV1` | `dense` | 32 (`out_channels`) | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointTransformerV2` | `dense` | 48 (`out_channels`) | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointVector` | `dense` | 32 | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `PointViT` | `subsampled` | 384 (`out_channels`) | `tested` | — | ✓ | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | in `FeaturePropagation` |
    | `PyGPointEncoder` | `dense` | 256 (`out_channels`) | `bridge` | `torch_geometric` | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `RandLANet` | `dense` | 32 (`out_channels`) | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `SpUNet` | `dense` | 20 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `SparseUNet14` | `dense` | 512 (`out_channels`) | `tested` | `spconv-cu118` (or another `spconv-*` build) | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |
    | `Uni3D` | `subsampled` | once built (`out_channels`) | `bridge` | `timm` | ✓ | in `FeaturePropagation` | ✓ | in `FeaturePropagation` | in `FeaturePropagation` |
    | `Volt` | `dense` | 256 (`out_channels`) | `tested` | — | in `FurthestPointSubsample` | ✓ | in `FurthestPointSubsample` | ✓ | ✓ |

    </div>

    ✓: the features of `model.backbone: {type: <Encoder>}` live where the proposal of the method needs them. *in `Adapter`*: the bare swap is refused with a message that names the adapter; `model.backbone: {type: <Adapter>, backbone: {type: <Encoder>}}` passes that rule.

    Adapters: `FeaturePropagation` turns any encoder into a `dense` one; `FurthestPointSubsample` turns any encoder into a `subsampled` one.

    *Width* is the number of channels the encoder returns with its constructor defaults, as it declares it (`Backbone.output_channels`); `(out_channels)` marks an encoder that has an option of that name to set it. The other stages take whatever width arrives, with these exceptions among the methods as shipped: Scale-Balanced-Grasp needs **256** channels (`DepthCylinderGroup` adds the seed features to its crop features). An encoder of another width is refused there by the config check, with both numbers and the ways out; `grasppanda compose` takes one of them by itself (the encoder's `out_channels`, or the projected variant of the grouper). *once built*: the width follows from a library model or a layout that only the constructor resolves, so the check builds the encoder once, without released weights.
<!-- generated:encoders:end -->

??? note "What the two adapters change: they are the only bridges, and never silent"

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

The contract says where features live, not what a feature has seen: the unit of length, the
receptive field and the recipe move with the encoder.

??? note "The defaults of each encoder, and what to re-tune first"

    These are the constructor defaults, which is what a bare `{type: ...}` gives you:

    | Encoder | Unit of length | What sets the receptive field | Seeds under its method | Re-tune first |
    |---|---|---|---|---|
    | `SparseUNet14` | `voxel_size`, 5 mm | Four stride-2 stages: the coarsest voxel is 16 × the grid, 8 cm at 5 mm | 1024 by `GraspnessFPS` from predicted graspness | — (reference of GSNet, EconomicGrasp and FineGrasp) |
    | `SpUNet` | `grid_size`, default **2 cm** (ScanNet rooms) | The same four stride-2 stages, with 2-3-4-6 residual blocks where `SparseUNet14` has one: more 3³ convolutions per level, so a feature reaches further in voxels | as above | `grid_size`; `out_channels` (default 20, class scores); `in_channels` (default 6); learning rate (the training check lost most seeds in two iterations at 1e-3) |
    | `PTv3` | `grid_size`, default **2 cm** (ScanNet rooms) | Four stride-2 poolings, and attention over patches of 1024 serialised voxels: counted in points, not in metres | as above | `grid_size` (0.005, the unit of the sparse-U-Net methods); `enc_patch_size` / `dec_patch_size` if clouds are small; learning rate and schedule |
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

Each table pairs two stages with their default options. `✓`: the pair passes; a number points
to the reason below the table, the first sentence of the error the config check prints after
its kind. A `fixed-by-compose` pair is accepted by `grasppanda compose`, which inserts what the
message names.

<!-- generated:pairs:begin -->
??? note "Encoder output and proposal: 5 of 10 pairs refused"

    | backbone ↓ · proposal → | `GraspnessFPS` | `SampledSeeds` | `ScoredSeeds` | `ScoredSeeds (select_by: graspness_only)` | `SubsampledSeeds` |
    |---|---|---|---|---|---|
    | `dense` | ✓ | ✓ | ✓ | ✓ | ✗ (1) |
    | `subsampled` | ✗ (2) | ✗ (3) | ✗ (4) | ✗ (4) | ✓ |

    1. [fixed-by-compose] SubsampledSeeds requires 'subsampled' scene features, but a dense encoder returns 'dense' ones (one feature per scene point).
    2. [fixed-by-compose] GraspnessFPS requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).
    3. [fixed-by-compose] SampledSeeds requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).
    4. [fixed-by-compose] ScoredSeeds requires 'dense' scene features, but a subsampled encoder returns 'subsampled' ones (features on a sub-sampled point set).

??? note "Proposal and view stage: 2 of 40 pairs refused"

    | proposal ↓ · view → | `DirectionViewStage` | `HealpixViewStage` | `IcosphereViewStage` | `LatticeViewStage` | `ObjectnessViewStage` | `RomaRotationViewStage` | `TemplateViewStage` | `TwoVectorViewStage` |
    |---|---|---|---|---|---|---|---|---|
    | `GraspnessFPS` | ✓ | ✓ | ✓ | ✓ | ✗ (1) | ✓ | ✓ | ✓ |
    | `SampledSeeds` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
    | `ScoredSeeds` | ✓ | ✓ | ✓ | ✓ | ✗ (2) | ✓ | ✓ | ✓ |
    | `ScoredSeeds (select_by: graspness_only)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
    | `SubsampledSeeds` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

    1. [fixed-by-compose] GraspnessFPS and ObjectnessViewStage both predict and supervise the 'objectness' of a seed; a detector needs exactly one owner for it.
    2. [fixed-by-compose] ScoredSeeds and ObjectnessViewStage both predict and supervise the 'objectness' of a seed; a detector needs exactly one owner for it.

??? note "View stage and assigner: 2 of 32 pairs refused"

    | view ↓ · assigner → | `CompactLabelAssigner` | `NearestLabelAssigner` | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | `ScaleBalancedAssigner` |
    |---|---|---|---|---|
    | `DirectionViewStage` | ✓ | ✓ | ✓ | ✓ |
    | `HealpixViewStage` | ✓ | ✓ | ✓ | ✓ |
    | `IcosphereViewStage` | ✓ | ✓ | ✓ | ✓ |
    | `LatticeViewStage` | ✓ | ✓ | ✓ | ✓ |
    | `ObjectnessViewStage` | ✗ (1) | ✗ (2) | ✓ | ✓ |
    | `RomaRotationViewStage` | ✓ | ✓ | ✓ | ✓ |
    | `TemplateViewStage` | ✓ | ✓ | ✓ | ✓ |
    | `TwoVectorViewStage` | ✓ | ✓ | ✓ | ✓ |

    1. [structural] ObjectnessViewStage computes its targets from 'dense' matched labels, but CompactLabelAssigner returns 'compact' ones.
    2. [fixed-by-compose] ObjectnessViewStage reads 'on_object' and 'view_width' of the matched labels, which NearestLabelAssigner returns only with `supervise: object_seeds` and `view_widths: true`.

??? note "Assigner and head bundle: 6 of 16 pairs refused"

    | assigner ↓ · head → | `DecoupledGridHead` | `EconomicHead` | `OperationHead` | `SWADHead` |
    |---|---|---|---|---|
    | `CompactLabelAssigner` | ✗ (1) | ✓ | ✗ (2) | ✗ (3) |
    | `NearestLabelAssigner` | ✓ | ✗ (4) | ✓ | ✓ |
    | `NearestLabelAssigner (supervise: object_seeds, view_widths: true)` | ✓ | ✗ (4) | ✓ | ✓ |
    | `ScaleBalancedAssigner` | ✓ | ✗ (5) | ✓ | ✓ |

    1. [structural] CompactLabelAssigner returns 'compact' matched labels, but DecoupledGridHead is supervised with 'dense' ones.
    2. [structural] CompactLabelAssigner returns 'compact' matched labels, but OperationHead is supervised with 'dense' ones.
    3. [structural] CompactLabelAssigner returns 'compact' matched labels, but SWADHead is supervised with 'dense' ones.
    4. [structural] NearestLabelAssigner returns 'dense' matched labels, but EconomicHead is supervised with 'compact' ones.
    5. [structural] ScaleBalancedAssigner returns 'dense' matched labels, but EconomicHead is supervised with 'compact' ones.

??? note "Grouper and head bundle: 22 of 56 pairs refused"

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

    1. [fixed-by-compose] CylinderGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    2. [fixed-by-compose] CylinderInteractionGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    3. [fixed-by-compose] DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
    4. [fixed-by-compose] DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
    5. [fixed-by-compose] DepthCylinderGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
    6. [fixed-by-compose] DepthCylinderProjectionGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
    7. [fixed-by-compose] DepthCylinderProjectionGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
    8. [fixed-by-compose] DepthCylinderProjectionGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
    9. [structural] DepthEmbedGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
    10. [structural] DepthEmbedGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
    11. [structural] DepthEmbedGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
    12. [fixed-by-compose] DepthLocalGroup emits features with axes ('seed', 'depth', 'channel'), but DecoupledGridHead expects ('seed', 'channel').
    13. [fixed-by-compose] DepthLocalGroup emits features with axes ('seed', 'depth', 'channel'), but EconomicHead expects ('seed', 'channel').
    14. [fixed-by-compose] DepthLocalGroup emits features with axes ('seed', 'depth', 'channel'), but SWADHead expects ('seed', 'channel').
    15. [structural] DepthPoolGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    16. [fixed-by-compose] LocalGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    17. [fixed-by-compose] MultiRadiusGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    18. [fixed-by-compose] MultiScaleGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    19. [fixed-by-compose] PyGConvGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    20. [fixed-by-compose] RoIAwareGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    21. [fixed-by-compose] RoIGridGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').
    22. [not-implemented] VoxelSetAbstractionGroup emits features with axes ('seed', 'channel'), but OperationHead expects ('seed', 'depth', 'channel').

??? note "View stage and head bundle: 0 of 32 pairs refused"

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

??? note "Reading the tables"

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
      --compact-labels`); `OperationHead` also needs the tolerance asset. `grasppanda
      compose` writes the label set of the head into the assigner and the data recipe.

## Every legal set of stages

<!-- generated:stage-sets:begin -->
The contracts accept **3014 of 17920** combinations of the registered stage classes (an assigner counts twice when it has options that switch optional fields on, a proposal when it has an option that leaves a seed score to the view stage). By head bundle and kind of encoder output:

| Head bundle | `dense` encoder | `subsampled` encoder | Shipped by |
|---|---|---|---|
| `DecoupledGridHead` | 880 | 230 | — |
| `EconomicHead` | 280 | 70 | EconomicGrasp, FineGrasp |
| `OperationHead` | 352 | 92 | GraspNet-baseline, Scale-Balanced-Grasp |
| `SWADHead` | 880 | 230 | GSNet |

Every set, one per line: [stage_sets.json](stage_sets.json) (the stage classes by index into `variants`; `methods` holds the set of each method as shipped). A sample of them was trained on real frames: [Sets of stages](measured_combinations.md#sets-of-stages).
<!-- generated:stage-sets:end -->

The detector `type` does not change which sets are legal. It names the method whose reference
behaviour the config claims to follow; a set that is no reference model is a new model,
whatever it is called.

??? note "Why HGGD is not in these tables"

    HGGD is not built on the seed-based skeleton. It proposes partial grasps from an image
    heat-map, has no view stage and no approach-aligned crop, and none of its stages fills a
    slot above; putting a seed-based stage into it, or its heat-map proposal into a seed-based
    detector, is not a swap the registries can express. Its image encoder is a slot of its own
    ([Image encoders](../slots/image_encoders.md)); its other parts are options of the detector
    (`grasppanda components HGGD`). Its "ResNet-34" is the block layout only, with a base width
    of 8 and four input channels, trained from scratch: a pretrained three-channel ResNet-34 is a
    different network with a different input.

## Traps

Swaps that pass every check above, or look as if they should, and are not what they seem. The
numbers are stable: code comments and error messages refer to a trap by its number.

??? note "The eleven traps, by number"

    1. **An object-level pretrained encoder as scene encoder.** Point-BERT, Point-MAE, ReCon,
        Uni3D were pretrained to turn one normalised object of a few thousand points into
        patch tokens or a single vector. Their architectures are registered here (`PointViT`,
        `PointMamba`, `Uni3D`) as sub-sampling scene encoders over patch tokens, built with random weights: features live on the token centres, not on
        every point. Their released weights are an initialisation from single objects, not
        scene features, and scaling the scene to an object's size changes the metres the
        cylinder radii and depth bins are written in (each preset says what it chose).

    2. **“PTv3 with Sonata, Concerto or Utonia weights” as “PTv3”.** The released
        self-supervised checkpoints are encoder-only, have other depths and widths, expect nine
        input channels (coordinates, colour, normal; zeros where missing) and were trained on a
        2 cm or rescaled grid. They load into `PTv3Sonata` / `PTv3Utonia`, not into `PTv3`;
        loading one into the wrong type fails on the key report. Changing only a weight path
        changes all of the above.

    3. **Keeping the grid when changing the encoder, or changing it freely.** The grid is the
        encoder's unit of length. The same layer sees a fourfold different piece of the scene
        at 5 mm and at 2 cm. Pretrained weights are tied to the grid they were trained on; a
        model trained from scratch is tied to the grid it was trained at. No contract checks
        this: `grid_size`, `voxel_size` and `radii` are yours to set, and the defaults of
        room-scale encoders are wrong for a tabletop (see *What an encoder swap changes*).

    4. **One PointNet++ for another.** Seeds come from furthest point sampling and neighbours
        from ball queries; the start index, the tie rule and the fill order differ between
        implementations, so two "identical" networks hand different seeds to the head. Here
        they are options with names (`fps_tie_break`, `fps_skip_near_origin`); an author
        checkpoint reproduces only with the author's values.

    5. **A head trained under one proposal, reused under another.** Targets depend on which
        seeds reach the head (see *Label statistics move*). Retrain the head, and recompute any
        prior such as the scale-balance histogram.

    6. **A seed-support grouper for a scene-support grouper “at another resolution”.** See
        *Reading the tables*, Grouper; the contracts refuse the pair with the wrong head.

    7. **View classification replaced by “the same thing with regression”.** The label files
        are indexed by the 300 template views. Regressing the direction is available
        (`direction-regression`, `direction-vmf`), but the regressed direction is snapped to
        that lattice for the label lookup, so its targets still come from the 300 label views,
        and its training crops use the frame of the snapped view while inference uses the
        regressed one (up to 9 degrees apart).

    8. **Scale-balanced weights on a model that does not read them.** Found by the swap
        matrix. `ScaleBalancedAssigner` with `SWADHead` or `TemplateViewStage` used to build,
        train and report losses while the weights were dropped. Every shipped view
        stage and head bundle now applies `MatchedLabels.weight` (a weighted mean over its
        supervised entries); a stage of your own that does not is reported as a warning.

    9. **A collision filter “on” compared across methods, and scores treated as one quantity.**
        Post-processors are swappable in `evaluation.variants`, and every result is reported
        with and without them. But GSNet's score is an unbounded regressed value, EconomicGrasp's a weighted sum of raw logits and `OperationHead`'s a
        product with a tolerance (`score_kind` in [Registered classes](../components.md)).
        Anything that thresholds or fuses scores must be fitted per head bundle.

    10. **A checkpoint carried across a swap.** See [Architecture](../design/ARCHITECTURE.md),
        §8.3. Nothing claims a checkpoint is portable across components: `runtime.load_from`
        fails on missing, unexpected or mis-shaped keys and prints the report. Stages you did not swap
        keep their weights only if you load them deliberately (`allow_weight_mismatch`), and
        they were trained for the features of the old encoder.

    11. **A weight file's licence inferred from its code.** Sonata-family weights are
        CC BY-NC 4.0 under Apache-2.0 code. Using them makes the
        trained model non-commercial; [Encoders checked against their authors'
        code](../model_zoo_backbones.md) lists each file.

## What is checked when

The config check runs before anything is built; model construction and the first batch check
what only they can see. Never checked: whether the grid, the radii and the recipe suit your
scenes, or whether the swap helps.

??? note "When each check runs, and how it fails"

    | When | What | How it fails |
    |---|---|---|
    | `grasppanda print-config <cfg> --check`, and first thing in `train` / `test` | Unknown keys and types; every rule of *What the checker compares*, with defaults resolved | `ConfigError` listing all problems, each with the config path and the file that set it |
    | Model construction (`train` / `test`, seconds later) | The same rules on the built stages; channel widths; values a constructor validates | `ContractError`, `ValueError` or `BuildError` naming the component and the numbers |
    | First batch | Shapes that depend on data: seeds per frame against the encoder's subsample, label views against the template, missing label assets | `ContractError` naming both sides |
    | Never | Whether `grid_size`, radii and the recipe suit your scenes; whether the swap helps | — |

## Ready-made swap configs

`grasppanda compose` writes a checked config for any combination, and `grasppanda probe` trains
it for a few iterations on real frames ([Compose and probe](../user_guides/combinations.md)).
`configs/experimental/` holds a few encoder swaps written that way, with no AP attached, each
stating what must be re-tuned and why
([`configs/experimental/README.md`](https://github.com/Daeda1used/GraspPanda/blob/main/configs/experimental/README.md)).

??? note "What the matrix test proves, and what it does not"

    `tests/unit/test_swap_matrix.py` runs on tiny synthetic scenes (two frames of 96 points,
    random dense and compact labels, a width histogram) on the CPU, with one tiny configuration
    per registered component. It proves that:

    - every registered scene encoder under every seed-based detector type either builds, trains
      one step with a gradient reaching the encoder and predicts, or is refused with the adapter
      by name, and the adapted model then does;
    - every grouper under every head bundle combines when the axes agree, and where they do not,
      the grouper inside its axis adapter trains and predicts, while a pair without an adapter is
      refused as `structural` or `not-implemented`;
    - every proposal under both kinds of encoder output does the same;
    - in the full product of proposals, view stages, assigners, groupers and head bundles under
      both kinds of encoder output, no set that passes the config check fails later: every
      accepted set trains one step and predicts under the barest and the strictest detector
      type, and every refused set is refused by the built detector too;
    - every head bundle applies a weight per seed as a weighted mean (uniform weights give the
      unweighted losses), the scoring proposal with `select_by: graspness_only` leaves objectness
      to the view stage that owns it, and `OperationHead` decodes without the objectness nobody
      predicts.

    What it does not prove: that a combination learns, or anything about real frames. A cell of
    [Does it run?](measured_combinations.md) is a probe on real frames; a synthetic probe of a
    composed config (`grasppanda probe <config> --synthetic`) is the same check as here at the
    sizes of the config.

## Adding a component to the matrix

A new stage is interchangeable when it says so in its contract and proves it in the matrix:

1. declare the contract attributes of its slot ([Registered classes](../components.md),
   [Add a component](add_component.md)) plus `status`, `source` and, if needed, `extras`;
2. add one tiny configuration to the tables of `tests/unit/test_swap_matrix.py` (the test
   fails until you do);
3. run `python -m tools.make_component_docs` to regenerate the registered classes and this page.
