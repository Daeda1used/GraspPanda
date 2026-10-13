# Changelog

All notable changes to GraspPanda are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

GraspPanda was rebuilt from scratch in October 2026. The releases up to v0.56 are an
earlier code base with a different design; they stay available under their tags and on the
`legacy` branch, and none of their code is part of this history.

### Initial public version of the rebuild

- **Core.** YAML configs with inheritance and three layers of validation; registries with
  lazy imports; typed data contracts between pipeline stages; run snapshots. Commands:
  `train`, `test`, `infer`, `demo`, `analyze`, `components`, `compose`, `probe`,
  `explore`, `sweep`, `prepare`, `print-config`, `env`.
- **Benchmarks.** Readers, derived label assets and evaluators for GraspNet-1Billion and
  GraspClutter6D. Both evaluators agree with the official tools on recorded frames (1,592
  and 254); the GraspNet-1Billion one runs about 40 times faster. One command scores a
  GraspNet-1Billion model on GraspClutter6D.
- **Methods as interchangeable components**, each equal to its reference implementation
  stage by stage with converted weights: GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline
  and Scale-Balanced-Grasp on one seed-based skeleton, and HGGD as a family of its own.
  `tools/convert_checkpoint.py` converts the authors' checkpoints strictly.
- **Scene encoders.** Sparse U-Net, SpUNet, OA-CNNs, PointNet++ (two layouts), PointNeXt,
  Point Transformer V3 (also with Sonata, Concerto and Utonia weights), LitePT, the
  transformer encoder of Scale-Balanced-Grasp, and adapters between dense and sub-sampled
  encoders. Released weights load strictly and reproduce the official outputs.
- **Interchange.** `grasppanda components` lists every component with its contract, status
  and source. A swap matrix test proves that every combination of registered stages either
  trains and predicts or is refused at config time with the conflict and the fix.
  `configs/experimental/` holds ready-made encoder swaps.
- **Combinations.** `grasppanda compose <method config> --set <slot>=<preset or component>`
  writes a checked config, inserting the one encoder adapter the stage contracts ask for;
  a refused combination prints the conflict and the fix. Presets
  (`grasppanda/presets/<slot>/`) hold the settings of every encoder and stage variant for
  table-top depth clouds. `grasppanda probe <config>` runs the real trainer for a few
  iterations on the `sample` split and predicts, ending with one verdict. The generated
  page `docs/advanced_guides/measured_combinations.md` records the combinations probed on
  real frames and the measured class of every component (universal, most, some,
  specific); see "Measured records" below for the numbers. No AP.
- **Inference and analysis.** `GraspInferencer` and `grasppanda demo` (depth image, point
  cloud, dataset frame or the built-in synthetic scene); `grasppanda analyze` splits an AP
  by scene, object and cause.
- **Speed.** Optional compiled kernels for furthest point sampling, ball query, cylinder
  query and nearest neighbours with bit-identical results; dense label grids are read where
  they are matched instead of being copied into every batch.

### Added: options for every slot

Options taken from published code and model libraries, by three routes (a bridge to an
optional library, source adapted from a repository under the MIT, BSD or Apache-2.0
licence and listed in `THIRD_PARTY_NOTICES.md`, released weights for a class that exists).
Each has a preset, a unit test of its contract, a row and a command on its slot page
(`docs/slots/`, starting at `docs/slots/index.md`) and a measured verdict per method.
**No accuracy is claimed for any of them.**

- **Image encoders** (`model.image_encoder`, registry `IMAGE_ENCODERS`). HGGD's image
  encoder is a slot: `TimmImageEncoder`, `HFImageEncoder` and `TorchvisionImageEncoder`
  (extras `timm`, `transformers`, `torchvision`) with 38 presets, depth as an explicit
  option, and a pyramid decoder that takes any strides and widths. Left out, the model is
  the reference network. `ImagePainted` and `ImagePaintedSubsampled` give the points of a
  seed-based method the image features at their pixel.
- **Scene encoders.** Point-based: `PointNet2MSG`, `PointNet2MSGDense`, `PointMetaBase`,
  `PointVector`, `ASSANet`, `PointMLP`, `DeLA`, `DGCNN`, `RandLANet`, the bridge
  `PyGPointEncoder` (extra `torch_geometric`) and three more PointNeXt sizes. Sparse:
  `MinkUNet` (14 to 101). Transformers and tokens: `PointTransformerV1`,
  `PointTransformerV2`, `Volt`, `OctFormer` (extra `ocnn`), `PointViT`, `PointMamba`
  (extra `mamba-ssm`), `Uni3D` (extra `timm`).
- **Released encoder weights.** `pretrained: <name or path>` on `SpUNet`, `MinkUNet`,
  `PTv3`, `PTv3Sonata`, `PTv3Utonia` and `LitePT`, with fifteen published checkpoints
  registered by name; `tools/convert_backbone_weights.py` (PointNeXt, VoteNet, Contrastive
  Scene Contexts) and `tools/convert_encoder_checkpoint.py` (Point-MAE, MaskPoint,
  PointMamba, Uni3D, Volt). Weights are never shipped; several are non-commercial.
- **Seed proposal.** `ScoredSeeds` (the selecting score, the candidate rule, the sampler
  and the padding as options) and `SampledSeeds`; a registry `SAMPLERS` with eleven
  samplers (FPS, feature and fusion FPS, score-weighted FPS, top-k, random, voxel grid,
  FPS then top-k, score NMS, bridges to `fpsample` and `torch_cluster`); 22 presets.
- **View.** `LatticeViewStage` (a lattice of any size, classification, angular class
  loss, soft arg-max, top-k), `IcosphereViewStage` (extra `trimesh`), `HealpixViewStage`
  (extra `healpix`), `DirectionViewStage`, `TwoVectorViewStage`, `RomaRotationViewStage`
  (extra `roma`); 18 presets.
- **Grouper.** `LocalGroup` and `DepthLocalGroup` (shape, encoder and pooling as
  options), `RoIGridGroup`, `RoIAwareGroup`, `VoxelSetAbstractionGroup`, `PyGConvGroup`
  (extra `torch_geometric`), `MultiRadiusGroup`, the axis adapters `DepthEmbedGroup` and
  `DepthPoolGroup`, and `DepthCylinderProjectionGroup`; 25 presets.
- **Heads and losses.** Losses are a typed slot: every loss term names its kind (`class`,
  `unit`, `value`, `bins`), every loss the kinds it fits, and the config check refuses
  another pairing with the losses that fit. Twenty loss presets, set per term
  (`--set head.score_loss=quality-focal`; extras `torchvision`, `kornia`), three of them
  wrappers (hard example mining, class balance, uncertainty weighting).
  `DecoupledGridHead` with four presets, and `swad-focal`, `economic-focal`.
- **Post-processing.** Twelve steps beside the collision filter and the NMS (`TopK`,
  `ScoreThreshold`, `FurthestGraspSampling`, `WorkspaceFilter`, `ApproachAngleFilter`,
  `TablePlaneFilter`, `MaskFilter`, `GripperLimits`, `GripperCentering`, `WidthFromCloud`,
  `ToFrame`, `GripperMeshCollisionFilter` with the extra `trimesh`); `--post STEP...` for
  `test`, `infer` and `demo`, `post=` for `GraspInferencer`, and `--set post=a+b` for
  `compose`; 15 presets.
- **Training recipe.** `compose` takes `--set optimizer=`, `schedule=`, `hooks=a+b`,
  `augment=a+b` and `recipe=` for every detector; 37 presets under `grasppanda/recipes/`.
  `TimmOptimizer`, `PytorchOptimizer`, `ScheduleFree`, `TimmSchedule` (extra `optim`),
  `ParamGroups` (rates per module, layer-wise decay), `EMAHook`, `SWAHook`, `FreezeHook`,
  `ScheduleFreeHook`, `AdaptiveGradClipHook`, and the augmentations `RandomJitter`,
  `RandomDropout`, `RandomShift`.
- **Slot documentation.** `docs/slots/index.md` (the pipeline, the task of every slot, the
  number of its options, one table that classifies every option from the measured
  records, how to try one), an overview of the scene encoders and one page per slot.

### Added: the combination explorer

- A documentation page (`docs/explorer/`) that answers a selection of method and slot
  options from the measured records: the verdict, the reason of a refusal or failure, the
  compose and probe commands, the preset; a grid of option by method per slot. Data
  generated by `tools/make_explorer.py`.
- `tools/make_explorer_contracts.py` stores what `grasppanda compose` answers for every
  option alone and every two options of two slots; the page shows them as a contract
  check, apart from measured verdicts.
- `grasppanda explore` serves the explorer on 127.0.0.1 from the installed package and
  checks every selection with the real `compose`; an accepted combination can be saved
  as a config. Nothing is trained.
- An animated README picture (`resources/showcase-{light,dark}.svg`) generated by
  `tools/make_showcase.py` from a captured transcript of the three commands it shows.

### Measured records

- 980 cells (770 `runs`, 11 `runs-unstable`, 193 `refused`, 6 `fails`), each one run of
  `compose` and `probe` on real frames. Beside one slot replaced at a time under every
  method the grid has groups for the image
  encoders of HGGD, post-processing, losses and recipe presets, and `cross`: eighteen
  requests that replace an encoder together with options of other slots, at least two per
  method (seventeen run; one is refused because its grouper and its head disagree on the
  axes of a crop).
- The contract rules accept 7456 sets of stage variants; 95 were probed, picked so that
  every stage variant appears in a probed set under each kind of encoder, and the others
  are listed as not probed.
- Every record names the run that measured it; a probe that runs out of GPU memory is
  retried alone and, if memory stays short, not recorded.

### Fixed

- A model on a data recipe that does not serve its labels is refused by the config check
  instead of failing at the first loss. Stages declare what they read from the training
  supervision (`needs_supervision`), dataset readers what a config makes them serve
  (`served_supervision`), and a rule compares the two at `data.train.dataset`, naming the
  field, the stage that reads it and the section or transform that would serve it.
  `grasppanda compose` answers with the shipped configs whose recipe serves the labels,
  or with the preset that brings the missing dataset section.
- The width of the seed features is one contract. An encoder declares the width it
  returns for its options (`Backbone.output_channels`), a grouper the width it needs
  (`Grouper.required_seed_channels`) and the variant that takes any width; the config
  check compares them without building anything. Compose sets a new encoder's
  `out_channels` to the width the grouper needs and otherwise puts
  `DepthCylinderProjectionGroup` in place; `--no-adapter` leaves the pair refused.
  Encoders of a fixed width used to pass the check under Scale-Balanced-Grasp and fail
  when built.
- A composed config can be the base of another compose. `compose -o` no longer replaces
  an existing file without `--force`, and never the base config or a file it inherits.
- `grasppanda probe`: a count below one and an unknown split are argument errors (exit
  status 2), not verdicts; a loss that is not finite in the very first iteration is
  `fails`, not `runs-unstable`.

### Changed

- `compose --help` and `probe --help` name the slots, the base, examples, the verdicts
  and the exit statuses; `grasppanda components <name>` names the presets of a component.
- The config hash of the HGGD configs changed (one more detector option); the model and
  its checkpoints did not.
- The config hash no longer includes the defaults of library classes (PyTorch's optimizers
  and schedulers, named by class), which differ between PyTorch versions: one config now
  hashes the same on every machine. Hashes written before this change differ; a checkpoint
  saved before it resumes with `runtime.allow_config_mismatch=true`, and
  `python -m tools.probe_grid rehash` rewrites the hashes of measured records.

### Known limits

- The model zoo is empty: no method has been trained and evaluated here to the admission
  rule yet. One author checkpoint has been scored on a whole test split (EconomicGrasp,
  Kinect); its result record is in `configs/economicgrasp/records/`.
- No option added beside the six ported methods has been trained and evaluated: a
  measured verdict says that a combination builds, trains for 20 iterations and predicts.
- Mixed precision (`train.amp`, preset `hooks=amp`) does not work with the shipped
  detectors: GSNet stops at a half/float mismatch in the label assigner, HGGD's first
  loss is not finite. `optimizer=soap` fails under HGGD with a CUDA error at the second
  step (it runs under GSNet). The collision filter that reads the full cloud fails under
  HGGD as shipped, whose dataset section keeps none; the message says how to supply it.
- A point augmentation that moves points stops at the first sample of a detector that
  reads images (HGGD); the config check does not see it.
- Most combinations of two or more replaced options have no measured record.
- The CPU-only wheel of spconv 2.3.8 miscomputes with more than one OpenMP thread; sparse
  layers detect this and stop with an explanation (see the installation guide).
