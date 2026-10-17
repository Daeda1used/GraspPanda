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
- The explorer's tab *Run a method*: `grasppanda explore --weights-dir DIR` (or
  `--weights NAME=PATH`) runs a ported method, or the composed selection, on the example
  scene or on an uploaded depth image with its intrinsics, and shows the picture, the best
  grasps, the counts through the post-processing and the `(G, 17)` array as a download, as
  `grasppanda demo` does. Weights only from the files the command line named, uploads up
  to 20 MB in a temporary directory, one run at a time. Greyed on the documentation site,
  with the command that enables it.
- An animated README picture (`resources/showcase-{light,dark}.svg`) generated by
  `tools/make_showcase.py` from a captured transcript of the three commands it shows.
- The README opens with a recording of the explorer in use, light and dark
  (`resources/explorer-{light,dark}.webp`), made by `tools/record_explorer_demo.py` from the
  built documentation site.
- The explorer redesigned as a product interface: a header with the verdict mix of the
  records and its caveats behind an information button; Build as a configuration column
  (method tiles, the pipeline as slot rows, a searchable picker per slot, a bottom sheet on
  a phone) beside a result card (status, changes, the four levels, numbers, the fall of
  every loss term, alerts, the commands in a terminal, folded details); Compare as a slot
  sidebar and a matrix of tiles at the chosen level with tooltips and a cell menu; Run with
  numbered steps, drop zones and a result card. One SVG icon set, colour tokens with dark
  variants, keyboard support throughout; `grasppanda explore` gets an app bar and a dark
  mode. The answer of the contract check stays an outline, apart from measured verdicts.

### Measured records

- 1262 cells (1014 `runs`, 174 `runs-empty`, 29 `runs-unstable`, each of those probed a
  second time, 41 `refused`, 4 `fails`), every one probed on real frames with judge 2 and
  the constant schedule. Beside one slot replaced at a time under every method the grid
  has groups for the image encoders of HGGD, every post-processing preset under all six
  methods, every loss preset in one term of every kind it fits, every recipe preset under
  every method it applies to, and `cross`: eighteen requests that replace an encoder
  together with options of other slots, at least two per method (all eighteen run).
- The contract rules accept 9104 sets of stage variants; 95 were probed, picked so that
  every stage variant appears in a probed set under each kind of encoder, and the others
  are counted as not probed.
- Every record names the run that measured it; a probe that runs out of GPU memory is
  retried alone and, if memory stays short, not recorded.
- `grasppanda probe` tells a stage that learns from one that is only wired in (judge 2):
  it keeps the loss of every iteration, gives every loss term its fall over the
  iterations in which it was active with the flags `flat`, `rising`, `zero_at_end` and
  `constant`, names the sub-modules (depth 2) that no gradient reached and the parts whose
  weights did not change, and reports the scores of the predicted grasps and how far the
  schedule lowered the learning rate within the probe. A run that predicts no grasp on any
  frame is the verdict `runs-empty` (exit status 0); the rest is reported, not judged.
  `tools/probe_grid.py` exports all of it, with the per-iteration losses in a side file;
  the records on the measured page were made by the first judge and say so until the grid
  is probed again.
- `grasppanda probe --schedule constant` trains every step at the recipe's peak learning
  rate instead of running the recipe's schedule over the few epochs of a probe, which
  decays GSNet to 0.63 and EconomicGrasp to 2.4 % of its rate within 20 iterations and
  keeps FineGrasp in its warm-up. The optimizer, its weight decay, clipping and hooks stay
  the recipe's and the config hash is unchanged; the report states the schedule and the
  rate. `--schedule recipe` (the default) runs the recipe as shipped.
- The measured grid covers post-processing, recipe and loss presets under every method
  they apply to: every post-processing preset under all six methods (84 cells), every recipe
  preset under all six (the five seed-based ones for a preset that names the scene encoder;
  217 cells), every loss preset in one term of every kind it accepts under every seed-based
  method with such a term (the last term of the kind in the pipeline), beside the
  hand-picked loss cells (134 cells). The 302 new cells are listed as not probed yet until
  their records are exported. Grid probes run with `--schedule constant`, except a cell
  whose request sets the schedule. The four groupers that need an option only a preset
  supplies no longer have cells under their type names (20 refusals that were not
  combinations); their presets keep theirs. `python -m tools.probe_grid refresh` rewrites
  the measured file for a changed grid without probing.
- `python -m tools.probe_grid run` takes `--predict-frames` (the frames every probe
  predicts on) and `--seed` (`runtime.seed` of every probe). A seeded run records the hash
  of the config with that seed, so probes of two seeds never stand for each other; without
  either option the commands and records are those of before.
- Level 2 of the measured grid, `python -m tools.learn_check`: does an option learn like
  the stage it replaces? A cell whose level-1 verdict is `runs`, `runs-empty` or
  `runs-unstable` trains for 250 iterations on the same 8 frames (seed 0, `--schedule
  constant`, prediction on 4 frames), and the mean of every loss term over the last 25
  iterations is put beside that of the method as shipped under the same settings. Outcome
  per cell: `on a par`, `worse`, `diverging`, `not comparable` (every term redefined by the
  cell), `fails` or `no reference`; a term the cell redefines (a loss preset, a head bundle,
  an assigner, a proposal or view stage that computes its loss with other code) is marked
  not comparable and the cell is judged on the others. The tolerance of a method is
  calibrated from its reference trained under two seeds: 1.25, or the first multiple of
  0.05 above the largest seed ratio of a term (EconomicGrasp 1.35, HGGD 1.6); the rising
  threshold likewise. Post-processing, which trains nothing, is left out. The runs are
  probe_grid run directories (`L2-<kind>`), resumable; `export` writes per-cell files
  under `$GRASPPANDA_CACHE/verification/L2/` and the committed summary
  `docs/advanced_guides/learns_like_reference.json`, `page` the page "Learns like the
  reference?" (in the docs nav), checked by the unit suite like the measured page. The
  six references and a pilot of 27 cells (3 per kind) are recorded.

### Added: classes of levels, and the options that fit each method

A class used to count the methods under which a component *worked* (built, trained 20
iterations, predicted), so "universal" meant "accepted by the contracts and did not crash".
Classes now count levels, and every method lists the options of each of its slots.

- `grasppanda.combine.levels` decides every measured cell at four levels from its record
  alone: **composes** (the config check accepts it), **runs** (the probe ran clean;
  `runs-unstable` counts as not running only after a repeat of the probe, read from a
  `repeat` field of the record, and is *unstable, not repeated* until then), **learns**
  (judge 2: every loss term active in at least half of the iterations fell by at least
  5 %, none flat, rising or constant, every unit reached by a gradient, every part
  changed) and **predicts** (judge 2: grasps on every predicted frame, scores not all
  the same). A record of the first judge runs at most and says *not checked* above. The
  class of an option is counted per level: `universal` only when it reached the level
  under every method it applies to, which include the five seed-based ones; `most`,
  `some`, `specific`, `none` for a stage decided under all five; else *N of M probed*,
  with the undecided methods named. A refusal by policy (`trust_remote_code`) waits for
  consent and is never `none`; a refusal shows its kind (structural, not-implemented,
  policy, not-a-combination) where the record has one. `grasppanda.combine.options`
  lists the options of every slot of a method from the grid and the records.
- The measured page and the slot index show the classes per level, one table per level,
  each with the sentence "one run, seed 0, 20 iterations, 8 frames of one scene; a class
  counts verdicts, not accuracy"; the definitions and thresholds are printed from
  `grasppanda.combine.levels`. Every probed cell shows its grasps per frame and, for
  judge 2, the fall of its loss terms and the levels it missed; a probe that predicted
  no grasp is greyed. The numbers below a table give the fall of every term and what kept
  a run from *learns* or *predicts*.
- `grasppanda compose <method config> --list [<slot>]` lists only the options of that
  method's slots, each with its verdict, levels, class, grasps per frame and fall per
  loss term; the options that do not fit with the reason, kind and lift; the options
  without a record as *not probed*, with the reason (`--json`: everything, every message
  whole). `--list` without a config still lists the presets. The wheel ships the measured
  records for it (`grasppanda/_measured/`).
- The explorer offers, in the lanes of a method, exactly the options of that method's
  list, shows the four levels of every record, the grasps per frame in every cell of the
  grid (greyed without a grasp), the class of an option per level with a level picker for
  the filters, the reason an option has no cell, and the kind of a refusal where compose
  reports one.
- Every shipped method config names `grasppanda compose <config> --list <slot>` for each
  of its slots, in a comment block and above the slot in its bases (config hashes
  unchanged).
- A generated page per method, `docs/methods/<method>.md` (`tools/make_method_pages.py`,
  with `--check` in the unit suite): every slot, what the method ships, the options that
  fit (verdict, levels, class, grasps, fall per term, the compose command), those that do
  not (reason, kind, lift) and those not probed. Linked from the method READMEs and from
  every slot page, whose class columns are now the class at *runs*, written from the
  records by the same tool. A test holds the explorer's lanes to the lists of `compose`.

### Added: authors' checkpoints on whole test splits

- The model-zoo page (`docs/model_zoo.md`) has two tables of `author-ckpt` numbers, the
  authors' released weights scored by this toolbox, with and without collision filter and
  with the reported number beside each. "Authors' checkpoints scored by this toolbox":
  GraspNet-1Billion, the whole test split (23,040 frames), for GraspNet-baseline,
  Scale-Balanced-Grasp, HGGD, EconomicGrasp and FineGrasp on RealSense and for HGGD and
  EconomicGrasp on Kinect. "GraspClutter6D transfers": the whole test split (3,055
  frames) for GraspNet-baseline, Scale-Balanced-Grasp, EconomicGrasp and HGGD. They are
  not reproductions. Each method's README states every difference from the reported
  number and what is and is not known about it.
- `python -m tools.record_result <run directory> --method <configs directory>` puts a
  whole-split evaluation into the zoo: it checks the run (every frame scored, clean code,
  hashed weights that are the file the metafile lists), copies its result records to
  `configs/<method>/records/` without the paths of the machine, and prints the metafile
  rows with the numbers computed from the per-scene values.
- A metafile row names the splits of its protocol (`seen` / `similar` / `novel`, or `test`
  for GraspClutter6D) and may state `AP0.8` and `AP0.4`; a measured row whose record file
  is missing is a validation error. One test holds every measured row of every method to
  its record: whole split, clean code, and exactly the numbers of the row.

### Added: the first model trained here

- GSNet, trained from scratch with the recipe of `gsnet/gsnet_sunet14_g1b-realsense_10e`
  (10 epochs, seed 0), is the first `reproduced` entry of the zoo: on the whole
  GraspNet-1Billion test split 68.83 / 60.35 / 25.20 AP without and 71.23 / 62.29 / 26.42
  with the collision filter, against 65.70 / 53.75 / 23.98 and 67.12 / 54.81 / 24.31 in the
  paper; on the whole GraspClutter6D test split 25.58 / 31.38 / 19.42 and 26.87 / 33.00 /
  20.04 (AP / AP0.8 / AP0.4). It is one training run with one seed, above the paper by
  more than the 1 AP of the admission gate, and the gap is not explained;
  `configs/gsnet/README.md` says what is known about it.
- Its weights are not published yet. A weights entry of `source: grasppanda` may have
  `url: null` with a `note` saying so, beside its sha256; the zoo, `GraspInferencer` and
  `grasppanda demo <config>` without `--weights` then say that the weights are not
  published instead of failing on the missing link.

### Added: refusals that say why, and the ones compose lifts

- Every refusal of the stage contracts and of the config check carries a kind:
  `structural` (a genuine incompatibility), `not-implemented` (the lift is named),
  `fixed-by-compose`, `not-a-combination` (a component without a required option, options
  that disagree) or `policy`. `grasppanda compose` prints it before each message and lists
  it in `refusal_kinds` of its `--json` report; the pair tables of the swapping page start
  each reason with it.
- `grasppanda compose` now inserts the axis adapter between a grouper and a head bundle
  whose axes differ (`DepthEmbedGroup` repeats a crop over the depth bins, `DepthPoolGroup`
  pools over them), sets the assigner options a stage reads, lets a scoring proposal
  predict graspness only next to `ObjectnessViewStage` (`ScoredSeeds` with `select_by:
  graspness_only`, a new value of that option), sets the decode option of a head whose
  score no stage predicts, and writes the label set of the head bundle into the assigner
  and the data recipe (`--set labels=dense|compact` asks for it). `--unset key.path` puts an
  option back to the default of its component. A data recipe refusal that remains prints
  the exact `--set`, `--unset` and `--opts` that rewrite it. `--no-adapter` leaves every
  such pair refused. Of the 198 refused cells of the earlier records, compose accepts 142;
  the records now hold their probes.
- `SWADHead`, `EconomicHead`, `DecoupledGridHead` and `TemplateViewStage` apply a weight
  per seed (a weighted mean over their supervised entries, as `OperationHead`), so
  `ScaleBalancedAssigner` combines with every shipped stage. A stage that does not apply
  one is reported as a warning, no longer refused.

### Added: the conversion command in the metafile

- A metafile weights entry of `source: author` states its `convert` command, exactly as
  the method's README prints it (a test holds the two together); a model whose weights
  were trained here may list the checkpoint the authors released for it as
  `author_weights`. The zoo's note, which `grasppanda demo <config>` prints without
  `--weights`, now gives that command and the file to pass. HGGD's README converts the
  file under the name the authors publish it (`HGGD_realsense_checkpoint`), and
  EconomicGrasp's gives the RealSense command beside the Kinect one.

### Added: run a method in five minutes

- Every method page (`docs/methods/<method>.md`) starts with *Run <method> in five
  minutes*: the authors' file, its licence and sha256, and that the project publishes no
  weights converted from it; the conversion command and the file it writes; the demo on
  the example scene with what it prints and writes, the measured wall time and grasp
  counts; the same from Python (`GraspInferencer`, six statements); the browser route
  (`grasppanda explore --weights-dir`, tab *Run a method*); the test on every 64th frame of
  `test_seen` with its AP beside the whole-split row, its provenance, collision filter and
  ground-truth-derived inputs. Every measurement names its date, commit and machine.
  GSNet's section offers the checkpoint released with the reference code, which runs but
  is far below the paper (19.23 / 39.75 AP on the 120 frames, without and with the
  collision filter), beside the weights trained here, which are not published.
- `tools/make_method_pages.py` writes the sections from the metafile, the config and
  `docs/methods/five_minutes.json` (the measured runs; nothing measured is in the
  generator). `tests/unit/test_method_quickstart.py` takes the commands from the pages:
  each parses with the parser of its program and names what exists, the demo, its camera
  variant and the Python block run on the CPU over each method's real pipeline with a toy
  detector and give the same grasps, the browser route picks each converted file for its
  method, and the measured runs agree with the metafiles and the configs.
- The README quick start, `docs/get_started/quickstart.md`, the inference guide and every
  method README link the six sections.

### Fixed

- `train --resume <checkpoint>` with another `--work-dir` or `runtime.run_id`, or with
  `runtime.allow_config_mismatch`, wrote into the run directory of the checkpoint; with a
  config mismatch it replaced that run's later checkpoints and `best.pth`. Such a resume
  now starts a new run directory below `runtime.work_dir`, whose `run.json` records
  `resumed_from`, and leaves the original run as it is; a mismatch without the flag is
  refused before anything is written. Training longer than planned is therefore a new run
  directory.
- A `_base_` entry that is not a file next to the config names a shipped config
  (`_base_/default_runtime.yaml`, `gsnet/gsnet_sunet14_g1b-realsense_10e.yaml`), so a
  child of a shipped config and the example project load from any directory and from an
  installed wheel; the example project inherits by name. A missing base is reported at the
  file that names it.
- With the CPU-only spconv wheel and more than one thread (the default), every config with
  a sparse encoder stopped at the self-check, and 163 unit tests failed. The wheel's
  convolutions and their backward now run on one PyTorch thread, the thread count is
  restored for the other operators, and the log of a run and the `probe` report say so
  once. The self-check stays and now compares the input gradient too. A synthetic probe of
  GSNet with PTv3 on the CPU with three threads: `wired` in 18 s (it stopped at the
  self-check before; 23 s with `OMP_NUM_THREADS=1`).
- `infer --input DIR` on a plain folder of frames ended in a raw traceback from the
  reader. It now checks the directory before anything runs (the reader of the section has
  a `root` parameter and reads its first frame) and otherwise stops with an `InputError`
  that names what the reader looked for and points to `demo --depth FILE --intrinsics` and
  `GraspInferencer`; `demo --depth DIR` says the same.
- A data loader with workers below a `TMPDIR` too deep for the sockets of its workers
  (beyond about 75 characters) waited forever for its first batch: every worker dropped
  its batches with `AF_UNIX path too long`, which only the worker printed. The loaders of
  `train` and `test` now refuse such a setting with the reason and the fix
  (`docs/faq.md`).
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

- Combinations that passed the config check and failed in the first iteration are now
  refused with the conflict and the fix: a head bundle or a view stage on another grid
  than the labels of the dataset (`num_angle`, `num_depth`, `num_views`, `label_views`);
  `SubsampledSeeds.num_seeds` against the points a sub-sampling encoder returns
  (`output_points`); `top_k` and `friction_range` out of range (`option_problems`); a
  point transform that moves points on a dataset section that serves images
  (`moves_points`); a post-processor that needs the cloud before sub-sampling on a
  section that drops it (`reads_full_points`; `compose --set post=collision...` switches
  `keep_full_points` on and says so); `train.amp: true` for a detector that does not
  declare `amp_ready`.
- `ASSANet` returned zeros for the constant input features most methods feed (its stem
  batch-normalises them) and was recorded as `runs`. It has the option `use_xyz` (on in
  the preset) and refuses constant features without it; a unit test runs every encoder
  preset on constant features.
- `grasppanda probe` on real frames records the parts of the model that received no
  gradient (`no_gradient`) and reports `runs-unstable` for one; it prints what every step
  of a post-processing chain left; a CUDA launch failure gets the note that out-of-memory
  has. The synthetic wiring check labels its scene on the grid of the dataset and reads
  the image size of a GraspClutter6D camera.
- Image-encoder bridges print one line when their library fetches pretrained weights,
  with the licence it states. The presets with non-commercial weights (`convnextv2-tiny`,
  `convnextv2-atto`, `hf-hiera-tiny`) ship with `pretrained: false`; a model that runs
  code from its hub repository (`hf-radio-v3-base`) is refused until the config says
  `accept_remote_code: true`.
- `compose --list image_encoder` lists the image encoders; a head refused for its label
  set names the shipped configs that serve it at once; `--set image_encoder=` under a
  seed-based method points to point painting.
- Install hints name `pip install "grasppanda[<extra>]"` when GraspPanda runs from an
  installed wheel and `pip install -e ".[<extra>]"` from the repository root of a checkout.
  For spconv they name the build that matches the installed PyTorch (`spconv-cu118`, ...,
  `spconv-cpu` for a CPU-only PyTorch) and list the others, instead of always
  `spconv-cu118`.
- `tools/probe_grid.py` refuses to record from a working tree with uncommitted changes
  (`--allow-dirty` overrides) and records the versions of the optional libraries of a run.
- `approach_to_rotation` was rewritten independently (cross products and a turn about the
  approach axis); the convention and its tests are unchanged.

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

- No method has passed the admission rule of the zoo yet, and the zoo has no weights to
  download. GSNet, the one model trained here, is above the paper by more than 1 AP and
  its weights are not published; a training with the unmodified reference code was not
  run here. The other numbers are `author-ckpt` (see above): each is one run with seed 0,
  and where a number differs from the reported one the cause is not known.
- No option added beside the six ported methods has been trained and evaluated: a
  measured verdict says that a combination builds, trains for 20 iterations and predicts.
- Mixed precision (`train.amp`, preset `hooks=amp`) does not work with the shipped
  detectors and is refused by the config check: GSNet stopped at a half/float mismatch in
  the label assigner, HGGD's first loss was not finite. `optimizer=soap` fails under HGGD
  with a CUDA error at the second step (it runs under GSNet).
- No shipped augmentation moves image and cloud together, so a detector that reads images
  (HGGD) has no geometric augmentation; point transforms that move points are refused there.
- The point-painting wrappers (`image-painted`, `image-painted-subsampled`) are not written
  by `compose` and have no cell in the measured records; the table of their probes in
  `docs/slots/image_encoders.md` was measured once by hand.
- `grasppanda probe --synthetic` (also the fallback when a dataset is missing) can report
  `fails` for FineGrasp with the encoders `spunet` and `minkunet34c-pointcontrast`, whose
  record on real frames is `runs`: after the second optimizer step on the random labels of
  the synthetic scene the attention of the grouper returns a value that is not finite. The
  cause is not found.
- 51 measured cells (48 sets of stages and 3 others) were probed from a working tree with
  uncommitted changes (`run` ends in `+dirty`): the commit they name does not identify the
  probed code exactly. The runner refuses that from now on.
- The unit tests of `OctFormer`, `PointMamba`, the HEALPix view stages and
  `hf-radio-v3-base` need `ocnn`, `mamba-ssm`, `healpix` and `open_clip_torch`, which the
  shared test environments do not hold: those tests skip there, and the measured cells of
  those options were probed with the libraries in a private environment. Runs recorded
  before this change do not name the versions of the optional libraries.
- The strict-loading checks of released third-party weights (integration tests) skip when
  the files are absent; they were run once with the files and are not repeated by CI.
- The slot pages differ in how they lay out their tables (method names as columns on
  some, abbreviations or one slash-separated cell on others).
- Most combinations of two or more replaced options have no measured record.
- The CPU-only wheel of spconv 2.3.8 miscomputes with more than one OpenMP thread; sparse
  layers detect this and stop with an explanation (see the installation guide).
