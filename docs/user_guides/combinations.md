# Try a combination

"GSNet with this encoder and that grouper" is four commands:

```bash
grasppanda components --slot backbones           # 1. what exists, with its contract
grasppanda compose <config> --set backbone=...   # 2. a checked config for the combination
grasppanda probe <composed config>               # 3. does it really train and predict?
grasppanda train <composed config>               # 4. the experiment itself
```

Steps 2 and 3 take about a minute together. They tell you whether a combination
**works**: it passes the contracts, builds, trains on real frames with finite losses and
predicts. They do not tell you how good it is. No AP is attached to anything on this
page; that takes step 4 and an evaluation on a benchmark split.

Which component was found to work under which method is on the generated page
[Measured combinations](../advanced_guides/measured_combinations.md); what a swap
changes besides the line in the config is in
[Swapping components](../advanced_guides/swapping.md). What each option of each slot is,
with a command per option, is in [Slots](../slots/index.md).

The same in a browser: the [combination explorer](../explorer/index.md) answers a
selection of method and options from the measured records and prints the commands, and
`grasppanda explore` serves it locally, with every selection checked by the `compose` of
your installation (nothing is trained).

## 1. See what exists

`grasppanda components` is the catalogue ([Components](../components.md)): every
registered component with the contract the config checker compares. `grasppanda compose
--list` shows the **presets**:

```text
$ grasppanda compose --list grouper
grouper  --set grouper=<preset or component>
preset                      component                     what it is
──────────────────────────  ────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────
ball                        LocalGroup                    LocalGroup with its defaults: a PointNet++ ball among the seeds, shared MLP and max.
box                         LocalGroup                    LocalGroup over a gripper-aligned box with cyclic padding: PointRCNN's RoI point pooling.
...                                                       (25 presets; shortened here)
voxel-set-abstraction       VoxelSetAbstractionGroup      VoxelSetAbstractionGroup with its defaults: PV-RCNN's keypoint feature for every seed.
  components without a preset: CylinderGroup, CylinderInteractionGroup

A component can also be named by its type (`--set grouper=CylinderGroup`): it is then
built with its constructor defaults, which `grasppanda components <type>` lists.
One preset with its notes: grasppanda compose --show <slot>=<preset>.
```

`grasppanda components <name>` ends with the presets of that component, and both commands
take the slot by either name (`grouper`, `groupers`).

A preset is the part of a config a newcomer cannot guess: the settings of one component
for table-top depth clouds of GraspNet-1Billion (grid size, radii, widths, sampling
rules) and, where its stage reads a label asset, the dataset section that serves it. Its
header says what must be re-tuned and why. Every scene encoder has one, because the
constructor defaults of most encoders are those of room-scale benchmarks: a bare
`{type: PTv3}` works on a 2 cm grid with six input channels. A stage whose constructor
defaults are the reference values of its method needs none.

```bash
grasppanda compose --show backbone=ptv3     # the file, with its notes
```

Presets live in `grasppanda/presets/<slot>/<name>.yaml`. They are config fragments, not
configs, and they are not inherited: two files of one `_base_` list may not set the same
value, and a preset by definition sets what the method's own model already sets. The
values of a preset therefore belong in the child config, which is where `compose` writes
them.

## 2. Compose

```text
$ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
      --set backbone=pointnet2 --set grouper=multi-scale -o gsnet_pointnet2_multi-scale.yaml
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=pointnet2 --set grouper=multi-scale

slot      component                        from
────────  ───────────────────────────────  ─────────────────────────────────────────────────────
backbone  PointNet2 in FeaturePropagation  preset `pointnet2`, in the adapter FeaturePropagation
proposal  GraspnessFPS                     base config
view      TemplateViewStage                base config
assigner  NearestLabelAssigner             base config
grouper   MultiScaleGroup                  preset `multi-scale`
head      SWADHead                         base config

what compose did
  • backbone: preset `pointnet2` (PointNet2). PointNet++ in the GraspNet-baseline layout: features
    on a 1024-point subsample.
  • grouper: preset `multi-scale` (MultiScaleGroup). MultiScaleGroup as FineGrasp uses it: a crop
    among the seeds at four radii, each by an EconomicGrasp crop module of its own, fused per seed
    by attention over the scales.
  • wrapped PointNet2 in the adapter FeaturePropagation (preset `feature-propagation`): GraspnessFPS
    requires 'dense' scene features, but PointNet2 returns 'subsampled' ones (features on a
    sub-sampled point set). An adapter changes the model; its notes below say how.

contracts
  ✓ backbone · proposal
  ✓ backbone · grouper
  ✓ proposal · view
  ✓ view · assigner
  ✓ assigner · head
  ✓ grouper · head
  ✓ view · head
  ✓ proposal · view · head

config check
  ✓ valid  (hash 76e5ab997651)

✓ written gsnet_pointnet2_multi-scale.yaml
  next: grasppanda probe gsnet_pointnet2_multi-scale.yaml
```

The file is a child config of the method config. Its header repeats the command, what
compose did and the re-tuning notes of every preset; its body is what you would have
written by hand (shortened here). Outside the shipped config tree the base is named by
its absolute path: the file can be moved on this machine, and on another one the command
in its header composes it again.

```yaml
_base_:
  - /…/configs/gsnet/gsnet_sunet14_g1b-realsense_10e.yaml

model:
  backbone:
    type: FeaturePropagation      # `subsampled` -> `dense`, written down
    mlp: [256, 256]
    neighbours: 3
    interpolation_power: 1.0
    backbone:
      type: PointNet2             # `subsampled`: features on 1024 points, 256 channels
      in_channels: 0              # coordinates only
      npoints: [2048, 1024, 512, 256]
      radii: [0.04, 0.1, 0.2, 0.3]
      nsamples: [64, 32, 16, 16]
      fps_skip_near_origin: true
      fps_tie_break: pointnet2_cuda
  grouper:
    type: MultiScaleGroup
    radius: 0.07                  # largest crop
    ...
```

What compose does beyond copying presets, each time saying so:

- **It inserts an encoder adapter** when the contract between encoder and proposal
  refuses the pair and exactly one registered adapter bridges it: `FeaturePropagation`
  around a sub-sampling encoder under a proposal that scores every point (above),
  `FurthestPointSubsample` around a dense encoder under a proposal that takes the
  encoder's subsample as its seeds. The adapter gets the seed count of the proposal and
  the sampling rules of the encoder it replaces. `--no-adapter` leaves the pair refused.
- **It fits a new encoder to the method.** Where the encoder has an `out_channels`
  option it is set to the seed width the grouper of the composed model fixes
  (`DepthCylinderGroup` with `seed_residual: gated` adds seed features to its crops) and
  otherwise to the width of the encoder it replaces, so the other stages keep the width
  they have in the reference model. Both widths are what the components declare
  (`Backbone.output_channels`, `Grouper.required_seed_channels`); nothing is built.
- **It puts the projected variant of a grouper in place** when the encoder has no such
  option and returns another width than the grouper fixes: `DepthCylinderProjectionGroup`,
  the same crop with a linear layer in front of its seed residual
  ([Grouper](../slots/grouper.md), section 5). That changes the model, and compose says
  so; `--no-adapter` leaves the pair refused.
- Where the method feeds real point features (`point_features: observation`,
  FineGrasp), `in_channels` is set to their number.
- **It adds the dataset section a stage reads**, when the preset carries one and the
  base config has none (`scale_statistic` for `ScaleBalancedAssigner`, `tolerance` for
  `OperationHead`, `graspness` for `GraspnessFPS`). The asset still has to be built; the
  preset says how. A component named by its *type* brings no section: the config check
  then refuses the config and compose names the preset that does.
- **It replaces, it does not merge.** A `--set` of the same component type as the base
  config's gets `_delete_: true`, so no option of the base leaks into what you asked for.
- **It does not overwrite.** `-o` refuses a file that exists unless you pass `--force`,
  and never writes onto the base config.

The base is a config that inherits fragments only, like the method configs under
`configs/<method>/`: a config chain is two levels deep, and the composed file is the
second. So a composed file cannot be inherited from, but it can be the *base of a
command*: compose reads the request in its header and composes again from its base with
the two requests merged.

```text
$ grasppanda compose gsnet_pointnet2_multi-scale.yaml --set head=swad --opts train.epochs=20 -o second.yaml
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=pointnet2 --set grouper=multi-scale --set head=swad --opts train.epochs=20
...
what compose did
  • gsnet_pointnet2_multi-scale.yaml is itself a composed config, and a config chain is limited to 2
    levels: this config is composed from its base, gsnet/gsnet_sunet14_g1b-realsense_10e, with its
    request and yours merged (a `--set` of yours replaces the one of the same slot).
```

A composed file you edited by hand, and a shipped child config such as the cross-dataset
`…-to-gc6d-d435` ones, are refused as a base with the config to compose from instead.

Further overrides go through `--opts`; a path below a slot you set addresses that
component wherever an adapter put it:

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=ptv3 \
    --opts model.backbone.grid_size=0.01 train.epochs=20 -o gsnet_ptv3_1cm.yaml
```

No rule about what fits what lives in compose. Every verdict is the contract rules of
the config checker, quoted, and a refused combination writes nothing. There are two
kinds of refusal.

**Stages that do not fit each other.** A head bundle comes with the assigner that feeds
it:

```text
$ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set head=economic
...
contracts
  ✓ backbone · proposal
  ✓ backbone · grouper
  ✓ proposal · view
  ✓ view · assigner
  ✗ assigner · head: NearestLabelAssigner returns 'dense' matched labels, but EconomicHead is
    supervised with 'compact' ones. A head bundle and the assigner that feeds it must agree on the
    label set; the dataset must serve that label set too.
  ✓ grouper · head
  ✓ view · head
  ✓ proposal · view · head
...
✗ refused: nothing was written. The messages above name the conflict and a fix.
```

Set the assigner too (`--set assigner=compact-label`) and the stages agree; what is then
left is the second kind.

**Stages that fit each other, on data that does not serve their labels.** A stage
declares what it reads from the training batch and a dataset what its config makes it
serve; the config check compares the two. GSNet's assigner and head pass every stage
contract inside EconomicGrasp, and are refused because EconomicGrasp's recipe serves
compact labels:

```text
$ grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
      --set assigner=NearestLabelAssigner --set head=swad
...
contracts
  ✓ backbone · proposal
  ✓ backbone · grouper
  ✓ proposal · view
  ✓ view · assigner
  ✓ assigner · head
  ✓ grouper · head
  ✓ view · head
  ✓ proposal · view · head

config check
  ✗ data.train.dataset: NearestLabelAssigner reads `supervision.objects`, which this training
    dataset does not serve. It serves compact labels in place of the dense label grids (its
    `compact_labels` section). A dataset serves one of the two label sets, and its transforms differ
    with it: the dense grids need a `SampleGraspLabels` transform and no `compact_labels` section,
    compact labels the section and no such transform.

✗ refused: nothing was written. The messages above name the conflict and a fix.
  → The data recipe is the base config's, and compose does not rewrite one. Shipped configs whose
    training data serves the labels of this request: gsnet/gsnet_sunet14_g1b-realsense_10e,
    gsnet/gsnet_sunet14_gc6d-d435_10e. Compose from one of them with the same `--set`.
```

The data recipe (which labels, how many points, the augmentation) belongs to the base
config. Compose adds a dataset *section* a preset brings; it does not turn a recipe for
compact labels into one for dense grids. The line after the arrow names the shipped
configs under whose recipe the very same request is served.

One more of the first kind, because it used to surface only when the model was built: an
encoder of another width under a grouper that fixes one. The grouper states the width it
needs, the encoder the width it returns, and the check compares the two. Compose takes
the way out by itself (above); with `--no-adapter` the pair is refused with both numbers:

```text
$ grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
      --set backbone=ptv3 --no-adapter
...
  ✗ backbone · grouper: DepthCylinderGroup as configured adds the seed features to its 256-channel
    crop features, but PTv3 returns 64-channel features. Use `type: DepthCylinderProjectionGroup`
    for the grouper (the same crop with a linear layer in front of the residual; `grasppanda
    compose` does that), or give the encoder 256 output channels where it has that option, or end
    the grouper's `mlp_channels` on 64, or set `seed_residual: none`.
```

**HGGD** is a detector family of its own. It has one component slot, the encoder of its
image network (`image_encoder`, see [Image encoders](../slots/image_encoders.md)); its other
parts are options of the detector, not registered components:

```text
$ grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --set image_encoder=resnet34
  • image_encoder: preset `resnet34` (TimmImageEncoder). ResNet-34 from timm with ImageNet-1k
    weights, colour only.
config check
  ✓ valid

$ grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --set backbone=ptv3
✗ HGGD has no slot 'backbone'. Its slots: image_encoder. HGGD is a detector family of its own: the
  stages of the seed-based methods (backbone, proposal, view, grouper, head) do not exist in it.
  Its other parts are options of the detector (`grasppanda components HGGD`); change them with
  `--opts model.<part>.<option>=<value>`.
```

Compose finds slots from the detector class (constructor parameters annotated
`Cfg[...]`, or `Cfg[...] | None` for a slot with a built-in default), so a detector whose
parts become registered components is composable without a change here. `--opts` and
`probe` work for HGGD as for every config.

## 3. Probe

```text
$ grasppanda probe gsnet_pointnet2_multi-scale.yaml
  config      gsnet_pointnet2_multi-scale.yaml  (hash 76e5ab997651)
  model       GSNet · 4,730,280 parameters
  data        8 frames of the config's dataset · 4 per iteration
  probe       as `grasppanda train` with data.train.dataset.split=sample data.train.num_workers=0 evaluation.interval=0 evaluation.save_best=null train.epochs=10 data.val.dataset.split=sample data.val.dataset.frame_step=4 data.val.num_workers=0
  device      cuda
  trained     20 of 20 iterations · 0.26 s per iteration (median, without data; data 0.31 s), measured on this machine now
  memory      peak 7.6 GiB on the GPU, this process
  losses      first → last
    loss/objectness             0.7068 → 0.6351
    loss/graspness              0.7984 → 0.3553
    loss/view                   1.885 → 0.3521
    loss/score                  0.2622 → 0.0127
    loss/width                  2.468 → 0.1995
    loss                        6.12 → 1.555
  seeds       real: min 1.00, mean 1.00 of the seed slots per iteration
              supervised: 1.00 → 1.00, min 1.00
  predicted   2 frame(s) · 233, 759 grasps

✓ runs: trained 20 iterations on 8 frames of the dataset and predicted on 2 frame(s), with finite losses and real seeds in every iteration.
  (16 s)
```

A probe is a miniature of `grasppanda train`, not a loop of its own: the real trainer,
the config's optimizer, schedule, accumulation, clipping and hooks, on the eight frames
of the split `sample` (scene 0 of GraspNet-1Billion, frames 0, 32, …, 224), then
prediction on two of them. Nothing is written. The line `probe` lists the overrides that
turn the config into that miniature; with them, `grasppanda train` runs the same thing.

**A probe is one run, and it does not repeat.** Run the command again and the hash, the
parameter count and the first column of losses are the same (the first losses to about
six digits); the last column, the seed fractions and the grasp counts are not. Training
on a GPU is not bit-reproducible here, and after the first optimizer step two runs are
two trajectories. Two probes of the shipped GSNet config on one machine: total loss 17.43
at the first iteration both times, 1.89 and 1.72 at the last; 233 and 279 grasps on the
two frames, then 164 and 188; the smallest fraction of real seeds 0.65, then 0.68. So
read the numbers as "of this run", and a verdict on the edge the same way: see *The
verdict*.

| Line | What it tells you |
|---|---|
| `losses` | Every loss term at the first and the last iteration. Falling on eight frames means the stages and the labels fit together, nothing more. |
| `seeds`, real | The fraction of seed slots that held a real seed. Below 1 the proposal let fewer points through than it has seeds; at 0 an iteration trained the proposal only. |
| `seeds`, supervised | The fraction of seeds that took part in the grasp losses: 1 for GSNet by construction, about a tenth for EconomicGrasp with an untrained proposal (a seed counts only within 5 mm of a label point), about half for GraspNet-baseline (seeds on an object). |
| `predicted` | Grasps per frame. A model trained for twenty iterations often predicts few or none, and the count changes from run to run; it is reported, not judged. An empty prediction still ran every stage: the detector drops the grasps of seeds that are not valid (no point passed the proposal, or the view stage classified the seed as background) at the very end. |
| `trained`, `memory` | Seconds per iteration and peak GPU memory of this process, on this machine at this moment. |

### The verdict

A probe ends with exactly one of five verdicts.

| Verdict | Meaning | Exit status |
|---|---|---|
| `runs` | On real frames: the detector was built, every requested training iteration ran with finite losses and at least one real seed, the weights stayed finite, and prediction returned one valid grasp group for every probed frame. A group may be empty: a model trained for a few iterations often predicts no grasp. | 0 |
| `runs-unstable` | On real frames: the detector was built, trained and predicted, but a loss or a weight became non-finite (a non-finite loss ends the training there), or an iteration had no real seed (no scene point passed the proposal, so only the proposal was trained), or no iteration supervised a seed, or fewer iterations ran or fewer frames were predicted than were asked for. | 1 |
| `fails` | An exception was raised after the config check had passed. The report names the phase (build, data, optimizer, loss, backward, step, hook, predict), the component it came from and the message. | 1 |
| `refused` | The config check refused the config (the same check as `grasppanda print-config --check --for train`). Nothing was built; the report carries the messages. | 2 |
| `wired` | No real frame was read. On a synthetic table-top scene with random labels the detector was built, back-propagated a loss through every stage and predicted. This shows that the stages connect; it says nothing about real data. | 0 with `--synthetic`, else 3 |

`runs-unstable` is what an untuned recipe looks like. A deep encoder under a learning
rate chosen for a shallow one can lose every seed for a few iterations and recover; the
verdict says which iterations, and the combination still works in the sense of this
page. Because a probe is one run and the seed fractions differ between runs, a
combination near that edge can be `runs` in one probe and `runs-unstable` in the next.
Treat the two verdicts as one answer, "it works", and the difference as a hint to watch
`metric/real_seeds` when you train.

`fails` is a combination that passed the config check and then broke. The probe says
where. The config check compares what components and datasets declare; what it cannot
know is whether options suit your data. An encoder named by its type instead of its
preset is built with its own defaults, and compose says so:

```text
$ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=OACNNs -o gsnet_oacnns_bare.yaml
...
what compose did
  • backbone: OACNNs with its constructor defaults; no preset was named. Its options are the
    component's own defaults, not a recipe for table-top clouds; that is what its preset holds:
    `--set backbone=oacnns`.
...
$ grasppanda probe gsnet_oacnns_bare.yaml
...
✗ fails: at loss (iteration 1), in the backbone OACNNs: ContractError: OACNNs: cloud 0 of the batch has no voxel left after 4 of 4 down-sampling steps. With crop_odd_extent=True the sparse grid is the bounding box of the cloud and loses the last cell of an odd axis at every level, so a cloud has to span at least 16 voxels along every axis. Use a smaller grid_size, or set crop_odd_extent=False to keep every voxel.
```

The default grid of OA-CNNs is 2 cm, for rooms; the preset sets 5 mm. That is what a
preset is for, and the probe is what tells you in a minute instead of after a queue.

A combination that fails for a reason the *contracts* could have known is a bug of the
contract rules: it should be refused before anything is built, or run. Two such families
were found by probing and are refused now, by the two rules shown in *Compose* above: a
model on a data recipe that does not serve its labels, and an encoder of another width
under a grouper that fixes one (which compose resolves with the grouper's projected
variant). If a probe shows you a third, please report it.

A probe that ends with `CUDA out of memory` is `fails` too, with a note: on a shared
machine that is a statement about the device at that moment, not about the combination.

### What a probe needs

The dataset and the label assets of its split `sample`. They are small; for
GraspNet-1Billion, RealSense:

```bash
grasppanda prepare graspnet1b --labels --split sample
grasppanda prepare graspnet1b --graspness --camera realsense --split sample
grasppanda prepare graspnet1b --compact-labels --split sample \
    --config economicgrasp/economicgrasp_sunet14_g1b-realsense_10e     # EconomicGrasp, FineGrasp
grasppanda prepare graspnet1b --graspness --camera realsense --split sample \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e             # FineGrasp
grasppanda prepare graspnet1b --tolerance --split sample \
    --config graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e
grasppanda prepare graspnet1b --view-grasps --camera realsense --split sample \
    --config hggd/hggd_r34w8_g1b-realsense_15e                         # HGGD
```

**The width histogram of Scale-Balanced-Grasp** is the exception. It is a statistic of
a whole training split, and eight frames leave bins of it empty, which its generator
refuses. So a config that reads it (`scale_statistic` in its training dataset: every
config composed from Scale-Balanced-Grasp, or with `--set assigner=scale-balanced`) is
probed with the reference's own file, which does not depend on the split. Import it once.
The import needs a config that names the file by its SHA-256, which compose writes:

```bash
SHA=$(sha256sum <reference repository>/statistic/scale_distribution.npy | cut -d' ' -f1)
grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
    --opts data.train.dataset.scale_statistic.source=legacy_import \
           data.train.dataset.scale_statistic.reference_sha256=$SHA -o sbg_reference_histogram.yaml
grasppanda prepare graspnet1b --scale-statistic --import-legacy \
    --scale-statistic-file <reference repository>/statistic/scale_distribution.npy \
    --config sbg_reference_histogram.yaml
```

From then on, either compose with those two `--opts`, or hand them to the probe:

```bash
grasppanda probe <config> --opts \
    data.train.dataset.scale_statistic.source=legacy_import \
    data.train.dataset.scale_statistic.reference_sha256=$SHA
```

Without it the probe of such a config says exactly this, runs the synthetic check and
ends `wired` (exit status 3). Training on the full split does not need the import:
`grasppanda prepare graspnet1b --scale-statistic --config <config>` builds the native
histogram of that split (about 3200 frames).

**Without the dataset, or without an asset,** the probe says what is missing and how to
build it, falls back to a wiring check on a synthetic scene and reports `wired`, never
`runs`:

```text
$ grasppanda probe gsnet_pointnet2_multi-scale.yaml --opts data.train.dataset.root=/data/not-mounted
  config      gsnet_pointnet2_multi-scale.yaml  (hash 3f3ae3d710e7)
  overrides   data.train.dataset.root=/data/not-mounted
  model       GSNet · 4,730,280 parameters
  dataset     NOT READ: the dataset is not on this machine. GraspNet-1Billion not found at /data/not-mounted. Set `root` in the dataset config, or the environment variable GRASPPANDA_GRASPNET1B_ROOT.
  data        SYNTHETIC scene with random labels: a wiring check, not a result
  device      cuda
  note        wiring check: model.proposal.graspness_threshold was lowered so that the untrained proposal lets points through to the stages behind it.
  trained     2 of 2 iterations · 0.15 s per iteration (median, without data; data 0.00 s), measured on this machine now
  memory      peak 3.9 GiB on the GPU, this process
  losses      first → last
    loss/objectness             0.9192 → 0.831
    loss/graspness              2.883 → 1.578
    loss/view                   16.27 → 13.15
    loss/score                  1.915 → 1.383
    loss/width                  1.982 → 1.154
    loss                        23.97 → 18.1
  seeds       supervised: 1.00 → 1.00, min 1.00
  predicted   2 frame(s) · 1024, 1024 grasps

✓ wired: trained 2 iterations on a synthetic scene with random labels and predicted on 2 frame(s), with finite losses.
  Real frames were not probed; `wired` says only that the stages connect.
  (5 s)
```

The synthetic scene is the ray-cast table top of `grasppanda demo`; its labels are noise.
`grasppanda probe <config> --synthetic` asks for that check directly, for example in a
project without data. `--json` prints the whole report for scripts; `--iters`,
`--frames`, `--predict-frames`, `--split`, `--device` and `--workers` change what is
run. A count below one or a split the dataset does not have is a wrong request, not a
verdict: the command says so and exits with status 2. `grasppanda probe --help` repeats
the verdicts, the exit statuses and what a probe needs.

## 4. Train

```bash
grasppanda train gsnet_pointnet2_multi-scale.yaml
```

The composed config is an ordinary config: everything in
[Training and testing](train_test.md) applies. What it is not is a tuned recipe. The
optimizer, schedule, batch size and point budget are the base method's, chosen by its
authors for its own stages; the header of the composed file lists what to re-tune first,
and `metric/real_seeds` and `metric/supervised_seeds` in the training log are the first
numbers to compare with a run of the reference config. A number from such a run has no
paper to be compared with: report it as trained here, next to the reference under the
same recipe, and call it experimental.

`configs/experimental/` holds eight such configs. They are compose's output for the
command in their header and are regenerated with the presets
(`python -m tools.make_combination_docs`).

## Reading the measured page

[Measured combinations](../advanced_guides/measured_combinations.md) is steps 2 and 3
run for a grid: every scene encoder under every seed-based method, every other stage
with one stage replaced at a time, a sample of the legal sets of stages (each composed
onto the method whose data recipe serves its labels), HGGD as it is and with every
image-encoder preset, the post-processing, loss and recipe presets under two or three
methods, and a few requests that replace several slots at once. The
[explorer](../explorer/index.md) shows the same records per selection. Every cell is a recorded
outcome; its two commands are in the measured file beside the page. Typing them repeats
the *experiment*: the same config (hash for hash), the same parameter count, the same
first losses. It does not repeat the later numbers: a cell is one run. In the tables

- `✓` is `runs`; `✓ own` is the method's own component (the cell composes to the shipped
  config, hash for hash); `✓ in Adapter` ran after compose inserted that adapter; `†`
  means compose also set `out_channels` or `in_channels` to fit the method;
- `≈ (n)` is `runs-unstable`, `✗ (n)` is `refused` (the config check's verdict: nothing
  was built) and `fails (n)` passed the config check and broke; `n` points to the reason
  below the table;
- the last column is the **measured class** of the component: `universal` (it works
  under all five seed-based methods), `most` (three or four), `some` (two), `specific`
  (exactly one), `none`. "Works" means `runs` or `runs-unstable`; `≈2` after a class
  says that two of those runs were unstable. A method's own component counts as
  working, so `specific` can be a stage that works where it is at home and nowhere
  else when it is replaced *alone*.

The table *Measured classes* near the top of the page lists every component under its
class, slot by slot: that is the short answer to "what works where".

A class is a statement about that grid: those presets, the methods' own recipes, one
stage replaced at a time. `specific` for a head bundle means that it cannot be swapped
in *alone*, because it needs its assigner and its labels; the table *Sets of stages*
shows combinations in which it runs (the sets are too many to probe each: the page says
how many were, and how they were picked), and *Several slots at once* a few requests
that also replace the encoder, a loss or the recipe. A set that reads dense label grids is refused
under EconomicGrasp and FineGrasp, whose recipes serve compact labels, and the reverse;
the page does not hold those cells, the config check says it when you compose one. And a
class says nothing about accuracy: `universal` means "builds, trains and predicts under
every method", not "is a good idea under every method".

The section *Passed the check, failed afterwards* lists the gaps this grid met:
combinations the config check accepted and a component then refused to build or run.
The toolbox treats each as a bug of its contract rules. An empty list says that no cell
of the grid is such a case; it is not a proof about configs outside the grid.

## Adding a preset

```yaml
# One sentence that says what this is.
#
# Free text: where it comes from, what it is not.
#
# What must be re-tuned, and why:
# * option: value, and the reason it will have to change
model:
  backbone:
    type: MyEncoder               # a comment that travels with the line
    grid_size: 0.005
```

Save it as `grasppanda/presets/<slot>/<name>.yaml`. A registered scene encoder without a
preset, a stage without a preset or a stated reason, and a component without measured
cells each fail a unit test that says what to run
(`tests/unit/test_combine_presets.py`, `tests/unit/test_combine_grid.py`).
