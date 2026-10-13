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
[Swapping components](../advanced_guides/swapping.md).

## 1. See what exists

`grasppanda components` is the catalogue ([Components](../components.md)): every
registered component with the contract the config checker compares. `grasppanda compose
--list` shows the **presets**:

```text
$ grasppanda compose --list grouper
grouper  --set grouper=<preset or component>
preset                      component           what it is
──────────────────────────  ──────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────
depth-cylinder-multi-scale  DepthCylinderGroup  DepthCylinderGroup as Scale-Balanced-Grasp uses it: scene-cloud cylinders at four radii per …
multi-scale                 MultiScaleGroup     MultiScaleGroup as FineGrasp uses it: a crop among the seeds at four radii, each by an …
  components without a preset: CylinderGroup, CylinderInteractionGroup

A component can also be named by its type (`--set grouper=CylinderGroup`): it is then
built with its constructor defaults, which `grasppanda components <type>` lists.
One preset with its notes: grasppanda compose --show <slot>=<preset>.
```

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
written by hand (shortened here; outside the shipped config tree the base is named by
its absolute path, so the file works from any directory):

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
  option it is set to the width of the encoder it replaces, so the other stages keep the
  width they have in the reference model. Where the method feeds real point features
  (`point_features: observation`, FineGrasp), `in_channels` is set to their number.
- **It adds the dataset section a stage reads**, when the preset carries one and the
  base config has none (`scale_statistic` for `ScaleBalancedAssigner`, `tolerance` for
  `OperationHead`, …). The asset still has to be built; the preset says how.
- **It replaces, it does not merge.** A `--set` of the same component type as the base
  config's gets `_delete_: true`, so no option of the base leaks into what you asked for.

Further overrides go through `--opts`; a path below a slot you set addresses that
component wherever an adapter put it:

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=ptv3 \
    --opts model.backbone.grid_size=0.01 train.epochs=20 -o gsnet_ptv3_1cm.yaml
```

No rule about what fits what lives in compose. Every verdict is the contract rules of
the config checker, quoted, and a refused combination writes nothing:

```text
$ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set head=economic
...
contracts
  ✓ backbone · proposal
  ✓ proposal · view
  ✓ view · assigner
  ✗ assigner · head: NearestLabelAssigner returns 'dense' matched labels, but EconomicHead is
    supervised with 'compact' ones. A head bundle and the assigner that feeds it must agree on the
    label set; the dataset must serve that label set too.
  ✓ grouper · head
  ✓ view · head
  ✓ proposal · view · head

config check
  ✗ data.train.dataset: transforms[2] (SampleGraspLabels) reads the dense label grids, which a
    dataset with `compact_labels` does not serve. Remove one of the two.

✗ refused: nothing was written. The messages above name the conflict and a fix.
```

Both messages say the same thing from two sides: EconomicGrasp's head comes with its
assigner and with a dataset that serves compact labels. Start from an EconomicGrasp or
FineGrasp config when you want that head.

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

| Line | What it tells you |
|---|---|
| `losses` | Every loss term at the first and the last iteration. Falling on eight frames means the stages and the labels fit together, nothing more. |
| `seeds`, real | The fraction of seed slots that held a real seed. Below 1 the proposal let fewer points through than it has seeds; at 0 an iteration trained the proposal only. |
| `seeds`, supervised | The fraction of seeds that took part in the grasp losses: 1 for GSNet by construction, about a tenth for EconomicGrasp with an untrained proposal (a seed counts only within 5 mm of a label point), about half for GraspNet-baseline (seeds on an object). |
| `predicted` | Grasps per frame. A model trained for twenty iterations often predicts few or none; the count is reported, not judged. |
| `trained`, `memory` | Seconds per iteration and peak GPU memory of this process, on this machine at this moment. |

### The verdict

A probe ends with exactly one of five verdicts.

| Verdict | Meaning | Exit status |
|---|---|---|
| `runs` | On real frames: the detector was built, every requested training iteration ran with finite losses and at least one real seed, the weights stayed finite, and prediction returned valid grasp groups for every probed frame. | 0 |
| `runs-unstable` | On real frames: training and prediction ran to the end, but a loss or a weight became non-finite, or an iteration had no real seed (no scene point passed the proposal, so only the proposal was trained), or no iteration supervised a seed. | 1 |
| `fails` | An exception was raised after the config check had passed. The report names the phase (build, data, optimizer, loss, backward, step, hook, predict), the component it came from and the message. | 1 |
| `refused` | The config check refused the config (the same check as `grasppanda print-config --check --for train`). Nothing was built; the report carries the messages. | 2 |
| `wired` | No real frame was read. On a synthetic table-top scene with random labels the detector was built, back-propagated a loss through every stage and predicted. This shows that the stages connect; it says nothing about real data. | 0 with `--synthetic`, else 3 |

`runs-unstable` is what an untuned recipe looks like. A deep encoder under a learning
rate chosen for a shallow one can lose every seed for a few iterations and recover; the
verdict says which iterations, and the combination still works in the sense of this
page.

`fails` is a combination that passed the config check and then broke. The probe says
where. EconomicGrasp with four attention heads in its crop composes (`✓ valid`), and then
(output shortened to its last line):

```text
$ grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
      --set grouper=CylinderInteractionGroup --opts model.grouper.attention_heads=4 -o eg.yaml
$ grasppanda probe eg.yaml
✗ fails: at build, in the grouper CylinderInteractionGroup: ValueError: num_heads must divide channels, got channels=259, num_heads=4
```

This is a gap of the config check, not the intended behaviour: a combination should be
refused before anything is built, or run. The check compares what components declare, and
a value that is only known once a component is built escapes it. The measured page lists
the cells it affects under *Passed the check, failed afterwards*; the probe is what catches
them, with the fix in the message.

One such gap was closed by a contract: Scale-Balanced-Grasp's crop adds the seed features
to its 256-channel crop features, so the encoder has to return 256 channels. The grouper
states that width, the check compares it with the encoder's (building the encoder once
when its config does not say), and for an encoder of another width `grasppanda compose`
puts `DepthCylinderProjectionGroup` in its place, the same crop with a linear layer in
front of the residual ([Grouper](../slots/grouper.md), section 5).

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
refuses. For a probe, import the reference's own file once
(`configs/scale_balanced_grasp/README.md`) and name it:

```bash
grasppanda probe <config> --opts \
    data.train.dataset.scale_statistic.source=legacy_import \
    data.train.dataset.scale_statistic.reference_sha256=<SHA-256 of the file>
```

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
run.

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
with one stage replaced at a time, the legal sets of stages, and HGGD as it is. Every
cell is a recorded run; its two commands are in the measured file beside the page, and
typing them repeats it. In the tables

- `✓` is `runs`; `✓ own` is the method's own component (the cell composes to the shipped
  config, hash for hash); `✓ in Adapter` ran after compose inserted that adapter; `†`
  means compose also set `out_channels` or `in_channels` to fit the method;
- `≈ (n)` is `runs-unstable`, `✗ (n)` is `refused` and `fails (n)` passed the config
  check and broke; `n` points to the reason below the table;
- the last column is the **measured class** of the component: `universal` (it works
  under all five seed-based methods), `most` (three or four), `some` (two), `specific`
  (exactly one), `none`. "Works" means `runs` or `runs-unstable`; `≈2` after a class
  says that two of those runs were unstable.

The table *Measured classes* near the top of the page lists every component under its
class, slot by slot: that is the short answer to "what can I put where".

A class is a statement about that grid: those presets, the methods' own recipes, one
stage replaced at a time. `specific` for a head bundle means that it cannot be swapped
in *alone*, because it needs its assigner and its labels; the stage sets at the end of
the page show the combinations in which it runs. And a class says nothing about
accuracy: `universal` means "builds, trains and predicts under every method", not "is a
good idea under every method".

The section *Passed the check, failed afterwards* is the list of gaps: combinations the
config check accepts and a component then refuses to build or run. The toolbox treats
each as a bug of its contract rules; what is listed there is known and not yet closed.

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
