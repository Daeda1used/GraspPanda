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
your installation (nothing is trained). Started with weights
(`grasppanda explore --weights-dir <dir>`), its tab *Run a method* also runs a ported
method, or your selection when the weights fit it, on one frame and shows the grasps:
[Run in the browser](inference.md#run-in-the-browser).

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

### The options that fit one method

Name a method config before `--list` and compose lists only the options of that method's
slots, each with what was measured for it: the verdict of its probe, the four levels
*composes*, *runs*, *learns* and *predicts* under that method, its class at each level
across the methods it applies to, the grasps per frame and the fall of the loss terms;
the options that do not fit with the reason, its kind and the lift where the record has
them; the options without a record as *not probed*, with the reason when there is one:

```text
$ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --list view
GSNet · view  (model.view)
measured on gsnet/gsnet_sunet14_g1b-realsense_10e (one run, seed 0, 20 iterations, 8 frames of one scene; a class counts verdicts, not accuracy)
...
view  --set view=...   shipped: TemplateViewStage (option template-view)
  fit: 19 (the config check accepts them)
option                   verdict  C R L P  grasps/frame  term fall    class C R L P
───────────────────────  ───────  ───────  ────────────  ───────────  ─────────────
template-view (shipped)  runs     ✓ ✓ ? ?  233, 279      not checked  most most ? ?
direction-regression     runs     ✓ ✓ ? ?  1024, 1024    not checked  most most ? ?
...
  do not fit: 2 (refused by the config check)
    objectness-view: ObjectnessViewStage reads 'on_object' and 'view_width' of the matched labels, ...
```

Without a slot it lists every slot of the method; `--json` prints everything, every
refusal message whole. The same lists are a page per method ([GSNet](../methods/gsnet.md),
[HGGD](../methods/hggd.md), ...) and the lanes of a method in the
[explorer](../explorer/index.md), so the three always offer the same options. A config
composed from a method config is listed as that method, with a note that the options were
measured on the method config. The shipped method configs name the command above each
slot. In a wheel the records come with the package; in a checkout they are read from
`docs/advanced_guides/measured_combinations.json`.

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
- **It inserts an axis adapter** between a grouper and a head bundle whose feature axes
  differ: `DepthEmbedGroup` (preset `depth-embed`) repeats a `(seed, channel)` crop over
  the depth bins of the head with a learned embedding of the bin, `DepthPoolGroup`
  (preset `depth-pool`) takes the maximum of a `(seed, depth, channel)` crop over the bins.
  `--no-adapter` leaves the pair refused.
- **It sets the options a stage declares for the other stages**: the assigner options
  that return a field a stage reads (`supervise: object_seeds`, `view_widths: true` for
  `ObjectnessViewStage`); graspness only for a scoring proposal next to
  `ObjectnessViewStage`, which then owns objectness (`ScoredSeeds` with `select_by:
  graspness_only`; `GraspnessFPS` keeps the options of the reference, so it is replaced by
  that `ScoredSeeds`); and the decode option of a head that would multiply in a score no
  stage predicts (`OperationHead` with `score_decode: score_x_tolerance`). Each changes the
  model and is one sentence of the report; `--no-adapter` leaves the pair refused.
- **It writes the label set of the head bundle into the recipe.** A head of dense label
  grids on a recipe of compact labels, or the other way round, gets the dataset section
  of a shipped recipe that serves them (the `compact_labels` section, or the
  `SampleGraspLabels` step) and the assigner of a shipped config that returns them; an
  assigner you set yourself is kept. `--set labels=dense` or `--set labels=compact` asks
  for it in so many words. Compact labels are an asset to build (`grasppanda prepare
  <dataset> --compact-labels --config <the composed config>`); dense grids are read from
  the label store of `grasppanda prepare <dataset> --labels`.
- **It adds the dataset section a stage reads**, when the preset carries one and the
  base config has none (`scale_statistic` for `ScaleBalancedAssigner`, `tolerance` for
  `OperationHead`, `graspness` for `GraspnessFPS`). The asset still has to be built; the
  preset says how. A component named by its *type* brings no section: the config check
  then refuses the config and compose prints the `--opts` that adds it and the preset that
  carries it.
- **It puts an option back to its default** with `--unset key.path` (a child config
  cannot delete what it inherits, so the default of the component is written):
  `--unset data.train.dataset.compact_labels`.
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
the config checker, quoted, and a refused combination writes nothing. Each refusal says
in brackets why it is one ([Swapping](../advanced_guides/swapping.md), *Why a pair is
refused*): `structural` (the components cannot work together), `not-implemented` (a lift
not written yet, named in the message), `fixed-by-compose` (what compose inserts by
itself, met with `--no-adapter` or in a config written by hand), `not-a-combination` (a
component without a required option, two options that disagree) or `policy`. The
`--json` report lists the same in `refusal_kinds`, in the order of `refusal`.

**Stages that do not fit each other.** A head bundle supervised with compact labels, fed
by an assigner of dense ones that you set yourself:

```text
$ grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
      --set assigner=NearestLabelAssigner
...
contracts
  ✓ backbone · proposal
  ✓ backbone · grouper
  ✓ proposal · view
  ✓ view · assigner
  ✗ assigner · head [structural]: NearestLabelAssigner returns 'dense' matched labels, but
    EconomicHead is supervised with 'compact' ones. A head bundle and the assigner that feeds it
    must agree on the label set; the dataset must serve that label set too.
  ✓ grouper · head
  ✓ view · head
  ✓ proposal · view · head

config check
  ✗ data.train.dataset [fixed-by-compose]: NearestLabelAssigner reads `supervision.objects`, which
    this training dataset does not serve. It serves compact labels in place of the dense label grids
    (its `compact_labels` section). A dataset serves one of the two label sets, and its transforms
    differ with it: the dense grids need a `SampleGraspLabels` transform and no `compact_labels`
    section, compact labels the section and no such transform.

✗ refused: nothing was written. The messages above name the conflict and a fix.
  → EconomicHead reads compact labels and NearestLabelAssigner returns dense ones; the request set
    the assigner (`--set assigner=...`). Without it compose gives EconomicHead an assigner and
    training data of compact labels; or keep NearestLabelAssigner with a head bundle that reads
    dense labels (DecoupledGridHead, OperationHead, SWADHead).
  → To make the training data serve the dense labels NearestLabelAssigner reads: `--set
    labels=dense`, which writes `--unset data.train.dataset.compact_labels --opts
    'data.train.dataset.transforms=[{type: RandomSample, num_points: 20000}, {type:
    SampleGraspLabels, point_fraction: 0.25, min_points: 300, collision: zero_scores, grids:
    deferred}, {type: RandomFlip, axis: x, prob: 0.5}, {type: ConstantFeatures, channels: 3}]'` (the
    recipe of gsnet/gsnet_sunet14_g1b-realsense_10e).
  → Or compose from a shipped config whose training data serves all of it:
    gsnet/gsnet_sunet14_g1b-realsense_10e, gsnet/gsnet_sunet14_gc6d-d435_10e.
```

The label sets are a genuine difference: a grid of every grasp per point and view cannot
be recovered from one grasp. Leave the assigner to compose and the head brings its own
labels; swap the head instead and compose rewrites the recipe:

```text
$ grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e --set head=swad
...
what compose did
  • head: preset `swad` (SWADHead). SWADHead as GSNet uses it: a score and a width per (angle, depth
    bin) of every seed, regressed; one decoded grasp per seed.
  • labels: dense. SWADHead is supervised with dense labels, and the training data of the base
    config serves compact labels: data.train.dataset now serves dense ones, `compact_labels` unset
    and the SampleGraspLabels step of gsnet/gsnet_sunet14_g1b-realsense_10e inserted before
    RandomFlip. The dense grids are read from the label store of `grasppanda prepare <dataset>
    --labels`, which a probe or a run needs (the config check does not look for it). The rest of the
    data recipe is the base config's.
  • assigner: NearestLabelAssigner, the assigner of gsnet/gsnet_sunet14_g1b-realsense_10e.
    CompactLabelAssigner returns compact matched labels and SWADHead is supervised with dense ones.
```

The rest of the data recipe (how many points, the augmentation) stays the base config's.

**What compose would insert, refused without it.** With `--no-adapter`, or in a config
written by hand, a pair that an adapter or an option joins is refused, and the message
names it:

```text
$ grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
      --set grouper=ball --no-adapter
...
  ✗ grouper · head [fixed-by-compose]: LocalGroup emits features with axes ('seed', 'channel'), but
    OperationHead expects ('seed', 'depth', 'channel'). An axis is never added or squeezed
    implicitly: wrap the grouper in the axis adapter DepthEmbedGroup (preset `depth-embed`), `{type:
    DepthEmbedGroup, num_depth: <the head's num_depth>, grouper: {...}}`, which `grasppanda compose`
    inserts, or use a grouper and a head bundle that agree. Groupers that emit ('seed', 'depth',
    'channel'): DepthCylinderGroup, DepthCylinderProjectionGroup, DepthEmbedGroup, DepthLocalGroup.
    Head bundles that expect ('seed', 'channel'): DecoupledGridHead, EconomicHead, SWADHead.
```

A stage set that is accepted but trains differently from what its names promise gets a
warning in the report and in the header of the composed file: a stage of a project that
does not apply the weight per seed `ScaleBalancedAssigner` returns trains unweighted.

One more that compose lifts by itself, because it used to surface only when the model was
built: an encoder of another width under a grouper that fixes one. The grouper states the width it
needs, the encoder the width it returns, and the check compares the two. Compose takes
the way out by itself (above); with `--no-adapter` the pair is refused with both numbers:

```text
$ grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
      --set backbone=ptv3 --no-adapter
...
  ✗ backbone · grouper [fixed-by-compose]: DepthCylinderGroup as configured adds the seed features
    to its 256-channel crop features, but PTv3 returns 64-channel features. Use `type:
    DepthCylinderProjectionGroup` for the grouper (the same crop with a linear layer in front of the
    residual; `grasppanda compose` does that), or give the encoder 256 output channels where it has
    that option, or end the grouper's `mlp_channels` on 64, or set `seed_residual: none`.
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
  trained     20 of 20 iterations · 0.26 s per iteration (median, without data; data 0.34 s), measured on this machine now
  memory      peak 7.6 GiB on the GPU, this process
  losses      first → last · fall from the first active value to the mean of the last quarter
    loss/objectness             0.7068 → 0.6153       fell 9 %
    loss/graspness              0.7984 → 0.3903       fell 51 %
    loss/view                   1.885 → 0.3763        fell 79 %
    loss/score                  0.2622 → 0.0159       fell 93 %
    loss/width                  2.468 → 0.1509        fell 93 %
    loss                        6.12 → 1.549          fell 74 %
  seeds       real: min 1.00, mean 1.00 of the seed slots per iteration
              supervised: 1.00 → 1.00, min 1.00
  predicted   2 frame(s) · 103, 134 grasps
  scores      min / median / max per frame: 0.104 / 0.11 / 0.116; 0.0932 / 0.105 / 0.112
  gradient    reached every one of 9 units of 5 part(s)
  changed     every part
  lr          0.001 at the first step → 0.00063 at the last, ×0.63 at the last step

✓ runs: trained 20 iterations on 8 frames of the dataset and predicted on 2 frame(s), with finite losses and real seeds in every iteration.
  (20 s)
```

A probe is a miniature of `grasppanda train`, not a loop of its own: the real trainer,
the config's optimizer, schedule, accumulation, clipping and hooks, on the eight frames
of the split `sample` (scene 0 of GraspNet-1Billion, frames 0, 32, …, 224), then
prediction on two of them. Nothing is written. The line `probe` lists the overrides that
turn the config into that miniature; with them, `grasppanda train` runs the same thing.

**The schedule of a probe.** To fit 20 iterations the probe sets `train.epochs` to as few
epochs as the eight frames need, and the recipe's learning-rate schedule runs over those
epochs: GSNet's decay of 0.95 per epoch ends at 0.63 of its rate, EconomicGrasp's cosine
at 2.4 %, and FineGrasp's warm-up of 800 iterations keeps the whole probe below 2.5 % of
its rate. The trends of two methods are then not comparable. `--schedule constant`
removes the scheduler (`scheduler=null` among the overrides) and trains every step at the
recipe's **peak** rate: the rate of the optimizer (with the per-group multipliers of
`ParamGroups`), or `max_lr` for OneCycleLR and CyclicLR. When the probe's batches hold
fewer frames than the recipe's, that peak is scaled by their ratio (the linear scaling
rule; FineGrasp's 0.008 for batches of 32 becomes 0.002 on the 8 frames of a grid probe),
and a note says so. A batch-norm momentum that the recipe decays per epoch
(`BNMomentumHook` of GraspNet-baseline, Scale-Balanced-Grasp and HGGD) is held at its
first epoch's value the same way (`hooks.<i>.decay=1` among the overrides, and a note):
decayed over epochs of a few iterations it reaches its floor of 0.001 within the probe,
and the running statistics that prediction normalises with stop following the weights
(Scale-Balanced-Grasp then predicted no grasp after 250 iterations). The optimizer with
its weight decay, the clipping, the accumulation and every other hook stay the recipe's,
and the config hash does not change. In place of the line `lr` the probe prints `schedule  constant:
0.001, the recipe's peak rate, at every step`. The default, `--schedule recipe`, runs the
recipe as shipped;
the [measured grid](../advanced_guides/measured_combinations.md) probes with `constant`.

**A probe is one run, and it does not repeat.** Run the command again and the hash, the
parameter count and the first column of losses are the same (the first losses to about
six digits); the last column, the seed fractions and the grasp counts are not. Training
on a GPU is not bit-reproducible here, and after the first optimizer step two runs are
two trajectories. Two probes of the shipped GSNet config on one machine: total loss 17.43
at the first iteration both times, 1.89 and 1.72 at the last; 233 and 279 grasps on the
two frames, then 164 and 188; the smallest fraction of real seeds 0.65, then 0.68. The
probe shown above predicted 1024 grasps on both frames in the run just before it. So
read the numbers as "of this run", and a verdict on the edge the same way: see *The
verdict*.

| Line | What it tells you |
|---|---|
| `losses` | Every loss term at the first and the last iteration, then its **fall**: from its first *active* value (reported and not exactly 0) to the mean of the last quarter of its active iterations, and its flags: `flat` (moved less than 5 %), `rising` (the last half of the active iterations more than 10 % above the first half), `zero at end` (0 or missing at the last iteration), `constant` (the very same value every time). A grasp term that is 0 until a seed is supervised starts where it becomes active, and the line says in how many iterations it was. Falling on eight frames means the stages and the labels fit together, nothing more; `--json` has every iteration's value. |
| `seeds`, real | The fraction of seed slots that held a real seed. Below 1 the proposal let fewer points through than it has seeds; at 0 an iteration trained the proposal only. |
| `seeds`, supervised | The fraction of seeds that took part in the grasp losses: 1 for GSNet by construction, about a tenth for EconomicGrasp with an untrained proposal (a seed counts only within 5 mm of a label point), about half for GraspNet-baseline (seeds on an object). |
| `predicted` | Grasps per frame. A model trained for twenty iterations often predicts few, and the count changes from run to run. When no frame has a grasp the verdict is `runs-empty`: every stage ran, and the detector dropped the grasps of seeds that are not valid (no point passed the proposal, or the view stage classified the seed as background) or below its score threshold, at the very end. |
| `scores` | The smallest, the median and the largest score of the grasps of each predicted frame, in the detector's own score unit. |
| `gradient` | The *units* of the model that never received a gradient (an L2 norm above 1e-8) before an optimizer step: sub-modules at depth 2, such as `backbone.encoder` or `head.score_net`, and parameters a top-level part holds itself. A top-level part none of whose units was reached is named alone and makes the run `runs-unstable` (a dead stage); a unit inside a part that learned is reported, not judged. |
| `changed` | The top-level parts whose trainable weights did not change between before the first optimizer step and after the last one (a relative change of at most 1e-6). A frozen encoder is expected here; a stage that should learn is not. |
| `lr` | The learning rate of the first optimizer at the first and at the last step, and their ratio. A schedule that steps per epoch decays with the epochs of the probe, which follow from the batch size and the eight frames: under GSNet's recipe, 0.95 per epoch at 2 iterations per epoch, the last of 20 steps runs at 0.95⁹ = 0.63 of the first rate. With `--schedule constant` the line is `schedule` and names the one rate every step used. |
| `trained`, `memory` | Seconds per iteration and peak GPU memory of this process, on this machine at this moment. |

### The verdict

A probe ends with exactly one of six verdicts.

| Verdict | Meaning | Exit status |
|---|---|---|
| `runs` | On real frames: the detector was built, every requested training iteration ran with finite losses and at least one real seed, every top-level part of the model that holds trainable parameters received a gradient in at least one iteration, the weights stayed finite, and prediction returned one valid grasp group for every probed frame, at least one of them with a grasp. | 0 |
| `runs-empty` | As `runs`, except that no probed frame has a grasp: prediction returned one valid grasp group for every probed frame and every group is empty. Every stage ran; a model trained for a few iterations often keeps no seed. A run that is also unstable is `runs-unstable`. | 0 |
| `runs-unstable` | On real frames: the detector was built, trained and predicted, but a loss or a weight became non-finite (a non-finite loss ends the training there), or an iteration had no real seed (no scene point passed the proposal, so only the proposal was trained), or no iteration supervised a seed, or a top-level part with trainable parameters received no gradient in any iteration, or fewer iterations ran or fewer frames were predicted than were asked for. | 1 |
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

`runs-empty` is common for a method that keeps only grasps above a score threshold: after
twenty iterations no seed may pass it yet (GSNet with the PointNeXt encoder predicts no
grasp on either frame of its probe). It ran every stage, so it works in the sense of this
page too; train longer before you read anything into it. The lines
`losses`, `gradient` and `changed` are what tells a stage that learns from one that is
only wired in: a flat term, a unit without a gradient or a part that did not change is
reported in every verdict and never hidden by a falling total.

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
were found by probing and are refused now, by two rules shown in *Compose* above: a
model on a data recipe that does not serve its labels (which compose resolves for the
label set of the head), and an encoder of another width under a grouper that fixes one
(which compose resolves with the grouper's projected variant). If a probe shows you a
third, please report it.

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
image-encoder preset, every post-processing preset under all six methods, every recipe
preset under every method it applies to, every loss preset in one term of every kind it
accepts, and a few requests that replace several slots at once. Grid probes run with
`--schedule constant`, so that the trends of different methods compare. The
[explorer](../explorer/index.md) shows the same records per selection. Every cell is a recorded
outcome; its two commands are in the measured file beside the page. Typing them repeats
the *experiment*: the same config (hash for hash), the same parameter count, the same
first losses. It does not repeat the later numbers: a cell is one run. In the tables

- `✓` is `runs`; `✓ own` is the method's own component (the cell composes to the shipped
  config, hash for hash); `✓ in Adapter` ran after compose inserted that adapter; `†`
  means compose also set `out_channels` or `in_channels` to fit the method;
- `∅` is `runs-empty` (it ran and predicted no grasp on either frame), `≈ (n)` is
  `runs-unstable`, `✗ (n)` is `refused` (the config check's verdict: nothing
  was built) and `fails (n)` passed the config check and broke; `n` points to the reason
  below the table;
- below the mark of a probe: the grasps it predicted per frame and, for a record of
  judge 2, the range of the fall of its loss terms and the levels it did not reach; a
  probe that predicted no grasp on any frame is greyed;
- the last column is the **class** of the component, level by level. A cell is decided
  at four levels from its record: **composes** (the config check accepts it), **runs**
  (the probe ran clean: `runs` or `runs-empty`; `runs-unstable` counts as not running only
  when a repeat of the probe was unstable too, and stays *unstable, not repeated* until
  then), **learns** (every loss term active in at least half of the iterations fell by at
  least 5 %, none flagged flat, rising or constant, every unit reached by a gradient,
  every part changed) and **predicts** (grasps on every predicted frame, scores not all
  the same). The last two need judge 2: a record of the first judge says *not checked*.
  The class counts the methods under which the option reached a level: `universal`
  (all five seed-based methods), `most` (three or four), `some` (two), `specific`
  (exactly one), `none`; an option probed under fewer methods, or with undecided cells,
  reads *N of M probed* with the rest named after it (*+1 not checked*). A refusal by
  policy (remote code) waits for your consent and is never counted as `none`;
- the columns *Fall per loss term* and *Learns, predicts* of the numbers below a table say
  what judge 2 saw in a run and what kept it from a level; a record made before that
  judge says `not checked`. The sections *Judges* and *Measured classes* of the page
  define every field and every threshold, from `grasppanda.combine.levels`.

The tables *Measured classes* near the top of the page list every component under its
class, one table per level: that is the short answer to "what works where". Every class
is "one run, seed 0, 20 iterations, 8 frames of one scene": it counts verdicts, not
accuracy.

A class is a statement about that grid: those presets, the methods' own recipes, one
stage replaced at a time. `specific` for a head bundle means that it cannot be swapped
in *alone*, because it needs its assigner and its labels; the table *Sets of stages*
shows combinations in which it runs (the sets are too many to probe each: the page says
how many were, and how they were picked), and *Several slots at once* a few requests
that also replace the encoder, a loss or the recipe. A set that reads dense label grids is refused
under EconomicGrasp and FineGrasp, whose recipes serve compact labels, and the reverse;
the page does not hold those cells, the config check says it when you compose one. And a
class says nothing about accuracy: `universal` at *runs* means "builds, trains and
predicts under every method", not "is a good idea under every method".

Whether an option that runs also *learns like the stage it replaces* is the next level:
[Learns like the reference?](../advanced_guides/learns_like_reference.md) trains the cells
that run for 250 iterations and compares every loss term with the method as shipped
under the same settings (`on a par`, `worse`, `diverging`). One run per cell, eight
frames of one scene: a statement about training dynamics, still not about accuracy.

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
