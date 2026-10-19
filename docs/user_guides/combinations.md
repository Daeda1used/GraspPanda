# Compose and probe

**Goal:** put another option into a slot of a method, check in about a minute that the
combination trains on real frames, then train it.

```bash
grasppanda compose <method config> --list <slot>
grasppanda compose <method config> --set <slot>=<option> -o my.yaml
grasppanda probe my.yaml
grasppanda train my.yaml
```

--8<-- "docs/includes/caveat_probe.md"

**You need:** GraspPanda installed. Steps 1 and 2 need no dataset. Step 3 reads the split
`sample` of the config's dataset with the assets of the method
([What a probe needs](#what-a-probe-needs)); without them it falls back to a wiring check
and says so.
The [explorer](../explorer/index.md) does steps 1 and 2 in a browser and shows what was
measured for every selection.

## 1. See the options of a slot

```bash
grasppanda compose <method config> --list backbone
```

With a method config, compose lists the options of that method's slot, each with what was
measured for it under that method, and the options that do not fit with the reason. Without
a slot it lists every slot of the method; without a method config, every option of the
slot. `grasppanda compose --show backbone=ptv3` prints one option with its notes. Which
slots a method has is on the [Components](../slots/index.md) page: the scene-encoder and
stage slots exist in the five seed-based methods, the image encoder in HGGD,
post-processing and the training recipe in all six.

??? note "Options, presets and components"

    An option is a **preset** or the name of a registered class. A preset is the part of a
    config a newcomer cannot guess: the settings of one component for table-top depth
    clouds of GraspNet-1Billion (grid size, radii, widths, sampling rules) and, where its
    stage reads a label asset, the dataset section that serves it. Its header says what must
    be re-tuned and why: GraspClutter6D objects are farther from the camera (0.70 m against
    0.44 m), so lengths may need re-tuning there. Every scene encoder has one, because the
    constructor defaults of most encoders are those of room-scale benchmarks: a bare
    `{type: PTv3}` works on a 2 cm grid with six input channels.

    A class named by its type (`--set grouper=CylinderGroup`) is built with its constructor
    defaults, which `grasppanda components <type>` lists. Presets live in
    `grasppanda/presets/<slot>/<name>.yaml`. They are config fragments, not configs, and are
    not inherited: compose writes their values into the child config. In a wheel the
    measured records come with the package; in a checkout they are read from
    `docs/advanced_guides/measured_combinations.json`.

## 2. Compose

Pick a method; each example puts one option into one slot:

=== "GraspNet-baseline"

    ```bash
    grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
        --set backbone=ptv3 -o baseline_ptv3.yaml
    ```

    PTv3 returns a feature for every point, and GraspNet-baseline takes the encoder's
    subsample as its seeds: compose wraps PTv3 in the adapter `FurthestPointSubsample`.

=== "GSNet"

    ```bash
    grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
        --set grouper=roi-grid -o gsnet_roi-grid.yaml
    ```

    The PV-RCNN RoI-grid pooling in place of GSNet's cylinder grouper; nothing else changes.

=== "Scale-Balanced-Grasp"

    ```bash
    grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
        --set backbone=ptv3 -o sbg_ptv3.yaml
    ```

    Compose wraps PTv3 in `FurthestPointSubsample` and, because PTv3 returns 64 channels
    where the grouper needs 256, puts the projected variant of the grouper in place.

=== "EconomicGrasp"

    ```bash
    grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
        --set grouper=multi-scale -o economicgrasp_multi-scale.yaml
    ```

    FineGrasp's grouper (crops at four radii, fused by attention) under EconomicGrasp.

=== "FineGrasp"

    ```bash
    grasppanda compose finegrasp/finegrasp_sunet14_g1b-realsense_10e \
        --set head=swad -o finegrasp_swad.yaml
    ```

    GSNet's head reads dense labels and FineGrasp trains on compact ones: compose rewrites
    the training data and the assigner to serve the dense labels.

=== "HGGD"

    ```bash
    grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e \
        --set image_encoder=resnet34 -o hggd_resnet34.yaml
    ```

    HGGD has its own stage slot, the image encoder; here a timm ResNet-34 with ImageNet
    weights.

Each example was probed and recorded as `runs` under its method; a unit test holds these
examples to the records. A refused combination writes nothing, and every refusal says in
brackets why: incompatible by design (`structural`), not supported yet (`not-implemented`),
needs a setting compose adds (`fixed-by-compose`), needs a preset's settings
(`not-a-combination`) or needs your consent (`policy`). The rules are on
[Swap rules](../advanced_guides/swapping.md).

??? note "What compose prints (the Scale-Balanced-Grasp example)"

    ```text
    grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e --set backbone=ptv3

    slot      component                       from
    ────────  ──────────────────────────────  ────────────────────────────────────────────────────
    backbone  PTv3 in FurthestPointSubsample  preset `ptv3`, in the adapter FurthestPointSubsample
    proposal  SubsampledSeeds                 base config
    view      ObjectnessViewStage             base config
    assigner  ScaleBalancedAssigner           base config
    grouper   DepthCylinderProjectionGroup    base config, projected
    head      OperationHead                   base config

    what compose did
      • backbone: preset `ptv3` (PTv3). Point Transformer V3 (encoder + decoder, Pointcept PT-v3m1) on a
        5 mm grid.
      • wrapped PTv3 in the adapter FurthestPointSubsample (preset `furthest-point-subsample`):
        SubsampledSeeds requires 'subsampled' scene features, but PTv3 returns 'dense' ones (one feature
        per scene point). An adapter changes the model; its notes below say how.
      • FurthestPointSubsample.npoint: 1024, the number of seeds of the proposal
        (`model.proposal.num_seeds`): the points the adapter keeps are the seeds.
      • FurthestPointSubsample: fps_skip_near_origin: true, fps_tie_break: pointnet2_cuda, the sampling
        rules of the encoder it replaces (PointNet2Transformer), so that the adapter keeps the very
        points that encoder ends on.
      • grouper: DepthCylinderProjectionGroup in place of DepthCylinderGroup. DepthCylinderGroup as
        configured adds the seed features to its 256-channel crop features, but FurthestPointSubsample
        returns 64-channel features. DepthCylinderProjectionGroup is the same crop with a linear layer
        in front of its seed residual; it changes the model (the base method has none). `--no-adapter`
        leaves the pair refused.

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
      ✓ valid  (hash 7f9112c56063)

    ✓ written sbg_ptv3.yaml
      next: grasppanda probe sbg_ptv3.yaml
    ```

    The file is a child config of the method config. Its header repeats the command, what
    compose did and the re-tuning notes of every preset; its body is what you would have
    written by hand:

    ```yaml
    _base_:
      - /…/configs/scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e.yaml

    model:
      backbone:
        type: FurthestPointSubsample  # `dense` -> `subsampled`, written down
        npoint: 1024                  # = model.proposal.num_seeds
        fps_skip_near_origin: true    # the rule of the encoder it replaces: same seeds
        fps_tie_break: pointnet2_cuda   # the rule of the encoder it replaces: same seeds
        backbone:
          type: PTv3                  # `dense`: one feature per input point
          in_channels: 3              # constant ones; geometry enters through the grid
          grid_size: 0.005
      grouper:
        type: DepthCylinderProjectionGroup   # the same crop with a linear layer in front of its seed residual
        radius: 0.08
        ...
    ```

    Outside the shipped config tree the base is named by its absolute path: the file can be
    moved on this machine, and on another one the command in its header composes it again.

??? note "What compose does for you, each time saying so"

    No rule about what fits what lives in compose: every verdict is the contract rules of
    the config checker, quoted. Where exactly one registered piece bridges a refused pair,
    compose puts it in and says so; `--no-adapter` leaves the pair refused.

    - **An encoder adapter**: `FeaturePropagation` around a sub-sampling encoder under a
      proposal that scores every point, `FurthestPointSubsample` around a dense encoder under
      a proposal that takes the encoder's subsample as its seeds. The adapter gets the seed
      count of the proposal and the sampling rules of the encoder it replaces.
    - **A fitted width**: where the new encoder has an `out_channels` option it is set to the
      seed width the grouper fixes, otherwise to the width of the encoder it replaces. Where
      the encoder has no such option, the projected variant of the grouper
      (`DepthCylinderProjectionGroup`) takes its place. Where the method feeds real point
      features (FineGrasp's normals), `in_channels` is set to their number.
    - **An axis adapter** between a grouper and a head bundle whose feature axes differ:
      `DepthEmbedGroup` (preset `depth-embed`) or `DepthPoolGroup` (preset `depth-pool`).
    - **The options a stage declares for the other stages**: assigner options that return a
      field a stage reads, graspness only for a scoring proposal next to
      `ObjectnessViewStage`, the decode option of a head that would multiply in a score no
      stage predicts.
    - **The label set of the head bundle**: a head of dense label grids on a recipe of
      compact labels, or the reverse, gets the dataset section and the assigner of a shipped
      recipe that serves them (`--set labels=dense|compact` asks for it explicitly). Compact
      labels are an asset to build (`grasppanda prepare <dataset> --compact-labels --config
      <the composed config>`).
    - **The dataset section a stage reads**, when its preset carries one (`tolerance`,
      `scale_statistic`, `graspness`). The asset still has to be built; the preset says how.
    - **Replace, never merge**: a `--set` of the base's own component type gets
      `_delete_: true`. `--unset key.path` writes a component default back. `-o` refuses an
      existing file without `--force` and never writes onto the base config.

    A composed file can be the base of a new command: compose reads the request in its header
    and composes again from its base with both requests merged (a config chain is two levels
    deep). Further overrides go through `--opts`; a path below a slot you set addresses that
    component wherever an adapter put it:

    ```bash
    grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
        --set backbone=ptv3 --opts model.backbone.grid_size=0.01 -o sbg_ptv3_1cm.yaml
    ```

??? note "Two refusals, and what they say"

    **Stages that do not fit each other.** A head bundle supervised with compact labels, fed
    by an assigner of dense ones that you set yourself:

    ```text
    $ grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
          --set assigner=NearestLabelAssigner
    ...
      ✗ assigner · head [structural]: NearestLabelAssigner returns 'dense' matched labels, but
        EconomicHead is supervised with 'compact' ones. A head bundle and the assigner that feeds it
        must agree on the label set; the dataset must serve that label set too.
    ...
    ✗ refused: nothing was written. The messages above name the conflict and a fix.
      → EconomicHead reads compact labels and NearestLabelAssigner returns dense ones; the request set
        the assigner (`--set assigner=...`). Without it compose gives EconomicHead an assigner and
        training data of compact labels; or keep NearestLabelAssigner with a head bundle that reads
        dense labels (DecoupledGridHead, OperationHead, SWADHead).
    ```

    Leave the assigner to compose and swap the head instead: compose then rewrites the
    recipe, as in the FineGrasp example above.

    **What compose would insert, refused without it.** With `--no-adapter`, or in a config
    written by hand:

    ```text
    $ grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
          --set grouper=ball --no-adapter
    ...
      ✗ grouper · head [fixed-by-compose]: LocalGroup emits features with axes ('seed', 'channel'), but
        OperationHead expects ('seed', 'depth', 'channel'). An axis is never added or squeezed
        implicitly: wrap the grouper in the axis adapter DepthEmbedGroup (preset `depth-embed`), `{type:
        DepthEmbedGroup, num_depth: <the head's num_depth>, grouper: {...}}`, which `grasppanda compose`
        inserts, or use a grouper and a head bundle that agree.
    ```

    A slot the method does not have is refused with the slots it has:

    ```text
    $ grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e --set backbone=ptv3
    ✗ HGGD has no slot 'backbone'. Its slots: image_encoder. HGGD is a detector family of its own: the
      stages of the seed-based methods (backbone, proposal, view, grouper, head) do not exist in it.
    ```

## 3. Probe it

```bash
grasppanda probe my.yaml
```

A probe is a miniature of `grasppanda train`: the real trainer with the config's
optimizer, schedule and hooks, 20 iterations on the eight frames of the split `sample` of
the config's dataset (GraspNet-1Billion: scene 0, frames 0, 32, …, 224), then prediction
on two of them. Nothing is written. The line `probe` of the report lists the overrides that
turn the config into that miniature.

??? note "What a probe prints (GraspNet-baseline with PTv3, the file of step 2, one run)"

    ```text
    $ grasppanda probe baseline_ptv3.yaml
      config      baseline_ptv3.yaml  (hash 147c5d6fd28d)
      model       GraspNetBaseline · 46,418,380 parameters
      data        8 frames of the config's dataset · 2 per iteration
      probe       as `grasppanda train` with data.train.dataset.split=sample data.train.num_workers=0 evaluation.interval=0 evaluation.save_best=null train.epochs=5 data.val.dataset.split=sample data.val.dataset.frame_step=4 data.val.num_workers=0
      device      cuda
      trained     20 of 20 iterations · 0.22 s per iteration (median, without data; data 0.20 s), measured on this machine now
      memory      peak 4.1 GiB on the GPU, this process
      losses      first → last · fall from the first active value to the mean of the last quarter
        loss/objectness             0.7379 → 0.3549       fell 43 %
        loss/view                   0.2098 → 0.05278      fell 74 %
        loss/score                  0.07834 → 0.02856     fell 59 %
        loss/angle                  0.525 → 0.4638        fell 17 %
        loss/width                  0.491 → 0.1616        fell 65 %
        loss/tolerance              1.145 → 0.3681        fell 65 %
        loss                        3.187 → 1.43          fell 52 %
      seeds       real: min 1.00, mean 1.00 of the seed slots per iteration
                  supervised: 0.51 → 0.50, min 0.47
      predicted   2 frame(s) · 492, 494 grasps
      scores      min / median / max per frame: -17.1 / 1.9 / 49.9; -17 / 2.19 / 32
      gradient    reached every one of 9 units of 4 part(s)
      changed     every part
      lr          0.001 at the first step → 0.001 at the last, ×1 at the last step

    ✓ runs: trained 20 iterations on 8 frames of the dataset and predicted on 2 frame(s), with finite losses and real seeds in every iteration.
      (28 s)
    ```

??? note "What each line of the report says"

    | Line | What it tells you |
    |---|---|
    | `losses` | Every loss term at the first and the last iteration, then its **fall**: from its first *active* value (reported and not exactly 0) to the mean of the last quarter of its active iterations, and its flags: `flat` (moved less than 5 %), `rising` (the last half of the active iterations more than 10 % above the first half), `zero at end` (0 or missing at the last iteration), `constant` (the very same value every time). Falling on eight frames means the stages and the labels fit together, nothing more; `--json` has every iteration's value. |
    | `seeds`, real | The fraction of seed slots that held a real seed. Below 1 the proposal let fewer points through than it has seeds; at 0 an iteration trained the proposal only. |
    | `seeds`, supervised | The fraction of seeds that took part in the grasp losses: about half for GraspNet-baseline (seeds on an object), 1 for GSNet by construction, about a tenth for EconomicGrasp with an untrained proposal (a seed counts only within 5 mm of a label point). |
    | `predicted` | Grasps per frame. A model trained for twenty iterations often predicts few, and the count changes from run to run. |
    | `scores` | The smallest, the median and the largest score of the grasps of each predicted frame, in the detector's own score unit. |
    | `gradient` | The *units* of the model that never received a gradient before an optimizer step: sub-modules at depth 2 and parameters a top-level part holds itself. A top-level part none of whose units was reached is a dead stage and makes the run `runs-unstable`. |
    | `changed` | The top-level parts whose trainable weights did not change between the first and the last optimizer step. A frozen encoder is expected here; a stage that should learn is not. |
    | `lr` | The learning rate of the first optimizer at the first and at the last step, and their ratio. With `--schedule constant` the line is `schedule` and names the one rate every step used. |

??? note "The schedule of a probe, and why a probe does not repeat"

    To fit 20 iterations the probe sets `train.epochs` to as few epochs as the eight frames
    need, and the recipe's learning-rate schedule runs over those epochs: GraspNet-baseline's
    first step (epoch 8) is never reached, so its rate stays (the line `lr` above); GSNet's
    decay of 0.95 per epoch ends at 0.63 of its rate, EconomicGrasp's cosine at 2.4 %, and
    FineGrasp's warm-up of 800 iterations keeps the whole probe below 2.5 % of its rate.
    `--schedule constant` removes the scheduler and trains every step at the recipe's
    **peak** rate, scaled by the ratio of batch sizes when the probe's batches hold fewer
    frames than the recipe's (FineGrasp's 0.008 for batches of 32 becomes 0.002), and holds a
    per-epoch batch-norm momentum decay (GraspNet-baseline, Scale-Balanced-Grasp, HGGD) at
    its first value. The config hash does not change. The default, `--schedule recipe`, runs
    the recipe as shipped; the probes behind
    [Does it run?](../advanced_guides/measured_combinations.md) use `constant`, so that the
    trends of different methods compare.

    A probe is one run. Run it again and the hash, the parameter count and the first losses
    are the same; the last losses, the seed fractions and the grasp counts are not. Training
    on a GPU is not bit-reproducible here, and after the first optimizer step two runs are
    two trajectories. Read the numbers as "of this run".

??? note "What a probe needs"

    The dataset and the label assets of its split `sample`. They are small; for
    GraspNet-1Billion, RealSense:
    {: #what-a-probe-needs }

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

    **The width histogram of Scale-Balanced-Grasp** is the exception. It is a statistic of a
    whole training split, and eight frames leave bins of it empty, which its generator
    refuses. So a config that reads it (`scale_statistic` in its training dataset: every
    config composed from Scale-Balanced-Grasp, or with `--set assigner=scale-balanced`) is
    probed with the reference's own file. Import it once, through a config that names the
    file by its SHA-256:

    ```bash
    sha256sum <reference repository>/statistic/scale_distribution.npy    # the <sha256> below
    grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
        --opts data.train.dataset.scale_statistic.source=legacy_import \
               data.train.dataset.scale_statistic.reference_sha256=<sha256> -o sbg_reference_histogram.yaml
    grasppanda prepare graspnet1b --scale-statistic --import-legacy \
        --scale-statistic-file <reference repository>/statistic/scale_distribution.npy \
        --config sbg_reference_histogram.yaml
    ```

    From then on, compose with those two `--opts`, or hand them to `grasppanda probe <config>
    --opts ...`. Training on the full split does not need the import:
    `grasppanda prepare graspnet1b --scale-statistic --config <config>` builds the native
    histogram of that split.

    **Without the dataset, or without an asset,** the probe says what is missing and how to
    build it, falls back to two iterations on a synthetic scene with random labels and
    reports `wired`, never `runs`. `--synthetic` asks for that check directly. `--iters`,
    `--frames`, `--predict-frames`, `--split`, `--device` and `--workers` change what is run;
    `--json` prints the whole report; `grasppanda probe --help` repeats the verdicts and exit
    statuses.

## 4. Read the verdict

A probe ends with exactly one of six verdicts. What the levels *composes*, *runs*, *learns*
and *predicts* add to them is on [How we measure](../evidence/index.md).

| Verdict | Meaning | Exit status |
|---|---|---|
| `runs` | On real frames: the detector was built, every requested training iteration ran with finite losses and at least one real seed, every top-level part of the model that holds trainable parameters received a gradient in at least one iteration, the weights stayed finite, and prediction returned one valid grasp group for every probed frame, at least one of them with a grasp. | 0 |
| `runs-empty` | As `runs`, except that no probed frame has a grasp: prediction returned one valid grasp group for every probed frame and every group is empty. Every stage ran; a model trained for a few iterations often keeps no seed. A run that is also unstable is `runs-unstable`. | 0 |
| `runs-unstable` | On real frames: the detector was built, trained and predicted, but a loss or a weight became non-finite (a non-finite loss ends the training there), or an iteration had no real seed (no scene point passed the proposal, so only the proposal was trained), or no iteration supervised a seed, or a top-level part with trainable parameters received no gradient in any iteration, or fewer iterations ran or fewer frames were predicted than were asked for. | 1 |
| `fails` | An exception was raised after the config check had passed. The report names the phase (build, data, optimizer, loss, backward, step, hook, predict), the component it came from and the message. | 1 |
| `refused` | The config check refused the config (the same check as `grasppanda print-config --check --for train`). Nothing was built; the report carries the messages. | 2 |
| `wired` | No real frame was read. On a synthetic table-top scene with random labels the detector was built, back-propagated a loss through every stage and predicted. This shows that the stages connect; it says nothing about real data. | 0 with `--synthetic`, else 3 |

`runs` and `runs-empty` mean the combination works in this sense: every stage ran. The
lines `losses`, `gradient` and `changed` tell a stage that learns from one that is only
wired in, in every verdict.

??? note "Reading `runs-unstable`, `runs-empty` and `fails`"

    `runs-unstable` is what an untuned recipe looks like. A deep encoder under a learning
    rate chosen for a shallow one can lose every seed for a few iterations and recover; the
    verdict says which iterations. Because a probe is one run, a combination near that edge
    can be `runs` in one probe and `runs-unstable` in the next. Treat the two as one answer,
    "it works", and watch `metric/real_seeds` when you train.

    `runs-empty` is common for a method that keeps only grasps above a score threshold:
    after twenty iterations no seed may pass it yet. Train longer before you read anything
    into it.

    `fails` is a combination that passed the config check and then broke, and the probe says
    where. The check compares what components and datasets declare; it cannot know whether
    options suit your data. An encoder named by its type is built with its own defaults:

    ```text
    $ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=OACNNs -o gsnet_oacnns_bare.yaml
    ...
      • backbone: OACNNs with its constructor defaults; no preset was named. ...
    $ grasppanda probe gsnet_oacnns_bare.yaml
    ...
    ✗ fails: at loss (iteration 1), in the backbone OACNNs: ContractError: OACNNs: cloud 0 of the batch has no voxel left after 4 of 4 down-sampling steps. ...
    ```

    The default grid of OA-CNNs is 2 cm, for rooms; its preset (`--set backbone=oacnns`)
    sets 5 mm. A combination that fails for a reason the contracts could have known is a bug
    of the contract rules: please report it. A probe that ends with `CUDA out of memory` is
    `fails` too, with a note: on a shared machine that says something about the device at
    that moment, not about the combination.

## 5. Train it

```bash
grasppanda train my.yaml
```

The composed config is an ordinary config: [Train and test](train_test.md) applies. It is
not a tuned recipe: optimizer, schedule, batch size and point budget are the base method's.
The header of the composed file lists what to re-tune first. Report a number from such a
run as trained here, next to the method as shipped under the same recipe.

## Where the measured answers are

- [Explorer](../explorer/index.md): every option under every method, with its verdict and
  the commands.
- [Does it run?](../advanced_guides/measured_combinations.md): these steps for every option
  under each method it fits; refusals and options without a record are named with the
  reason.
- [Does it train like the shipped component?](../advanced_guides/learns_like_reference.md):
  the options that run, trained longer beside the component the method ships.
- `configs/experimental/`: composed configs kept in the repository, each with the command in
  its header.

??? note "Adding a preset"

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
    cells each fail a unit test that says what to run (`tests/unit/test_combine_presets.py`,
    `tests/unit/test_combine_grid.py`). How to write the component itself:
    [Add a component](../advanced_guides/add_component.md).

## Next

- [Swap rules](../advanced_guides/swapping.md): why a pair of stages is refused.
- [Add a component](../advanced_guides/add_component.md): bring your own option.
