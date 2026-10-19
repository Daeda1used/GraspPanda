# Swap a component

Replace one stage of a method with one line, check it, try it on real frames, then train it.
The encoder is the usual first swap.

![A swap composed, a wrong swap refused with the fix, and a probe that runs (FineGrasp)](../assets/showcase-light.svg#only-light)
![A swap composed, a wrong swap refused with the fix, and a probe that runs (FineGrasp)](../assets/showcase-dark.svg#only-dark)

## Pick the encoder slot of your method

<div class="grid cards" markdown>

-   :material-cube-scan: **Scene encoder**

    ---

    `--set backbone=` under GraspNet-baseline, GSNet, Scale-Balanced-Grasp, EconomicGrasp
    and FineGrasp. For example `ptv3`.

    [All scene encoders →](../slots/scene_encoders.md)

-   :material-image-outline: **Image encoder**

    ---

    `--set image_encoder=` under HGGD. For example `dinov2-small-reg-depth-stem`.

    [All image encoders →](../slots/image_encoders.md)

</div>

Every other slot works the same way: [all slots](../slots/index.md). Whether an option works
under your method is in the [explorer](../explorer/index.md) before you run anything.

## What you need

| To | You need | You get |
|---|---|---|
| see what works where | nothing: the [explorer](../explorer/index.md) | the measured record of each swap |
| compose a swap | the install | a checked config, or a refusal with its reason |
| `probe --synthetic` | the install | `wired`: the stages connect on a synthetic scene |
| probe on real frames | scene 0 of [GraspNet-1Billion](../user_guides/datasets.md) and the method's assets for it | one verdict: `runs`, `runs-empty`, `runs-unstable`, `fails` or `refused` |
| train | the whole training split, its assets and a GPU | a model with weights, and its AP from `grasppanda test` |

To probe the examples below, PTv3 needs a spconv extra and the DINOv2 encoder the `timm` extra
([Install](installation.md)).

## 1. Compose

=== "GraspNet-baseline"

    ```bash
    grasppanda compose \
        graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
        --set backbone=ptv3 -o baseline_ptv3.yaml
    ```

=== "GSNet"

    ```bash
    grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
        --set backbone=ptv3 -o gsnet_ptv3.yaml
    ```

=== "Scale-Balanced-Grasp"

    ```bash
    grasppanda compose \
        scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
        --set backbone=ptv3 -o sbg_ptv3.yaml
    ```

=== "EconomicGrasp"

    ```bash
    grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e \
        --set backbone=ptv3 -o economicgrasp_ptv3.yaml
    ```

=== "FineGrasp"

    ```bash
    grasppanda compose finegrasp/finegrasp_sunet14_g1b-realsense_10e \
        --set backbone=ptv3 -o finegrasp_ptv3.yaml
    ```

=== "HGGD"

    ```bash
    grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e \
        --set image_encoder=dinov2-small-reg-depth-stem -o hggd_dinov2.yaml
    ```

`compose` checks every pair of stages and writes the config. Where the next stage needs an
adapter, it inserts one and says so. Most swaps that cannot work are refused with the reason
and the fix, before anything is built; the probe of step 2 catches the rest. `--set` repeats
for [several slots at once](../advanced_guides/measured_combinations.md#several-slots-at-once).

??? note "Why was my swap refused?"

    Each refusal names its kind: incompatible by design, not supported yet, needs a setting
    compose adds, needs a preset's settings, or needs your consent. The rules behind them,
    and the lifts compose applies: [Swap rules](../advanced_guides/swapping.md).

## 2. Probe

A probe trains the swapped model for a few iterations on real frames, predicts, and ends with
one verdict. Point GraspPanda to your
[GraspNet-1Billion download](../user_guides/datasets.md) and build the method's assets for the
`sample` split, a few frames of scene 0:

=== "GraspNet-baseline"

    ```bash
    export GRASPPANDA_GRASPNET1B_ROOT=/path/to/graspnet1b
    grasppanda prepare graspnet1b --labels --tolerance \
        --split sample --config baseline_ptv3.yaml
    grasppanda probe baseline_ptv3.yaml
    ```

=== "GSNet"

    ```bash
    export GRASPPANDA_GRASPNET1B_ROOT=/path/to/graspnet1b
    grasppanda prepare graspnet1b --labels --graspness \
        --split sample --config gsnet_ptv3.yaml
    grasppanda probe gsnet_ptv3.yaml
    ```

=== "Scale-Balanced-Grasp"

    Scale-Balanced-Grasp also reads a grasp-width histogram of the whole training split,
    which a few frames cannot fill. A probe reads the file of the
    [reference repository](https://github.com/mahaoxiang822/Scale-Balanced-Grasp/tree/995d6f8)
    instead (`statistic/scale_distribution.npy`, written `<file>` below), in two steps
    ([What a probe needs](../user_guides/combinations.md#what-a-probe-needs)).
    First add the two `--opts` that name the file by its SHA-256, and import it once:

    ```bash
    export GRASPPANDA_GRASPNET1B_ROOT=/path/to/graspnet1b
    sha256sum <file>   # prints the <sha256> of the next command
    grasppanda compose sbg_ptv3.yaml -o sbg_ptv3_probe.yaml --opts \
        data.train.dataset.scale_statistic.source=legacy_import \
        data.train.dataset.scale_statistic.reference_sha256=<sha256>
    grasppanda prepare graspnet1b --scale-statistic --import-legacy \
        --scale-statistic-file <file> --config sbg_ptv3_probe.yaml
    ```

    Then build the other assets and probe the config that carries the two `--opts`:

    ```bash
    grasppanda prepare graspnet1b --labels --tolerance \
        --split sample --config sbg_ptv3_probe.yaml
    grasppanda probe sbg_ptv3_probe.yaml
    ```

=== "EconomicGrasp"

    ```bash
    export GRASPPANDA_GRASPNET1B_ROOT=/path/to/graspnet1b
    grasppanda prepare graspnet1b --labels --graspness --compact-labels \
        --split sample --config economicgrasp_ptv3.yaml
    grasppanda probe economicgrasp_ptv3.yaml
    ```

=== "FineGrasp"

    ```bash
    export GRASPPANDA_GRASPNET1B_ROOT=/path/to/graspnet1b
    grasppanda prepare graspnet1b --labels --graspness --compact-labels \
        --split sample --config finegrasp_ptv3.yaml
    grasppanda probe finegrasp_ptv3.yaml
    ```

=== "HGGD"

    ```bash
    export GRASPPANDA_GRASPNET1B_ROOT=/path/to/graspnet1b
    grasppanda prepare graspnet1b --labels --view-grasps \
        --split sample --config hggd_dinov2.yaml
    grasppanda probe hggd_dinov2.yaml
    ```

`grasppanda prepare graspnet1b --check --split sample --config <your config>` lists what is
still missing. Without the dataset, the probe runs its synthetic check and reports `wired`.

--8<-- "docs/includes/caveat_probe.md"

## 3. Train

A swapped model has no weights until you train it. Training needs the whole training split
with the same assets, and a GPU:

```bash
# list what is still missing
grasppanda prepare graspnet1b --check --config <your config>
grasppanda train <your config>
grasppanda test <your config> --weights <run>/checkpoints/best.pth
```

`test` scores it with the collision filter on and off. [Train and test](../user_guides/train_test.md)
has the details, and
[Does it train like the shipped component?](../advanced_guides/learns_like_reference.md)
shows how swaps that train trained beside the component the method ships.
