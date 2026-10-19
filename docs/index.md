---
template: home.html
title: GraspPanda
hide: [navigation, toc]
---

## What do you want to do?

<div class="grid cards" markdown>

-   :material-play-circle-outline: **Run a method**

    A published detector from its released weights: first grasps in minutes.

    [Run a method →](get_started/run_a_method.md)

-   :material-swap-horizontal: **Swap a component**

    One line swaps an encoder, a grouper or a loss; compose checks it first.

    [Swap a component →](get_started/swap.md)

-   :material-chart-box-outline: **Train and evaluate**

    One evaluator per dataset, collision filter on and off, every number with its record.

    [Train and test →](user_guides/train_test.md)

-   :material-camera-outline: **Your camera**

    A depth image and its intrinsics; HGGD also reads the colour image.

    [Your camera and Python →](user_guides/inference.md)

-   :material-puzzle-outline: **Add a component**

    Register your own encoder, head or detector without forking.

    [Add a component →](advanced_guides/add_component.md)

-   :material-shield-check-outline: **See the evidence**

    Where each option runs and learns, under which method, and why not elsewhere.

    [How we measure →](evidence/index.md)

</div>

## Methods and datasets

<div class="gp-strip" markdown>
<span class="gp-strip__label">Seed-based skeleton</span>
[GraspNet-baseline <small>CVPR 2020</small>](methods/graspnet_baseline.md){ .gp-chip }
[GSNet <small>ICCV 2021</small>](methods/gsnet.md){ .gp-chip }
[Scale-Balanced-Grasp <small>CoRL 2022</small>](methods/scale_balanced_grasp.md){ .gp-chip }
[EconomicGrasp <small>ECCV 2024</small>](methods/economicgrasp.md){ .gp-chip }
[FineGrasp <small>arXiv 2025</small>](methods/finegrasp.md){ .gp-chip }
</div>

<div class="gp-strip" markdown>
<span class="gp-strip__label">Own family</span>
[HGGD <small>RA-L 2023</small>](methods/hggd.md){ .gp-chip }
</div>

<div class="gp-strip" markdown>
<span class="gp-strip__label">Datasets</span>
[GraspNet-1Billion](user_guides/datasets.md){ .gp-chip }
[GraspClutter6D](user_guides/datasets.md#graspclutter6d){ .gp-chip }
</div>

## How a swap goes

<div class="gp-steps" markdown>

<div markdown>
**Compose**

`grasppanda compose` writes a checked config, or says why the swap cannot work.
</div>

<div markdown>
**Probe**

`grasppanda probe` trains it briefly on real frames and gives one verdict.
</div>

<div markdown>
**Train**

`grasppanda train` and `test` give its accuracy, with the collision filter on and off.
</div>

</div>
