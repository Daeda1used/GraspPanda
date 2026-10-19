# Scale-Balanced-Grasp

> [Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes](https://openreview.net/pdf?id=tiPHpS4eA4)
> (CoRL 2022). Reference code:
> [mahaoxiang822/Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp)
> at `995d6f8`. Its `LICENSE` file is MIT; the code derives from graspnet-baseline, which is
> under a non-commercial research licence.

GraspNet-baseline with four changes aimed at small-scale grasps: a backbone with
point-transformer layers, crops at four radii fused with a gated seed-feature residual,
an objectness target that asks for graspable views, and loss weights that are larger for
rare grasp widths.

**Run it in five minutes:** [the method page](https://daeda1used.github.io/GraspPanda/methods/scale_balanced_grasp/#run-in-five-minutes): the weights, grasps on the example scene, the same from Python and in the browser, and a first number on the benchmark, each with what it printed when it was measured.

## Configs

| Config | Trains on | Tests on |
|---|---|---|
| [sbg_pointnet2-transformer_g1b-realsense_18e](sbg_pointnet2-transformer_g1b-realsense_18e.yaml) | GraspNet-1Billion, RealSense | GraspNet-1Billion, RealSense |
| [sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e](sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e.yaml) | GraspNet-1Billion, RealSense | GraspClutter6D, RealSense D435 |

The core method is implemented: the reference's `GraspNet_MSCQ` network with its
scale-balanced loss. Noisy-clean mixing, object-balanced sampling and ScaleAP are not
(see *Scope* under Notes). The options of every slot, with what was measured for each:
[the method page](https://daeda1used.github.io/GraspPanda/methods/scale_balanced_grasp/#the-slots),
or in a terminal `grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e --list`
(one slot: `--list grouper`).

## Results on GraspNet-1Billion

AP on seen / similar / novel over the whole test split (90 scenes, 23,040 frames, all
scored), RealSense, protocol `graspnet1b-sv-v1`. The workspace is cropped with the
ground-truth segmentation, as in the reference (`gt_assisted: workspace`).

| Provenance | Collision filter | AP | Source |
|---|---|---|---|
| `reported` | off | 58.95 / 52.97 / 22.63 | paper, Table 2, "Ours" |
| `author-ckpt` | off | 57.78 / 51.66 / 22.12 | [record](records/sbg_pointnet2-transformer_g1b-realsense_18e_author-ckpt_test_raw.json) |
| `reported` | on | 63.83 / 58.46 / 24.63 | paper, Table 2, "Ours + CD" |
| `author-ckpt` | on | 62.29 / 56.80 / 24.00 | [record](records/sbg_pointnet2-transformer_g1b-realsense_18e_author-ckpt_test_cd.json) |

`author-ckpt` is the authors' `checkpoint-rs.tar`, converted and run through
`grasppanda test` at commit `4bc5333`, one run with seed 0. No model has been trained here
yet. **Measured minus reported**: −1.17 / −1.31 / −0.51 AP without the collision filter
and −1.54 / −1.66 / −0.63 AP with it, below the paper by more than 1 AP on seen and
similar. The gap is not explained.

<details markdown="1"><summary>What is known about the gap, and a 20-frame check with the reference code</summary>

The paper's rows are quoted from Table 2 of the paper (arXiv 2212.05275); whether they
use object-balanced sampling is not stated in the table.

- On the 20 frames below, the pristine reference code and this implementation score the
  same when fed the same points, and the stage-by-stage comparison agrees. Twenty frames
  cannot show whether the two agree on a split to a tenth of an AP.
- Object-balanced sampling, which the paper describes as the method's inference step
  (it re-samples the seeds per object instance), is not implemented here. Table 2 does
  not say whether its "Ours" rows use it.
- The collision filter adds 4.51 / 5.14 / 1.88 AP here and 4.88 / 5.49 / 2.00 AP in the
  paper.
- Not investigated: the reference code with this checkpoint on the whole split, with and
  without object-balanced sampling; whether the released checkpoint is the one behind
  Table 2; how much a number moves with the seed of the point sampling.

What was measured with the pristine reference code is a **20-frame subset**, not a split:
scenes 100–104, frames 0 / 64 / 128 / 192, scored by `GraspNet1BEvaluator`.

| Predictions from | Collision filter | AP on the 20 frames |
|---|---|---|
| pristine reference code | off | 56.46 |
| pristine reference code | on (reference filter) | 61.71 |
| GraspPanda, converted weights, same input points | off | 56.46 |

The subset is not a benchmark number and must not be quoted as one.

</details>

## Results on GraspClutter6D

The same checkpoint on the whole GraspClutter6D test split (235 scenes, 3,055 frames of
the RealSense D435), protocol `graspclutter6d-sv-v1`, config
[sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e](sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e.yaml). Percent.

| Collision filter | AP | AP0.8 | AP0.4 | Provenance |
|---|---|---|---|---|
| not stated | 22.69 | 27.57 | 16.95 | reported |
| off | 21.77 | 26.49 | 16.46 | author-ckpt ([record](records/sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e_author-ckpt_test_raw.json)) |
| on | 22.71 | 27.61 | 17.01 | author-ckpt ([record](records/sbg_pointnet2-transformer_g1b-realsense-to-gc6d-d435_18e_author-ckpt_test_cd.json)) |

The filtered run matches the reported row to 0.06 AP.

<details markdown="1"><summary>Against the reported row</summary>

`reported` is Table IX of the GraspClutter6D paper (arXiv 2504.06866 v3), which states
neither the collision filter nor which checkpoint was scored. Measured minus reported
(AP / AP0.8 / AP0.4): −0.92 / −1.08 / −0.49 without the filter, +0.02 / +0.04 / +0.06 with
it. Whether the paper ran with a filter was not investigated and cannot be read from it.
The input is cropped to the image rectangle around the ground-truth segmentation, as the
official tool does (`gt_assisted: workspace`); this is not the crop the weights were
trained with, and on a 24-scene subset the training crop scored 3.1 ± 0.9 AP higher
without the filter (`docs/user_guides/cross_dataset.md`).

</details>

## Weights

GraspPanda publishes no weights for this method. Download the authors' `checkpoint-rs.tar`
(link and SHA-256 in the [metafile](metafile.yaml)) and convert it once:

```bash
python -m tools.convert_checkpoint checkpoint-rs.tar sbg_realsense.pth \
    --source-format scale-balanced-grasp --camera realsense
```

## Train and test

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --tolerance \
    --config scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e
grasppanda prepare graspnet1b --scale-statistic \
    --config scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e
grasppanda test scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
    --weights sbg_realsense.pth
grasppanda train scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e
```

Training follows the reference recipe: Adam, one-cycle learning rate with peak 1e-3 over
18 epochs, batch 2, batch-norm momentum 0.5 halved every 2 epochs. The native width
statistic reads 3200 training frames, about 1.3 s each on the development machine, so the
step takes roughly 70 minutes; it has not been run at that setting here. Both evaluation
variants are always reported: `raw` (the reference's shipped test command turns its
collision filter off) and `cd`.

## Notes

<details markdown="1"><summary>How it is built</summary>

`configs/_base_/models/scale_balanced_grasp.yaml`; what differs from
[GraspNet-baseline](../graspnet_baseline/README.md), on which the reference code builds:

| Slot | Component | Difference |
|---|---|---|
| `backbone` | `PointNet2Transformer` | two set-abstraction levels (2048 and 1024 points), each followed by a vector-attention layer over 16 nearest points; no feature propagation |
| `proposal` | `SubsampledSeeds` | none |
| `view` | `ObjectnessViewStage` | `min_graspable_views: 10`: a positive seed lies on an object *and* has more than ten views with a score target above 0.1; only those seeds supervise the view scores |
| `assigner` | `ScaleBalancedAssigner` | also returns a loss weight per seed |
| `grouper` | `DepthCylinderGroup` | radius 0.08 m × (0.25, 0.5, 0.75, 1), an MLP per radius, fused by a linear layer; `seed_residual: gated` adds `sigmoid(W f) · f` of the seed feature |
| `head` | `OperationHead` | weighted losses; `score_supervision: seeds_with_a_supervised_depth`; `score_decode: score_x_objectness_x_tolerance` |

Which method ships what in each slot:
[Methods](https://daeda1used.github.io/GraspPanda/methods/#what-each-method-ships).

</details>

<details markdown="1"><summary>Scope: what is not implemented</summary>

- **Noisy-clean mixing** (training data). The reference builds a second, clean cloud per
  frame from the object models at their annotated poses plus a synthetic table, cropped
  to the points within 8 mm of the sensor cloud, and feeds it to training next to the
  sensor cloud (how the two are mixed was not studied here). It would need a derived
  asset of clean clouds per frame and a point transform that mixes the two.
- **Object-balanced sampling** (test time). The reference re-samples the 1024 seeds so
  that every object instance gets the same number, using instance masks from a second,
  separately trained segmentation network, and interpolates seed features onto the new
  seeds. Here it would be a proposal that replaces `SubsampledSeeds`, with an `instance`
  mask input of `predicted` provenance, plus that segmentation network and its weights.
- **ScaleAP**, the reference's scale-wise evaluation.

</details>

<details markdown="1"><summary>Scale-balanced weights: the width histogram</summary>

The view and grasp losses of a seed are multiplied by `1 − ln(count / max count)` of the
bin, in a histogram of grasp widths, that holds the width of the seed's best grasp over
all views. Kept from the reference: a width on a bin edge or outside the histogram takes
the weight of bin 0, and a seed without any usable grasp takes the width of the first
cell of its label grid.

The histogram is a statistic of the training labels. The reference loads it from
`statistic/scale_distribution.npy`, relative to the working directory, at import; the
script that produced the file is not published. Here it is a derived asset that the
dataset attaches to every training sample (`data.train.dataset.scale_statistic`):

| `source` | What it is | Status |
|---|---|---|
| `native` | generated by `grasppanda prepare graspnet1b --scale-statistic`: widths of the best grasp of the *graspable* seeds of every 8th training frame, 32 equal bins | the shipped default; its shape agrees with the reference file, its counts do not reproduce it |
| `legacy_import` | the reference's own file, imported with `--import-legacy --scale-statistic-file` and pinned by its SHA-256, `909357a67265e93052ea309e455215298dbb067b8f4f3c2b511e015db7329cbd` | read exactly; use it to train with exactly the reference's weights |

How the native definition was chosen: the reference file holds 683 218 entries between
0.0030 and 0.1000 m. Its bin fractions were compared with three candidate populations.
Best-grasp widths per annotated point, per scene instance or per object model (tried on
ten scenes) have a different shape and widths below the file's range. Best-grasp widths
of graspable seeds on a sample of training frames have the file's shape: correlation of
the 32 bin fractions 0.997 with a largest difference of 0.4 percentage points on 80
frames (0.994 and 0.55 on the 40 frames of the integration test), with no width outside
the file's range. This is an inference from the shape of a distribution, not a
reproduction.

</details>

<details markdown="1"><summary>Equivalence with the reference code</summary>

`tests/integration/test_sbg_parity.py` compares this implementation with activations
recorded from the pristine reference (`tools/make_sbg_fixtures.py`), in true float32:

- conversion is strict and complete: 190 tensors, 1 577 196 parameters on both sides;
- four test frames: identical seeds, identical neighbour sets in both attention layers,
  identical views and objectness decisions; no crop of 65 536 differs in its neighbours;
- every stage fed what the reference fed it agrees within 9.3e-6 of the stage's largest
  value (the level of the other seed-based comparisons). One intermediate, the gate, is
  compared with a double-precision evaluation instead: the reference lies 1.6e-5 from it
  and this implementation 1.5e-5;
- end to end from the cloud, the features the heads receive agree within 4.1e-5 and the
  head scores within 3.4e-5. The excess is rounding of the seed features (which agree to
  6e-7) amplified 30 to 70 times by the gated residual, whose inputs reach magnitudes of
  130 to 150; decoded scores agree within 1.4e-6;
- one training batch of two frames: label matching, score, width, tolerance and view
  targets exact; the graspable target identical (369 of 775 object seeds); the
  scale-balanced weight of all 2048 seeds identical; six loss terms within 2e-7
  relative; total gradient norm within 1e-6 relative.

</details>

<details markdown="1"><summary>Reference behaviours kept as named options</summary>

- **Sampling ties** (`backbone.fps_tie_break: pointnet2_cuda`). In one of the four
  compared frames two points are equally far from the points sampled before them. The
  reference kernel and a lowest-index rule pick them in opposite order, which swaps two
  rows of the first level and with them eleven neighbour lists of the attention layer.
  The backbone therefore follows the kernel's order.
- **The reference environment.** PyTorch 1.10 enables TF32 by default; with it the
  reference differs from itself by up to 0.1 on the seed features and chooses other
  views for 2 to 8 seeds per frame. The fixtures are recorded with TF32 off.

</details>

<details markdown="1"><summary>How this port differs from the reference</summary>

- Noisy-clean mixing, object-balanced sampling and ScaleAP are not implemented (the
  reference's training data mixing, its test-time seed re-sampling and its scale-wise
  metric).
- The width histogram of the loss weights is generated here by a definition inferred
  from the shape of the reference's file; the file itself can be imported to train with
  exactly its weights.

</details>

## Citation

<details markdown="1"><summary>BibTeX</summary>

```bibtex
@inproceedings{ma2022scalebalanced,
  title     = {Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes},
  author    = {Ma, Haoxiang and Huang, Di},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2022}
}
```

</details>
