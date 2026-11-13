---
title: How good does it become?
---

# How good does it become?

A swap that [runs](measured_combinations.md) and
[trains like the shipped component](learns_like_reference.md) has not shown one point of AP.
Here a few are trained to the end of their recipe and scored on the whole GraspNet-1Billion
test split, as the [model zoo](../model_zoo.md) scores a method. It is the one check that
measures what a swap is worth, and so the one that says what the two cheaper checks are
worth.

**One training per row, seed 0.** Trained again from one config and seed, GSNet moved by up
to 1.12 AP over three trainings and EconomicGrasp by up to 1.44 over two: a difference
below about 1 AP is not a finding.

## Scene encoders under GSNet {#encoders}

GSNet with another scene encoder and nothing else changed: the recipe, the data and the
evaluator are GSNet's. AP on RealSense, 23,040 frames, as seen / similar / novel; *Loss* is
the total training loss over the last epoch. `sunet14` is the encoder GSNet ships.

<div class="gp-figures" markdown>

--8<-- "docs/includes/trained_encoders.md"

</div>

- **SpUNet: no difference shown.** With 2.6 times the parameters it ends 0.2 to 0.7 AP below
  the shipped encoder, inside what one seed moves.
- **PTv3: 10.4 AP lower on seen scenes, and under-trained.** After ten epochs its training
  loss, 0.96, is above the shipped encoder's after five, 0.85. GSNet's recipe is Adam at 1e-3
  from the first step, without weight decay, tuned for a sparse CNN: the row measures an
  encoder under a recipe made for another one.
- **LitePT: 14.2 AP lower on seen scenes, and unsteady.** Its training loss rose from 1.34
  after the second epoch to 1.66 after the fourth, and ends at 1.18.

### Encoder or recipe {#recipe}

The shipped encoder, PTv3 and LitePT again, under the recipe PTv3 is trained with
elsewhere: the preset [`adamw-onecycle`](../slots/recipe.md), AdamW with weight decay, a
warm-up, then a cosine down to almost zero (*one-cycle* below).

<div class="gp-figures" markdown>

--8<-- "docs/includes/trained_recipes.md"

</div>

- **Shipped encoder: the recipe is not shown to matter.** +0.95 AP on seen and −0.59 on
  novel scenes without the collision filter, inside what one seed moves.
- **PTv3: the recipe was more than half of the gap.** It gains 6.8 AP on seen scenes and
  ends 4.6 AP below the shipped encoder under the same recipe; on novel scenes the two are
  0.25 apart. Its loss, 0.84, stays above the shipped encoder's 0.74.
- **LitePT: steadier, and a third of the gap closed.** Its loss now falls in every epoch
  and ends at 1.02. It gains 5.7 AP on seen scenes and ends 9.5 AP below the shipped
  encoder under the same recipe.

## The same swap under EconomicGrasp {#economicgrasp}

EconomicGrasp with PTv3 and nothing else changed, under EconomicGrasp's recipe: Adam at 1e-3
as GSNet's, on a cosine that ends at zero. The first row is the
[model zoo's](../model_zoo.md).

<div class="gp-figures" markdown>

--8<-- "docs/includes/trained_economicgrasp.md"

</div>

- **PTv3: 3.8 AP lower on seen scenes.** The gap GSNet keeps under one-cycle (4.6), not the
  one under its own recipe (10.4). Both recipes with the smaller gap anneal the rate to
  almost zero; one run each does not say that this is why. The first row trained again
  ends 0.33 AP lower on seen scenes: the gap is not the spread of trainings.
- **Under-trained here too.** After ten epochs its loss, 3.31, is where the shipped encoder
  was after four (3.33).

## What the cheaper checks had said {#cheaper-checks}

<div class="gp-figures" markdown>

| Swap | [Does it run?](measured_combinations.md) | [Trains like shipped?](learns_like_reference.md) | Trained in full: seen, filter off |
|---|---|---|---|
| `spunet` | runs | on a par | −0.62 AP, within a seed |
| `ptv3` | runs | on a par | −10.39 AP; −4.56 under one-cycle |
| `litept` | runs | on a par | −14.23 AP; −9.47 under one-cycle |

</div>

All three passed both, and so did PTv3 under EconomicGrasp. In 250 iterations on eight
frames PTv3 under GSNet brought the graspness loss lower than the shipped encoder did; over
ten epochs on 25,600 frames it fell behind. The short checks sort out the combinations that
fail or diverge. They do not rank the ones that pass.

## FineGrasp's changes, one at a time {#finegrasp}

FineGrasp is EconomicGrasp with three changes, each a group of config entries: graspness
normalised per object, normals as encoder input, and a crop at four radii. Each is switched
on alone, and all three together, under EconomicGrasp's recipe. A row *trained again*
repeats the row above it from the same config and seed: the yardstick.

<div class="gp-figures" markdown>

--8<-- "docs/includes/trained_changes.md"

</div>

- **No change alone shows a gain, and one a loss.** `graspness` moves no split by more than
  0.7 AP; `normals` ends 0.96 to 1.29 AP below both trainings of EconomicGrasp on seen
  scenes; `multi-scale`, trained twice, ends below both on every split in both trainings,
  by 1.52 / 3.06 / 1.95 AP on average.
- **All three: above both on seen scenes, between them elsewhere.** +1.20 and +1.53 AP on
  seen. That is 1.0 to 2.5 AP below the authors' checkpoint, where FineGrasp trained with
  its own recipe is 1.2 to 3.1 below: the recipe is not what separates the model trained
  here from theirs.
- **Trainings spread.** The two of EconomicGrasp differ by up to 1.44 AP, the two of
  `multi-scale` by up to 1.95. `all` is being trained again.

## Run it {#run}

Each experiment is a sweep file under
[`projects/`](https://github.com/Daeda1used/GraspPanda/tree/main/projects), with no code
beside it and the result records of every row:

```bash
grasppanda sweep projects/encoder_comparison/gsnet_encoders.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8
grasppanda sweep projects/finegrasp_ablation/changes.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8
```
