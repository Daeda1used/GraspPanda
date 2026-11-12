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

**One training per row, seed 0.** GSNet, trained three times from one config and seed, moved
by up to 1.12 AP: a difference below about 1 AP is not a finding.

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

Whether PTv3 closes the gap under the recipe it is trained with elsewhere is being measured:
both encoders under the recipe preset [`adamw-onecycle`](../slots/recipe.md).

## What the cheaper checks had said {#cheaper-checks}

| Swap | [Does it run?](measured_combinations.md) | [Trains like shipped?](learns_like_reference.md) | Trained in full: seen, filter off |
|---|---|---|---|
| `spunet` | runs | on a par | −0.62 AP, within a seed |
| `ptv3` | runs | on a par | −10.39 AP |
| `litept` | runs | on a par | training |

All three passed both. In 250 iterations on eight frames PTv3 brought the graspness loss
lower than the shipped encoder did; over ten epochs on 25,600 frames it fell behind. The
short checks sort out the combinations that fail or diverge. They do not rank the ones that
pass.

## Run it {#run}

Each experiment is a sweep file under
[`projects/`](https://github.com/Daeda1used/GraspPanda/tree/main/projects), with no code
beside it and the result records of every row:

```bash
grasppanda sweep projects/encoder_comparison/gsnet_encoders.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8
```
