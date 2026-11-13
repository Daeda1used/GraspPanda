# Encoder comparison

**Support level: experimental.** One training run per encoder, one seed.

What does a scene encoder from another paper do under a grasp detection method when
nothing else changes? This project trains GSNet with the encoder it ships and with three
others, with GSNet's recipe, data and evaluator, and scores each on the whole
GraspNet-1Billion test split. It is the first use of the swappable stages beyond a
[probe](https://daeda1used.github.io/GraspPanda/evidence/): a probe says that a combination
trains, this says how good it becomes. Two questions the first table leaves open follow:
[is it the encoder or the recipe](#encoder-or-recipe), and
[is it GSNet](#the-same-swap-under-economicgrasp).

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense
grasppanda sweep projects/encoder_comparison/gsnet_encoders.yaml --dry-run        # four runs, each config checked
grasppanda sweep projects/encoder_comparison/gsnet_encoders.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8                                               # one run per GPU
```

`evaluation.num_workers` gives the evaluator processes of its own: scoring 23,040 frames in
the training process takes hours.

Every run writes into `work_dirs/gsnet_encoders/<encoder>/seed0/`, and the table of the
four into `work_dirs/gsnet_encoders/summary_test.md`. A run that is recorded is not repeated.

| Name | Encoder | Config |
|---|---|---|
| `sunet14` | the 14-layer sparse U-Net GSNet ships | [`gsnet_sunet14_g1b-realsense_10e`](../../configs/gsnet/gsnet_sunet14_g1b-realsense_10e.yaml) |
| `ptv3` | Point Transformer V3 | [`gsnet_ptv3_g1b-realsense_10e`](../../configs/experimental/gsnet_ptv3_g1b-realsense_10e.yaml) |
| `spunet` | SpUNet, the same kind of network three times as deep | [`gsnet_spunet_g1b-realsense_10e`](../../configs/experimental/gsnet_spunet_g1b-realsense_10e.yaml) |
| `litept` | LitePT-S | [`gsnet_litept_g1b-realsense_10e`](../../configs/experimental/gsnet_litept_g1b-realsense_10e.yaml) |

## Results

AP on the whole GraspNet-1Billion test split (RealSense, 23,040 frames) as seen / similar /
novel. The loss is the total training loss over the last epoch; the heads and labels are
the same in every run, so it compares.

| Encoder | Parameters | Collision filter off | On | Loss |
|---|---|---|---|---|
| `sunet14` | 15.4 M | 69.46 / 60.71 / 25.42 | 71.69 / 62.77 / 26.59 | 0.76 |
| `spunet` | 39.9 M | 68.84 / 60.22 / 25.04 | 70.98 / 62.22 / 26.36 | 0.79 |
| `ptv3` | 46.4 M | 59.07 / 52.55 / 21.91 | 62.28 / 55.62 / 23.48 | 0.96 |
| `litept` | 12.9 M | 55.23 / 50.03 / 21.60 | 59.18 / 53.50 / 23.04 | 1.18 |

**No difference is shown for SpUNet.** With 2.6 times the parameters it ends 0.2 to 0.7 AP
below the encoder GSNet ships. GSNet itself, trained three times from this config and seed,
moved by up to 1.12 AP ([its README](../../configs/gsnet/README.md)): one run cannot tell
the two apart.

**PTv3 is 10.4 AP lower on seen scenes, and under-trained.** The gap is far outside what a
seed explains, and the loss says where it comes from: after ten epochs PTv3 stands at 0.96,
above where the shipped encoder was after five (0.85). It has not fitted the training
scenes. GSNet's recipe is Adam at 1e-3 from the first step, without weight decay, and its
rate is still 6.3e-4 when training stops: tuned for a sparse CNN. The row measures an
encoder with a recipe made for another one, which is what the header of a composed config
says under *what must be re-tuned*.

**LitePT is 14.2 AP lower on seen scenes, and did not train steadily.** Its loss rose from
1.34 after the second epoch to 1.66 after the fourth before it fell again, and ends at 1.18,
above the shipped encoder's after two epochs (1.05). It is the same recipe: 1e-3 from the
first step, no warm-up. [Below](#encoder-or-recipe) it is trained under another one.

The 250-iteration check of the documentation had called all three swaps on a par with
the shipped encoder: it sorts out what fails or diverges and does not rank what passes
([How good does it become?](https://daeda1used.github.io/GraspPanda/advanced_guides/trained_swaps/)).

Records: [`records/`](records/), with the loss of every epoch, and
[`run3`](../../configs/gsnet/repeats/) of GSNet for `sunet14`. Unit tests hold the table and
these sentences to them.

## Encoder or recipe

[`gsnet_recipe.yaml`](gsnet_recipe.yaml) asks what the PTv3 and LitePT rows leave open. It
trains GSNet with the shipped encoder, with PTv3 and with LitePT under the recipe Pointcept
trains PTv3 with, the recipe preset `adamw-onecycle` (*one-cycle* below): AdamW with weight
decay 0.05, a warm-up over the first 5 % of the steps, then a cosine down to almost zero.
With the rows above, each of the three stands under both recipes.

```bash
grasppanda sweep projects/encoder_comparison/gsnet_recipe.yaml --gpus 0,1,2 \
    --opts evaluation.num_workers=8
```

| Encoder | Recipe | Collision filter off | On | Loss |
|---|---|---|---|---|
| `sunet14` | GSNet's | 69.46 / 60.71 / 25.42 | 71.69 / 62.77 / 26.59 | 0.76 |
| `sunet14` | one-cycle | 70.41 / 61.48 / 24.83 | 72.63 / 63.54 / 26.19 | 0.74 |
| `ptv3` | GSNet's | 59.07 / 52.55 / 21.91 | 62.28 / 55.62 / 23.48 | 0.96 |
| `ptv3` | one-cycle | 65.85 / 57.95 / 24.58 | 68.50 / 60.40 / 25.92 | 0.84 |
| `litept` | GSNet's | 55.23 / 50.03 / 21.60 | 59.18 / 53.50 / 23.04 | 1.18 |
| `litept` | one-cycle | 60.94 / 54.55 / 23.48 | 64.89 / 57.94 / 24.97 | 1.02 |

**The recipe is not shown to change the shipped encoder.** It fits the training scenes
closer (0.74 against 0.76) and ends +0.95 AP on seen and −0.59 on novel scenes without the
collision filter: every split inside what one seed moves.

**For PTv3 the recipe was more than half of the gap.** It gains 6.8 AP on seen scenes and
ends 4.6 AP below the shipped encoder under the same recipe, where it was 10.4 below under
GSNet's; on novel scenes the two are 0.25 apart. Its loss, 0.84, stays above the shipped
encoder's 0.74: these ten epochs do not fit it either. What is left of the gap is the
encoder, or a schedule still too short for it; that is not measured.

**For LitePT the recipe steadied the training and closed a third of the gap.** Its loss
now falls in every epoch and ends at 1.02. It gains 5.7 AP on seen scenes and ends 9.5 AP
below the shipped encoder under the same recipe, where it was 14.2 below under GSNet's.

Records: [`records/`](records/), the runs of this file by `<encoder>_adamw-onecycle`.

## The same swap under EconomicGrasp

Is the gap GSNet's?
[`economicgrasp_ptv3_g1b-realsense_10e`](../../configs/experimental/economicgrasp_ptv3_g1b-realsense_10e.yaml)
is EconomicGrasp with PTv3 for its encoder and nothing else changed, trained with
EconomicGrasp's recipe: Adam at 1e-3 without warm-up or weight decay, as GSNet's, but on a
cosine that ends at zero. The first row is the model zoo's
[EconomicGrasp](../../configs/economicgrasp/README.md#results-on-graspnet-1billion): the
same config and seed with the encoder it ships.

```bash
grasppanda train experimental/economicgrasp_ptv3_g1b-realsense_10e
grasppanda test experimental/economicgrasp_ptv3_g1b-realsense_10e \
    --weights <the checkpoint of epoch 10> --opts evaluation.num_workers=8
```

| Encoder | Collision filter off | On | Loss |
|---|---|---|---|
| `sunet14` | 69.34 / 61.41 / 26.21 | 71.05 / 62.54 / 26.81 | 2.98 |
| `ptv3` | 65.52 / 57.11 / 23.91 | 67.14 / 58.43 / 24.49 | 3.31 |

**PTv3 ends 3.8 AP lower on seen scenes, 2.3 on novel.** That is the gap GSNet keeps under
the one-cycle recipe (4.6), not the one under GSNet's own (10.4). Both recipes with the
smaller gap anneal the rate to almost zero, and GSNet's stops at 6.3e-4; the two methods
differ in more than that, and one run each does not say that this is why.

**It is under-trained here too.** After ten epochs its loss, 3.31, is where the shipped
encoder was after four (3.33). The loss terms are EconomicGrasp's, so the column compares
within this table only.

Records: [`records/`](records/) as `economicgrasp_ptv3`, and EconomicGrasp's
[reproduced records](../../configs/economicgrasp/records/) for the first row.
