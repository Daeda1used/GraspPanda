# FineGrasp's changes, one at a time

**Support level: experimental.** One seed; one to three training runs per row.

FineGrasp is EconomicGrasp with three changes, and here it is built as exactly that: three
groups of config entries. This project switches each on alone, and all three together,
under EconomicGrasp's recipe (one process, batches of four, Adam at 1e-3), data and
evaluator, and scores each run on the whole GraspNet-1Billion test split. No code is
written for it: the sweep file is the experiment.

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --graspness --camera realsense          # EconomicGrasp's labels
grasppanda prepare graspnet1b --graspness --camera realsense \
    --config finegrasp/finegrasp_sunet14_g1b-realsense_10e            # normalised per object
grasppanda prepare graspnet1b --compact-labels \
    --config economicgrasp/economicgrasp_sunet14_g1b-realsense_10e
grasppanda sweep projects/finegrasp_ablation/changes.yaml --dry-run   # six runs, each config checked
grasppanda sweep projects/finegrasp_ablation/changes.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8                                   # one run per GPU
```

Every run writes into `work_dirs/changes/<change>/seed0/`, and the table of the six into
`work_dirs/changes/summary_test.md`. A run that is recorded is not repeated.

| Name | What changes | Config entries |
|---|---|---|
| `graspness` | Graspness is normalised within every object before the frame, so a small object beside an easy one keeps seeds. | the label asset: `graspness.normalization: instance` |
| `normals` | The encoder reads `[x, y, z, normal]` per point instead of three ones. | `model.point_features`, `backbone.in_channels`, two data transforms |
| `multi-scale` | Every seed is cropped at four radii; attention over the scales fuses the four. | `model.grouper`: `MultiScaleGroup` around EconomicGrasp's crop |
| `graspness+multi-scale` | The crop with per-object graspness. | the two groups |
| `normals+multi-scale` | The crop with normals as encoder input. | the two groups |
| `all` | The three together: FineGrasp's model and data, trained as EconomicGrasp is. | all of the above |

The run without a change is the model zoo's
[EconomicGrasp row](../../configs/economicgrasp/README.md#results-on-graspnet-1billion):
the same config and seed. A test holds `all` to the shipped FineGrasp config, entry by
entry (`tests/unit/test_sweep.py`).

**Why.** Trained here with its own recipe, FineGrasp ends 1.2 to 3.1 AP below its paper,
while EconomicGrasp reaches its authors' checkpoint
([FineGrasp's README](../../configs/finegrasp/README.md#results-on-graspnet-1billion)).
`all` separates the two things that could explain it: if the FineGrasp model gains over
EconomicGrasp under this recipe, the gap lies in FineGrasp's recipe as it was run here; if
it does not, in the three changes as they are generated here.

## Results

AP on the whole GraspNet-1Billion test split (RealSense, 23,040 frames) as seen / similar /
novel. A row *trained again* repeats the row above it from the same config and seed. The
three trainings of EconomicGrasp lie 0.50 / 1.57 / 1.12 AP apart without the collision
filter ([its README](../../configs/economicgrasp/README.md#results-on-graspnet-1billion)),
and two trainings of a change up to 1.29 / 1.95 / 0.70: a change that moves a split by less
is not a finding. The two rows that pair the crop with one other change were trained once.

| Change | Collision filter off | On |
|---|---|---|
| none: EconomicGrasp | 69.34 / 61.41 / 26.21 | 71.05 / 62.54 / 26.81 |
| none, trained again | 69.01 / 60.02 / 25.09 | 70.65 / 61.10 / 25.74 |
| none, trained a third time | 68.84 / 61.59 / 25.70 | 70.53 / 62.85 / 26.38 |
| `graspness` | 70.03 / 61.03 / 26.21 | 71.47 / 62.12 / 26.73 |
| `graspness`, trained again | 69.42 / 60.66 / 26.06 | 70.94 / 61.82 / 26.64 |
| `normals` | 68.05 / 60.78 / 26.23 | 70.11 / 62.00 / 26.77 |
| `normals`, trained again | 68.09 / 60.98 / 26.08 | 69.94 / 62.36 / 26.65 |
| `multi-scale` | 68.10 / 58.63 / 24.05 | 69.84 / 59.77 / 24.71 |
| `multi-scale`, trained again | 67.21 / 56.68 / 23.35 | 69.09 / 57.92 / 24.06 |
| `graspness+multi-scale` | 68.07 / 59.53 / 24.89 | 69.65 / 60.68 / 25.49 |
| `normals+multi-scale` | 68.93 / 59.25 / 25.40 | 70.47 / 60.21 / 25.95 |
| `all` | 70.54 / 61.27 / 25.41 | 71.72 / 62.09 / 25.96 |
| `all`, trained again | 69.25 / 60.34 / 25.66 | 70.67 / 61.36 / 26.24 |

**No difference is shown for `graspness`.** Its first training ends 0.69 AP above the three
trainings of EconomicGrasp on seen scenes without the collision filter, its second 0.08;
on the other splits both lie inside their range. Between the means that is
+0.66 / −0.16 / +0.47 AP.

**`normals` ends lower on seen scenes.** Both of its trainings end below all three of
EconomicGrasp's there, by 0.75 to 1.29 AP without the collision filter, and lie 0.04 AP
apart; on the other splits both lie inside the range of the three, but for +0.02 AP on
novel. Between the means that is −0.99 / −0.13 / +0.49 AP. Its training loss is
EconomicGrasp's: 2.97 in both trainings against 2.98 over the last epoch.

**`multi-scale` alone ends lower.** Both of its trainings end below all three of
EconomicGrasp's on every split, with the collision filter and without. Between the means
that is 1.41 / 3.35 / 1.97 AP without the filter. It is not under-trained: its training
loss is the lower one, 2.77 and 2.78 against EconomicGrasp's 2.98 over the last epoch with
the same labels and loss terms. Nor does it lean on its training mode: on 160 training
frames its loss is 2.86 with the layers as in training and 2.73 as in evaluation, measured
on the first training ([`tools/train_fit.py`](../../tools/train_fit.py)). Why the lower
loss comes with the lower AP is not known.

**`all`, the FineGrasp model, is not shown to differ from EconomicGrasp.** Its first
training ends above the three trainings of EconomicGrasp on seen scenes, by 1.20 to 1.70 AP
without the collision filter; its second ends inside their range on every split. Between
the means the difference is +0.83 / −0.20 / −0.13 AP without the filter and
+0.45 / −0.44 / −0.21 with it. Over its two trainings it is 1.0 to 3.3 AP below the authors'
checkpoint, where FineGrasp trained with its own recipe is 1.2 to 3.1 below: the recipe is
not what separates the model trained here from the authors'.

**The parts do not add up: the crop needs both other changes.** The crop loses
1.41 / 3.35 / 1.97 AP alone, and every training of `all` ends above every training of
`multi-scale` on every split. With per-object graspness alone the crop still ends below all
three trainings of EconomicGrasp on every split, −0.99 / −1.48 / −0.78 AP from their mean
without the filter; with normals alone below them on similar scenes only,
−0.13 / −1.76 / −0.27. One run each.

Records: [`records/`](records/), with the loss of every epoch; EconomicGrasp's
[reproduced records](../../configs/economicgrasp/records/) for the first row and its
[repeats](../../configs/economicgrasp/repeats/) for the two below it. Unit tests hold the
table and these sentences to them.
