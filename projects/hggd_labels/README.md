# HGGD's labels

**Support level: experimental.** One training run per label density, one seed.

HGGD is trained on grasp labels that a converter thins first: of the grasps annotated in a
frame it keeps one per 2 cm and 60°. Trained here on those labels, HGGD ends 8 to 10 AP
below its authors' checkpoint on seen scenes, and the checkpoint does not fit them: its
training loss falls on denser labels
([HGGD's README](../../configs/hggd/README.md#results-on-graspnet-1billion)). This project
trains HGGD on denser labels and changes nothing else: model, recipe, evaluator and seed
are those of the model zoo's row. No code is written for it: the sweep file is the
experiment.

```bash
grasppanda prepare graspnet1b --labels
grasppanda prepare graspnet1b --view-grasps --camera realsense \
    --config hggd/hggd_r34w8_g1b-realsense_15e \
    --opts data.train.dataset.view_grasps.nms_distance=0.01 \
           data.train.dataset.view_grasps.nms_angle=30.0          # the labels of `1cm-30deg`
grasppanda sweep projects/hggd_labels/labels.yaml --dry-run       # two runs, each config checked
grasppanda sweep projects/hggd_labels/labels.yaml --gpus 0,1 \
    --opts evaluation.num_workers=8                               # one run per GPU
```

A run reads labels of its own, and `prepare` builds them from the overrides of the run:
the two `--opts` above are the two lines of `1cm-30deg` in [`labels.yaml`](labels.yaml).
Every run writes into `work_dirs/labels/<labels>/seed0/`.

| Name | Suppression of the labels | Labels a frame | Asset, 100 scenes |
|---|---|---|---|
| none: the model zoo's row | 2 cm, 60°: the released converter | 131 | 0.2 GB |
| `1cm-30deg` | 1 cm, 30°: a line the converter has commented out | 419 | 0.5 GB |
| `unsuppressed` | none | 4611 | 4.8 GB |

## Results

AP on the whole GraspNet-1Billion test split (RealSense, 23,040 frames) as seen / similar /
novel. *On* is HGGD's collision filter, then its NMS: what the released test script dumps.

| Labels | Collision filter off | On |
|---|---|---|
| the converter's: HGGD in the model zoo | 48.01 / 41.42 / 18.88 | 55.30 / 48.60 / 21.70 |
| `1cm-30deg` | 57.83 / 48.53 / 22.71 | 63.61 / 54.31 / 24.95 |
| `unsuppressed` | 62.40 / 49.54 / 21.55 | 67.14 / 54.01 / 23.72 |
| the authors' checkpoint | 58.36 / 47.95 / 22.08 | 63.72 / 52.91 / 24.16 |

**On `1cm-30deg` HGGD lands on its authors' checkpoint.** It ends
−0.53 / +0.58 / +0.63 AP from it without the collision filter and −0.11 / +1.40 / +0.79
with it, where the converter's labels leave it −10.35 / −6.53 / −3.20 and
−8.42 / −4.31 / −2.46. Nothing but the two label options differs between the two
trainings: +9.82 AP on seen scenes without the filter comes from the labels alone.

**It is inside the model zoo's gate against the repository's numbers.** HGGD's repository
lists 64.45 / 53.59 / 24.59; with the filter the model trained on `1cm-30deg` is
−0.84 / +0.72 / +0.36 AP from that, where the gate is 1 AP on each split. The paper's
numbers, 59.36 / 51.20 / 22.17, do not state the filter either; without it the model is
−1.53 / −2.67 / +0.54 from them.

**Without suppression it ends above the checkpoint on seen scenes, not on novel.** On all
4,611 labels a frame HGGD ends +4.04 / +1.59 / −0.53 AP from the checkpoint without the
collision filter and +3.42 / +1.10 / −0.44 with it; with the filter that is
+2.69 / +0.42 / −0.87 from the repository's numbers, outside the gate on seen scenes.
Against `1cm-30deg` it is +4.57 / +1.01 / −1.16 AP without the filter: on seen scenes far
beyond the 1.75 AP by which training again moved EconomicGrasp, on novel inside it. Of the
three densities `1cm-30deg` is the one on the checkpoint; the checkpoint's loss does not
tell the two denser ones apart: 1.85 on `1cm-30deg`, 1.67 on the unsuppressed labels.

**The gap opens in the first epochs.** On 160 frames of the training scenes the model on
`1cm-30deg` is at 48.57 AP after four epochs and 52.18 after the fifteenth, the one on the
unsuppressed labels at 49.62 and 58.04; on the converter's labels it is at 32.65 and 43.76
([`validation.json`](records/validation.json)).

**What it does not say.** One training per row: HGGD has not been trained twice from one
config here, and two other methods moved by 1.12 and 1.75 AP when they were. Whether these
are the labels the checkpoint was trained on is not known either; they are the ones a
commented line of the released converter writes, and the checkpoint fits them better than
the converter's. The shipped config keeps the converter's labels, as released.

## On Kinect

The same two overrides under HGGD's Kinect config, trained and scored on the Kinect frames
of the test split:

```bash
grasppanda prepare graspnet1b --view-grasps --camera kinect \
    --config hggd/hggd_r34w8_g1b-kinect_15e --opts <the two overrides of `1cm-30deg`>
grasppanda train hggd/hggd_r34w8_g1b-kinect_15e --opts <the same two>
```

| Labels | Collision filter off | On |
|---|---|---|
| the converter's: HGGD in the model zoo | 49.73 / 40.52 / 17.41 | 54.46 / 44.57 / 18.83 |
| `1cm-30deg` | 59.08 / 46.05 / 17.13 | 62.77 / 49.45 / 18.28 |
| the authors' checkpoint | 56.98 / 43.97 / 18.56 | 60.30 / 46.85 / 19.22 |

**On Kinect the converter's labels leave HGGD below the checkpoint as on RealSense**:
−7.25 / −3.45 / −1.15 AP without the collision filter and −5.84 / −2.28 / −0.39 with it.
`1cm-30deg` gains +9.35 / +5.53 / −0.28 and +8.31 / +4.88 / −0.55 over them: on seen
scenes about as much as on RealSense (+9.82), on novel nothing, where RealSense gained
+3.83.

**The denser labels end above the checkpoint on seen and similar scenes, and below it on
novel**: +2.10 / +2.08 / −1.43 AP without the filter and +2.47 / +2.60 / −0.94 with it.
Against the repository's Kinect numbers, 61.17 / 47.02 / 19.37, that is
+1.60 / +2.43 / −1.09 AP with the filter: outside the zoo's gate on every split. On this
camera neither label set lands on the checkpoint.

Records: [`records/`](records/), and HGGD's
[records](../../configs/hggd/records/) for the authors' checkpoint and the model zoo's row.
Unit tests hold the tables and these sentences to them.
