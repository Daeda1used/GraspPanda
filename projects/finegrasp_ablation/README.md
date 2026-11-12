# FineGrasp's changes, one at a time

**Support level: experimental.** One training run per change, one seed.

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
grasppanda sweep projects/finegrasp_ablation/changes.yaml --dry-run   # four runs, each config checked
grasppanda sweep projects/finegrasp_ablation/changes.yaml --gpus 0,1,2,3 \
    --opts evaluation.num_workers=8                                   # one run per GPU
```

Every run writes into `work_dirs/changes/<change>/seed0/`, and the table of the four into
`work_dirs/changes/summary_test.md`. A run that is recorded is not repeated.

| Name | What changes | Config entries |
|---|---|---|
| `graspness` | Graspness is normalised within every object before the frame, so a small object beside an easy one keeps seeds. | the label asset: `graspness.normalization: instance` |
| `normals` | The encoder reads `[x, y, z, normal]` per point instead of three ones. | `model.point_features`, `backbone.in_channels`, two data transforms |
| `multi-scale` | Every seed is cropped at four radii; attention over the scales fuses the four. | `model.grouper`: `MultiScaleGroup` around EconomicGrasp's crop |
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
novel. GSNet, trained three times from one config and seed, moved by up to 1.12 AP
([its README](../../configs/gsnet/README.md)): a smaller difference is not a finding.

| Change | Collision filter off | On |
|---|---|---|
| none: EconomicGrasp | 69.34 / 61.41 / 26.21 | 71.05 / 62.54 / 26.81 |
| `graspness` | 70.03 / 61.03 / 26.21 | 71.47 / 62.12 / 26.73 |
| `normals` | training | |
| `multi-scale` | training | |
| `all` | training | |

**No difference is shown for `graspness`.** Normalising graspness per object moves no split
by more than 0.7 AP: +0.69 on seen and −0.38 on similar without the collision filter.

Records: [`records/`](records/), and EconomicGrasp's
[reproduced records](../../configs/economicgrasp/records/) for the first row. Unit tests
hold the table and these sentences to them.
