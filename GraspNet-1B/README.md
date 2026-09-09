# GraspNet-1B

Choose a method in the browser and **Load preset**, or generate an editable configuration:

```bash
./panda init --method graspness -o graspness.local.yaml
./panda weights graspness --camera realsense
# Set dataset_root in graspness.local.yaml to the directory containing scenes/.
./panda run graspness.local.yaml
```

Run commands from the repository root after [installation](../docs/INSTALL.md). Configuration generation does not require data, weights or a GPU, and never overwrites an existing file. [Data & weights](../docs/DOWNLOADS.md) explains downloads and preprocessing. [Using GraspPanda](../docs/USAGE.md) covers training, checkpoint resume and evaluation.

## Configuration examples

Examples are defined in one [template catalogue](examples.yaml). Generate only the files you need:

```bash
./panda init --list
./panda init --example compose-baseline -o composition.local.yaml
./panda init --example train-hggd -o training.local.yaml
```

| Workflow | Example names |
|---|---|
| Inference | `infer-graspness` or `--method METHOD` for any runnable preset |
| Training | `train-baseline`, `train-economicgrasp`, `train-hggd`, `train-finegrasp`, `train-gtg2` |
| Losses and augmentation | `train-controls`, `compose-hggd` |
| Point encoders and grouping | `compose-baseline`, `compose-ptv3`, `compose-octformer`, `compose-pcm`, `compose-pointmamba`, `compose-pointmeta`, `compose-pointvector`, `compose-reslfe` |
| EconomicGrasp composition | `compose-economicgrasp`; [native stages, grouping and training controls](../docs/MODULES.md#economicgrasp-components) |
| Grouped seed interaction | `compose-seed-interaction`; [parameters and method compatibility](../docs/MODULES.md#grouped-seed-interaction) |
| Grasp prediction heads | `compose-economic-head`; [branch and attention settings](../docs/MODULES.md#economicgrasp-interactive-head) |
| Image encoders | `compose-hggd`, `compose-dino`, `compose-vmamba` |
| Parameter sweeps | `sweep-baseline`; use `./panda sweep FILE --preview` before execution |

Each generated file includes its method, operation and settings. Set your local paths, prepare the method's labels for training, and use `./panda run FILE` (or `./panda sweep FILE` for a sweep). The [module guide](../docs/MODULES.md) explains compatible replacements and parameters. Files named `*.local.yaml` are ignored by Git.

## Observation protocols

Single-view point clouds, RGB-D, fused views, active perception and temporal sequences have different input contracts. The selected preset preserves its method's protocol. Find operation availability, limitations, PDFs and original implementations in [Methods & papers](../docs/METHODS.md).

Original implementations are fetched into `upstream/` by protocol. Runtime caches, weights and experiment outputs are generated locally; see the [local directory layout](../docs/INSTALL.md#files-created-locally).
