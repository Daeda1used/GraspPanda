# Example project

**Support level: example.** A template for adding a method without forking GraspPanda.
It trains, tests and infers end to end on a CPU in seconds, on synthetic scenes whose
labels are simple functions of the point coordinates. What it learns is a toy task:
copy the structure, not the numbers.

What it shows:

| File | Shows |
|---|---|
| `toy_components.py` | Registering components, declaring slot contracts (`output`, `out_axes`, `in_axes`), receiving injected arguments (`INJECTED`), a config check that compares contracts before anything is built, a `loss` that returns `loss/*` terms and a metric, and a post-processor. |
| `toy_data.py` | A dataset addressed by `SampleKey` whose labels reach the model through `Supervision` only, and an evaluator that reads dumped predictions and returns per-frame precision tables. |
| `configs/toy_detector.yaml` | Inheriting shipped base configs by name (`_base_/default_runtime.yaml`, found from any directory and from an installed wheel) and loading out-of-tree code with `custom_imports`. |
| `configs/toy_train.yaml` | Everything a run needs: data sections, evaluator, post-processing variants, the metric that selects the best checkpoint. |

## Validate

From the repository root (or from the parent of a copy of this folder: nothing in it
points into the checkout):

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml --check
grasppanda print-config projects/example_project/configs/toy_train.yaml --for train --check

# a typo is an error, with a suggestion
grasppanda print-config projects/example_project/configs/toy_detector.yaml --opts model.head.num_angles=6

# an incompatible swap is rejected before anything runs
grasppanda print-config projects/example_project/configs/toy_detector.yaml --opts model.grouper.type=ToyDepthGrouper
```

## Train, test, infer

```bash
CFG=projects/example_project/configs/toy_train.yaml

grasppanda train $CFG --opts runtime.run_id=demo
grasppanda test  $CFG --weights work_dirs/toy_train/demo/checkpoints/best.pth
grasppanda infer $CFG --weights work_dirs/toy_train/demo/checkpoints/best.pth --input val --output out/toy_val
```

`train` evaluates on the validation scenes after every epoch and keeps `best.pth` by
`raw/AP`. Both variants of `evaluation.variants` are always reported: `raw` (the
predictions as the model emits them) and `centre` (the same predictions after
`ToyCentreFilter`). The filter removes grasps, and the toy evaluator counts a missing
grasp among the top ten as a failure, so `centre` scores lower: the point is that both
numbers appear side by side, each with its own result record. See the
[training and testing guide](../../docs/user_guides/train_test.md).

## From Python

```python
import torch
from grasppanda.config import Config
from grasppanda.registry import DETECTORS
from grasppanda.structures import Observation, PointSet, SampleKey

config = Config.load("projects/example_project/configs/toy_detector.yaml")
model = DETECTORS.build(config["model"]).eval()
observation = Observation(
    keys=[SampleKey("demo", 0, "none", 0)],
    points=PointSet.from_dense(torch.rand(1, 2048, 3)),
)
grasps = model.predict(observation)[0]  # a GraspGroup of 64 grasps (untrained: arbitrary)
```
