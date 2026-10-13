# Quick start

Everything on this page works today and needs no dataset. It uses the
[example project](https://github.com/Daeda1used/GraspPanda/tree/main/projects/example_project),
a tiny untrained detector wired exactly like a real method.

## 1. Validate a config

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml --check
```

```text
✓ projects/example_project/configs/toy_detector.yaml is valid
  inherits    default_runtime.yaml, gsnet_10e.yaml
  components  optimizer: Adam · scheduler: ExponentialLR · model: ToyDetector · …
  hash        13d2bf9633e8
```

## 2. See what a run would use

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml --defaults
```

prints the config after inheritance, overrides and interpolation, with every default
written out. This is exactly what a run stores as `config.resolved.yaml`.

## 3. Make a mistake on purpose

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --opts model.head.num_angles=6
```

```text
✗ 1 config problem in projects/example_project/configs/toy_detector.yaml

  model.head.num_angles
    ToyHead has no parameter 'num_angles'. Did you mean 'num_angle'? …
    set in: <command line>
```

An incompatible component swap is rejected the same way, before anything is built:

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --opts model.grouper.type=ToyDepthGrouper
```

## 4. Build and run from Python

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
grasps = model.predict(observation)[0]
print(len(grasps), grasps.to_graspnet_array().shape)   # 64 (64, 17)
```

The toy detector is untrained; its grasps are arbitrary. Training, evaluation and
pretrained models arrive with milestone M1.

## Next

- [Configuration guide](../user_guides/config.md)
- [Add your own component](../advanced_guides/add_component.md)
