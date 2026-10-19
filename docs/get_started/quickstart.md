# Tour the config (optional)

Ten minutes with the [example project](https://github.com/Daeda1used/GraspPanda/tree/main/projects/example_project),
no dataset needed: how a config is checked, what a run uses, and how a mistake is caught
before anything is built. The example is a tiny untrained detector, wired like a real method.

## 1. Validate a config

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --check
```

```text
✓ projects/example_project/configs/toy_detector.yaml is valid
  inherits    default_runtime.yaml, adam_10e.yaml
  components  optimizer: Adam · scheduler: ExponentialLR · model: ToyDetector · model.backbone: ToyBackbone · model.grouper: ToyGrouper · model.head: ToyHead
  hash        53bed832c67f
```

The config inherits two shipped building blocks: the runtime defaults and a plain training
recipe (`_base_/schedules/adam_10e.yaml`: Adam, a learning rate of 1e-3, 10 epochs).

## 2. See what a run would use

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --defaults
```

It prints the config after inheritance, overrides and interpolation, with every default
written out. This is exactly what a run stores as `config.resolved.yaml`.

??? example "The output"

    ```yaml
    train:
      amp: false
      accumulate: 1
      epochs: 10
      grad_clip: null
      grad_clip_mode: norm
      ema: null
      sync_bn: false
      scheduler_unit: epoch
    evaluation:
      interval: 1
      variants:
        raw: []
      evaluator: null
      save_best: null
      split: test
      weights: model
      num_workers: 0
    runtime:
      work_dir: work_dirs/toy_detector
      seed: 0
      deterministic: false
      checkpoint:
        interval: 1
        keep_last: 3
      log:
        interval: 50
        backends:
        - text
      run_id: null
      device: auto
      resume: false
      allow_config_mismatch: false
      load_from: null
      allow_weight_mismatch: false
      ops: {}
    optimizer:
      type: Adam
      lr: 0.001
      weight_decay: 0.0
      betas:
      - 0.9
      - 0.999
      eps: 1.0e-08
      amsgrad: false
      foreach: null
      maximize: false
      capturable: false
      differentiable: false
      fused: null
      decoupled_weight_decay: false
    scheduler:
      type: ExponentialLR
      gamma: 0.95
      last_epoch: -1
    custom_imports:
    - ../toy_components.py
    model:
      type: ToyDetector
      num_seeds: 64
      backbone:
        type: ToyBackbone
        out_channels: 64
        in_channels: 3
      grouper:
        type: ToyGrouper
        radius: 0.05
        nsample: 16
      head:
        type: ToyHead
        num_angle: 12
        num_depth: 4
        depth_step: 0.01
        max_width: 0.1
        width_weight: 10.0
    data: null
    hooks: []
    ```

## 3. Make a mistake on purpose

A misspelt key is an error that names the closest parameter:

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --opts model.head.num_angles=6
```

```text
✗ 1 config problem in projects/example_project/configs/toy_detector.yaml

  model.head.num_angles
    ToyHead has no parameter 'num_angles'. Did you mean 'num_angle'? Parameters: num_angle, num_depth, depth_step, max_width, width_weight, in_channels.
    set in: <command line>
```

A swap that cannot work is refused the same way, before anything is built:

```bash
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --opts model.grouper.type=ToyDepthGrouper
```

```text
✗ 1 config problem in projects/example_project/configs/toy_detector.yaml

  model
    ToyDepthGrouper emits features with axes ('seed', 'depth', 'channel'), but ToyHead expects ('seed', 'channel'). Use a grouper and a head that agree; a depth axis is never added or squeezed implicitly.
    set in: projects/example_project/configs/toy_detector.yaml
```

Both exit with status 2, so a script can stop on them.

## 4. Build and run from Python

```python
import torch
from grasppanda.config import Config
from grasppanda.registry import DETECTORS
from grasppanda.structures import Observation, PointSet, SampleKey

config = Config.load("projects/example_project/configs/toy_detector.yaml")
model = DETECTORS.build(config["model"]).eval()
points = PointSet.from_dense(torch.rand(1, 2048, 3))
observation = Observation(keys=[SampleKey("demo", 0, "none", 0)], points=points)
grasps = model.predict(observation)[0]
print(len(grasps), grasps.to_graspnet_array().shape)   # 64 (64, 17)
```

The toy detector is untrained, so its grasps are arbitrary.

## Next

- Run a real method from its released weights:
  [GraspNet-baseline](../methods/graspnet_baseline.md#run-in-five-minutes) ·
  [GSNet](../methods/gsnet.md#run-in-five-minutes) ·
  [Scale-Balanced-Grasp](../methods/scale_balanced_grasp.md#run-in-five-minutes) ·
  [EconomicGrasp](../methods/economicgrasp.md#run-in-five-minutes) ·
  [FineGrasp](../methods/finegrasp.md#run-in-five-minutes) ·
  [HGGD](../methods/hggd.md#run-in-five-minutes)
- The whole config language: [Configuration](../user_guides/config.md)
- Your own component, without forking: [Add a component](../advanced_guides/add_component.md)
