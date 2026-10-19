# Configuration

**Goal:** read a config, start your own from a shipped one, change it, and check it before
anything runs.

A config is one YAML file. It describes the whole experiment: model, data, optimisation,
evaluation and runtime. No behaviour is reachable only through a command-line flag.

**You need:** GraspPanda installed ([Install](../get_started/installation.md)). Every
command on this page runs without a dataset.

## 1. Read a config

Every component is written as `{type: Name, ...arguments}`. The toy detector of the
[example project](https://github.com/Daeda1used/GraspPanda/tree/main/projects/example_project),
shortened:

```yaml
_base_:
  - _base_/default_runtime.yaml
  - _base_/schedules/adam_10e.yaml       # Adam, lr 1e-3, x0.95 per epoch, 10 epochs

model:
  type: ToyDetector
  backbone: {type: ToyBackbone, out_channels: 64}
  head:     {type: ToyHead, num_angle: 12, num_depth: 4}
```

`type` names a registered class of the slot's registry: `model` in `DETECTORS`,
`model.backbone` in `BACKBONES`, `optimizer` in `OPTIMIZERS` (every `torch.optim`
optimizer is there). `grasppanda components` lists every registered class with its
contract; the [Registered classes](../components.md) page shows the same list.

??? note "The top-level keys"

    The top level is a closed set. An unknown key is an error.

    | Key | Content |
    |---|---|
    | `_base_` | Parent files to inherit from. |
    | `custom_imports` | Out-of-tree modules to import before validation. |
    | `model` | The detector. |
    | `data` | `train`, `val`, `test`: each `{dataset, batch_size, num_workers, shuffle, drop_last}`. |
    | `optimizer`, `scheduler` | Optimisation. |
    | `train` | `epochs`, `amp`, `accumulate`, `grad_clip`, `grad_clip_mode`, `ema`, `sync_bn`, `scheduler_unit`. |
    | `evaluation` | `interval`, `evaluator`, `variants`, `save_best`, `split`, `weights`, `num_workers`. |
    | `hooks` | Extra training hooks. |
    | `runtime` | `work_dir`, `run_id`, `device`, `seed`, `deterministic`, `resume`, `allow_config_mismatch`, `load_from`, `allow_weight_mismatch`, `checkpoint`, `log`, `ops`. |

    What each key does during a run: [Train and test](train_test.md).

## 2. Start from a shipped config {#inheritance}

Name a shipped config in `_base_` and change what you need:

```yaml
_base_: [economicgrasp/economicgrasp_sunet14_g1b-realsense_10e.yaml]

train:
  epochs: 20        # overrides the schedule of the method
```

An entry of `_base_` is a path relative to the file. When no such file exists, it is the
name of a shipped config, with or without `.yaml`: `_base_/default_runtime.yaml` for a
building block, `<method>/<config>` for a whole method. This works from any directory and
from an installed wheel. A base that is neither is an error, reported at the file that
names it (`set in: <that file>`).

### Merge rules {#merge-rules}

1. **At most two levels.** A config may inherit from a file that itself inherits from
   `_base_/` building blocks. A third level is an error.
2. **Sibling bases must not disagree.** If two files of one `_base_` list set the same
   value, loading fails and names both files. There is no "last one wins".
3. **Mappings merge deeply, unless `type` changes.** A child that sets another `type`
   *replaces* the inherited node, and the dropped keys are printed:

    ```yaml
    # base:   optimizer: {type: Adam, lr: 1e-3, betas: [0.9, 0.99]}
    optimizer: {type: SGD, momentum: 0.9}
    # result: optimizer: {type: SGD, momentum: 0.9}     (lr and betas are dropped)
    ```

4. **`_delete_: true`** replaces a mapping that has no `type`.
5. **Lists replace wholesale.**

## 3. Change a value from the command line

```bash
grasppanda print-config <config> --opts train.epochs=20 optimizer.lr=3e-4 hooks.0.interval=5
```

Values are read as YAML, so `true`, `null`, `1e-3` and `[1, 2]` mean what they mean in a
file. Overrides go through the same validation as the file, and overriding `type`
follows merge rule 3.

## 4. Check it

```bash
grasppanda print-config projects/example_project/configs/toy_train.yaml --check
```

Validation runs before any dataset is opened or any module is built, and reports every
problem at once. Add `--for train` to also require what `train` needs, or `--no-check` to
print a broken config while you debug it.

??? note "Expected output"

    ```text
    ✓ projects/example_project/configs/toy_train.yaml is valid
      inherits    toy_detector.yaml, default_runtime.yaml, adam_10e.yaml
      components  evaluation.variants.centre.0: ToyCentreFilter · evaluation.evaluator: ToyEvaluator · optimizer: Adam · scheduler: ExponentialLR · model: ToyDetector · model.backbone: ToyBackbone · model.grouper: ToyGrouper · model.head: ToyHead · data.train.dataset: ToyScenes · data.val.dataset: ToyScenes · data.test.dataset: ToyScenes
      hash        be81c4b1301a
    ```

    The hash identifies the experiment. It ignores `runtime.work_dir`, `runtime.run_id`,
    `runtime.resume` and `runtime.allow_config_mismatch`, so moving, naming or resuming a
    run does not change it.

??? note "Troubleshooting: what a refused config looks like"

    Three layers check a config: the schema (top-level and fixed-structure keys), the
    signature of every component (`{type: X}` against `X.__init__`) and the contracts
    between components. A misspelt key and a misspelt argument, both given with `--opts`:

    ```text
    ✗ 2 config problems in projects/example_project/configs/toy_train.yaml

      runtim
        unknown top-level key 'runtim'. Did you mean 'runtime'? Valid keys: custom_imports, model, data, optimizer, scheduler, train, evaluation, hooks, runtime.
        set in: <command line>

      model.head.num_angles
        ToyHead has no parameter 'num_angles'. Did you mean 'num_angle'? Parameters: num_angle, num_depth, depth_step, max_width, width_weight, in_channels.
        set in: <command line>
    ```

    A contract message names both sides, for example `ToyDepthGrouper emits features with
    axes ('seed', 'depth', 'channel'), but ToyHead expects ('seed', 'channel').`

??? note "Interpolation: `${path.to.value}`"

    `${path.to.value}` refers to another value. It is resolved after all merging and all
    overrides, so changing the source changes every reference:

    ```yaml
    runtime:
      work_dir: work_dirs/${config_name}/seed${runtime.seed}
    ```

    `${config_name}` is the file name of the config without its extension.

??? note "Out-of-tree components: `custom_imports`"

    ```yaml
    custom_imports:
      - my_package.my_heads          # an importable module
      - ../my_components.py          # or a file, relative to this config
    ```

    A `.py` path is loaded as a standalone module, so a single-file plugin next to its
    config works without installation. See [Add a component](../advanced_guides/add_component.md).

??? note "What a run records"

    `config.resolved.yaml` (everything above, expanded, defaults written out), the original
    files and overrides, the git commit and diff, a code archive and the environment. The
    run directory is described in [Train and test](train_test.md#the-run-directory).

??? note "Notes on YAML"

    - `lr: 1e-3` is a number here. (Plain PyYAML reads it as a string.)
    - Duplicate keys in one mapping are an error rather than "last one wins".

## Next

- [Datasets](datasets.md): put GraspNet-1Billion on disk and build what your method reads.
- [Train and test](train_test.md): run a config.
- [Compose and probe](combinations.md): let `grasppanda compose` write a config with another
  component in one slot.
