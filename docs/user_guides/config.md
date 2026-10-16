# Configuration

A GraspPanda config is a YAML file. It describes the whole experiment: model, data,
optimisation, evaluation and runtime. No behaviour is reachable only through a
command-line flag.

## Components

Every component is written as `{type: Name, ...arguments}`:

```yaml
model:
  type: ToyDetector
  backbone: {type: ToyBackbone, out_channels: 64}
  head:     {type: ToyHead, num_angle: 12, num_depth: 4}
optimizer: {type: Adam, lr: 1e-3}
scheduler: {type: ExponentialLR, gamma: 0.95}
```

`type` names a class in the registry that owns that slot: `model` is resolved in
`DETECTORS`, `model.backbone` in `BACKBONES`, `optimizer` in `OPTIMIZERS` (every
`torch.optim` optimizer is available), and so on. List what is registered with:

```bash
grasppanda print-config --list backbones
grasppanda print-config <config> --list all      # also shows the config's custom imports
```

`grasppanda components` shows the same registries as a catalogue: each component with
its contract, status, source and the shipped configs that use it, and
`grasppanda components <name>` its options and what it can be combined with. The
[Components](../components.md) page is that catalogue; which component can replace
which is in [Swapping components](../advanced_guides/swapping.md).

## Top-level keys

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

## Inheritance

```yaml
_base_:
  - ../_base_/default_runtime.yaml
  - ../_base_/schedules/gsnet_10e.yaml

train:
  epochs: 20        # overrides the schedule
```

An entry of `_base_` is a path relative to the file. When there is no such file, it is
the name of a shipped config, with or without `.yaml`: `_base_/default_runtime.yaml` for
a building block, `gsnet/gsnet_sunet14_g1b-realsense_10e` for a whole method. This works
from any directory and from an installed wheel, so a config of your own can start from a
shipped one:

```yaml
_base_: [gsnet/gsnet_sunet14_g1b-realsense_10e.yaml]   # uses both levels (rule 1)

train:
  epochs: 20
```

A file next to the config comes first. A base that is neither is an error reported at the
file that names it (`set in: <that file>`).

### Merge rules

1. **At most two levels.** A config may inherit from a file that itself inherits from
   `_base_/` building blocks. A third level is an error.
2. **Sibling bases must not disagree.** If two files of one `_base_` list set the same
   value, loading fails and names both files. There is no "last one wins" to remember.
3. **Mappings merge deeply — unless `type` changes.** When a child sets a different
   `type`, the inherited node is *replaced*:

    ```yaml
    # base:   optimizer: {type: Adam, lr: 1e-3, betas: [0.9, 0.99]}
    optimizer: {type: SGD, momentum: 0.9}
    # result: optimizer: {type: SGD, momentum: 0.9}     (lr and betas are dropped)
    ```

    Arguments of the old component can never leak into the new one. The dropped keys
    are printed.
4. **`_delete_: true`** replaces a mapping that has no `type`.
5. **Lists replace wholesale.**

## Interpolation

`${path.to.value}` refers to another value. It is resolved after all merging and all
overrides, so changing the source changes every reference:

```yaml
runtime:
  work_dir: work_dirs/${config_name}/seed${runtime.seed}
```

`${config_name}` is the file name of the config without its extension.

## Command-line overrides

```bash
grasppanda print-config <config> --opts train.epochs=20 optimizer.lr=3e-4 hooks.0.interval=5
```

- Values are parsed as YAML, so `true`, `null`, `1e-3` and `[1, 2]` mean what they mean
  in a file.
- Overrides go through exactly the same validation as file contents.
- Overriding `type` follows merge rule 3.

## Validation

Validation runs before any dataset is opened or any module is built, and reports every
problem at once.

| Layer | Checks | Example message |
|---|---|---|
| Schema | Top-level and fixed-structure keys and their types. | `unknown top-level key 'runtim'. Did you mean 'runtime'?` |
| Component signatures | Each `{type: X}` node against `X.__init__`: the name exists in the right registry, every key is a parameter, required parameters are present, value types match. | `ToyHead has no parameter 'num_angles'. Did you mean 'num_angle'?` |
| Cross-field rules | Contracts between components and between sections. | `ToyDepthGrouper emits features with axes ('seed', 'depth', 'channel'), but ToyHead expects ('seed', 'channel').` |

```bash
grasppanda print-config <config> --check                # validate, print a summary
grasppanda print-config <config> --for train --check    # also require what `train` needs
grasppanda print-config <config> --no-check             # print a broken config to debug it
```

## Out-of-tree components

```yaml
custom_imports:
  - my_package.my_heads          # an importable module
  - ../my_components.py          # or a file, relative to this config
```

A `.py` path is loaded as a standalone module, so a single-file plugin next to its
config works without installation. See [Add a component](../advanced_guides/add_component.md).

## What a run records

`config.resolved.yaml` (everything above, expanded, defaults written out), the original
files and overrides, the git commit and diff, a code archive and the environment. The
config hash printed by `--check` identifies the experiment; it ignores `runtime.work_dir`,
`runtime.run_id`, `runtime.resume` and `runtime.allow_config_mismatch`, so that moving,
naming or resuming a run does not change its identity. What each key does is described in
[Training and testing](train_test.md).

## Notes on YAML

- `lr: 1e-3` is a number here. (Plain PyYAML reads it as a string.)
- Duplicate keys in one mapping are an error rather than "last one wins".
