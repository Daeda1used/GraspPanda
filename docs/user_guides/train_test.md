# Training and testing

`grasppanda train`, `grasppanda test` and `grasppanda infer` take a config and nothing
else. Every setting of a run lives in the config; the few flags the commands accept are
shorthand for overrides and end up in the run snapshot like any other override.

!!! info "What you can run today"
    The engine is implemented and tested on synthetic data. The dataset readers, the
    detectors and the benchmark evaluator arrive with their own milestones, so the
    examples below use the [example project](https://github.com/Daeda1used/GraspPanda/tree/main/projects/example_project):
    a toy detector, a synthetic dataset and a toy evaluator that exercise every code
    path on a CPU in seconds. No benchmark number has been produced with this engine yet.

## Try it

```bash
CFG=projects/example_project/configs/toy_train.yaml

grasppanda train $CFG --opts runtime.run_id=demo
grasppanda test  $CFG --weights work_dirs/toy_train/demo/checkpoints/best.pth
grasppanda infer $CFG --weights work_dirs/toy_train/demo/checkpoints/best.pth \
                      --input val --output out/toy_val
```

```text
ʕ•ᴥ•ʔ  GraspPanda 0.1.0  ·  6-DoF grasp detection in clutter
  train
  config      projects/example_project/configs/toy_train.yaml  (hash 08b7a4c78cd8)
  run         work_dirs/toy_train/demo
  device      cpu · 1 process · seed 0 · deterministic off
  model       ToyDetector · 7,601 parameters (7,601 trainable)
  data        32 frames · 8 batches of 4 per epoch
  schedule    5 epochs · Adam(lr=0.005, weight_decay=0.0) · ExponentialLR(gamma=0.95) per epoch
  evaluation  data.val every 1 epoch(s) · variants raw, centre · best by raw/AP
  hooks       EvaluationHook, CheckpointHook, TextLoggerHook
epoch 1/5  iter 4/8  loss 3.9746  loss/score 3.8631  loss/width 0.1115  cell_accuracy 0.0410  lr 5.000e-03  data 0.001s  iter 0.012s  eta 0:00:00
...
evaluation after epoch 1 (8 frames, 0.1 s)
protocol toy-synthetic-v0 · dataset toy · split val · cameras toy · ground-truth-assisted inputs: none
variant  frames  AP     AP_tight
───────  ──────  ─────  ────────
raw      8/8     1.31   0.68
centre   8/8     15.46  7.75
best checkpoint: raw/AP = 1.3054 at epoch 1
```

The three commands exit with `0` on success, `2` when the config is invalid (nothing was
started) and `1` when the run failed, with a one-paragraph reason on stderr.

## The run directory

Every invocation works in `<runtime.work_dir>/<run id>/`. The run id is
`runtime.run_id`, or the start time (`20261012-101500`, with `-test` or `-infer`
appended for those commands) when it is not set.

```text
work_dirs/toy_train/demo/
├── config.resolved.yaml      the config with every default written out
├── config.sources/           the files it was merged from, and overrides.txt
├── run.json  git.json  env.json  code.tar.gz  diff.patch
├── log.txt
├── checkpoints/              epoch_0001.pth … and best.pth
├── predictions/<variant>/    scene_0100/<camera>/0000.npy, GraspNet (G, 17) arrays
├── results/                  <variant>.json (test) · epoch_0005/<variant>.json (training)
└── resumes/<time>/           the snapshot of each later invocation that resumed this run
```

One directory is one run. A fresh `train`, and every `test` and `infer`, refuses to
write into a directory that already holds a run. `run.json` also records the protocol
id and the ground-truth-assisted inputs once an evaluation has run.

## What the config controls

| Key | Meaning |
|---|---|
| `data.train` / `val` / `test` | `{dataset, batch_size, num_workers, shuffle, drop_last}`. `batch_size` is per process. Training shuffles unless `shuffle: false`; prediction never shuffles. |
| `optimizer`, `scheduler` | Any `torch.optim` optimizer and scheduler, `{type: ..., ...}`. |
| `train.epochs` | Length of the run. |
| `train.amp` | Mixed precision for the training forward pass. Prediction always runs in full precision. |
| `train.accumulate` | Batches whose gradients are summed before one optimizer step. |
| `train.grad_clip`, `train.grad_clip_mode` | Limit and `norm` (global L2 norm) or `value` (element-wise clamp). |
| `train.ema` | Decay of the moving average of the weights; `null` turns it off. |
| `train.sync_bn` | Synchronised batch norm across processes (CUDA only). |
| `train.scheduler_unit` | `epoch` (default) or `iteration`: when the scheduler steps. |
| `evaluation.interval` | Evaluate on `data.val` every N epochs and after the last one; `0` turns it off. |
| `evaluation.evaluator` | The scorer. |
| `evaluation.variants` | Named post-processing chains, each scored separately. |
| `evaluation.save_best` | `<variant>/<metric>` that selects `best.pth`; higher is better. |
| `evaluation.split` | The `data` section `test` and `infer` read (default `test`). |
| `evaluation.weights` | `model` or `ema`: which weights are scored. |
| `evaluation.num_workers` | Worker processes of the evaluator. |
| `runtime.work_dir`, `runtime.run_id` | Where the run lives. Neither enters the config hash. |
| `runtime.device` | `auto`, `cpu`, `cuda`, `cuda:N`. |
| `runtime.seed`, `runtime.deterministic` | See [Reproducibility](#reproducibility). |
| `runtime.resume`, `runtime.allow_config_mismatch` | See [Resuming](#resuming). |
| `runtime.load_from`, `runtime.allow_weight_mismatch` | See [Loading weights](#loading-weights). |
| `runtime.checkpoint` | `{interval, keep_last}`. |
| `runtime.log` | `{interval, backends}` with backends `text`, `tensorboard`, `wandb`. |
| `hooks` | Extra hooks; see [Hooks](#hooks). |

## The training loop

The loop is one method,
[`Trainer.train`](https://github.com/Daeda1used/GraspPanda/blob/main/grasppanda/engine/trainer.py),
36 lines long. Per batch:

1. `losses = model.loss(observation, supervision)`, under autocast when `train.amp` is on.
2. Entries whose key starts with `loss/` are summed. Everything else is a metric: it is
   logged, never summed. If the sum is NaN or infinite the run stops and names the term,
   the iteration and the frames of the batch:

    ```text
    ✗ NonFiniteLossError: non-finite loss at epoch 1, iteration 3: loss/aux = nan.
      Frames in the batch: tiny/scene_0000/cam/0002, tiny/scene_0000/cam/0005
    ```

3. `total / train.accumulate` is back-propagated.
4. Every `train.accumulate` batches, and on the last batch of an epoch, the gradients
   are clipped and applied, the weight average is updated, and the scheduler steps when
   `train.scheduler_unit` is `iteration`. A tail group smaller than `accumulate` therefore
   takes a smaller step; with a scheduler per iteration, "iteration" means optimizer step.

After the last batch the scheduler steps when its unit is `epoch`.

### Schedulers that need the length of the run

`OneCycleLR` (`total_steps`), `CosineAnnealingLR` (`T_max`) and `PolynomialLR`
(`total_iters`) are given the number of scheduler steps of the run, so the config does
not repeat what `train.epochs` and the dataset size already say:

```yaml
scheduler: {type: OneCycleLR, max_lr: 1e-3}     # no total_steps
train: {epochs: 10, scheduler_unit: iteration}
```

A value written in the config is kept. `ReduceLROnPlateau` and schedulers that take
Python functions or other schedulers are rejected with an explanation.

### Parameter groups and several optimizers

By default one optimizer covers every trainable parameter. A detector that needs more
defines `configure_optimizers`, which receives the `optimizer` node of the config:

```python
def configure_optimizers(self, optimizer: dict):
    return [
        {"params": self.backbone.parameters(), "lr": optimizer["lr"] * 0.1},
        {"params": self.head.parameters()},
    ]
    # or several optimizers: {"backbone": self.backbone.parameters(), "head": ...}
```

Return parameter groups for one optimizer, or a mapping `name -> parameters | optimizer`
for several. Every optimizer steps on the same total loss and gets its own scheduler.

## Hooks

Whatever is not the arithmetic of a step is a hook. Three are added by default from the
config: a logger per `runtime.log.backends` entry, in-training evaluation (when
`data.val`, `evaluation.evaluator` and `evaluation.interval > 0` are set) and
checkpointing. Others are listed under `hooks`:

```yaml
hooks:
  - {type: BNMomentumHook, initial: 0.5, decay: 0.5, step: 2, minimum: 0.001}
  - {type: StageTimerHook, sync: true}
```

| Hook | Does |
|---|---|
| `TextLoggerHook` | One line every `runtime.log.interval` iterations and on the last batch of an epoch, to the console and `log.txt`: running means of every loss term and metric, learning rate, gradient norm (when clipping by norm), data and iteration time, peak GPU memory, ETA. |
| `TensorBoardLoggerHook`, `WandbLoggerHook` | The same scalars to TensorBoard (`<run>/tensorboard`) or Weights & Biases. Optional: `pip install "grasppanda[tensorboard]"` or `"grasppanda[wandb]"`. A missing package fails before training starts. |
| `EvaluationHook` | Scores `data.val`; see [Evaluation](#evaluation). |
| `CheckpointHook` | `epoch_NNNN.pth` every `interval` epochs and after the last one, the newest `keep_last` kept, `best.pth` by `evaluation.save_best`. |
| `BNMomentumHook` | Sets the momentum of every batch-norm layer to `max(initial * decay ** (epoch // step), minimum)` before each epoch. |
| `StageTimerHook` | Seconds per iteration spent in the forward pass of each stage (child module) of the detector, as `forward.<stage>` in the log. With `sync: true` it waits for the GPU around every stage, which makes the numbers honest and the training slower. |

A hook listed under `hooks` replaces the default of the same class. Hooks run in
`priority` order (evaluation 70, checkpointing 80, loggers 90, everything else 50), so a
checkpoint always knows the metrics of its epoch. To write one, subclass
`grasppanda.engine.hooks.Hook`, annotate the constructor parameters, register it with
`@HOOKS.register()` and load the module through `custom_imports`:

```python
@HOOKS.register()
class FreezeBackbone(Hook):
    def __init__(self, epochs: int = 1) -> None:
        self.epochs = epochs

    def before_epoch(self, state) -> None:          # state.epoch: index of the epoch to run
        state.model.backbone.requires_grad_(state.epoch >= self.epochs)
```

`state` is a `TrainState`: model, optimizers, schedulers, epoch and iteration counters,
the outputs of the last iteration, the latest evaluation metrics, the run directory.

## Checkpoints

A checkpoint holds the model, optimizer(s), scheduler(s), gradient scaler, weight
average, the random-number states of every process, epoch, iteration, best metric,
config hash and package version. It is written to a temporary file and renamed, so an
interrupted job never leaves a truncated file, and it contains only tensors and plain
values: files are read with `weights_only=True` and loading never executes pickled code.

### Resuming

```bash
grasppanda train <config> --resume              # runtime.resume=true: newest checkpoint in the work dir
grasppanda train <config> --resume path/to/epoch_0004.pth
```

The run continues in the directory that owns the checkpoint (a checkpoint file that
lies outside a run directory starts a new run). A checkpoint written under a different
config hash is refused:

```text
✗ ResumeError: …/epoch_0001.pth was written under config hash f77ba11a0ed6, but the current
  config hashes to 590caa0a40d9: this is a different experiment. Resume with the original
  config (config.resolved.yaml in the run directory), or set
  runtime.allow_config_mismatch=true to continue anyway.
```

`runtime.work_dir`, `runtime.run_id`, `runtime.resume` and
`runtime.allow_config_mismatch` are not part of the hash. Anything else is, including
`train.epochs`: training longer than planned needs `runtime.allow_config_mismatch=true`.

### Loading weights

`runtime.load_from` (`--weights` for `test` and `infer`) loads weights and nothing else.
The key report is always printed, and any difference is an error unless
`runtime.allow_weight_mismatch: true`; there is no silent non-strict load.

```text
✗ WeightMismatchError: weights from w.pth: 1 matched, 0 missing, 5 unexpected, 3 with a different shape
  unexpected (in the file, not in the model):
    encoder.1.weight
    encoder.1.running_mean
    ...
  different shape:
    encoder.0.weight: file (8, 3), model (16, 3)
    head.weight: file (1, 8), model (1, 16)
    ...
Nothing was loaded. Fix the model config or the file, or set
runtime.allow_weight_mismatch=true to load the keys that match.
```

Accepted layouts: a GraspPanda checkpoint, a bare state dict, or a mapping with a
`state_dict` / `model_state_dict` entry; a uniform `module.` prefix is stripped. Each of
these is stated in the report. Files that pickle other objects must be converted first.

## Evaluation

In-training evaluation and `grasppanda test` run the same routine:

1. `model.predict` runs over the split. The raw predictions are written once, to
   `predictions/raw/`. An empty prediction is a valid result and is written too.
2. For every entry of `evaluation.variants` with a non-empty chain, the chain is applied
   to each frame's raw grasps, with the observation of that frame, and the result is
   written to `predictions/<variant>/`. A variant with an empty chain *is* the raw
   predictions. The name `raw` is reserved for it.
3. The evaluator scores each variant against **every** key of the split, so a frame
   without a prediction is reported as missing rather than skipped.
4. One result record per variant is written, and metrics are reported as
   `<variant>/<metric>`.

```yaml
evaluation:
  evaluator: {type: ToyEvaluator}
  variants:
    raw: []
    centre: [{type: ToyCentreFilter, radius: 0.15}]
  save_best: raw/AP
```

### Result records

`results/<variant>.json` holds what was measured and under which conditions: `protocol`,
`dataset`, `split`, `cameras`, `requested` / `scored` / `coverage` / `missing`,
`postprocessing` (the chain with every parameter), `gt_assisted`, `evaluator` (class,
config and the info it reported), `config_hash`, `code` (commit, dirty flag, version),
`weights` (path, sha256, `model` or `ema`, epoch, the config hash they were trained
under), `seed`, `metrics` and `per_scene`. Per-scene numbers are the mean of the
evaluator's per-frame precision tables, overall and per column, as fractions in `[0, 1]`.

`gt_assisted` lists the inference inputs that were derived from ground truth, collected
from `observation.gt_assisted()` over all frames.

### Partial results

When not every requested frame was scored, the record is still written, but with
`"partial": true`, `"metrics": null` and the aggregates under `"partial_metrics"`, so
nothing that reads `metrics` can mistake them for a headline number. The console prints
coverage and per-scene numbers only:

```text
variant  frames  AP
───────  ──────  ──────────────────
raw      6/6     1.00
best1    5/6     withheld (partial)
best1: PARTIAL, 5 of 6 frames scored (83.3 %), 1 missing. Per-scene mean precision only:
  cam/scene_0020  frames 5  0.5000
```

A partial variant contributes no `<variant>/<metric>` and never becomes the best
checkpoint.

### `grasppanda test`

```bash
grasppanda test <config> --weights <checkpoint> [--split val] [--work-dir DIR] [--opts ...]
```

`--weights` is `runtime.load_from`, `--split` is `evaluation.split` (a `data` section:
`train`, `val` or `test`). Which frames a section holds is the dataset's business:
`--opts data.test.dataset.<parameter>=...`. To score the averaged weights of a run
trained with `train.ema`, set `evaluation.weights=ema`.

### `grasppanda infer`

```bash
grasppanda infer <config> --weights <checkpoint> --input <split|dir> --output <dir>
```

Predicts and writes, scores nothing and needs no evaluator. `--output DIR` makes `DIR`
the run directory (`runtime.work_dir` and `runtime.run_id`), so the grasps land in
`DIR/predictions/<variant>/` next to the snapshot that says how they were made.
`--input` is a `data` section, or a directory that is handed to the reader of
`data.<evaluation.split>` as its `root` (`data.<split>.dataset.root=DIR`).

!!! warning "Limits of `infer` today"
    Grasps are written as GraspNet `(G, 17)` arrays; a native `.npz` format does not
    exist yet. Reading a plain folder of RGB-D frames needs a dataset reader with a
    `root` parameter, and none is shipped yet.

## Distributed training

```bash
torchrun --nproc_per_node=4 -m grasppanda.cli.main train <config>
torchrun --nproc_per_node=4 -m grasppanda.cli.main test  <config> --weights <checkpoint>
```

Nothing in the config says "distributed": the processes read their rank from the
environment `torchrun` sets. Then

- every process builds the model from the same seed and draws its own random numbers
  afterwards (`runtime.seed + rank`);
- the training set is one common shuffle dealt out to the processes; `batch_size` is per
  process;
- `train.sync_bn: true` converts batch norm to its synchronised form (on CUDA; with
  several CPU processes it is an error, because PyTorch has no such layer for the CPU);
- for evaluation each process predicts a disjoint share of the frames (no padding), the
  keys are merged and de-duplicated by `SampleKey`, and the main process scores;
- only the main process writes files and prints; a checkpoint stores the random state
  of every process.

!!! warning "What has and has not been exercised"
    Tested: two CPU processes with the gloo backend, in the unit suite (training,
    in-training evaluation, `test`, identical weights on both processes) and, by hand,
    an exact resume. Also by hand: the wrapper, synchronised batch norm and mixed
    precision on one GPU in a one-process NCCL group. **Not tested: several GPUs.** The
    development machine has one. While the main process scores an evaluation the others
    wait in a collective; its timeout is raised to 24 hours for that reason, which has
    not been observed under NCCL. `DistributedDataParallel` runs with its defaults: a
    detector with parameters that receive no gradient in some iterations needs
    `find_unused_parameters`, which is not exposed yet.

## Reproducibility

- `runtime.seed` seeds Python, NumPy and PyTorch. The order of an epoch depends on the
  seed and the epoch only, and the seeds of data-loader workers on seed, rank and epoch,
  so neither depends on how many epochs ran in this process.
- Prediction runs under its own seed (`runtime.seed + rank`) and puts the random state
  back afterwards. Evaluating during training therefore does not change what is
  trained, and the same weights give the same predictions in training and in
  `grasppanda test` (same seed, batch size and number of processes), even when the data
  pipeline or the detector draws random numbers at inference time.
- `runtime.deterministic: true` selects deterministic kernels where PyTorch has them and
  warns where it has none. It costs speed.
- With `deterministic: true`, training N epochs and training N/2, stopping and resuming
  give bit-identical weights. This is a unit test on CPU (with dropout, augmentation
  noise, batch norm, Adam, a one-cycle schedule, the weight average, clipping and
  accumulation all on, with and without worker processes) and was checked once by hand
  on a GPU with mixed precision. It is a property of this engine with those toy
  components; a detector that uses an operator without a deterministic implementation
  will not have it.

## From Python

```python
from grasppanda.config import Config
from grasppanda.engine import commands

config = Config.load("projects/example_project/configs/toy_train.yaml", ["train.epochs=2"],
                     command="train")
state = commands.train(config)            # state.run.dir, state.model, state.best
```

`commands.test(config)` returns an `Evaluation` (`.records`, `.metrics`, `.partial`) and
`commands.infer(config)` a `Dump` (`.root`, `.keys`, `.gt_assisted`). To validate a config
without running it: `grasppanda print-config <config> --for train --check`.
