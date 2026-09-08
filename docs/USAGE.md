# Using GraspPanda

## Browser

Run `./panda ui` and open **http://127.0.0.1:7860**.

1. In **Experiments**, select a protocol and method, then **Load preset**.
2. Set the dataset root containing `scenes/`. **Download registered weights** prepares each required network for the selected camera and verifies its checksum.
3. Select an operation, adjust its inputs and click **Run current form**.
4. In **Runs & results**, select the job to view logs, predictions, loss curves and checkpoints. Cancel it or export the experiment as a ZIP.

Expand **Method details & input requirements** for the selected method's input protocol and original implementation. **Compose modules** exposes registered replacements. **Configuration editor** lets you generate, edit, validate and run exact JSON. For a native recipe, the preset defines fixed inputs; disabled frame fields do not override them. Supported recipes accept a primary checkpoint override.

The **Guide** tab includes installation, downloads, module instructions and a data/GPU readiness check. The **Methods & papers** tab contains the complete method table.

## Operations

| UI operation | Configuration `action` | Inputs and behavior |
|---|---|---|
| Predict grasps | `infer` | Selected frames and a matching checkpoint; saves predictions and a preview. |
| Run native recipe | `pipeline_smoke` | Method-specific fixed scene, sequence, author sample or component. Its input contract is shown on selection. |
| Short training run | `train_check` | Repeats a labelled sample for `training_steps`; saves loss curves and a checkpoint. Fixed native batch sizes apply. |
| Single training step | `train_smoke` | One native labelled optimizer step. |
| Train across epochs | `train` | Native loader, augmentation, optimizer and schedule; supports initialization or checkpoint resume. |
| Evaluate predictions | `evaluate` | Complete split predictions with a matching manifest; runs the official evaluator. |
| Check model environment | `probe` | Fixed import or synthetic model diagnostic; no dataset required. |

The available choices depend on the method. The internal action identifiers are retained for configuration compatibility. See [Methods & papers](METHODS.md) for scope and [Compose modules](MODULES.md) for training settings.

## CLI

```bash
./panda list
./panda doctor
./panda weights hggd --camera realsense
./panda verify hggd --dataset-root /data/GraspNet-1B
./panda verify asgrasp
./panda run GraspNet-1B/examples/infer-graspness.example.yaml
```

`verify` executes the method preset. `run` accepts YAML or JSON; edit the example paths before running. Relative data/checkpoint paths resolve from the repository root. `./panda fetch` restores missing pinned source checkouts without changing existing checkouts or the version lock.

Set a default dataset path for the UI and CLI presets:

```bash
export GRASPPANDA_DATASET_ROOT=/data/GraspNet-1B
./panda ui
```

## Train, resume and reuse

Prepare the method's labels using [Data & weights](DOWNLOADS.md). For native epoch training, select **Train across epochs** and expand **Training & evaluation settings**. Set both batch limits to `0` for complete splits. Increase `timeout_minutes` in the configuration editor for long runs.

`initialize` loads model weights and starts a fresh optimizer. `resume` restores the model, optimizer and epoch with strict loading; keep the same component/data settings and set `epochs` above the saved epoch. SBG also restores its OneCycle schedule, which requires the original final-epoch horizon.

To use a completed run's `checkpoint.pt`, click **Prepare inference from checkpoint** in **Runs & results**, open **Configuration editor**, review the generated configuration and **Run edited JSON**. Frame adapters retain the module choices. Native recipes retain their fixed input protocol. CenterGrasp's RGB checkpoint stays paired with the SGDF model that provided its embedding targets.

## Outputs and evaluation

### Configuration sweeps

In **Configuration editor**, generate or edit the base experiment, expand **Configuration sweep**, and enter a parameter grid. **Preview sweep** shows every exact configuration without running. **Run sweep** validates all combinations and input paths before queueing them together in the shared runtime.

```json
{"seed": [0, 1], "modules.backbone.embed_dim": [32, 64]}
```

This example creates four PointMLP experiments when the base selects `backbone: pointmlp`. Use dotted configuration paths; lists such as layer widths must be nested inside the list of candidate values. To compare different architectures with different parameters, vary the entire `modules.backbone` mapping. Unsupported method/parameter combinations and duplicate configurations are rejected. A grid is limited to 128 experiments.

```bash
./panda sweep GraspNet-1B/examples/sweep-baseline.yaml --preview
./panda sweep GraspNet-1B/examples/sweep-baseline.yaml
```

Edit the dataset/checkpoint paths first. Sweep YAML contains `base` and `grid`; its Cartesian product uses sorted parameter names and the supplied value order. Jobs run sequentially, each with its own seed, exact configuration and provenance. Group manifests are generated in `outputs/runs/sweeps/`; each run also carries `sweep.json`. Inspect or cancel individual jobs in **Runs & results** and compare their settings and results in **Compare**. Ctrl+C cancels the CLI sweep's remaining jobs.

### Experiment artifacts

Each experiment creates its own directory under `outputs/runs/`, containing configuration, logs, provenance, results and available predictions/checkpoints. These files are generated on your machine. The UI and CLI share a persistent queue and run one job at a time; a CLI client connects automatically to a running UI worker.

Closing the browser leaves the job running. Stopping the server cancels the active process group. After a restart, abandoned jobs are marked interrupted; they do not silently resume. Use a separate `--runs-dir` for an independent CLI queue, with your own GPU allocation.

For official AP, predict every frame of one test split with the same method, camera and preprocessing. Set `action: evaluate` and `prediction_dir` to that complete prediction directory, retaining its configuration and checkpoint. Evaluation rejects incomplete or changed files. AP is `null` until evaluation completes. Compare AP only across matching observation, supervision, split and postprocessing protocols.

`official_gt_workspace` uses dataset segmentation and poses; `depth_only` uses a different protocol. HGGD/RNG use `native_demo`. Fused scenes, sequences and RGB/stereo samples require their own presets.

## Remote access and troubleshooting

The server binds to localhost. Use SSH forwarding on a remote machine:

```bash
ssh -L 7860:localhost:7860 USER@SERVER
```

Optional `GRASPPANDA_USER` and `GRASPPANDA_PASSWORD` enable authentication. Keep this local research workbench behind trusted access.

| Problem | Next step |
|---|---|
| Missing data or weights | Check the reported path, camera and method-specific preprocessing. |
| CUDA / import failure | Run `./panda doctor`, then follow [Installation](INSTALL.md). |
| Checkpoint mismatch | Match the architecture; use strict loading or the documented component transfer policy. |
| No grasp candidates | Inspect depth units, valid points, camera and checkpoint; candidate count is not accuracy. |
| Slow first invocation | Allow time for native kernel compilation and preprocessing. |
| Failed job | Inspect its log, correct the input and submit a new run. |
