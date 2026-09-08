# Compose modules

Module replacement is an explicit contract, not a shape-only switch. Supported choices preserve coordinates, units, sample indices, label assignment and decoder semantics. The rest of each method remains authoritative.

## Supported slots

| Method | Slot | Choices | Contract |
|---|---|---|---|
| GraspNet Baseline / PointNet2 port | `backbone` | `upstream`, `pointnet` | Camera XYZ in metres → 1,024 seed coordinates, original point indices and 256-channel features |
| GraspNet Baseline / PointNet2 port | `crop` | `upstream`, `multiscale` | Oriented neighborhoods → 256-channel features with the original four depth bins |
| Graspness | `backbone` | `upstream`, `pointnet`, `sparse_unet18` | Sparse features → 512 channels with unchanged coordinate mapping and voxel-to-point correspondence |

`pointnet` is a GraspPanda PointNet-style shared-MLP/global-pooling encoder. It does not claim checkpoint equivalence to the original PointNet paper. The baseline upstream encoder is PointNet++; returning to `upstream` restores it. The Graspness PointNet-style adapter operates on voxel coordinates and features. `sparse_unet18` uses the pinned implementation's MinkUNet18D. `multiscale` combines three native cylindrical crops at radius factors 0.5, 1 and 1.5 with a learned projection; it is not the entire SBG method.

HGGD, RNG and other methods currently retain their native components. Their image proposals, local refinement targets and decoding contracts cannot be replaced by a Graspness slot solely because tensor dimensions match.

## Checkpoint policies

- `strict`: every checkpoint key and parameter shape must match. Use this for original models and a saved checkpoint of the same composition.
- `reuse_unchanged`: discard checkpoint parameters only inside explicitly replaced modules, retain their constructor initialization, and load all remaining parameters strictly. The result records exactly which keys were initialized or discarded.

Selecting a different encoder and silently accepting all missing keys would conceal implementation mistakes. GraspPanda rejects mismatches outside the chosen slots.

## Run a bounded training check

Download the baseline weights and prepare its training labels as described in [downloads](DOWNLOADS.md). Copy this configuration to `compose.local.yaml` and set your dataset path:

```yaml
dataset: graspnet1b
method: graspnet_baseline
action: train_check
dataset_root: /data/GraspNet-1B
checkpoint: checkpoints/graspnet_baseline/checkpoint-rs.tar
camera: realsense
split: train
scene: 0
frame: 0
num_points: 4096
workspace: official_gt_workspace
modules:
  backbone: pointnet
  crop: multiscale
checkpoint_policy: reuse_unchanged
training_steps: 3
learning_rate: 0.0001
```

```bash
./panda run compose.local.yaml --runs-dir outputs/cli-runs
```

The check repeats one labelled frame without augmentation, computes the native loss, and requires finite losses/gradients and nonzero parameter updates. It also verifies updates in every replaced component. This is a bounded optimization diagnostic, not a multi-epoch training schedule or accuracy result. HGGD, GraNet and fusion use batch size 2; other point methods use batch size 1. RNG uses anchor batch 2 and up to 48 local patches. CenterGrasp checks its SGDF and RGB objectives separately. The general `batch_size` and `epochs` fields apply to the full native `train` action, not this diagnostic.

The output directory contains `checkpoint.pt`, `result.json`, the configuration, provenance and logs. Results include loss components, input-label hashes, transfer details and updates. The UI plots total loss and can export the run.

## Use the saved model

Choose a completed training check in **Runs & results**, click **Prepare inference from checkpoint**, review the generated JSON in **Experiments**, then click **Run edited JSON**. This retains the same module choices and switches to `strict` checkpoint loading for a test frame.

For CLI use, change `action` to `infer`, `checkpoint` to the saved file, `checkpoint_policy` to `strict`, `split` to `test_seen` and `scene` to `100`. Keep `modules` unchanged. Short training does not establish quality, especially when an encoder was newly initialized; an empty graspable-point set is reported explicitly.

## Train a composed model across epochs

Baseline and Graspness accept these same module choices in `action: train`. SBG also exposes its native epoch trainer. The original loaders, augmentation, loss, optimizer and learning-rate schedule remain in use. Object/collision labels load through bounded caches instead of eagerly occupying memory for every scene.

To test the epoch workflow, change the example above:

```yaml
action: train
train_checkpoint_mode: initialize
epochs: 1
batch_size: 2
train_batch_limit: 3
eval_batch_limit: 1
data_workers: 0
```

`initialize` loads model weights with the selected transfer policy and starts a fresh optimizer at epoch 0. `resume` requires `strict`, restores the optimizer and epoch, and requires the same composition and data settings. Set `epochs` to the final epoch number, greater than the saved epoch. SBG's OneCycle schedule requires the original final-epoch horizon when resuming; initialize a new run to change that horizon.

A nonzero training batch limit selects consecutive frames from `scene`/`frame` before native shuffling and augmentation. The native validation loop uses a prefix of test_seen for Baseline/SBG; Graspness has no validation loop in its released trainer. These validation losses are diagnostics, not benchmark AP or a recommended model-selection protocol. Set both limits to **0** for complete splits and increase `timeout_minutes` for a long run. Epoch-boundary seeds are controlled by the configured seed plus epoch.

Each run saves native epoch checkpoints under `training/` and the last checkpoint as `checkpoint.pt`. The UI exposes initialization/resume, batch limits, worker count, losses and checkpoint inference. Bounded epoch execution and resume are checked; full-split convergence is not claimed.

## Add a component

Register a slot/choice in `grasppanda/components.py`, implement the adapter under `grasppanda/modules/`, and preserve its documented semantic contract. Test indices and gradient flow, strict checkpoint transfer, real-label optimization, checkpoint reload and downstream decoding. Document the input contract and available operation in the method table. Additional dataset integration also requires readers, supervision and an evaluator; metadata registration alone is insufficient.
