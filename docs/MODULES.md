# Compose modules

Module replacement is an explicit contract, not a shape-only switch. Supported choices preserve coordinates, units, sample indices, label assignment and decoder semantics. The rest of each method remains authoritative.

## Component selection and parameters

A component can be a name (`backbone: pointnet`) or a mapping containing `type` and its parameters. Both forms serialize into the experiment and checkpoint. Unknown parameters are rejected.

| Method | Slot | Choices |
|---|---|---|
| Baseline / PointNet2 port | `backbone` | `upstream`, `pointnet`, `pointnext`, `pointmlp` |
| Baseline / PointNet2 port | `crop` | `upstream`, `multiscale`, `cylinder` |
| Graspness | `backbone` | `upstream`, `pointnet`, `sparse_unet18` |
| Graspness | `crop` | `upstream`, `cylinder`, `finegrasp` |

The baseline encoder returns original-input seed indices and 256-channel features. Graspness encoders retain sparse coordinate correspondence and 512-channel features. Crop adapters retain the native decoder's depth/view semantics.

| Component | Parameters |
|---|---|
| Baseline `pointnet` | `local_channels`, `global_channels`, `fusion_channels` (layer widths); `activation`: relu/gelu/silu; `normalization`: batch/group/none; `dropout` |
| `pointnext` | `width`, `blocks` (five stage depths), `nsample`, `radius` (metres), `radius_scaling`, `expansion`, `activation`, `reduction`: max/mean/sum, `decoder_layers` |
| `pointmlp` | `embed_dim`, `dim_expansion`, `pre_blocks`, `pos_blocks`, `stage_points`, `k_neighbors`, `decoder_channels`, `decoder_blocks`, `res_expansion`, `activation`, `normalize`: anchor/center |
| `multiscale` | `radius_factors` relative to the method's native cylinder radius |
| `finegrasp` | `nsample`, `radius_factors`; native local attention and Transformer fusion across radius groups |
| `cylinder` | `hidden_channels`, `radius_factors`, `nsample`, `pooling`: max/mean/attention, `activation`, `normalization` |

`pointnext` uses the [official PointNeXt encoder and decoder](https://github.com/guochengqian/PointNeXt) ([NeurIPS 2022 paper](https://arxiv.org/pdf/2206.04670)), followed by a feature projection and original-input seed sampling. Its OpenPoints operator has a separate compiled namespace to coexist with legacy grasp operators. Architecture settings do not automatically provide pretrained weights.

`pointmlp` adapts the [official PointMLP segmentation blocks](https://github.com/ma-xu/pointMLP-pytorch/blob/main/part_segmentation/model/pointMLP.py) ([ICLR 2022 paper](https://openreview.net/pdf?id=3Pbra-_u76D)): geometric affine normalization, residual MLP extraction and native feature propagation. An XYZ embedding replaces the normals-dependent input; the part-category token and segmentation classifier are omitted. Dense features are projected to 256 channels and sampled at original-input seed indices. This is a grasp backbone adaptation, without pretrained grasp weights. Its native propagation fusion retains ReLU; `activation` controls embedding and residual blocks.

PointMLP stage parameters are four-element lists. `stage_points` defaults to `[1024, 256, 64, 16]` and must decrease; each `k_neighbors` value must fit that stage's input. Fixed stage sizes bound neighborhood memory as the input cloud grows. `dim_expansion` multiplies the preceding channel width; `decoder_channels` specifies output widths from coarse to fine. Train any changed architecture before evaluating it.

`finegrasp` uses the author's [CylinderGroup and GroupTransformerFusion](https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/robo_orchard_lab/models/finegrasp) from [FineGrasp (2025)](https://arxiv.org/pdf/2507.05978). This replacement retains Graspness seed features and decoder; it does not replace the complete method with FineGrasp. Start with `radius_factors: [0.25, 0.5, 0.75, 1.0]` and `nsample: 16`, then train the new grouping layers.

`pointnet` is a toolbox shared-MLP/global-pooling adapter. `cylinder` exposes native oriented queries with configurable feature extraction and pooling; it is not a complete reproduction of a named attention or scale-balanced method. `sparse_unet18` uses MinkUNet18D. These choices are available only in their registered semantic slots.

```yaml
modules:
  backbone:
    type: pointnext
    width: 32
    blocks: [1, 2, 2, 2, 1]
    radius: 0.05
    nsample: 32
  crop:
    type: cylinder
    hidden_channels: [64, 128]
    radius_factors: [0.5, 1.0, 1.5]
    pooling: attention
checkpoint_policy: reuse_unchanged
```

In the UI, select the component names under **Compose modules**, then enter parameters keyed by slot in **Component parameters by slot**. Do not repeat `type` in this parameter editor; the selector supplies it. Full YAML/JSON configurations use the mapping form above.

## Training controls

Baseline, its PointNet2 port and Graspness accept `loss` and `augmentation` overrides in supported short-training or epoch-training actions. Other methods reject these controls until their supervision adapters implement them.

```yaml
loss:
  weights:
    width: 0.4
augmentation:
  mode: custom
  rotation_axis: x
  rotation_degrees: 15
  translation: 0.01
  jitter_std: 0.001
  jitter_clip: 0.003
```

Loss weights are absolute coefficients. Unspecified components retain the native coefficients and all masks/reductions remain native. Baseline defaults: objectness/view = 1; score/angle/width/tolerance = 0.2. Graspness defaults: objectness = 1, graspness = 10, view = 100, score = 15, width = 10. Identical defaults retain the original loss tensor.

An empty augmentation mapping preserves the action's original behavior. `mode: none` disables augmentation; `mode: native` selects the author's transform. `mode: custom` applies a camera-axis rotation sampled uniformly within the specified degrees and an independent translation sampled within the specified metre range. Point coordinates and object poses transform together; sparse coordinates are recomputed. Clipped Gaussian jitter affects observations while clean grasp supervision is retained. Image crops, scaling and scene mixing require different label/intrinsic contracts and are not implied by these point-only controls.

Loss/augmentation settings are training-only. Resume requires the same settings; checkpoint inference retains architecture settings and clears training-only options automatically.

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

`initialize` loads model weights with the selected transfer policy and starts a fresh optimizer at epoch 0. `resume` requires `strict`, restores the optimizer and epoch, and requires the same composition and data settings. Restored model and optimizer tensors are checked before the next update. Native CUDA point operators use atomic gradient accumulation, so later loss trajectories are not guaranteed to be bitwise identical. Set `epochs` to the final epoch number, greater than the saved epoch. SBG's OneCycle schedule requires the original final-epoch horizon when resuming; initialize a new run to change that horizon.

A nonzero training batch limit selects consecutive frames from `scene`/`frame` before native shuffling and augmentation. The native validation loop uses a prefix of test_seen for Baseline/SBG; Graspness has no validation loop in its released trainer. These validation losses are diagnostics, not benchmark AP or a recommended model-selection protocol. Set both limits to **0** for complete splits and increase `timeout_minutes` for a long run. Epoch-boundary seeds are controlled by the configured seed plus epoch.

Each run saves native epoch checkpoints under `training/` and the last checkpoint as `checkpoint.pt`. The UI exposes initialization/resume, batch limits, worker count, losses and checkpoint inference. Bounded epoch execution and resume are checked; full-split convergence is not claimed.

## Add a component

Register a slot/choice in `grasppanda/components.py`, implement the adapter under `grasppanda/modules/`, and preserve its documented semantic contract. Test indices and gradient flow, strict checkpoint transfer, real-label optimization, checkpoint reload and downstream decoding. Document the input contract and available operation in the method table. Additional dataset integration also requires readers, supervision and an evaluator; metadata registration alone is insufficient.
