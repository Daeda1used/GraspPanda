# Candidate graph experiments

GtG2 generates grasp candidates with GPG, constructs a local graph for each candidate and averages independently trained graph regressors. Its graph slots are separate from scene-point encoders and cylinder decoders. See [Methods & papers](METHODS.md) for the paper, original implementation and availability.

## Prepare, train and predict

Install the shared runtime using [Installation](INSTALL.md). Download the selected GraspNet training scenes and object models; `dex_models` avoids rebuilding the object geometry cache. Dataset links are in [Data & weights](DOWNLOADS.md). This workflow creates candidate labels from official collision and force-closure primitives; it does not require Graspness maps.

```bash
./panda init --example train-gtg2 -o gtg2.local.yaml
# Set dataset_root and label_root in gtg2.local.yaml.
./panda prepare-gtg2 --config gtg2.local.yaml
./panda run gtg2.local.yaml
```

Preparation inherits the configuration's camera, frame, seed, training scenes and crop geometry. Explicit preparation CLI arguments override those values; keep any geometry or seed override in the training configuration too. `--scoring-batch` controls temporary scoring memory without reducing the candidate set. Preparation can take substantial time and storage; completed frames are reused on subsequent invocations.

For an initial bounded experiment, set `trainer.scenes: [0, 1]`, `trainer.folds: [0, 1]`, `modules.crop.candidate_limit: 256`, `epochs: 2`, `batch_size: 16`, `train_batch_limit: 1` and `eval_batch_limit: 1` **before preparation**. Each model trains on one scene and validates on the other. This is a functional experiment, not the full training protocol. Set candidate and batch limits to `0` for complete data, and omit `trainer.scenes` to select scenes 0–99.

In the browser, select **GtG2 reconstruction → Load preset**, choose the prepared graph root under **Training & evaluation settings**, and configure the graph under **Compose modules**. Select explicit `gtg_sage` / `gtg_gatv2` and `grasp_graph` types before entering component parameters. **Method training stages** controls scene folds and sampling. Preparation currently uses the CLI command above.

After training, choose **Prepare inference from checkpoint** in **Runs & results** and run the generated JSON. It keeps the graph configuration and selects every trained ensemble member's best validation checkpoint. For CLI use, copy the training configuration, set `action: infer`, `split: test_seen`, `scene: 100`, the chosen `frame`, and `checkpoint` to the generated `checkpoint.pt`; remove `trainer`, `loss`, `augmentation`, `optimizer` and `scheduler` overrides.

Inference permits a different `candidate_limit` and `gpg_threads`; other graph and encoder settings must match the checkpoint. Set `candidate_limit: 0` to score all eligible candidates even when the checkpoint used bounded training preparation. `batch_size` controls graph inference batches. Predictions use the standard GraspNet format and can enter the shared [official evaluation workflow](USAGE.md#experiment-artifacts).

## Configure the graph

| Slot / setting | Choices and defaults |
|---|---|
| `backbone: gtg_sage` | `hidden: 64`, `layers: 3`, `aggregation: max` (`mean` / `sum` supported) |
| `backbone: gtg_gatv2` | `hidden: 64`, `layers: 3`, `heads: 4`, `dropout: 0`, `add_self_loops: true`, `share_weights: false`; hidden width must divide evenly across heads |
| `crop: grasp_graph` | `k: 5`, `max_points: 70` per region, `include_outside: true`, `encoding: binary` (XYZ + one flag); `onehot` uses XYZ + two type flags |
| Eligibility | `min_inside_train: 50`, `min_inside_infer: 70`; minimum counts and point cap must exceed `k` |
| Local geometry | `bound_size: 0.03`, `gripper_depth: 0.06`, `gripper_height: 0.03`, `voxel_size: 0.005`, `plane_threshold: 0.002`; distances in metres |
| Candidate generation | `dual_cloud: true`, `gpg_samples: 1000000`, `gpg_threads: 4`, `max_width: 0.1`, `candidate_limit: 0` |
| Candidate NMS | `nms_translation: 0.005` metres, `nms_rotation_degrees: 1`; either threshold at zero disables suppression |
| Training pose variants | `train_depths: [0.03, 0.04, 0.05]`, `train_width_offsets: [0.02, 0.04]`; zero width offset is always included |

The global scene-point `num_points` field is unused; retain its default. The graph's `crop.voxel_size` governs candidate preprocessing, while global `voxel_size` controls optional final collision filtering. `collision_thresh: 0` disables that final filter. All candidate generation uses `official_gt_workspace`, including dataset segmentation and camera poses.

Raw inside/outside points, candidate identities and scores are stored as numeric arrays. Cache keys include input content hashes, preparation geometry, seed and source versions. Changes to graph width, depth, feature encoding, `k`, sampling cap or outside-feature use reuse the same raw cache. Geometry changes require preparation again. Modified or corrupted cached arrays are rejected.

GATv2 uses the [PyG 2.6.1 implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html) of [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491), ICLR 2022. It is an experimental encoder replacement requiring training.

## Objectives, augmentation and resume

`loss.functions.score` supports `upstream` / `mse`, `l1`, `smooth_l1`, `huber` and `charbonnier`, with the shared [regression parameters](MODULES.md#loss-formulations). `loss.weights.score` defaults to `1` and must remain positive. Targets are `1.2 - friction` for valid grasps, `-1` for collisions and `-0.5` for other invalid candidates.

Empty augmentation settings preserve the released model's batch-wide random half-turn about the gripper approach axis. `mode: none` disables it. Custom augmentation runs before graph construction, preserves the quality target and never changes validation inputs:

```yaml
augmentation:
  mode: custom
  half_turn_probability: 0.5
  point_dropout: 0.1
loss:
  functions:
    score: {type: huber, delta: 0.2}
optimizer: {type: adamw, weight_decay: 0.01}
scheduler: {type: cosine, min_lr_ratio: 0.1}
```

The default optimizer is Adam with the configuration's learning rate. AdamW, SGD and Lion, plus constant/cosine/multistep schedules, use the shared controls. Schedules count updates separately for each ensemble member. A singleton final training batch joins the preceding batch so batch normalization remains valid without dropping a graph.

`trainer.folds` selects distinct residues 0–9. A fold holds out scenes where `scene % 10 == fold`. Defaults select five folds and all 100 training scenes. `sampling: balanced` keeps all nonnegative samples and samples up to `negative_per_class` from each negative class every `refresh_epochs`; `sampling: all` uses every prepared training graph. Validation uses all eligible held-out graphs unless explicitly batch-limited.

Epoch checkpoints include every member's model, best validation state, optimizer, schedule and random states. Set `train_checkpoint_mode: resume` and select a checkpoint under `training/`. Keep data, graph, fold, loss, augmentation and optimization settings unchanged. With a configured schedule, retain the original final epoch horizon. A checkpoint saved before validation resumes that pending validation before further training. CUDA runs are not promised to reproduce identical subsequent parameter trajectories.

## Reconstruction choices

The adapter retains the released GNN's position encoder and transformation block. It resolves inconsistent graph/model feature dimensions with explicit four-feature binary encoding, includes the surrounding points, corrects edge conversion and aligns each regression prediction with one scalar target. Missing preparation/inference helpers are reconstructed around the available geometry code and official evaluator primitives.

The five-member, 500-epoch preset and refreshed negative sampling follow the paper's described protocol; they are not implemented by the released short training script. Raw and inpainted proposals use one declared geometry configuration, frame-specific camera poses and widths capped at 0.1 m. Deterministic score ties replace the source's zero-threshold quadratic NMS. These choices and configurable alternatives constitute a toolbox reconstruction; they do not establish the paper's reported accuracy.
