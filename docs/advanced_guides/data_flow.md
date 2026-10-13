# Data contracts

Stages exchange small typed structures instead of a shared dictionary. Each structure
checks its own invariants, so a mismatch fails where it is introduced, with a message
that names the stage.

## Observation and Supervision

```python
losses = model.loss(observation, supervision)     # training
grasps = model.predict(observation)               # inference: no ground truth in reach
```

- [`Observation`](../api/structures.md) holds what a model may see: depth, RGB,
  intrinsics, points, per-point features, and masks.
- `Supervision` holds labels and is only ever passed to `loss`.
- A mask carries its `Provenance` (`sensor`, `calibration`, `predicted`,
  `ground_truth`). `observation.gt_assisted()` lists inputs derived from ground truth,
  and results are stamped with that list. The workspace crop that most GraspNet-family
  pipelines apply at test time is derived from the ground-truth segmentation; here that
  is visible instead of implicit.
- The dense grasp labels of an object (`ObjectGraspLabels`) hold their score and width
  grids either as tensors or as a reader that leaves them in the label store. Code that
  consumes them calls `gather_grids` and gets the same values either way; see
  [Data loading](data_loading.md).

## Points and features

| Structure | Invariant |
|---|---|
| `PointSet` | `(N, 3)` coordinates with batch offsets, a reference frame and an **identity**. A sub-sampled set remembers its parent. |
| `PointFeatures` | `(N, C)` features **bound to a `PointSet`**. `feats.expect(points, user=...)` raises if they belong to a different set. |

This is what stops per-point features of a sub-sampled cloud from being indexed as if
they lived on the full cloud, and what stops a voxelisation's inverse map from being
reused with another point set.

## Stage outputs

| Structure | Produced by | Notes |
|---|---|---|
| `SeedSet` | proposal | Seeds, optional features, **named** scores (`graspness`, `objectness`, …) and a validity mask. No score is mandatory. |
| `ApproachSet` | view stage | One approach direction per seed, optionally a crop frame and a template index. |
| `GroupFeatures` | grouper | Features with **named axes**, e.g. `("seed", "channel")` or `("seed", "depth", "channel")`. A head states the axes it expects. |
| `GraspGroup` | head bundle | The single output type of every detector. |

## GraspGroup

Geometry follows GraspNet-1Billion; all lengths are metres.

| Field | Meaning |
|---|---|
| `translation` | Grasp point. |
| `rotation` | Proper rotation. Column 0 approach axis, column 1 closing axis, column 2 finger-height axis. |
| `width` | Jaw opening. |
| `depth` | How far the fingers reach past the grasp point along the approach axis. Not a camera depth. |
| `score` | Ranking score, higher is better; `score_kind` says what it means. |
| `frame`, `gripper`, `key` | Reference frame, gripper geometry, sample identity. |

```python
grasps.validate()                       # finite, orthonormal, det +1, widths ≥ 0
grasps.sort_by_score()                  # stable: ties keep their order on every platform
grasps.to_graspnet_array()              # (G, 17), only at the I/O boundary
```

GraspNet label scores are friction coefficients: lower is better and −1 means invalid.
They are labels, not confidences, and exist only inside `Supervision`.

## Operators

`grasppanda.ops` holds pure-PyTorch reference implementations that run on CPU and define
correct behaviour: furthest point sampling, ball and cylinder query, kNN, grasp NMS,
voxelisation, view templates. Their conventions (first sample is index 0, first hits in
index order, strict thresholds, 0-based indices) are documented in the
[API reference](../api/ops.md) and pinned by tests against brute-force oracles.
