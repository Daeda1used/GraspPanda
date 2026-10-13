# Post-processing: what happens to predicted grasps before a robot or an evaluator gets them

*Words such as seed, contract, preset, adapter, label set and point painting are explained in the [slot index](index.md#words-used-on-these-pages).*

## 1. The task of the slot

A detector returns up to a thousand grasps per frame, ranked by a score. A robot wants a
handful that its own gripper can execute, that do not hit the scene or the table, that come
from a direction it can reach, on the object it was asked for, in its own frame. A
post-processor is one such step: it removes grasps, re-orders them, or corrects a pose or a
width. Steps are chained; a chain is called a *variant*.

It is the last stage of the pipeline: `detector.predict(observation)` → **variant chain** →
`GraspGroup` written to disk, scored, drawn or sent to the robot. Every method, HGGD
included, emits the same `GraspGroup`, so this is the one slot where everything is
interchangeable between all six methods.

No post-processor has weights and none is trained. No AP is claimed for any of them: the
benchmark evaluators reward none of the execution filters below, and a variant that removes
grasps is always reported beside `raw`, never instead of it.

## 2. Input and output

```python
class PostProcessor:
    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup: ...
```

| | |
|---|---|
| `grasps` | The grasps of **one** frame. `translation (G, 3)` grasp point, metres, in `grasps.frame` (`camera`: x right, y down, z forward). `rotation (G, 3, 3)`: column 0 approach axis, column 1 closing axis, column 2 finger height. `width (G,)` opening, `depth (G,)` how far the fingers reach past the grasp point along the approach, `score (G,)` higher is better (its scale depends on the method: `score_kind`), `gripper` the geometry the numbers refer to, `key` the frame. `G` may be 0. |
| `observation` | The batch the grasps were predicted from; a step picks its frame by `grasps.key`. `points`: the cloud the network saw. `full_points`: the cloud before sub-sampling (only when the dataset section has `keep_full_points: true`; the GraspNet-family configs do, HGGD's does not). `masks`: named boolean masks over `points`; no shipped dataset fills them. `rgb`, `depth`, `intrinsics` when the method reads images. Never ground truth. |
| returns | A `GraspGroup` of the same frame: fewer grasps, other order, or changed `translation` / `width` / `depth` / `frame`. |

Who consumes it: `grasppanda test` (each variant of `evaluation.variants` is dumped and
scored separately, with the number of grasps after every step in the result record),
`grasppanda infer`, `grasppanda demo`, and `GraspInferencer` in your code.

What the neighbours require:

* A step that reads a cloud requires the grasps in the cloud's frame. A step that changes the
  frame (`ToFrame`) must therefore be last; anything after it that reads the cloud raises
  `ContractError: grasps are in frame 'base' but the scene cloud is in frame 'camera'`.
* Steps that change a pose or a width (`GripperCentering`, `WidthFromCloud`) go **before** a
  collision filter, selection steps (`TopK`, `FurthestGraspSampling`) go **after** the filters.
* The evaluators apply their own NMS and top-50, so a selection step changes what a robot
  receives, not a score.

Contract attributes of a post-processor class: `status`, `source`, `extras` (optional
libraries), and `primary` (the option `name=value` sets on a command line). The config
checker validates the options of every node of `evaluation.variants.*` against the
constructor (names and types); what a step needs from the observation is checked when it
runs, with a `ContractError` that names the missing input and how to supply it.

## 3. Options at a glance

"Name" is what you write after `--post` or `--set post=`; the component type is what you
write as `type:` in a config. All are universal (every method) unless the last column says
otherwise. "Probe" is the verdict of the measured records
([measured combinations](../advanced_guides/measured_combinations.md), group `post`)
under GSNet / GraspNet-baseline / HGGD. Weights: none anywhere (nothing is trained).

| Name | Component | What it does | Idea from (sources not re-opened for this page; no code was copied) | Code | Extra | Probe | Reads beyond the grasps |
|---|---|---|---|---|---|---|---|
| `top-k` | `TopK` | keep the `k` best | AnyGrasp (Fang et al., arXiv 2212.08333, §IV-D: top-100), GPD `num_selected` | written here | — | runs / runs / runs | nothing |
| `score-threshold` | `ScoreThreshold` | keep scores above an absolute, quantile or relative limit | 6-DOF GraspNet `choose_fn` (Mousavian et al., arXiv 1905.10520) | written here | — | runs / runs / runs | nothing |
| `furthest-sampling` | `FurthestGraspSampling` | spread: furthest point sampling over grasp points among the best candidates | Contact-GraspNet `select_grasps` (Sundermeyer et al., arXiv 2103.14127) | written here on `grasppanda.ops.furthest_point_sample` | — | runs / runs / runs | nothing |
| `nms` | `GraspNMS` | one grasp per 3 cm and 30° | graspnetAPI rule | existed | — | runs / runs / runs | nothing |
| `workspace` | `WorkspaceFilter` | grasp point inside a box, in the camera frame or (with `transform`) the robot's | GPD `workspace_grasps`, AnyGrasp SDK `lims` | written here | — | runs / runs / runs | nothing |
| `approach-cone` | `ApproachAngleFilter` | approach within a cone about the table normal, the camera axis or a vector | GPD `filter_approach_direction`, AnyGrasp SDK `approach_steering`, ThinkGrasp's gravity cone | written here | — | runs / runs / runs | a cloud, for `table_normal` (the plane is fitted) |
| `table-plane` | `TablePlaneFilter` | no gripper corner below the table plane | ZeroGrasp's table test (Iwase et al., arXiv 2504.10857, CC BY-NC code: not copied) | written here, plane fitted by RANSAC | — | runs / runs / runs | a cloud (or a given plane) |
| `target-mask` | `MaskFilter` | keep grasps near a masked point (a target object) | Contact-GraspNet `filter_grasps`, AnyGrasp SDK `apply_object_mask` | written here | — | not probed: no dataset fills `masks`; run with a stand-in mask on real frames, section 4 | `observation.masks[<name>]`: **needs your segmentation** |
| `gripper-limits` | `GripperLimits` | opening (and depth) within what a real gripper does; remove or clip | AnyGrasp SDK `max_gripper_width`, GPD apertures | written here | — | runs / runs / runs | nothing |
| `center-gripper` | `GripperCentering` | shift along the closing axis to the middle of the contacts | AnyGrasp §IV-D | written here | — | runs / runs / runs | a cloud |
| `width-from-cloud` | `WidthFromCloud` | opening = extent between the fingers + clearance | ZeroGrasp's width refinement, single-view form | written here | — | runs / runs / runs | a cloud |
| `collision` | `ModelFreeCollisionFilter` | four gripper boxes against the voxelised **full** cloud (GSNet test script) | GraspNet-1Billion collision detector | existed | — | runs / runs / runs | `full_points`: HGGD's dataset sections drop that cloud, so `compose` switches `keep_full_points` on for `data.val` and `data.test` and says so |
| `collision-strict` | `ModelFreeCollisionFilter` | HGGD's parameters: one point rejects, on the network's cloud | HGGD test script | existed | — | runs / runs / runs | `points` |
| `mesh-collision` | `GripperMeshCollisionFilter` | scene points inside a gripper **mesh** (STL/OBJ/PLY of your gripper, or built-in boxes) | GraspGen's cloud-against-gripper-mesh check (Murali et al., arXiv 2507.13097), as containment | **route L**: [trimesh](https://github.com/mikedh/trimesh) (MIT) loads, splits, hulls; placement and containment in PyTorch | `trimesh` | runs / runs / runs | a cloud |
| `to-base-frame` | `ToFrame` | grasps into the robot base frame; optionally the grasp point at the fingertips | — | written here on `GraspGroup.transform` | — | runs / runs / runs | nothing; must be last |

Fifteen names, thirteen component types (`collision` and `collision-strict` are two parameter
sets of one class, as are any `name=value` variants you write).

### Measured with released weights on two real frames

Every preset alone on the raw predictions of GraspNet-1Billion frames
`scene_0100/realsense/0000` and `scene_0130/realsense/0128`; GSNet with a converted
checkpoint, HGGD with the authors' RealSense checkpoint (both local, non-commercial, not
committed). "in → out" per frame; time is the faster of the two frames on a shared GPU, after
a warm-up call. This is evidence that a step runs and what it does to real predictions, not a
benchmark result.

| Name | GSNet: in → out | ms | HGGD: in → out | ms |
|---|---|---|---|---|
| `top-k` | 1024 → 50; 1024 → 50 | 0.2 | 447 → 50; 412 → 50 | 0.2 |
| `score-threshold` | 1024 → 512; 1024 → 512 | 0.6 | 447 → 224; 412 → 206 | 0.6 |
| `furthest-sampling` | 1024 → 50; 1024 → 50 | 0.4 | 447 → 50; 412 → 50 | 0.4 |
| `nms` | 1024 → 129; 1024 → 158 | 41.1 | 447 → 137; 412 → 137 | 42.8 |
| `workspace` | 1024 → 1024; 1024 → 1024 | 0.4 | 447 → 447; 412 → 412 | 0.4 |
| `approach-cone` | 1024 → 913; 1024 → 707 | 6.6 | 447 → 398; 412 → 277 | 20.3 |
| `table-plane` | 1024 → 737; 1024 → 633 | 6.8 | 447 → 159; 412 → 222 | 3.6 |
| `target-mask` (stand-in mask: points within 6 cm of the best grasp) | 1024 → 181; 1024 → 68 | 1.7 | 447 → 90; 412 → 42 | 1.1 |
| `gripper-limits` | 1024 → 389; 1024 → 453 | 0.4 | 447 → 287; 412 → 291 | 3.7 |
| `center-gripper` | 1024 kept; 1023 and 998 moved, median 6.7 and 5.6 mm | 12.4 | 447 and 412 kept; 422 and 377 moved, median 2.6 and 6.3 mm | 3.3 |
| `width-from-cloud` | 1024 kept; 201 and 127 narrowed, median by 7.9 and 14.4 mm | 9.2 | 447 and 412 kept; 129 and 69 narrowed, median by 12.4 and 17.1 mm | 3.0 |
| `collision` | 1024 → 816; 1024 → 565 | 13.7 | fails with the message of section 5: no full cloud | — |
| `collision-strict` | 1024 → 826; 1024 → 617 | 9.6 | 447 → 212; 412 → 125 | 10.5 |
| `mesh-collision` | 1024 → 605; 1024 → 465 | 12.4 | 447 → 156; 412 → 138 | 37.3 |
| `to-base-frame` | 1024 → 1024; 1024 → 1024 | 0.2 | 447 → 447; 412 → 412 | 0.2 |

## 4. How to use each option

### Three ways to add steps

**On a command line**, with weights you already have. Steps run left to right; `name=value`
sets the main option, `name:option=value,option=value` any option (values in YAML):

```bash
grasppanda demo gsnet/gsnet_sunet14_g1b-realsense_10e --weights gsnet.pth --depth depth.png \
    --intrinsics 927.17,927.37,651.32,349.62 --depth-scale 1000 \
    --post center-gripper width-from-cloud collision table-plane approach-cone=45 \
           gripper-limits=0.08 nms furthest-sampling=20

grasppanda test gsnet/gsnet_sunet14_g1b-realsense_10e --weights gsnet.pth --post collision nms top-k=50
grasppanda infer hggd/hggd_r34w8_g1b-realsense_15e --weights hggd.pth --post collision-strict nms top-k=20
```

`test` and `infer` add the chain as the variant `post`, scored or written beside the
variants of the config (`raw`, `cd`). `demo` applies it after `--variant` /
`--collision-filter`, if given, and prints what every step left. Real output (GSNet,
converted checkpoint, a GraspNet-1Billion frame):

```text
  grasps    1024 predicted in 3.49 s (first call, includes warm-up); after GripperCentering 1024 →
  WidthFromCloud 1024 → ModelFreeCollisionFilter 872 → GripperMeshCollisionFilter 675 →
  TablePlaneFilter 613 → ApproachAngleFilter 585 → GripperLimits 364 → GraspNMS 51 →
  FurthestGraspSampling 20 → ToFrame 20
```

and the same chain with `collision-strict` and `mesh-collision:cloud=points` under HGGD:

```text
  grasps    447 predicted in 7.21 s (first call, includes warm-up); after GripperCentering 447 →
  WidthFromCloud 447 → ModelFreeCollisionFilter 247 → GripperMeshCollisionFilter 128 →
  TablePlaneFilter 109 → ApproachAngleFilter 106 → GripperLimits 106 → GraspNMS 28 →
  FurthestGraspSampling 20 → ToFrame 20
```

**As a composed config** (works for every detector, HGGD too):

```bash
grasppanda compose --list post                       # the presets, one line each
grasppanda compose --show post=approach-cone         # one preset with its notes
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set post=collision+nms+top-k=20 -o my.yaml
grasppanda probe my.yaml                             # builds, trains 20 iterations, predicts, runs the chain
```

The probe prints one `post` line per predicted frame, the number of grasps predicted and
what every step left (here from `hggd/hggd_r34w8_g1b-realsense_15e` with the same chain):

```text
  predicted   2 frame(s) · 384, 353 grasps
  post        384 → ModelFreeCollisionFilter 115 → GraspNMS 48 → TopK 20
  post        353 → ModelFreeCollisionFilter 131 → GraspNMS 94 → TopK 20
```

```text
what compose did
  • post: evaluation variant `post` = ModelFreeCollisionFilter → GraspNMS → TopK, applied in
    this order to the raw predictions of every frame and scored beside the variants of the
    base config (`grasppanda demo --variant post` applies it to one frame).
config check
  ✓ valid  (hash 6f7add9f2286)
```

**In a config**, as any other variant (name it as you like; `raw` must stay empty):

```yaml
evaluation:
  variants:
    robot:
      - {type: GripperCentering, max_shift: 0.02}
      - {type: WidthFromCloud, clearance: 0.01, mode: shrink}
      - {type: ModelFreeCollisionFilter, voxel_size: 0.01, collision_threshold: 0.01}
      - {type: TablePlaneFilter, clearance: 0.005}
      - {type: ApproachAngleFilter, direction: table_normal, max_angle: 60}
      - {type: GripperLimits, max_width: 0.08}
      - {type: GraspNMS}
      - {type: TopK, k: 20}
      - {type: ToFrame, transform: calibration/T_base_camera.txt, frame: base, origin: fingertips}
```

**In Python:**

```python
from grasppanda.apis import GraspInferencer
from grasppanda.structures import Mask, Provenance

inferencer = GraspInferencer("gsnet/gsnet_sunet14_g1b-realsense_10e", weights="gsnet.pth")
grasps = inferencer(depth=depth, intrinsics=K, depth_scale=1000, post="collision+nms+top-k=20")

# a target object: put your segmentation on the cloud, then filter on it
observation = inferencer.observe(depth=depth, intrinsics=K, depth_scale=1000)
observation.masks["target"] = Mask(on_target, Provenance.PREDICTED)   # (N,) bool over observation.points
grasps = inferencer.predict(observation, post="target-mask+collision+top-k=10")
```

### The settings that matter

| Name | Main option (`name=value`) | Choose it like this |
|---|---|---|
| `top-k` | `k` (50) | 50 is what the GraspNet-1Billion evaluator scores; a robot that tries candidates in order needs 10 to 20. After the filters, not before. |
| `score-threshold` | `threshold` (0.5, `mode: quantile`) | `quantile` and `relative` need no knowledge of the score scale; `mode=absolute` is tuned per method (GSNet's score is unbounded, EconomicGrasp's is a sum of logits, HGGD's is in [0, 1]). `min_keep=1` keeps the best grasp whatever its score. |
| `furthest-sampling` | `num` (50) | `candidates` (300) is how far down the ranking the spread may reach; `null` lets a poor but remote grasp in. |
| `nms` | none: `nms:translation_threshold=0.05` | 0.03 m and 30° are the benchmark rule; both must be undercut to suppress. |
| `workspace` | `z` ([0.2, 1.0]) | `workspace:x=[-0.3,0.3],y=[-0.2,0.2],z=[0.3,0.8]`. With `transform` (4×4 `T_base_camera`, nested lists or a `.npy`/text file) the box is in the robot base frame; the grasps stay in the camera frame. |
| `approach-cone` | `max_angle` (60°) | Top-down picking: 30 to 45. `direction=camera` for a camera that looks straight down; `direction=[x,y,z]` for a shelf. `min_keep=1` returns the best-aligned grasp when none passes. |
| `table-plane` | `clearance` (0.005 m) | `finger_length` is the benchmark gripper's 0.06 m. Give `plane=[a,b,c,d]` when the table is not the largest plane in view. |
| `target-mask` | `mask` (`target`) | `max_distance`: 0.005 m is exact for seed-based methods (grasp points are cloud points); HGGD computes centres and needs 0.01 to 0.02. `invert=true` avoids an object. |
| `gripper-limits` | `max_width` (0.08 m) | Franka hand 0.08, Robotiq 2F-85 0.085, 2F-140 0.14. `mode=clip` only for a consumer that clips itself. |
| `center-gripper` | `max_shift` (0.02 m) | Before a collision filter. `voxel_size` (0.005) trades exactness for speed. |
| `width-from-cloud` | `clearance` (0.01 m) | `mode=shrink` never widens; `fit` also widens when points lie within `margin` outside the prediction. Before a collision filter. |
| `collision` | none | `voxel_size` 0.01 and `collision_threshold` 0.01 are the GSNet reference; EconomicGrasp's script uses a 0.005 voxel. Needs the full cloud. |
| `collision-strict` | none | One point rejects: strict on a noisy cloud. Reads the network's cloud, so it runs everywhere. |
| `mesh-collision` | `mesh` (none: built-in boxes) | See below. `max_points` 2 or 3 tolerates depth noise. |
| `to-base-frame` | `transform` | **Replace the placeholder.** `origin=fingertips` gives a tool centre point (depth 0); `grasp_point` keeps the GraspNet convention. Last step. |

### A mesh of your own gripper

```bash
pip install -e ".[trimesh]"
```

```yaml
- type: GripperMeshCollisionFilter
  mesh: meshes/my_gripper_collision.stl   # the open gripper, any format trimesh reads
  scale: 0.001                            # the file is in millimetres
  mesh_transform:                         # file coordinates -> tool frame (this example: file z is the approach axis)
    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
  bodies: exact                           # or convex: hull of every body, faster
  max_points: 2
```

The tool frame: origin in the middle between the fingertips, x along the approach (out of the
gripper, into the object), y along the closing direction, z along the finger height, metres.
Points between the fingers are outside the mesh, so the object to be grasped does not count;
a neighbour an open finger would hit does. The mesh is tested as it is (usually fully open),
with no approach sweep: put it after `collision`, not in its place. The unit test writes the
built-in gripper to an STL in millimetres with another axis convention, loads it back with
`scale` and `mesh_transform`, and gets the same collisions in both `bodies` modes.

### Adding your own step

A class in `grasppanda/models/postprocessors/`, registered with `@POSTPROCESSORS.register()`,
with `__call__(grasps, observation) -> GraspGroup`, explicit constructor options, `status`,
`source` and, for the `name=value` shorthand, `primary`. Helpers for the cloud of the frame,
points in gripper frames, a rigid transform from a config and a plane fit are in
`grasppanda/models/postprocessors/scene.py`. Add a preset under `grasppanda/presets/post/`
(a tripwire test asks for one) and it appears in `--list post` and in the probed grid.

## 5. Pitfalls met while building this

* **`collision` needs the cloud before sub-sampling, which HGGD's dataset sections drop.**
  It used to pass the config check and stop when the chain ran. Now `compose --set
  post=collision...` writes `keep_full_points: true` into `data.val` and `data.test` of the
  composed config and says so, and a config written by hand without it is refused by the
  config check, naming the section the command reads: `data.val` for `grasppanda probe` and
  for evaluation during training (`--opts data.val.dataset.keep_full_points=true`),
  `data.test` for `grasppanda test`. To test the cloud the network saw instead, write the
  step as `collision:cloud=points` (`collision-strict` already does). The newer steps
  (`center-gripper`, `width-from-cloud`, `table-plane`, `approach-cone`,
  `mesh-collision`) read the full cloud when it is there and the network's cloud otherwise.
* **A full cloud is half a million points.** Centring 1024 grasps against 519 612 points took
  0.8 s and gigabytes of work memory; both refiners now down-sample to 5 mm voxels first
  (12 ms). `WidthFromCloud` adds half a voxel per side, so it errs wide.
* **`table-plane` removes far more under HGGD (447 → 159) than under GSNet (1024 → 737)** on
  the same frame. Not investigated further: HGGD's cloud is the whole view with the table,
  the GraspNet-family cloud is cropped around the objects, and HGGD's grasps carry one fixed
  depth. Look at the picture of `grasppanda demo` before trusting a clearance.
* **A fitted "table" is the largest plane.** A frame that shows mostly a wall or a box lid
  gets that plane; below `min_inliers` (10 % of the cloud) the step refuses and asks for
  `plane`. The normal is oriented towards the camera, so the camera must be above the table.
* **trimesh's own containment needs `rtree` or `embreex`**, which the `trimesh` wheel does not
  pull in. The filter therefore uses trimesh for loading, body split, watertightness, convex
  hulls and box primitives, and tests containment itself (ray parity, or half-spaces for
  convex bodies).
* **A probe trains a detector for 20 iterations**, so the grasps its post-processing record
  counts are those of an almost untrained network (with 2 iterations GSNet predicts none).
  The probe shows that a step runs on real frames under a method; the table in section 3 was
  measured with released weights.
* **`gripper-limits=0.08` removes more than half of GSNet's grasps** on these frames
  (1024 → 389): the heads predict wide openings. `width-from-cloud` before it narrows the
  ones the cloud shows to be narrower.
* **The demo's picture is in the camera frame.** After a `ToFrame` step the printed table and
  `grasps.npy` are in the new frame; the picture shows the grasps before that step.

## 6. Not wrapped

Sources and licences below were recorded by a survey of the literature and code on
2026-10-12. They were not opened again when this page was written (only trimesh was
installed and run), so check the linked repository before relying on a licence.

| Candidate | Why not |
|---|---|
| GraRe re-ranker ([Minakanmi-Yuki/grare](https://github.com/Minakanmi-Yuki/grare), MIT; weights on the Hub without a licence) | A learned re-ranker per detector and camera that needs RGB, MobileSAM and a Point-MAE checkpoint: three more models and their licences, and no checkpoint for GSNet or FineGrasp. The most promising follow-up; it needs a second score per grasp on `GraspGroup` first. |
| GraspGen-X collision filter and discriminator ([NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX), Apache-2.0 code, NVIDIA Open Model weights) | The collision filter is a padded distance test that needs the target object's points removed per grasp (an instance mask); `GripperMeshCollisionFilter` covers the use without one. The discriminator scores segmented single objects in another gripper convention. |
| ZeroGrasp contact refinement ([sh8/zerograsp](https://github.com/sh8/zerograsp)) | CC BY-NC 4.0 and derived from the graspnet-baseline detector: not copied. Its two ideas that work on a sensor cloud are here as `WidthFromCloud` and `TablePlaneFilter`, written independently. |
| AnyGrasp SDK filters | A licensed binary. Its documented steps (top-k, width limit, approach steering, object mask, centring) are here as generic filters. |
| Contact-GraspNet `select_grasps`, GraspGen, SceneCollisionNet, CabiNet | NVIDIA non-commercial source licences. The selection idea is `FurthestGraspSampling`; the learned collision networks need weights trained on synthetic scenes and a gripper cloud. |
| C-SJO (Generalizing-Grasp) | No licence file; the released script assigns grasps to objects with ground-truth poses. |
| GPD cluster consensus, deepen-until-contact, antipodal check ([atenpas/gpd](https://github.com/atenpas/gpd), BSD-2) | C++ inside a candidate generator, with constants for GPD's handle heuristics; not a library to bridge. Small enough to write here later. |
| Soft-NMS, score fusion, score calibration | Change scores, so they are tied to a method's score scale; they need `accepts_score_kind` on the contract and a second score on `GraspGroup`. |
| cuRobo / RM4D reachability | Needs a robot description and a calibrated base frame; outside what a benchmark frame defines. `WorkspaceFilter` with `transform` is the cheap stand-in. |
| Open3D for the mesh check | Not in the shared extras environment; trimesh does what is needed. |
