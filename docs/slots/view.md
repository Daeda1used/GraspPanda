# Slot `view`: approach direction and orientation

Nothing on this page carries an AP. "Runs" means: the combination passes the contract
check, builds, trains 20 iterations on real GraspNet-1Billion frames with finite losses
and predicts. How good a choice is takes a full training and an evaluation.

## 1. What the slot does

`model.view` is the third stage of the seed-based detectors (GSNet, EconomicGrasp,
FineGrasp, GraspNet-baseline, Scale-Balanced-Grasp):

```text
backbone → proposal → VIEW → (assigner, training only) → grouper → head bundle
```

A 6-DoF grasp orientation has three degrees of freedom. The view stage fixes the first
two, the **approach direction** (the unit vector from the gripper towards the object).
The head bundle then predicts the third one, the in-plane rotation about that
direction, together with depth and width, from a crop of the scene that the grouper
cuts *in the frame the view stage chose*. So the view stage decides what the head gets
to look at.

| | |
|---|---|
| Called as | `approach, seeds = view(seeds)` |
| Receives | `SeedSet`: `M` seeds of the whole batch (frames concatenated), `seeds.points.coords (M, 3)` in metres in the camera frame, `seeds.feats (M, C)`; `C` is injected by the detector (`in_channels`) |
| Returns | `ApproachSet` with one row per seed, and the `SeedSet` (a stage may add a residual to `feats`, add a score to `scores`, or mark seeds invalid) |
| `ApproachSet.direction` | `(M, 3)` unit vectors, camera frame. **Read by the decoder of every head bundle**: the grasp rotation is `approach_to_rotation(direction, in-plane angle)` |
| `ApproachSet.rotation` | `(M, 3, 3)`, column 0 = the direction, column 1 horizontal (`approach_to_rotation`). **Read by the grouper at inference** as the crop frame. Required |
| `ApproachSet.view_index` | `(M,)` index into `view.template`, the lattice the label files are indexed by (300 Fibonacci views). **Read by the assigner in training**, which looks up the labels of that view and returns the training crop frame (object pose × template frame of that view). Required by both assigners |
| `ApproachSet.logits` | whatever the stage needs for its own loss |
| Owns | `view.loss(approach, matched) -> {"loss/…": scalar, "metric/…": scalar}` and its targets, computed from the labels the assigner matched |
| Must have | the attribute `template` (a `ViewTemplate` with as many views as the label set): the detector hands it to the assigner |
| Declares | `label_formats` (kinds of matched labels it can read: `dense` grids `view_friction (M, V, A, D)`, `compact` scores `view_graspness (M, V)`), `needs_matched` (optional label fields it cannot do without), `seed_scores` (scores it adds to the seeds), `applies_seed_weights` (whether its loss honours the per-seed weight of a scale-balanced assigner) |

Three facts about the neighbours shape every option here:

1. **The labels know 300 directions.** GraspNet-1Billion and GraspClutter6D grasp labels
   are indexed by a 300-view Fibonacci lattice, and the assigners address labels by
   `view_index` into it. A stage may predict on another lattice, or no lattice at all,
   but it must *snap* its choice to one of those 300 for the label lookup, and its
   targets can never be finer than they are.
2. **Train and inference crops differ by the snap.** In training the crop frame is the
   label frame of the snapped view; at inference it is the frame of the direction the
   stage returned. For a stage whose directions are the 300 label views the two are the
   same up to the re-indexing of the labels under the object pose; for every other
   stage they differ by up to the lattice spacing (table in section 5).
3. **The head still owns the in-plane angle.** All registered head bundles decode
   `approach_to_rotation(direction, angle bin)`. A stage that regresses a *full*
   rotation therefore contributes its first column; the rest is an auxiliary output
   (`RegressedApproach.frame`) that is supervised and logged. A head bundle that takes
   the full frame from the view stage does not exist yet (section 7).

## 2. Which option when

| You want | Take | Why |
|---|---|---|
| the reference behaviour of GSNet / EconomicGrasp / FineGrasp | `template-view` | a score per label view, training view sampled from the scores |
| the reference behaviour of GraspNet-baseline / Scale-Balanced-Grasp | `objectness-view`, `objectness-view-graspable` | also predicts objectness, which their decoders read |
| the same view scoring, but with seed weights or as a base for the options below | `lattice-view` | computes what `template-view` computes (tested with shared weights), and applies the weights of `ScaleBalancedAssigner` |
| a smaller output layer / coarser choice | `lattice-fibonacci-120`, `icosphere-162`, `healpix-192` | fewer classes; each view is supervised by its nearest label view |
| only approaches away from the camera | `lattice-hemisphere-60` | the 60-of-120 template of RGB Matters |
| a classification problem instead of 300 regressions | `lattice-best-view` | softmax cross-entropy on the most graspable view |
| ... where near misses are cheap | `lattice-angular-cost` | adds the angular class loss of RGB Matters |
| a direction between lattice points at inference | `lattice-soft-argmax` | softmax-weighted mean of the top-k view directions |
| training crops closer to what inference will choose | `lattice-topk` | samples the training view among the 10 best views instead of all 300 |
| no lattice in the output at all | `direction-regression` | 3 numbers per seed, cosine loss |
| ... with a confidence of the direction | `direction-vmf` | direction + concentration, von Mises-Fisher likelihood |
| a full grasp frame from two vectors | `two-vector`, `two-vector-baseline` | Gram-Schmidt in the order of the 6-D representation, or of Contact-GraspNet |
| a full rotation in a representation of your choice | `rotation-6d`, `rotation-9d`, `rotation-quaternion` (or `representation: rotvec`) | the mappings of RoMa |

All options that are not marked "reference" are **universal**: they read dense or
compact labels, add no seed score, need no optional label field and apply seed
weights, so the contract rules accept them behind every proposal and assigner and in
front of every grouper and head bundle. The one refusal you will meet is under
Scale-Balanced-Grasp, and it is the head's: its decoder multiplies by the objectness
of a seed, which only `ObjectnessViewStage` predicts (section 4, last example).

## 3. Catalogue

Fit per method is the verdict of the probe on real frames
(`docs/advanced_guides/measured_combinations.md`, probed on 2026-10-13): GS = GSNet,
EG = EconomicGrasp, FG = FineGrasp, GB = GraspNet-baseline, SB = Scale-Balanced-Grasp.

<!-- measured:begin -->
| Preset (`--set view=…`) | Component | What it predicts | GS | EG | FG | GB | SB | Scope |
|---|---|---|---|---|---|---|---|---|
| `template-view` | `TemplateViewStage` | score per label view, 300 Fibonacci views | runs | runs | runs | runs | refused | reference (GSNet family) |
| `objectness-view` | `ObjectnessViewStage` | objectness + score per view | refused | refused | refused | runs | runs | reference (GraspNet-baseline): needs object labels per seed and view widths, owns objectness |
| `objectness-view-graspable` | `ObjectnessViewStage` | objectness (graspable seeds) + score per view | refused | refused | refused | runs | runs | reference (Scale-Balanced-Grasp): as above |
| `lattice-view` | `LatticeViewStage` | score per view, 300 Fibonacci views, seed weights applied | runs | runs | runs | runs | refused | universal |
| `lattice-fibonacci-120` | `LatticeViewStage` | score per view, 120 Fibonacci views | runs | runs | runs | runs | refused | universal |
| `lattice-hemisphere-60` | `LatticeViewStage` | score per view, 60 camera-facing views | runs | runs | runs | runs | refused | universal |
| `lattice-best-view` | `LatticeViewStage` | classification of the best view (cross-entropy) | runs | runs | runs | runs | refused | universal |
| `lattice-angular-cost` | `LatticeViewStage` | classification + angular class loss | runs | runs | runs | runs | refused | universal |
| `lattice-soft-argmax` | `LatticeViewStage` | score per view; soft-arg-max read-out (top 8, T = 0.05) | runs | runs | runs | runs | refused | universal |
| `lattice-topk` | `LatticeViewStage` | score per view; training view sampled among the top 10 | runs | runs | runs | runs | refused | universal |
| `icosphere-162` | `IcosphereViewStage` | score per view, 162 icosphere vertices | runs | runs | runs | runs | refused | universal |
| `icosphere-642` | `IcosphereViewStage` | score per view, 642 icosphere vertices | runs | runs | runs | runs | refused | universal |
| `healpix-192` | `HealpixViewStage` | score per view, 192 HEALPix pixels | runs | runs | runs | runs | refused | universal |
| `healpix-768` | `HealpixViewStage` | score per view, 768 HEALPix pixels | runs | runs | runs | runs | refused | universal |
| `direction-regression` | `DirectionViewStage` | regressed approach vector, cosine loss | runs | runs | runs | runs | refused | universal |
| `direction-vmf` | `DirectionViewStage` | regressed approach + concentration, vMF likelihood | runs | runs | runs | runs | refused | universal |
| `two-vector` | `TwoVectorViewStage` | two vectors, Gram-Schmidt keeping the approach | runs | runs | runs | runs | refused | universal |
| `two-vector-baseline` | `TwoVectorViewStage` | two vectors, Gram-Schmidt keeping the baseline | runs | runs | runs | runs | refused | universal |
| `rotation-6d` | `RomaRotationViewStage` | full rotation, 6-D | runs | runs | runs | runs | refused | universal |
| `rotation-9d` | `RomaRotationViewStage` | full rotation, 9-D + SVD | runs | runs | runs | runs | refused | universal |
| `rotation-quaternion` | `RomaRotationViewStage` | full rotation, quaternion | runs | runs | runs | runs | refused | universal |
<!-- measured:end -->

Sources, as opened for this page:

| Component | Idea from | Code | Licence | Extra to install |
|---|---|---|---|---|
| `TemplateViewStage`, `ObjectnessViewStage` | GSNet (ICCV 2021), GraspNet-1Billion (CVPR 2020) | ported earlier, written for GraspPanda | Apache-2.0 | none |
| `LatticeViewStage` | the same view scoring on a free lattice; hemisphere template of RGB Matters ([arXiv 2103.02184](https://arxiv.org/abs/2103.02184), "Accepted by ICRA 2021") | written for GraspPanda; the angular class loss is adapted from [GouMinghao/rgb_matters@2e176728f6](https://github.com/GouMinghao/rgb_matters/blob/2e176728f6/rgbd_graspnet/net/loss.py) (`Class_Loss`) | MIT (Copyright (c) 2022 Gou Minghao), see `THIRD_PARTY_NOTICES.md` | none |
| `IcosphereViewStage` | subdivided icosahedron | bridge to [trimesh](https://github.com/mikedh/trimesh) `trimesh.creation.icosphere` (5.1.1 here) | MIT | `pip install "grasppanda[trimesh]"` |
| `HealpixViewStage` | HEALPix (Gorski et al., 2005, ApJ 622, 759, as cited by the library) | bridge to [healpix](https://github.com/ntessore/healpix) `healpix.pix2vec` (2025.1 here) | BSD-3-Clause | `pip install "grasppanda[healpix]"` |
| `DirectionViewStage` | regression of the approach vector: the approach network of Generalizing-Grasp ([arXiv 2404.01727](https://arxiv.org/abs/2404.01727), "Accepted at CVPR 2024"); with `concentration`, the von Mises-Fisher direction likelihood of vMF-Contact ([arXiv 2411.03591](https://arxiv.org/abs/2411.03591), "Accepted to ICRA 2025") | written for GraspPanda (both upstream repositories have no licence file) | Apache-2.0 | none |
| `TwoVectorViewStage` | 6-D rotation representation (Zhou et al., [arXiv 1812.07035](https://arxiv.org/abs/1812.07035), CVPR 2019); baseline/approach order of Contact-GraspNet ([arXiv 2103.14127](https://arxiv.org/abs/2103.14127), ICRA 2021) | written for GraspPanda (the idea only) | Apache-2.0 | none |
| `RomaRotationViewStage` | 6-D (Zhou et al.), 9-D + SVD (Levinson et al., [arXiv 2006.14616](https://arxiv.org/abs/2006.14616)), quaternion, rotation vector | bridge to [RoMa](https://github.com/naver/roma) (Brégier, "Oral presentation at 3DV 2021"; 1.6.1 here): `special_gramschmidt`, `special_procrustes`, `unitquat_to_rotmat`, `rotvec_to_rotmat` | BSD-3-Clause (Copyright (c) 2020 NAVER Corp.) | `pip install "grasppanda[roma]"` |

There are no pretrained weights for any of the new stages: all start from random
initialisation. Released checkpoints of the six ported methods contain
`TemplateViewStage` or `ObjectnessViewStage`; loading one into a model with another view
stage leaves the view stage (and nothing else) untrained.

What each parameterisation demands:

| Stage | From the labels | From the assigner | From the head bundle |
|---|---|---|---|
| lattice stages (`Lattice`, `Icosphere`, `Healpix`) | per seed, the graspness of all 300 label views: dense `view_friction (M, 300, A, D)` or compact `view_graspness (M, 300)` encoded with the same `max_friction` | label lookup at `view_index`; `weight` is applied when returned | nothing |
| `DirectionViewStage` | the same; reduced to the single most graspable view | the same | nothing |
| `TwoVectorViewStage`, `RomaRotationViewStage` | the same, plus for the frame target the grid of the snapped view (dense `friction (M, A, D)`) or its labelled angle bin (compact `angle (M,)`) | additionally `crop_rotation`, the label frame of the snapped view | `num_angle` equal to the stage's (checked by the contract rules) |

## 4. How to use each option

Every option is one `--set`:

```bash
grasppanda compose --list view                       # the table of presets
grasppanda compose --show view=lattice-soft-argmax   # one preset with its notes
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set view=rotation-6d -o my.yaml
grasppanda probe my.yaml                             # 20 iterations on real frames
grasppanda train my.yaml
```

```text
$ grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set view=rotation-6d -o my.yaml
slot      component              from
────────  ─────────────────────  ────────────────────
backbone  SparseUNet14           base config
proposal  GraspnessFPS           base config
view      RomaRotationViewStage  preset `rotation-6d`
assigner  NearestLabelAssigner   base config
grouper   CylinderGroup          base config
head      SWADHead               base config
...
contracts
  ✓ backbone · proposal
  ✓ proposal · view
  ✓ view · assigner
  ✓ assigner · head
  ✓ grouper · head
  ✓ view · head
  ✓ proposal · view · head

config check
  ✓ valid  (hash 8347abcb50ef)
```

The same as a YAML fragment, to paste into a config that inherits a method:

```yaml
model:
  view:
    type: RomaRotationViewStage
    representation: 6d            # 6d | 9d | quaternion | rotvec
    num_angle: 12                 # must equal the head bundle's
```

### Lattice stages

```yaml
model:
  view:
    type: LatticeViewStage        # or IcosphereViewStage / HealpixViewStage
    num_views: 300                # IcosphereViewStage: subdivisions: 2; HealpixViewStage: nside: 4
    hemisphere: false             # true: only approaches away from the camera
    target: score                 # score | best_view
    selection: argmax             # evaluation: argmax | soft_argmax
    train_selection: multinomial  # training: multinomial | topk | argmax
    top_k: null                   # used by soft_argmax and topk
    temperature: 1.0              # softmax temperature of soft_argmax
    angular_weight: 0.0           # best_view only: weight of the angular class loss
    loss: {type: SmoothL1, weight: 100.0}   # best_view: {type: CrossEntropy, weight: 1.0}
```

- **Size.** `LatticeViewStage` takes any `num_views`; `IcosphereViewStage` has
  `10 · 4**subdivisions + 2` views (12, 42, 162, 642, 2562); `HealpixViewStage` has
  `12 · nside**2` (48, 192, 300, 768 for nside 2, 4, 5, 8). Up to 300 views every view
  reads its own label column; beyond 300 neighbouring views share one (section 5).
- **`target: score`** is GSNet's: regress the min-max normalised graspness of every
  view. **`target: best_view`** makes it a classification of the most graspable view
  and logs `metric/view_accuracy` and `metric/view_error_deg`; use
  `train_selection: argmax` with it, and a `CrossEntropy` loss.
- **`angular_weight`** adds `loss/view_angular`: the expected `(1 - cos)/2` between the
  predicted view distribution and the best view. Set the weight of `loss` to 0 to train
  with it alone.
- **`selection: soft_argmax`** changes inference only. Choose `temperature` by the range
  of the outputs: about 0.05 for regressed scores in [0, 1], 1.0 for logits. Keep
  `top_k` small (8) so that two separate good regions are not averaged into the bad
  region between them.
- **`train_selection: topk`** restricts GSNet's training-time sampling to the `top_k`
  best views.

### Regression stages

```yaml
model:
  view:
    type: DirectionViewStage
    concentration: false          # true: von Mises-Fisher likelihood, logs metric/view_concentration
    direction_weight: 1.0
---
model:
  view:
    type: TwoVectorViewStage
    keep: approach                # approach (6-D order) | baseline (Contact-GraspNet order)
    direction_weight: 1.0         # loss/view: 1 - cos to the most graspable label view
    frame_weight: 1.0             # loss/view_frame: squared Frobenius distance, half-turn aware
    num_angle: 12
```

- `loss/view` pulls the direction to the most graspable label view;
  `metric/view_error_deg` is the mean angle to it.
- `loss/view_frame` (two-vector and RoMa stages) compares the regressed frame with the
  label frame of the snapped view at its best in-plane angle, taking the smaller of the
  distances to that frame and to its half-turn copy (a parallel-jaw grasp turned by 180°
  about its approach is the same grasp). `metric/view_frame_error_deg` is the mean
  rotation angle.
- `representation` of `RomaRotationViewStage`: `6d` and `9d` are continuous
  parameterisations; `quaternion` and `rotvec` are not and are there for comparison.
  `rotvec` is unit-tested and has no preset and no probe record.
- `residual: true` (all new stages) adds the hidden features of the stage to the seed
  features that the grouper and the head consume, as GSNet's stage does.

### Under Scale-Balanced-Grasp

```text
$ grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e --set view=lattice-view
  ✗ proposal · view · head: OperationHead as configured reads the seed score 'objectness' when it
    decodes, but neither SubsampledSeeds nor LatticeViewStage predicts it. Use a stage that does
    (proposals: GraspnessFPS; view stages: ObjectnessViewStage) or another decode option of the
    head.
```

The scale-balanced weights themselves are not the obstacle (the new stages apply them;
`TemplateViewStage` is refused for ignoring them). Take the decode option of the head
that does not read objectness:

```bash
grasppanda compose scale_balanced_grasp/sbg_pointnet2-transformer_g1b-realsense_18e \
    --set view=lattice-view --set head=operation -o sbg_lattice.yaml
```

<!-- sbg:begin -->
```text
$ grasppanda probe sbg_lattice.yaml --opts data.train.dataset.scale_statistic.source=legacy_import \
      data.train.dataset.scale_statistic.reference_sha256=<SHA-256 of the reference's width histogram>
  model       ScaleBalancedGrasp · 1,484,060 parameters
  data        8 frames of the config's dataset · 2 per iteration
  device      cuda
  trained     20 of 20 iterations · 0.33 s per iteration (median, without data; data 0.19 s), measured on this machine now
  memory      peak 9.4 GiB on the GPU, this process
  losses      first → last
    loss/view                   12.3 → 0.3184
    loss/score                  0.106 → 0.06177
    loss/angle                  0.4933 → 0.477
    loss/width                  0.4659 → 0.2944
    loss/tolerance              0.8858 → 0.4744
    loss                        14.25 → 1.626
  seeds       real: min 1.00, mean 1.00 of the seed slots per iteration
              supervised: 0.51 → 0.50, min 0.47
  predicted   2 frame(s) · 1024, 1024 grasps

✓ runs: trained 20 iterations on 8 frames of the dataset and predicted on 2 frame(s), with finite losses and real seeds in every iteration.
  (16 s)
```

The two `--opts` import the width histogram of the reference implementation, which a probe
cannot build from a few frames (`configs/scale_balanced_grasp/README.md`).
<!-- sbg:end -->

## 5. Pitfalls met while building this

- **A finer lattice does not give finer labels.** Measured with 200 000 random
  directions (`grasppanda.ops` lattices, this machine):

  | Lattice | Views | Random direction to nearest view, mean / max | View to its nearest label view, mean / max | Label columns read |
  |---|---|---|---|---|
  | Fibonacci (the labels) | 300 | 4.5° / 9.0° | 0° | 300 |
  | Fibonacci | 120 | 7.1° / 14.1° | 4.6° / 7.7° | 120 |
  | Fibonacci | 1200 | 2.2° / 4.4° | 4.5° / 8.2° | 300 |
  | icosphere | 42 | 11.9° / 20.8° | 4.8° / 7.7° | 42 |
  | icosphere | 162 | 6.1° / 10.8° | 4.5° / 7.7° | 162 |
  | icosphere | 642 | 3.0° / 5.4° | 4.5° / 8.1° | 300 |
  | HEALPix | 48 | 11.2° / 23.4° | 4.5° / 6.7° | 48 |
  | HEALPix | 192 | 5.6° / 11.5° | 4.5° / 7.9° | 190 |
  | HEALPix | 300 | 4.5° / 9.2° | 4.4° / 8.3° | 274 |
  | HEALPix | 768 | 2.8° / 5.6° | 4.5° / 8.4° | 300 |

  The third column is the price of a coarse lattice; the fourth is the train/inference
  crop mismatch of any lattice that is not the label lattice. HEALPix with 192 views
  already maps two pairs of views onto the same label column.
- **`TemplateViewStage(num_views=N)` with N ≠ 300 is not "another lattice size".** Its
  template is also the label lattice, so the assigner refuses the labels at the first
  loss. Use `LatticeViewStage(num_views=N)`.
- **Soft-arg-max needs a temperature that fits the outputs.** With regressed scores and
  `temperature: 1.0` the weights are nearly uniform and the mean direction is
  meaningless; with all views voting (`top_k: null`) opposite views can cancel (the
  stage then falls back to the best view).
- **A single best view is a brittle target.** `best_view`, `direction-regression` and
  the frame stages reduce the 300 scores of a seed to their arg-max; on a symmetric
  object two distant views can tie and the target flips between them. `target: score`
  and `angular_weight` do not have this problem to the same degree.
- **Flip augmentation yields mirrored label frames.** `crop_rotation` of a mirrored
  object pose is a reflection; the frame loss skips those seeds (their direction loss
  still counts).
- **The vMF loss is a log-density and goes negative** when the predicted concentration
  is high. That is not a bug; do not compare its value with the cosine loss.
- **`RomaRotationViewStage` imports RoMa in every forward** (through the cached
  `require`), not as a stored module attribute: a module attribute would make the
  detector impossible to deep-copy.
- **healpy is GPL-2.0**, so the bridge uses the BSD-licensed `healpix` package, which
  ships wheels and needs only NumPy.

## 6. Not wrapped

| Candidate | Opened | Why not |
|---|---|---|
| Contact-GraspNet code (two-vector head, contact labels, 5-point loss) | [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) | its `License.pdf` is the "NVIDIA Source Code License for Contact-GraspNet"; section 3.3 limits use to non-commercial research and evaluation. The idea of its parameterisation is `two-vector-baseline`; its contact-point labels and control-point loss need a label asset the toolbox does not have |
| vMF-Contact code | [YitianShi/vMF-Contact](https://github.com/YitianShi/vMF-Contact) | no licence file (the licence request returns 404). The closed form of the likelihood is textbook and is written here; its evidential posterior update is not |
| Generalizing-Grasp code (regressed view with a neighbourhood term, `2θ` angle regression) | [mahaoxiang822/Generalizing-Grasp](https://github.com/mahaoxiang822/Generalizing-Grasp) | no licence file. `direction-regression` is the cosine part of the idea; its soft neighbourhood term is not included |
| S4G (6-D rotation per point, symmetric loss) | [yzqin/s4g-release](https://github.com/yzqin/s4g-release) | no licence file. The same representation and a half-turn-aware loss are in `rotation-6d` / `two-vector` |
| DexGraspNet 2.0 (diffusion on seed features) | [PKU-EPIC/DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2) | no licence file; a generative head is a head bundle, not a view stage |
| OrbitGrasp, Edge Grasp Network | [BoceHu/orbitgrasp](https://github.com/BoceHu/orbitgrasp) (MIT), [HaojHuang/Edge-Grasp-Network](https://github.com/HaojHuang/Edge-Grasp-Network) (MIT) | detectors of their own (equivariant networks, edge features) with their own simulated labels; they do not fit a seed-feature view stage. Only the licences were checked here |
| healpy, astropy-healpix | [healpy](https://github.com/healpy/healpy) (GPL-2.0), [astropy-healpix](https://github.com/astropy/astropy-healpix) (BSD-3-Clause) | GPL; and a dependency on astropy for 12·nside² vectors. `healpix` does the same with NumPy only |
| PyTorch3D rotation conversions | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | a compiled package for functions RoMa has in pure PyTorch |
| Top-k views per seed at inference (several grasps per location) | - | `ApproachSet` is one row per seed and the heads return one grasp per seed; it needs a change of the skeleton, not a view stage |
| A head bundle that takes the regressed in-plane rotation | - | belongs to the head slot; until it exists the frame of the rotation stages is auxiliary |
