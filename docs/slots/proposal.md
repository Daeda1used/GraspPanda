# Slot: seed proposal (`model.proposal`)

Options for the stage that decides **where grasps are predicted**. Every option on this page was
composed under all five seed-based methods and, where the contract rules accept it, probed on
real GraspNet-1Billion frames on 2026-10-13; the verdicts are copied from the
[measured records](../advanced_guides/measured_combinations.md). **No AP has been
measured for any option that is new here**: a probe trains 20 iterations on 8 frames and predicts
on 2, which shows that a combination builds, trains with finite losses and produces grasps, not
how good it is.

## 1. The task of the slot

A seed-based detector (GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline, Scale-Balanced-Grasp)
predicts one grasp per *seed*: a 3-D point with a feature vector. The proposal turns the scene
features of the encoder into a fixed number of seeds per frame, usually 1024. It is the only stage
that decides which parts of the scene the rest of the model ever looks at: the view stage picks an
approach direction per seed, the grouper crops the scene around each seed, the head predicts the
grasp there. A proposal owns whatever it predicts to make that decision (point-wise graspness and
objectness in the GSNet family), with the targets and the loss.

```text
backbone ──► proposal ──► view ──► (assigner) ──► grouper ──► head
 features     seeds        approach  labels        crops        grasps
```

Every published proposal is the same five choices, and this page offers each as an option:

| Choice | What it decides | Where it is set |
|---|---|---|
| scorer | what is predicted per scene point (objectness, graspness, nothing) | the proposal class |
| selecting score | which score cuts and ranks | `select_by` |
| candidate rule | threshold, product threshold or rank cut | `graspness_threshold`, `objectness_threshold`, `top_fraction` |
| sampler | which candidates become seeds | `sampler: {type: ...}`, a component of its own |
| too few points | what a frame with fewer candidates than seeds gets | `too_few` |

## 2. Input and output

`proposal(feats, observation) -> (SeedSet, scores)` and `proposal.loss(scores, supervision) -> dict`.

| | Type | Content |
|---|---|---|
| in | `PointFeatures` | `feats`: `(N, C)` features of the encoder, bound to a point set. `N` is the number of points of the whole batch; frames are told apart by `points.batch_offsets`. `C` is injected (`in_channels`). |
| in | `Observation` | `points`: the scene `PointSet`, `(N, 3)` coordinates in metres in the camera frame. |
| out | `SeedSet` | `points`: `B × num_seeds` seeds as a child set of the scene (`parent_index` says which scene point each seed is); `feats`: `(M, C)` features of the seeds, still attached to the encoder's graph; `scores`: named per-seed scores (`graspness`, `objectness`), detached; `valid`: `(M,)`, false for padding; `source`: how the seeds were chosen. |
| out | scores | whatever the proposal's own `loss` needs (`PointScores` for a scoring proposal, `None` otherwise). |

What the neighbours require:

* **Every frame yields exactly `num_seeds` seeds**, so that later stages work on dense `(B, M, …)`
  tensors. Padding is marked `valid = False` and never becomes a grasp; the share of real seeds is
  logged as `metric/real_seeds`.
* **Seeds are scene points.** Assigners match labels through `parent_index`. All options here pick
  existing points; none invents positions.
* **Groupers that gather among the seeds** (`CylinderGroup` and relatives, `support = "seeds"`) see
  the seed density as an input: 1024 seeds by top-k and 1024 by furthest point sampling give the
  same grouper very different neighbourhoods.
* **Seeds are chosen from the predictions in training too**, so a sampler changes which examples
  the view stage and the head are trained on from the first iteration.

The contract attributes, and what the config check (`grasppanda compose`, `grasppanda validate`,
the detector's constructor) does with them:

| Attribute | Values | Rule |
|---|---|---|
| `requires` | `dense`, `subsampled`, `any` | must equal the encoder's `output`; `compose` inserts the adapter (`FeaturePropagation` to make a sub-sampling encoder dense, `FurthestPointSubsample` for the opposite). |
| `seed_scores` | names of the scores it predicts | one owner per score: a proposal that predicts `objectness` is refused next to `ObjectnessViewStage`, which predicts it too. A head that decodes with a score needs a stage that predicts it. |
| `sampler_inputs` (proposals with a sampler) against the sampler's `needs` | `scores`, `feats` | a sampler that reads scores is refused inside a proposal that predicts none. |

## 3. Options at a glance

**Proposal classes** (registry `PROPOSALS`):

| Class | Scorer | Requires | Predicts | Status |
|---|---|---|---|---|
| `GraspnessFPS` | objectness + graspness | dense | graspness, objectness | reference port (GSNet, EconomicGrasp, FineGrasp); selection and sampler hard-wired, with the two quirks of the reference sampling kernel |
| `SubsampledSeeds` | none | subsampled | nothing | reference port (GraspNet-baseline, Scale-Balanced-Grasp); the encoder's subsample is the seed set |
| `ScoredSeeds` | objectness + graspness (same targets and losses as `GraspnessFPS`) | dense | graspness, objectness | **new**: selecting score, candidate rule, sampler and padding are options |
| `SampledSeeds` | none | dense | nothing | **new**: a sampler picks among all scene points; the dense counterpart of `SubsampledSeeds` |

**Samplers** (registry `SAMPLERS`, slot `model.proposal.sampler`). All weights are "random init":
a sampler has no parameters, and the scorer of `ScoredSeeds` is trained with the model.

| Sampler | What it is | Paper | Code | Reads | Extra |
|---|---|---|---|---|---|
| `FPS` | furthest point sampling on coordinates (D-FPS); start `first` or `random` | PointNet++, Qi et al., NeurIPS 2017 | `grasppanda.ops.furthest_point_sample` | coordinates | none |
| `FeatureFPS` | furthest point sampling in feature space (F-FPS) | [3DSSD](https://arxiv.org/abs/2002.10187), Yang et al., CVPR 2020 (oral) | written from the definition; distance conventions read in [3DSSD@8bc7605](https://github.com/dvlab-research/3DSSD/blob/8bc7605/lib/utils/layers_util.py#L84-L111) (MIT) | features | none |
| `FusionFPS` | part of the seeds by F-FPS, the rest by D-FPS (FS) | 3DSSD, CVPR 2020 | as above | features | none |
| `ScoreWeightedFPS` | semantics-guided FPS (S-FPS): maximise score^γ × distance | [SASA](https://arxiv.org/abs/2201.01976), Chen et al., AAAI 2022 | written from the definition; kernel convention read in [SASA@cd79f60](https://github.com/blakechen97/SASA/blob/cd79f60/pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu.cu#L414-L463) (Apache-2.0) | scores | none |
| `TopK` | the best-scored points, no spread | [IA-SSD](https://arxiv.org/abs/2203.11139), Zhang et al., CVPR 2022 (oral); the last step of [GraNet](https://github.com/wang-h-w/GraNet/tree/d9992e1), Wang et al., IROS 2023 | `torch.sort`; read in [IA-SSD@6b042b5](https://github.com/yifanzhang713/IA-SSD/blob/6b042b5/pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py#L276-L280) (Apache-2.0) | scores | none |
| `RandomSampler` | uniform, without replacement, own seeded generator | sampling ablation of GSNet (Wang et al., ICCV 2021); RandLA-Net, Hu et al., CVPR 2020 | `torch.randperm` | coordinates | none |
| `VoxelGridSampler` | one point per occupied voxel (the best-scored one when scores exist) | regular subsampling as in [KPConv](https://arxiv.org/abs/1904.08889), Thomas et al., ICCV 2019 | `grasppanda.ops.voxelize` | coordinates (scores optional) | none |
| `FPSThenTopK` | FPS for coverage, then the best-scored of the rest | selection rule of [Contact-GraspNet](https://arxiv.org/abs/2103.14127), Sundermeyer et al., ICRA 2021 | written from the description (the reference code is under the NVIDIA Source Code License and was not copied) | scores | none |
| `ScoreNMS` | best score first, no two seeds closer than a radius | Algorithm 1 of [S4G](https://arxiv.org/abs/1910.14218), Qin et al., CoRL 2019 | written from the paper (the repository has no licence file) | scores | none |
| `FpsampleFPS` | CPU furthest point sampling, bucket-based QuickFPS and four other methods | QuickFPS, Han et al., IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2023 (as cited by the library) | bridge to [`fpsample`](https://github.com/leonardodalinky/fpsample) 1.0.2 (MIT) | coordinates | `pip install "grasppanda[fpsample]"` |
| `TorchClusterFPS` | the furthest point sampling kernel of PyTorch Geometric (CPU and CUDA) | PointNet++, NeurIPS 2017 | bridge to [`torch_cluster.fps`](https://github.com/rusty1s/pytorch_cluster) 1.6.3 (MIT) | coordinates | `pip install "grasppanda[torch-cluster]"` |

**Ready-made options** (presets: the name to write after `--set proposal=`). "Fit" is the verdict
of the probe under GSNet / EconomicGrasp / FineGrasp / GraspNet-baseline / Scale-Balanced-Grasp,
in that order, in the words of the measured page.

<!-- FIT TABLE -->
| `--set proposal=` | What it is | GSNet | EconomicGrasp | FineGrasp | GraspNet-baseline | Scale-Balanced-Grasp | Real seeds | Scope |
|---|---|---|---|---|---|---|---|---|
| `graspness-fps` | `GraspnessFPS`: GSNet's proposal as ported (reference) | runs | runs | runs | refused | refused | 0.96 to 1.00 | graspness family |
| `SubsampledSeeds` | `SubsampledSeeds`: the encoder's subsample (reference) | runs with adapter | runs with adapter | runs with adapter | runs | runs | 1.00 to 1.00 | universal |
| `scored-fps` | `ScoredSeeds` defaults: GSNet's rule, plain `FPS` | runs | runs | runs | refused | refused | 0.98 to 1.00 | graspness family |
| `graspness-ffps` | graspable points, `FeatureFPS` (F-FPS, 3DSSD) | runs | runs | runs | refused | refused | 0.93 to 1.00 | graspness family |
| `graspness-fusion-fps` | graspable points, `FusionFPS` (FS, 3DSSD) | runs | runs | runs | refused | refused | 0.93 to 1.00 | graspness family |
| `graspness-sfps` | graspable points, `ScoreWeightedFPS` (S-FPS, SASA) | runs | runs | runs | refused | refused | 0.96 to 1.00 | graspness family |
| `graspness-topk` | graspable points, `TopK` (IA-SSD, GraNet) | runs | runs | runs | refused | refused | 0.98 to 1.00 | graspness family |
| `graspness-random` | graspable points, `RandomSampler` | runs | runs | runs | refused | refused | 0.94 to 1.00 | graspness family |
| `graspness-voxel` | graspable points, `VoxelGridSampler` 1 cm | runs | runs | runs | refused | refused | 0.83 to 1.00 | graspness family |
| `graspness-fps-topk` | graspable points, `FPSThenTopK` (Contact-GraspNet's rule) | runs | runs | runs | refused | refused | 0.93 to 1.00 | graspness family |
| `graspness-nms` | graspable points, `ScoreNMS` 1 cm (S4G's rule) | runs | runs | runs | refused | refused | 0.68 to 0.93 | graspness family |
| `graspness-fpsample` | graspable points, `FpsampleFPS` (QuickFPS, extra `fpsample`) | runs | runs | runs | refused | refused | 0.98 to 1.00 | graspness family |
| `graspness-torch-cluster` | graspable points, `TorchClusterFPS` (extra `torch-cluster`) | runs | runs | runs | refused | refused | 0.97 to 1.00 | graspness family |
| `objectness-fps` | `select_by: objectness`: predicted foreground, `FPS` | runs | runs | runs | refused | refused | 1.00 to 1.00 | graspness family |
| `product-fps` | `select_by: product`: P(object) × graspness > 0.1, `FPS` (MultiGraspNet's rule) | runs | runs | runs | refused | refused | 0.94 to 1.00 | graspness family |
| `graspness-top-fraction` | `top_fraction: 0.1`: rank cut, `FPS` (DexGraspNet 2.0's rule) | runs | runs | runs | refused | refused | 1.00 to 1.00 | graspness family |
| `graspness-fps-fill` | `too_few: best_rest`: GSNet's rule without empty frames | runs | runs | runs | refused | refused | 1.00 to 1.00 | graspness family |
| `dense-fps` | `SampledSeeds` defaults: `FPS` over all scene points | runs | runs | runs | runs with adapter | runs with adapter | 1.00 to 1.00 | universal |
| `dense-ffps` | all scene points, `FeatureFPS` (F-FPS, 3DSSD) | runs | runs | runs | runs with adapter | runs with adapter | 1.00 to 1.00 | universal |
| `dense-fusion-fps` | all scene points, `FusionFPS` (FS, 3DSSD) | runs | runs | runs | runs with adapter | runs with adapter | 1.00 to 1.00 | universal |
| `dense-random` | all scene points, `RandomSampler` | runs | runs | runs | runs with adapter | runs with adapter | 1.00 to 1.00 | universal |
| `dense-voxel` | all scene points, `VoxelGridSampler` 2 cm | runs | runs | runs | runs with adapter | runs with adapter | 0.84 to 0.86 | universal |
| `dense-fpsample` | all scene points, `FpsampleFPS` (extra `fpsample`) | runs | runs | runs | runs with adapter | runs with adapter | 1.00 to 1.00 | universal |
| `dense-torch-cluster` | all scene points, `TorchClusterFPS` (extra `torch-cluster`) | runs | runs | runs | runs with adapter | runs with adapter | 1.00 to 1.00 | universal |

"Real seeds" is the mean share of seed slots that held a real seed during the 20 probe iterations (smallest and largest over the methods that ran), with a scorer that started untrained: a low value is the padding of frames in which few points passed the candidate rule yet, not a defect.

How to read the last column: the `graspness-*`, `objectness-*`, `product-*` and `scored-*` options
are **specific to the graspness family** (GSNet, EconomicGrasp, FineGrasp): they predict objectness,
and the view stage of GraspNet-baseline and Scale-Balanced-Grasp predicts it too, so the contract
rule "one owner per score" refuses the pair with a message that names both stages. The `dense-*`
options predict nothing and are **universal**: each ran under all five methods.

Every cell of this table has a record: the 15 new scoring presets ran under each of the three methods
of the graspness family, the seven `dense-*` presets under all five methods, and the 32 refusals are
the contract rule above. The CPU swap matrix (`tests/unit/test_swap_matrix.py`) also builds, trains
one step and predicts with `ScoredSeeds` and `SampledSeeds` under every detector type. To measure
the options again, for example on another dataset:

```bash
python -m tools.probe_grid run --name mine --only 'stage/*/proposal/*'
python -m tools.probe_grid export --name mine && python -m tools.make_combination_docs
```

## 4. How to use each option

### One command

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set proposal=graspness-sfps -o my.yaml
grasppanda probe my.yaml            # 20 iterations on 8 real frames, then predict
grasppanda train my.yaml
```

Real output of the first command (tail):

```text
proposal  ScoredSeeds           preset `graspness-sfps`
view      TemplateViewStage     base config
assigner  NearestLabelAssigner  base config
grouper   CylinderGroup         base config
head      SWADHead              base config

what compose did
  • proposal: preset `graspness-sfps` (ScoredSeeds). Graspable points, then semantics-guided
    furthest point sampling (S-FPS of SASA, AAAI 2022) with graspness as the weight: spread the
    seeds, prefer graspable points.

contracts
  ✓ backbone · proposal
  ✓ proposal · view
  ...
config check
  ✓ valid  (hash 1a8cdb6139be)
```

A universal option under a method whose encoder sub-samples; `compose` wraps the encoder in
`FeaturePropagation` by itself:

```bash
grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --set proposal=dense-ffps -o gnb_ffps.yaml
grasppanda probe gnb_ffps.yaml --iters 6 --frames 4
```

```text
  model       GraspNetBaseline · 1,158,060 parameters
  trained     6 of 6 iterations · 2.01 s per iteration (median, without data; data 0.20 s), measured on this machine now
  seeds       real: min 1.00, mean 1.00 of the seed slots per iteration
              supervised: 0.49 → 0.49, min 0.47
  predicted   2 frame(s) · 282, 35 grasps

✓ runs: trained 6 iterations on 4 frames of the dataset and predicted on 2 frame(s), with finite losses and real seeds in every iteration.
```

### As a YAML fragment

Any sampler inside either proposal; this is everything a config needs:

```yaml
model:
  proposal:
    type: ScoredSeeds
    num_seeds: 1024
    select_by: graspness_and_objectness   # graspness | objectness | product
    graspness_threshold: 0.1
    objectness_threshold: 0.5
    top_fraction: null                    # e.g. 0.1: a rank cut instead of the thresholds
    too_few: pad                          # best_rest: fill with the best other points
    sampler:
      type: ScoreWeightedFPS
      gamma: 1.0
```

```yaml
model:
  proposal:
    type: SampledSeeds
    num_seeds: 1024
    sampler: {type: FeatureFPS, metric: concat, coord_weight: 1.0}
```

`ScoredSeeds` reads point-wise graspness labels: the config needs the `graspness` section of the
training dataset (the presets carry it; `grasppanda prepare graspnet1b --graspness --camera
<camera> --config <config>` builds the labels once). `SampledSeeds` needs no label of its own.

### The settings that matter

| Setting | Choose |
|---|---|
| `select_by` | `graspness_and_objectness` is GSNet's rule. `graspness` drops the objectness test (objectness is still trained). `objectness` makes every predicted object point a candidate: foreground sampling with a learned mask. `product` cuts `P(object) × graspness`, MultiGraspNet's rule ([arXiv 2602.06504](https://arxiv.org/abs/2602.06504), RA-L 2026): a soft version of the first. |
| `graspness_threshold` | belongs to the label definition: 0.1 for graspness normalised to [0, 1] per frame (GSNet) or per instance (FineGrasp). Change the label, change the threshold. |
| `top_fraction` | a rank cut: the best share of each frame's points, whatever their values. [DexGraspNet 2.0](https://arxiv.org/abs/2410.23004) selects "object points that rank top 1%" in graspness before FPS to 1024 points. With 15 000 points per frame the share must be at least 0.07 to leave 1024 candidates; the preset uses 0.1. |
| `too_few` | `pad` (GSNet): a frame without candidates has no real seed and trains the proposal only, which happens in the first iterations of every run. `best_rest`: such a frame gets the best-scored other points as real seeds. |
| `sampler.gamma` (`ScoreWeightedFPS`) | 0 is FPS started at the best point, 1 is SASA's reference, large values approach top-k. `distance: squared` is the released kernel, `euclidean` the paper's formula; `squared` with γ equals `euclidean` with γ/2. |
| `sampler.metric`, `sampler.coord_weight` (`FeatureFPS`, `FusionFPS`) | `concat` is the squared L2 distance of `[xyz, features]` as in the released 3DSSD code; `sum` is `coord_weight·|Δxyz| + |Δfeatures|`, Eq. 1 of the paper. With 512 feature channels and `coord_weight: 1` the coordinates (metres) hardly count; raise it or set `normalize: true` when the seeds cluster. |
| `sampler.voxel_size` (`VoxelGridSampler`), `sampler.min_distance` (`ScoreNMS`) | they decide how many real seeds a scene yields. Read `metric/real_seeds` from the probe and adapt. |
| `sampler.seed` (`RandomSampler`, `FPS` with `start: random`) | an integer makes two evaluations pick the same seeds. |

### Pretrained weights

There are none to load for a new option and none are needed: a sampler has no parameters. A
checkpoint of GSNet, EconomicGrasp or FineGrasp loads strictly into a model whose proposal is
`ScoredSeeds` (the scorer is the same `score` layer), so a sampler can be swapped **at test time**
on a trained model; that is a different experiment from training with the sampler (GSNet's own
sampling ablation retrains per sampler), and the config should say which one it is.

### List what is installed

```bash
grasppanda components --slot samplers
grasppanda components ScoredSeeds
```

## 5. Pitfalls met while building this

* **`fpsample` 1.0.2, bucket methods, do not begin at `start_idx`.** Measured here on 5000 random
  points, 256 picks: `vanilla` returns exactly the indices of `grasppanda.ops.furthest_point_sample`;
  `bucket` (QuickFPS) is repeatable and spreads as well (smallest spacing 0.138 against 0.140) but
  returns another seed set and does not contain the start point; the `npdu` methods assume points in
  scan order and gave a smallest spacing of 0.007 on a shuffled cloud, worse than random sampling
  (0.017). Use `method: vanilla` when the seed set must equal the reference.
* **CPU bridges are not faster here.** 20 000 → 1024 points took 6 to 12 ms with `fpsample`
  (`bucket`, h = 3 to 9) and 10 ms with the kernel GraspPanda already ships; the bridges are for
  CPU-only inference and for people who want the same kernel as another code base.
* **`torch_cluster.fps` takes a ratio, not a count,** and starts at a random point by default. The
  bridge converts the count, trims the result and sets `random_start: false`; with that it returned
  exactly the reference indices in the unit test.
* **Three distances are called F-FPS.** The 3DSSD paper adds L2 distances with a weight λ, its
  released code takes the squared L2 distance of the concatenation without λ, and SASA's code adds
  `cdist(xyz) + γ·cdist(features)`. `metric` and `coord_weight` name the convention; the unit test
  compares both with a brute-force distance matrix.
* **S-FPS: paper and kernel differ by a factor of two in γ.** The released kernel multiplies the
  *squared* distance by `score^γ`; `distance` names the convention and the test pins the relation.
* **3DSSD's fusion sampling returns 2 M points.** A proposal has a fixed seed count, so `FusionFPS`
  splits the count between the two kinds (`feature_fraction`); that is a deviation, stated in the
  docstring.
* **A score above 1 must not be clamped.** A first version clamped graspness to [0, 1] before the
  sampler; the predicted graspness is a regression output and is not bounded by 1, every value above
  1 then tied, and top-k fell back to "the first k points" (caught by the unit test). The selecting
  score is now only kept non-negative.
* **One value read back per pick makes a sampler slow on a shared GPU.** The first version of the
  general kernel read every pick as a Python number and measured feature distances by subtracting
  an `N × C` array: 1024 picks among 15 000 points with 512 channels took 18 s on the GPU of this
  machine while other jobs were running, and `dense-fusion-fps` probed at 14 s per iteration. Picks
  now stay on the device and a feature distance is one matrix-vector product per pick: 0.19 s for
  the same call (`FusionFPS` 0.5 s, `ScoreWeightedFPS` 0.3 s, `ScoreNMS` 0.2 s, plain `FPS` 0.01 s).
  Feature-space sampling over the whole scene is still the most expensive option here.
* **Probes failed with "CUDA error: unspecified launch failure" once**, in two processes at the same
  second and at unrelated places (a loss, an assigner), while the GPU held 46 of 48 GB for
  other jobs. The cells were probed again and their records replaced.
* **A stage preset used to have to be a stage of a shipped reference model** (a tripwire test).
  Options beyond the reference models are now listed in that test by component.

## 6. Not wrapped

| Candidate | Link opened | Reason |
|---|---|---|
| Object-balanced seeds (quota per instance): OBS / FS of Scale-Balanced-Grasp, per-object quantile of DexGraspNet 2.0 | [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) (an MIT licence file, on code derived from graspnet-baseline, whose terms are non-commercial), [DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2) (no licence file) | needs an instance mask as a test-time input; `Observation.masks` is declared but no dataset or transform fills it. The sampler side would be a small wrapper around any sampler here; the missing part is the mask provider with its provenance (ground truth, a segmentation network, a height threshold are different experiments). Follow-up. |
| GraNet's cascaded selection (object points, then ten value classes, graph convolutions between levels) | [GraNet@d9992e1](https://github.com/wang-h-w/GraNet/tree/d9992e1) (MIT) | not a proposal alone: it brings its own graph encoder (DGL) and a ten-class label that no dataset here builds. Its last step (top-k) is the `graspness-topk` option. |
| MultiGraspNet's graspness loss (weighted binary cross-entropy on the soft target) | [arXiv 2602.06504](https://arxiv.org/abs/2602.06504) | a primitive of the loss slot, not of this one; its candidate rule is the `product-fps` option. Written from the paper; no code of the work was used. |
| RandLA-Net code | [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | CC BY-NC-SA 4.0; random sampling needs no code from it. |
| S4G code, Contact-GraspNet code | [s4g-release](https://github.com/yzqin/s4g-release) (no licence file), [contact_graspnet](https://github.com/NVlabs/contact_graspnet) (NVIDIA licence) | licences; the two selection rules are small and were written from the papers (`ScoreNMS`, `FPSThenTopK`). |
| PyTorch3D `sample_farthest_points`, Open3D `farthest_point_down_sample`, mmcv `PointsSampler` | [mmcv@d2111bd](https://github.com/open-mmlab/mmcv/blob/d2111bd/mmcv/ops/points_sampler.py) | not installed in the shared environment (PyTorch3D and mmcv need a compiled build matching the local PyTorch); they would add a third and fourth spelling of the same D-FPS. |
| Voted seeds (VoteGrasp), mask queries (M2T2), image heat-map centres (HGGD, RNGNet, KGN) | none opened for this page | they emit virtual seeds or a variable number of regions; the seed-based skeleton needs scene points with `parent_index` (label matching). Another detector family. |
| A sampler option inside the encoders (`PointNet2`, `FurthestPointSubsample`), where 3DSSD, IA-SSD and SASA place theirs | — | outside this slot's lane (backbone package). The `SAMPLERS` registry is ready for it. |
