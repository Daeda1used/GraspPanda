# Grasp heads and losses: what is predicted per seed and how it is penalised

## 1. The task of the slot

A seed-based detector (GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline,
Scale-Balanced-Grasp) picks seed points, chooses an approach direction per seed, crops the
scene around the seed along that direction, and then has to say **which grasp** to execute
there: the in-plane rotation, how deep the fingers go, how wide they open, and how good the
grasp is. That last step is the *head bundle* (`model.head`). It is the last stage of the
network: `grouper` → **head** → `GraspGroup`.

A head bundle is four parts that cannot be mixed across bundles, because they share one
grasp parameterisation: the network head, the target encoder (labels → what the head is
trained on), the loss, and the decoder (outputs → grasps). So the bundle is swapped as a
whole. *Inside* a bundle, and inside the proposal and view stages before it, every loss term
is a small config node of its own (`{type: SmoothL1, weight: 10.0}`); those are swapped one
by one. This page covers both: head bundles, and the loss primitives that go into the loss
terms of any stage.

No AP is claimed for any option of this page. What was checked is that each one builds,
trains a few iterations on real frames with finite losses and predicts (the measured
records), and a unit test of its contract.

## 2. Input and output

### The head bundle

```python
class HeadBundle(nn.Module):
    def forward(self, crops: GroupFeatures) -> Predictions: ...        # raw outputs, with gradients
    def encode(self, matched: MatchedLabels, ...) -> Targets: ...      # labels -> targets
    def loss(self, predictions, targets, ...) -> dict[str, Tensor]:    # named, weighted terms
    def decode(self, raw, seeds, approach, gripper) -> GraspGroup: ... # one grasp per seed
```

| | |
|---|---|
| in: crop features | from the grouper, one row per valid seed. Their named axes are the grouper's `out_axes`: `('seed', 'channel')` (`CylinderGroup`, `CylinderInteractionGroup`, `MultiScaleGroup`) or `('seed', 'depth', 'channel')` (`DepthCylinderGroup`, one crop per depth bin). |
| in: matched labels | from the assigner, the labels of the scene point nearest to every seed, already reduced to the chosen view: `dense` (a friction and a width per in-plane angle and depth bin) or `compact` (one best grasp per seed). |
| out, training | a dictionary of named loss terms, each already multiplied by its `weight`; the trainer sums them and logs each as `loss/<name>`. |
| out, prediction | a `GraspGroup` with one grasp per seed: `translation` (the seed, metres, camera frame), `rotation` (approach from the view stage, in-plane angle from the head), `depth`, `width` (metres), `score`. |

Contract attributes of a head bundle, and what the config checker does with them:

| attribute | meaning | rule |
|---|---|---|
| `in_axes` | axes of the crop features it expects | must equal the grouper's `out_axes` |
| `label_format` | `dense` or `compact` | must equal the assigner's `label_format` (and the dataset must serve that label set) |
| `score_kind` | meaning of the decoded score | scores of different kinds are not comparable; shown in the catalogue |
| `applies_seed_weights` | whether its loss honours a weight per seed from the assigner | an assigner that returns weights needs a head that applies them |
| `loss_terms` | loss option → kind of term (below) | every loss node is checked against it |

### A loss primitive

```python
class Loss(nn.Module):
    def forward(self, prediction, target, mask=None, weights=None) -> Tensor  # weight x mean over the supervised elements
```

A stage declares the **kind** of every loss option it has, a primitive the kinds it
**accepts**. There are four kinds:

| kind | what the stage hands to the loss | examples |
|---|---|---|
| `class` | `(N, C)` logits against `(N,)` class indices | objectness of points or seeds; angle, depth and quality classes |
| `unit` | prediction and target of equal shape, the target a quality in `[0, 1]` | graspness, view scores of `TemplateViewStage`, the score grid of `SWADHead` |
| `value` | prediction and target of equal shape, the target unbounded | widths, tolerances, the log-friction scores of the GraspNet-baseline family (up to ln 10) |
| `bins` | `(N, C)` logits over bins against `(N,)` fractional bin positions | the width of `DecoupledGridHead` with `width: bins` |

Which term of which stage is which kind (the stages of the five shipped methods):

| stage | option (write it as `<slot>.<option>`) | kind | loss in the shipped configs |
|---|---|---|---|
| `GraspnessFPS` (GSNet, EconomicGrasp, FineGrasp) | `proposal.objectness_loss` | `class` | CrossEntropy |
| | `proposal.graspness_loss` | `unit` | SmoothL1 |
| `TemplateViewStage` (GSNet, EconomicGrasp, FineGrasp) | `view.loss` | `unit` | SmoothL1 |
| `LatticeViewStage`, `IcosphereViewStage`, `HealpixViewStage` ([view slot](view.md)) | `view.loss` | `unit` with `target: score`, `class` with `target: best_view` | SmoothL1 / CrossEntropy |
| `ObjectnessViewStage` (GraspNet-baseline, Scale-Balanced-Grasp) | `view.objectness_loss` | `class` | CrossEntropy |
| | `view.view_loss` | `value` | MSE |
| `SWADHead` (GSNet) | `head.score_loss` | `unit` | SmoothL1 |
| | `head.width_loss` | `value` | SmoothL1 |
| `EconomicHead` (EconomicGrasp, FineGrasp) | `head.angle_loss`, `head.depth_loss`, `head.score_loss` | `class` | CrossEntropy |
| | `head.width_loss` | `value` | SmoothL1 |
| `OperationHead` (GraspNet-baseline, Scale-Balanced-Grasp) | `head.angle_loss` | `class` | CrossEntropy |
| | `head.score_loss`, `head.width_loss`, `head.tolerance_loss` | `value` | SmoothL1 |
| `DecoupledGridHead` (new) | `head.score_loss` | `unit` | SmoothL1 |
| | `head.angle_loss` | `class` | CrossEntropy |
| | `head.depth_loss` | `class` or `value` (option `depth`) | CrossEntropy / SmoothL1 |
| | `head.width_loss` | `value`, `class` or `bins` (option `width`) | SmoothL1 / CrossEntropy / DistributionFocal |

The config checker refuses a loss in a term of another kind and names the losses that fit.
Two further attributes of a primitive matter:

* `prediction`: `value` or `logit`. A loss that trains the output of a `unit` term as a
  logit (`QualityFocal`, `Varifocal`, `GHMC`, `FocalLoss`, `BinaryCrossEntropy`) changes
  what the raw output means. The stages read it through the sigmoid where they consume it:
  `GraspnessFPS` compares the graspness with its threshold, `SWADHead` decodes its score.
* `applies_weights`: false for a loss that scores the supervised set as a whole
  (`DiceLoss`, `LovaszSoftmaxLoss`). It is refused in a term that receives a weight per
  seed from the assigner (Scale-Balanced-Grasp).

`grasppanda components <Loss>` prints both and the accepted kinds.

## 3. Options at a glance

### Losses

"Name" is the preset to write after `--set <slot>.<option>=`; "component" is the `type:` of
a config node. "Probed" is what the measured records hold
([measured combinations](../advanced_guides/measured_combinations.md), group `loss`): the
terms the loss was put into on real frames, all with the verdict **runs** unless stated.
Every loss is universal in the sense that it goes into every term of a kind it accepts, in
any stage of any seed-based method; that it does is a unit test on tiny models of four
methods (`tests/unit/test_losses_zoo.py`). Nothing here has weights.

| Name | Component | Fits | What it is | Paper | Code, licence | Extra | Probed in |
|---|---|---|---|---|---|---|---|
| `focal` | `FocalLoss` | `class`, `unit` | sigmoid focal loss: turns down well-classified elements; one-vs-all over classes | Lin et al., ICCV 2017, [arXiv 1708.02002](https://arxiv.org/abs/1708.02002) | route L: `torchvision.ops.sigmoid_focal_loss`, [pytorch/vision](https://github.com/pytorch/vision), BSD-3-Clause | `torchvision` | GSNet `proposal.objectness_loss`; FineGrasp `head.score_loss` |
| `softmax-focal` | `SoftmaxFocalLoss` | `class` | focal loss over mutually exclusive classes | same paper | route L: `kornia.losses`, [kornia/kornia](https://github.com/kornia/kornia), Apache-2.0 | `kornia` | GSNet `proposal.objectness_loss`; EconomicGrasp `head.angle_loss` |
| `quality-focal` | `QualityFocal` | `unit` | focal loss for a soft target: sigmoid cross-entropy focused on large errors | Generalized Focal Loss, Li et al., NeurIPS 2020, [arXiv 2006.04388](https://arxiv.org/abs/2006.04388) | route A: [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) @ `cfd5d3a`, Apache-2.0 | — | GSNet `head.score_loss`; EconomicGrasp `proposal.graspness_loss`. Refused: GraspNet-baseline `head.score_loss` (a `value` term) |
| `distribution-focal` | `DistributionFocal` | `bins` | cross-entropy to the two bins around a continuous target | same paper | route A: mmdetection, as above | — | GSNet, as the width loss of the head preset `decoupled-grid-bins` (no other shipped stage has a `bins` term) |
| `varifocal` | `Varifocal` | `unit` | positives weighted by their target quality, negatives focused | VarifocalNet, Zhang et al., CVPR 2021, [arXiv 2008.13367](https://arxiv.org/abs/2008.13367) | route A: mmdetection, as above | — | GSNet `head.score_loss`; FineGrasp `view.loss` |
| `ghm-c` | `GHMC` | `unit`, `class` | gradient harmonising, classification: every range of gradient norms counts alike | Li et al., AAAI 2019, [arXiv 1811.05181](https://arxiv.org/abs/1811.05181) | route A: mmdetection, as above | — | GSNet `head.score_loss`; GraspNet-baseline `head.angle_loss` |
| `ghm-r` | `GHMR` | `value`, `unit` | gradient harmonising, regression | same paper | route A: mmdetection, as above | — | GSNet `head.width_loss`; GraspNet-baseline `head.score_loss` |
| `balanced-l1` | `BalancedL1` | `value`, `unit` | Smooth-L1 with a larger gradient for small errors | Libra R-CNN, Pang et al., CVPR 2019, [arXiv 1904.02701](https://arxiv.org/abs/1904.02701) | route A: mmdetection, as above | — | GSNet `head.width_loss`; GraspNet-baseline `head.width_loss` |
| `huber` | `Huber` | `value`, `unit` | quadratic below `delta`, linear above | — | PyTorch `huber_loss` | — | GSNet `head.width_loss`; Scale-Balanced-Grasp `view.view_loss`, `head.width_loss` |
| `charbonnier`, `cauchy` | `RobustLoss` (`kind:` also `welsch`, `geman_mcclure`) | `value`, `unit` | robust penalties that grow slowly for large errors | special cases of Barron, CVPR 2019, [arXiv 1701.03077](https://arxiv.org/abs/1701.03077) | route L: `kornia.losses`, Apache-2.0 | `kornia` | GSNet `head.width_loss` (both); GraspNet-baseline `head.tolerance_loss` (`charbonnier`); EconomicGrasp `head.width_loss` (`cauchy`). `welsch`, `geman_mcclure`: unit test only |
| `dice` | `DiceLoss` | `class` | one minus the overlap of predicted and labelled classes over the set | V-Net, Milletari et al., [arXiv 1606.04797](https://arxiv.org/abs/1606.04797) | route L: `kornia.losses`, Apache-2.0 | `kornia` | GSNet, EconomicGrasp `proposal.objectness_loss`. Refused: Scale-Balanced-Grasp `head.angle_loss` (seed weights) |
| `lovasz-softmax` | `LovaszSoftmaxLoss` | `class` | surrogate of the mean intersection-over-union of the set | Berman et al., CVPR 2018, [arXiv 1705.08790](https://arxiv.org/abs/1705.08790) | route L: `kornia.losses`, Apache-2.0 | `kornia` | GSNet, FineGrasp `proposal.objectness_loss` |
| `label-smoothing` | `LabelSmoothingCrossEntropy` | `class` | cross-entropy with smoothed targets and optional class weights | Szegedy et al., CVPR 2016, [arXiv 1512.00567](https://arxiv.org/abs/1512.00567) | PyTorch `cross_entropy` | — | GSNet `proposal.objectness_loss`; EconomicGrasp `head.depth_loss` |
| `poly1` | `Poly1CrossEntropy` | `class` | cross-entropy plus `epsilon x (1 - p_t)` | PolyLoss, Leng et al., [arXiv 2204.12511](https://arxiv.org/abs/2204.12511) (its arXiv record names ICLR) | the Poly-1 formula on PyTorch `cross_entropy`, written here | — | GSNet `proposal.objectness_loss`; FineGrasp `head.angle_loss` |
| `bce` | `BinaryCrossEntropy` | `unit` | sigmoid cross-entropy to a soft or binary target | — | PyTorch `binary_cross_entropy_with_logits` | — | GSNet, FineGrasp `proposal.graspness_loss` |
| `ohem`, `ohem-smooth-l1` | `TopK` (a wrapper: `loss:` is any primitive) | every kind | online hard example mining: only the hardest fraction of the elements counts | Shrivastava et al., CVPR 2016, [arXiv 1604.03540](https://arxiv.org/abs/1604.03540); top-k of point errors as in Contact-GraspNet, [arXiv 2103.14127](https://arxiv.org/abs/2103.14127) | written here (a `topk` over the penalties) | — | GSNet `proposal.objectness_loss`, `head.score_loss`; FineGrasp `head.depth_loss`; GraspNet-baseline `view.view_loss` |
| `class-balanced` | `ClassBalanced` (a wrapper) | `class` | every class weighted by the inverse of its effective number of samples | Cui et al., CVPR 2019, [arXiv 1901.05555](https://arxiv.org/abs/1901.05555) | the weight formula of the paper, written here (reference: [richardaecn/class-balanced-loss](https://github.com/richardaecn/class-balanced-loss), MIT) | — | GSNet `proposal.objectness_loss`; EconomicGrasp `head.score_loss`; Scale-Balanced-Grasp `head.angle_loss` |
| `uncertainty-weighting` | `UncertaintyWeighted` (a wrapper) | every kind | multi-task weighting: a learned factor per wrapped term, which pays for being small | Kendall et al., CVPR 2018, [arXiv 1705.07115](https://arxiv.org/abs/1705.07115) | the formula as in the `UW` weighting of [LibMTL](https://github.com/median-research-group/LibMTL) (MIT), written here, one log-variance per term | — | all five loss terms of GSNet; all six of GraspNet-baseline |
| — | `CrossEntropy`, `L1`, `MSE`, `SmoothL1` | `class` / `value`, `unit` | the primitives of the shipped configs | — | PyTorch | — | the shipped configs |

### Head bundles

"Fit" is the verdict of the measured records (group `stage`, one stage replaced under the
method) for GSNet / EconomicGrasp / FineGrasp / GraspNet-baseline / Scale-Balanced-Grasp.
All are randomly initialised; none has released weights.

| Name | Component | What it is | From | Fit | Why |
|---|---|---|---|---|---|
| `swad` | `SWADHead` | a score and a width per (angle, depth) cell, regressed; arg-max cell | GSNet (existed) | runs / refused / refused / refused / refused | dense labels, crops without a depth axis |
| `swad-focal` | `SWADHead` | `swad` with `QualityFocal` on the score grid and a fixed friction range; the score is decoded through the sigmoid | preset of this page | runs / refused / refused / refused / refused | as `swad` |
| `economic` | `EconomicHead` | angle, depth and quality classes plus a width | EconomicGrasp (existed) | refused / runs / runs / refused / refused | compact labels |
| `economic-focal` | `EconomicHead` | `economic` with `SoftmaxFocalLoss` on the three class terms and the arg-max restricted to supervised bins; extra `kornia` | preset of this page | refused / runs / runs / refused / refused | as `economic` |
| `operation`, `operation-objectness` | `OperationHead` | score, angle class, width and tolerance per depth crop | GraspNet-baseline family (existed) | refused / refused / refused / runs / runs | crops with a depth axis |
| `decoupled-grid` | `DecoupledGridHead` | the score grid of SWAD, but angle and depth come from two classifiers of their own and the width is predicted once per depth bin | written here after the idea of the grasp head of FGC-GraspNet (Lu et al., ICRA 2022, [arXiv 2206.11141](https://arxiv.org/abs/2206.11141)); no upstream code | runs / refused / refused / refused / refused | dense labels, crops without a depth axis: swaps with `swad` |
| `decoupled-grid-regression` | `DecoupledGridHead` | depth and width regressed (a continuous depth) | as above | runs / refused / refused / refused / refused | as above |
| `decoupled-grid-class` | `DecoupledGridHead` | depth and width classified (width into 16 bins, decoded as the bin centre) | as above | runs / refused / refused / refused / refused | as above |
| `decoupled-grid-bins` | `DecoupledGridHead` | width as a distribution over 16 bins, trained with the distribution focal loss, decoded as the expectation | as above; loss from Generalized Focal Loss | runs / refused / refused / refused / refused | as above |

Why the refusals: EconomicGrasp and FineGrasp train on `compact` labels, which hold one
grasp per seed and not the grid a score-grid head is supervised with; GraspNet-baseline and
Scale-Balanced-Grasp crop once per depth bin, so their features have a `depth` axis. The
messages say so (section 5). A head preset can be moved to another method together with the
stages it depends on, e.g. under GraspNet-baseline
`--set view=template-view --set grouper=CylinderGroup --set head=decoupled-grid` (accepted by
`compose`; not probed);
which stage sets are legal is in [swapping components](../advanced_guides/swapping.md).

## 4. How to use each option

### One loss term

```bash
grasppanda compose --list loss                    # the presets, one line each
grasppanda compose --show loss=quality-focal      # one preset with its notes
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set head.score_loss=quality-focal --set proposal.objectness_loss=focal -o gsnet_focal.yaml
grasppanda probe gsnet_focal.yaml                 # a few iterations on real frames, about a minute
grasppanda train gsnet_focal.yaml
```

The left side of `--set` is `<slot>.<loss option>` from the table of section 2; the right
side is a preset or a component type (`--set head.width_loss=Huber`, built with its
defaults). A preset carries no weight: compose keeps the weight of the loss it replaces and
says so. Real output of the command above:

```text
what compose did
  • head.score_loss: preset `quality-focal` (QualityFocal) in place of SmoothL1. Quality focal loss
    (Generalized Focal Loss): sigmoid cross-entropy to a soft target, focused on the cells whose
    prediction is far off. The weight 15.0 of the replaced loss is kept.
  • proposal.objectness_loss: preset `focal` (FocalLoss) in place of CrossEntropy. Sigmoid focal
    loss (RetinaNet) through torchvision: cross-entropy that turns down elements the network already
    gets right. The weight 1.0 of the replaced loss is kept.
...
config check
  ✓ valid  (hash fa1aebf6f2a3)
```

and the lines it writes into the composed config:

```yaml
model:
  proposal:
    objectness_loss:
      type: FocalLoss
      alpha: 0.25                 # positives x alpha, negatives x (1 - alpha); -1: off
      gamma: 2.0                  # exponent of (1 - p_t)
      weight: 1.0                 # kept from the CrossEntropy it replaces; the penalties differ in scale: re-tune
  head:
    score_loss:
      type: QualityFocal
      beta: 2.0                   # exponent of |target - sigmoid(prediction)|
      weight: 15.0                # kept from the SmoothL1 it replaces; the penalties differ in scale: re-tune
```

The same by hand, in your own config or on any command line that takes `--opts`:

```bash
grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e \
    --opts model.proposal.objectness_loss.type=FocalLoss
```

An override of `type` alone keeps the other keys of the node, so it works only when the new
loss has the options of the old one (here: `weight`). Otherwise replace the whole node in a
YAML file.

Libraries: `pip install "grasppanda[torchvision]"` for `focal`, `pip install "grasppanda[kornia]"`
for `softmax-focal`, `dice`, `lovasz-softmax`, `charbonnier`, `cauchy` and the head preset
`economic-focal`. Without the library the config still loads; building the model raises an
error that names the extra.

### Choosing a loss for a term

* **Objectness (`class`, two classes, many easy points):** `focal` (try `alpha: -1` first:
  table-top objectness is far less imbalanced than detection anchors), `softmax-focal`,
  `class-balanced`, `ohem`, or the set losses `dice` and `lovasz-softmax`.
* **Angle, depth and quality classes (`class`, classes exclude each other):**
  `softmax-focal`, `label-smoothing`, `poly1`, `class-balanced`, `ohem`.
* **Graspness, view scores, the SWAD score grid (`unit`):** `quality-focal`, `varifocal`,
  `ghm-c`, `bce` train a logit; `balanced-l1`, `huber`, `ghm-r`, `charbonnier`, `cauchy`,
  `ohem-smooth-l1` keep the output a value.
* **Widths, tolerances, the scores of OperationHead (`value`):** `huber`, `balanced-l1`,
  `ghm-r`, `charbonnier`, `cauchy`, `ohem-smooth-l1`. Set the knee (`delta`, `beta`, `mu`)
  in the units of the target: SWAD and Economic widths are metres x 10.
* **A width as a distribution (`bins`):** `distribution-focal`, with the head preset
  `decoupled-grid-bins`.

### Wrappers: mining, class balance, multi-task weighting

`TopK`, `ClassBalanced` and `UncertaintyWeighted` take another loss as `loss:`.

```yaml
model:
  proposal:
    objectness_loss:
      type: TopK
      fraction: 0.25              # share of the supervised elements that is kept
      loss: {type: FocalLoss, weight: 1.0, alpha: -1, gamma: 2.0}
```

`uncertainty-weighting` is a preset without a loss of its own: compose wraps the loss that
is in the term, with its weight. Set it on every term that should take part in the balance:

```bash
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \
    --set proposal.objectness_loss=uncertainty-weighting \
    --set proposal.graspness_loss=uncertainty-weighting --set view.loss=uncertainty-weighting \
    --set head.score_loss=uncertainty-weighting --set head.width_loss=uncertainty-weighting
```

```yaml
  head:
    score_loss:
      type: UncertaintyWeighted
      initial_log_variance: -0.6931   # -ln 2: the factor starts at 1
      loss:
        type: SmoothL1
        weight: 15.0
        beta: 1.0
```

A term that is not wrapped keeps its fixed weight. The learned value is the parameter
`<stage>.<term>.log_variance` of the checkpoint. The log-variances are model parameters: an
optimizer that decays all parameters pulls them towards 0, so exclude them from weight
decay. Wrapped terms, and so the total loss, can become negative.

### A head bundle

```bash
grasppanda compose --list head
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set head=swad-focal -o gsnet_swad-focal.yaml
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set head=decoupled-grid-bins -o gsnet_grid-bins.yaml
grasppanda compose economicgrasp/economicgrasp_sunet14_g1b-realsense_10e --set head=economic-focal -o eg_focal.yaml
```

`DecoupledGridHead` in a config, with the options that matter:

```yaml
model:
  head:
    type: DecoupledGridHead
    num_angle: 12
    num_depth: 4
    depth_step: 0.01
    depth: class                  # class | regression (a continuous depth between the bins)
    width: bins                   # regression | class (bin centre) | bins (expectation)
    width_bins: 16                # for class and bins: 16 over 0.1 m is a bin every 6.7 mm
    score_target: log_friction_fixed_range
    friction_range: [0.1, 1.0]    # the range of the GraspNet-1Billion labels
    score_loss: {type: SmoothL1, weight: 15.0, beta: 1.0}
    angle_loss: {type: CrossEntropy, weight: 1.0}
    depth_loss: {type: CrossEntropy, weight: 1.0}
    width_loss: {type: DistributionFocal, weight: 1.0}
```

* **Classification or regression of depth and width** is the pair of options `depth` and
  `width`; the presets `decoupled-grid-class` and `decoupled-grid-regression` are the two
  ends. The kind of `depth_loss` and `width_loss` follows the option, so a loss must be
  changed with it (the presets do).
* The class terms are cross-entropies of order ln 12 and ln 4 at the start, the regressed
  width a distance in metres x 10: watch `loss/angle` and `loss/depth` beside `loss/score`
  and re-tune the weights. The weights of the presets are starting points.
* `width_decode_gain: 1.2` opens the gripper 20 % wider than learned, as the other bundles
  of the family do; 1.0 decodes what was learned.
* A loss and a head combine: `--set head=decoupled-grid --set head.score_loss=varifocal`.

### What a probe shows

`grasppanda probe` trains 20 iterations on 8 real frames and predicts on 2. Figures of the
measured records of this page's options (random initialisation; evidence that the option
trains and decodes, not a result):

| Combination | Parameters | Total loss, first → last | Grasps per frame |
|---|---|---|---|
| `decoupled-grid` under GSNet | 15 321 491 | 17.55 → 5.12 | 510, 475 |
| `decoupled-grid-regression` under GSNet | 15 321 104 | 19.08 → 4.425 | 987, 949 |
| `decoupled-grid-class` under GSNet | 15 329 231 | 19.33 → 7.311 | 1024, 1024 |
| `decoupled-grid-bins` under GSNet | 15 329 231 | 19.32 → 7.462 | 1024, 1024 |

| Loss preset, method, term | Term value, first → last |
|---|---|
| `focal`, GSNet, `proposal.objectness_loss` | 0.3003 → 0.1454 |
| `quality-focal`, GSNet, `head.score_loss` | 3.306 → 0.06559 |
| `balanced-l1`, GSNet, `head.width_loss` | 7.802 → 1.089 |
| `quality-focal`, EconomicGrasp, `proposal.graspness_loss` | 1.052 → 0.3497 |

The other cells, with the refusals, are in the sections "Head bundles" and "Losses" of the
[measured combinations](../advanced_guides/measured_combinations.md).

## 5. Pitfalls

* **A loss in the wrong kind of term is refused, with the alternatives.** Real output of
  `grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set head.width_loss=focal`:

  ```text
  ✗ assigner · head: SWADHead.width_loss is a `value` term (a prediction and a target of equal
    shape, the target an unbounded quantity), but FocalLoss fits `class` or `unit` terms. Losses for
    a `value` term: BalancedL1, GHMR, Huber, L1, MSE, RobustLoss, SmoothL1, TopK,
    UncertaintyWeighted.
  ```

* **The scores of the GraspNet-baseline family are not qualities in [0, 1].** The score and
  view targets of `OperationHead` and `ObjectnessViewStage` are log-friction scores up to
  ln 10, so they are `value` terms and the sigmoid losses are refused there
  (`OperationHead.score_loss is a `value` term ..., but QualityFocal fits `unit` terms`).
* **A logit loss changes what the raw output is.** After `quality-focal`, `varifocal`,
  `ghm-c`, `focal` or `bce` in a `unit` term, the stage output is a logit. The shipped
  stages read it through the sigmoid; a score threshold fitted on the unbounded score of
  `swad` does not carry over to `swad-focal`, whose score lies in `[0, 1]`.
* **Weights are kept, scales are not.** A focal term is much smaller than the Smooth-L1 or
  cross-entropy it replaces, a distribution focal term much larger than a regressed width.
  Compare the logged `loss/...` values of both runs before trusting a weight.
* **Set losses cannot weight seeds.** `dice` and `lovasz-softmax` are refused in the terms
  of Scale-Balanced-Grasp that receive the per-seed weights of its assigner; the message
  names the losses that can.
* **Label smoothing and the extra class of EconomicHead.** With `unmatched_class: true`
  every classifier has one never-supervised output; smoothing gives it target mass. Use
  `decode_argmax: supervised` (as `economic-focal` does) or `unmatched_class: false`.
* **`SWADHead` with `score_target: log_friction_batch_norm`** (the GSNet reference
  behaviour) normalises targets with the extrema of the batch; for a focal-type score loss
  `log_friction_fixed_range` gives a target that means the same in every batch
  (`swad-focal` sets it).
* **GHM with `momentum > 0`** keeps running counts as a buffer of the module; it is saved
  with the checkpoint.
* **Adding a constructor option to a loss used by the shipped configs changes the config
  hash of every method**, and with it every measured record. Label smoothing and class
  weights are therefore a class of their own (`LabelSmoothingCrossEntropy`), and
  `CrossEntropy` keeps its constructor.
* **Head presets do not cross label formats.** `--set head=decoupled-grid` under
  EconomicGrasp:

  ```text
  ✗ assigner · head: CompactLabelAssigner returns 'compact' matched labels, but DecoupledGridHead is
    supervised with 'dense' ones. A head bundle and the assigner that feeds it must agree on the
    label set; the dataset must serve that label set too.
  ```

## 6. Not wrapped

Looked at for this slot and left out. Licences are what the repository states (licence
file, or the repository metadata where no file was found).

| Candidate | Why not (yet) |
|---|---|
| The grasp head of [luyh20/FGC-GraspNet](https://github.com/luyh20/FGC-GraspNet) as code | no licence file in the repository (its metadata says MIT), and it derives from graspnet-baseline, whose licence is non-commercial research only. The idea (separate outputs) is what `DecoupledGridHead` implements, without upstream code. |
| Heads and losses of [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) and its descendants as code | non-commercial research licence. The ported methods are written here from papers and behaviour. |
| [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) (per-point contact head with width bins) | its `License.pdf` is the NVIDIA Source Code License for Contact-GraspNet (non-commercial; see [View](view.md), section 6); not copied. Its top-k selection of point errors is the idea behind `TopK`. |
| [sh8/ZeroGrasp](https://github.com/sh8/ZeroGrasp) | CC BY-NC 4.0. |
| Gradient surgery between loss terms (PCGrad: [WeiChengTseng/Pytorch-PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad), BSD-3-Clause; Jacobian descent: [TorchJD/torchjd](https://github.com/TorchJD/torchjd), MIT) | they need one backward pass per loss term inside the training step; a loss primitive only returns a value. A trainer hook, not an option of this slot. Of the multi-task weightings only the one that is a function of the loss values (uncertainty weighting) is wrapped. |
| Ranking losses ([allegro/allRank](https://github.com/allegro/allRank), Apache-2.0; [teddykoker/torchsort](https://github.com/teddykoker/torchsort), Apache-2.0) | a loss over the order of the grasps of a scene needs the frame boundaries, which a primitive does not see; torchsort has a compiled extension. Not started. |
| Ordinal regression of the quality classes ([Raschka-research-group/coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch), MIT) | needs another output layer (one logit per threshold) in `EconomicHead`, i.e. a new term kind and a head option. Not started. |
| Label and feature distribution smoothing for imbalanced regression ([YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression), MIT) | needs the label statistics of the training set per term. Not started. |
| Rotated-box losses of [open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate) (Apache-2.0) | they compare oriented boxes in the image plane; no seed-based stage predicts one. |
| A decoupled head for depth-axis crops or for compact labels | `DecoupledGridHead` reads dense label grids and crops without a depth axis only, so it is refused under four of the five methods as shipped. |
| Per-seed weights in `SWADHead`, `EconomicHead`, `DecoupledGridHead` | `applies_seed_weights` is false for the three; only `OperationHead` and `ObjectnessViewStage` honour the weights of `ScaleBalancedAssigner`. |
| Loss slots for HGGD | its loss terms are functions inside the detector, not config nodes. |
