<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/banner-dark.png">
  <img src="resources/banner-light.png" alt="GraspPanda: the toolbox for 6-DoF grasp detection in clutter" width="820">
</picture>

**English** | [简体中文](README_zh-CN.md)

[![tests](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml/badge.svg)](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-2da44e)](https://daeda1used.github.io/GraspPanda)
[![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-lightgrey)](#-installation-and-quick-start)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c)](https://pytorch.org)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#-roadmap)

[Documentation](https://daeda1used.github.io/GraspPanda) ·
[Quick start](#-installation-and-quick-start) ·
[Model zoo](#-model-zoo) ·
[Roadmap](#-roadmap) ·
[Design](docs/design/ARCHITECTURE.md) ·
[Contributing](CONTRIBUTING.md)

</div>

**GraspPanda is to 6-DoF grasp detection in clutter what MMDetection is to object detection:**
one codebase where every stage of the pipeline can be swapped from a YAML file, where
published methods are ported and checked against their reference implementations stage by
stage, and where every number is produced by the same pinned evaluation.

🐼 *The panda is the animal, not the robot arm. GraspPanda is not tied to any robot.*

> [!NOTE]
> **Alpha.** Working today: the config system, both benchmark readers, both evaluators
> (equal to the official tools stage by stage), six methods as swappable components
> (GSNet, EconomicGrasp, FineGrasp, HGGD, GraspNet-baseline, Scale-Balanced-Grasp; each equal to its
> reference implementation stage by stage with converted weights), options for every slot
> of the pipeline ([Slots](docs/slots/index.md)), and the `train` / `test` / `infer` / `demo` /
> `analyze` / `sweep` / `compose` / `probe` / `explore` commands. The model zoo is still empty on
> purpose: a method enters only after it has been trained *and* evaluated here, and the
> first from-scratch reproduction is running. No accuracy is claimed for an option that is
> not part of a ported method.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/showcase-dark.svg">
  <img src="resources/showcase-light.svg" alt="A terminal composes GSNet with another scene encoder and grouper, a wrong head swap is refused with the conflict and the fix, and a probe on real frames ends with the verdict runs; above it the slots of the pipeline flip to the chosen options" width="830">
</picture>

*Real output, lines dropped, none invented. A combination is one command; a wrong one is refused with the fix before anything is built; a probe says whether it trains. No accuracy is claimed by a probe.*

**[Try combinations in the browser →](https://daeda1used.github.io/GraspPanda/explorer/)** every slot, every option: the measured verdict of 980 recorded combinations (740 probed on real frames by a probe of their own), and for any two swaps the answer of the real `compose`. Or locally, checked by your own installation: `grasppanda explore`.

<img src="resources/demo.webp" alt="Predicted grasps on a cluttered GraspNet-1Billion test scene, rendered by GraspPanda" width="720">

*GSNet predictions on a GraspNet-1Billion test frame (top 40 after the collision filter), computed by GraspPanda's implementation with weights trained by the reference code. Scene image: GraspNet-1Billion, CC BY-NC-SA 4.0.*
</div>

## 📰 News

- **2026-10** — First public version of the rebuild. Six methods as interchangeable components, each equal to its reference implementation stage by stage; eleven scene encoders; `grasppanda components`, `demo` and `analyze`; GraspClutter6D end to end. Releases up to v0.56 are an earlier, unrelated code base and stay available under their tags and on the `legacy` branch.
- **2026-10** — `grasppanda compose` builds a combination from a method config and a preset, `grasppanda probe` trains it for a few iterations on real frames, and a generated page shows the measured result of every recorded combination (208 at that time; which component works under which method, and what the contract rules refuse).
- **2026-10** — [Options for every slot](docs/slots/index.md), taken from published code and model libraries: 248 in all (image and scene encoders, seed proposal, view, grouper, heads and losses, post-processing, training recipe). Each has a row and a command on its slot page and a measured verdict per method; none has been trained and evaluated yet, so none carries an accuracy.
- **2026-10** — The [combination explorer](https://daeda1used.github.io/GraspPanda/explorer/): pick a method, swap a slot, read the measured verdict, the command and the reason of a refusal; two swaps without a record get the stored answer of `compose`; `grasppanda explore` serves it locally and checks any selection with the real `compose`.
- **2026-10** — More combinations that used to pass the check and fail in the first iteration are refused with the fix: a head or a view stage on another grid than the labels, a seed count the encoder does not return, a point augmentation next to images, a collision filter without the full cloud, mixed precision under a detector not written for it. A probe on real frames now reports a part of the model that received no gradient.
- **2026-10** — Two more families of wrong combinations are refused when a config is checked instead of failing later: stages on a data recipe that does not serve their labels, and an encoder whose width a grouper cannot take (`compose` then fits the encoder or puts the projected grouper in place, and says so).

## 🤔 Why GraspPanda

Grasp detection in clutter has strong methods and a great benchmark, but comparing them is harder than it should be.

- **Numbers with and without collision filtering get mixed.** GSNet reports 65.70 AP on GraspNet-1Billion (RealSense, seen) without the filter and 67.12 with it. Tables that quote one against the other differ by more than many claimed improvements.
- **The same method yields different numbers.** HGGD's own README lists 59.36 AP (RealSense, seen) as its paper result and 64.45 as its repository result; two later papers that re-ran it report 58.35 and 63.37.
- **Each method lives in its own repository**, with its own data pipeline, its own label files, its own compiled operators and often its own copy of the evaluator.

GraspPanda answers with one pipeline, one evaluator, and a rule: **only advertise what has been verified**.

## ✨ Features

| | |
|---|---|
| 🎯 **Certified reproductions** ⏳ | A method enters the model zoo only after it is trained and evaluated in this toolbox, within 1 AP of the published number or with the gap explained. Every number is labelled `reported`, `author-ckpt` or `reproduced`. |
| 🧩 **Swappable components** | Backbone, seed proposal, approach view, grouping, head bundle, loss and post-processing are config slots. An ablation is one base config plus one override. |
| 🛡️ **Typed contracts, strict configs** | Unknown keys, wrong types and incompatible swaps are errors with a suggested fix, raised before anything is built. Nothing is silently interpolated, projected or squeezed. |
| 📏 **One pinned evaluation** | A clean-room re-implementation of the official GraspNet-1Billion and GraspClutter6D protocols that equals the official tools stage by stage (1,592 recorded frames, zero mismatches), about 40x faster and with a ninth of the memory. Collision filter on *and* off are always reported. |
| 🔍 **Honest inputs** | `predict()` never sees ground truth. Inputs derived from it, such as the ground-truth workspace crop most pipelines use at test time, are declared in the config and stamped on results. |
| 📦 **Small core** | `torch`, `numpy`, `pyyaml`. Sparse convolution and other heavy pieces are optional extras; operators have pure-PyTorch reference implementations. |
| 🧾 **Reproducible runs** | Each run snapshots its resolved config, overrides, code, git state and environment. |

⏳ marks what is still in progress (the first from-scratch reproduction is running); everything else works today.

### A config is the whole experiment

```yaml
_base_:
  - ../../../configs/_base_/default_runtime.yaml
  - ../../../configs/_base_/schedules/gsnet_10e.yaml

custom_imports: [../toy_components.py]      # out-of-tree code, no fork needed

model:
  type: ToyDetector
  backbone: {type: ToyBackbone, out_channels: 64}
  grouper:  {type: ToyGrouper, radius: 0.05, nsample: 16}
  head:     {type: ToyHead, num_angle: 12, num_depth: 4}
```

```bash
# swap a component: kwargs of the old type are dropped, not silently merged
grasppanda print-config <config> --opts optimizer.type=SGD --check

# see everything a run would use, defaults included
grasppanda print-config <config> --defaults
```

### One pipeline, typed contracts

<div align="center">
<img src="resources/pipeline.png" alt="Pipeline slots: backbone, proposal, view, grouper and head bundle, with the data contract each one emits" width="820">
</div>

### Every slot has options

An option is a name after `--set <slot>=`: a component with settings for table-top depth
clouds. Each slot page starts with the task of the slot and its input and output, then
gives one row and one command per option. [The index](docs/slots/index.md) classifies
every option from the measured records as universal, most, some or specific.

| Slot | Options | What they are |
|---|---|---|
| [Image encoder](docs/slots/image_encoders.md) | 38 | HGGD's own encoder and bridges to timm, transformers and torchvision; point painting for the seed-based methods |
| [Scene encoder](docs/slots/scene_encoders.md) | 47 | point-based networks, sparse-voxel networks, transformers and pretrained 3-D encoders, with released weights where they fit |
| [Seed proposal](docs/slots/proposal.md) | 24 | scored or plain seeds, with eleven samplers |
| [View](docs/slots/view.md) | 21 | view lattices of any size, regressed directions, full rotations |
| [Grouper](docs/slots/grouper.md) | 32 | crops among the seeds or of the scene: shapes, encoders, poolings, RoI pooling, graph layers |
| [Head bundle and assigner](docs/slots/head_loss.md) | 10 + 4 | the heads of the ported methods, a decoupled grid head, and the label assigners they go with |
| [Loss](docs/slots/head_loss.md) | 20 | focal, quality, robust and set losses, mining, class balance, uncertainty weighting, per loss term |
| [Post-processing](docs/slots/post.md) | 15 | collision, table, workspace and gripper filters, NMS, selection, change of frame |
| [Training recipe](docs/slots/recipe.md) | 37 | optimizers, schedules, hooks, augmentation (one of them, mixed precision, is refused for every shipped detector today) |

## 🗺️ Overview

✅ available · 🚧 in progress · 📅 planned

| Area | Status | What |
|---|---|---|
| **Core** | ✅ | YAML configs with inheritance and strict three-layer validation · registries with lazy auto-import · build-time injection · run snapshots |
| **Data contracts** | ✅ | `Observation` / `Supervision` · `PointSet` / `PointFeatures` · `SeedSet` · `ApproachSet` · `GroupFeatures` · `GraspGroup` |
| **Operators** | ✅ | furthest point sampling · ball and cylinder query · kNN · interpolation · grasp NMS · voxelisation · space-filling curves · view templates (pure PyTorch reference, CPU-testable) · optional compiled CPU and CUDA kernels for the four hot operators, checked against the reference on the device they run on ([details](docs/advanced_guides/operators.md)) |
| **CLI** | ✅ `train`, `test`, `infer`, `demo`, `analyze`, `components`, `compose`, `probe`, `explore`, `sweep`, `prepare`, `print-config`, `env` · 📅 `profile` | |
| **Datasets** | ✅ GraspNet-1Billion and GraspClutter6D readers, split manifests, content-hashed label assets (equal to the reference preprocessing on recorded frames) | |
| **Evaluation** | ✅ GraspNet-1Billion and GraspClutter6D evaluators, result records with coverage and provenance | |
| **Methods** | ✅ GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline, Scale-Balanced-Grasp (one seed-based skeleton) and HGGD (a family of its own): each equal to its reference stage by stage with converted weights · 🚧 full-split evaluation and from-scratch reproduction · 📅 Contact-GraspNet | |
| **Scene encoders** | ✅ compared with the official code where weights are released: sparse U-Net (spconv), SpUNet (MinkUNet-34C layout), OA-CNNs, PointNet++ (two forms), PointNeXt, PTv3, PTv3 with Concerto and Utonia weights, LitePT ([details](docs/model_zoo_backbones.md)) · ✅ usable, not compared: the other encoders of the [slot pages](docs/slots/scene_encoders.md) · every combination of encoder and method either trains or is refused at config time with the fix ([swap matrix](docs/advanced_guides/swapping.md)) | |

## 🏆 Model zoo

Nothing has been certified yet, so this table lists **targets**: numbers reported by the
original authors on GraspNet-1Billion (RealSense, AP on seen / similar / novel). The last
column fills in as each method passes the gate.

What has been measured is how the weights released by the authors score here: five
methods on the whole GraspNet-1Billion test split and four of them on GraspClutter6D,
with and without collision filter, the reported number beside each. Those `author-ckpt`
numbers are in the tables of [docs/model_zoo.md](docs/model_zoo.md), each backed by a
committed result record; they are not reproductions, so the last column stays empty.

| Method | Venue | Collision filter | Reported AP (seen / similar / novel) | Reproduced here |
|---|---|:---:|---|:---:|
| GraspNet-baseline | CVPR 2020 | on | 47.47 / 42.27 / 16.61 ("In repo" row of its README) | — |
| GSNet | ICCV 2021 | off | 65.70 / 53.75 / 23.98 | — |
| GSNet | ICCV 2021 | on | 67.12 / 54.81 / 24.31 | — |
| EconomicGrasp | ECCV 2024 | off | 68.21 / 61.19 / 25.48 | — |
| HGGD | RA-L 2023 | see note | 59.36 / 51.20 / 22.17 ("In paper" row of its README) · 64.45 / 53.59 / 24.59 ("In repo" row; also the arXiv version) | — |
| Scale-Balanced-Grasp | CoRL 2022 | off | 58.95 / 52.97 / 22.63 | — |
| Scale-Balanced-Grasp | CoRL 2022 | on | 63.83 / 58.46 / 24.63 | — |
| FineGrasp | arXiv 2025 | off | 71.67 / 62.83 / 27.40 | — |
| FineGrasp | arXiv 2025 | on | 73.71 / 64.56 / 28.14 | — |

Note: HGGD publishes two sets of numbers and does not state the collision filter; its released test script filters collisions before dumping. Which settings separate the numbers is one of the things this toolbox is built to pin down.

Admission rule: config + weights (sha256) + training log + result record from the pinned
evaluator, with `|reproduced − reported| ≤ 1 AP` on each split or a documented gap.

## 🚀 Installation and quick start

Install PyTorch for your CUDA version first, then:

```bash
git clone https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
pip install -e .
```

Three commands to see the framework work, no dataset required:

```bash
grasppanda env                                                              # environment report
grasppanda train projects/example_project/configs/toy_train.yaml            # trains a toy detector on CPU in seconds
grasppanda print-config projects/example_project/configs/toy_detector.yaml \
    --opts model.grouper.type=ToyDepthGrouper                               # rejected, with the reason
```

### First grasps in five minutes

No dataset is needed for a first picture. The zoo holds no weights yet and the authors'
checkpoints are under their own licences, so no grasp checkpoint is downloaded for you.
(The one exception to "nothing is fetched": an image-encoder preset with `pretrained: true`
lets `timm`, `transformers` or `torchvision` download ImageNet-style weights on first use and
prints one line with their licence; see [image encoders](docs/slots/image_encoders.md).)

**1. Check the installation** (no download, CPU, seconds): a toy detector, trained on the spot.

```bash
grasppanda train projects/example_project/configs/toy_train.yaml --opts runtime.run_id=toy
grasppanda demo  projects/example_project/configs/toy_train.yaml \
    --weights work_dirs/toy_train/toy/checkpoints/best.pth --example
```

`--example` is a synthetic table-top scene that ships with the package. The toy's grasps
mean nothing; the command shows that everything runs.

**2. A real method.** Download an author checkpoint (links and licences in
`configs/<method>/README.md`), convert it once, run it:

```bash
python -m tools.convert_checkpoint checkpoint-rs.tar baseline_realsense.pth \
    --source-format graspnet-baseline --camera realsense
grasppanda demo graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --weights baseline_realsense.pth --example --collision-filter
```

It prints the best grasps and writes `grasps.npy` (GraspNet 17-column layout, camera
frame, metres), `grasps.png` and `grasps.gif`.

**3. Your camera.** A depth image and its intrinsics are enough for the point-cloud
methods; HGGD also reads the registered colour image:

```bash
grasppanda demo <config> --weights <file> --depth depth.png --intrinsics fx,fy,cx,cy \
    --workspace -0.25 -0.2 0.35 0.25 0.2 0.8
```

The models were trained on GraspNet-1Billion (RealSense D435 and Azure Kinect, table-top
scenes); results on other cameras have not been measured. The same pipeline is available
in Python as `GraspInferencer`, and `grasppanda analyze` says where the AP of an
evaluation comes from: see [Inference and visualisation](docs/user_guides/inference.md).

### Try a combination

```bash
grasppanda compose --list grouper                            # the options of a slot, one line each
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=ptv3 -o gsnet_ptv3.yaml
grasppanda probe gsnet_ptv3.yaml                             # 20 iterations on real frames, then predict
grasppanda train gsnet_ptv3.yaml

# two slots at once, with options added in 2026-10 (recorded: runs)
grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=point-transformer-v2 \
    --set grouper=cylinder-vector-attention -o gsnet_ptv2_attention.yaml
grasppanda explore                                           # the same in a browser, checked by compose
```

`compose` writes a checked config from a method config and presets, inserts an adapter
where the stage contracts ask for one and says so; a combination the contracts refuse is
reported with the conflict and the fix, and a composed config can be composed again.
`probe` ends with one verdict (`runs`, `runs-unstable`, `fails`, `refused`). Which
component works under which method has been measured on real frames, not inferred: see
[Slots](docs/slots/index.md), [Try a combination](docs/user_guides/combinations.md) and
[Measured combinations](docs/advanced_guides/measured_combinations.md). No accuracy is
claimed for a combination until it has been trained and evaluated.

### Benchmarks

With GraspNet-1Billion on disk (`export GRASPPANDA_GRASPNET1B_ROOT=...`, and the sparse
backend: `pip install -e ".[spconv-cu118]"`):

```bash
grasppanda prepare graspnet1b --check                        # verify the dataset tree
grasppanda prepare graspnet1b --labels                       # memory-mapped grasp labels
grasppanda prepare graspnet1b --graspness --camera realsense # point-wise graspness labels
grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e
grasppanda test  gsnet/gsnet_sunet14_g1b-realsense_10e --weights <checkpoint>
```

A model trained on GraspNet-1Billion is scored on GraspClutter6D with one command (see
[Cross-dataset evaluation](docs/user_guides/cross_dataset.md)):

```bash
grasppanda test gsnet/gsnet_sunet14_g1b-realsense-to-gc6d-d435_10e --weights <checkpoint>
```

`grasppanda components` lists what can go into each slot. Swapping the scene encoder is one
override; an incompatible one is rejected with the reason:

```bash
grasppanda print-config gsnet/gsnet_sunet14_g1b-realsense_10e --check \
    --opts "model.backbone={type: PTv3, in_channels: 3, grid_size: 0.005}"
```

Build a detector from a config and run it:

```python
import torch
from grasppanda.config import Config
from grasppanda.registry import DETECTORS
from grasppanda.structures import Observation, PointSet, SampleKey

config = Config.load("projects/example_project/configs/toy_detector.yaml")
model = DETECTORS.build(config["model"]).eval()

observation = Observation(
    keys=[SampleKey("demo", 0, "none", 0)],
    points=PointSet.from_dense(torch.rand(1, 2048, 3)),
)
grasps = model.predict(observation)[0]     # GraspGroup: translation, rotation, width, depth, score
array = grasps.to_graspnet_array()         # (64, 17), the GraspNet interchange layout
```

The example detector is a wiring template: it is untrained and its grasps are arbitrary.
`pip install grasppanda` arrives with the first release.

## 🧭 Roadmap

- [x] **M0 — Core.** Config system, registries, data contracts, reference operators, CLI, CI, docs skeleton, brand.
- [ ] **M1 — GraspNet-1Billion, evaluator, GSNet.** ✅ Evaluator agrees with the official API value for value. ✅ GSNet equals the reference implementation stage by stage with converted weights. 🚧 Training from scratch.
- [ ] **M2 — EconomicGrasp and HGGD.** A second method of the same family, and one that deliberately does not fit it. ✅ Both equal their reference stage by stage with the authors' weights. 🚧 Full-split evaluation and training.
- [ ] **M3 — Scale-Balanced-Grasp, GraspNet-baseline, FineGrasp.** First controlled comparison table: same backbone, same budget, same inputs, same evaluator. ✅ All three equal their reference stage by stage with the authors' weights. 📅 The table.
- [ ] **M4 — GraspClutter6D and Contact-GraspNet.** ✅ Reader, evaluator (equal to the official tool on 254 recorded frames), label assets, transfer configs for five methods (checked on a 312-frame subset). 📅 Full-split transfer table, training on GraspClutter6D, Contact-GraspNet.
- [ ] **M5 — Release.** PyPI, model zoo, documentation site.

## 📚 Tutorials

- [Configuration guide](docs/user_guides/config.md): inheritance, the merge rules, overrides, validation.
- [Slots](docs/slots/index.md): the task of every stage, its options and which fits which method.
- [Try a combination](docs/user_guides/combinations.md): catalogue, compose, probe, train.
- [Add a component](docs/advanced_guides/add_component.md): register a backbone or a head bundle without forking.
- [Data contracts](docs/advanced_guides/data_flow.md): what each stage receives and returns.
- [Architecture](docs/design/ARCHITECTURE.md): the full design, with the evidence behind each decision.

## 📝 Citation

If GraspPanda helps your research, please cite it:

```bibtex
@software{grasppanda,
  title  = {GraspPanda: A Toolbox for 6-DoF Grasp Detection in Clutter},
  author = {{GraspPanda Contributors}},
  year   = {2026},
  url    = {https://github.com/Daeda1used/GraspPanda}
}
```

Please also cite the methods and datasets you use; each method page lists its reference.

## 🙏 Acknowledgements

GraspPanda borrows ideas from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg),
[MMEngine](https://github.com/open-mmlab/mmengine),
[Detectron2](https://github.com/facebookresearch/detectron2) and
[Pointcept](https://github.com/Pointcept/Pointcept), and builds on the
[GraspNet-1Billion](https://graspnet.net) and
[GraspClutter6D](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) benchmarks
and on the authors of the methods it reproduces. The methods and the evaluators are written
from the papers and from behavioural study of the reference code; no source file of those
repositories is copied. Some other components are adapted from code under the MIT, BSD or
Apache-2.0 licence, each listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with
repository, commit and licence, and components that bridge to an optional library contain
none of its code ([CONTRIBUTING.md](CONTRIBUTING.md), [NOTICE](NOTICE)).

## 📄 License

GraspPanda is released under the [Apache License 2.0](LICENSE). Datasets, upstream
repositories and released weights keep their own licences, recorded per method.
