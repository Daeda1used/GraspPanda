<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/banner-dark.png">
  <img src="resources/banner-light.png" alt="GraspPanda: a toolbox for 6-DoF grasp detection in cluttered scenes" width="820">
</picture>

**English** | [简体中文](README_zh-CN.md)

[![tests](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml/badge.svg)](https://github.com/Daeda1used/GraspPanda/actions/workflows/test.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-2da44e)](https://daeda1used.github.io/GraspPanda)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#-roadmap)

[Documentation](https://daeda1used.github.io/GraspPanda) ·
[Get started](https://daeda1used.github.io/GraspPanda/get_started/) ·
[Explorer](https://daeda1used.github.io/GraspPanda/explorer/) ·
[Model zoo](#-model-zoo) ·
[Design](https://daeda1used.github.io/GraspPanda/design/ARCHITECTURE/) ·
[Contributing](CONTRIBUTING.md)

<a href="https://daeda1used.github.io/GraspPanda/explorer/"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/explorer-dark.webp">
  <img src="resources/explorer-light.webp" alt="The combination explorer: the six methods in their two families; GraspNet-baseline with a PTv3 scene encoder from another paper, which runs, and with EconomicGrasp's head bundle, refused as incompatible by design with the reason; HGGD with a DINOv2 image encoder; Compare on post-processing under all six methods, then on the scene encoder of the five seed-based methods" width="820">
</picture></a>

</div>

**GraspPanda is an open toolbox for 6-DoF parallel-jaw grasp detection in cluttered scenes.**
It starts from GraspNet-1Billion and also reads GraspClutter6D. Six published methods are
ported and match their reference code stage by stage. The stages of the seed-based skeleton,
HGGD's image encoder, post-processing and the training recipe are slots you can swap with one
line. Each option was composed under every method it fits, those accepted were
probed on real frames, and the few without a record say why.

<details>
<summary>Why GraspPanda</summary>

<br>

Grasp detection in clutter has strong methods and a great benchmark, but comparing them is
harder than it should be.

- **Numbers with and without the collision filter get mixed.** GSNet reports 65.70 AP on
  GraspNet-1Billion (RealSense, seen) without the filter and 67.12 with it. Tables that quote
  one against the other differ by more than many claimed improvements.
- **The same method yields different numbers.** HGGD's README lists 59.36 AP (RealSense,
  seen) as its paper result and 64.45 as its repository result; two later papers that re-ran
  it, [RNGNet](https://arxiv.org/abs/2406.01767) and [GraRe](https://arxiv.org/abs/2608.00946),
  report 58.35 and 63.37.
- **Each method lives in its own repository**, with its own data pipeline, label files and
  compiled operators, and often its own copy of the evaluator.

GraspPanda answers with one pipeline, one pinned evaluator per dataset, and a rule: only
advertise what has been verified.

</details>

## ✨ Features

| | |
|---|---|
| 🧩 **Swap any stage**<br>Encoders, seed proposal, view, grouper, heads, losses, post-processing and the training recipe are slots. | 🔬 **Measured, not assumed**<br>Each option shows under how many methods it runs and learns, and why the others are left out. |
| 🎯 **Faithful ports**<br>Six methods in two families, each equal to its reference stage by stage with converted weights. | 📏 **Pinned evaluators**<br>GraspNet-1Billion and GraspClutter6D, equal to the official tools; collision filter on and off. The GraspNet-1Billion one equals `graspnetAPI` value for value on 1,592 recorded frames ([test](tests/integration/test_eval_parity.py)). |
| 🛡️ **Wrong swaps stop early**<br>Most wrong combinations are refused before anything is built, with the conflict and the fix; a short probe catches the rest. | 📦 **Small core**<br>`torch`, `numpy`, `scipy`, `pyyaml`, `pillow` and `tqdm`; sparse convolution and model libraries are extras. |

## 🚀 Quick start

Install PyTorch for your CUDA version, then GraspPanda with the extras the examples below
use, and check it with a toy detector:

```bash
git clone https://github.com/Daeda1used/GraspPanda.git && cd GraspPanda
pip install -e ".[spconv-cu118,timm]"   # match spconv to your CUDA
grasppanda env                                                     # what is installed
grasppanda train projects/example_project/configs/toy_train.yaml   # a toy detector, trained on the CPU in under a minute
```

Swap a stage with one line: a scene encoder from another paper under GraspNet-baseline, and a
pretrained image encoder under HGGD.

```bash
grasppanda compose graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e \
    --set backbone=ptv3 -o baseline_ptv3.yaml
grasppanda compose hggd/hggd_r34w8_g1b-realsense_15e \
    --set image_encoder=dinov2-small-reg-depth-stem -o hggd_dinov2.yaml
# each probe trains briefly on GraspNet-1Billion if it is set up; otherwise a synthetic check: `wired`
grasppanda probe baseline_ptv3.yaml
grasppanda probe hggd_dinov2.yaml
```

`compose` checks the stage contracts and writes the config, inserting an adapter where one is
needed. PTv3 runs under 5/5 seed-based methods; under 4/5 it trains like shipped, on a par
with the method's own encoder (worse under FineGrasp). The DINOv2 encoder runs under HGGD and
trains on a par with HGGD's own. A probe shows that a combination trains, not how good it is;
a swapped model has no weights until you train it.

**Next:** [run a method](https://daeda1used.github.io/GraspPanda/get_started/run_a_method/) from its released weights, first grasps in minutes ·
[swap a component](https://daeda1used.github.io/GraspPanda/get_started/swap/) ·
[open the explorer](https://daeda1used.github.io/GraspPanda/explorer/)

## 🧩 What's inside

| Method | Venue | Family | Reads |
|---|---|---|---|
| [GraspNet-baseline](https://daeda1used.github.io/GraspPanda/methods/graspnet_baseline/) | CVPR 2020 | seed-based skeleton | depth |
| [GSNet](https://daeda1used.github.io/GraspPanda/methods/gsnet/) | ICCV 2021 | seed-based skeleton | depth |
| [Scale-Balanced-Grasp](https://daeda1used.github.io/GraspPanda/methods/scale_balanced_grasp/) | CoRL 2022 | seed-based skeleton | depth |
| [EconomicGrasp](https://daeda1used.github.io/GraspPanda/methods/economicgrasp/) | ECCV 2024 | seed-based skeleton | depth |
| [FineGrasp](https://daeda1used.github.io/GraspPanda/methods/finegrasp/) | arXiv 2025 | seed-based skeleton | depth |
| [HGGD](https://daeda1used.github.io/GraspPanda/methods/hggd/) | RA-L 2023 | own family | RGB-D |

**Datasets:** GraspNet-1Billion and GraspClutter6D, each with a reader, label assets and an
evaluator equal to the official one. **Slots**, and the methods each applies to:

| Slots | Applies to |
|---|---|
| post-processing, training recipe | All 6 methods |
| scene encoder, seed proposal, view, assigner, grouper, head bundle, loss | Seed-based · 5 methods |
| image encoder | HGGD only |

<details>
<summary>Every slot, and how a detector is put together</summary>

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="resources/pipeline-dark.png">
  <img src="resources/pipeline.png" alt="GraspNet-1Billion and GraspClutter6D feed two detector families. The seed-based skeleton has the slots scene encoder, seed proposal, view, assigner, grouper, head bundle, and losses; HGGD has its own family with an image encoder slot and fixed heat-map and region stages. Post-processing and the training recipe are shared by all methods." width="820">
</picture>

| Slot | `--set` key | What goes in |
|---|---|---|
| [Image encoder](https://daeda1used.github.io/GraspPanda/slots/image_encoders/) | `image_encoder` | HGGD's own encoder and bridges to timm, transformers and torchvision |
| [Scene encoder](https://daeda1used.github.io/GraspPanda/slots/scene_encoders/) | `backbone` | point-based, sparse-voxel and transformer encoders, with released weights where they fit |
| [Seed proposal](https://daeda1used.github.io/GraspPanda/slots/proposal/) | `proposal` | scored or plain seeds, with their samplers |
| [View](https://daeda1used.github.io/GraspPanda/slots/view/) | `view` | view lattices of any size, regressed directions, full rotations |
| [Assigner](https://daeda1used.github.io/GraspPanda/slots/head_loss/) | `assigner` | how labels are matched to the predicted grasps |
| [Grouper](https://daeda1used.github.io/GraspPanda/slots/grouper/) | `grouper` | crops among the seeds or of the scene: shapes, encoders, poolings |
| [Head bundle](https://daeda1used.github.io/GraspPanda/slots/head_loss/) | `head` | the heads of the ported methods and a decoupled grid head |
| [Loss](https://daeda1used.github.io/GraspPanda/slots/losses/) | `<stage>.<term>_loss`, e.g. `head.score_loss` | focal, quality, robust and set losses, set per loss term |
| [Post-processing](https://daeda1used.github.io/GraspPanda/slots/post/) | `post` | collision, table and workspace filters, NMS, selection |
| [Training recipe](https://daeda1used.github.io/GraspPanda/slots/recipe/) | `optimizer`, `schedule`, `hooks`, `augment`, `recipe` | optimizers, schedules, hooks, augmentation; presets that tune the scene encoder apply to the seed-based methods only |

Where each option runs, under which method and why not elsewhere: the
[explorer](https://daeda1used.github.io/GraspPanda/explorer/) and the
[evidence pages](https://daeda1used.github.io/GraspPanda/evidence/).

</details>

## 🏆 Model zoo

Targets: what the authors report on GraspNet-1Billion (RealSense, AP on seen / similar /
novel), and in the last column the model trained here, where there is one.

| Method | Venue | Collision filter | Reported AP (seen / similar / novel) | Reproduced here |
|---|---|:---:|---|:---:|
| GraspNet-baseline | CVPR 2020 | not stated | 27.56 / 26.11 / 10.55 (paper) | — |
| GraspNet-baseline | CVPR 2020 | on | 47.47 / 42.27 / 16.61 (repository) | — |
| GSNet | ICCV 2021 | off | 65.70 / 53.75 / 23.98 | 68.83 / 60.35 / 25.20 |
| GSNet | ICCV 2021 | on | 67.12 / 54.81 / 24.31 | 71.23 / 62.29 / 26.42 |
| Scale-Balanced-Grasp | CoRL 2022 | off | 58.95 / 52.97 / 22.63 | — |
| Scale-Balanced-Grasp | CoRL 2022 | on | 63.83 / 58.46 / 24.63 | — |
| EconomicGrasp | ECCV 2024 | off | 68.21 / 61.19 / 25.48 | — |
| FineGrasp | arXiv 2025 | off | 71.67 / 62.83 / 27.40 | — |
| FineGrasp | arXiv 2025 | on | 73.71 / 64.56 / 28.14 | — |
| HGGD | RA-L 2023 | not stated | 59.36 / 51.20 / 22.17 (paper) · 64.45 / 53.59 / 24.59 (repository) | — |

A method enters the zoo when it is trained here within 1 AP of the reported number, or with
the gap explained. None has passed so far, so GraspPanda publishes no trained weights yet;
the methods run from their released weights
([Run a method](https://daeda1used.github.io/GraspPanda/get_started/run_a_method/)).

<details>
<summary>Notes on these numbers</summary>

<br>

- **GSNet** was trained here from scratch with the recipe of its config: one run, one seed.
  It scores above the paper by more than the gate allows, and why is not known. What is
  known: the port equals the reference implementation stage by stage, a model trained with
  the reference code (on a modified label set) is also above the paper, and the authors'
  original code is not public
  ([details](https://daeda1used.github.io/GraspPanda/model_zoo/#trained-and-evaluated-here)).
  Its weights are not published yet.
- **The authors' released weights** of the other methods were scored on the whole test splits
  here, with and without the collision filter. Those `author-ckpt` numbers are in the
  [model zoo](https://daeda1used.github.io/GraspPanda/model_zoo/), each backed by a committed
  result record. They are not reproductions.
- **HGGD** publishes two sets of numbers and does not state the collision filter; its
  released test script filters collisions before it saves the grasps.
- **Admission rule:** config, weights with sha256, training log and a result record of the
  pinned evaluator, within 1 AP on each split or with the gap documented.

</details>

## 📰 News

- **2026-10** — Swaps that train were also trained beside the component the method ships, to see whether they [train like shipped](https://daeda1used.github.io/GraspPanda/advanced_guides/learns_like_reference/).
- **2026-10** — The documentation is rebuilt around tasks: run a method, swap a component, read the evidence.

<details>
<summary>Older news</summary>

<br>

- **2026-10** — Refusals say why: each one names its kind, and `compose` lifts the ones an adapter or a setting can fix.
- **2026-10** — The [combination explorer](https://daeda1used.github.io/GraspPanda/explorer/): pick a method, swap a slot, read the measured record and the reason of a refusal. `grasppanda explore` serves it locally.
- **2026-10** — Options for every slot, taken from published code and model libraries, each probed under the methods it fits.
- **2026-10** — `grasppanda compose` builds a combination from a method config and presets; `grasppanda probe` trains it briefly on real frames.
- **2026-10** — First public version of the rebuild. Releases up to v0.56 are an earlier, unrelated code base and stay under their tags and on the `legacy` branch.

</details>

## 🧭 Roadmap

<details>
<summary>Milestones</summary>

<br>

- [x] **M0 — Core.** Config system, registries, data contracts, reference operators, CLI, CI, docs, brand.
- [ ] **M1 — GraspNet-1Billion, its evaluator, a first method.** ✅ Evaluator equal to the official one. ✅ GSNet equal to its reference stage by stage, trained here and scored on the whole test split. 🚧 Its gap to the paper is not explained; its weights are not published.
- [ ] **M2 — EconomicGrasp, and HGGD as a second family.** ✅ EconomicGrasp and HGGD equal to their reference stage by stage with the authors' weights. 🚧 Training here.
- [ ] **M3 — The other seed-based methods.** ✅ GraspNet-baseline, Scale-Balanced-Grasp and FineGrasp equal to their reference. 📅 A controlled comparison: same encoder, budget, inputs and evaluator.
- [ ] **M4 — GraspClutter6D.** ✅ Reader, evaluator equal to the official one, label assets, transfer configs. 📅 Training on GraspClutter6D; Contact-GraspNet.
- [ ] **M5 — Release.** PyPI, model zoo, documentation site.

</details>

## 📝 Citation

<details>
<summary>BibTeX</summary>

```bibtex
@software{grasppanda,
  title  = {GraspPanda: A Toolbox for 6-DoF Grasp Detection in Clutter},
  author = {{GraspPanda Contributors}},
  year   = {2026},
  url    = {https://github.com/Daeda1used/GraspPanda}
}
```

Please also cite the methods and datasets you use; each method page lists its reference.

</details>

## 🙏 Acknowledgements

GraspPanda borrows ideas from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg),
[MMEngine](https://github.com/open-mmlab/mmengine),
[Detectron2](https://github.com/facebookresearch/detectron2) and
[Pointcept](https://github.com/Pointcept/Pointcept), and builds on the
[GraspNet-1Billion](https://graspnet.net) and
[GraspClutter6D](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) benchmarks and
the authors of the methods it ports. Methods and evaluators are written from the papers and
from the behaviour of the reference code; no file of those repositories is copied. Code
adapted from MIT, BSD or Apache-2.0 projects is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); [NOTICE](NOTICE) says how the datasets,
upstream code and released weights are licensed.

🐼 *The panda is the animal, not the robot arm: GraspPanda is not tied to any robot.*

## 📄 License

[Apache License 2.0](LICENSE). Datasets, upstream repositories and released weights keep
their own licences, stated per method.
