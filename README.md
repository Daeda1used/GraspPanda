<div align="center">

# GraspPanda

### An All-in-One Research Toolbox for Visual Grasping

**From published methods to your next grasping model.**

[![License: MIT](https://img.shields.io/badge/license-MIT-167568)](LICENSE)
[![Datasets](https://img.shields.io/badge/datasets-4-365c92)](docs/DATASETS.md)
[![Python 3.11](https://img.shields.io/badge/python-3.11-365c92)](docs/INSTALL.md)
[![Source checks](https://github.com/Daeda1used/GraspPanda/actions/workflows/ci.yml/badge.svg)](https://github.com/Daeda1used/GraspPanda/actions/workflows/ci.yml)

[Get started](#quick-start) · [Methods & papers](docs/METHODS.md) · [Compose modules](docs/MODULES.md) · [Data & weights](docs/DOWNLOADS.md)

</div>

![Visual observations, configurable grasp models and reproducible experiments in one shared runtime.](docs/assets/overview.svg)

GraspPanda brings **parallel-jaw and dexterous grasping** into one configurable research workspace. Reproduce a method, replace compatible components, and follow an experiment from **configuration to training, prediction and comparison**—in the browser or from the CLI.

**One shared runtime · Dataset-specific protocols · Composable models.**

| Research setting | Dataset | Representative workflows |
|---|---|---|
| Real clutter, parallel-jaw grasping | **GraspNet-1B** | Graspness, HGGD, Scale-Balanced-Grasp, EconomicGrasp and more |
| Real clutter across cameras | **GraspClutter6D** | Native Contact-GraspNet and cross-dataset transfer |
| Synthetic reconstruction and grasping | **ZeroGrasp-11B** | ZeroGrasp, with shared RGB encoders |
| Dexterous hands in clutter | **DexGraspNet 2.0** | Native diffusion, ISAGrasp and GraspTTA baselines |

See [dataset protocols](docs/DATASETS.md) and the [method catalogue](docs/METHODS.md) for exact operations and evaluation scope.

## One toolbox. Every stage of the experiment.

| Layer | Explore |
|---|---|
| **Grasp methods** | GraspNet Baseline, Graspness, HGGD, Scale-Balanced-Grasp, EconomicGrasp, RegionNormalizedGrasp, FineGrasp and more |
| **Visual encoders** | PointNet families, sparse CNNs, point transformers, point Mamba models, DINO, VMamba, EfficientViT and FastViT |
| **Model components** | Local grouping, sampling, seed interaction, score heads and pose refinement; stage and layer controls for registered encoders |
| **Training controls** | Loss formulations, augmentation, optimizers, schedules, checkpoint transfer and resume |
| **Experiment workflow** | Verified starter data, editable presets, sweeps, queued runs, grasp previews, 3D hand geometry, loss curves and checkpoint reuse |

Compatibility checks cover **geometry, sample indices and supervision** as well as tensor shapes. Single-view, fused-view and temporal workflows retain their own input contracts. Original implementations, pinned dependencies and shared native operators sit behind the same interface.

## Quick start

Use Linux x86-64 with a supported NVIDIA GPU. Install [uv, CUDA 11.8 and the system prerequisites](docs/INSTALL.md), then:

```bash
git clone --depth 1 https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
./panda install
./panda ui
```

Open **http://127.0.0.1:7860**:

| Your starting point | In the browser |
|---|---|
| **No dataset yet** | **Try the author sample** → download weights → run ASGrasp's included stereo sample |
| **Try another dataset** | Select **Dataset** → **Get dataset inputs** → prepare data and weights → check inputs → run |
| **GraspNet ready** | **Start a GraspNet experiment** → set your data path → download weights → check inputs → run |
| **A new model idea** | Choose a method → **Compose modules** → train the replacement → reuse its checkpoint |

Use **Check current form** to verify required inputs before starting. View predictions and checkpoints in **Runs & results**; expand **Compose modules** to build your own model.

See the [sample and dataset guide](docs/DOWNLOADS.md) for exact commands and download links. You can browse methods and generate JSON presets before installing the GPU runtime.

## Documentation

| Guide | Contents |
|---|---|
| [Install](docs/INSTALL.md) | Requirements, shared environment and troubleshooting |
| [Datasets](docs/DATASETS.md) | Input protocols, starter downloads, native training and shared components |
| [Data & weights](docs/DOWNLOADS.md) | Dataset archives, pretrained weights and preparation |
| [Usage](docs/USAGE.md) | Browser, CLI, training, resume and experiment sweeps |
| [Modules](docs/MODULES.md) | Compatible components and configuration |
| [Methods & papers](docs/METHODS.md) | Available operations, limitations, paper PDFs and original code |

The source stays small: `grasppanda/` contains the toolbox and installation recipes; `docs/` contains the guides. Data, weights, downloaded implementations and experiment outputs are created locally.

## Build on GraspPanda

Add a method, component or dataset through the [extension guide](docs/REFERENCE.md#extending-grasppanda). Contributions with clear input contracts and reproducible configurations are welcome. When using an integrated method, cite its original paper; source and weight terms are listed in [third-party notices](docs/THIRD_PARTY.md).

[MIT license](LICENSE) · [Report an issue](https://github.com/Daeda1used/GraspPanda/issues)
