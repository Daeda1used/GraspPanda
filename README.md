# 🐼 GraspPanda

**A modular toolbox for visual grasping on GraspNet-1B.** Run methods, compose compatible components, and manage experiments in a browser or CLI with one shared environment.

## Get started

Requires Linux x86-64, an NVIDIA GPU and the CUDA 11.8 compiler toolkit. Install [uv and system prerequisites](docs/INSTALL.md), then:

```bash
git clone --depth 1 https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
./panda install
./panda ui
```

Open **http://127.0.0.1:7860**. Choose a method → **Load preset** → set your dataset root → **Download registered weights** → **Run current form**. View predictions and checkpoints in **Runs & results**.

**No dataset yet?** Follow the [sample walkthrough](docs/DOWNLOADS.md#start-without-graspnet), or [download GraspNet](docs/DOWNLOADS.md#graspnet-1b). To use the CLI, generate a configuration with `./panda init --method graspness`.

Prefer a source archive? Download and extract the [latest release](https://github.com/Daeda1used/GraspPanda/releases/latest), then use the same install commands.

## Guides

| Start here | Details |
|---|---|
| [Install](docs/INSTALL.md) · [Data & weights](docs/DOWNLOADS.md) | System requirements, dataset setup and author downloads |
| [Use the toolbox](docs/USAGE.md) · [Examples](GraspNet-1B/README.md) | Inference, training, resume and experiment sweeps |
| [Compose modules](docs/MODULES.md) | Compatible encoders, grouping, losses and optimization |
| [Methods & papers](docs/METHODS.md) | Available operations, limitations, PDF and source links |

[MIT license](LICENSE) · [Third-party terms](docs/THIRD_PARTY.md) · [Extend the toolbox](docs/EXTENDING.md)
