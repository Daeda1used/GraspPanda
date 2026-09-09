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

**No dataset yet?** Start with the ASGrasp preset and its bundled stereo sample, or follow [Data & weights](docs/DOWNLOADS.md) to download GraspNet. Use `./panda init --method graspness` for a local CLI configuration. Sources, weights and experiment outputs are prepared locally.

## Guides

| Start here | Details |
|---|---|
| [Install](docs/INSTALL.md) · [Data & weights](docs/DOWNLOADS.md) | System requirements, dataset setup and author downloads |
| [Use the toolbox](docs/USAGE.md) · [Examples](GraspNet-1B/README.md) | Inference, training, resume and experiment sweeps |
| [Compose modules](docs/MODULES.md) | Compatible encoders, grouping, losses and optimization |
| [Methods & papers](docs/METHODS.md) | Available operations, limitations, PDF and source links |

[MIT license](LICENSE) · [Third-party terms](docs/THIRD_PARTY.md) · [Extend the toolbox](docs/EXTENDING.md)
