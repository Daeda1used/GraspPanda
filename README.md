# 🐼 GraspPanda

**A modular toolbox for visual grasping on GraspNet-1B.** Run methods, compose compatible components, and manage experiments in a browser or CLI with one shared environment.

## Quick start

Requires Linux x86-64, an NVIDIA GPU and the CUDA 11.8 compiler toolkit. First install [uv and system prerequisites](docs/INSTALL.md).

```bash
git clone --depth 1 https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
./panda install
./panda ui
```

Open **http://127.0.0.1:7860**. Choose a method → **Load preset** → set your dataset root → **Download registered weights** → **Run current form**. Find your predictions and checkpoints in **Runs & results**.

**No dataset yet?** Try the [included author sample](docs/DOWNLOADS.md#start-without-graspnet), or follow the [GraspNet download guide](docs/DOWNLOADS.md#graspnet-1b).

## Documentation

| Guide | Contents |
|---|---|
| [Install](docs/INSTALL.md) | Requirements, shared environment and troubleshooting |
| [Data & weights](docs/DOWNLOADS.md) | Dataset archives, pretrained weights and preparation |
| [Usage](docs/USAGE.md) | Browser, CLI, training, resume and experiment sweeps |
| [Modules](docs/MODULES.md) | Compatible components and configuration |
| [Methods & papers](docs/METHODS.md) | Available operations, limitations, paper PDFs and original code |

[MIT license](LICENSE) · [Third-party terms](docs/THIRD_PARTY.md) · [Contributing](docs/REFERENCE.md#extending-grasppanda)
