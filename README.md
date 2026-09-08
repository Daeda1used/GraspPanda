# 🐼 GraspPanda

**A modular toolbox for visual grasping.** Run GraspNet-1B methods, compose compatible components, and manage experiments through a browser or the command line—all in one shared environment.

## Get started

Requires Linux x86-64, an NVIDIA GPU and CUDA toolkit 11.8. Install [uv and the system prerequisites](docs/INSTALL.md), then:

```bash
git clone https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
bash tools/bootstrap.sh
./panda ui
```

Open **http://127.0.0.1:7860**. Choose a method → **Load preset** → set your dataset root → **Download registered weights** → **Run current form**. Find predictions, checkpoints and exports in **Runs & results**.

**No dataset yet?** Choose **ASGrasp** and load its preset to start with the author's bundled stereo sample. See [Data & weights](docs/DOWNLOADS.md) for GraspNet downloads and preparation.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/INSTALL.md) | Shared runtime, prerequisites and troubleshooting |
| [GraspNet-1B](GraspNet-1B/README.md) | Dataset entry and editable examples |
| [Usage](docs/USAGE.md) | UI, CLI, sweeps, training, resume and evaluation |
| [Data & weights](docs/DOWNLOADS.md) | Official downloads, local paths and preprocessing |
| [Modules](docs/MODULES.md) | Compatible replacements and checkpoint transfer |
| [Methods & papers](docs/METHODS.md) | Support status, paper PDFs and original implementations |
| [Extension guide](docs/EXTENDING.md) | Add a method or component |

Sources, weights and experiment outputs are downloaded or generated locally.

[MIT license](LICENSE) · [Third-party terms](docs/THIRD_PARTY.md)
