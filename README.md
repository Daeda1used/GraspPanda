# 🐼 GraspPanda

**A modular toolbox for visual grasping on GraspNet-1B.** Run methods, compose compatible components, and manage experiments in a browser or CLI with one shared environment.

## Get started

Requires Linux x86-64, an NVIDIA GPU and the CUDA 11.8 compiler toolkit. Install [uv and system prerequisites](docs/INSTALL.md), then:

```bash
git clone https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
bash tools/bootstrap.sh
./panda ui
```

Open **http://127.0.0.1:7860**. Choose a method → **Load preset** → set your dataset root → **Download registered weights** → **Run current form**. View predictions and checkpoints in **Runs & results**.

**No dataset yet?** Start with the ASGrasp preset and its bundled stereo sample, or follow [Data & weights](docs/DOWNLOADS.md) to download GraspNet. Sources, weights and experiment outputs are prepared locally.

## Guides

- [GraspNet-1B & examples](GraspNet-1B/README.md)
- [Usage: UI, CLI, training, resume and sweeps](docs/USAGE.md)
- [Compatible modules](docs/MODULES.md)
- [Methods, availability, paper PDFs and original code](docs/METHODS.md)

[MIT license](LICENSE) · [Third-party terms](docs/THIRD_PARTY.md) · [Extend the toolbox](docs/EXTENDING.md)
