# Installation

## Requirements

- Linux, Python ≥ 3.10
- PyTorch ≥ 2.1, installed for your CUDA version
- NumPy 1.26 or 2.x

## Install

Install PyTorch first, following [pytorch.org](https://pytorch.org/get-started/locally/).
Then install GraspPanda from source:

```bash
git clone https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
pip install -e .
```

`pip install grasppanda` arrives with the first release.

## Optional extras

The core depends on `torch`, `numpy` and `pyyaml` only. Heavier pieces are extras:

| Extra | Installs | Needed for |
|---|---|---|
| `spconv-cu118`, `spconv-cu121`, `spconv-cu124`, `spconv-cu126`, `spconv-cpu` | the matching spconv wheel | sparse backbones (from M1) |
| `vis` | Open3D | the interactive window of `grasppanda demo --show`; pictures and animations need no extra |
| `dev` | pytest, ruff, codespell, pre-commit | contributing |
| `docs` | mkdocs-material, mkdocstrings | building this site |

```bash
pip install -e ".[spconv-cu118,dev]"
```

Pick the spconv extra that matches the CUDA build of your PyTorch
(`python -c "import torch; print(torch.version.cuda)"`). From an installed wheel the same
extras are installed with `pip install "grasppanda[spconv-cu118]"`. When a component
needs an extra that is missing, the error names the command for your install (wheel or
checkout) and, for spconv, the build for your PyTorch:

```text
Sparse convolution (backend 'spconv') needs 'spconv.pytorch', which is not installed.
Install it with: pip install "grasppanda[spconv-cpu]". That is the build for your PyTorch
(CPU only); the others: spconv-cu118 | spconv-cu121 | spconv-cu124 | spconv-cu126.
```

The shipped configs are part of the package, so a config of your own can inherit from
them by name from any directory (`_base_: [gsnet/gsnet_sunet14_g1b-realsense_10e.yaml]`,
see [Configs](../user_guides/config.md#inheritance)).

!!! note "The CPU-only spconv wheel runs single-threaded"
    `spconv-cpu` (the PyPI package `spconv` 2.3.8) returns wrong, non-repeatable features
    and gradients when PyTorch runs more than one thread. GraspPanda therefore runs each
    sparse convolution of a CPU tensor, forward and backward, on one thread and gives the
    other operators their threads back; the log of a run and the `probe` report say so
    once (`sparse convolution of CPU tensors runs on one thread ...`). It still compares the
    first sparse convolution of a CPU tensor and its gradient with dense ones, and stops
    with an explanation if they differ. No thread setting is needed. A CUDA build of
    spconv computes CPU tensors correctly with any number of threads and does not need a
    GPU to run, so its sparse convolutions stay multi-threaded on the CPU.

The four hot point operators (furthest point sampling, ball query, cylinder query, kNN)
have optional compiled kernels that are built on first use when a C++ compiler is
present; without one the pure-PyTorch reference runs. See
[Operators and backends](../advanced_guides/operators.md).

## Verify

```bash
grasppanda env
```

prints the interpreter, PyTorch build, visible GPUs and which optional backends are
installed. Paste this output into bug reports.
