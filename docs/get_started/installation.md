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
(`python -c "import torch; print(torch.version.cuda)"`).

!!! warning "The CPU-only spconv wheel"
    `spconv-cpu` (the PyPI package `spconv` 2.3.8) returns wrong, non-repeatable features
    when OpenMP runs more than one thread. GraspPanda compares the first sparse
    convolution of a CPU tensor with a dense one and stops with an explanation instead of
    computing wrong results. On a machine without a GPU, either start Python with
    `OMP_NUM_THREADS=1`, or install a CUDA build of spconv: its CPU path is correct with
    any number of threads and does not need a GPU to run.

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
