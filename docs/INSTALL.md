# Installation

## Supported runtime

The tested configuration is Ubuntu 22.04 x86-64, Python 3.11.16, PyTorch 2.5.1+cu118, NumPy 1.23.5, CUDA toolkit 11.8 and an RTX A6000. A compatible NVIDIA driver and the **compiler toolkit** are required; the CUDA runtime bundled with PyTorch does not provide `nvcc`. See [NVIDIA's CUDA 11.8 installation guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html).

The lock includes Linux-specific CUDA wheels. Windows, macOS, CPU-only execution and newer GPU architectures are not supported by this tested runtime. Native build speed and memory requirements vary; reduce `MAX_JOBS` if compilation exhausts memory.

## Install once

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then the system libraries:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config python3 python3-dev \
  git ca-certificates curl unzip \
  libopenblas-dev libeigen3-dev libcgal-dev libboost-all-dev \
  libgmp-dev libmpfr-dev libgl1 libglib2.0-0 libgomp1 libpcl-dev
export GRASPPANDA_CUDA_HOME=/usr/local/cuda-11.8
./panda install
./panda doctor
```

`./panda install` checks system prerequisites before downloading packages, fetches pinned sources for integrated methods and builds native extensions into the same `.venv`. Reference-only papers do not add downloads. It does not download GraspNet or the external checkpoint collection. Compilation can take substantial time on a new machine. Existing verified wheels are reused when their recorded build conditions match.

```bash
./panda weights hggd --camera realsense
./panda ui
```

Do not create a separate environment per method. To refresh locked Python packages while retaining the separately built native extensions, use:

```bash
uv sync --python 3.11 --locked --inexact --no-dev
```

An exact `uv sync` removes packages outside the lock. Rerun the installer after changing Python, PyTorch, CUDA or the GPU architecture. Import isolation happens through separate processes, not separate environments.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing `nvcc` or wrong CUDA release | Set `GRASPPANDA_CUDA_HOME` to the CUDA 11.8 toolkit directory. |
| Undefined symbol / incompatible CUDA extension | Rebuild with `./panda install` using the locked environment; do not reuse wheels from a different ABI. |
| Out of memory during compilation | Set `MAX_JOBS=2` before running the installer. |
| Google Drive quota or academic mirror unavailable | Retry later or manually download the exact registered file; paths and checksums are in `grasppanda/resources/checkpoints.json`. |
| A source revision has changed | Restore the pinned checkout or intentionally update its lock and rerun validation. |
| Run directory already has a worker | CLI commands attach automatically. A second UI still needs another port/run directory; restart older workers after upgrading. |

Build logs are in `logs/`. `./panda doctor` reports the active interpreter, GPU, core extensions and source revisions; it is a readiness check, not a full method benchmark.

## Files created locally

The clone contains source, guides, example configurations and dependency locks. GitHub source archives omit repository automation and use the same installation commands; Git is still required to fetch the original implementations. Installation and use create the following ignored directories:

| Directory | Created by |
|---|---|
| `.venv/`, `upstream/`, `environments/` | Shared runtime installer; downloaded sources and native build cache |
| `checkpoints/` | Weight downloader |
| `outputs/` | Experiment queue, predictions, trained checkpoints and exports |
| `logs/` | Installation and source-download diagnostics |

Keep GraspNet at a path of your choice and select that path in the UI. Use `./panda init --method graspness` or choose a [configuration example](../GraspNet-1B/README.md#configuration-examples) to generate a `*.local.yaml` file before editing. Local paths and generated experiments stay out of commits. `pyproject.toml`, `uv.lock`, source pins and compatibility patches are required installation inputs.

<details>
<summary>Native component build details</summary>

Flash3D builds its native hierarchy and Transformer Engine in the same Python environment. The installer downloads a checksum-pinned CUDA 12.2 compiler and a private CUDA 12 runtime for the Flash3D extension; Transformer Engine builds against the shared CUDA 11.8 toolkit. No system toolkit or Python package is replaced. Native files live in versioned directories under `environments/native/flash3d/`. Before selecting a new build, the installer verifies forward/backward computation, an optimizer update and strict checkpoint reloading without a dataset. A failed check leaves the previous build selected. This additional compilation can take substantial time; its build tools and downloads are cached locally.

VMamba builds its CUDA selective-scan extension during installation. The first invocation compiles its Triton cross-scan kernels into a local cache. When a container mounts only `libcuda.so.1`, GraspPanda creates a linker alias under `environments/triton-driver/`; system libraries are not modified. An explicit `TRITON_LIBCUDA_PATH` takes precedence.

LitePT uses a checksum-locked FlashAttention wheel matching the shared Python/PyTorch/CUDA ABI. Its PointROPE CUDA extension is compiled locally for the selected GPU; `rope_backend: torch` selects the author’s PyTorch rotation implementation. Attention still requires a supported Ampere-or-newer GPU.

Point Transformer V2 builds the pinned Pointcept pointops with a separate Python/CUDA namespace. It shares the same interpreter, PyTorch and CUDA toolkit with the other point encoders. No Pointcept training environment or pretrained model is downloaded.

The ResLFE cylindrical component builds the pinned DeepLA CUDA operators under `environments/build/deepla-ops/`. Existing installations need another `./panda install` after upgrading; no separate environment is required. Original downloaded source remains unchanged.

GtG2 candidate generation builds GPG against system PCL using the shared Python interpreter. Matching verified binaries are reused locally. The build copy receives a deterministic sampling seed and an array binding; the original source stays intact. [Candidate graph experiments](MODULES.md#candidate-graph-experiments) describes data preparation and training.

PointCNN++ builds its pinned author CUDA/CUTLASS extension into a versioned local artifact using the same interpreter and CUDA 11.8 toolkit. No separate Python environment is created. The installer verifies the native encoder, gradients and tensor checkpoint loading before activating the artifact; a failed build leaves an existing installation in place. Allow additional compilation time on the first installation.

Swin3D builds its pinned attention and KNN CUDA operators in a versioned artifact under `environments/native/swin3d/`, using the same Python, PyTorch and CUDA toolkit. Forward/backward computation, an optimizer update, scene isolation and strict tensor loading are checked before activation. Source, compatibility patch and artifact hashes are verified when loading; a failed build leaves the previous artifact selected.

</details>
