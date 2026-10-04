# Installation

## Supported runtime

Use Ubuntu 22.04 x86-64 with an NVIDIA Ampere, Ada or Hopper GPU (compute capability 8.0, 8.6, 8.9 or 9.0). The shared environment locks Python 3.11.16, PyTorch 2.5.1+cu118 and NumPy 1.23.5; CUDA toolkit 11.8 is required. This runtime has been validated on an RTX A6000. A compatible NVIDIA driver and the **compiler toolkit** are required; the CUDA runtime bundled with PyTorch does not provide `nvcc`. See [NVIDIA's CUDA 11.8 installation guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html).

The lock includes Linux-specific CUDA wheels. Windows, macOS, CPU-only execution and newer GPU architectures are not supported by this tested runtime. Native build speed and memory requirements vary; reduce `MAX_JOBS` if compilation exhausts memory.

## Install once

To browse first, use `./panda list`, `./panda docs downloads` or generate a JSON preset with `./panda init --method hggd -o hggd.local.json`. These commands only need system Python 3.10+. The installer prepares the locked Python 3.11 environment used for all model operations.

### Choose the storage volume first

For a NAS or dedicated data disk, configure a fresh checkout **before installing**:

```bash
./panda storage --root /mnt/nas/GraspPanda
./panda storage
./panda install
```

The first command links the shared environment, downloaded sources, build cache, checkpoints, outputs and logs to the selected volume. The second reports their actual destinations. Existing data or links to another location are never replaced; use a fresh checkout when changing the storage layout. This command does not move an existing Python environment.

The `panda` launcher also places package, model, CUDA-kernel and browser artifact caches under `environments/artifact-cache/`. Explicit cache environment variables take precedence. Keep dataset roots on the same volume when using **Download starter data** or `panda data --root`; datasets are independently selectable. Mount the volume before starting the toolbox. Use one running queue per run directory on one host; a shared NAS is storage, not a distributed scheduler.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then the system libraries:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config python3 python3-dev \
  git ca-certificates curl unzip p7zip-full \
  libopenblas-dev libeigen3-dev libcgal-dev libboost-all-dev \
  libgmp-dev libmpfr-dev libgl1 libegl1 libglu1-mesa libglib2.0-0 libgomp1 libpcl-dev
export GRASPPANDA_CUDA_HOME=/usr/local/cuda-11.8
./panda install
./panda doctor
```

`./panda install` checks system prerequisites before downloading packages, fetches pinned sources for integrated methods and builds native extensions into the same `.venv`. Reference-only papers do not add downloads. It does not download datasets or the external checkpoint collection. Dataset starters and checkpoints are separate, explicit actions in the UI or CLI. Compilation can take substantial time on a new machine. Existing verified wheels are reused when their recorded build conditions match.

```bash
./panda weights hggd --camera realsense
./panda ui
```

All integrated methods use one environment. Rerun `./panda install` after updating the toolbox; its `uv sync --locked --inexact` step retains separately built native extensions and its builders reuse compatible cached wheels. An exact `uv sync` removes packages outside the lock. Rebuild after changing Python, PyTorch, CUDA or the GPU architecture. Import isolation happens through separate processes.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing `nvcc` or wrong CUDA release | Set `GRASPPANDA_CUDA_HOME` to the CUDA 11.8 toolkit directory. |
| No CUDA GPU visible / unsupported architecture | Check the NVIDIA driver and `CUDA_VISIBLE_DEVICES`; use one of the GPU architectures listed above. The installer checks this before native compilation. |
| GraspGen cannot initialize EGL | Install `libegl1` and `libglu1-mesa`; ensure the NVIDIA graphics/EGL driver is visible. CUDA compute alone does not provide headless rendering. |
| TARGO native operator build fails | Check the CUDA 11.8 compiler and system C++ prerequisites. Build logs are in `environments/artifact-cache/targo/`; the author checkout stays unchanged. |
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

With `panda storage --root`, these entries are ignored links to the chosen volume; the GitHub checkout still contains only source and guides.

Keep each dataset at a path of your choice and select that path in the UI. Use `./panda init --method graspness` or choose a [configuration example](USAGE.md#configuration-examples) to generate a `*.local.yaml` file before editing. Local paths and generated experiments stay out of commits. `pyproject.toml`, `uv.lock`, source pins and compatibility patches are required installation inputs.

<details>
<summary>Additional native component requirements</summary>

All components use the same Python environment. The installer fetches pinned build inputs, verifies compatible artifacts and keeps caches under `environments/`; rerun `./panda install` after upgrading.

| Component | Installation detail |
|---|---|
| Flash3D | Downloads a private CUDA 12.2 compiler/runtime for its extension; Transformer Engine uses CUDA 11.8. System toolkits and shared Python packages are retained. Allow substantial first-build time. |
| VMamba | Builds selective scan; Triton compiles cross-scan kernels on first use. A local driver linker alias is created when needed; `TRITON_LIBCUDA_PATH` takes precedence. |
| LitePT | Uses the locked FlashAttention wheel and builds PointROPE. `rope_backend: torch` selects the author's rotation implementation; attention requires Ampere or newer. |
| PTv2 / ResLFE | Compiles Pointcept / DeepLA operators with separate namespaces to coexist with legacy grasp operators. |
| GtG2 | Builds the GPG candidate generator against system PCL. |
| DexGraspNet 2.0 | Builds the native primitive-distance CUDA loss in the shared environment. Prediction and training do not require Isaac Gym; the author simulation evaluation has separate legacy requirements. |
| PointCNN++ / Swin3D | Builds the pinned CUDA operators using the shared CUDA 11.8 toolkit; versioned artifacts are activated only after verification. |

</details>
