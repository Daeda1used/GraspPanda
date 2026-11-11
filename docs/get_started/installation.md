# Install

GraspPanda runs on Linux with Python 3.10 or newer and PyTorch 2.1 or newer. Installing
takes a few minutes; checking it takes under a minute and needs no dataset.

## 1. Install PyTorch

Install a PyTorch built for CUDA 12.6 or older ([pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

A plain `pip install torch` now brings a build for CUDA 13, for which sparse convolution
(step 3) has no build, and which a driver older than CUDA 13 does not run on a GPU.

## 2. Install GraspPanda

```bash
git clone https://github.com/Daeda1used/GraspPanda.git
cd GraspPanda
pip install -e .
```

There is no release on PyPI yet: `pip install grasppanda` arrives with the first one.

## 3. Add what your method needs

As shipped, GraspNet-baseline, Scale-Balanced-Grasp and HGGD need nothing more. GSNet,
EconomicGrasp and FineGrasp use sparse convolution: add the spconv extra that matches the CUDA
build of your PyTorch.

```bash
# prints your CUDA build: 12.6 means spconv-cu126
python -c "import torch; print(torch.version.cuda)"
pip install -e ".[spconv-cu126]"
```

A CPU-only PyTorch prints `None`: use `spconv-cpu`.

An option from another library, such as an image encoder from timm, needs its extra too. Its slot page names it, and a missing extra stops with the command for your install.

??? note "Every extra"

    | Extra | Installs | Needed for |
    |---|---|---|
    | `spconv-cu118`, `spconv-cu121`, `spconv-cu124`, `spconv-cu126`, `spconv-cpu` | spconv for your CUDA build | GSNet, EconomicGrasp, FineGrasp, and the scene encoders built on sparse convolution (sparse-voxel networks, PTv3, LitePT) |
    | `timm`, `transformers`, `torchvision` | a model library | image encoders, and the Uni3D scene encoder (`timm`) |
    | `kornia` | Kornia | its losses |
    | `torch_geometric` | PyG | the PyG scene encoder and grouper |
    | `trimesh`, `roma`, `healpix` | a geometry library | gripper-mesh collision, some view stages |
    | `ocnn`, `mamba-ssm` | octree and selective-scan operators | the OctFormer and PointMamba scene encoders |
    | `optim` | timm, pytorch_optimizer, schedulefree | optimizers and schedules beyond `torch.optim` |
    | `fpsample`, `torch-cluster` | sampling libraries | the seed samplers that bridge to them |
    | `vis` | Open3D | the window of `grasppanda demo --show`; pictures need no extra |
    | `tensorboard`, `wandb` | a logger | logging to that service |
    | `dev`, `docs` | test, lint and docs tools | contributing, building this site |

    Extras combine: `pip install -e ".[spconv-cu126,timm]"`. From an installed wheel the
    same extras install with `pip install "grasppanda[spconv-cu126]"`.

??? note "Where it was run"

    | Machine | Python | PyTorch | What was run |
    |---|---|---|---|
    | RTX A6000 (48 GB) | 3.12 | 2.7.1, CUDA 11.8 | development: every record of this site |
    | RTX 3090 (24 GB) | 3.12 | 2.7.1, CUDA 11.8 | the unit suite, the first-number test of every method, training |
    | RTX 3090 (24 GB) | 3.11 | 2.14.1, CUDA 12.6 | the unit suite, the first-number test of every method |
    | no GPU | 3.10 | 2.14.1, CUDA 13.0 | the quick start from a fresh clone: everything but sparse convolution |

    At one precision setting the released checkpoints of the five seed-based methods score
    the same AP within 0.01 on the first three; HGGD does so only in float32, the default
    ([Reproducibility](../user_guides/train_test.md#reference)). 24 GB is enough for every
    shipped recipe; FineGrasp's needs its 32 frames spread over processes or accumulated
    steps there ([its README](https://github.com/Daeda1used/GraspPanda/tree/main/configs/finegrasp#train-and-test)).

??? note "spconv on a CPU-only machine"

    `spconv-cpu` (spconv 2.3.8) returns wrong, non-repeatable results when PyTorch runs
    more than one thread. GraspPanda therefore runs each sparse convolution of a CPU tensor
    on one thread and gives the other operators their threads back; a run's log and the
    `probe` report say so once. It also compares the first sparse convolution with a dense
    one and stops with an explanation if they differ. No setting is needed. A CUDA build of
    spconv computes CPU tensors correctly with any number of threads.

??? note "Compiled operators"

    Furthest point sampling, ball query, cylinder query and nearest neighbours have compiled
    kernels that are built on first use when a C++ compiler is present. Without one, the
    pure-PyTorch reference runs. See [Operators and backends](../advanced_guides/operators.md).

## 4. Check the install

```bash
grasppanda env
grasppanda train projects/example_project/configs/toy_train.yaml
```

`env` prints the interpreter, the PyTorch build, the GPUs and the optional backends, and ends
with `✓` or with what keeps a method from running here and the command that fixes it; paste
it into bug reports. `train` trains a toy detector on synthetic scenes, on the CPU, in under a
minute.
It ends with `✓ trained`: everything is wired. The toy's grasps mean nothing.

??? example "What `train` prints (shortened)"

    ```text
    ʕ•ᴥ•ʔ  GraspPanda <version>  ·  6-DoF grasp detection in clutter
      train
      config      projects/example_project/configs/toy_train.yaml  (hash be81c4b1301a)
      run         work_dirs/toy_train/20261018-120000
      device      cpu · 1 process · seed 0 · deterministic off
      model       ToyDetector · 7,601 parameters (7,601 trainable)
      data        32 frames · 8 batches of 4 per epoch
      schedule    5 epochs · Adam(lr=0.005, weight_decay=0.0) · ExponentialLR(gamma=0.95) per epoch
      evaluation  data.val every 1 epoch(s) · variants raw, centre · best by raw/AP
    epoch 1/5  iter 8/8  loss 3.9372  loss/score 3.8366  loss/width 0.1006  ...
    evaluation after epoch 1 (8 frames, 0.4 s)
    ...
    ✓ trained epochs 1-5 in 15.6 s
      checkpoint  work_dirs/toy_train/20261018-120000/checkpoints/epoch_0005.pth
      best        raw/AP = 91.3105 at epoch 5  (work_dirs/toy_train/20261018-120000/checkpoints/best.pth)
    ```

    The run directory is named after the time it started. The losses and the AP of the toy
    differ from run to run on another machine.

## Next

[Run a method](run_a_method.md) from its released weights, or [swap a component](swap.md).
