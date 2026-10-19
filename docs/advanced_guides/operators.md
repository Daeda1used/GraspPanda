# Operators and their backends

`grasppanda.ops` holds the geometric operators the models are built from. Each has a
pure-PyTorch **reference** implementation that runs on any device and *defines* the
result. Four of them can also run on optional **compiled** kernels (C++/OpenMP and
CUDA), which are used only where they return exactly what the reference returns.

Nothing on this page is needed to use GraspPanda: after a plain `pip install` without
a compiler every operator runs on the reference. The kernels make inference and
training faster; they never change a result.

## Conventions

The conventions are part of the contract: which point wins a tie decides which seeds a
checkpoint sees, so a ported checkpoint only reproduces if they are kept.

| Operator | Result | Rules a backend must reproduce |
|---|---|---|
| `furthest_point_sample(points, num_samples, skip_near_origin, tie_break)` | `(B, num_samples)` indices | The first sample is index 0. The squared distance is `x² + y² + z²` added left to right. `tie_break="lowest_index"` takes the first of several equally far points; `"pointnet2_cuda"` takes the one the PointNet++ CUDA kernel takes (bit-reversed thread id, then index, with `min(512, 2^⌊log₂N⌋)` threads). `skip_near_origin` never picks a point with squared norm ≤ 1e-3 after the first. |
| `ball_query(query, support, radius, nsample)` | `(B, M, nsample)` indices, `(B, M)` hit counts | Strictly inside: `d² < radius²`. The first `nsample` hits in index order; missing slots repeat the first hit; a query without hits returns index 0 in every slot and a count of 0. |
| `cylinder_query(query, support, rotation, radius, hmin, hmax, nsample)` | as `ball_query` | With `d = Rᵀ(p − q)`: `hmin < d.x < hmax` and `d.y² + d.z² < radius²`, all strict; the axis is column 0 of `rotation`. Same ordering and padding. |
| `knn(query, support, k)` | `(B, M, k)` distances and indices | Ascending Euclidean distance; equidistant neighbours in index order (a stable sort); indices start at 0. |

`three_interpolate`, `inverse_distance_weights`, `voxelize`, `grasp_nms`, the view
templates and the serialisation codes have a single implementation.

## Backends

| Backend | What runs | When |
|---|---|---|
| `reference` | The pure-PyTorch operator. | Always available. |
| `compiled` | The kernels of `grasppanda._ops`. | A library has been built and its self-check passed on the device. Selecting it explicitly is an error otherwise. |
| `auto` (default) | `compiled` where it is available and passed its self-check, else `reference`. | |

The kernels take **float32** tensors on a CPU or a CUDA device. Other floating types
always run the reference, as does `knn` when gradients are requested or `k > 64`, and
`cylinder_query` under autocast or with TF32 matrix products (see below).

### What "identical" means, and how it is ensured

A kernel has to return the same integers as the reference for every input, and the same
distances bit for bit. That is checked three times: by construction, at run time on the
first call on a device, and in the test suite.

??? note "The three checks"

    1. **By construction.** Every kernel expression mirrors one expression of the
       reference (`grasppanda/_ops/src/common.h`):
       * no fused multiply-add: a fused `a·b + c` rounds once where PyTorch rounds twice
         (`-ffp-contract=off` on the host, `__fmul_rn` / `__fadd_rn` on CUDA);
       * three-term sums in PyTorch's order. Where the reference spells the order out
         (furthest point sampling) so do the kernels. Where it calls `sum(-1)` (ball
         query, kNN) PyTorch adds left to right on the CPU but `(x + z) + y` on CUDA, and
         the kernels follow the device;
       * thresholds are rounded to float32 the way PyTorch rounds a Python number that is
         compared with a float32 tensor;
       * the root of a kNN distance is taken by PyTorch, not by the kernel: PyTorch's
         float32 square root on the CPU is accurate to one unit in the last place but is
         not the correctly rounded one (it differed in 18 % of a million random values);
       * NaN and infinite coordinates behave as in `torch.minimum`, `argmax` and `sort`.
    2. **At run time.** The first time an operator runs on a device, its kernel is compared
       with the reference on small random and lattice clouds (ties, repeated points, empty
       and oversized neighbourhoods, points exactly on a surface), and the kernels' squared
       distances are compared with PyTorch's bit for bit. A kernel that differs is not
       used; one log line says so.
    3. **In the test-suite.** `tests/unit/test_ops.py` runs every oracle test on both
       backends; `tests/unit/test_ops_equivalence.py` and
       `tests/integration/test_ops_cuda.py` compare the backends on some 2 700 (CPU) and
       4 700 (CUDA) randomized calls; the stage-parity tests of all six ported methods
       (`tests/integration/test_<method>_parity.py`) and the backbone-oracle tests run with
       the default backend, `auto`, so with the kernels wherever they are built.

??? note "The cylinder query"

    The reference rotates the offsets with a batched matrix product, that is through a
    BLAS whose rounding is not specified: MKL and cuBLAS return different last bits for
    the same input. No kernel can reproduce that. The cylinder kernel therefore computes
    its own rotated offset together with a bound on how far any correct float32 evaluation
    can be from it, and gives a verdict only for support points that are further from the
    cylinder surface than that bound. A query that meets a point inside the band before it
    is complete is handed back and answered by the reference itself. On recorded
    GraspNet-1Billion frames that happens to 2 of 4096 queries, on synthetic table tops to
    13 of 65 536; the result is the reference's in every case.

    Under autocast, or with `torch.backends.cuda.matmul.allow_tf32`, the reference's
    matrix product runs in half precision or TF32. The kernel is then not used, because a
    float32 kernel would be *more* accurate than the reference and so disagree with it.

## Selecting a backend

In a config, per operator (`fps`, `ball_query`, `cylinder_query`, `knn`); operators
that are not mentioned use `auto`:

```yaml
runtime:
  ops: {fps: reference, knn: compiled}
```

The values are validated when the config is loaded. The selection is applied by
`grasppanda.ops.backends.configure`, which is also the Python interface:

```python
from grasppanda.ops import backends

backends.configure({"fps": "reference"})      # for this process
with backends.override(knn="reference"):       # temporarily, for a comparison
    ...
print(backends.report())
```

`grasppanda train`, `test` and `infer` apply `runtime.ops` when the run starts, and the
backend each operator settled on is written to `env.json` in the run directory.

The environment variable `GRASPPANDA_OPS` overrides the configuration for one process,
either with one backend for all operators or with a list:

```bash
GRASPPANDA_OPS=reference grasppanda test ...
GRASPPANDA_OPS=fps=reference,knn=compiled python my_script.py
```

`grasppanda env --json` and the `env.json` of every run record what each operator
uses on the CPU and on the current CUDA device, and why if that is not what `auto`
would prefer:

```json
"ops": {
  "fps": {"requested": "auto", "cpu": "compiled", "cuda": "compiled"},
  "knn": {"requested": "auto", "cpu": "compiled", "cuda": "reference",
          "note": {"cuda": "the compiled library has no kernels for 'cuda' tensors: ..."}},
  "library": {"path": ".../grasppanda_ops_3a33d4588d6dade9.so", "cuda": false,
              "detail": "CPU kernels only (no CUDA 11.x toolkit ...)"}
}
```

## Building the kernels

The kernels are compiled the first time an operator needs them, with PyTorch's own
extension tooling (`torch.utils.cpp_extension`), and cached. The build takes a few
seconds: the kernels use no PyTorch headers and are called through `ctypes`.

| Needed | For | Without it |
|---|---|---|
| A C++ compiler (`$CXX` or `c++`) and `ninja` (`pip install ninja`) | CPU kernels | One log line per process; every operator runs on the reference. Selecting `reference` (`GRASPPANDA_OPS=reference`) silences it. |
| A CUDA toolkit of the **major** version PyTorch was built with | CUDA kernels | CPU kernels only; operators on CUDA tensors run on the reference. |

```bash
python -m grasppanda._ops.build            # build if needed, then report
python -m grasppanda._ops.build --force    # rebuild, also after a recorded failure
python -m grasppanda._ops.build --cpu-only
```

??? note "Cache, toolkit, failures and forked workers"

    * **Cache.** `$GRASPPANDA_OPS_CACHE`, by default `<cache root>/ops` (`$GRASPPANDA_CACHE`
      or `~/.cache/grasppanda`). One directory per PyTorch version, variant (CPU or CUDA),
      platform and source revision, so several environments share a cache without clashing
      and an upgrade rebuilds. Nothing compiled is ever part of the package.
    * **Toolkit.** `$CUDA_HOME`, `nvcc` on the path and `/usr/local/cuda*` are examined; a
      toolkit of exactly PyTorch's CUDA version wins, then any of the same major version.
      The kernels are compiled to PTX for the oldest GPU architecture PyTorch supports and
      specialised by the driver on first use, so one build serves every GPU of a machine
      and a build made on a login node serves its compute nodes. `TORCH_CUDA_ARCH_LIST`
      overrides the architecture list as for any PyTorch extension.
    * **Failures.** A failed build is recorded with its log next to where the library would
      be, reported in one log line, and not retried until the compiler changes or `--force`
      is given. A failed CUDA build still yields the CPU kernels.
    * **Never on first use.** `GRASPPANDA_OPS_BUILD=0` uses a library only if it is already
      built.
    * **Forked workers** run the CPU kernels on one thread (GNU OpenMP does not survive a
      fork); the parent uses `torch.get_num_threads()`.

## Speed

On synthetic table-top clouds the kernels made the four operators 4 to 62 times faster
on the GPU of the measurements and 5 to 250 times on its CPU (`tools/bench_ops.py`).
End to end, two pipelines were timed on that GPU: the forward pass of the PointNet++
encoder of GraspNet-baseline (4.2 to 4.3 times faster) and **`GSNet.predict`, the only
whole model timed** (2.6 to 3.3 times). The other five methods were not timed end to
end.

??? note "How it was measured"

    Measured with `tools/bench_ops.py`: each case on both backends back to back, the
    device synchronised around every call; times are minimum / median in milliseconds and
    the ratio is the median of the per-pair ratios. Inputs are synthetic table-top clouds.
    The machine was shared while measuring (an RTX A6000 with six other jobs on it and a
    load average of 25 on 16 cores), so the ratios are more meaningful than the times.
    "reference" is the current pure-PyTorch implementation.

??? note "Operators on CUDA (RTX A6000, torch 2.7.1+cu118, 7 runs)"

    | case | batch | reference (ms) | compiled (ms) | ratio |
    |---|---:|---:|---:|---:|
    | fps: 1024 of 3000, ties: first | 1 | 55.87 / 57.80 | 2.02 / 2.54 | 25.7x |
    | fps: 1024 of 3000, ties: PointNet++ kernel | 1 | 88.67 / 91.55 | 3.03 / 3.31 | 29.1x |
    | fps: 1024 of 8000, ties: first | 1 | 55.56 / 56.09 | 4.49 / 6.44 | 8.6x |
    | fps: 1024 of 8000, ties: PointNet++ kernel | 1 | 88.41 / 89.02 | 4.86 / 6.24 | 14.2x |
    | fps: 1024 of 15000, ties: first | 1 | 55.60 / 55.99 | 9.67 / 10.40 | 5.4x |
    | fps: 1024 of 15000, ties: PointNet++ kernel | 1 | 88.42 / 88.69 | 9.08 / 10.98 | 8.1x |
    | cylinder_query: 1024 x 1024 seeds, 16 neighbours | 1 | 1.57 / 1.66 | 0.30 / 0.31 | 5.1x |
    | ball_query: 1024 x 15000, 16 neighbours | 1 | 3.54 / 3.95 | 0.30 / 0.40 | 11.6x |
    | cylinder_query: 1024 x 15000, 16 neighbours | 1 | 8.36 / 9.50 | 0.32 / 0.87 | 11.2x |
    | ball_query: 1024 x 15000, 64 neighbours | 1 | 2.50 / 4.17 | 0.23 / 0.58 | 8.0x |
    | cylinder_query: 1024 x 15000, 64 neighbours | 1 | 8.81 / 9.53 | 0.95 / 1.20 | 8.2x |
    | knn: 15000 x 1024, k=3 | 1 | 4.15 / 4.39 | 0.26 / 0.41 | 12.4x |
    | knn: 1024 x 15000, k=1 | 1 | 6.85 / 7.95 | 0.30 / 0.42 | 16.6x |
    | ball_query: 1024 x 20000, 16 neighbours | 1 | 3.86 / 5.12 | 0.74 / 1.17 | 4.2x |
    | cylinder_query: 1024 x 20000, 16 neighbours | 1 | 17.37 / 22.52 | 1.64 / 2.17 | 10.4x |
    | ball_query: 1024 x 20000, 64 neighbours | 1 | 4.74 / 5.22 | 0.49 / 1.26 | 4.1x |
    | cylinder_query: 1024 x 20000, 64 neighbours | 1 | 13.92 / 17.35 | 1.27 / 1.62 | 8.6x |
    | knn: 20000 x 1024, k=3 | 1 | 4.64 / 6.64 | 0.42 / 0.78 | 10.1x |
    | knn: 1024 x 20000, k=1 | 1 | 11.78 / 13.46 | 0.67 / 0.89 | 13.5x |
    | fps: 1024 of 3000, ties: first | 4 | 53.82 / 54.23 | 2.03 / 2.10 | 25.8x |
    | fps: 1024 of 3000, ties: PointNet++ kernel | 4 | 89.66 / 93.41 | 2.34 / 3.07 | 29.2x |
    | fps: 1024 of 8000, ties: first | 4 | 54.52 / 56.22 | 4.33 / 5.60 | 10.0x |
    | fps: 1024 of 8000, ties: PointNet++ kernel | 4 | 89.82 / 94.32 | 4.94 / 7.00 | 13.3x |
    | fps: 1024 of 15000, ties: first | 4 | 55.83 / 58.59 | 8.23 / 9.76 | 6.6x |
    | fps: 1024 of 15000, ties: PointNet++ kernel | 4 | 89.23 / 92.71 | 8.61 / 11.31 | 8.2x |
    | cylinder_query: 1024 x 1024 seeds, 16 neighbours | 4 | 2.40 / 2.41 | 0.14 / 0.14 | 17.4x |
    | ball_query: 1024 x 15000, 16 neighbours | 4 | 11.69 / 12.72 | 0.31 / 0.40 | 32.4x |
    | cylinder_query: 1024 x 15000, 16 neighbours | 4 | 41.30 / 46.58 | 0.65 / 1.00 | 46.7x |
    | ball_query: 1024 x 15000, 64 neighbours | 4 | 10.66 / 13.88 | 0.48 / 0.68 | 20.3x |
    | cylinder_query: 1024 x 15000, 64 neighbours | 4 | 48.22 / 55.53 | 0.97 / 1.88 | 37.5x |
    | knn: 15000 x 1024, k=3 | 4 | 15.05 / 18.48 | 0.84 / 1.20 | 13.1x |
    | knn: 1024 x 15000, k=1 | 4 | 29.47 / 32.34 | 0.41 / 0.62 | 54.5x |
    | ball_query: 1024 x 20000, 16 neighbours | 4 | 16.28 / 18.17 | 0.45 / 0.87 | 23.6x |
    | cylinder_query: 1024 x 20000, 16 neighbours | 4 | 52.57 / 59.43 | 0.67 / 0.91 | 58.3x |
    | ball_query: 1024 x 20000, 64 neighbours | 4 | 15.67 / 19.60 | 0.61 / 1.03 | 19.1x |
    | cylinder_query: 1024 x 20000, 64 neighbours | 4 | 52.46 / 70.64 | 1.03 / 1.21 | 53.8x |
    | knn: 20000 x 1024, k=3 | 4 | 18.88 / 21.17 | 1.07 / 1.43 | 14.5x |
    | knn: 1024 x 20000, k=1 | 4 | 38.52 / 44.77 | 0.54 / 0.66 | 62.1x |

??? note "Operators on the CPU (4 threads, 3 runs), a selection"

    `python tools/bench_ops.py --device cpu` prints all cases.

    | case | batch | reference (ms) | compiled (ms) | ratio |
    |---|---:|---:|---:|---:|
    | fps: 1024 of 3000, ties: first | 1 | 36.42 / 36.59 | 1.52 / 1.52 | 24.0x |
    | fps: 1024 of 15000, ties: first | 1 | 62.96 / 67.61 | 7.43 / 7.43 | 9.1x |
    | fps: 1024 of 15000, ties: PointNet++ kernel | 1 | 109.69 / 114.88 | 14.25 / 14.26 | 8.0x |
    | fps: 1024 of 15000, ties: PointNet++ kernel | 4 | 750.58 / 777.96 | 14.36 / 14.39 | 54.1x |
    | cylinder_query: 1024 x 1024 seeds, 16 neighbours | 1 | 16.95 / 18.18 | 1.87 / 1.90 | 9.7x |
    | ball_query: 1024 x 20000, 16 neighbours | 1 | 118.66 / 156.01 | 0.62 / 0.62 | 252.7x |
    | ball_query: 1024 x 20000, 64 neighbours | 1 | 122.85 / 182.34 | 2.03 / 2.06 | 60.5x |
    | cylinder_query: 1024 x 20000, 16 neighbours | 1 | 207.94 / 247.31 | 4.35 / 4.52 | 50.7x |
    | cylinder_query: 1024 x 20000, 64 neighbours | 1 | 219.31 / 230.34 | 13.26 / 54.79 | 4.9x |
    | knn: 20000 x 1024, k=3 | 1 | 655.94 / 793.83 | 20.55 / 22.15 | 38.6x |
    | knn: 1024 x 20000, k=1 | 1 | 999.58 / 1012.06 | 14.89 / 15.31 | 66.1x |
    | ball_query: 1024 x 20000, 64 neighbours | 4 | 594.42 / 658.26 | 7.01 / 7.04 | 93.9x |
    | cylinder_query: 1024 x 20000, 64 neighbours | 4 | 810.68 / 937.14 | 48.78 / 50.09 | 17.6x |
    | knn: 1024 x 20000, k=1 | 4 | 2875.05 / 3010.67 | 61.12 / 73.31 | 45.7x |

??? note "End to end: the encoder of GraspNet-baseline, and GSNet"

    End to end on the same GPU (`python tools/bench_ops.py --models`, 7 runs): the PointNet++
    encoder of GraspNet-baseline alone, and the whole of GSNet on recorded GraspNet-1Billion
    test frames with a checkpoint of the reference code (5 713 graspable candidates in the
    first):

    | case | batch | reference (ms) | compiled (ms) | ratio |
    |---|---:|---:|---:|---:|
    | PointNet2 backbone forward, 20000 points | 1 | 138.76 / 151.57 | 28.30 / 33.90 | 4.3x |
    | PointNet2 backbone forward, 20000 points | 4 | 178.48 / 202.99 | 40.94 / 46.77 | 4.2x |
    | GSNet.predict, 15000 points per frame | 1 | 122.85 / 130.97 | 21.72 / 41.19 | 3.3x |
    | GSNet.predict, 15000 points per frame | 4 | 137.18 / 171.34 | 39.04 / 59.69 | 2.6x |

    The reference itself became about twice as fast when its sampling loop was rewritten
    to work in place. Against the operators as they were before that (commit `d52eadd`),
    measured in one process with the three variants back to back:

    | case | before (ms) | reference now (ms) | compiled (ms) | before / compiled |
    |---|---:|---:|---:|---:|
    | PointNet2 forward, 20000 points, batch 1 | 293.8 / 309.4 | 135.5 / 151.9 | 33.1 / 43.4 | 8.6x |
    | PointNet2 forward, 20000 points, batch 4 | 313.8 / 337.6 | 162.1 / 191.2 | 41.9 / 54.7 | 6.6x |
    | GSNet.predict, batch 1 | 215.7 / 248.7 | 113.5 / 150.3 | 21.1 / 28.9 | 8.2x |
    | GSNet.predict, batch 4 | 235.2 / 260.9 | 126.2 / 143.4 | 37.4 / 60.1 | 4.5x |

    What is left of a GSNet frame with the kernels is the sparse U-Net (17 to 20 ms of
    26 ms); seed selection takes 6 ms, of which sampling 1024 of 5 713 candidates is about
    3.5 ms. The CUDA sampling kernel runs one block per cloud, which occupies a single
    multiprocessor of the GPU: its time grows linearly with the number of candidates
    (about 0.55 ns per point and sample) and is the first thing to improve if sampling
    matters again.

## Changing an operator or adding a kernel

* The reference comes first. Change behaviour there, with an oracle test written out
  in `tests/unit/test_ops.py`.
* A kernel is accepted when it returns identical integers and bit-identical distances
  on `tests/unit/test_ops_equivalence.py` and `tests/integration/test_ops_cuda.py`, and
  when the stage-parity tests of the six methods and the backbone oracle tests pass
  with it as default. Add the cases your change makes interesting to the randomized
  comparisons, and a self-check to `grasppanda/ops/backends.py`.
* Where the reference goes through arithmetic that cannot be mirrored (a BLAS, a
  library function that is not correctly rounded), let PyTorch compute that part, or
  decide only what is certain and hand the rest back, as the cylinder query does.
