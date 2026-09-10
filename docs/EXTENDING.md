# Extending GraspPanda

## Structure

```text
GraspNet-1B/              Dataset guide and configuration entry point
grasppanda/              Configuration, UI and experiment queue
  methods/               Method-specific inference, data and training adapters
  integrations/          Dataset contracts, readiness and evaluation
  modules/               Interchangeable components and architecture options
  training/              Shared trainers, objectives, augmentation and optimization
  resources/             Built-in templates, source pins, weight records and patches
  runtime/               Installation and isolated worker entry points
docs/                    User and extension guides
```

GraspPanda resolves one Python/PyTorch/CUDA environment. Each job runs in its own process to isolate upstream module names and import-time arguments. A SQLite queue owns scheduling, cancellation, timeouts and run records. Original repositories stay pinned in the local `upstream/` directory; adapters and scoped overlays supply compatibility changes.

## Add a method

1. Add its source URL, pinned revision, input protocol and license information to `grasppanda/resources/`.
2. Register checkpoint URLs, exact sizes, hashes and network roles in `checkpoints.json`.
3. Place method-specific adapters and trainers under `grasppanda/methods/`, or add a native recipe. Register actual operations in `grasppanda/config.py`; reference-only papers belong only in [Methods & papers](METHODS.md).
4. Define input readiness in the dataset provider. Write outputs using the official grasp representation and save the experiment manifest.
5. Exercise the relevant inference, loss, gradient, checkpoint reload and cancellation paths locally. Describe the supported operation in [Methods & papers](METHODS.md), with the paper PDF and original implementation.

Keep local data, weights, credentials, logs, development reports and generated configurations out of commits. Add reusable configuration templates to `grasppanda/resources/examples.yaml`; `./panda init` writes the selected experiment locally. The `.gitignore` resource allowlist must include any new non-Python installation input. Include relevant validation in the pull request description. Source availability alone is insufficient to enable an operation.

## Add a component

Register a choice in `grasppanda/components.py` and implement it under `grasppanda/modules/`. Match coordinates, units, sample indices, feature dimensions, neighborhoods, supervision, loss and decoder semantics. A matching tensor shape alone does not establish interchangeability. See [Compose modules](MODULES.md) for existing contracts and strict checkpoint transfer.

Declare accepted parameters and cross-stage constraints in `grasppanda/module_options.py`; the UI uses this schema for its parameter reference, and sweeps validate each expanded experiment against it. Pin external component sources in `component_sources.lock.json` and add any native build steps to `grasppanda/runtime/build_components.py`. Keep adaptation differences explicit in the component documentation.

Graspness, SBG and HGGD have different proposal, grouping and refinement structures. Replacing a branch may require its labels, losses and decoder as well as its forward method. Verify gradient flow into every replaced component and strict reload of the resulting checkpoint.

## Add a dataset

Register metadata in `grasppanda/datasets.py` and an executable provider under `grasppanda/integrations/`. Supply readers, camera conventions, target readiness, evaluation and compatible method adapters. Separate single-view, fused-view, active and temporal observation protocols. A metadata-only registration cannot run experiments.

## Compatibility

Build native operators against the locked runtime. Version source patches under `grasppanda/resources/patches/`; generate overlays and build products locally under `environments/`. Do not edit downloaded checkouts or silently discard learned parameters to make weights load.

| Component | Compatibility behavior |
|---|---|
| Legacy PyTorch helpers | Use standard-library equivalents for removed `torch._six` aliases. |
| Point/sparse operators | Compile PointNet2, KNN, MinkowskiEngine and PyTorch3D against the shared ABI. |
| HGGD / RNG | Select camera before import; retain native preprocessing and decoding. Remove profiling counters only. |
| GFLA | Adapt launcher imports and weight paths; compile the original C++ NMS. |
| CenterGrasp | Python 3.11 dataclass factories and the MPlib FCL bridge; retain native ICP and collision filtering. |
| ZeroGrasp | Adapt octree object-range arguments for the single-image sample. |
| MotionGrasp | Retain checkpoint tracker dimensions and explicit attention masks. |
| Graspness modern | Restore the deterministic view-lattice buffer; keep learned parameter matching strict. |
| GraspFast | Scope native KNN imports and map checkpoint keys without changing learned tensors. |
| ASGrasp | Fetch the pinned GSNet submodule and retain the author's RGB/stereo input path. |

Source pins and the Python lock make dependencies traceable. System compilers and hardware still affect native builds; a port does not by itself establish numerical equivalence to the historical author environment.
