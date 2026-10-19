# GraspPanda Architecture

| | |
|---|---|
| Status | **Design record (v1.0), written before the implementation.** It keeps the decisions and the reasons for them. Where the code, the guides or the evidence pages differ from it, they are authoritative. |
| Date | 2026-10-12 |
| Scope | Design for v1: single-view 6-DoF parallel-jaw grasp detection in cluttered scenes. |
| Evidence | Every claim about an upstream repository was read from source at the pinned commit listed in Appendix A. No model was trained, no checkpoint was loaded and no AP was measured while writing this document. AP numbers quoted here are author-reported unless marked otherwise. |

Section numbers are kept as they were: docstrings in the code cite them (§5.2 to §11.2).
Every section is folded; open the one you need.

**Superseded since the record was written**

- §1 lists a web UI as out of scope. `grasppanda explore` now serves the combination
  explorer, a local web page that composes, compares and runs a method on one frame
  ([Explorer](../explorer/index.md)).
- §2, principle 7 ("share on second use"). The registries also hold options wrapped from
  papers and libraries that no shipped method uses, most scene encoders among them. Each
  was composed under every method it fits and, where accepted, probed on real frames; the
  few without a record say why ([Evidence](../evidence/index.md)).
- §4 and §8.2. Six methods are ported: five on the seed-based skeleton and HGGD. The
  dense-contact family (Contact-GraspNet) was not implemented. HGGD's image encoder is a
  registered slot (`image_encoder`); its other parts are options of the detector.
  `grasppanda/combine/` (compose, probe, the measured records) and `grasppanda/presets/`
  were added; tests live in `tests/unit/` and `tests/integration/`.
- §8 and Appendix A.2 were written with GSNet as the first user of the seed-based skeleton:
  the flow of §8.5, the constants of §8.6 and most examples are GSNet's. The skeleton now
  carries five methods on equal terms (GraspNet-baseline, GSNet, Scale-Balanced-Grasp,
  EconomicGrasp, FineGrasp); which slots each method has is on
  [Components](../slots/index.md).
- §12.2 to §12.4. No command downloads weights or a sample subset: `prepare` builds assets
  from a dataset on disk, `demo` runs given weights on a frame or the built-in example
  scene, `analyze` reads result records, and `infer` writes GraspNet arrays only.
  `compose`, `probe`, `components` and `explore` were added. `GraspInferencer` takes a
  config and a weights file. First steps: [Get started](../get_started/index.md).
- §13.3 and §13.4. No weights are published yet ([Model zoo](../model_zoo.md)). The site is
  organised as Get started, Explorer, Methods, Components, Guides, Evidence and Reference.

## 1. Goals and scope

<details markdown="1">
<summary>§1 in full</summary>

**One line.** GraspPanda is to 6-DoF grasp detection in clutter what MMDetection is to object detection: certified reproductions and controlled component-swap experiments, all scored by one pinned, official-equivalent evaluation.

**Four requirements, and how the design answers each.**

| Requirement | Design answer |
|---|---|
| Dataset-driven | A benchmark is a *protocol*: dataset version + split manifest + camera + allowed inputs + pinned evaluator (§11.3). Every result is stamped with its protocol id; only protocol-conformant runs enter the model zoo. GraspNet-1Billion first, GraspClutter6D second. |
| Low-code | One YAML file plus one command reproduces a method. First result within five commands after `pip install` (§12.4). |
| More than reproduction | Backbone, seed proposal, approach view, grouping, head bundle, loss, augmentation and post-processing are config slots with typed contracts (§8). An ablation is a base config plus an override. |
| Professional | Model zoo with provenance, docs site, CI, PyPI release on tag, changelog, citation file, brand assets (§13, §14). |

**v1 in scope.** Single-view RGB-D or point-cloud input; 6-DoF parallel-jaw grasps; cluttered tabletop scenes; GraspNet-1Billion and GraspClutter6D.

**Reserved, not implemented in v1** (the architecture leaves room, §8.7):
multi-view fusion track (Generalizing-Grasp; own protocol and own leaderboard), suction (SuctionNet-1B), simulated clearing (pile/packed, GSR/DR).

**Out of scope.** Planar grasps, dexterous hands, object-centric datasets (ACRONYM, GraspGen), web UI, and wrapping upstream repositories in subprocesses as the core implementation.

</details>

## 2. Design principles

<details markdown="1">
<summary>§2 in full</summary>

Each rule exists because the previous GraspPanda (v0.62) or an upstream codebase broke it.

1. **Depth before breadth.** A method enters the model zoo only after it has been trained *and* evaluated inside this toolbox (§13.1). v0.62 had 39 method entries and no certified reproduction.
2. **Every number carries its provenance.** Three classes, never mixed in one column: `reported` (paper/README), `author-ckpt` (author weights evaluated here), `reproduced` (trained here).
3. **Observation and supervision are separate objects.** `predict(observation)` cannot reach ground truth. Any GT-derived input (workspace mask from GT segmentation, GT instance masks) is an explicit, named config attribute and is stamped on the result (§7.1).
4. **Slots have typed contracts.** A swap is allowed only when contracts match. A mismatch is a config-check error with an explanation — never a silent interpolation, projection or axis squeeze (§8.3).
5. **One owner per constant.** `num_view`, `num_angle`, `num_depth`, depth bins, gripper width, voxel size and seed count each live in exactly one object; everything else receives them by injection (§8.6).
6. **The head bundle is atomic.** Head, target encoder, loss and decoder share one parameterization and are swapped as a unit (§8.4).
7. **Share on second use.** A component becomes a public, registered slot only when two in-tree methods use it. Until then it stays private to its detector.
8. **Small core, optional weight.** Core depends on `torch`, `numpy` and a few pure-Python packages. Sparse convolution, visualisation, logging backends and accelerated ops are extras. No `sys.path` edits, no top-level packages named `pointnet2`/`knn`/`models`/`utils`.
9. **A run is fully described by its snapshot.** No behaviour is reachable only through a CLI flag (§5.6).
10. **Clean room.** Upstream logic is studied; code is written here. No file is copied (§14.3).

</details>

## 3. Lessons from reference frameworks

<details markdown="1">
<summary>§3 in full</summary>

Verified against source on 2026-10-12: PaddleSeg `release/2.10`, MMEngine `main`, Detectron2 `main`, Pointcept `main`.

| Framework | Borrow | Avoid (with evidence) |
|---|---|---|
| **PaddleSeg** | YAML + `_base_`; every component is `{type: Name, **kwargs}` built recursively; a rule-based `ConfigChecker` that runs at load time; decorator registration; one-command train/val/predict. | `--opts` only asserts that *intermediate* keys exist and then assigns the leaf unconditionally, so a misspelt leaf key silently creates a new entry (`cvlibs/config.py:228-235`). Top-level keys are read with `dic.get`, so unknown keys are ignored. The builder resolves a `type` by scanning *all* registries in order (`cvlibs/builder.py:82-87`), so equal names in two registries collide silently. Duplicate registration only warns and overwrites (`cvlibs/manager.py:112-115`). The checker *mutates* the config to sync `num_classes` (`allow_update=True`). Unbounded `_base_` depth. Components register only if someone remembered to import them in `__init__.py`. Build errors are re-raised as `RuntimeError(str(e))`, dropping the traceback (`builder.py:65-75`). More than ten run settings exist only as CLI flags (`--do_eval`, `--save_interval`, `--resume_model`, `--seed`, `--use_ema`, `--log_iters`, `--keep_checkpoint_max`, … in `tools/train.py`). |
| **MMEngine** | `Registry(locations=[...])` lazily imports its modules on first lookup; duplicate names raise unless `force=True`; `custom_imports`; `_delete_`; hook-based engine; dumping the resolved config. | Dict merge keeps stale base keys when a child changes `type` unless the user remembers `_delete_=True` (`config.py:_merge_a_into_b`). No key validation: an unknown kwarg fails only inside a constructor, possibly hours into a job. A failed lookup returns `None` and the error offers no candidate names (`registry/build_functions.py:98-106`). Scope hierarchy and `DefaultScope` switching add a lot of machinery for a single-package toolbox. Python configs execute arbitrary code and exist in two dialects (text and lazy-import). `{{_base_.x}}` is substituted before child overrides apply, so overriding `x` does not propagate. `runner.py` is 2413 lines. |
| **Detectron2** | `tools/plain_train_net.py`: a complete training loop in 221 readable lines. `ShapeSpec`: children are built with the parent-computed input shape instead of the user repeating channel counts. `Instances`: one typed container for predictions. | yacs `CfgNode` requires every key of every component to be pre-declared in one central `defaults.py`. |
| **Pointcept** | Run snapshot: resolved config dumped into the experiment directory, code copied alongside (`scripts/train.sh:87`). Offset-based batching for variable-size point clouds. spconv/PTv3 integration patterns. | The launcher executes the *copied* code via `PYTHONPATH`, so what runs is not what is installed. Config values are rewritten implicitly at load (`data.train.loop = epoch // eval_epoch`, `num_worker=None → cpu_count()`; `engines/defaults.py:122,133`). |

Resulting choices: YAML with strict three-layer validation (§5.3), type-aware merge (§5.2), one flat registry family with lazy auto-import and nearest-name errors (§6), build-time injection instead of repeated channel counts (§6.3), a ~300-line trainer with hooks (§12.1), and a snapshot that records code without running from the copy (§5.6).

</details>

## 4. Repository layout

<details markdown="1">
<summary>§4 in full</summary>

```
GraspPanda/
├── configs/
│   ├── _base_/
│   │   ├── datasets/            graspnet1b_realsense.yaml, graspnet1b_kinect.yaml, graspclutter6d_d435.yaml
│   │   ├── models/              gsnet.yaml, economicgrasp.yaml, hggd.yaml, …   (full default model per method)
│   │   ├── schedules/           one file per training recipe
│   │   └── default_runtime.yaml
│   └── <method>/                README.md, metafile.yaml, <method>_<backbone>_<dataset>-<camera>_<schedule>.yaml
├── grasppanda/
│   ├── config/                  loader, schema, validation, cross-field checker, snapshot
│   ├── registry.py              Registry, build(), lazy auto-import, plugin discovery
│   ├── structures/              Observation, Supervision, PointSet, PointFeatures, SeedSet, ApproachSet,
│   │                            GroupFeatures, GraspGroup, Gripper, ReferenceFrame, SampleKey
│   ├── datasets/                manifest, derived-asset store, graspnet1b/, graspclutter6d/, samplers
│   ├── transforms/              point-cloud and RGB-D preprocessing and augmentation
│   ├── models/
│   │   ├── backbones/           PointNet++, SparseUNet (spconv), PTv3, 2-D ResNet …
│   │   ├── proposals/           graspness / objectness + FPS, OBS (HGGD's heat-map proposal stays in its detector until reused)
│   │   ├── views/               300-view classification, regression, anchors
│   │   ├── groupers/            cylinder crop: single / multi-scale / multi-depth
│   │   ├── assigners/           seed ↔ grasp-label matching shared by the GraspNet family
│   │   ├── heads/               head bundles: head + target encoder + loss + decoder
│   │   ├── losses/              primitive losses used inside stages
│   │   ├── postprocessors/      collision filter, grasp NMS, width clip, re-rankers
│   │   └── detectors/           GSNet, EconomicGrasp, HGGD, …
│   ├── ops/                     FPS, ball/cylinder query, kNN, grasp NMS: pure-PyTorch reference + optional backends
│   │   └── sparse/              sparse-conv facade: spconv | warpconvnet | minkowski (legacy)
│   ├── _ops/                    compiled extension namespace (built only if a kernel proves necessary)
│   ├── evaluation/              graspnet1b/, graspclutter6d/, protocols/, result records, statistics
│   ├── engine/                  trainer, hooks, optim builders, checkpoint, distributed helpers
│   ├── apis/                    GraspInferencer and friends
│   ├── cli/                     `grasppanda` entry point, one module per sub-command
│   ├── hub/                     weight index, download, sha256 verification
│   ├── visualization/
│   └── utils/
├── tools/                       repo-only scripts: dist_train.sh, gen_model_zoo.py, make_parity_fixtures.py, convert_checkpoint.py
├── projects/                    example_project/ and experimental or third-party methods, each with a support-level badge
├── docs/                        MkDocs Material site; design/ holds this file
├── tests/                       unit/, contracts/, parity/, smoke/
├── resources/                   logo, banner, architecture diagram, demo GIF sources
└── pyproject.toml  README.md  README_zh-CN.md  LICENSE  NOTICE  CITATION.cff  CHANGELOG.md  CONTRIBUTING.md
```

Packaging notes:

- **Configs ship in the wheel.** `configs/` stays at the repository root for visibility and is mapped to `grasppanda/_configs` at build time. `grasppanda train gsnet/gsnet_sunet14_g1b-realsense_10e` resolves a config by name whether the package was installed from PyPI or in editable mode. Without this, "one command after `pip install`" would require cloning the repository.
- **The CLI lives in the package**, not in `tools/`, for the same reason. `tools/` keeps only scripts that make sense inside a checkout.
- **Method folders are organised by method, benchmark is in the file name** (`<method>_<backbone>_<dataset>-<camera>_<schedule>.yaml`), matching PaddleSeg and MMDetection. The benchmark-first view is generated: the model-zoo page is grouped by protocol.

</details>

## 5. Configuration system

<details markdown="1">
<summary>§5 in full: 5.1 Format and top-level schema · 5.2 Inheritance and merge rules · 5.3 Strict validation · 5.4 Command-line overrides · 5.5 Cross-field consistency checker · 5.6 Run snapshot</summary>

### 5.1 Format and top-level schema

YAML, loaded with `yaml.safe_load`. The top level is a closed set of keys:

| Key | Content |
|---|---|
| `_base_` | List of relative paths to parent files. |
| `custom_imports` | Out-of-tree components, imported before validation: module names, or `.py` paths relative to the config that lists them. |
| `model` | The detector: `{type: GSNet, backbone: {...}, ...}`. |
| `data` | `train`, `val`, `test`, each `{dataset: {type: ...}, batch_size, num_workers, sampler}`. |
| `optimizer`, `scheduler` | `{type: ...}`. |
| `train` | `epochs`, `amp`, `grad_clip`, `accumulate`, `ema`, `sync_bn`. |
| `evaluation` | `interval`, `evaluator`, `variants` (post-processing chains, §11.4), `save_best`. |
| `hooks` | Extra hooks beyond the defaults. |
| `runtime` | `work_dir`, `seed`, `deterministic`, `resume`, `load_from`, `log`, `checkpoint`, `ops` (backend selection), `dist`. |

Target user experience — the whole file a user writes for a backbone ablation:

```yaml
_base_:
  - ../_base_/models/gsnet.yaml
  - ../_base_/datasets/graspnet1b_realsense.yaml
  - ../_base_/schedules/gsnet_10e.yaml
  - ../_base_/default_runtime.yaml

model:
  backbone: {type: PTv3, in_channels: 3}     # was SparseUNet14; the old kwargs are dropped, not merged
```

### 5.2 Inheritance and merge rules

1. **Depth ≤ 2.** A leaf may inherit from a file that itself inherits from `_base_/` primitives; a third hop is an error. Deep chains are where readers lose track of what a config means.
2. **Sibling bases must not conflict.** If two files in one `_base_` list define the same leaf path, loading fails and names both files. There is therefore no order-dependence to remember (PaddleSeg resolves such conflicts by list position).
3. **Dict onto dict: deep merge — unless `type` changes.** When the child node sets a `type` different from the parent's, the parent node is *replaced*. This removes MMEngine's stale-kwargs trap and makes "swap one component" a one-line edit. The log lists the dropped keys.
4. `_delete_: true` replaces an untyped dict explicitly.
5. **Lists replace wholesale.** Individual elements are reachable through overrides (`hooks.0.interval=5`).
6. **Interpolation `${a.b.c}`** refers to another config path and is resolved *after* all merges and overrides, so overriding the source propagates to every reference.
7. Dataset roots are not interpolated from the environment. A dataset resolves its root from the config value, then `$GRASPPANDA_DATA`, then `./data`. Machine paths and `runtime.work_dir` are excluded from the config hash.

### 5.3 Strict validation

Validation runs before any dataset is opened or any module is constructed. `grasppanda print-config --check` runs exactly the same code.

| Layer | What is checked | Failure example |
|---|---|---|
| 1. Schema | Top-level and fixed-structure keys (`train`, `runtime`, loader blocks) against a declared schema. | `unknown key 'runtim'. Did you mean 'runtime'?` |
| 2. Component signatures | For every `{type: X, ...}` node: `X` exists in the registry that owns that slot; every kwarg is a parameter of `X.__init__`; every required parameter is present or injected (§6.3); scalar types match annotations. Registered classes may not accept `**kwargs`. | `SWADHead has no parameter 'num_angles'. Did you mean 'num_angle'?` |
| 3. Cross-field rules | Relations between components (§5.5). | `CylinderGroup emits axes (seed,) but OperationHead expects (seed, depth).` |

All problems found in one pass are reported together, each with its config path and the file it came from.

### 5.4 Command-line overrides

`--opts model.backbone.type=PTv3 train.epochs=20 hooks.0.interval=5`

- Values are parsed as YAML scalars, so `true`, `null`, `1e-3`, `[1, 2]` mean the same as in a file.
- Overrides are applied to the merged tree and the full validation of §5.3 then runs. An override cannot create a key that validation would reject in a file.
- Overriding `type` follows merge rule 3: kwargs of the previous type are dropped and listed.
- The override list is stored in the snapshot.

### 5.5 Cross-field consistency checker

The checker **never edits the config**. A value is either stated once and injected where needed, or stated twice and required to agree. PaddleSeg's auto-sync is convenient but leaves the user unsure which value won.

| Rule | Must agree across | Mechanism |
|---|---|---|
| Grasp parameterization: `num_view`, `num_angle`, `num_depth`, depth bins, angle range | view stage, head bundle (head / targets / loss / decoder), label grid of the dataset | Owned by the head bundle's codec and the view template; the dataset declares its label grid; mismatch is an error unless an explicit re-binning transform is configured. |
| Gripper: max width, finger height, depth base | decoder clamp, collision filter, evaluator | Owned by one `Gripper` object referenced by all three. |
| Camera | dataset, evaluator, metadata of loaded weights | Compared; mismatch on weights is an error unless `runtime.allow_camera_mismatch: true`. |
| Point budget: sampled points ≥ seeds; voxel size | sampler transform, proposal stage, sparse backbone | Compared; voxel size is owned by the backbone. |
| Preprocessing fingerprint | transforms that precede label alignment, derived-label assets | Asset manifest hash must match (§10.2). |
| Inputs | `model.required_inputs` ⊆ what the dataset's observation provides | Missing input is an error. GT-provenance inputs set `gt_assisted` on the run and are rejected by protocols that forbid them. |
| Stage contracts | backbone output kind ↔ proposal requirement; grouper axes ↔ head axes; channel counts | Checked statically from class metadata, again at build time from real modules. |
| Splits | `val` scenes disjoint from `train` scenes; `save_best` never monitors a test split | Compared on scene ids from the manifests. |
| Protocol conformance | evaluator constants, split manifest hash, post-processing declared | Non-conformant runs are stamped `protocol: custom` and cannot enter the zoo. |

Rules are small classes registered in a list, so a project can add its own.

### 5.6 Run snapshot

Every `train`, `test` and `sweep` run writes into `work_dir/<run_id>/`:

| File | Content |
|---|---|
| `config.resolved.yaml` | Fully expanded config, interpolations resolved, defaults of every component filled in. |
| `config.sources/` | The original leaf file and each base, unmodified, plus `overrides.txt`. |
| `run.json` | Command line, start time, host, seed, `config_hash`, protocol id, parent run on resume. |
| `git.json`, `diff.patch` | Commit, dirty flag, uncommitted diff. For a wheel install: package version and `RECORD` hash instead. |
| `code.tar.gz` | Tracked files of `grasppanda/` and `configs/`. Stored for the record; the run never executes from it. |
| `env.json` | Python, torch, CUDA, cuDNN, GPU, driver, selected op/sparse backends, `pip freeze`. |
| `data.json` | Dataset fingerprint, split-manifest hashes, derived-asset manifest hashes. |

`grasppanda print-config <cfg> [--opts ...]` prints `config.resolved.yaml` to stdout without creating a run.

</details>

## 6. Registry and builder

<details markdown="1">
<summary>§6 in full: 6.1 Registries · 6.2 Behaviour · 6.3 Building and injection</summary>

### 6.1 Registries

`DATASETS`, `TRANSFORMS`, `BACKBONES`, `PROPOSALS`, `VIEWS`, `GROUPERS`, `ASSIGNERS`, `HEADS`, `LOSSES`, `DETECTORS`, `POSTPROCESSORS`, `EVALUATORS`, `HOOKS`, `OPTIMIZERS`, `SCHEDULERS`.

`OPTIMIZERS` and `SCHEDULERS` are pre-populated from `torch.optim` so that `{type: AdamW, lr: 1e-3}` works without wrappers.

### 6.2 Behaviour

- **One flat `Registry` class.** No scopes, no parent/child hierarchy.
- **Lazy auto-import.** Each registry declares the package it owns (`BACKBONES → grasppanda.models.backbones`). On first lookup every module below that package is imported. Adding a file is enough; nobody edits `__init__.py`. `import grasppanda` stays cheap.
- **Optional dependencies never break import.** A module that needs `spconv` imports it lazily. If the extra is missing, building the component raises `SparseUNet14 needs spconv: pip install -e ".[spconv-cu118]"`.
- **Out-of-tree code.** `custom_imports: [my_pkg.my_heads]` or `custom_imports: [../my_components.py]` in a config, or a `grasppanda.plugins` entry point for pip-installed plugins. A `.py` path is loaded as a standalone module, so a single-file plugin next to its config needs neither installation nor a `sys.path` edit. `projects/` uses the same mechanism; no fork is needed.
- **Duplicate names raise.** Overriding requires an explicit `force=True`.
- **Helpful failures.** A failed lookup lists the registered names, the closest matches by edit distance, and whether the name exists in a *different* registry (`'SWADHead' is registered in HEADS, not in GROUPERS`).
- **Metadata on the class.** Each component may declare `status` (`experimental` / `tested` / `reproduced`), `requires` (extras), and its contract attributes (§8.3). `grasppanda print-config --list backbones` and the docs tables are generated from these.

### 6.3 Building and injection

A slot's registry is determined by the parameter annotation of the parent class, not by searching every registry for a name:

```python
@DETECTORS.register()
class GSNet(SeedGraspDetector):
    def __init__(
        self,
        backbone: Cfg[Backbone],
        proposal: Cfg[Proposal],
        view: Cfg[ViewStage],
        grouper: Cfg[Grouper],
        head: Cfg[HeadBundle],
        assigner: Cfg[Assigner],
    ):
        self.backbone = BACKBONES.build(backbone)
        self.proposal = PROPOSALS.build(proposal, in_channels=self.backbone.out_channels)
        self.view = VIEWS.build(view, in_channels=self.backbone.out_channels)
        self.grouper = GROUPERS.build(grouper, in_channels=self.backbone.out_channels)
        self.head = HEADS.build(
            head, in_channels=self.grouper.out_channels, view_template=self.view.template
        )
        ...
```

- `Cfg[Backbone]` tells the static validator which registry owns the slot, so layer 2 of §5.3 can walk the whole tree without constructing anything.
- `REGISTRY.build(node, **injected)` fills only parameters the user left unset. If the user set one and it disagrees with the injected value, that is an error — never a silent override. This is Detectron2's `ShapeSpec` idea generalised.
- A parameter that is meant to be injected is declared with the sentinel default `INJECTED`; the validator treats it as optional, and construction fails clearly if nobody supplied it.
- `REGISTRY.build` also accepts an already-constructed object, so the Python API and unit tests can assemble a detector without YAML.

</details>

## 7. Data contracts

<details markdown="1">
<summary>§7 in full: 7.1 Observation and Supervision · 7.2 Points and features · 7.3 Stage outputs · 7.4 GraspGroup</summary>

All structures are plain dataclasses holding tensors. They live in `grasppanda/structures/` and have no dependency on any model.

### 7.1 Observation and Supervision

```python
@dataclass(frozen=True)
class SampleKey:  # identity of one evaluation unit
    dataset: str
    scene: int
    camera: str
    frame: int


class Provenance(Enum):  # where a piece of input information came from
    SENSOR
    CALIBRATION
    PREDICTED
    GROUND_TRUTH


@dataclass
class Observation:  # everything a model may see at inference time
    key: SampleKey
    depth: Tensor | None  # (H, W) metres
    rgb: Tensor | None  # (3, H, W) in [0, 1]
    intrinsics: Tensor  # (3, 3)
    points: PointSet | None  # camera frame, metres
    point_feats: Tensor | None  # (N, C): ones / rgb / normals, as declared
    masks: dict[str, Mask]  # 'workspace', 'instance', …; each Mask carries data + Provenance
    frame: ReferenceFrame


@dataclass
class Supervision:  # training-only; never passed to predict()
    objects: list[
        ObjectGraspLabels
    ]  # per object: points, friction scores, widths, collision mask, pose
    point_labels: dict[
        str, Tensor
    ]  # 'graspness', 'objectness', 'instance', … aligned to observation.points
    extras: dict[str, Any]  # method-specific targets (heat-maps, tolerance, …)
```

- A detector implements `loss(observation, supervision) -> dict[str, Tensor]` and `predict(observation) -> list[GraspGroup]`. `predict` receives no supervision, no evaluator handle and no dataset handle.
- A dataset config declares its observation explicitly, including every mask and its source. A model declares `required_inputs`. The checker matches them (§5.5).
- **Ground-truth leakage is a named attribute, not a hidden default.** The workspace crop used by the GraspNet family is derived from the GT segmentation mask at test time (§8.1). Here that is spelled `workspace: {type: SegmentationBBoxWorkspace, source: ground_truth, outlier: 0.02}`. Any `GROUND_TRUTH` provenance on an inference input sets `gt_assisted: [workspace]` on the run and appears as a column in result tables.
- Batches concatenate variable-size clouds with batch offsets; fixed-size batches expose a dense `(B, N, …)` view without copying.

### 7.2 Points and features

| Structure | Fields | Invariant |
|---|---|---|
| `PointSet` | `coords (N,3)`, `batch_offsets`, `frame`, `id`, optional `parent_id` + `index` | `id` is unique per set. A sub-sampled set records which parent it came from and, when it is a true subset, the index into it. |
| `PointFeatures` | `feats (N,C)`, `points: PointSet` | Features cannot exist without their set. Consumers assert the set identity they expect. |
| `VoxelMap` (internal to sparse backbones) | voxel coords, `inverse_map`, `voxel_size`, `source_id` | `devoxelize()` returns `PointFeatures` bound to the source set and raises if asked to map onto any other set. An `inverse_map` therefore cannot be reused across backbones or point sets. |

A backbone states what it returns: `output = "dense"` (features on the *input* set — sparse U-Net, PTv3) or `output = "subsampled"` (features on a child set — PointNet++ set abstraction). Stages that need one or the other say so; conversion between them is an explicit, configured module (`FeaturePropagation`), never an implicit interpolation.

### 7.3 Stage outputs

| Structure | Fields |
|---|---|
| `SeedSet` | `points: PointSet` (child of the scene set), `feats (M,C)`, `scores: dict[str, Tensor]` with declared semantics (`graspness`, `objectness`, `heatmap`), `valid (M,)`, `source` (how the seeds were chosen). No score is mandatory. |
| `ApproachSet` | `direction (M,3)` unit vectors, optional `rotation (M,3,3)`, optional `view_index (M,)` with the `ViewTemplate` it indexes, optional `logits`. Aligned one-to-one with a `SeedSet`. |
| `GroupFeatures` | `feats` with **named axes**, e.g. `(seed, channel)` or `(seed, depth, channel)`; `seeds`; `local_frame`. An axis is never squeezed implicitly. |
| `RawPredictions` | Method-specific tensors plus the id of the codec that can decode them. Keeps gradients. |

Too few seeds, an empty heat-map or a failed segmentation produce `valid = False` entries, not zero-coordinate grasps. Each detector documents its padding policy.

### 7.4 GraspGroup

The single output type of every detector and the single input type of every post-processor and evaluator.

| Field | Shape | Meaning |
|---|---|---|
| `translation` | `(G,3)` | Grasp point in metres, in `frame`. |
| `rotation` | `(G,3,3)` | Proper rotation (`det = +1`, orthonormal within tolerance). Columns: approach axis, closing axis, height axis. |
| `width`, `depth`, `height` | `(G,)` | Jaw opening; finger insertion depth along the approach axis past the grasp point; finger thickness. Metres. |
| `score` | `(G,)` | Ranking score, higher is better. `score_kind` names the semantics. |
| `object_id` | `(G,)` optional | Instance association when a method provides one, else `-1`. |
| `frame`, `gripper`, `key` | | Reference frame, gripper definition, `SampleKey`. |

- `validate()` checks finiteness, `SO(3)` membership and non-negative widths; it runs at the detector boundary in debug and test modes.
- `depth` here is insertion depth, never camera-z depth. The grasp point is the benchmark's grasp origin, not a robot TCP; conversion to a TCP is a function of `Gripper`.
- **Scores.** Raw GraspNet label scores are the minimum friction coefficient μ for force closure: lower is better and `-1` means invalid. They are not confidences. μ exists only inside `Supervision`. A detector's `score_kind` is declared by its head bundle (for example `predicted_quality`, an uncalibrated regression of a label-derived quantity); no method's score is assumed to be a calibrated probability.
- The GraspNet 17-D array `[score, width, height, depth, R(9), t(3), object_id]` appears only in `GraspGroup.from_graspnet_array()` / `.to_graspnet_array()`, i.e. at the import/export boundary.

</details>

## 8. Grasp pipeline abstraction

<details markdown="1">
<summary>§8 in full: 8.1 What the reference methods actually do · 8.2 Detector families · 8.3 Stage contracts of the seed-based family · 8.4 Head bundle · 8.5 Training-time flow and the assigner · 8.6 Ownership of constants · 8.7 Reference behaviour, controlled variants, reserved extensions</summary>

### 8.1 What the reference methods actually do

Read from source at the pinned commits (Appendix A). This table, not the papers' block diagrams, drives the abstraction.

| | GraspNet-baseline / SBG | GSNet (graspness_unofficial) | EconomicGrasp | HGGD |
|---|---|---|---|---|
| Input | Depth → cloud, xyz | Depth → cloud, constant-ones features | Same as GSNet | 4-channel RGB-D image 640×360 + 25 600-point cloud |
| Workspace crop | Box around **GT segmentation** in table frame, ±2 cm, also at test time | Same | Same | None |
| Scene encoder | PointNet++ (baseline) / 2 SA + 2 transformer blocks (SBG) → 1024 seeds, 256-d | Sparse U-Net, per-point 512-d | Same architecture as GSNet | 2-D encoder-decoder; no 3-D scene network |
| Proposal | 2-way "objectness" on the 1024 seeds; optional OBS needs a second segmentation network at test time (SBG) | Objectness + graspness per point → mask → FPS to 1024 | Same code as GSNet | Heat-map peaks → up to 48 **partial grasps** `(u, v, θ, width, Δz)` |
| Approach | 300 view scores; argmax in both modes | 300 view scores; train: one view sampled by multinomial from predicted scores; test: argmax | Same code as GSNet | None. Rotation is Euler-factored: θ in 2-D, 7×7 learned (γ, β) anchors in 3-D |
| Grouping | Cylinders over the **scene cloud**, xyz only, one per depth → explicit depth axis `(seed, 4)`; SBG adds four radii | One cylinder (r 0.05, 16 neighbours) **among the seeds**, with features; no depth axis | Same grouper + neighbour self-attention | Unrotated ball of radius = predicted width around the lifted centre; 512 points; image features attached and **detached** |
| Head | Score, angle class, width per depth, plus tolerance | SWAD: score and width grids `(12, 4)`, regression | Angle (13-way), depth (5-way), score (6-way) classification + width regression, with a validity mask | 49 anchor logits + per-anchor translation offsets |
| GT inside the training forward | Label matching runs mid-forward; the **crop rotation comes from the labels** (object pose × template); crops are centred at the matched **GT grasp point** rather than the seed | Same, but crops are centred at the seed | Same as GSNet | None; targets are computed after the forward from the network's own proposals |
| Output score | `score × P(objectness) × tolerance / 0.05` (SBG) | Max of regressed grid (unbounded) | Bin-centre-weighted sum of **raw logits** (no softmax) | Anchor sigmoid |
| Collision filter in shipped test path | Shipped command turns it off (SBG) | Default on in `test.py`, shipped command turns it off | Off | **On**, and also rejects grasps with ≤ 2 points between the fingers |
| Recipe | SBG: Adam 1e-3, OneCycle, 18 epochs, batch 2, BN-momentum decay; "validation" on `test_seen` | Adam 1e-3, ×0.95/epoch, 10 epochs, batch 4, 15 000 points, flip + x-rotation | Adam 1e-3, cosine, 20 epochs by default (README command: 10), batch 4, 20 000 points, flip | AdamW 1e-2, StepLR, 15 epochs, batch 4 × 2 accumulation, value clipping, BN-momentum decay; one optimizer over two networks |

Consequences:

1. GraspNet-baseline, GSNet, SBG and EconomicGrasp share one skeleton with a small number of real variation points (encoder output kind, proposal, view selection policy, grouping support and axes, head bundle).
2. **HGGD shares none of the four seed-based stages.** It has no per-point graspness, no 3-D scene backbone, no view classification and no approach-aligned crop; its proposal stage emits partial grasps, not points. It must be its own detector.
3. Three of four families crop the workspace with ground-truth segmentation at inference and HGGD does not. A comparison table that does not show this column is comparing different input conditions.

### 8.2 Detector families

Every detector implements the same outer contract and nothing more is required of it:

```python
class BaseGraspDetector(nn.Module):
    required_inputs: ClassVar[set[str]]  # e.g. {"points"} or {"rgb", "depth", "points"}

    def loss(self, observation, supervision) -> dict[str, Tensor]: ...
    def predict(self, observation) -> list[GraspGroup]: ...
    def configure_optimizers(self, cfg): ...  # optional override
```

Train and eval behaviour follow `nn.Module.training`; there is no constructor flag (upstream models need a second instance to validate during training).

| Family | Methods in v1 | Internal stages |
|---|---|---|
| Seed-based (`SeedGraspDetector`) | GraspNet-baseline, GSNet, SBG, EconomicGrasp, FineGrasp | backbone → proposal → view → (assigner, training only) → grouper → head bundle |
| Image-to-region (`HGGD`) | HGGD; RNGNet later if approved | image encoder → heat-map proposal → lift → region grouping → local encoder → anchor head bundle |
| Dense contact (`ContactGraspNet`) | Contact-GraspNet | PointNet++ encoder → per-point contact head bundle |

All three share structures, operators, post-processors, evaluators, engine and CLI. Only the first family uses the `PROPOSALS` / `VIEWS` / `GROUPERS` / `ASSIGNERS` slots. HGGD passes a `PartialGraspSet` (3-D centre, in-plane angle, width, depth offset, heat score) between its stages; it never fills a graspness field. If RNGNet later reuses HGGD's proposal or region grouping, those become registered slots at that point (principle 7).

The seed-based skeleton is extracted in M2, when EconomicGrasp becomes its second user, and re-examined in M3 against GraspNet-baseline and SBG. HGGD lands in M2 precisely to test that the outer contract, the data structures and the engine did not overfit to GSNet.

### 8.3 Stage contracts of the seed-based family

```python
class Backbone(nn.Module):
    output: ClassVar[Literal["dense", "subsampled"]]
    out_channels: int

    def forward(self, points: PointSet, feats: Tensor | None) -> PointFeatures: ...


class Proposal(nn.Module):  # scene features -> seeds
    requires: ClassVar[Literal["dense", "subsampled", "any"]]

    def forward(self, feats: PointFeatures, obs: Observation) -> SeedSet: ...
    def loss(self, seeds: SeedSet, sup: Supervision) -> dict[str, Tensor]: ...


class ViewStage(nn.Module):  # seeds -> one approach per seed
    template: ViewTemplate  # e.g. Fibonacci lattice, 300 views

    def forward(self, seeds: SeedSet) -> ApproachSet: ...
    def loss(self, approach: ApproachSet, matched: MatchedLabels) -> dict[str, Tensor]: ...


class Assigner:  # training only
    def __call__(self, seeds, approach, sup) -> MatchedLabels:
        ...
        # label grids at the chosen view, crop frame, crop centre, validity mask


class Grouper(nn.Module):
    support: ClassVar[Literal["seeds", "scene"]]
    out_axes: tuple[str, ...]  # ("seed",) or ("seed", "depth")
    out_channels: int

    def forward(
        self, support: PointFeatures, centres: Tensor, approach: ApproachSet
    ) -> GroupFeatures: ...


class HeadBundle(nn.Module):  # see 8.4
    in_axes: tuple[str, ...]
    codec: GraspCodec

    def forward(self, groups: GroupFeatures, seeds: SeedSet) -> RawPredictions: ...
    def loss(self, raw: RawPredictions, matched: MatchedLabels) -> dict[str, Tensor]: ...
    def decode(self, raw: RawPredictions, seeds: SeedSet, approach: ApproachSet) -> GraspGroup: ...
```

What the checker does with these declarations:

| Swap | Verdict | Reason given to the user |
|---|---|---|
| `SparseUNet14` → `PTv3` (with its decoder) under GSNet | Allowed | Both `dense`; channel counts are injected. Encoder-only PTv3 checkpoints (Sonata, Concerto, Utonia) are `subsampled` and are a different registered type. |
| `SparseUNet14` → `PointNet2` under `GraspnessFPS` | Rejected | Proposal requires dense features; the backbone returns a 1024-point subsample. Add a `FeaturePropagation` neck explicitly or change the proposal. |
| `CylinderGroup` → multi-scale cylinder, same axes | Allowed | `out_axes` unchanged. |
| `CylinderGroup` `(seed,)` under an operation/tolerance head | Rejected | Head expects `(seed, depth)`. |
| `SWADHead` ↔ `EconomicHead` | Allowed as a bundle swap | Checker also verifies the label asset each bundle needs. |
| `SWADHead` → operation/tolerance bundle | Allowed only with a depth-axis grouper and the tolerance asset | Checker lists every missing piece at once. |
| `GraspnessFPS` → object-balanced sampling | Allowed if an `instance` mask input is configured | Its provenance (`predicted` or `ground_truth`) is recorded on the run. |
| HGGD local PointNet → PointNet++ | Allowed inside HGGD | Same region contract. |
| HGGD heat-map proposal into a seed-based slot | Rejected | Different family: emits partial grasps, not seeds. |

Any change of structure or feature distribution requires retraining or declared fine-tuning. Loading a checkpoint into a swapped architecture fails on the key report (§12.1); nothing claims a checkpoint is portable across components.

### 8.4 Head bundle

A head bundle is one registered class that owns four things sharing one `GraspCodec`: the prediction module, the target encoder, the loss, and the decoder. `HEADS` contains bundles only; there is no way to configure a head with another bundle's decoder.

```yaml
head:
  type: SWADHead
  num_angle: 12            # the codec checks these against the dataset's label grid
  num_depth: 4
  width_target_scale: 10.0
  width_decode_gain: 1.2   # upstream applies 1.2 only in the decoder; kept as a named, documented option
  score_target: log_friction_batch_norm
  loss: {score: {type: SmoothL1, weight: 15.0}, width: {type: SmoothL1, weight: 10.0}}
```

- The codec is the single owner of the angle and depth grids, the flattening order of the angle × depth grid, the width scale, and the mapping to and from `GraspGroup`. Upstream writes `np.pi / 12` and `(i + 1) * 0.01` directly in the decoder, independent of `NUM_ANGLE` / `NUM_DEPTH`; a mismatch there changes results silently.
- Primitive losses inside a bundle are swappable where that is semantically safe (`SmoothL1` ↔ `L1`). The structure of the targets is not.
- A bundle may own non-gradient state. HGGD's (γ, β) anchors are learned by a shifting procedure during the first epochs and are stored in the checkpoint; they are buffers of its anchor head bundle.
- Proposal and view stages follow the same pattern on a smaller scale: each owns its targets and its loss, and loss weights are attributes of the stage that produces the loss.

### 8.5 Training-time flow and the assigner

Seed-based detectors cannot compute targets before the forward pass, because targets depend on which seeds and which view the network chose. The flow, as verified for GSNet:

1. Backbone, then proposal. Seeds are selected from **predictions** in training as well as in inference.
2. View stage. Training samples one view per seed from the predicted distribution; inference takes the argmax.
3. **Assigner** (training only). Transforms each object's grasp points and view grid by its GT pose, re-indexes object-frame views to the camera-frame template, matches each seed to its nearest label point, and returns the label grids at the chosen view **together with the crop rotation**. The training crop frame is object pose × template rotation, not the camera-frame template rotation that inference uses.
4. Grouper, with the rotation from step 3 in training or from step 2 in inference.
5. Head bundle forward; stage losses are summed.

Ground truth enters only at step 3 and only through the `Supervision` argument of `loss()`. `predict()` has no step 3. The assigner is a registered component because all five seed-based methods share its first half; their differences are parameters (distance threshold for a valid match: none in GSNet, 5 mm in EconomicGrasp; crop centre: seed or matched label point).

### 8.6 Ownership of constants

| Constant | Where upstream repeats it (GSNet) | Single owner here |
|---|---|---|
| 300 views | `loss_utils`, `label_generation` (`expand(…, 300)`), `generate_graspness`, label file layout | `ViewTemplate` of the view stage; the dataset declares its label grid |
| 12 angles, π/12 step | constant, decoder literal, `view_grasp_num = 48` | `GraspCodec` |
| 4 depths, 0.01 m step | constant, decoder literal, the same 48 | `GraspCodec` |
| 1024 seeds, graspness threshold 0.1 | module globals | proposal stage |
| Max width 0.1 m, finger height 0.02 m | label generation, decoder clamp, evaluator clip | `Gripper` |
| Voxel size 0.005 m | dataset default, `train.py`, `test.py` | sparse backbone |
| Points per frame | dataset default 20 000, CLI 15 000 | sampling transform |
| Workspace margin 0.02 m | four files | workspace transform (fingerprinted) |
| Friction thresholds 0.8 (point graspness), 0.6 (view graspness) | generator script, label matching | asset generator parameters; view-stage target encoder |
| Cylinder 0.05 / −0.02 / 0.04 / 16 | two files | grouper |
| Collision voxel 0.01 m, threshold 0.01 m, approach 0.05 m | `test.py`, detector defaults (which differ) | collision post-processor |

### 8.7 Reference behaviour, controlled variants, reserved extensions

**Reproduction configs replicate upstream behaviour, including quirks that move numbers.** Each quirk becomes a named option whose reference value is what upstream does; the "corrected" behaviour is a controlled variant with its own experiment name. Appendix A.2 lists the quirks found so far (for example: EconomicGrasp's score is a weighted sum of raw logits; its angle and depth heads carry an extra, never-supervised class that still takes part in the arg-max; GSNet normalises score targets with per-batch extrema).

**Reserved for after v1** — nothing below is implemented, but nothing in v1 blocks it:

| Extension | Hook already in the design |
|---|---|
| Multi-view track (Generalizing-Grasp) | `SampleKey` can carry a view-set id; `Observation` can hold several frames with extrinsics or a fused cloud with provenance; protocols are separate objects, so a scene-level, object-balanced evaluation is a new protocol and a new leaderboard, never a row in the single-view table. |
| Suction (SuctionNet-1B) | Same scenes and readers; a `SuctionGroup` output type and its evaluator sit beside `GraspGroup`. |
| Simulated clearing (GSR / DR) | Evaluators consume either dumped predictions or a policy callable; `GraspInferencer` is that callable. |

</details>

## 9. Operators, backends and dependencies

<details markdown="1">
<summary>§9 in full: 9.1 grasppanda.ops · 9.2 Sparse convolution · 9.3 Dependency policy</summary>

### 9.1 `grasppanda.ops`

Every operator has a pure-PyTorch reference implementation that runs on CPU and is the definition of correct behaviour. Faster backends are optional and must reproduce the reference on a shared test-suite before they are selectable.

| Operator | Reference behaviour | Optional backends | What parity must pin down |
|---|---|---|---|
| `furthest_point_sample` | Iterative; first sample is index 0. | pointops, `torch_cluster.fps`, `pytorch3d.sample_farthest_points` | Start index and tie-breaking decide which points become seeds, hence checkpoint parity. The upstream CUDA kernel also skips points with squared norm ≤ 1e-3; exposed as a compatibility flag. |
| `ball_query`, `cylinder_query` | Chunked pairwise test; first `nsample` hits in index order; documented padding rule. Cylinder: axis is column 0 of the given rotation, axial range `[hmin, hmax]`, grouped coordinates centred, divided by radius and rotated into the crop frame. | pytorch3d, pointops; a CUDA kernel under `grasppanda._ops` only if profiling shows the need | Ordering, padding, axis convention. |
| `knn`, `three_interpolate` | `cdist` + `topk`, chunked; 0-indexed (upstream `knn` is 1-indexed). | `torch_cluster.knn`, `pytorch3d.knn_points` | Tie order. |
| `voxelize` | Floor of `xyz / voxel_size`, `unique` with inverse map; representative rule `first` or `mean`. | backend-native | Representative rule when features are not constant. |
| `grasp_nms` | Translation / rotation-distance suppression. | — | Used as a model post-processor only. The evaluator has its own pinned copy (§11.1) so that changing this operator can never change AP. |

`runtime.ops: {fps: auto}` selects per operator; `auto` means the fastest installed backend that passed parity. The choice is recorded in `env.json`.

A pure-PyTorch cylinder query over 1024 seeds × 20 000 points is on the order of 10⁷–10⁸ pairwise tests per batch, which is practical on a GPU with chunking; no compiled extension is assumed for v1.

### 9.2 Sparse convolution

Sparse backbones are written against a thin facade (`SubMConv3d`, `SparseConv3d`, `SparseInverseConv3d`, batch norm, activation, tensor construction, `features_at(inverse_map)`).

| Backend | Role | Facts (checked on PyPI 2026-10-12) |
|---|---|---|
| spconv 2.3.8 | Default. | Wheels for cu118 / cu121 / cu124 / cu126 and CPU, CPython 3.9–3.13, Linux x86-64; released 2024-12-15; no cu128 or newer wheel. |
| WarpConvNet | Optional, later. | 1.8.2 on PyPI (2026-08-10), by the author of MinkowskiEngine. `LICENSE.md` is Apache-2.0 while the PyPI classifier says "Other/Proprietary". PyPI has a source distribution only; binary wheels exist for cu124–cu130, none for cu118. |
| MinkowskiEngine 0.5.4 | Legacy, opt-in. Only to load author checkpoints and to record reference activations. | Unmaintained; needs patched system headers on CUDA 12; blocks ONNX/TensorRT export. Never imported by default. |

`SparseUNet14` is the first shared backbone: the Graspness `MinkUNet14D` and EconomicGrasp `TDUnet` are architecturally identical (BasicBlock, one block per stage, planes 32-64-128-256-192-192-192-192, 5³ stem, four stride-2 down/up stages, 1×1 projection to 512).

`tools/convert_checkpoint.py` maps MinkowskiEngine weights to the facade layout. It is accepted only with a stage-by-stage activation comparison against the legacy backend on fixed frames (M1 gate). The dev machine's driver (525.89.02) fixes the development environment to cu118 wheels until a cu12x build is shown to work on it.

### 9.3 Dependency policy

| Group | Packages |
|---|---|
| Core | `torch` (installed by the user for their CUDA), `numpy` (1.26 and 2.x), `scipy`, `pyyaml`, `pillow`, `tqdm`, `packaging`. |
| `spconv-cu118`, `spconv-cu121`, `spconv-cu124`, `spconv-cu126`, `spconv-cpu` | The matching spconv wheel. |
| `warp` | `warpconvnet`. |
| `vis` | Viewer and rendering backend (choice deferred to M0; must work headless for docs and the demo GIF). |
| `log` | `tensorboard`, `wandb`. |
| `parity` | `graspnetAPI`, for evaluator parity tests only. |
| `dev`, `docs` | Lint, test and site tooling. |

Mesh and SDF readers needed by the evaluator are implemented in-tree with NumPy so that evaluation stays in the core install. Forbidden: committed CUDA installers; vendored top-level packages named `pointnet2` or `knn_pytorch` (each upstream installs one under the same global name and the last install silently wins); any `sys.path` manipulation; import-time side effects such as argument parsing.

</details>

## 10. Datasets and derived assets

<details markdown="1">
<summary>§10 in full: 10.1 GraspNet-1Billion reader · 10.2 Derived assets and grasppanda prepare · 10.3 Transforms · 10.4 GraspClutter6D reader</summary>

### 10.1 GraspNet-1Billion reader

- Samples are addressed by `SampleKey(scene, camera, frame)` through split manifests shipped with the package: `train` (scenes 0–99), `test_seen` (100–129), `test_similar` (130–159), `test_novel` (160–189), plus `train_dev` / `val_dev` (§11.5) and `sample`.
- Raw data stays where it is and is never modified. Layout expected: `scenes/scene_XXXX/{realsense,kinect}/{rgb,depth,label,meta,annotations,camera_poses.npy,cam0_wrt_table.npy}`, `models/`, `grasp_label/`, `collision_label/`.
- Grasp labels per object: points `(Np,3)`, friction scores and widths `(Np,300,12,4)`; collision masks per scene, indexed by the object's position in the frame's `cls_indexes`. The angle and depth grids (`a·π/12`, `(d+1)·0.01 m`) are declared once in the dataset's label-grid descriptor; upstream code never reads them from the label file and re-types them in the decoder.
- Labels are served from a memory-mapped, per-object store with an LRU, built once by `prepare` and verified value-for-value against the source arrays. This replaces "load all 88 objects into RAM" (the reason upstream needs a simplified label set to fit in memory).

### 10.2 Derived assets and `grasppanda prepare`

An asset is a directory `<asset_root>/<dataset>/<kind>/<hash>/` with a `manifest.json` recording generator name and version, parameters, input fingerprints, preprocessing fingerprint, units and frames, whether ground truth was used, and per-shard completion. Writes are atomic; generation is resumable; a missing shard is an error, never an all-zero label.

`hash = sha256(generator id + version, parameters, preprocessing spec, dataset fingerprint)`.

**Point-wise graspness is the motivating case.** Upstream stores one array per frame, row-aligned to `cloud[depth > 0 & workspace_mask]`, with no key (`graspnet_dataset.py:149,187`). The mask recipe is copy-pasted in four files. Changing the outlier margin, the mask rule or the camera misaligns every label without any error. In addition, each value is min-max normalised over the masked set, so even the values depend on the mask.

Design:

| Aspect | Upstream | GraspPanda |
|---|---|---|
| Key | Row position in the masked cloud | Pixel: a dense `H×W` map, `NaN` outside the generation mask |
| Mismatch | Silent | A requested pixel that is `NaN`, or a manifest whose preprocessing fingerprint differs from the dataset's, is an error |
| dtype / size | float64, 223 GB for both cameras on the dev machine | float16, compressed; dtype is a hashed parameter |
| Cost | Re-reads every object's full label file for every frame | Object-level graspness is computed once per scene and reused for its 256 frames; only the scene-point → object-point nearest-neighbour step is per frame |

The existing upstream-format labels on disk can be imported (`prepare graspness --import-legacy`), re-keyed to pixels by recomputing the mask and asserting row counts. Imported assets are marked `legacy-import` because their generator version is unknown; they are for parity debugging, not for zoo runs.

Other assets use the same store: tolerance labels for GraspNet-baseline and SBG, compact per-scene labels for EconomicGrasp, 2-D targets for HGGD, GraspClutter6D graspness.

### 10.3 Transforms

The pipeline has an explicit **label-alignment boundary**: everything up to and including workspace masking is fingerprinted and tied to asset hashes. Sampling and augmentation come after it and carry labels with them (indices are gathered; rigid transforms are applied to points and object poses together).

Reference recipes follow upstream *code*, and each axis is named in the config: Graspness uses a horizontal flip plus a ±30° rotation about the camera x-axis (the source comment says z; the matrix is an x-rotation); EconomicGrasp uses the flip only.

### 10.4 GraspClutter6D reader

Facts below were read from the dataset itself and from `SeungBack/graspclutter6dAPI` at `ddb2ee7`. The dataset README documents only download and installation, not the layout.

| Topic | Fact |
|---|---|
| Layout | BOP style: `scenes/<6-digit id>/{rgb,depth,label,mask,mask_visib}/`, `scene_camera.json`, `scene_gt.json`, `scene_gt_info.json`; 1000 scenes. Object models in millimetres (`models/`) and metres (`models_m/`), 200 objects. Grasp labels reuse the GraspNet format (`grasp_label/obj_%06d_labels.npz`, `collision_label/%06d.npz`). |
| Frames and cameras | 52 images per scene: `image_id = 4 · ann_id + c`, `ann_id` 0–12, `c` = 1 RealSense D415, 2 RealSense D435, 3 Azure Kinect, 4 Zivid. So 13 frames per camera per scene. |
| Depth | 16-bit, with a per-camera `depth_scale` in `scene_camera.json` (1 for the RealSense ids, 0.1 for Azure Kinect and Zivid). Not uniformly millimetres. |
| Resolution | Differs by camera (1920×1080, 3840×2160, 1920×1200 in the scene inspected). |
| Poses | `scene_gt.json`: object → camera, translation in millimetres. Scenes contain repeated instances of one object id. |
| Splits | `split_info/grasp_train_scene_ids.json`: 413 scenes; `grasp_test_scene_ids.json`: 235; disjoint. This matches the paper's 413 / 235. The remaining 352 scenes belong to neither grasp split. The JSON files are authoritative. |
| Licence | CC BY-SA 4.0; about 220 GB on Hugging Face. |

Reader requirements that follow: camera is selected by image-id arithmetic, never by directory; depth scale and intrinsics are read per image; instances, not object ids, key the labels; nothing is ever written into the dataset tree (the official tool writes cache pickles there when they are missing).

</details>

## 11. Evaluation

<details markdown="1">
<summary>§11 in full: 11.1 The evaluator is re-implemented, not vendored · 11.2 Parity gate · 11.3 Protocols · 11.4 Result records and post-processing variants · 11.5 Validation, statistics and latency</summary>

### 11.1 The evaluator is re-implemented, not vendored

Reasons:

1. **Licence.** The force-closure code inside `graspnetAPI` is Dex-Net, whose files carry the UC Berkeley notice permitting use "for educational, research, and not-for-profit purposes" only. It cannot be placed in an Apache-2.0 package, and copying it would break the clean-room rule.
2. **Weight.** Importing the official evaluator pulls in open3d, autolab_core, cvxopt (imported unconditionally, never used for two-finger grasps), scikit-learn, scikit-image, matplotlib, IPython, transforms3d, and a compiled NMS extension, with `numpy==1.23.4` pinned in the PyPI release.
3. **Robustness.** See the table at the end of this section.

**Pinned reference semantics.** `graspnetAPI` evaluation path as of v1.2.10, unchanged through 1.2.11 and through the current upstream head (`eb57dd2`, which differs from 1.2.11 only in `setup.py`). History matters: commit `6704cc8` (2020-11-13) reduced the friction levels from twelve to six and changed per-object selection; the global top-50 cut and the width clip arrived by 2020-12-21 (`b34c54b`). Numbers produced with earlier API versions are not comparable and are labelled as such in zoo tables.

**Specification** (order is part of the definition):

1. Clip width to `[0, 0.1]` m.
2. Global grasp NMS in descending score order: a kept grasp suppresses later grasps whose translation distance is `< 0.03` m **and** whose rotation angle is `< 30°` (both strict). Grasps on different objects can suppress each other.
3. Assign each grasp to the object whose model point (model voxel-downsampled at 0.008 m) is nearest to the grasp translation.
4. Per object, keep the 10 highest-scoring grasps.
5. Globally, keep grasps whose score is at least the 50th-ranked score (ties included).
6. Collision and emptiness, against **ground-truth model points plus a synthetic table slab**, cropped to the assigned object's bounding box ± 0.05 m. Gripper: finger width 0.01, height 0.02, depth base 0.02 m. Fewer than 10 points between the fingers counts as empty. Failing grasps score −1. The grasp's own `height` and `object_id` columns are ignored.
7. Force closure: a two-contact antipodal test on the object's signed-distance field along the fingertip line `t + depth · x̂`, for μ from 1.2 down to 0.2. The grasp's score is the smallest μ that still holds, or −1.
8. Re-sort by predicted score. For `k = 1…50` and each μ in `{0.2, 0.4, 0.6, 0.8, 1.0, 1.2}`: `precision@k = #(0 < score ≤ μ in the top k) / k`. Missing grasps count as failures.
9. `AP` is the mean over scenes × 256 frames × 50 × 6. `AP0.8` and `AP0.4` are the same mean restricted to μ = 0.8 and μ = 0.4; they are a community convention, not computed by the official API.

**Data the evaluator needs.** Only each object's 100³ SDF, its grid transform, and the 1000-voxel surface sample that is frozen inside the official `dex_models/*.pkl` (it determines out-of-bounds SDF values, and rebuilding it draws a fresh unseeded random sample). `grasppanda prepare eval-cache` extracts these with a restricted unpickler that maps the pickled classes to inert stubs: no upstream code is imported or executed. The cache is about 8 MB per object; the official path loads 1.2–3.2 GB of pickles per scene.

| Official behaviour | Here |
|---|---|
| Models and Dex-Net objects reloaded for every scene and worker; `eval_all` reads about 172 GB. | Per-object cache, loaded once per worker. |
| NMS is a compiled extension that never frees its buffers; worker pool has no task limit; an exception appears only after all scenes finish; a killed worker makes `join()` block forever. This is a plausible, **not established**, explanation for the reported stall near scene 188. | Pure NumPy NMS; workers recycled per scene chunk; time-outs; errors raised immediately with the `SampleKey`. |
| Empty prediction array raises `IndexError` (by reading; forks patch this). Missing file raises. | Empty prediction is valid and scores zeros, as the forks do. Missing prediction is an error and lowers coverage. Both documented. |
| Ties are broken by unstable sorts (`std::sort`, `argsort`), so tied inputs are platform-dependent. | Stable, documented tie rule. |
| `np.int` in the SDF builder; `sklearn` meta-package; `transforms3d==0.3.1`. | NumPy 1.26 and 2.x; no such dependencies. |

`GraspClutter6DEvaluator` shares steps 2–5 and 7–9. It differs as the official tool does: 13 frames per camera, `max_width = 0.14` m, a pre-NMS filter keeping grasps within 0.05 m of any object model point, no table in the collision scene, float64 poses, and a normalised closing axis. A workspace crop is part of the reference *input pipeline* there, not of the evaluator.

### 11.2 Parity gate

No AP produced by this toolbox is reported before this gate passes.

- **Oracle.** The official API in its own pinned environment, used only to record outputs. Recorded per frame: post-NMS selection, object assignment, collision mask, per-grasp friction score, and the `(50, 6)` precision table.
- **Fixtures.** Dumps from at least two detectors with different score distributions (author checkpoints), covering seen, similar and novel scenes and both cameras; plus constructed edge cases: empty frame, fewer than 50 grasps, out-of-range widths, tied scores, slightly non-orthonormal rotations.
- **Criterion.** Stage-by-stage equality of the recorded intermediates and exact equality of the precision tables for tie-free inputs; equality up to the documented tie rule otherwise.
- **Known fine print to replicate.** Object poses pass through quaternion → Euler → matrix and are stored as float32; SDF query coordinates are cast to float32 before the grid transform; model points are voxel-downsampled with Open3D semantics. Each is reproduced deliberately and covered by a fixture.
- **CI.** A small fixture set built on synthetic objects (so no dataset content is committed), with expected values recorded once from the oracle.

### 11.3 Protocols

A protocol is a frozen YAML file under `grasppanda/evaluation/protocols/`.

| Protocol | Unit | Splits | Evaluator constants | Headline report |
|---|---|---|---|---|
| `graspnet1b-sv-v1` | scene × camera × frame, 256 frames | seen 100–129, similar 130–159, novel 160–189 | §11.1 | RealSense and Kinect × seen / similar / novel × AP / AP0.8 / AP0.4 |
| `graspclutter6d-sv-v1` | scene × camera × frame, 13 frames | 235 test scenes from the split JSON | §11.1, GC6D differences | D435 as in the paper; other cameras reported when run |

Each protocol defines two **input presets**: `reference` (inputs as the upstream family uses them, including the GT-derived workspace crop; marked `gt_assisted`) and `visual` (no GT-derived input). Zoo entries are produced under `reference` for comparability with publications; controlled tables show both.

Documented data caveats: `test_novel` scenes 188 and 189 contain only seen or similar objects (graspnetAPI issue 47); some label grasps are themselves in collision (issue 64).

### 11.4 Result records and post-processing variants

- Raw predictions are written once per run. Each entry of `evaluation.variants` is a post-processing chain applied to those predictions and scored separately. Reference configs define `raw` and `cd` (model-free collision filter on the observed cloud), so collision-detection on and off are always both reported.
- The model-side **collision filter** (uses the sensor cloud; part of inference) and the evaluator-side **collision check** (uses GT models; part of scoring) have different names everywhere to prevent the confusion that produced mixed baselines in recent papers (GSNet's 67.12 with filter quoted against others' numbers without).
- A result record (`results/<variant>.json`) holds: protocol id and input preset, dataset version, camera, split, coverage (frames scored / frames in manifest), post-processing chain with parameters, provenance of workspace and segmentation inputs, evaluator version, config hash, code commit, weight sha256, seed, and per-scene metric arrays.
- **Partial coverage prints no headline AP**, only per-scene numbers with the coverage figure.

### 11.5 Validation, statistics and latency

- **Validation split.** `train_dev` / `val_dev` are a fixed scene-level partition of training scenes 0–99. Frames of one scene never straddle the two. No test split is ever used for model selection; `save_best` on a test split is a checker error. (graspnet-baseline and SBG evaluate on `test_seen` during training; not replicated.)
- **Reproduction recipes** train on all 100 training scenes for a fixed schedule and use the final checkpoint, as the authors do. Model selection on `val_dev` is for controlled experiments that need it.
- **Seeds and variance.** Controlled experiments run three seeds. Uncertainty is computed with the scene as the cluster (per-scene AP, then a paired scene-level bootstrap for method differences), because the 256 frames of a scene are not independent. Seed-to-seed spread is reported separately.
- **Latency** is reported per stage (preprocessing, encoder, proposal, view, grouping, head, decode, each post-processor) at batch size 1 after warm-up with CUDA synchronisation; device and precision are recorded.
- **Optional metric.** SBG's scale-wise evaluation is available as `ScaleAP` under its own name. It buckets by predicted width and uses top-5 per object and 20 ranks, so it is never shown in an AP column.

</details>

## 12. Engine, CLI and Python API

<details markdown="1">
<summary>§12 in full: 12.1 Trainer · 12.2 CLI · 12.3 Python API · 12.4 First result in five commands</summary>

### 12.1 Trainer

One readable file of roughly 300 lines, in the spirit of Detectron2's `plain_train_net.py`; everything optional is a hook.

```python
for epoch in range(start_epoch, cfg.train.epochs):
    hooks.before_epoch(state)
    for batch in train_loader:
        with autocast(enabled=cfg.train.amp):
            losses = model.loss(batch.observation, batch.supervision)  # dict[str, Tensor]
            total = sum(losses.values())
        scaler.scale(total / cfg.train.accumulate).backward()
        if state.should_step():
            clip_gradients(model, cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        hooks.after_iter(state, losses)
    hooks.after_epoch(state)  # scheduler, checkpoint, evaluation
```

| Concern | Behaviour |
|---|---|
| Hooks | `Checkpoint`, `Logger` (text, TensorBoard, W&B), `EMA`, `Evaluation`, `Visualizer`, `StageTimer`, `NaNGuard`. |
| Resume | Restores model, optimizer(s), scheduler, scaler, EMA, RNG states, epoch/iteration and best metric. Refuses to resume when `config_hash` differs unless explicitly allowed. |
| `load_from` | Weights only. Always prints a matched / missing / unexpected key report; no silent `strict=False`. |
| Distributed | `torchrun` DDP. Sampler and worker seeds derive from `runtime.seed` and rank. Predictions padded by the distributed sampler are de-duplicated by `SampleKey` before evaluation. |
| AMP | Off by default for reproduction configs; a stage can opt out with a local `autocast(False)` where sparse or geometric ops are unstable in half precision. |
| Multiple optimizers | A detector may override `configure_optimizers()` to return named parameter groups or several optimizers; the loop does not assume one. |
| In-training evaluation | Runs `predict` on the `val` manifest (scene-level hold-out from training scenes, §11.5), scores it with the configured evaluator and keeps the best checkpoint by the configured metric. |

### 12.2 CLI

`grasppanda <command> <config-or-name> [--opts k=v ...]`

| Command | Purpose |
|---|---|
| `train` | Train from a config. |
| `test` | Evaluate weights on a split under a protocol; writes a result record (§11.4). |
| `infer` | Run on a folder of RGB-D frames or a dataset split and dump `GraspGroup`s (native `.npz`, or GraspNet 17-D `.npy` for interchange). |
| `demo` | One-frame prediction and visualisation; fetches the sample data and zoo weights if missing. |
| `prepare` | Download the sample subset, verify a dataset tree, generate derived assets (graspness, tolerance, method-specific labels). Resumable and content-hashed (§10.2). |
| `print-config` | Print the resolved config; `--check` runs all validation; `--list <registry>` shows registered components and their status. |
| `analyze` | Parameters, FLOPs, peak memory, and per-stage latency (warm-up, CUDA sync, batch size 1, device and precision recorded). |
| `env` | Print the interpreter, PyTorch build, GPUs and optional backends, for bug reports. |
| `sweep` | Expand a grid of overrides × seeds, run or resume each cell, and aggregate into Markdown / CSV / LaTeX tables with scene-clustered confidence intervals (§11.5). |

### 12.3 Python API

```python
from grasppanda import GraspInferencer

inferencer = GraspInferencer(
    "gsnet_sunet14_g1b-realsense_10e"
)  # zoo name: config + sha256-verified weights
grasps = inferencer(
    depth=depth, rgb=rgb, intrinsics=K, depth_scale=1000.0
)  # GraspGroup, camera frame, metres
inferencer.visualize(grasps, top_k=50)
```

When a user feeds their own frames, the inferencer compares the input conditions with those the weights were trained under and warns on differences that matter (for example: weights trained on clouds cropped with a GT-derived workspace mask, input cloud uncropped).

### 12.4 First result in five commands

```bash
pip install -e ".[spconv-cu118,vis]"            # torch matching your CUDA must already be installed
grasppanda prepare sample                             # small GraspNet-1B subset with labels and object models
grasppanda demo gsnet_sunet14_g1b-realsense_10e           # downloads weights, shows predicted grasps on a cluttered scene
grasppanda train gsnet/gsnet_sunet14_g1b-sample_smoke # minutes-long run that proves the training loop end to end
grasppanda test  gsnet_sunet14_g1b-realsense_10e --opts data.test.dataset.split=sample
```

The sample-split result is labelled as partial coverage and prints no headline AP (§11.4).

</details>

## 13. Model zoo, weights, documentation and brand

<details markdown="1">
<summary>§13 in full: 13.1 Admission gate · 13.2 Per-method files · 13.3 Weights · 13.4 Documentation · 13.5 Brand and README</summary>

### 13.1 Admission gate

A zoo entry is created only when all of the following exist: config, weights with sha256, training log, result record produced by the pinned evaluator, and the run snapshot.

Two measurements are required per method and camera, because author-reported numbers are not always reproducible even with author weights (HGGD RealSense seen: its README lists 59.36 as "In paper" and 64.45 as "In repo", the arXiv version prints 64.45, and two third-party re-evaluations give 58.35 and 63.37):

| Leg | Compares | Purpose |
|---|---|---|
| A. `author-ckpt` | Author weights evaluated here vs. the author-reported number | Validates our data pipeline, model port and evaluator against the publication. |
| B. `reproduced` | Trained here vs. author-reported number, with leg A alongside | Certifies training. |

Gate: `|reproduced − reported| ≤ 1.0 AP` on each of seen / similar / novel. If the gate fails, the entry may still be published, but its README must state the gap, leg A's number, and the diagnosed or suspected cause. Collision-detection on/off is always a separate row; a number without its CD flag is never published.

### 13.2 Per-method files

`configs/<method>/README.md` — paper, figure, results table (backbone, camera, seen/similar/novel AP, CD flag, provenance class, links to config / weights / log).

`configs/<method>/metafile.yaml` — machine-readable source for the generated top-level zoo table and the weight index:

```yaml
method: GSNet
paper: {title: "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection", venue: ICCV, year: 2021}
upstream: {repo: graspnet/graspness_unofficial, commit: b5abf5a, license: "SJTU academic, non-commercial"}
implementation: clean-room
models:
  - name: gsnet_sunet14_g1b-realsense_10e
    config: configs/gsnet/gsnet_sunet14_g1b-realsense_10e.yaml
    protocol: graspnet1b-sv-v1
    camera: realsense
    weights: {url: ..., sha256: ..., license: CC-BY-NC-SA-4.0}
    log: ...
    results:
      - {provenance: reported,   collision_detection: false, seen: 65.70, similar: 53.75, novel: 23.98, source: paper}
      - {provenance: reproduced, collision_detection: false, seen: null,  similar: null,  novel: null, evaluator: grasppanda-eval-1.0, seeds: 3}
```

### 13.3 Weights

Hosted on Hugging Face Hub with GitHub Releases as mirror; URLs are pinned to a revision, never to a moving branch; sha256 is verified on download. No self-hosted CDN (OpenMMLab's download domain expired in June 2026 and took every checkpoint link with it).

### 13.4 Documentation

MkDocs Material on GitHub Pages, built in CI, API reference generated from docstrings.

Get Started · User Guides (config, dataset preparation, train/test, inference, visualisation) · Advanced Guides (add a backbone / head bundle / loss / dataset / metric / hook; data flow) · Model Zoo · API · FAQ · Changelog.

English is the source language. `README_zh-CN.md` mirrors `README.md`; CI fails when their section structure diverges.

### 13.5 Brand and README

- In robotics "Panda" usually means the Franka arm. Logo and tagline therefore show the *animal* — a panda holding or formed by a parallel-jaw gripper — and the README states once that the toolbox is not tied to any robot.
- Assets in `resources/`: logo (SVG source, PNG export, light and dark variants), README banner, architecture overview diagram, demo GIF of predictions on a cluttered scene.
- README order: language switch · logo and tagline · badges (CI, PyPI, docs, license, Python) · quick links · demo GIF · news · features · overview table (methods, components, datasets) · model-zoo summary · install and quick start · tutorials · citation (BibTeX + `CITATION.cff`) · acknowledgements · license.

</details>

## 14. Engineering quality

<details markdown="1">
<summary>§14 in full: 14.1 Tests (pytest, CPU) · 14.2 Continuous integration · 14.3 Licensing and clean-room policy · 14.4 Project files</summary>

### 14.1 Tests (pytest, CPU)

| Area | What is asserted |
|---|---|
| Config | Merge rules, type-change replacement, depth limit, sibling-base conflict, interpolation after override, every error path with its suggestion text. |
| Registry | Auto-import, duplicate rejection, cross-registry hint, missing-extra message, `custom_imports`. |
| Structures | `GraspGroup` ↔ 17-D array round trip; rotation validity; frame transforms; `PointFeatures` identity checks. |
| Ops | Reference vs. each optional backend (indices, ordering, tie-breaking, padding); cylinder query against a brute-force oracle. |
| Transforms / losses | Deterministic outputs on fixed seeds; label alignment survives augmentation. |
| Models | Tiny-input forward and backward for every detector; head-bundle encode → decode round trip on synthetic labels. |
| Contracts | Incompatible configs must fail, each with the expected message (negative tests are first-class). |
| Evaluator | Value-level agreement with recorded official outputs on small fixtures (§11.2). |
| Checkpoint | Save → resume reproduces the next iteration's loss. |

### 14.2 Continuous integration

- `lint`: pre-commit with ruff and codespell.
- `test`: Python 3.10 and 3.12 × NumPy 1.26 and 2.x × oldest and newest supported torch, CPU wheels (spconv publishes a CPU wheel, so sparse backbones are covered).
- `smoke` ("TIPC-lite"): train a few iterations on the sample subset → infer → compare predictions and metrics with a stored record within a stated tolerance.
- `docs`: build the site; fail on broken links.
- `release`: on tag, build and publish to PyPI through trusted publishing; the version comes from the tag, so PyPI cannot lag behind GitHub.

### 14.3 Licensing and clean-room policy

| Item | Position |
|---|---|
| Toolbox code | Apache-2.0. |
| Method implementations | Written here from papers and from behavioural study of upstream code. No upstream file is copied or adapted line by line. Equivalence is proven by tests against upstream *outputs* (activations and predictions recorded in a separate environment), not by shared code. |
| Non-commercial upstream (graspnet-baseline, graspness_unofficial: SJTU academic licence; Contact-GraspNet, M2T2: NVIDIA non-commercial) | Reference only. Several repositories with an MIT or Apache headline (SBG, EconomicGrasp, GraNet, FineGrasp; FGC-GraspNet is reported as MIT by GitHub but has no licence file) vendor files derived from graspnet-baseline, whose licence claims rights over derivatives, so their headline licence is not relied upon either. |
| No licence file (RNGNet, Generalizing-Grasp, DexGraspNet2) | All rights reserved. Author weights may be evaluated locally; nothing is redistributed. |
| Official evaluator | `graspnetAPI` bundles Dex-Net code under the UC Berkeley research-only licence (headers in `graspnetAPI/utils/dexnet/**`). It is therefore **not vendored**; see §11.1. |
| Weights trained here | GraspNet-1Billion is CC BY-NC-SA 4.0; weights are released under the same terms. |
| Records | Each `metafile.yaml` records upstream repo, commit and licence, weight licence and dataset licence. `NOTICE` and `THIRD_PARTY.md` aggregate them. |

### 14.4 Project files

`CONTRIBUTING.md`, issue forms, PR template, `CHANGELOG.md` (Keep a Changelog), `CITATION.cff`, `projects/example_project/` (a minimal out-of-tree head bundle with its config and test).

</details>

## Appendix A. Sources and upstream facts

<details markdown="1">
<summary>Appendix A in full: A.1 Pinned sources · A.2 Upstream behaviours a reproduction must replicate or consciously change</summary>

### A.1 Pinned sources

The reference implementations were read at these commits; they are never copied (clean-room rule, §14.3).

| Source | Commit |
|---|---|
| graspnet/graspness_unofficial | `b5abf5a` |
| EconomicGrasp | `7361bb6` |
| THU-VCLab/HGGD | `69580ff` |
| mahaoxiang822/Scale-Balanced-Grasp | `995d6f8` |
| graspnet/graspnet-baseline | `fb2f089` |
| graspnet/graspnetAPI | see §11.1 |
| Contact-GraspNet (PyTorch port) | `fa2d966` |
| FineGrasp | — |
| PaddleSeg `release/2.10`, MMEngine `main`, Detectron2 `main`, Pointcept `main` | fetched 2026-10-12 |

### A.2 Upstream behaviours a reproduction must replicate or consciously change

| Method | Behaviour | Where | Handling |
|---|---|---|---|
| GraspNet-baseline, GSNet, SBG, EconomicGrasp | Workspace crop is built from the GT segmentation mask at test time. | `dataset/graspnet_dataset.py` (`get_data`), `utils/data_utils.py:121-156` | Explicit workspace transform with `ground_truth` provenance; `gt_assisted` on results. |
| GSNet | Train/eval is a constructor flag; a training-mode model cannot decode. | `models/graspnet.py:24-35,84-88` | `loss()` / `predict()` on one instance. |
| GSNet | Training view is sampled by multinomial from detached, min-max-normalised predicted scores. | `models/modules.py:45-58` | View-stage option `train_selection: multinomial`. |
| GSNet | Score targets are `log(u_max / μ) / log(u_max / u_min)` with extrema taken over the whole batch. | `utils/label_generation.py:133-138` | Named target option; per-dataset constants as a controlled variant. |
| GSNet | Seed → label matching has no distance threshold; a seed on the table still receives a full label grid. | `utils/label_generation.py:73-81` | Assigner parameter `max_match_distance: null`. |
| GSNet | Stored graspness covers every masked point including table and background (nearest object point, no threshold), min-max normalised per frame without epsilon. | `dataset/generate_graspness.py:102-118` | Generator parameters; loss mask stays as upstream. |
| GSNet, SBG, EconomicGrasp | Decoder multiplies predicted width by 1.2; the loss target has no such factor. | `models/graspnet.py:109` and equivalents | Codec option `width_decode_gain`. |
| GSNet | `test.py` enables collision filtering by default (voxel 0.01 m), while the shipped command and README numbers disable it. | `test.py:29,89-93`, `command_test.sh` | Both variants always reported. |
| GSNet | Augmentation comment says z-axis; the matrix rotates about the camera x-axis. | `dataset/graspnet_dataset.py:71-91` | Follow the code; axis named in config. |
| GSNet, EconomicGrasp | FPS starts at index 0 and ignores points with squared norm ≤ 1e-3. | `pointnet2/_ext_src/src/sampling_gpu.cu:90-106` | Reference operator flag. |
| EconomicGrasp | Output score is Σ cₖ · logitₖ with bin centres cₖ; no softmax is applied anywhere. | `models/economicgrasp.py:134-137` (HEAD) | Codec option `score_decode: logit_weighted`; `softmax_expectation` as a controlled variant. |
| EconomicGrasp | Angle and depth heads have 13 and 5 outputs; the extra class is never a supervised target but takes part in the decoder's arg-max at the pinned commit. Upstream commit `4119bdc` (2026-04-12) restricts the arg-max to the supervised classes; the published numbers predate it. | `models/modules_economicgrasp.py`, `models/economicgrasp.py:139-146` (HEAD of the pin) | A named decoder option; the reference config states which behaviour it replicates. |
| EconomicGrasp | "Economic" labels keep all 300 views; only angle × depth is collapsed to the best pair per (point, view). Scores are quantised to 0.1 and widths to truncated millimetres (uint8). | `dataset/generate_economic.py:96-120` | Asset generator parameters. |
| EconomicGrasp | A seed is a valid training sample only within 5 mm of its matched label point; the literal 0.005 is unrelated to the voxel size. | `utils/label_generation.py:152-158` | Assigner parameter with its own name. |
| EconomicGrasp | Collision filter reuses the sparse-conv voxel size (0.005 m); the test dataset ignores `--voxel_size`. | `test.py:28-38,87-91` (HEAD) | Separate, separately named parameters. |
| GraspNet-baseline, SBG | Training crops are centred at the matched GT grasp point, inference crops at the seed. | `models/graspnet.py:106-111` | Assigner output `crop_centre`. |
| SBG | In-training "validation" runs on `test_seen`. | `train.py:78-79,200-235` (HEAD) | Not replicated; scene-level hold-out (§11.5). |
| SBG | `test_similar` is defined as scenes 130–190 in its dataset class. | `dataset/graspnet_dataset.py:45-46` | Not replicated; official manifests. |
| SBG | Scale-balanced weight: bin of the best grasp over all views; widths outside every bin silently fall into bin 0; prior loaded from a relative path at import. | `models/loss.py:22-46` | Prior becomes a hashed statistic asset computed from training labels. |
| SBG | `evaluate_scale.py` buckets by *predicted* width (< 0.04, 0.04–0.07, ≥ 0.07 m), keeps top-5 per object and uses `TOP_K = 20`. | `evaluate_scale.py:51,113,156-171` | Offered as an optional metric under its own name; never mixed with AP. |
| SBG | OBS needs a second segmentation network at test time and its batch-norm layers are left in training mode. | `test_obs.py:63-102` | OBS is a proposal variant with a `predicted` instance-mask input. |
| HGGD | The test script filters collisions (and grasps with ≤ 2 points between the fingers) before dumping; the commented-out line affects only a logging variable. | `test_graspnet.py:255-270`, `dataset/evaluation.py:343-349`, `dataset/collision_detector.py:84-92` | Static reading; to be confirmed by running. Repository numbers are therefore to be treated as with-collision-filter until shown otherwise. |
| HGGD | "Joint training" sums two losses into one optimizer step; image features are detached, so no gradient crosses from the local network to the image network. | `models/anchornet.py:216`, `dataset/evaluation.py:28-38` | Gradient boundary is part of the recipe; removing it is a named variant. |
| HGGD | Shipped shell script and argparse defaults disagree (loss weights, learning rate, batch size, `local_k`). | `train_graspnet.sh`, `train_utils.py` | Reference config follows the shell script; stated in the method README. |
| HGGD | The test loop needs label files for test scenes and writes no dump for a frame without 2-D proposals. | `dataset/base_grasp_dataset.py:287`, `test_graspnet.py:197-199` | `predict()` needs no labels; an empty result is an explicit empty `GraspGroup`. |

</details>
