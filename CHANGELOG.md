# Changelog

All notable changes to GraspPanda are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

GraspPanda was rebuilt from scratch in October 2026. The releases up to v0.56 are an
earlier code base with a different design; they stay available under their tags and on the
`legacy` branch, and none of their code is part of this history.

### Initial public version of the rebuild

- **Core.** YAML configs with inheritance and three layers of validation; registries with
  lazy imports; typed data contracts between pipeline stages; run snapshots. Commands:
  `train`, `test`, `infer`, `demo`, `analyze`, `components`, `sweep`, `prepare`,
  `print-config`, `env`.
- **Benchmarks.** Readers, derived label assets and evaluators for GraspNet-1Billion and
  GraspClutter6D. Both evaluators agree with the official tools on recorded frames (1,592
  and 254); the GraspNet-1Billion one runs about 40 times faster. One command scores a
  GraspNet-1Billion model on GraspClutter6D.
- **Methods as interchangeable components**, each equal to its reference implementation
  stage by stage with converted weights: GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline
  and Scale-Balanced-Grasp on one seed-based skeleton, and HGGD as a family of its own.
  `tools/convert_checkpoint.py` converts the authors' checkpoints strictly.
- **Scene encoders.** Sparse U-Net, SpUNet, OA-CNNs, PointNet++ (two layouts), PointNeXt,
  Point Transformer V3 (also with Sonata, Concerto and Utonia weights), LitePT, the
  transformer encoder of Scale-Balanced-Grasp, and adapters between dense and sub-sampled
  encoders. Released weights load strictly and reproduce the official outputs.
- **Interchange.** `grasppanda components` lists every component with its contract, status
  and source. A swap matrix test proves that every combination of registered stages either
  trains and predicts or is refused at config time with the conflict and the fix.
  `configs/experimental/` holds ready-made encoder swaps.
- **Combinations.** `grasppanda compose <method config> --set <slot>=<preset or component>`
  writes a checked config, inserting the one encoder adapter the stage contracts ask for;
  a refused combination prints the conflict and the fix. Presets
  (`grasppanda/presets/<slot>/`) hold the settings of every encoder and stage variant for
  table-top depth clouds. `grasppanda probe <config>` runs the real trainer for a few
  iterations on the `sample` split and predicts, ending with one verdict. The generated
  page `docs/advanced_guides/measured_combinations.md` records 208 combinations probed on
  real frames (145 run, 6 run unstably, 51 are refused by the contract rules, 6 fail) and
  the measured class of every component (universal, most, some, specific). No AP.
- **Inference and analysis.** `GraspInferencer` and `grasppanda demo` (depth image, point
  cloud, dataset frame or the built-in synthetic scene); `grasppanda analyze` splits an AP
  by scene, object and cause.
- **Speed.** Optional compiled kernels for furthest point sampling, ball query, cylinder
  query and nearest neighbours with bit-identical results; dense label grids are read where
  they are matched instead of being copied into every batch.

### Known limits

- The model zoo is empty: no method has been trained and evaluated here to the admission
  rule yet. One author checkpoint has been scored on a whole test split (EconomicGrasp,
  Kinect); its result record is in `configs/economicgrasp/records/`.
- An encoder without an `out_channels` option (LitePT, PTv3 and its pretrained variants,
  PointNeXt, PointNet2Dense) passes the config check under Scale-Balanced-Grasp and fails
  when built, because no contract states the width an encoder returns. `grasppanda probe`
  reports it with the fix.
- The CPU-only wheel of spconv 2.3.8 miscomputes with more than one OpenMP thread; sparse
  layers detect this and stop with an explanation (see the installation guide).
