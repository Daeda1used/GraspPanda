# Configs

```
configs/
├── _base_/                 building blocks: datasets/, models/, schedules/, default_runtime.yaml
├── <method>/               one folder per ported method
│   ├── README.md           what is ported, how it was checked, its numbers
│   ├── metafile.yaml       its weights, and the reported and measured numbers
│   ├── *.yaml              runnable configs
│   └── records/            the result records behind the measured numbers of the metafile
└── experimental/           ready-made swaps of a method's scene encoder (no accuracy claimed)
```

Runnable configs are named `<method>_<backbone>_<dataset>-<camera>_<schedule>.yaml` and are
addressed by name, whatever the family:
`grasppanda print-config graspnet_baseline/graspnet-baseline_pointnet2_g1b-realsense_18e` or
`grasppanda print-config hggd/hggd_r34w8_g1b-realsense_15e`.

Rules of the config language are in the
[configuration guide](../docs/user_guides/config.md). In short: at most two levels of
`_base_`, a changed `type` replaces the inherited node, and unknown keys are errors.

## What a method folder means

A method gets a folder when its port equals the reference implementation stage by stage,
checked with weights released with that implementation. The folder does not mean the method passed the model-zoo
gate. Admission needs a training here and a whole-split evaluation within 1 AP of the reported
number, or with the gap explained; no method has passed yet, so the zoo has no weights to
download.

A number of a metafile that was measured here (`author-ckpt`, `reproduced`) names its result
record under `records/`. `python -m tools.record_result` checks a whole-split run, puts its
records there and prints the rows; see
[Recording a result](../docs/model_zoo.md#recording-a-result).
