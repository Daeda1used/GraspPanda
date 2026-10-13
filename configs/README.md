# Configs

```
configs/
├── _base_/                 building blocks: datasets/, models/, schedules/, default_runtime.yaml
└── <method>/               README.md, metafile.yaml and runnable configs of one method
```

Runnable configs are named `<method>_<backbone>_<dataset>-<camera>_<schedule>.yaml`
and are addressed by name: `grasppanda print-config gsnet/gsnet_sunet14_g1b-realsense_10e`.

Rules of the config language are in the
[configuration guide](../docs/user_guides/config.md). In short: at most two levels of
`_base_`, a changed `type` replaces the inherited node, and unknown keys are errors.

Method folders are added as each method passes the model-zoo gate; see the roadmap in
the top-level README.
