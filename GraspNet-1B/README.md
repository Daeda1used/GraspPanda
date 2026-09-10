# GraspNet-1B

Method presets and component compositions for GraspNet-1B. Start with the [first-frame walkthrough](../docs/DOWNLOADS.md#run-your-first-graspnet-frame), or the [author sample](../docs/DOWNLOADS.md#start-without-graspnet) if you have not downloaded the dataset.

## Configuration examples

Use `--method` for an original method preset, or choose a composition, training or sweep from the [template catalogue](examples.yaml). Generate only the files you need:

```bash
./panda init --list
./panda init --example compose-baseline -o composition.local.yaml
./panda init --example train-hggd -o training.local.yaml
```

Run commands from the repository root after [installation](../docs/INSTALL.md). Each template contains the relevant settings; omitted fields use shared defaults. Generation does not access data, weights or the GPU and never overwrites an existing file. Set your local paths, prepare the method's labels for training, and use `./panda run FILE` (or `./panda sweep FILE` for a sweep). The [module guide](../docs/MODULES.md) explains compatible replacements and parameters; [operating instructions](../docs/USAGE.md) cover training, resume and evaluation. Generated `*.local.yaml` files stay local.

## Observation protocols

Single-view point clouds, RGB-D, fused views, active perception and temporal sequences have different input contracts. The selected preset preserves its method's protocol. Find operation availability, limitations, PDFs and original implementations in [Methods & papers](../docs/METHODS.md).

Original implementations are fetched into `upstream/` by protocol. Runtime caches, weights and experiment outputs are generated locally; see the [local directory layout](../docs/INSTALL.md#files-created-locally).
