# FAQ

## Is GraspPanda only for GraspNet-1Billion or GSNet?

No. GraspPanda starts from GraspNet-1Billion, the benchmark most methods report on, and also
reads GraspClutter6D with an evaluator of its own. It ports methods of two families:
GraspNet-baseline, GSNet, Scale-Balanced-Grasp, EconomicGrasp and FineGrasp on one seed-based
skeleton, and HGGD on its own. GSNet is one of them. Every option shows under which methods
it was measured to work: see [Methods](methods/index.md) and [Evidence](evidence/index.md).

![Two detector families fed by GraspNet-1Billion and GraspClutter6D: the slots of the seed-based skeleton, HGGD's image encoder and fixed stages, and the post-processing and training recipe all methods share](assets/pipeline.png#only-light)
![Two detector families fed by GraspNet-1Billion and GraspClutter6D: the slots of the seed-based skeleton, HGGD's image encoder and fixed stages, and the post-processing and training recipe all methods share](assets/pipeline-dark.png#only-dark)

## Which method should I run first?

The chooser on [Run a method](get_started/run_a_method.md) shows what each method reads, what
it needs installed and where its weights are. GraspNet-baseline, Scale-Balanced-Grasp and HGGD
need no extra; GSNet, EconomicGrasp and FineGrasp need spconv ([Install](get_started/installation.md)).

## Does a probe tell me accuracy?

No.

--8<-- "docs/includes/caveat_probe.md"

A probe trains a combination for a few iterations and says whether it runs, learns and
predicts. Accuracy needs a training on the whole split and `grasppanda test`; the
[model zoo](model_zoo.md) lists what was trained and evaluated here.

## Can I download trained weights?

Not from this project yet. Every method runs from weights released with its reference code:
you download them yourself, under their licence, and convert them once
([Run a method](get_started/run_a_method.md)). GSNet was trained here, but its weights are not
published yet. A method enters the model zoo only after it has been trained and evaluated here.

## Is GraspPanda tied to the Franka Panda robot?

No. The panda is the animal. Grasps are expressed for a generic parallel-jaw gripper in the
camera frame; nothing depends on a particular arm.

## Why YAML and not Python configs?

Strict validation is far easier to guarantee for data than for code: every key can be
checked against a schema or a constructor signature before anything runs. Logic belongs in
components, which you add through `custom_imports` without forking.

## Why did my override drop some keys?

Changing a node's `type` replaces the node, so arguments of the old component cannot leak
into the new one. The dropped keys are printed. See
[Configuration](user_guides/config.md#merge-rules).

## Why does a component not accept `**kwargs`?

Every config key must be checkable against a signature. A component that swallows arbitrary
keys would turn typos into silent no-ops.

## Do I need a GPU?

Not for the framework core: every operator has a pure-PyTorch reference implementation and
the test suite runs on CPU. Training a grasp detector does need one.

## Does a number depend on the GPU?

Not since a run states its precision. On recent GPUs PyTorch lets convolutions round to
TF32 unless told otherwise, and with it HGGD's released checkpoint scored up to 0.9 AP
apart on an RTX A6000, an RTX 3090 and that RTX 3090 with a newer PyTorch (120 frames of the
seen test split). A run therefore computes in float32 unless its config says
`runtime.tf32: true`, and HGGD then agrees within 0.02 AP on the two stacks tried. The
checkpoints of the five seed-based methods scored the same within 0.01 AP wherever the
setting was the same ([Reproducibility](user_guides/train_test.md#reference)).

## Why is `num_workers` refused below my `TMPDIR`?

A data-loader worker hands each batch to the training process through a Unix socket in the
temporary directory, and a socket path may be at most 107 bytes long (103 outside Linux).
Below a deeper `TMPDIR` the worker cannot create it, drops the batch, and the loader waits
forever. So `grasppanda train` and `test` stop at once with the reason. Point `TMPDIR` to a
shorter directory (`TMPDIR=/tmp grasppanda train ...`) or set `data.train.num_workers=0`. A
Python release that falls back to a short directory by itself is recognised and not refused.

## How do I report a bug?

Open an issue with the output of `grasppanda env` and, for a config problem, of
`grasppanda print-config <config>`.
