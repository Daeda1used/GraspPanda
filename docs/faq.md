# FAQ

## Is GraspPanda tied to the Franka Panda robot?

No. The panda is the animal. Grasps are expressed for a generic parallel-jaw gripper in
the camera frame; nothing depends on a particular arm.

## Why is the model zoo empty?

A method enters only after it has been trained and evaluated in this toolbox. Listing
methods that merely run would repeat the mistake this project was restarted to fix.

## Why YAML and not Python configs?

Strict validation is far easier to guarantee for data than for code: every key can be
checked against a schema or a constructor signature before anything runs. Logic belongs
in components, which you add through `custom_imports` without forking.

## Why did my override drop some keys?

Changing a node's `type` replaces the node, so arguments of the old component cannot
leak into the new one. The dropped keys are printed. See
[Configuration](user_guides/config.md#merge-rules).

## Why does a component not accept `**kwargs`?

Every config key must be checkable against a signature. A component that swallows
arbitrary keys would turn typos into silent no-ops.

## Do I need a GPU?

Not for the framework core: all operators have pure-PyTorch reference implementations
and the test-suite runs on CPU. Training a grasp detector does need one.

## Why is `num_workers` refused below my `TMPDIR`?

A data-loader worker hands every batch to the training process through a Unix socket
that it creates in the temporary directory, and the path of a socket may be at most 107
bytes long (103 outside Linux). Below a deeper `TMPDIR` the worker cannot create it: it
prints `OSError: AF_UNIX path too long`, drops the batch, and the loader waits for that
batch forever. So the loaders of `grasppanda train` and `test` stop at once with the
reason instead. Point `TMPDIR` to a shorter directory (`TMPDIR=/tmp grasppanda train
...`) or set `data.train.num_workers=0`. A Python release that falls back to a short
directory by itself is recognised and not refused.

## How do I report a bug?

Open an issue and include the output of `grasppanda env` and, for config problems, of
`grasppanda print-config <config>`.
