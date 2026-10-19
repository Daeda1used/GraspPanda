# Get started

From install to your first swap in three steps. Nothing needs a dataset until you probe or
train on real frames.

<div class="grid cards" markdown>

-   :material-download-outline: **1 · Install**

    ---

    Install from source, then check it with a toy detector that trains in under a minute.

    [Install →](installation.md)

-   :material-play-circle-outline: **2 · Run a method**

    ---

    Pick a method and get grasps from its released weights, on the example scene or your camera.

    [Run a method →](run_a_method.md)

-   :material-swap-horizontal: **3 · Swap a component**

    ---

    Replace an encoder or another stage with one line, check it, try it, train it.

    [Swap a component →](swap.md)

-   :material-file-cog-outline: **Optional · Tour the config**

    ---

    See how a config is checked, and how a mistake is caught before anything is built.

    [Tour the config →](quickstart.md)

</div>

## Words used on these pages

| Word | What it means |
|---|---|
| **slot** | a stage of a detector you can replace: `--set <slot>=…` on the command line, `model.<slot>` in a config |
| **option** | what goes into a slot: a preset or a component name |
| **seed-based skeleton** | the pipeline five methods share: scene encoder → seeds → view → grouper → head; HGGD is its own family |
| **scene encoder** | the point-cloud encoder of the seed-based methods; its slot is `backbone` |
| **image encoder** | the image encoder of HGGD; its slot is `image_encoder` |
| `composes` · `runs` · `learns` · `predicts` | the levels a swap is measured at: config accepted · trains on real frames without error · its losses fall · it ranks grasps |
| **probe** | a short training on real frames that ends with one verdict, such as `runs` or `refused`; `wired` means a synthetic scene only |
| **trains like shipped** | a swap trained longer beside the component the method ships: `on a par`, `worse`, `diverging`, `did not train` or `not comparable` |

The exact definitions and thresholds: [How we measure](../evidence/index.md).
