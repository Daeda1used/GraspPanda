# Add a component

**Goal:** add an encoder, a stage, a head bundle, a post-processor, a transform, a hook or
an evaluator without forking: write a class, register it, and point a config at it.

**You need:** GraspPanda installed and one Python file. The
[example project](https://github.com/Daeda1used/GraspPanda/tree/main/projects/example_project)
does exactly this and is the best starting point.

## 1. Write and register it

```python
# my_components.py
from grasppanda.models import Backbone
from grasppanda.registry import BACKBONES

@BACKBONES.register()
class MyBackbone(Backbone):
    """One sentence that says what it is: the Registered classes page shows it."""

    output = "dense"                      # the contract: features live on the input points

    def __init__(self, in_channels: int = 3, out_channels: int = 256, depth: int = 4) -> None:
        ...                               # the whole class: see the fold below
```

Subclass the base class of the slot (`Backbone`, `Grouper`, `HeadBundle`, …): it decides
the registry and the config slot. Annotate every constructor parameter, since the
annotations are what config values are validated against.

??? note "The whole class, with what the Registered classes page prints"

    ```python
    # my_components.py
    from torch import nn

    from grasppanda.models import Backbone
    from grasppanda.registry import BACKBONES


    @BACKBONES.register()
    class MyBackbone(Backbone):
        """One sentence that says what it is: the Registered classes page shows it."""

        output = "dense"                      # the contract: features live on the input points
        status = "tested"                     # what you proved, not what you hope
        source = "My method (Me et al., 2026)"  # the paper, method or library it implements
        extras = ("spconv",)                  # optional dependencies, if any

        def __init__(self, in_channels: int = 3, out_channels: int = 256, depth: int = 4) -> None:
            super().__init__()
            self.out_channels = out_channels
            ...
    ```

    `status`, `source`, `extras` and the first sentence of the docstring are what
    `grasppanda components` and the [Registered classes](../components.md) page print.
    Components inside the `grasppanda` package must declare `source` on the class itself (a
    subclass does not inherit a citation), add one tiny configuration to
    `tests/unit/test_swap_matrix.py` if they are stages of the seed-based skeleton, and
    regenerate the pages with `python -m tools.make_component_docs`; the unit suite fails
    until all three are done.

## 2. Point a config at it

```yaml
custom_imports: [../my_components.py]     # or an importable module name

model:
  backbone: {type: MyBackbone, depth: 6}
```

Put these lines in a config that inherits the method you build on
([Configuration](../user_guides/config.md#inheritance)). Which slots each method has is on
the [Components](../slots/index.md) page.

## 3. Check it, then probe it

```bash
grasppanda print-config my_config.yaml --check
grasppanda probe my_config.yaml
```

**Expected result:** `✓ my_config.yaml is valid` with a hash, then a probe verdict. A probe
trains the config for 20 iterations on real frames and predicts
([Compose and probe](../user_guides/combinations.md)).

??? note "Troubleshooting: what registration and the check refuse"

    - **`**kwargs`.** Registration refuses it: a component that swallows arbitrary keys would
      turn typos into silent no-ops.
    - **Annotations under `TYPE_CHECKING`.** Import annotation names at runtime; the
      validator reads them.
    - **A key the class does not have.** `MyBackbone has no parameter 'depths'. Did you mean
      'depth'? Parameters: …`, before anything is built.
    - **A contract a neighbour cannot meet.** The message names both sides; see *Contracts*.

??? note "Slots that hold other components"

    Annotate the parameter with `Cfg[Base]` and build it yourself, so you can hand the
    child what you already know:

    ```python
    from grasppanda.models import Backbone, BaseGraspDetector, HeadBundle
    from grasppanda.registry import BACKBONES, DETECTORS, HEADS, Cfg


    @DETECTORS.register()
    class MyDetector(BaseGraspDetector):
        def __init__(self, backbone: Cfg[Backbone], head: Cfg[HeadBundle]) -> None:
            super().__init__()
            self.backbone = BACKBONES.build(backbone)
            self.head = HEADS.build(head, in_channels=self.backbone.out_channels)
    ```

    `Cfg[Backbone]` tells the validator which registry owns the slot, so nested configs are
    checked without constructing anything.

??? note "Injected arguments"

    A child should not make the user repeat what the parent knows. Declare such a parameter
    with the `INJECTED` default:

    ```python
    from grasppanda.registry import HEADS, INJECTED


    @HEADS.register()
    class MyHead(HeadBundle):
        in_axes = ("seed", "channel")

        def __init__(self, num_angle: int = 12, in_channels: int = INJECTED) -> None: ...
    ```

    - The config may omit `in_channels`; the parent supplies it through `HEADS.build(...)`.
    - If the config sets it *and* disagrees with the parent, building fails. Nothing is
      silently overridden.

??? note "Contracts"

    Declare what your component consumes and produces as class attributes; the base classes
    document which ones exist, and the [Registered classes](../components.md) page explains
    each next to the components that declare it:

    | Slot | Contract attributes |
    |---|---|
    | backbone (scene encoder) | `output`, `output_channels(options)` |
    | proposal | `requires`, `seed_scores`, `needs_supervision` |
    | view stage | `seed_scores`, `label_formats`, `needs_matched`, `applies_seed_weights` |
    | assigner | `label_format`, `optional_outputs`, `needs_supervision` |
    | grouper | `support`, `out_axes`, `required_seed_channels(options)` |
    | head bundle | `in_axes`, `label_format`, `score_kind`, `applies_seed_weights`, `needs_supervision`, `required_seed_scores(options)` |
    | image encoder | `in_channels`, `input_layout`, `out_channels`, `out_strides` (`grasppanda.models.image_encoders.base.ImageEncoder`) |

    The first six are the stages of the seed-based skeleton. It compares them for every
    detector built on it, on the config and again on the built stages
    (`stage_contract_problems` in `grasppanda.models.detectors.seed_based`). A stage
    therefore needs no checking code of its own: declare the attributes truthfully and an
    illegal combination is refused with a message that names both sides. Say only what the
    code does: a head that sets `applies_seed_weights = True` without reading
    `MatchedLabels.weight` turns a config error into a silently wrong model. No config check
    compares the widths of an image encoder: the decoder that follows is built for whatever
    widths arrive.

    Two of them reach beyond the pair of stages next to each other:

    - `output_channels(options)` is the width a backbone returns, told from its options
      alone; return the same number the built module has as `out_channels`
      (`tests/unit/test_swap_matrix.py` compares them for your tiny config and for your
      defaults). The default reads an integer `out_channels` option, so an encoder that has
      one needs nothing more. Return `None` only where the width is known once the encoder
      is built (a bridge whose width is that of a library model): the config check then
      builds the encoder once, without released weights, and the test asks for the reason
      (`WIDTH_KNOWN_ONCE_BUILT`). A grouper that fixes the width of the seed features states
      it with `required_seed_channels(options)` and names in `projected_variant` the grouper
      that takes any width, which `grasppanda compose` puts in its place.
    - `needs_supervision` lists what a stage reads from the training `Supervision`, by where
      it sits (`objects`, `objects.tolerance`, `point_labels.<name>`, `extras.<name>`), each
      with the option that makes the stage read it, or `None` when it always does. A dataset
      reader says what a config makes it serve with `served_supervision(node)`, and the
      config check refuses a training dataset that does not serve what the model reads.
      `tests/unit/test_supervision_contract.py` trains every legal stage set on a batch that
      carries exactly the declared fields and without each of them in turn, so a stage that
      reads more than it declares fails there. A detector that is not built on the skeleton
      declares `needs_supervision` on itself.

    A component that owns slots of another kind can compare its children's contracts before
    anything is built by implementing `check_config`:

    ```python
    @classmethod
    def check_config(cls, node):
        grouper = GROUPERS.get(node["grouper"]["type"])
        head = HEADS.get(node["head"]["type"])
        if grouper.out_axes != head.in_axes:
            yield f"{grouper.__name__} emits {grouper.out_axes}, but {head.__name__} expects {head.in_axes}."
    ```

    Each yielded message becomes a config error at that node.

??? note "Head bundles"

    A head bundle is one class that owns the prediction module, the target encoder, the
    loss and the decoder. They share one grasp parameterization (angle and depth grids,
    width scale, score meaning), so they are swapped together or not at all. There is no
    way to configure a head with another bundle's decoder.

??? note "Optional dependencies"

    Import heavy dependencies inside the component, through `require`, so that
    `import grasppanda` keeps working with the core install:

    ```python
    from grasppanda.utils.imports import require

    class SparseThing(Backbone):
        def __init__(self, ...) -> None:
            spconv = require("spconv.pytorch", extra="spconv-cu118", purpose="SparseThing")
    ```

    A user without the extra gets `SparseThing needs 'spconv.pytorch' … pip install -e ".[spconv-cu118]"`.

## Share it

Put it under `projects/<your_project>/` with a README that states a support level; see
[`projects/README.md`](https://github.com/Daeda1used/GraspPanda/blob/main/projects/README.md).
A pip-installable plugin can expose a `grasppanda.plugins` entry point to be imported
automatically.

## Next

- [Compose and probe](../user_guides/combinations.md): measure your option under the
  methods that have its slot.
- [Data contracts](data_flow.md): the structures stages exchange.
