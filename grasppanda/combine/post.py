"""Post-processing chains from short text: ``collision+nms+top-k=50``.

A chain is what ``evaluation.variants.<name>`` holds: a list of post-processor nodes,
applied in order to the predictions of a frame. This module turns the text a person
types after ``--post`` (``grasppanda test``, ``infer``, ``demo``) or after
``--set post=`` (``grasppanda compose``) into that list.

One step is

``name``
    a preset of ``grasppanda/presets/post/`` (``top-k``) or a registered post-processor
    with its constructor defaults (``TopK``);
``name=value``
    the same with its main option set: the option the component names as ``primary``
    (``top-k=50``, ``score-threshold=0.3``, ``gripper-limits=0.085``);
``name:option=value,option=value``
    any options, values in YAML (``workspace:x=[-0.3,0.3],z=[0.2,0.9]``).

Steps are joined with ``+`` and run left to right.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
from collections.abc import Sequence
from typing import Any

from grasppanda.config import parse_yaml
from grasppanda.registry import POSTPROCESSORS, component_parameters

from .presets import POST_SLOT, presets

__all__ = [
    "POST_SLOT",
    "POST_VARIANT",
    "PostSpecError",
    "chain_nodes",
    "chain_override",
    "step_node",
]

#: The evaluation variant a chain given on a command line is written to.
POST_VARIANT = "post"

_STEP = re.compile(r"\+(?=[A-Za-z])")


class PostSpecError(ValueError):
    """A post-processing step that cannot be understood, with what would be."""


def _split_options(text: str) -> list[str]:
    """Split ``a=1,b=[2,3]`` at the commas outside brackets."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [part for part in (*parts, current) if part.strip()]


def _base_node(name: str) -> dict[str, Any]:
    members = presets().get(POST_SLOT, {})
    if name in members:
        return copy.deepcopy(members[name].node)
    if name in POSTPROCESSORS:
        return {"type": name}
    known = [*members, *POSTPROCESSORS.names()]
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    hint = f" Did you mean {' or '.join(repr(item) for item in close)}?" if close else ""
    raise PostSpecError(
        f"no post-processing step '{name}'.{hint} Presets: {', '.join(members) or 'none'}. "
        f"Components: {', '.join(POSTPROCESSORS.names())}."
    )


def step_node(spec: str) -> dict[str, Any]:
    """The component node of one step (see the module docstring for the syntax)."""
    spec = spec.strip()
    match = re.match(r"^([A-Za-z][\w-]*)\s*([:=])?(.*)$", spec, flags=re.S)
    if match is None:
        raise PostSpecError(
            f"cannot read the post-processing step {spec!r}; write `name`, `name=value` or "
            "`name:option=value,option=value`"
        )
    name, mark, rest = match.groups()
    node = _base_node(name)
    cls = POSTPROCESSORS.get(node["type"])
    params, _ = component_parameters(cls)
    if mark is None:
        if rest.strip():
            raise PostSpecError(f"cannot read the post-processing step {spec!r}")
        return node
    if mark == "=":
        primary = getattr(cls, "primary", None)
        if primary is None:
            raise PostSpecError(
                f"{node['type']} names no main option, so `{name}=<value>` has nothing to set. "
                f"Write `{name}:<option>=<value>`; its options: {', '.join(params)}."
            )
        items = [f"{primary}={rest}"]
    else:
        items = _split_options(rest)
    for item in items:
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or key not in params:
            raise PostSpecError(
                f"`{item.strip()}` in {spec!r}: {node['type']} has no option '{key}'. "
                f"Its options: {', '.join(params)}."
            )
        try:
            node[key] = parse_yaml(value)
        except Exception:
            node[key] = value.strip()
    return node


def chain_nodes(specs: Sequence[str] | str) -> list[dict[str, Any]]:
    """The nodes of a chain given as one text or several; ``+`` joins steps."""
    if isinstance(specs, str):
        specs = [specs]
    steps = [step for spec in specs for step in _STEP.split(spec) if step.strip()]
    if not steps:
        raise PostSpecError("an empty post-processing chain; name at least one step")
    return [step_node(step) for step in steps]


def chain_override(specs: Sequence[str] | str, variant: str = POST_VARIANT) -> str:
    """The config override that defines ``evaluation.variants.<variant>`` as this chain."""
    return f"evaluation.variants.{variant}={json.dumps(chain_nodes(specs))}"
