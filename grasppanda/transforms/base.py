"""Base classes of data transforms and the fingerprint of a preprocessing chain.

A data pipeline has two halves, separated by the *label-alignment boundary*:

* :class:`ImageTransform` runs before it, on a :class:`~grasppanda.transforms.frame.Frame`
  (depth image, organised cloud, validity mask). These steps decide *which pixels
  become points*. Every derived per-point label is aligned to exactly that set, so their
  configuration is hashed into a fingerprint that derived-label assets must match.
* :class:`PointTransform` runs after it, on a
  :class:`~grasppanda.transforms.frame.PointSample`. Sampling and augmentation live
  here; they may reorder, drop or move points only together with the labels.

Keeping the two halves in different types makes it impossible to configure a sampling
step where it would silently change label alignment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from grasppanda.registry import TRANSFORMS, component_parameters

if TYPE_CHECKING:
    from .frame import Frame, PointSample

__all__ = [
    "ImageTransform",
    "PointTransform",
    "Transform",
    "canonical_json",
    "config_fingerprint",
    "fingerprint",
]


def _plain(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, where) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item, where) for key, item in sorted(value.items())}
    raise TypeError(
        f"{where} is a {type(value).__name__}; transform arguments must be numbers, strings, "
        "booleans, lists or mappings so that they can be fingerprinted"
    )


def canonical_json(value: Any) -> str:
    """Serialise plain data so that equal content always gives equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@TRANSFORMS.base
class Transform:
    """Maps one sample to one sample.

    A subclass stores every constructor argument under an attribute of the same name,
    normalised to a canonical type (``float(margin)``), so that :meth:`config` can
    describe the instance exactly.
    """

    #: Bump when behaviour changes for unchanged arguments; part of every fingerprint.
    version: ClassVar[int] = 1

    def __call__(self, sample: Any) -> Any:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """``{type, version, **arguments}``: what this instance does, as plain data."""
        name = type(self).__name__
        node: dict[str, Any] = {"type": name, "version": self.version}
        parameters, _ = component_parameters(type(self))
        for parameter in parameters:
            if not hasattr(self, parameter):
                raise TypeError(
                    f"{name} must keep its constructor argument '{parameter}' as "
                    f"self.{parameter} so that its configuration can be fingerprinted"
                )
            node[parameter] = _plain(getattr(self, parameter), f"{name}.{parameter}")
        return node

    def __repr__(self) -> str:
        arguments = {k: v for k, v in self.config().items() if k not in ("type", "version")}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"


class ImageTransform(Transform):
    """A step before the label-alignment boundary. Fingerprinted."""

    #: True when the step reads ground truth (e.g. the segmentation mask). The dataset
    #: only hands ground truth to a pipeline that declares it, and assets built on such
    #: a pipeline are flagged.
    uses_ground_truth: ClassVar[bool] = False

    def __call__(self, frame: Frame) -> Frame:
        raise NotImplementedError


class PointTransform(Transform):
    """A step after the label-alignment boundary. Carries labels along with the points."""

    def __call__(self, sample: PointSample) -> PointSample:
        raise NotImplementedError


def config_fingerprint(configs: Sequence[Mapping[str, Any]]) -> str:
    """Hash of a list of transform configs, as returned by :meth:`Transform.config`."""
    text = canonical_json([dict(config) for config in configs])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fingerprint(transforms: Sequence[ImageTransform]) -> str:
    """Hash of a preprocessing chain: the identity of the point set labels align to."""
    for transform in transforms:
        if not isinstance(transform, ImageTransform):
            raise TypeError(
                f"{type(transform).__name__} is not an ImageTransform; only steps before the "
                "label-alignment boundary can be fingerprinted"
            )
    return config_fingerprint([transform.config() for transform in transforms])
