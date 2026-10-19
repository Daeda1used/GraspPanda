"""GraspNet-baseline: GraspNet-1Billion, A Large-Scale Benchmark for General Object
Grasping (CVPR 2020).

The method as a configuration of registered stages:

=============  ==========================  ==============================================
slot           reference component         what it does
=============  ==========================  ==============================================
``backbone``   ``PointNet2``               1024 seeds with 256 channels (sub-sampled)
``proposal``   ``SubsampledSeeds``         the backbone's subsample is the seed set
``view``       ``ObjectnessViewStage``     objectness + 300 view scores, arg-max view
``assigner``   ``NearestLabelAssigner``    labels, crop frame and crop centre from labels
``grouper``    ``DepthCylinderGroup``      one scene-cloud cylinder per depth bin
``head``       ``OperationHead``           score, angle class, width, tolerance
=============  ==========================  ==============================================

``configs/_base_/models/graspnet_baseline.yaml`` is the full reference model. The
implementation is written from the paper and from the behaviour of
``graspnet/graspnet-baseline`` (commit ``fb2f089``); with the authors' checkpoint
converted by ``tools/convert_checkpoint.py`` it is compared stage by stage against
activations recorded from that implementation
(``tests/integration/test_baseline_parity.py``).

What sets it apart from the methods with dense features (GSNet, EconomicGrasp, FineGrasp)
is visible in the contracts, slot by slot: the backbone returns ``subsampled`` features,
so a proposal that scores every scene point is refused; the grouper has
``support = "scene"`` and emits ``(seed, depth, channel)``, so a head written for
``(seed, channel)`` is refused.

Behaviours of the reference that change numbers are named options whose defaults
reproduce it: the training crop is centred at the matched label point, not at the seed
(``assigner.crop_centre``); only seeds on an object are supervised
(``assigner.supervise``); the score target is scaled by the worst friction level of
the batch (``score_target`` of the view stage and the head); widths are widened in the
decoder only (``head.width_decode_gain``); the decoded score is multiplied by the
predicted tolerance (``head.score_decode``); background seeds yield no grasp
(``view.reject_background``).

Known, deliberate differences from the reference:

* One model instance trains and predicts; the mode follows ``model.train()`` /
  ``model.eval()``.
* Losses are means over the supervised entries; the reference divides the sum by the
  count plus 1e-6. Losses over an empty selection are zero rather than NaN.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal

from grasppanda.models.base import Assigner, Backbone, Grouper, HeadBundle, Proposal, ViewStage
from grasppanda.models.heads.friction import SCORE_TARGETS
from grasppanda.registry import DETECTORS, Cfg
from grasppanda.structures import ContractError, Gripper

from .seed_based import SeedGraspDetector

__all__ = ["GraspNetBaseline"]

#: Options that the view stage and the head bundle state separately and must agree on:
#: both derive their targets from one score definition.
_SHARED_TARGET_OPTIONS = ("score_target", "max_friction")


def _target_problems(view: Any, head: Any, get: Callable[[Any, str], Any]) -> Iterator[str]:
    """Compare the score encoding of the view stage and the head bundle.

    Only when both use the score definition of :mod:`grasppanda.models.heads.friction`:
    a stage with a target of its own kind (a view graspness, a normalised score grid)
    shares no definition with the other, even where an option happens to have the same
    name.
    """
    if not all(get(stage, "score_target") in SCORE_TARGETS for stage in (view, head)):
        return
    for option in _SHARED_TARGET_OPTIONS:
        left, right = get(view, option), get(head, option)
        if left is not None and right is not None and left != right:
            yield (
                f"the view stage has {option} = {left!r} and the head bundle has {option} = "
                f"{right!r}. The view target is the best grasp score of a view, so both must "
                "encode scores the same way."
            )


@DETECTORS.register()
class GraspNetBaseline(SeedGraspDetector):
    """The GraspNet-1Billion baseline; see the module docstring and ``graspnet_baseline.yaml``.

    The arguments are those of :class:`SeedGraspDetector`. On top of its contract checks,
    two things that a pair of stages states separately are required to agree: the score
    encoding of the view stage and the head bundle, and the number of depth bins of the
    grouper and the head bundle.
    """

    status = "tested"
    source = "GraspNet-baseline (Fang et al., CVPR 2020)"

    def __init__(
        self,
        backbone: Cfg[Backbone],
        proposal: Cfg[Proposal],
        view: Cfg[ViewStage],
        grouper: Cfg[Grouper],
        head: Cfg[HeadBundle],
        assigner: Cfg[Assigner],
        gripper: Gripper | None = None,
        point_features: Literal["ones", "observation"] = "ones",
    ) -> None:
        super().__init__(backbone, proposal, view, grouper, head, assigner, gripper, point_features)
        problems = list(_target_problems(self.view, self.head, lambda o, n: getattr(o, n, None)))
        crops = getattr(self.grouper, "num_depth", None)
        codec = getattr(self.head, "codec", None)
        if crops is not None and codec is not None and crops != codec.num_depth:
            problems.append(
                f"{type(self.grouper).__name__} makes {crops} crop(s) per seed, but "
                f"{type(self.head).__name__} is configured for num_depth = {codec.num_depth}. "
                "The grouper needs one far end (hmax) per depth bin of the head."
            )
        if problems:
            raise ContractError(f"{type(self).__name__}: " + " ".join(problems))

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """The skeleton's contract checks, plus what the config states twice."""
        yield from super().check_config(node)
        view, head, grouper = node.get("view"), node.get("head"), node.get("grouper")
        if isinstance(view, Mapping) and isinstance(head, Mapping):
            yield from _target_problems(view, head, lambda o, n: o.get(n))
        if isinstance(grouper, Mapping) and isinstance(head, Mapping):
            reach, bins = grouper.get("hmax"), head.get("num_depth")
            if isinstance(reach, (list, tuple)) and isinstance(bins, int) and len(reach) != bins:
                yield (
                    f"the grouper lists {len(reach)} far end(s) (hmax), but the head bundle has "
                    f"num_depth = {bins}. The grouper needs one crop per depth bin of the head."
                )
