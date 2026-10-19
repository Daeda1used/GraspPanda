"""EconomicGrasp: An Economic Framework for 6-DoF Grasp Detection (ECCV 2024).

The method as a configuration of registered stages; the first three are those of GSNet,
which EconomicGrasp builds on:

=============  ============================  ============================================
slot           reference component           what it does
=============  ============================  ============================================
``backbone``   ``SparseUNet14``              per-point features from a sparse U-Net
``proposal``   ``GraspnessFPS``              objectness + graspness → 1024 seeds by FPS
``view``       ``TemplateViewStage``         300 view scores, one approach per seed
``assigner``   ``CompactLabelAssigner``      one labelled grasp per seed at its view
``grouper``    ``CylinderInteractionGroup``  cylinder crop whose neighbours attend
``head``       ``EconomicHead``              angle, depth and quality classes, width
=============  ============================  ============================================

What makes it "economic" is the supervision: the dense angle × depth grid of every
annotated point and view is collapsed to its best cell before training
(:mod:`grasppanda.datasets.compact_labels`), so a seed is supervised with one grasp
instead of 48 and a batch carries megabytes of labels instead of gigabytes.

``configs/_base_/models/economicgrasp.yaml`` is the full reference model. The
implementation is written from the paper and from the behaviour of
``iSEE-Laboratory/EconomicGrasp`` (commit ``7361bb6``); with the authors' checkpoint
converted by ``tools/convert_checkpoint.py`` it is compared stage by stage against
activations recorded from that implementation
(``tests/integration/test_economicgrasp_parity.py``).

Behaviours of the reference that change numbers are named options whose defaults
reproduce it. Shared with GSNet: the training view is sampled
(``view.train_selection``), widths are widened in the decoder only
(``head.width_decode_gain``), the sampling kernel skips points next to the origin
(``proposal.fps_skip_near_origin``). Its own: a seed is supervised only within 5 mm of
its label point (``assigner.max_match_distance``) and only at a view that kept its
label (``assigner.shared_views``), the classifiers carry a never-supervised class that
the decoder's arg-max still considers (``head.unmatched_class``,
``head.decode_argmax``), and the decoded score is a weighted sum of raw logits
(``head.score_decode``).

Known, deliberate differences from the reference:

* One model instance trains and predicts; the mode follows ``model.train()`` /
  ``model.eval()``.
* A frame with fewer graspable candidates than seeds yields fewer grasps instead of
  repeats of its first grasp; a frame with none yields an empty group.
* Which of two template views that share a labelled view keeps the label is a fixed
  rule here. In the reference it is the outcome of an indexed assignment with repeated
  indices, which is undefined on a GPU and differs between runs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

from grasppanda.models.base import Assigner, Backbone, Grouper, HeadBundle, Proposal, ViewStage
from grasppanda.registry import ASSIGNERS, DETECTORS, HEADS, Cfg
from grasppanda.structures import ContractError, Gripper

from .seed_based import SeedGraspDetector

__all__ = ["EconomicGrasp"]


def _label_problems(assigner: Any, head: Any) -> Iterator[str]:
    """Compare the label formats of assigner and head; works on classes and instances."""

    def name(component: Any) -> str:
        return component.__name__ if isinstance(component, type) else type(component).__name__

    produced = getattr(assigner, "label_format", "dense")
    consumed = getattr(head, "label_format", "dense")
    if produced != consumed:
        yield (
            f"{name(assigner)} returns '{produced}' matched labels, but {name(head)} is "
            f"supervised with '{consumed}' ones. A head bundle and the assigner that feeds it "
            "must agree on the label set; the dataset must serve that label set too."
        )


@DETECTORS.register()
class EconomicGrasp(SeedGraspDetector):
    """Economic-supervision grasp detector; see the module docstring and
    ``economicgrasp.yaml``.

    The arguments are those of :class:`SeedGraspDetector`. In addition to the stage
    contracts checked there, the assigner and the head bundle must agree on the kind of
    grasp labels (dense grids or compact, one grasp per point and view).
    """

    status = "tested"
    source = "EconomicGrasp (Wu et al., ECCV 2024)"

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
        problems = list(_label_problems(self.assigner, self.head))
        if problems:
            raise ContractError(f"{type(self).__name__}: " + " ".join(problems))

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Stage contracts, plus the label format shared by assigner and head."""
        yield from super().check_config(node)
        classes = {}
        for slot, registry in (("assigner", ASSIGNERS), ("head", HEADS)):
            child = node.get(slot)
            if not (isinstance(child, Mapping) and "type" in child):
                return
            classes[slot] = registry.get(child["type"])
        yield from _label_problems(**classes)
