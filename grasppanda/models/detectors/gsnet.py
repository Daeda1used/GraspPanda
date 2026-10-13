"""GSNet: Graspness Discovery in Clutters for Fast and Accurate Grasp Detection (ICCV 2021).

The method as a configuration of registered stages:

=============  ==========================  ==============================================
slot           reference component         what it does
=============  ==========================  ==============================================
``backbone``   ``SparseUNet14``            per-point features from a sparse U-Net
``proposal``   ``GraspnessFPS``            objectness + graspness → 1024 seeds by FPS
``view``       ``TemplateViewStage``       300 view scores, one approach per seed
``assigner``   ``NearestLabelAssigner``    labels and crop frame at the chosen view
``grouper``    ``CylinderGroup``           approach-aligned cylinder among the seeds
``head``       ``SWADHead``                score and width over 12 angles × 4 depths
=============  ==========================  ==============================================

``configs/_base_/models/gsnet.yaml`` is the full reference model. The implementation is
written from the paper and from the behaviour of ``graspnet/graspness_unofficial``
(commit ``b5abf5a``); with the authors' checkpoint converted by
``tools/convert_checkpoint.py`` it is compared stage by stage against activations
recorded from that implementation (``tests/integration/test_gsnet_parity.py``).

Behaviours of the reference that change numbers are named options whose defaults
reproduce it: the training view is sampled (``view.train_selection``), label matching
has no distance limit (``assigner.max_match_distance``), score targets are normalised
with per-batch extrema (``head.score_target``), widths are widened in the decoder only
(``head.width_decode_gain``), and the sampling kernel skips points next to the origin
(``proposal.fps_skip_near_origin``).

Known, deliberate differences from the reference:

* One model instance trains and predicts; the mode follows ``model.train()`` /
  ``model.eval()``.
* A frame with fewer graspable candidates than seeds yields fewer grasps instead of
  repeats of its first grasp; a frame with none yields an empty group (the reference
  reads out of bounds there).
* Losses over an empty selection are zero rather than NaN, and a seed with constant
  view scores is sampled uniformly rather than raising.
"""

from __future__ import annotations

from grasppanda.registry import DETECTORS

from .seed_based import SeedGraspDetector

__all__ = ["GSNet"]


@DETECTORS.register()
class GSNet(SeedGraspDetector):
    """Graspness-based grasp detector; see the module docstring and ``gsnet.yaml``."""

    status = "tested"
    source = "GSNet (Wang et al., ICCV 2021)"
