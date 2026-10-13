"""Scale-Balanced-Grasp: Towards Scale Balanced 6-DoF Grasp Detection in Cluttered
Scenes (CoRL 2022).

The method keeps the flow of GraspNet-baseline and changes four of its parts. As a
configuration of registered stages:

=============  ==========================  ==============================================
slot           reference component         what differs from GraspNet-baseline
=============  ==========================  ==============================================
``backbone``   ``PointNet2Transformer``    two abstraction levels, each with an attention
                                           layer; no feature propagation
``proposal``   ``SubsampledSeeds``         —
``view``       ``ObjectnessViewStage``     ``min_graspable_views: 10``: a positive needs
                                           more than ten graspable views
``assigner``   ``ScaleBalancedAssigner``   also returns a loss weight per seed
``grouper``    ``DepthCylinderGroup``      four radii of 0.08 m × (0.25 … 1), fused, plus
                                           a gated residual of the seed features
``head``       ``OperationHead``           weighted losses; score supervised on all depth
                                           bins of a supervised seed; decoded score also
                                           times P(object)
=============  ==========================  ==============================================

``configs/_base_/models/scale_balanced_grasp.yaml`` is the full reference model: the
``GraspNet_MSCQ`` network of ``mahaoxiang822/Scale-Balanced-Grasp`` (commit ``995d6f8``)
with its scale-balanced loss, as that repository's ``train.py`` and ``test.py`` build
it. With the authors' checkpoint converted by ``tools/convert_checkpoint.py`` it is
compared stage by stage against activations recorded from that implementation
(``tests/integration/test_sbg_parity.py``).

The scale-balanced loss weights come from a histogram of grasp widths over the training
labels. The model does not own that histogram: the dataset serves it with every batch
(``scale_statistic:`` section of the dataset config), so it can neither be loaded from
a path relative to the working directory nor go stale against the label set unnoticed.

Not part of this detector, and not implemented: *noisy-clean mixing*, a training-data
option that also feeds clouds synthesised from the object models at their annotated
poses, and *object-balanced sampling*, an inference-time re-sampling of the seeds that
needs instance masks from a second, separately trained segmentation network.
"""

from __future__ import annotations

from grasppanda.registry import DETECTORS

from .graspnet_baseline import GraspNetBaseline

__all__ = ["ScaleBalancedGrasp"]


@DETECTORS.register()
class ScaleBalancedGrasp(GraspNetBaseline):
    """Scale-Balanced-Grasp; see the module docstring and ``scale_balanced_grasp.yaml``.

    The arguments and the checks are those of :class:`GraspNetBaseline`: the method is a
    different choice of components and options in the same flow.
    """

    status = "tested"
    source = "Scale-Balanced-Grasp (Ma and Huang, CoRL 2022)"
