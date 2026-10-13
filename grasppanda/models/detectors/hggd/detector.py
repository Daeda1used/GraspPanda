"""HGGD: Efficient Heatmap-Guided 6-DoF Grasp Detection in Cluttered Scenes (RA-L 2023).

HGGD shares no stage with the seed-based family. It reads an RGB-D *image*, proposes
partial grasps from a heat-map, and completes each from a small point region:

==============  ==========================  ===============================================
stage           module                      what it does
==============  ==========================  ===============================================
input           :mod:`.image`               ``[depth, R, G, B]`` at the network resolution
image network   :class:`.ImageNet`          heat-map, in-plane angle classes and offsets
proposal        :class:`.HeatmapStage`      heat-map peaks → partial grasps; 2-D targets
regions         :class:`.RegionStage`       lift centres, attach image features, ball crop
region network  :class:`.LocalNet`          PointNet + partial-grasp info
anchors         :class:`.AnchorHead`        ``(gamma, beta)`` anchors, offsets, decoding
==============  ==========================  ===============================================

The stages are private to this detector; partial grasps travel between them as a
:class:`.PartialGraspSet`. ``configs/_base_/models/hggd.yaml`` is the full reference
model. The implementation is written from the paper and from the behaviour of
``THU-VCLab/HGGD`` (commit ``69580ff``); with the authors' checkpoints converted by
``tools/convert_checkpoint.py`` it is compared stage by stage against values recorded
from that implementation (``tests/integration/test_hggd_parity.py``).

Training. No ground truth enters the forward pass: the dense image targets are built
from the per-view labels, and the region targets are matched *after* the forward pass to
the centres the network itself proposed. Both losses are summed into one step; the
feature map is handed to the region stage without gradient, so the region loss does not
reach the image network (``encoder.detach_features``).

Known, deliberate differences from the reference:

* One model instance trains and predicts; the mode follows ``model.train()`` /
  ``model.eval()`` and selects the proposal budget and score threshold.
* Camera intrinsics come from the observation, not from per-camera constants.
* A frame without proposals or without a region yields an explicit empty group (the
  reference writes no file, or fails); a training batch without regions still trains
  the image network (the reference skips the step).
* The reference's random draws at inference (region jitter, sampling start points,
  re-sampling of small regions) come from PyTorch's generators here as well, so
  predictions are reproducible under a seed but not bit-equal to a reference run;
  ``run_stages`` accepts recorded draws for comparisons.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor

from grasppanda.models.base import BaseGraspDetector
from grasppanda.models.image_encoders import ImageEncoder
from grasppanda.registry import DETECTORS, IMAGE_ENCODERS, Cfg
from grasppanda.structures import ContractError, GraspGroup, Gripper, Observation, Supervision

from .anchor_head import AnchorHead
from .heatmap import HeatmapStage
from .image import encode_image
from .image_net import ImageNet
from .local_net import LocalNet
from .options import (
    AnchorOptions,
    EncoderOptions,
    ImageOptions,
    LocalOptions,
    ProposalOptions,
    RegionOptions,
    options_from,
)
from .region import RegionStage
from .structures import ImagePredictions, RegionDraws, RegionSet

__all__ = ["HGGD", "StageOutputs"]

#: Name of the per-view grasp labels in ``Supervision.extras``.
LABELS = "view_grasps"


@dataclass(eq=False)
class StageOutputs:
    """Everything the stages of one forward pass produced.

    Attributes:
        image: ``(B, 4, U, V)`` network input.
        raw: Outputs of the image network (with gradient).
        maps: The same after squashing, blurring and clamping (no gradient).
        regions: Regions with the partial grasps they belong to.
        proposals: Number of partial grasps per frame, before regions were cut.
        logits: ``(R, A)`` anchor logits.
        offsets: ``(R, A, 3)`` raw centre offsets.
    """

    image: Tensor
    raw: ImagePredictions
    maps: ImagePredictions
    regions: RegionSet
    proposals: Tensor
    logits: Tensor
    offsets: Tensor


@DETECTORS.register()
class HGGD(BaseGraspDetector):
    """Heat-map guided grasp detector; see the module docstring and ``hggd.yaml``.

    Args:
        gripper: Gripper the predictions refer to; ``None`` is the benchmark gripper.
        image: Network input; :class:`~.options.ImageOptions`.
        encoder: Image network; :class:`~.options.EncoderOptions`.
        image_encoder: The encoder of the image network, any registered image encoder.
            ``None`` is the reference encoder described by ``encoder.base_width`` and
            ``encoder.blocks``; another encoder is followed by a pyramid decoder of
            ``encoder.decoder_channels``.
        proposal: Heat-map stage; :class:`~.options.ProposalOptions`.
        region: Region stage; :class:`~.options.RegionOptions`.
        local: Region network; :class:`~.options.LocalOptions`.
        anchors: Rotation anchors; :class:`~.options.AnchorOptions`.
    """

    status = "tested"
    source = "HGGD (Chen et al., RA-L 2023)"
    required_inputs = frozenset({"rgb", "depth", "intrinsics", "points"})
    #: The per-view grasp labels the losses are computed from.
    needs_supervision: ClassVar[dict[str, tuple[str, Any] | None]] = {f"extras.{LABELS}": None}

    def __init__(
        self,
        gripper: Gripper | None = None,
        image: ImageOptions | None = None,
        encoder: EncoderOptions | None = None,
        proposal: ProposalOptions | None = None,
        region: RegionOptions | None = None,
        local: LocalOptions | None = None,
        anchors: AnchorOptions | None = None,
        image_encoder: Cfg[ImageEncoder] | None = None,
    ) -> None:
        super().__init__()
        self.gripper = options_from(Gripper, gripper)
        self.image_options = options_from(ImageOptions, image)
        encoder = options_from(EncoderOptions, encoder)
        local = options_from(LocalOptions, local)
        self.proposal = HeatmapStage(proposal)
        built = None if image_encoder is None else IMAGE_ENCODERS.build(image_encoder)
        if built is not None and (encoder.base_width, tuple(encoder.blocks)) != (8, (3, 4, 6, 3)):
            raise ContractError(
                f"{type(self).__name__}: encoder.base_width and encoder.blocks describe the "
                f"reference encoder, but image_encoder puts {type(built).__name__} in its "
                "place. Set them on the image_encoder node (HGGDResidualEncoder has both "
                "options) or remove them."
            )
        self.image_net = ImageNet(
            in_channels=4,
            base_width=encoder.base_width,
            blocks=encoder.blocks,
            head_channels=encoder.head_channels,
            num_angles=self.proposal.num_angles,
            stride=self.proposal.stride,
            detach_features=encoder.detach_features,
            encoder=built,
            decoder_channels=encoder.decoder_channels,
        )
        self.region = RegionStage(region)
        self.anchors = AnchorHead(anchors, gripper=self.gripper)
        self.local_net = LocalNet(
            feature_channels=self.image_net.feature_channels,
            num_anchors=self.anchors.num_classes,
            info_embedding=local.info_embedding,
            hidden=local.hidden,
            dropout=local.dropout,
        )
        problems = list(self._problems(self.image_options, self.proposal.options))
        if problems:
            raise ContractError(f"{type(self).__name__}: " + " ".join(problems))
        self._epoch: int | None = None

    # ------------------------------------------------------------------ config check

    @staticmethod
    def _problems(image: ImageOptions, proposal: ProposalOptions) -> Iterator[str]:
        width, height = image.size
        if proposal.stride not in (2, 4, 8, 16):
            yield f"proposal.stride must be 2, 4, 8 or 16, got {proposal.stride}."
        elif width % proposal.stride or height % proposal.stride:
            yield (
                f"image.size {tuple(image.size)} is not a multiple of proposal.stride "
                f"{proposal.stride}, so the angle maps would not cover the image."
            )
        if proposal.grid_size > min(width, height):
            yield f"proposal.grid_size {proposal.grid_size} exceeds the image {tuple(image.size)}."

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """Problems between the option groups, visible in the config node alone."""
        groups = {"image": ImageOptions, "proposal": ProposalOptions, "anchors": AnchorOptions}
        built = {}
        for name, kind in groups.items():
            value = node.get(name)
            if value is not None and not isinstance(value, Mapping):
                return
            try:
                built[name] = options_from(kind, value)
            except TypeError:
                return  # reported by the signature check
        yield from cls._problems(built["image"], built["proposal"])
        anchors = built["anchors"]
        if not 1 <= anchors.top_k <= anchors.num_anchors**2:
            yield (
                f"anchors.top_k is {anchors.top_k}, but there are only "
                f"{anchors.num_anchors**2} anchors (num_anchors squared)."
            )

    # ----------------------------------------------------------------------- inputs

    def _scene_points(self, observation: Observation) -> list[Tensor]:
        name = type(self).__name__
        missing = sorted(self.required_inputs - observation.provides())
        if missing:
            raise ContractError(
                f"{name} reads {sorted(self.required_inputs)} from the observation, but "
                f"{missing} is missing. A scene dataset serves the images with "
                "`images: [rgb, depth]`; the cloud is the usual `points`."
            )
        points = observation.points
        batch = len(observation.keys)
        if not (
            points.batch_size == batch
            and observation.rgb.shape[0] == batch
            and observation.depth.shape[0] == batch
            and observation.intrinsics.shape[0] == batch
        ):
            raise ContractError(
                f"{name}: the observation lists {batch} frame(s) but holds "
                f"{points.batch_size} cloud(s), {observation.rgb.shape[0]} colour image(s), "
                f"{observation.depth.shape[0]} depth image(s) and "
                f"{observation.intrinsics.shape[0]} camera matrices"
            )
        if points.frame != "camera":
            raise ContractError(
                f"{name} lifts image pixels into the camera frame; the cloud is in frame "
                f"'{points.frame}'"
            )
        width, height = self.image_options.size
        sensor_height, sensor_width = observation.depth.shape[-2:]
        if (
            sensor_width % width
            or sensor_height % height
            or (sensor_width // width != sensor_height // height)
        ):
            raise ContractError(
                f"{name}: the {sensor_width} x {sensor_height} sensor image is not an integer "
                f"multiple of the network image {width} x {height} (image.size)"
            )
        offsets = (
            [0, len(points)] if points.batch_offsets is None else points.batch_offsets.tolist()
        )
        return [points.coords[offsets[b] : offsets[b + 1]] for b in range(batch)]

    # ----------------------------------------------------------------------- stages

    def run_stages(
        self,
        observation: Observation,
        *,
        image: Tensor | None = None,
        candidates: list[Tensor] | None = None,
        draws: list[RegionDraws | None] | None = None,
    ) -> StageOutputs:
        """Run the stage flow once and return every intermediate result.

        :meth:`loss` and :meth:`predict` are thin wrappers around this method. The
        keyword arguments replace steps that cannot be reproduced otherwise by recorded
        values, which is how the stages behind them are compared with a reference
        implementation: ``image`` the encoded network input, ``candidates`` the
        candidate pixels per frame, ``draws`` the random choices of the region stage.
        """
        points = self._scene_points(observation)
        size = tuple(self.image_options.size)
        if image is None:
            image = encode_image(
                observation.rgb,
                observation.depth,
                size,
                resize=self.image_options.resize,
                depth_clip=self.image_options.depth_clip,
            )
        raw = self.image_net(image)
        proposals, maps = self.proposal(raw, training=self.training, candidates=candidates)
        regions = self.region(
            proposals,
            points,
            raw.features,
            observation.depth,
            observation.intrinsics,
            size,
            draws=draws,
        )
        if self.training and len(regions) == 1:
            # Batch statistics of a single region are undefined; treat it as none.
            regions = RegionSet(
                points=regions.points[:0],
                grasps=regions.grasps.select(slice(0, 0)),
                kept=torch.zeros_like(regions.kept),
                support=regions.support,
            )
        if len(regions):
            grasps = regions.grasps
            # The side input in the units the reference feeds: radians, pixels, millimetres.
            info = torch.stack(
                [grasps.theta, grasps.width, grasps.depth_offset * 1000.0], dim=1
            ).to(torch.float32)
            logits, offsets = self.local_net(regions.points, info)
        else:
            logits = image.new_zeros(0, self.anchors.num_classes)
            offsets = image.new_zeros(0, self.anchors.num_classes, 3)
        counts = torch.bincount(proposals.frame, minlength=len(observation.keys))
        return StageOutputs(image, raw, maps, regions, counts, logits, offsets)

    # ------------------------------------------------------------------------- loss

    def on_train_epoch(self, epoch: int, max_epochs: int) -> None:
        """Called before every training epoch (``EpochSignalHook``); ``epoch`` is 0-based.

        The anchor positions follow the labels during the first
        ``anchors.shift_epochs`` epochs, and the label pool starts empty in each.
        """
        self._epoch = epoch
        self.anchors.clear_pool()

    def loss(self, observation: Observation, supervision: Supervision) -> dict[str, Tensor]:
        """Losses of the image network and of the region network, plus logged metrics."""
        name = type(self).__name__
        labels = supervision.extras.get(LABELS)
        if labels is None:
            raise ContractError(
                f"{name} trains on per-view grasp labels (supervision.extras['{LABELS}']), "
                "which this supervision does not carry. Set `view_grasps` in the dataset "
                "config and build the asset with `grasppanda prepare <dataset> --view-grasps`."
            )
        if len(labels) != len(observation.keys):
            raise ContractError(
                f"{name}: labels of {len(labels)} frame(s) for a batch of {len(observation.keys)}"
            )
        shifting = self.training and self.anchors.shift_epochs > 0
        if shifting and self._epoch is None:
            raise ContractError(
                f"{name} moves its rotation anchors during the first "
                f"{self.anchors.shift_epochs} epoch(s) (anchors.shift_epochs) and therefore "
                "has to be told the epoch. Add `- {type: EpochSignalHook}` to `hooks`, call "
                "`model.on_train_epoch(epoch, max_epochs)` yourself, or set "
                "`anchors.shift_epochs: 0` to keep the anchors fixed."
            )
        out = self.run_stages(observation)
        size = tuple(self.image_options.size)
        targets = self.proposal.encode(labels, observation.depth, size)
        losses = self.proposal.loss(out.raw, targets)
        matched = self.anchors.match(out.regions, labels, observation.intrinsics)
        moved = False
        if shifting and self._epoch < self.anchors.shift_epochs:
            moved = self.anchors.pool(matched)
        similarity, offset = self.anchors.encode(out.regions.grasps.theta, matched)
        losses.update(self.anchors.loss(out.logits, out.offsets, similarity, offset))
        frames = max(len(observation.keys), 1)
        device = out.image.device
        supervised = sum(len(rows) > 0 for rows in matched)
        losses["metric/proposals"] = out.proposals.float().mean()
        losses["metric/regions"] = torch.tensor(len(out.regions) / frames, device=device)
        losses["metric/supervised_regions"] = torch.tensor(
            supervised / max(len(matched), 1), device=device
        )
        losses["metric/anchors_moved"] = torch.tensor(float(moved), device=device)
        return losses

    # --------------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, observation: Observation) -> list[GraspGroup]:
        """One :class:`GraspGroup` per frame: region by region, anchors by descending score.

        A frame without proposals, or whose proposals have too little support, yields
        an empty group. Scores are the sigmoid of the anchor logits
        (``score_kind = "anchor_probability"``). Call ``model.eval()`` first.
        """
        return self.decode(self.run_stages(observation), observation)

    @torch.no_grad()
    def decode(self, out: StageOutputs, observation: Observation) -> list[GraspGroup]:
        """Turn the stage outputs of a forward pass into per-frame grasps (float32)."""
        scale = observation.depth.shape[-1] / self.image_options.size[0]
        groups = self.anchors.decode(
            out.logits, out.offsets, out.regions, observation.intrinsics, scale, observation.keys
        )
        results = []
        for frame, group in enumerate(groups):
            single = group._with(
                **{
                    name: getattr(group, name).to(torch.float32)
                    for name in ("translation", "rotation", "width", "depth", "score", "height")
                }
            )
            # The reference frame is a rotation only up to the camera's |fx - fy|.
            tolerance = self.anchors.rotation_tolerance(observation.intrinsics[frame])
            results.append(single.validate(atol=tolerance))
        return results
