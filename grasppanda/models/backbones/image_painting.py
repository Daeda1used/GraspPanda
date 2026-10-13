"""Point painting: image features for the scene encoders of the seed-based family.

The seed-based methods read a point cloud. These two adapters put an image encoder next
to any registered scene encoder: the colour image is encoded once, every point is
projected into the image with the camera matrix, the feature maps are sampled there
bilinearly, and the result is given to the 3-D network

``fusion: input``
    as extra input channels (PointPainting, PointAugmenting): the last
    ``paint_channels`` of the wrapped encoder's ``in_channels`` are the painted features,
    so set its ``in_channels`` to what the detector feeds plus ``paint_channels``;
``fusion: output``
    as a residual on the returned features (late fusion): the wrapped encoder is
    unchanged, and the projection starts at zero, so training starts from the plain
    3-D model.

What the data pipeline has to deliver, and does: the colour image
(``images: [rgb]`` of a scene dataset, ``observation.rgb``), the camera matrix
(``observation.intrinsics``) and points in the camera frame. A dataset that serves images
refuses point-space augmentation (a flip or a rotation of the cloud would leave the image
behind), so the pixel of a point is its projection, exactly; no per-point pixel index has
to travel through the batch. Sub-sampling does not matter: the projection is per point.

Two classes, because where the features live is a class-level contract:
:class:`ImagePainted` wraps a ``dense`` encoder and is ``dense``,
:class:`ImagePaintedSubsampled` wraps a ``subsampled`` one and is ``subsampled``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.models.base import Backbone
from grasppanda.models.image_encoders import ImageEncoder
from grasppanda.registry import BACKBONES, IMAGE_ENCODERS, Cfg
from grasppanda.structures import ContractError, Observation, PointFeatures, PointSet

__all__ = ["ImagePainted", "ImagePaintedSubsampled", "sample_maps"]


def sample_maps(
    maps: list[Tensor], points: PointSet, intrinsics: Tensor, image_size: tuple[int, int]
) -> Tensor:
    """Sample every ``(B, C_i, h_i, w_i)`` map at the pixel of every point.

    Args:
        maps: Feature maps of the batch of images, any resolutions.
        points: Points in the camera frame of their image.
        intrinsics: ``(B, 3, 3)`` camera matrices of the full-size images.
        image_size: ``(H, W)`` of those images.

    Returns:
        ``(N, sum C_i)`` features; zero for points that project outside the image.
    """
    if points.frame != "camera":
        raise ContractError(
            f"point painting projects points with the camera matrix; the cloud is in frame "
            f"'{points.frame}'"
        )
    batch = points.batch_index
    coords = points.coords
    camera = intrinsics[batch]
    depth = coords[:, 2].clamp_min(1e-6)
    u = camera[:, 0, 0] * coords[:, 0] / depth + camera[:, 0, 2]
    v = camera[:, 1, 1] * coords[:, 1] / depth + camera[:, 1, 2]
    height, width = image_size
    # Pixel centres are at integer coordinates; grid_sample wants [-1, 1] over the edges.
    grid = torch.stack([(u + 0.5) / width * 2 - 1, (v + 0.5) / height * 2 - 1], dim=-1)
    inside = (grid.abs() <= 1).all(dim=-1, keepdim=True).to(coords.dtype)
    sampled = []
    for feature in maps:
        out = coords.new_zeros(len(coords), feature.shape[1])
        for index in range(feature.shape[0]):
            mine = batch == index
            if mine.any():
                values = F.grid_sample(
                    feature[index : index + 1],
                    grid[mine].view(1, -1, 1, 2).to(feature.dtype),
                    mode="bilinear",
                    padding_mode="border",  # a border pixel reads its own cell, not a fade
                    align_corners=False,
                )
                out[mine] = values[0, :, :, 0].t().to(out.dtype)
        sampled.append(out * inside)
    return torch.cat(sampled, dim=1)


class _Painted(Backbone):
    """Shared implementation; see the module docstring.

    Args:
        backbone: The wrapped scene encoder.
        image_encoder: Any registered image encoder with the input layout ``rgb``.
        fusion: ``input`` or ``output``; see the module docstring.
        paint_channels: Width the sampled image features are projected to (``input``).
            The wrapped encoder's ``in_channels`` must be that much larger than what the
            detector feeds (3 constant ones for GSNet: ``in_channels: 35`` with 32).
        image_scale: The colour image is resampled by this factor before it is encoded
            (0.5 turns a 1280 x 720 sensor image into 640 x 360: a quarter of the memory).
    """

    uses_observation: ClassVar[bool] = True
    #: Compose inserts an adapter where a contract asks for it; no contract asks for this one.
    compose_inserts: ClassVar[bool] = False
    required_inputs: ClassVar[frozenset[str]] = frozenset({"rgb", "intrinsics"})

    def __init__(
        self,
        backbone: Cfg[Backbone],
        image_encoder: Cfg[ImageEncoder],
        fusion: Literal["input", "output"] = "input",
        paint_channels: int = 32,
        image_scale: float = 1.0,
    ) -> None:
        super().__init__()
        name = type(self).__name__
        if not 0.0 < image_scale <= 1.0:
            raise ValueError(f"{name}: image_scale must be in (0, 1], got {image_scale}")
        self.image_scale = image_scale
        if fusion not in ("input", "output"):
            raise ValueError(f"{name}: fusion must be input or output, got {fusion!r}")
        self.fusion = fusion
        self.paint_channels = paint_channels
        self.image_encoder = IMAGE_ENCODERS.build(image_encoder)
        if self.image_encoder.input_layout != "rgb":
            raise ContractError(
                f"{name} paints from the colour image; {type(self.image_encoder).__name__} "
                f"declares the input layout '{self.image_encoder.input_layout}'. Use an "
                "encoder with depth: rgb_only (the points already carry the depth)."
            )
        self.backbone = BACKBONES.build(backbone)
        painted = paint_channels if fusion == "input" else 0
        #: What the detector feeds; the wrapped encoder reads the painted channels after it.
        self.in_channels = self.backbone.in_channels - painted
        if self.in_channels < 0:
            raise ValueError(
                f"{name} with fusion: input hands {paint_channels} painted channels to "
                f"{type(self.backbone).__name__}, which has in_channels="
                f"{self.backbone.in_channels}. Set its in_channels to what the detector "
                f"feeds plus paint_channels (3 + {paint_channels} for constant ones)."
            )
        if getattr(self.backbone, "output", None) != self.output:
            raise ValueError(
                f"{name} wraps encoders that return `{self.output}` features, but "
                f"{type(self.backbone).__name__} declares "
                f"output={getattr(self.backbone, 'output', None)!r}. {self._other()}"
            )
        self.out_channels = self.backbone.out_channels
        sampled = sum(self.image_encoder.out_channels)
        width = paint_channels if fusion == "input" else self.out_channels
        self.project = nn.Linear(sampled, width)
        if fusion == "output":  # start as the plain 3-D model
            nn.init.zeros_(self.project.weight)
            nn.init.zeros_(self.project.bias)

    @classmethod
    def _other(cls) -> str:
        other = "ImagePaintedSubsampled" if cls.output == "dense" else "ImagePainted"
        return f"Use {other}."

    @classmethod
    def check_config(cls, node: dict[str, Any]) -> Iterator[str]:
        """What is wrong with the node, before anything is built."""
        if not isinstance(node.get("image_encoder"), dict):
            yield (
                f"{cls.__name__} needs an `image_encoder` node (for example the node of a "
                "preset under grasppanda/presets/image_encoder/)."
            )
        inner = node.get("backbone")
        if isinstance(inner, dict) and "type" in inner:
            output = getattr(BACKBONES.get(inner["type"]), "output", None)
            if output != cls.output:
                yield (
                    f"{cls.__name__} wraps encoders that return `{cls.output}` features, but "
                    f"{inner['type']} returns `{output}`. {cls._other()}"
                )

    def _paint(self, points: PointSet, observation: Observation) -> Tensor:
        if observation is None or observation.rgb is None or observation.intrinsics is None:
            raise ContractError(
                f"{type(self).__name__} reads the colour image and the camera matrix, but the "
                "observation has none. Serve them with `images: [rgb]` in the dataset node "
                "(and remove point-space augmentation such as RandomFlip / RandomRotation, "
                "which a dataset with images refuses)."
            )
        rgb = observation.rgb
        image = rgb
        if self.image_scale != 1.0:
            image = F.interpolate(
                rgb,
                scale_factor=self.image_scale,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        maps = self.image_encoder(image)
        return self.project(sample_maps(maps, points, observation.intrinsics, rgb.shape[-2:]))

    def forward(
        self, points: PointSet, feats: Tensor, observation: Observation | None = None
    ) -> PointFeatures:
        if self.fusion == "input":
            painted = self._paint(points, observation)
            return self.backbone(points, torch.cat([feats, painted.to(feats.dtype)], dim=1))
        out = self.backbone(points, feats)
        painted = self._paint(out.points, observation)
        return PointFeatures(out.feats + painted.to(out.feats.dtype), out.points)


@BACKBONES.register()
class ImagePainted(_Painted):
    """Give a `dense` scene encoder image features sampled at the pixel of every point.

    See the module docstring; arguments as :class:`_Painted`: ``backbone``,
    ``image_encoder``, ``fusion`` (``input`` or ``output``), ``paint_channels``,
    ``image_scale``.
    """

    output = "dense"
    status = "tested"
    extras = ()
    source = (
        "PointPainting (Vora et al., CVPR 2020), PointAugmenting (Wang et al., CVPR 2021); "
        "written here"
    )

    def __init__(
        self,
        backbone: Cfg[Backbone],
        image_encoder: Cfg[ImageEncoder],
        fusion: Literal["input", "output"] = "input",
        paint_channels: int = 32,
        image_scale: float = 1.0,
    ) -> None:
        super().__init__(backbone, image_encoder, fusion, paint_channels, image_scale)


@BACKBONES.register()
class ImagePaintedSubsampled(_Painted):
    """Give a `subsampled` scene encoder image features sampled at the pixel of every point.

    With ``fusion: output`` the features are sampled at the pixels of the sub-sampled
    points the wrapped encoder returns. Arguments as :class:`ImagePainted`.
    """

    output = "subsampled"
    status = "tested"
    extras = ()
    source = (
        "PointPainting (Vora et al., CVPR 2020), PointAugmenting (Wang et al., CVPR 2021); "
        "written here"
    )

    def __init__(
        self,
        backbone: Cfg[Backbone],
        image_encoder: Cfg[ImageEncoder],
        fusion: Literal["input", "output"] = "input",
        paint_channels: int = 32,
        image_scale: float = 1.0,
    ) -> None:
        super().__init__(backbone, image_encoder, fusion, paint_channels, image_scale)
