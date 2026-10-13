"""Options of HGGD's stages.

The stages are private to the detector (no second method uses them yet), so they are
not registered components. Their options are plain dataclasses instead: the config
validator checks every key and type of ``model.image``, ``model.proposal`` … against
these declarations, and each value has exactly one owner. Every default is the value of
the reference implementation (THU-VCLab/HGGD at ``69580ff``, with the options of its
shipped ``train_graspnet.sh`` / ``test_graspnet.sh``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal, TypeVar

__all__ = [
    "AnchorOptions",
    "EncoderOptions",
    "ImageOptions",
    "LocalOptions",
    "ProposalOptions",
    "RegionOptions",
    "options_from",
]

T = TypeVar("T")


def options_from(cls: type[T], value: Any) -> T:
    """``value`` as an instance of the options class ``cls``.

    Accepts ``None`` (all defaults), an instance, or the mapping a config node arrives as.
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        known = {field.name for field in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(value) - known)
        if unknown:
            raise TypeError(
                f"{cls.__name__} has no option {unknown}; its options are {sorted(known)}"
            )
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in value.items()})
    raise TypeError(f"expected {cls.__name__} options or a mapping, got {type(value).__name__}")


@dataclass(frozen=True)
class ImageOptions:
    """The image the first network sees.

    Attributes:
        size: ``(width, height)`` of the network image. The sensor image must be an
            integer multiple of it.
        resize: ``pil_bicubic`` reproduces Pillow's ``Image.resize`` (two antialiased
            bicubic passes, colour rounded to 8 bits after each), which the reference
            feeds its network; ``torch_bicubic`` is PyTorch's antialiased interpolation.
        depth_clip: Bound of the depth channel after subtracting its mean, metres.
    """

    size: tuple[int, int] = (640, 360)
    resize: Literal["pil_bicubic", "torch_bicubic"] = "pil_bicubic"
    depth_clip: float = 1.0


@dataclass(frozen=True)
class EncoderOptions:
    """The image network.

    Attributes:
        base_width: Channels of the stem; the four stages have 2, 4, 8 and 16 times as
            many. (The block layout is ResNet-34's; the width is an eighth of it.)
        blocks: Residual blocks per stage.
        head_channels: Hidden width of the five map heads.
        detach_features: Hand the feature map to the region stage without gradient, so
            that the region loss does not train the image network. ``false`` is a
            different training recipe.
    """

    base_width: int = 8
    blocks: tuple[int, int, int, int] = (3, 4, 6, 3)
    head_channels: int = 32
    detach_features: bool = True


@dataclass(frozen=True)
class ProposalOptions:
    """The heat-map stage; see :mod:`.heatmap` for the algorithm.

    Attributes:
        num_angles: In-plane angle classes; class ``k`` covers
            ``[-pi/2 + k pi/K, -pi/2 + (k + 1) pi/K)``.
        stride: Pixels of the network image per cell of the angle maps (and of the
            feature map handed to the region stage).
        anchor_width: Width, in pixels of the network image, that a zero width offset
            decodes to.
        anchor_depth: Depth offset, in metres, that a depth offset of ``0.5`` decodes to.
        blur_kernel, blur_sigma: Gaussian blur of the squashed heat-map.
        grid_size: Edge of the cells the candidates are drawn from, pixels.
        max_proposals, train_max_proposals: Partial grasps kept per frame.
        candidate_factor: Grid cells examined per kept proposal.
        score_threshold, train_score_threshold: Heat a candidate pixel must exceed.
        min_distance: A candidate is dropped when an accepted one is closer than this
            many pixels *and* within ``min_angle`` degrees.
        min_angle: See ``min_distance``.
        heatmap_weight, offset_weight, angle_weight: Weights of the three loss terms.
            The reference's shell script and its argument defaults disagree
            (``--reg-b 5`` against a default of 1); these are the script's.
        label_min_score: Labels below this quality are ignored by the dense targets.
        max_labels: Labels used per frame; more are sub-sampled at random.
        peak_radius: Radius in pixels of the Gaussian drawn per label centre; its
            standard deviation is ``(2 radius + 1) / 6``.
        peak_min_distance: A label centre closer than this to an earlier one draws no peak.
        depth_window: Half-size, in sensor pixels, of the window whose median is the
            surface depth below a label.
        positive_count_threshold: Soft label count ``2 / (1 + exp(-n)) - 1`` from which
            a cell is a positive of its angle class. The reference value 0.5 makes a
            cell with a single label a negative.
    """

    num_angles: int = 6
    stride: int = 8
    anchor_width: float = 50.0
    anchor_depth: float = 0.02
    blur_kernel: int = 9
    blur_sigma: float = 10.0
    grid_size: int = 8
    max_proposals: int = 48
    train_max_proposals: int = 128
    candidate_factor: int = 10
    score_threshold: float = 0.01
    train_score_threshold: float = 0.0
    min_distance: float = 8.0
    min_angle: float = 30.0
    heatmap_weight: float = 1.0
    offset_weight: float = 5.0
    angle_weight: float = 1.0
    label_min_score: float = 0.8
    max_labels: int = 5000
    peak_radius: int = 10
    peak_min_distance: float = 8.0
    depth_window: int = 2
    positive_count_threshold: float = 0.5


@dataclass(frozen=True)
class RegionOptions:
    """The region stage; see :mod:`.region`.

    Attributes:
        num_points: Points per region.
        min_points: A region needs *more* than this many scene points.
        feature_neighbours: Feature-map cells pooled per scene point.
        jitter: Standard deviation, in metres, of the noise added to region coordinates
            before sampling. The reference applies it at inference too.
        zero_depth_window: Half-size in sensor pixels of the window searched for depth
            around a centre pixel that has none.
        zero_depth_centres: ``append`` reproduces the reference, which moves the centres
            of pixels without depth to the end of the centre list while the grasp
            attributes keep their order; ``in_place`` keeps each centre with its grasp.
    """

    num_points: int = 512
    min_points: int = 32
    feature_neighbours: int = 8
    jitter: float = 5e-4
    zero_depth_window: int = 4
    zero_depth_centres: Literal["append", "in_place"] = "append"


@dataclass(frozen=True)
class LocalOptions:
    """The region network.

    Attributes:
        info_embedding: Width of the embedding of ``(theta, width, depth offset)``.
        hidden: Width of the anchor and offset read-outs.
        dropout: Dropout probability on the region vector.
    """

    info_embedding: int = 32
    hidden: int = 256
    dropout: float = 0.3


@dataclass(frozen=True)
class AnchorOptions:
    """The rotation anchors; see :mod:`.anchor_head`.

    Attributes:
        num_anchors: Anchors per out-of-plane angle; ``num_anchors ** 2`` classes.
        top_k: Anchors decoded per region (``--local-k 10`` of the test script; the
            argument default is 3).
        offset_scale: Metres a clipped offset of ``1`` moves the grasp centre.
        max_gamma: Anchors with ``|gamma|`` of at least this many degrees are dropped.
        grasp_depth: Insertion depth of every decoded grasp, metres.
        finger_pixels: Finger thickness, in sensor pixels, whose back-projection is the
            decoded ``height``.
        frame: ``pixel_keypoints`` builds the in-plane axes from pixel offsets
            back-projected with ``fx`` and ``fy`` separately, as the reference does;
            they are orthogonal only up to ``|fx - fy| / f``. ``orthonormal`` builds an
            exact rotation.
        match_radius: Labels closer than this to a region centre are its labels, metres.
        max_matches: Nearest labels kept per region (the argument default of the
            training script, which its shell script does not set).
        positive_similarity: Rotation similarity from which an anchor is a positive.
        offset_weight: Weight of the offset loss (``--offset-d 1`` of the shell script;
            the argument default is 5).
        shift_epochs: Epochs during which the anchor positions follow the labels; 0
            freezes them.
        shift_pool: Matched labels pooled before the anchors move.
    """

    num_anchors: int = 7
    top_k: int = 10
    offset_scale: float = 0.02
    max_gamma: float = 60.0
    grasp_depth: float = 0.02
    finger_pixels: float = 20.0
    frame: Literal["pixel_keypoints", "orthonormal"] = "pixel_keypoints"
    match_radius: float = 0.02
    max_matches: int = 500
    positive_similarity: float = 0.99
    offset_weight: float = 1.0
    shift_epochs: int = 5
    shift_pool: int = 1_000_000
