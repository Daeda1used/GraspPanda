"""Bridges to the image-model libraries: timm, Hugging Face transformers, torchvision.

One class per library; each exposes every model of its library that can return spatial
feature maps. The three share how the input is treated, because the libraries do not:

* **Colour** arrives in ``[0, 1]`` and is normalised here with the mean and standard
  deviation the pretrained weights expect (read from the library where it states them).
* **Depth** is an explicit choice, ``depth``:

  ``rgb_only``
      the encoder sees colour only (``in_channels`` 3). Pretrained weights are used
      exactly as released.
  ``fourth_channel``
      the first convolution (or patch embedding) is widened to four inputs. The colour
      filters keep their pretrained values; the depth filters start at zero
      (``depth_init: zero``, the network starts as the colour model) or at the mean of
      the colour filters (``depth_init: mean``). No library ships weights that were
      pretrained on RGB-D: the depth filters are always trained here, and with
      ``frozen`` the widened convolution is the one layer of the model that still is.
      (timm's own ``in_chans=4`` repeats the colour filters instead; it is not used.)
  ``depth_stem``
      a small convolutional stem of its own turns depth into a three-channel residual
      that is added to the normalised colour image; its last layer starts at zero.
      The pretrained network is untouched.

* **Image size.** Convolutional networks take any size. Transformers often do not: a
  ViT needs a multiple of its patch size, window attention a multiple of its window, and
  some carry position embeddings of one fixed size. ``size_multiple`` resamples the
  image up to the next multiple; ``input_size`` resamples it to one fixed ``(H, W)``
  (build the model for that size with ``model_args``). The maps come back at the
  resampled size divided by the stride; the consumer resamples them to what it needs.
* **Strides and widths** are measured with one forward pass at construction
  (``probe_size``, or ``input_size`` when set), so they are what the model does, not
  what a table says.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from grasppanda.registry import IMAGE_ENCODERS
from grasppanda.structures import ContractError
from grasppanda.utils.imports import require

from .base import ImageEncoder

__all__ = ["HFImageEncoder", "TimmImageEncoder", "TorchvisionImageEncoder"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _widen_image_convolutions(model: nn.Module, init: str) -> None:
    """Give every convolution that reads the 3-channel image a fourth input channel.

    Usually that is one layer. Re-parameterisable stems (FastViT, RepViT) read the image
    with several parallel convolutions; all of them are widened. The modules are changed
    in place, so a library's own convolution class (TensorFlow-style padding, weight
    standardisation) keeps its behaviour.
    """
    found = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d) and module.in_channels == 3 and module.groups == 1
    ]
    if not found:
        raise ContractError(
            "depth: fourth_channel needs a model whose first layer is a convolution over "
            "three channels; this one has none. Use depth: depth_stem or rgb_only."
        )
    for module in found:
        weight = module.weight.detach()
        extra = weight.mean(dim=1, keepdim=True) if init == "mean" else weight[:, :1] * 0.0
        module.weight = nn.Parameter(torch.cat([weight, extra], dim=1))
        module.in_channels = 4
    # Libraries that check the channel count against their configuration.
    for other in model.modules():
        if getattr(other, "num_channels", None) == 3:
            other.num_channels = 4
        config = getattr(other, "config", None)
        if getattr(config, "num_channels", None) == 3:
            config.num_channels = 4


_ANNOUNCED: set[tuple[str, str]] = set()


def _announce_weights(bridge: str, library: str, name: str, licence: str | None) -> None:
    """Say once per process that pretrained weights come through a library, which
    downloads them on first use, and under which licence; to standard error."""
    if (library, name) in _ANNOUNCED:
        return
    _ANNOUNCED.add((library, name))
    terms = str(licence) if licence else "not stated by the library; read the model card"
    marks = ("-nc", "noncommercial", "non-commercial")
    restricted = any(mark in terms.lower() for mark in marks)
    print(
        f"{bridge}: pretrained weights '{name}' come through {library}, which downloads "
        f"them on first use. Licence: {terms}"
        + (" (NON-COMMERCIAL)" if restricted else "")
        + ". They are not part of GraspPanda.",
        file=sys.stderr,
    )


class _LibraryEncoder(ImageEncoder):
    """What the three bridges share: input treatment, freezing, measured contract."""

    def _setup(
        self,
        model: nn.Module,
        *,
        depth: str,
        depth_init: str,
        frozen: bool,
        mean: Sequence[float],
        std: Sequence[float],
        probe_size: Sequence[int],
        input_size: Sequence[int] | None,
        size_multiple: int,
    ) -> None:
        if depth not in ("rgb_only", "fourth_channel", "depth_stem"):
            raise ValueError(f"depth must be rgb_only, fourth_channel or depth_stem, got {depth!r}")
        if depth_init not in ("zero", "mean"):
            raise ValueError(f"depth_init must be zero or mean, got {depth_init!r}")
        self.depth = depth
        self.frozen = frozen
        if size_multiple < 1:
            raise ValueError(f"size_multiple must be at least 1, got {size_multiple}")
        self.input_size = None if input_size is None else (int(input_size[0]), int(input_size[1]))
        self.size_multiple = int(size_multiple)
        self.in_channels = 3 if depth == "rgb_only" else 4
        self.input_layout = "rgb" if depth == "rgb_only" else "rgbd"
        self.register_buffer("mean", torch.tensor(list(mean)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(list(std)).view(1, 3, 1, 1), persistent=False)
        self.model = model
        if frozen:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        # Added after freezing: what handles depth is always trained.
        if depth == "fourth_channel":
            _widen_image_convolutions(model, depth_init)
        self.depth_stem: nn.Module | None = None
        if depth == "depth_stem":
            self.depth_stem = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 3, 1),
            )
            nn.init.zeros_(self.depth_stem[-1].weight)
            nn.init.zeros_(self.depth_stem[-1].bias)
        height, width = (int(side) for side in probe_size)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            maps = self.forward(torch.zeros(1, self.in_channels, height, width))
        self.train(was_training)
        if not maps or any(m.ndim != 4 for m in maps):
            raise ContractError(f"{type(self).__name__}: the model returned no (B, C, h, w) maps")
        self.out_channels = tuple(int(m.shape[1]) for m in maps)
        self.out_strides = tuple(round(height / m.shape[2]) for m in maps)

    def train(self, mode: bool = True) -> _LibraryEncoder:
        super().train(mode)
        if mode and getattr(self, "frozen", False):
            self.model.eval()  # frozen batch-norm statistics stay frozen
        return self

    def _maps(self, x: Tensor) -> list[Tensor]:
        raise NotImplementedError

    def forward(self, image: Tensor) -> list[Tensor]:
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ContractError(
                f"{type(self).__name__} with depth: {self.depth} takes a "
                f"(B, {self.in_channels}, H, W) image ({self.input_layout}), got "
                f"{tuple(image.shape)}"
            )
        x = (image[:, :3] - self.mean) / self.std
        if self.depth == "fourth_channel":
            x = torch.cat([x, image[:, 3:]], dim=1)
        elif self.depth_stem is not None:
            x = x + self.depth_stem(image[:, 3:])
        height, width = x.shape[-2:]
        target = self.input_size or (
            height + -height % self.size_multiple,
            width + -width % self.size_multiple,
        )
        if target != (height, width):
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        return self._maps(x)


@IMAGE_ENCODERS.register()
class TimmImageEncoder(_LibraryEncoder):
    """Any timm model that returns feature maps (`features_only`), by its timm name.

    Covers the convolutional and hierarchical families (ResNet, ConvNeXt V2,
    EfficientNetV2, MobileNetV4, RepViT, FastViT, EfficientViT, Swin V2, MaxViT, Hiera …)
    and the plain ViTs (DINOv2, DINOv3, EVA-02, SigLIP 2 …), which return maps at their
    patch stride.

    Args:
        name: timm model name, with or without a pretrained tag
            (``convnextv2_tiny.fcmae_ft_in22k_in1k``). ``timm.list_models()`` lists them.
        pretrained: Load the released weights of the tag (downloaded by timm, or read
            from ``checkpoint``).
        checkpoint: Local weight file to use instead of a download.
        out_indices: Which feature levels to return; ``None`` is timm's default (all
            levels of a pyramid, the last blocks of a ViT).
        depth: ``rgb_only``, ``fourth_channel`` or ``depth_stem``; see the module docstring.
        depth_init: Initial depth filters of ``fourth_channel``: ``zero`` or ``mean``.
        frozen: Keep the library model fixed (no gradient, evaluation-mode statistics).
            What handles depth is trained all the same.
        model_args: Further keyword arguments of ``timm.create_model`` (``img_size`` for
            models with a fixed input size, ``drop_path_rate`` …).
        input_size: Resample every image to this ``(H, W)`` before the model (models with
            a fixed input size).
        size_multiple: Resample every image up to the next multiple of this (patch size,
            window size).
        probe_size: ``(H, W)`` of the forward pass that measures strides and widths. Use
            the network image size for models with a fixed input size.
    """

    status = "bridge"
    source = "PyTorch Image Models (the timm model zoo), via timm"
    extras = ("timm",)

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        checkpoint: str | None = None,
        out_indices: Sequence[int] | None = None,
        depth: Literal["rgb_only", "fourth_channel", "depth_stem"] = "rgb_only",
        depth_init: Literal["zero", "mean"] = "zero",
        frozen: bool = False,
        model_args: dict[str, Any] | None = None,
        input_size: Sequence[int] | None = None,
        size_multiple: int = 1,
        probe_size: Sequence[int] = (224, 224),
    ) -> None:
        super().__init__()
        timm = require("timm", extra="timm", purpose="TimmImageEncoder")
        arguments = dict(model_args or {})
        if out_indices is not None:
            arguments["out_indices"] = tuple(out_indices)
        if checkpoint is not None:
            arguments["pretrained_cfg_overlay"] = {"file": str(checkpoint)}
        try:
            model = timm.create_model(
                name,
                pretrained=pretrained or checkpoint is not None,
                features_only=True,
                **arguments,
            )
        except (RuntimeError, AttributeError, TypeError) as error:
            raise ContractError(
                f"TimmImageEncoder: timm cannot build '{name}' as a feature extractor "
                f"({error}). Check the name with `timm.list_models('*{name.split('.')[0][:8]}*')`; "
                "a few architectures have no `features_only` mode."
            ) from error
        self.name = name
        cfg = getattr(model, "pretrained_cfg", None) or {}
        if pretrained and checkpoint is None:
            _announce_weights("TimmImageEncoder", "timm", name, cfg.get("license"))
        fmt = getattr(model, "output_fmt", "NCHW")  # Swin and friends: channels last
        self._channels_last = str(getattr(fmt, "value", fmt)) == "NHWC"
        self._setup(
            model,
            depth=depth,
            depth_init=depth_init,
            frozen=frozen,
            mean=cfg.get("mean", _IMAGENET_MEAN),
            std=cfg.get("std", _IMAGENET_STD),
            probe_size=input_size or probe_size,
            input_size=input_size,
            size_multiple=size_multiple,
        )

    def _maps(self, x: Tensor) -> list[Tensor]:
        maps = self.model(x)
        if self._channels_last:
            maps = [m.permute(0, 3, 1, 2).contiguous() for m in maps]
        return list(maps)


@IMAGE_ENCODERS.register()
class HFImageEncoder(_LibraryEncoder):
    """Any Hugging Face vision model, through `AutoBackbone` or `AutoModel`, by its hub name.

    ``AutoBackbone`` models (DINOv2, DINOv3, Hiera, ConvNeXt V2, Swin V2, ResNet …) return
    feature maps themselves. Models without a backbone class (SigLIP 2, Depth Anything's
    encoder is a DINOv2 backbone and needs none) go through ``AutoModel``: the hidden
    states of the chosen layers are reshaped to maps, class and register tokens dropped.

    Args:
        name: Hub id (``facebook/dinov2-with-registers-small``) or a local directory
            with ``config.json`` (and weights).
        pretrained: Load the weights; ``False`` builds the architecture from its
            configuration alone.
        config: Build from this configuration instead of ``name`` (a mapping with
            ``model_type`` and the fields of that model's configuration class); random
            weights, nothing is downloaded.
        api: ``backbone`` (``AutoBackbone``), ``model`` (``AutoModel`` with hidden
            states) or ``depth_backbone`` (the backbone inside a depth-estimation model,
            ``AutoModelForDepthEstimation``: Depth Anything V2). ``radio`` is for
            NVIDIA's RADIO models, whose hub code returns ``(summary, spatial tokens)``
            and normalises the image itself (``model_args: {trust_remote_code: true}``,
            ``mean: [0, 0, 0]``, ``std: [1, 1, 1]``; needs ``open_clip_torch``).
        out_indices: Stages (``backbone``) or hidden states (``model``) to return;
            ``None`` is the last one.
        depth: ``rgb_only``, ``fourth_channel`` or ``depth_stem``; see the module docstring.
        depth_init: Initial depth filters of ``fourth_channel``: ``zero`` or ``mean``.
        frozen: Keep the library model fixed. What handles depth is trained all the same.
        mean, std: Colour normalisation; the ImageNet statistics unless given (SigLIP
            uses 0.5 and 0.5).
        model_args: Further keyword arguments of ``from_pretrained``.
        input_size: Resample every image to this ``(H, W)`` before the model (models with
            a fixed input size).
        size_multiple: Resample every image up to the next multiple of this (patch size,
            window size).
        probe_size: ``(H, W)`` of the forward pass that measures strides and widths.
        accept_remote_code: Must be ``True`` when ``model_args`` sets
            ``trust_remote_code``: the model code of the hub repository then runs on
            this machine, and the config has to say that this is wanted.
    """

    status = "bridge"
    source = "Hugging Face Transformers (AutoBackbone / AutoModel), via transformers"
    extras = ("transformers",)

    @staticmethod
    def _remote_code_message(name: Any) -> str:
        return (
            f"HFImageEncoder: `model_args.trust_remote_code: true` runs Python code "
            f"downloaded from the hub repository '{name}' on this machine. Read that "
            "repository, then state that you accept it with `accept_remote_code: true` "
            "(`--opts model.image_encoder.accept_remote_code=true`)."
        )

    @classmethod
    def check_config(cls, node: Mapping[str, Any]) -> Iterator[str]:
        """Model code from the hub runs only when the config accepts it in so many words."""
        arguments = node.get("model_args")
        asked = isinstance(arguments, Mapping) and arguments.get("trust_remote_code")
        if asked and not node.get("accept_remote_code", False):
            yield cls._remote_code_message(node.get("name"))

    def __init__(
        self,
        name: str | None = None,
        pretrained: bool = True,
        config: dict[str, Any] | None = None,
        api: Literal["backbone", "model", "depth_backbone", "radio"] = "backbone",
        out_indices: Sequence[int] | None = None,
        depth: Literal["rgb_only", "fourth_channel", "depth_stem"] = "rgb_only",
        depth_init: Literal["zero", "mean"] = "zero",
        frozen: bool = False,
        mean: Sequence[float] = _IMAGENET_MEAN,
        std: Sequence[float] = _IMAGENET_STD,
        model_args: dict[str, Any] | None = None,
        input_size: Sequence[int] | None = None,
        size_multiple: int = 1,
        probe_size: Sequence[int] = (224, 224),
        accept_remote_code: bool = False,
    ) -> None:
        super().__init__()
        transformers = require("transformers", extra="transformers", purpose="HFImageEncoder")
        if (name is None) == (config is None):
            raise ValueError("HFImageEncoder needs exactly one of `name` and `config`")
        loaders = {
            "backbone": "AutoBackbone",
            "model": "AutoModel",
            "depth_backbone": "AutoModelForDepthEstimation",
            "radio": "AutoModel",
        }
        if api not in loaders:
            raise ValueError(f"api must be one of {', '.join(loaders)}, got {api!r}")
        auto = getattr(transformers, loaders[api])
        arguments = dict(model_args or {})
        indices = None if out_indices is None else [int(i) for i in out_indices]
        if api == "backbone" and indices is not None:
            arguments["out_indices"] = indices
        if config is not None:
            fields = dict(config)
            built = transformers.AutoConfig.for_model(fields.pop("model_type"), **fields)
            if "out_indices" in arguments:
                built.out_indices, built.out_features = arguments.pop("out_indices"), None
            model = auto.from_config(built)
        elif pretrained:
            if arguments.get("trust_remote_code") and not accept_remote_code:
                raise ContractError(self._remote_code_message(name))
            if not os.path.isdir(str(name)):
                card = f"see https://huggingface.co/{name}"
                _announce_weights("HFImageEncoder", "transformers", str(name), card)
            model = auto.from_pretrained(name, **arguments)
        else:
            built = transformers.AutoConfig.from_pretrained(name)
            if "out_indices" in arguments:
                built.out_indices, built.out_features = arguments.pop("out_indices"), None
            model = auto.from_config(built)
        model = getattr(model, "vision_model", model)  # two-tower models: the image tower
        if api == "depth_backbone":
            model = model.backbone
        self.name = name
        self.api = api
        self._indices = indices or [-1]
        patch = getattr(model.config, "patch_size", 1)
        self._patch = int(patch[0] if isinstance(patch, (list, tuple)) else patch)
        self._setup(
            model,
            depth=depth,
            depth_init=depth_init,
            frozen=frozen,
            mean=mean,
            std=std,
            probe_size=input_size or probe_size,
            input_size=input_size,
            size_multiple=size_multiple,
        )

    def _maps(self, x: Tensor) -> list[Tensor]:
        height, width = x.shape[-2] // self._patch, x.shape[-1] // self._patch
        if self.api == "radio":
            return [self._grid(self.model(x)[1], height, width, x)]
        if self.api != "model":
            maps = self.model(pixel_values=x).feature_maps
            # A backbone configured not to reshape (Depth Anything's) returns tokens.
            return [m if m.ndim == 4 else self._grid(m, height, width, x) for m in maps]
        arguments = {"interpolate_pos_encoding": True} if self._interpolates() else {}
        states = self.model(pixel_values=x, output_hidden_states=True, **arguments).hidden_states
        maps = []
        for index in self._indices:
            tokens = states[index]
            if tokens.ndim == 4:  # hierarchical models: (B, h, w, C)
                maps.append(tokens.permute(0, 3, 1, 2).contiguous())
            else:
                maps.append(self._grid(tokens, height, width, x))
        return maps

    @staticmethod
    def _grid(tokens: Tensor, height: int, width: int, x: Tensor) -> Tensor:
        """``(B, prefix + h * w, C)`` tokens as a ``(B, C, h, w)`` map, prefix dropped."""
        prefix = tokens.shape[1] - height * width
        if prefix < 0:
            raise ContractError(
                f"HFImageEncoder: {tokens.shape[1]} tokens do not cover the "
                f"{height} x {width} patches of a {tuple(x.shape[-2:])} image; the model "
                "has a fixed input size or is not a plain ViT. Use api: backbone, or "
                "input_size."
            )
        grid = tokens[:, prefix:].transpose(1, 2)
        return grid.reshape(tokens.shape[0], -1, height, width)

    def _interpolates(self) -> bool:
        import inspect

        return "interpolate_pos_encoding" in inspect.signature(self.model.forward).parameters


@IMAGE_ENCODERS.register()
class TorchvisionImageEncoder(_LibraryEncoder):
    """A torchvision classification network as a pyramid encoder, by its torchvision name.

    Works for the families whose body is a chain of stages: the ResNets (ResNet,
    ResNeXt, Wide ResNet, RegNet) and everything with a ``features`` sequence (ConvNeXt,
    EfficientNet and V2, MobileNet V2 and V3, MNASNet, DenseNet, VGG). The classifier is
    dropped; one map is returned per resolution, the last one computed at it.

    Args:
        name: Name for ``torchvision.models.get_model`` (``resnet34``, ``convnext_tiny``).
        weights: Name of the weight enum entry (``DEFAULT``, ``IMAGENET1K_V1``), or
            ``None`` for random weights.
        checkpoint: Local state dict of the torchvision model to load instead.
        min_stride: Coarsest resolutions only: maps finer than this stride are not returned.
        depth: ``rgb_only``, ``fourth_channel`` or ``depth_stem``; see the module docstring.
        depth_init: Initial depth filters of ``fourth_channel``: ``zero`` or ``mean``.
        frozen: Keep the library model fixed. What handles depth is trained all the same.
        model_args: Further keyword arguments of ``get_model``.
        input_size: Resample every image to this ``(H, W)`` before the model (models with
            a fixed input size).
        size_multiple: Resample every image up to the next multiple of this (patch size,
            window size).
        probe_size: ``(H, W)`` of the forward pass that measures strides and widths.
    """

    status = "bridge"
    source = "The torchvision model zoo, via torchvision"
    extras = ("torchvision",)

    def __init__(
        self,
        name: str,
        weights: str | None = None,
        checkpoint: str | None = None,
        min_stride: int = 2,
        depth: Literal["rgb_only", "fourth_channel", "depth_stem"] = "rgb_only",
        depth_init: Literal["zero", "mean"] = "zero",
        frozen: bool = False,
        model_args: dict[str, Any] | None = None,
        input_size: Sequence[int] | None = None,
        size_multiple: int = 1,
        probe_size: Sequence[int] = (224, 224),
    ) -> None:
        super().__init__()
        models = require(
            "torchvision.models", extra="torchvision", purpose="TorchvisionImageEncoder"
        )
        if weights is not None and checkpoint is None:
            _announce_weights(
                "TorchvisionImageEncoder",
                "torchvision",
                f"{name} ({weights})",
                "see the weights table of the torchvision documentation",
            )
        net = models.get_model(name, weights=weights, **dict(model_args or {}))
        if checkpoint is not None:
            net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if hasattr(net, "layer1") and hasattr(net, "conv1"):
            stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            steps = [stem, net.layer1, net.layer2, net.layer3, net.layer4]
        elif hasattr(net, "stem") and hasattr(net, "trunk_output"):  # RegNet
            steps = [net.stem, *net.trunk_output.children()]
        elif isinstance(getattr(net, "features", None), nn.Sequential):
            steps = list(net.features.children())
        else:
            raise ContractError(
                f"TorchvisionImageEncoder cannot cut '{name}' into stages: it supports the "
                "ResNet and RegNet families and models with a `features` sequence. Use "
                "TimmImageEncoder for transformers (timm has the same architectures)."
            )
        self.name = name
        self.min_stride = min_stride
        self._emit: tuple[int, ...] | None = None
        model = nn.ModuleList(steps)
        self._setup(
            model,
            depth=depth,
            depth_init=depth_init,
            frozen=frozen,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            probe_size=input_size or probe_size,
            input_size=input_size,
            size_multiple=size_multiple,
        )

    def _maps(self, x: Tensor) -> list[Tensor]:
        height = x.shape[-2]
        if self._emit is None:  # the measuring pass: the last step at every resolution
            outputs = []
            for step in self.model:
                x = step(x)
                outputs.append(x)
            if any(output.ndim != 4 for output in outputs):
                raise ContractError("TorchvisionImageEncoder: a stage returns no (B, C, h, w) map")
            last: dict[int, int] = {}
            for index, output in enumerate(outputs):
                last[output.shape[-2]] = index
            self._emit = tuple(
                index
                for size, index in sorted(last.items(), reverse=True)
                if round(height / size) >= self.min_stride
            )
            return [outputs[index] for index in self._emit]
        maps = []
        for index, step in enumerate(self.model):
            x = step(x)
            if index in self._emit:
                maps.append(x)
        return maps
