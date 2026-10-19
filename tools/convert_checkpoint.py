"""Convert an upstream checkpoint into a GraspPanda state dict.

``--source-format`` names the implementation that wrote the file; it is required, because
two methods release a file of the same name (``checkpoint-rs.tar``):

``graspnet-baseline``
    The GraspNet of ``graspnet/graspnet-baseline`` (PointNet++ backbone, e.g. the authors'
    ``checkpoint-rs.tar``), converted for ``configs/_base_/models/graspnet_baseline.yaml``.
``graspness-unofficial``
    GSNet: the GraspNet of ``graspnet/graspness_unofficial`` (MinkowskiEngine backbone,
    e.g. the authors' ``minkuresunet_realsense.tar``), converted for
    ``configs/_base_/models/gsnet.yaml``.
``scale-balanced-grasp``
    The ``GraspNet_MSCQ`` of ``mahaoxiang822/Scale-Balanced-Grasp`` (the baseline layout
    with attention blocks and one crop module per radius, e.g. the authors'
    ``checkpoint-rs.tar``), converted for ``configs/_base_/models/scale_balanced_grasp.yaml``.
``economicgrasp``
    The network of ``iSEE-Laboratory/EconomicGrasp`` (the GSNet backbone, e.g. the authors'
    ``economicgrasp_kinect.tar``), converted for
    ``configs/_base_/models/economicgrasp.yaml``.
``finegrasp``
    The ``FineGrasp`` of ``HorizonRobotics/robo_orchard_lab`` (the EconomicGrasp layout
    with one crop module per radius and a transformer over the scales; the authors'
    ``finegrasp_pipeline/model.safetensors``), converted for
    ``configs/_base_/models/finegrasp.yaml``.
``hggd``
    The two networks and the learned rotation anchors of ``THU-VCLab/HGGD`` (e.g. the
    authors' ``HGGD_realsense_checkpoint``: keys ``anchor``, ``local``, ``gamma``,
    ``beta``), converted for ``configs/_base_/models/hggd.yaml``. The optimizer state in
    the file is not a model tensor and is left behind.

The commands of the model zoo (``convert`` in ``configs/<method>/metafile.yaml``)::

    python -m tools.convert_checkpoint checkpoint-rs.tar baseline_realsense.pth \
        --source-format graspnet-baseline --camera realsense
    python -m tools.convert_checkpoint minkuresunet_realsense.tar gsnet_realsense.pth \
        --source-format graspness-unofficial --camera realsense
    python -m tools.convert_checkpoint checkpoint-rs.tar sbg_realsense.pth \
        --source-format scale-balanced-grasp --camera realsense
    python -m tools.convert_checkpoint economicgrasp_realsense.tar economicgrasp_realsense.pth \
        --source-format economicgrasp --camera realsense
    python -m tools.convert_checkpoint finegrasp_pipeline/model.safetensors \
        finegrasp_realsense.pth --source-format finegrasp --camera realsense
    python -m tools.convert_checkpoint HGGD_realsense_checkpoint hggd_realsense.pth \
        --source-format hggd --camera realsense

What the conversion does, and nothing else:

* renames every tensor to the module that plays the same role here;
* re-lays MinkowskiEngine kernels in the sparse facade's layout
  (:func:`grasppanda.ops.sparse.convert_minkowski_kernel`);
* turns 1×1 convolutions (sparse, ``Conv1d``, ``Conv2d``) into the weights of the linear
  layers that replace them;
* cuts the stacked query / key / value projection of an ``nn.MultiheadAttention`` into
  the three linear layers that replace it.

No value is changed. Every source tensor must be consumed and every tensor of the
target model must be filled with the right shape, otherwise the conversion fails with a
report; nothing is ever loaded with ``strict=False``.

A converted file is accepted as equivalent only through the stage-by-stage comparison
with activations recorded from the upstream implementation
(``tests/integration/test_<method>_parity.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from grasppanda.ops.sparse import convert_minkowski_kernel

__all__ = [
    "FORMAT_METHODS",
    "SOURCE_FORMATS",
    "ConversionError",
    "build_parser",
    "convert_economicgrasp_reference",
    "convert_finegrasp_reference",
    "convert_graspness_reference",
    "convert_graspnet_baseline",
    "convert_hggd_reference",
    "convert_scale_balanced_grasp",
    "load_safetensors",
    "main",
    "verify_against",
]

#: Edge length of a cubic MinkowskiEngine kernel by its number of cells.
_KERNEL_SIZE = {125: 5, 27: 3, 8: 2}

#: Reference backbone layer → SparseUNet14 submodule.
_BACKBONE = {
    "conv0p1s1": "stem.conv",
    "bn0": "stem.norm",
    "final": "project",
}
for _stage, _stride in enumerate((1, 2, 4, 8), start=1):  # encoder: conv{n}p{stride}s2, bn{n}
    _BACKBONE[f"conv{_stage}p{_stride}s2"] = f"down.{_stage - 1}.conv"
    _BACKBONE[f"bn{_stage}"] = f"down.{_stage - 1}.norm"
    _BACKBONE[f"block{_stage}.0"] = f"encoder.{_stage - 1}"
for _stage, _stride in enumerate((16, 8, 4, 2)):  # decoder: convtr{n}p{stride}s2, bntr{n}
    _BACKBONE[f"convtr{_stage + 4}p{_stride}s2"] = f"up.{_stage}.conv"
    _BACKBONE[f"bntr{_stage + 4}"] = f"up.{_stage}.norm"
    _BACKBONE[f"block{_stage + 5}.0"] = f"decoder.{_stage}"

#: Residual block member → _BasicBlock member.
_BLOCK = {
    "conv1": "conv1",
    "norm1": "norm1",
    "conv2": "conv2",
    "norm2": "norm2",
    "downsample.0": "shortcut.conv",
    "downsample.1": "shortcut.norm",
}

#: Reference head layer → GSNet submodule.
_HEADS = {
    "graspable.conv_graspable": "proposal.score",
    "rotation.conv1": "view.hidden",
    "rotation.conv2": "view.score",
    "swad.conv1": "head.hidden",
    "swad.conv_swad": "head.output",
}

_BATCH_NORM = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


class ConversionError(RuntimeError):
    """The source checkpoint does not have the expected structure."""


def _backbone_key(key: str) -> tuple[str, str]:
    """Split ``<layer>.<tensor>`` of the reference backbone into (our module, tensor name)."""
    for layer in sorted(_BACKBONE, key=len, reverse=True):
        if key.startswith(layer + "."):
            rest = key[len(layer) + 1 :]
            if layer.startswith("block"):
                for member, ours in _BLOCK.items():
                    if rest.startswith(member + "."):
                        return f"{_BACKBONE[layer]}.{ours}", rest[len(member) + 1 :]
                break
            return _BACKBONE[layer], rest
    raise ConversionError(f"unexpected backbone tensor 'backbone.{key}'")


def _translate(key: str, value: Tensor) -> tuple[str, Tensor]:
    if key.startswith("backbone."):
        module, tensor = _backbone_key(key[len("backbone.") :])
        if tensor == "kernel":
            size = 1 if value.ndim == 2 else _KERNEL_SIZE.get(value.shape[0])
            if size is None:
                raise ConversionError(f"'{key}' has an unsupported kernel volume {value.shape[0]}")
            return f"backbone.{module}.weight", convert_minkowski_kernel(value, size)
        if tensor == "bias":  # MinkowskiEngine keeps a (1, C) bias
            return f"backbone.{module}.bias", value.reshape(-1).clone()
        if tensor.startswith("bn.") and tensor[3:] in _BATCH_NORM:
            return f"backbone.{module}.{tensor[3:]}", value.clone()
        raise ConversionError(f"unexpected backbone tensor '{key}'")
    for layer, module in _HEADS.items():
        if key in (f"{layer}.weight", f"{layer}.bias"):
            tensor = key.rsplit(".", 1)[1]
            # Conv1d weights are (out, in, 1); the linear layer takes (out, in).
            return f"{module}.{tensor}", value.flatten(
                1
            ).clone() if tensor == "weight" else value.clone()
    match = re.fullmatch(r"crop\.mlps\.layer(\d+)\.(conv\.weight|bn\.bn\.(\w+))", key)
    if match:
        layer = f"grouper.mlp.{match.group(1)}"
        if match.group(2) == "conv.weight":  # Conv2d (out, in, 1, 1)
            return f"{layer}.linear.weight", value.flatten(1).clone()
        if match.group(3) in _BATCH_NORM:
            return f"{layer}.norm.{match.group(3)}", value.clone()
    raise ConversionError(f"unexpected tensor '{key}' in a graspness_unofficial checkpoint")


def convert_graspness_reference(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map the state dict of the reference ``GraspNet`` onto GraspPanda's ``GSNet``.

    Raises :class:`ConversionError` on any tensor that does not belong to that network.
    """
    converted: dict[str, Tensor] = {}
    for key, value in source.items():
        ours, tensor = _translate(key, value)
        if ours in converted:
            raise ConversionError(f"'{key}' maps to '{ours}', which was already filled")
        converted[ours] = tensor
    return converted


#: EconomicGrasp layer (a 1×1 convolution or a linear layer) → EconomicGrasp submodule here.
_ECONOMIC_LAYERS = {
    "view.conv1": "view.hidden",
    "view.conv2": "view.score",
    "cy_group.local_interaction_module.norm1": "grouper.interaction.norm",
    "grasp_head.global_interaction_module.norm1": "head.interaction.norm",
}
#: The four projections of the reference attention block, in its order.
for _slot, _role in enumerate(("query", "key", "value", "output")):
    _ECONOMIC_LAYERS[f"cy_group.local_interaction_module.msa.linears.{_slot}"] = (
        f"grouper.interaction.{_role}"
    )
    _ECONOMIC_LAYERS[f"grasp_head.global_interaction_module.msa.linears.{_slot}"] = (
        f"head.interaction.{_role}"
    )
for _token in ("angle", "depth", "width", "score"):
    _ECONOMIC_LAYERS[f"grasp_head.conv_{_token}_feature"] = f"head.project.{_token}"
    _ECONOMIC_LAYERS[f"grasp_head.conv_{_token}"] = f"head.predict.{_token}"

#: Shared MLP of the reference crop → shared MLP of CylinderInteractionGroup.
_ECONOMIC_MLPS = {"mlps": "grouper.mlp", "mlps2": "grouper.post_mlp"}


def _translate_economicgrasp(key: str, value: Tensor) -> tuple[str, Tensor]:
    if key.startswith(("backbone.", "graspable.")):  # the same layers as the Graspness network
        return _translate(key, value)
    layer, _, tensor = key.rpartition(".")
    if layer in _ECONOMIC_LAYERS and tensor in ("weight", "bias"):
        # Conv1d weights are (out, in, 1); linear and layer-norm tensors pass unchanged.
        flat = value.flatten(1) if tensor == "weight" and value.ndim == 3 else value
        return f"{_ECONOMIC_LAYERS[layer]}.{tensor}", flat.clone()
    match = re.fullmatch(r"cy_group\.(mlps2?)\.layer(\d+)\.(conv\.weight|bn\.bn\.(\w+))", key)
    if match:
        layer = f"{_ECONOMIC_MLPS[match.group(1)]}.{match.group(2)}"
        if match.group(3) == "conv.weight":  # Conv2d (out, in, 1, 1)
            return f"{layer}.linear.weight", value.flatten(1).clone()
        if match.group(4) in _BATCH_NORM:
            return f"{layer}.norm.{match.group(4)}", value.clone()
    raise ConversionError(f"unexpected tensor '{key}' in an EconomicGrasp checkpoint")


def convert_economicgrasp_reference(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map the state dict of the reference ``economicgrasp`` onto GraspPanda's ``EconomicGrasp``.

    Raises :class:`ConversionError` on any tensor that does not belong to that network.
    """
    converted: dict[str, Tensor] = {}
    for key, value in source.items():
        try:
            ours, tensor = _translate_economicgrasp(key, value)
        except ConversionError as error:
            raise ConversionError(
                str(error).replace("a graspness_unofficial", "an EconomicGrasp")
            ) from None
        if ours in converted:
            raise ConversionError(f"'{key}' maps to '{ours}', which was already filled")
        converted[ours] = tensor
    return converted


# ----------------------------------------------------------------------- FineGrasp

#: Layer of the reference's ``nn.TransformerEncoderLayer`` → member of ``_EncoderLayer``.
_FINEGRASP_ENCODER = {
    "self_attn.out_proj": "interaction.output",
    "norm1": "interaction.norm",
    "linear1": "expand",
    "linear2": "contract",
    "norm2": "norm",
}


def _translate_finegrasp(key: str, value: Tensor) -> dict[str, Tensor]:
    """The tensor(s) of the FineGrasp model here that one reference tensor fills."""
    match = re.fullmatch(r"cy_groups\.(\d+)\.(.+)", key)
    if match:  # one EconomicGrasp crop module per radius, smallest first
        ours, tensor = _translate_economicgrasp(f"cy_group.{match.group(2)}", value)
        return {ours.replace("grouper.", f"grouper.branches.{match.group(1)}.", 1): tensor}
    if key in ("fuse_multi_scale.attn.weight", "fuse_multi_scale.attn.bias"):
        return {f"grouper.fusion.score.{key.rsplit('.', 1)[1]}": value.clone()}
    match = re.fullmatch(r"fuse_multi_scale\.transformer\.layers\.(\d+)\.(.+)\.(\w+)", key)
    if match:
        layer = f"grouper.fusion.layers.{match.group(1)}"
        member, tensor = match.group(2), match.group(3)
        if member in _FINEGRASP_ENCODER and tensor in ("weight", "bias"):
            return {f"{layer}.{_FINEGRASP_ENCODER[member]}.{tensor}": value.clone()}
        if member == "self_attn" and tensor in ("in_proj_weight", "in_proj_bias"):
            # nn.MultiheadAttention stacks the query, key and value projections.
            if value.shape[0] % 3:
                raise ConversionError(f"'{key}' is not three stacked projections: {value.shape}")
            name = tensor.removeprefix("in_proj_")
            return {
                f"{layer}.interaction.{role}.{name}": part.clone()
                for role, part in zip(("query", "key", "value"), value.chunk(3, dim=0))
            }
    if key.startswith(("backbone.", "graspable.", "view.", "grasp_head.")):
        ours, tensor = _translate_economicgrasp(key, value)  # the same layers as EconomicGrasp
        return {ours: tensor}
    raise ConversionError(f"unexpected tensor '{key}' in a FineGrasp checkpoint")


def convert_finegrasp_reference(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map the state dict of the reference ``FineGrasp`` onto GraspPanda's FineGrasp model
    (``EconomicGrasp`` with a ``MultiScaleGroup``).

    Raises :class:`ConversionError` on any tensor that does not belong to that network.
    """
    converted: dict[str, Tensor] = {}
    for key, value in source.items():
        try:
            filled = _translate_finegrasp(key, value)
        except ConversionError as error:
            message = str(error)
            for other in ("a graspness_unofficial", "an EconomicGrasp"):
                message = message.replace(f"in {other} checkpoint", "in a FineGrasp checkpoint")
            raise ConversionError(message) from None
        for ours, tensor in filled.items():
            if ours in converted:
                raise ConversionError(f"'{key}' maps to '{ours}', which was already filled")
            converted[ours] = tensor
    return converted


#: safetensors dtype tag → NumPy dtype (the format stores little-endian data).
_SAFETENSORS_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "?",
}


def load_safetensors(path: Path) -> dict[str, Tensor]:
    """Read a ``.safetensors`` file: an 8-byte header length, a JSON header that gives
    dtype, shape and byte range of every tensor, then the raw little-endian data.

    Written out here so that converting such a checkpoint needs no extra dependency.
    """
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
        data = handle.read()
    header.pop("__metadata__", None)
    tensors = {}
    for name, entry in header.items():
        if entry["dtype"] not in _SAFETENSORS_DTYPES:
            raise ConversionError(f"'{name}' has the unsupported dtype {entry['dtype']}")
        begin, end = entry["data_offsets"]
        array = np.frombuffer(data[begin:end], dtype=_SAFETENSORS_DTYPES[entry["dtype"]])
        tensors[name] = torch.from_numpy(array.reshape(entry["shape"]).copy())
    return tensors


# ---------------------------------------------------------------------------- HGGD

#: Map head of the reference image network → ImageNet submodule.
_HGGD_MAPS = {
    "hmap": "heatmap",
    "cls_mask_conv": "angle",
    "theta_offset_conv": "angle_offset",
    "width_offset_conv": "width_offset",
    "depth_offset_conv": "depth_offset",
}
#: Member of a reference residual block (an ``nn.Sequential`` by position) → ours.
_HGGD_BLOCK = {
    "residual_function.0": "conv1",
    "residual_function.1": "norm1",
    "residual_function.3": "conv2",
    "residual_function.4": "norm2",
    "shortcut.0": "shortcut.0",
    "shortcut.1": "shortcut.1",
}
#: Layer of the reference region network → LocalNet submodule. Layers marked ``True``
#: are 1×1 ``Conv1d`` whose ``(out, in, 1)`` weight becomes a linear ``(out, in)`` one.
_HGGD_LOCAL = {
    "pointnet.stn.conv1": ("align.points.0.linear", True),
    "pointnet.stn.conv2": ("align.points.1.linear", True),
    "pointnet.stn.conv3": ("align.points.2.linear", True),
    "pointnet.stn.bn1": ("align.points.0.norm", False),
    "pointnet.stn.bn2": ("align.points.1.norm", False),
    "pointnet.stn.bn3": ("align.points.2.norm", False),
    "pointnet.stn.fc1": ("align.vector.0.linear", False),
    "pointnet.stn.fc2": ("align.vector.1.linear", False),
    "pointnet.stn.bn4": ("align.vector.0.norm", False),
    "pointnet.stn.bn5": ("align.vector.1.norm", False),
    "pointnet.stn.fc3": ("align.out", False),
    "pointnet.conv1": ("embed.linear", True),
    "pointnet.bn1": ("embed.norm", False),
    "pointnet.conv2": ("fuse.linear", True),
    "pointnet.bn2": ("fuse.norm", False),
    "pointnet.conv3": ("lift.linear", True),
    "pointnet.bn3": ("lift.norm", False),
    "point_layer.0": ("reduce", False),
    "point_layer.1": ("reduce_norm", False),
    "info_layer": ("info", False),
    "anchor_mlp.0": ("anchor.hidden", False),
    "anchor_mlp.1": ("anchor.norm", False),
    "anchor_mlp.3": ("anchor.out", False),
    "offset_mlp.0": ("offset.hidden", False),
    "offset_mlp.1": ("offset.norm", False),
    "offset_mlp.3": ("offset.out", False),
}


def _hggd_image_key(key: str) -> str:
    """Name of a tensor of the reference ``AnchorGraspNet`` inside our ``ImageNet``."""
    match = re.fullmatch(r"backbone\.net\.conv1\.([01])\.(\w+)", key)
    if match:
        return f"stem.{match.group(1)}.{match.group(2)}"
    match = re.fullmatch(r"backbone\.net\.conv([2-5])_x\.(\d+)\.(.+)\.(\w+)", key)
    if match and match.group(3) in _HGGD_BLOCK:
        stage, block = int(match.group(1)) - 2, match.group(2)
        return f"stages.{stage}.{block}.{_HGGD_BLOCK[match.group(3)]}.{match.group(4)}"
    match = re.fullmatch(r"trconv\.(\d)\.deconv\.([01])\.(\w+)", key)
    if match:
        return f"ups.{match.group(1)}.{('conv', 'norm')[int(match.group(2))]}.{match.group(3)}"
    match = re.fullmatch(r"(\w+)\.0\.conv\.([01])\.(\w+)", key)
    if match and match.group(1) in _HGGD_MAPS:
        member = ("conv", "norm")[int(match.group(2))]
        return f"{_HGGD_MAPS[match.group(1)]}.{member}.{match.group(3)}"
    match = re.fullmatch(r"(\w+)\.1\.(weight|bias)", key)
    if match and match.group(1) in _HGGD_MAPS:
        return f"{_HGGD_MAPS[match.group(1)]}.out.{match.group(2)}"
    raise ConversionError(f"unexpected tensor 'anchor.{key}' in an HGGD checkpoint")


def _hggd_local_key(key: str, value: Tensor) -> tuple[str, Tensor]:
    layer, _, tensor = key.rpartition(".")
    if layer not in _HGGD_LOCAL:
        raise ConversionError(f"unexpected tensor 'local.{key}' in an HGGD checkpoint")
    ours, pointwise = _HGGD_LOCAL[layer]
    if pointwise and tensor == "weight":
        if value.ndim != 3 or value.shape[-1] != 1:
            raise ConversionError(
                f"'local.{key}' should be a 1x1 convolution, got {tuple(value.shape)}"
            )
        return f"{ours}.weight", value.flatten(1).clone()
    return f"{ours}.{tensor}", value.clone()


def convert_hggd_reference(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    """Map an HGGD checkpoint onto GraspPanda's ``HGGD``.

    ``checkpoint`` holds the state dicts ``anchor`` (image network) and ``local``
    (region network) and the anchor positions ``gamma`` and ``beta``. Raises
    :class:`ConversionError` when a part is missing or a tensor is not known.
    """
    absent = [name for name in ("anchor", "local", "gamma", "beta") if name not in checkpoint]
    if absent:
        raise ConversionError(
            f"not an HGGD checkpoint: {absent} missing (found {sorted(checkpoint)[:8]})"
        )
    converted: dict[str, Tensor] = {}

    def put(name: str, tensor: Tensor, source: str) -> None:
        if name in converted:
            raise ConversionError(f"'{source}' maps to '{name}', which was already filled")
        converted[name] = tensor

    for key, value in checkpoint["anchor"].items():
        put(f"image_net.{_hggd_image_key(key)}", value.clone(), f"anchor.{key}")
    for key, value in checkpoint["local"].items():
        ours, tensor = _hggd_local_key(key, value)
        put(f"local_net.{ours}", tensor, f"local.{key}")
    for name in ("gamma", "beta"):
        anchors = torch.as_tensor(checkpoint[name])
        if anchors.ndim != 1:
            raise ConversionError(f"'{name}' should be a vector of anchors, got {anchors.shape}")
        put(f"anchors.{name}", anchors.detach().cpu().double().clone(), name)
    return converted


# ------------------------------------------------------------------ graspnet-baseline

#: Layers of the reference's small heads → members of the modules that replace them.
_BASELINE_LAYERS = {
    "conv1": "hidden1",
    "conv2": "hidden2",
    "bn1": "norm1",
    "bn2": "norm2",
}
#: Reference module → (GraspPanda module, name of its output layer).
_BASELINE_HEADS = {
    "view_estimator.vpmodule": ("view", "output"),
    "grasp_generator.operation": ("head.operation", "output"),
    "grasp_generator.tolerance": ("head.tolerance", "output"),
}


#: Members of the reference's attention block → members of ``_VectorAttention``.
_ATTENTION = {
    "fc1": "embed",
    "fc2": "project",
    "fc_delta.0": "position.0",
    "fc_delta.2": "position.2",
    "fc_gamma.0": "attention.0",
    "fc_gamma.2": "attention.2",
    "w_qs": "query",
    "w_ks": "key",
    "w_vs": "value",
}


def _translate_baseline(
    key: str, value: Tensor, family: str = "graspnet-baseline"
) -> tuple[str, Tensor]:
    """Rename one tensor of a network of the GraspNet-baseline family.

    ``family`` selects the parts in which Scale-Balanced-Grasp differs: attention blocks
    in the backbone, one crop module per radius, a fusion layer and a gate.
    """
    backbone = "view_estimator.backbone."
    multi_scale = family == "scale-balanced-grasp"
    if key.startswith(backbone):
        rest = key[len(backbone) :]
        # The set abstraction keeps the parameter names of its lineage; see PointNet2.
        if re.fullmatch(
            r"(sa\d+\.mlp_module|fp\d+\.mlp)\.layer\d+\.(conv\.weight|bn\.bn\.\w+)", rest
        ):
            return f"backbone.{rest}", value.clone()
        match = re.fullmatch(r"(transformer\d+)\.([\w.]+)\.(weight|bias)", rest)
        if multi_scale and match and match.group(2) in _ATTENTION:
            ours = f"backbone.{match.group(1)}.{_ATTENTION[match.group(2)]}.{match.group(3)}"
            return ours, value.clone()
        raise ConversionError(f"unexpected backbone tensor '{key}'")
    crop = r"crop(\d)" if multi_scale else r"crop()"
    match = re.fullmatch(
        rf"grasp_generator\.{crop}\.mlps\.layer(\d+)\.(conv\.weight|bn\.bn\.(\w+))", key
    )
    if match:
        scale = int(match.group(1)) - 1 if multi_scale else 0  # crop1 is the smallest radius
        layer = f"grouper.scales.{scale}.{match.group(2)}"
        if match.group(3) == "conv.weight":  # Conv2d (out, in, 1, 1)
            return f"{layer}.linear.weight", value.flatten(1).clone()
        if match.group(4) in _BATCH_NORM:
            return f"{layer}.norm.{match.group(4)}", value.clone()
    if multi_scale:
        for name, ours in (("fuse_multi_scale", "grouper.fuse"), ("gate_fusion.0", "grouper.gate")):
            if key == f"grasp_generator.{name}.weight":  # Conv1d (out, in, 1)
                return f"{ours}.weight", value.flatten(1).clone()
            if key == f"grasp_generator.{name}.bias":
                return f"{ours}.bias", value.clone()
    for prefix, (module, output) in _BASELINE_HEADS.items():
        if not key.startswith(prefix + "."):
            continue
        layer, _, tensor = key[len(prefix) + 1 :].partition(".")
        ours = output if layer == "conv3" else _BASELINE_LAYERS.get(layer)
        if ours is None:
            break
        if layer.startswith("conv") and tensor in ("weight", "bias"):
            # Conv1d weights are (out, in, 1); the linear layer takes (out, in).
            return f"{module}.{ours}.{tensor}", (
                value.flatten(1).clone() if tensor == "weight" else value.clone()
            )
        if layer.startswith("bn") and tensor in _BATCH_NORM:
            return f"{module}.{ours}.{tensor}", value.clone()
        break
    raise ConversionError(f"unexpected tensor '{key}' in a {family} checkpoint")


def convert_graspnet_baseline(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map the state dict of the reference ``GraspNet`` onto GraspPanda's ``GraspNetBaseline``.

    Every reference tensor has exactly one counterpart: the layout is the same, 1×1
    convolutions become linear layers. Raises :class:`ConversionError` on any tensor
    that does not belong to that network.
    """
    converted: dict[str, Tensor] = {}
    for key, value in source.items():
        ours, tensor = _translate_baseline(key, value)
        if ours in converted:
            raise ConversionError(f"'{key}' maps to '{ours}', which was already filled")
        converted[ours] = tensor
    return converted


def convert_scale_balanced_grasp(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map the state dict of the reference ``GraspNet_MSCQ`` onto ``ScaleBalancedGrasp``.

    The network is the GraspNet-baseline layout with attention blocks in the backbone,
    one crop module per radius (``crop1`` is the smallest), a fusion layer and a gate.
    Raises :class:`ConversionError` on any tensor that does not belong to that network.
    """
    converted: dict[str, Tensor] = {}
    for key, value in source.items():
        ours, tensor = _translate_baseline(key, value, "scale-balanced-grasp")
        if ours in converted:
            raise ConversionError(f"'{key}' maps to '{ours}', which was already filled")
        converted[ours] = tensor
    return converted


def _model_state(convert):
    """Adapt a state-dict converter to a loaded training checkpoint."""
    return lambda checkpoint: convert(checkpoint.get("model_state_dict", checkpoint))


#: Source format → (converter of the loaded file, config the weights are converted for), in
#: the order of the methods (GraspNet-baseline, GSNet, Scale-Balanced-Grasp, EconomicGrasp,
#: FineGrasp, HGGD).
SOURCE_FORMATS = {
    "graspnet-baseline": (
        _model_state(convert_graspnet_baseline),
        "_base_/models/graspnet_baseline.yaml",
    ),
    "graspness-unofficial": (_model_state(convert_graspness_reference), "_base_/models/gsnet.yaml"),
    "scale-balanced-grasp": (
        _model_state(convert_scale_balanced_grasp),
        "_base_/models/scale_balanced_grasp.yaml",
    ),
    "economicgrasp": (
        _model_state(convert_economicgrasp_reference),
        "_base_/models/economicgrasp.yaml",
    ),
    "finegrasp": (_model_state(convert_finegrasp_reference), "_base_/models/finegrasp.yaml"),
    "hggd": (convert_hggd_reference, "_base_/models/hggd.yaml"),
}


#: Source format → the method whose released weights it reads.
FORMAT_METHODS = {
    "graspnet-baseline": "GraspNet-baseline",
    "graspness-unofficial": "GSNet",
    "scale-balanced-grasp": "Scale-Balanced-Grasp",
    "economicgrasp": "EconomicGrasp",
    "finegrasp": "FineGrasp",
    "hggd": "HGGD",
}


def verify_against(model: torch.nn.Module, converted: Mapping[str, Tensor]) -> None:
    """Raise unless ``converted`` fills exactly the tensors of ``model`` with their shapes."""
    target = model.state_dict()
    missing = sorted(target.keys() - converted.keys())
    unexpected = sorted(converted.keys() - target.keys())
    wrong = sorted(
        f"{key}: converted {tuple(converted[key].shape)}, model {tuple(target[key].shape)}"
        for key in target.keys() & converted.keys()
        if converted[key].shape != target[key].shape
    )
    if missing or unexpected or wrong:
        lines = [f"the converted state dict does not fit {type(model).__name__}:"]
        for title, items in (
            ("not filled", missing),
            ("no such tensor in the model", unexpected),
            ("shape mismatch", wrong),
        ):
            if items:
                shown = ", ".join(items[:8]) + (
                    f" … ({len(items) - 8} more)" if len(items) > 8 else ""
                )
                lines.append(f"  {title} ({len(items)}): {shown}")
        raise ConversionError("\n".join(lines))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_file(
    source: Path,
    config: str | None = None,
    *,
    source_format: str,
    camera: str | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Convert a checkpoint file; returns the state dict and its provenance record.

    ``source_format`` (a key of :data:`SOURCE_FORMATS`) has no default: two methods release
    a file of the same name. ``config`` defaults to the reference model config of
    ``source_format``.
    """
    from grasppanda.config import Config
    from grasppanda.registry import DETECTORS

    if source_format not in SOURCE_FORMATS:
        raise ConversionError(
            f"unknown source format '{source_format}'; supported: {', '.join(SOURCE_FORMATS)}"
        )
    convert, default_config = SOURCE_FORMATS[source_format]
    config = config or default_config
    if source.suffix == ".safetensors":
        checkpoint: Any = load_safetensors(source)
    else:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    converted = convert(checkpoint)
    resolved = Config.load(config)
    verify_against(DETECTORS.build(resolved["model"]), converted)
    meta = {
        "source_format": source_format,
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "model_config": str(config),
        "model": resolved["model"],
        "camera": camera,
        "tensors": len(converted),
        "parameters": sum(
            t.numel() for k, t in converted.items() if t.is_floating_point() and "running_" not in k
        ),
    }
    return converted, meta


def build_parser() -> argparse.ArgumentParser:
    """The command line of the converter (also used to check the commands the docs print)."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert_checkpoint", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "source", type=Path, help="upstream checkpoint (.tar / .pth / .safetensors)"
    )
    parser.add_argument("destination", type=Path, help="where to write the converted file")
    parser.add_argument(
        "--source-format",
        choices=list(SOURCE_FORMATS),
        required=True,
        help="which upstream implementation produced the checkpoint: "
        + ", ".join(f"{name} ({method})" for name, method in FORMAT_METHODS.items()),
    )
    parser.add_argument(
        "--config",
        help="config (file or shipped name) of the model the weights are converted for "
        "(default: the reference model of the source format)",
    )
    parser.add_argument(
        "--camera", help="camera the weights were trained for, recorded as metadata"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        converted, meta = convert_file(
            args.source, args.config, camera=args.camera, source_format=args.source_format
        )
    except ConversionError as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 1
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": converted, "meta": meta}, args.destination)
    print(
        f"wrote {args.destination}: {meta['tensors']} tensors, {meta['parameters']:,} parameters, "
        f"source sha256 {meta['source_sha256'][:12]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
