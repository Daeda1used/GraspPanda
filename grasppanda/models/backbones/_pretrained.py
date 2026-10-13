"""Start a scene encoder from released weights: the ``pretrained`` option of the encoders.

Several encoder classes of the toolbox keep the module tree of the code base they come
from, so that a published checkpoint loads into them key for key. This module is the one
way to do that from a config::

    model:
      backbone:
        type: SpUNet
        pretrained: gaussiancross-scannet-lin     # a name of RELEASED, or a path
        pretrained_mismatch: skip                 # the stem differs: 3 input channels here

``pretrained`` is a file path (``~`` and ``$VARIABLES`` are expanded) or one of the names
in :data:`RELEASED`, which stands for ``$GRASPPANDA_CACHE/weights_components/<file>``.
Nothing is downloaded: when the file is absent the error prints the command that
fetches it. The file is read without executing pickled code.

What is loaded is the encoder alone. The loader finds the prefix under which the file
stores it (``backbone.``, ``module.backbone.``, ...), ignores what lies outside (class
heads, projection heads of a pre-training method) and reports every tensor of the
encoder: loaded, absent from the file, of another shape, or in the file but not in the
model. ``pretrained_mismatch`` decides about the middle two: ``error`` (the default)
refuses, ``skip`` leaves those tensors at their initialisation and says which. A grasp
model that feeds three constant channels into a network published for colour and normal
has exactly one such tensor, the stem kernel; ``skip`` is how its presets start.

A grasp checkpoint trained from such a model holds the encoder like any other. Loading
it (``--weights``, ``runtime.load_from``, a resumed run) still builds the encoder first,
so the released file must stay where it is, or the config must set ``pretrained: null``.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from grasppanda.utils.paths import cache_root

__all__ = [
    "RELEASED",
    "PretrainedError",
    "PretrainedReport",
    "ReleasedWeights",
    "load_pretrained",
    "read_state",
    "resolve_pretrained",
    "select_condition",
]

_LOG = logging.getLogger("grasppanda")

#: Directory under the cache root where released component weights are kept.
WEIGHTS_DIRECTORY = "weights_components"

Mismatch = Literal["error", "skip"]


class PretrainedError(RuntimeError):
    """Released weights cannot be found, read, or do not fit the encoder."""


@dataclass(frozen=True)
class ReleasedWeights:
    """One published checkpoint that fits an encoder class of the toolbox.

    Attributes:
        file: File name under ``$GRASPPANDA_CACHE/weights_components``.
        url: Where the file is published (a revision-pinned link).
        component: Registered class it loads into.
        options: Constructor options with which every tensor of the class is in the file
            with the same shape (a strict load). A grasp config changes ``grid_size``
            freely; changing ``in_channels`` or a width needs ``pretrained_mismatch: skip``.
        licence: Licence of the weights as their publisher states it; ``NC`` marks
            non-commercial terms.
        megabytes: Size of the file.
        trained_on: Data and task of the checkpoint.
        paper: Work the weights belong to.
    """

    file: str
    url: str
    component: str
    options: Mapping[str, Any]
    licence: str
    megabytes: int
    trained_on: str
    paper: str


_HF = "https://huggingface.co"
_GAUSSIANCROSS = f"{_HF}/RayYoh/GaussianCross/resolve/3cbceee8b88113780cec6a734fea8b14e2baae4f"
_VOLT = f"{_HF}/KadirYilmaz/Volt/resolve/9dcb91d919983daa2aac99c37a9a6b2a948bb67e"
_PTV3 = f"{_HF}/Pointcept/PointTransformerV3/resolve/5e25cfaf759289b03db1f90f143b3dc5172929d3"
_SONATA = f"{_HF}/facebook/sonata/resolve/df99897472c09f91ba9288da0a034aacffc0b010"
_CONCERTO = f"{_HF}/Pointcept/Concerto/resolve/efc130718a564eba0ce4899318f3c20b19d9c752"
_LITEPT = f"{_HF}/prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002"

_SONATA_SMALL = {
    "enc_depths": [2, 2, 2, 6, 2],
    "enc_channels": [32, 64, 128, 256, 512],
    "enc_num_head": [2, 4, 8, 16, 32],
}

#: Released checkpoints that were downloaded and loaded strictly into their class
#: (``tests/integration/test_backbone_released_weights.py``). Other files of the same
#: releases fit in the same way; ``docs/slots/scene_encoders_sparse.md`` lists them.
RELEASED: dict[str, ReleasedWeights] = {
    "gaussiancross-scannet-lin": ReleasedWeights(
        "spunet_gaussiancross_scannet_lin.pth",
        f"{_GAUSSIANCROSS}/scannet/semseg-spunet-base-lin/model/model_best.pth",
        "SpUNet",
        {"out_channels": None},
        "MIT",
        157,
        "GaussianCross self-supervised pre-training on ScanNet: the linear-probing "
        "checkpoint of the release, whose class layer lies outside the encoder",
        "GaussianCross (Yao et al., ACM MM 2025)",
    ),
    "gaussiancross-scannet": ReleasedWeights(
        "spunet_gaussiancross_scannet.pth",
        f"{_GAUSSIANCROSS}/scannet/semseg-spunet-base/model/model_best.pth",
        "SpUNet",
        {},
        "MIT",
        470,
        "GaussianCross pre-training, fine-tuned for ScanNet semantic segmentation",
        "GaussianCross (Yao et al., ACM MM 2025)",
    ),
    "volt-teacher-scannet": ReleasedWeights(
        "spunet_volt_teacher_scannet.pth",
        f"{_VOLT}/teacher_weights/scannet_unet_teacher.pth",
        "SpUNet",
        {},
        "MIT",
        470,
        "ScanNet semantic segmentation; a teacher network of Volt",
        "Volt (Yilmaz et al., ECCV 2026)",
    ),
    "ptv3-scannet200": ReleasedWeights(
        "ptv3_scannet200.pth",
        f"{_PTV3}/scannet200-semseg-pt-v3m1-0-base/model/model_best.pth",
        "PTv3",
        {},
        "MIT",
        554,
        "supervised ScanNet200 semantic segmentation",
        "Point Transformer V3 (Wu et al., CVPR 2024)",
    ),
    "ptv3-s3dis-rpe": ReleasedWeights(
        "ptv3_s3dis_rpe.pth",
        f"{_PTV3}/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth",
        "PTv3",
        {"enable_rpe": True, "enc_patch_size": [128] * 5, "dec_patch_size": [128] * 4},
        "MIT",
        554,
        "supervised S3DIS semantic segmentation, relative position bias",
        "Point Transformer V3 (Wu et al., CVPR 2024)",
    ),
    "ptv3-nuscenes": ReleasedWeights(
        "ptv3_nuscenes.pth",
        f"{_PTV3}/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth",
        "PTv3",
        {"in_channels": 4},
        "MIT",
        554,
        "supervised nuScenes lidar segmentation (outdoor)",
        "Point Transformer V3 (Wu et al., CVPR 2024)",
    ),
    "sonata-small": ReleasedWeights(
        "sonata_small.pth",
        f"{_SONATA}/sonata_small.pth",
        "PTv3Sonata",
        _SONATA_SMALL,
        "CC BY-NC 4.0 (NC)",
        154,
        "Sonata self-supervised pre-training, small encoder",
        "Sonata (Wu et al., CVPR 2025)",
    ),
    "sonata": ReleasedWeights(
        "sonata.pth",
        f"{_SONATA}/sonata.pth",
        "PTv3Sonata",
        {},
        "CC BY-NC 4.0 (NC)",
        434,
        "Sonata self-supervised pre-training, base encoder",
        "Sonata (Wu et al., CVPR 2025)",
    ),
    "concerto-base-origin": ReleasedWeights(
        "concerto_base_origin.pth",
        f"{_CONCERTO}/concerto_base_origin.pth",
        "PTv3Sonata",
        {},
        "CC BY-NC 4.0 (NC)",
        433,
        "Concerto joint 2-D/3-D self-supervised pre-training: the model of the paper",
        "Concerto (Zhang et al., NeurIPS 2025)",
    ),
    "litept-small-structured3d": ReleasedWeights(
        "litept_small_structured3d.pth",
        f"{_LITEPT}/structured3d-semseg-litept-small-v1m1/model/model_best.pth",
        "LitePT",
        {},
        "MIT",
        152,
        "supervised Structured3D semantic segmentation (synthetic rooms)",
        "LitePT (Yue et al., CVPR 2026)",
    ),
    "litept-small-nuscenes": ReleasedWeights(
        "litept_small_nuscenes.pth",
        f"{_LITEPT}/nuscenes-semseg-litept-small-v1m1/model/model_best.pth",
        "LitePT",
        {"in_channels": 4},
        "MIT",
        152,
        "supervised nuScenes lidar segmentation (outdoor)",
        "LitePT (Yue et al., CVPR 2026)",
    ),
    "ptv3-ppt-s3dis": ReleasedWeights(
        "ptv3_ppt_s3dis.pth",
        f"{_PTV3}/s3dis-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth",
        "PTv3",
        {
            "enable_rpe": True,
            "enc_channels": [32, 64, 128, 256, 384],
            "enc_num_head": [2, 4, 8, 16, 24],
            "enc_patch_size": [128] * 5,
            "dec_channels": [64, 64, 128, 256],
            "dec_num_head": [4, 4, 8, 16],
            "dec_patch_size": [128] * 4,
            "pretrained_condition": 0,
        },
        "MIT",
        446,
        "Point Prompt Training: supervised jointly on ScanNet, S3DIS and Structured3D "
        "with one set of normalisation layers per dataset (0 ScanNet, 1 S3DIS, 2 Structured3D)",
        "PPT (Wu et al., CVPR 2024) with Point Transformer V3",
    ),
    "pointcontrast-nce": ReleasedWeights(
        "minkunet34c_pointcontrast_nce.pth",
        "https://www.dropbox.com/s/uzr88m1lab0abbl/nce.pth?dl=1",  # codespell:ignore nce
        "MinkUNet",
        {"out_channels": 32},
        "unstated (code MIT; ScanNet terms apply)",
        303,
        "PointContrast self-supervised pre-training (PointInfoNCE) on pairs of partial "
        "ScanNet RGB-D frames; Res16UNet34C, 3 input channels",
        "PointContrast (Xie et al., ECCV 2020)",
    ),
    "mask3d-scannet": ReleasedWeights(
        "minkunet34c_mask3d_scannet_pretrained.ckpt",
        "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/"
        "scannet_pretrained/scannet_pretrained.ckpt",
        "MinkUNet",
        {"out_channels": 20, "stem_kernel": 5},
        "unstated (code MIT; ScanNet terms apply)",
        159,
        "the Res16UNet34C backbone inside Mask3D, trained for instance segmentation of "
        "ScanNet (the model its S3DIS models are fine-tuned from); 3 input channels",
        "Mask3D (Schult et al., ICRA 2023)",
    ),
    "openscene-scannet-openseg": ReleasedWeights(
        "minkunet18a_openscene_scannet_openseg.pth.tar",
        "https://cvg-data.inf.ethz.ch/openscene/models/scannet_openseg.pth.tar",
        "MinkUNet",
        {
            "layers": [2] * 8,
            "planes": [32, 64, 128, 256, 128, 128, 96, 96],
            "stem_kernel": 5,
            "out_channels": 768,
            "final_bias": False,
        },
        "unstated (code Apache-2.0; ScanNet terms apply)",
        187,
        "OpenScene: MinkUNet18A distilled on ScanNet (2 cm voxels) to predict per-point "
        "OpenSeg / CLIP features (768 channels); its input is geometry alone, three "
        "constant ones per point, which is what GSNet feeds",
        "OpenScene (Peng et al., CVPR 2023)",
    ),
}


def weights_directory() -> Path:
    """``$GRASPPANDA_CACHE/weights_components``: where released files are looked up."""
    return cache_root() / WEIGHTS_DIRECTORY


def resolve_pretrained(spec: str, *, user: str) -> Path:
    """The file ``spec`` stands for: a name of :data:`RELEASED` or a path."""
    if spec in RELEASED:
        entry = RELEASED[spec]
        path = weights_directory() / entry.file
        if not path.is_file():
            raise PretrainedError(
                f"{user}: the released weights `{spec}` ({entry.paper}; {entry.licence}; "
                f"{entry.megabytes} MB) are not on this machine. Fetch them with\n"
                f"  mkdir -p {path.parent} && curl -L -o {path} {entry.url}\n"
                "or set `pretrained: null` to build the encoder without them (for example "
                "to load a grasp checkpoint that already holds the encoder)."
            )
        return path
    path = Path(os.path.expandvars(spec)).expanduser()
    if not path.is_file():
        known = ", ".join(sorted(RELEASED))
        raise PretrainedError(
            f"{user}: pretrained=`{spec}` is neither a file nor a known release "
            f"(known names: {known})."
        )
    return path


class _Inert:
    """Stands in for every class or function a checkpoint pickles besides tensors.

    Training checkpoints carry objects of their code base (a scheduler, numpy scalars
    of the best metric, an argument namespace). Nothing of that is needed and none of
    it is imported or run: the name resolves to this class, which ignores its
    arguments and its state.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> _Inert:
        return object.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass

    # A pickled container of a foreign type is filled through these.
    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def append(self, value: Any) -> None:
        pass

    def extend(self, values: Any) -> None:
        pass

    def add(self, value: Any) -> None:
        pass


class _Dict(dict):
    """A dictionary that accepts pickled attributes (a state dict's ``_metadata``)."""


_STORAGE_DTYPES = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


class _StoredTensor(_Inert):
    """A tensor of the archive that has not been read yet."""

    def __new__(cls, *args: Any, **kwargs: Any) -> _StoredTensor:
        return object.__new__(cls)

    def __init__(self, storage: Any, offset: Any, size: Any, stride: Any, *unused: Any) -> None:
        self.storage, self.offset, self.size, self.stride = storage, offset, size, stride


def _read_inert(path: Path) -> Any:
    """Read the pickle of a zip-format PyTorch checkpoint, resolving nothing.

    The pickle may name any class of the code base that wrote the file. Here a name is
    never imported: a tensor becomes a :class:`_StoredTensor`, an ordered dictionary a
    dictionary, everything else an :class:`_Inert`.
    """

    class Reader(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) == ("collections", "OrderedDict"):
                return _Dict
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return _StoredTensor
            if (module, name) == ("torch._utils", "_rebuild_parameter"):
                return lambda data, *unused: data
            if module == "torch" and name in _STORAGE_DTYPES:
                return _STORAGE_DTYPES[name]
            return _Inert

        def persistent_load(self, pid: Any) -> Any:
            # ("storage", storage type, key, location, element count)
            return (pid[1], str(pid[2]))

    with zipfile.ZipFile(path) as archive, archive.open(_pickle_name(archive, path)) as stream:
        return Reader(stream).load()


def _pickle_name(archive: zipfile.ZipFile, path: Path) -> str:
    pickles = [name for name in archive.namelist() if name.endswith("/data.pkl")]
    if len(pickles) != 1:
        raise pickle.UnpicklingError(f"{path} is not a zip-format PyTorch checkpoint")
    return pickles[0]


def _materialise(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Read the stored tensors of ``state`` from the archive; other values are kept."""
    out = dict(state)
    if not any(isinstance(value, _StoredTensor) for value in out.values()):
        return out
    with zipfile.ZipFile(path) as archive:
        root = _pickle_name(archive, path)[: -len("data.pkl")]
        for key, value in out.items():
            if not isinstance(value, _StoredTensor):
                continue
            dtype, record = value.storage
            flat = torch.frombuffer(bytearray(archive.read(f"{root}data/{record}")), dtype=dtype)
            size = tuple(int(n) for n in value.size)
            stride = tuple(int(n) for n in value.stride)
            out[key] = torch.as_strided(flat, size, stride, int(value.offset)).clone()
    return out


def _torch_load(path: Path) -> Any:
    """Tensors and plain values of a checkpoint, without running code named by it.

    ``torch.load(weights_only=True)`` first; a training checkpoint that pickles objects
    of its code base is read by :func:`_read_inert` instead.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # the restricted unpickler fails in several ways on foreign objects
        return _read_inert(path)


def read_state(path: Path) -> dict[str, Tensor]:
    """The tensors of a weight file (``.pth`` or ``.safetensors``), keys as stored."""
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise PretrainedError(
                f"reading {path} needs the `safetensors` package (pip install safetensors)"
            ) from error
        return dict(load_file(str(path)))
    try:
        payload = _torch_load(path)
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if isinstance(payload, Mapping) and isinstance(payload.get(key), Mapping):
                payload = payload[key]
                break
        if not isinstance(payload, Mapping):
            raise PretrainedError(f"{path} holds a {type(payload).__name__}, not a state dict")
        payload = _materialise(path, payload)
    except PretrainedError:
        raise
    except Exception as error:
        raise PretrainedError(
            f"cannot read {path} without executing pickled code: {error}\n"
            "Convert it to a plain state dict first."
        ) from error
    return {key: value for key, value in payload.items() if isinstance(value, Tensor)}


@dataclass
class PretrainedReport:
    """What happened to every tensor of the encoder."""

    source: str
    prefix: str
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    outside: int = 0

    @property
    def strict(self) -> bool:
        """Every tensor of the encoder came from the file, and the file's encoder has no
        tensor the class lacks."""
        return not (self.missing or self.mismatched or self.unexpected)

    def summary(self) -> str:
        text = (
            f"pretrained weights from {self.source} (prefix `{self.prefix}`): "
            f"{len(self.loaded)} tensors loaded"
        )
        for label, keys in (
            ("absent from the file", self.missing),
            ("of another shape", self.mismatched),
            ("in the file but not in the model", self.unexpected),
        ):
            if keys:
                shown = ", ".join(keys[:4]) + (", ..." if len(keys) > 4 else "")
                text += f"; {len(keys)} {label} ({shown})"
        if self.outside:
            text += f"; {self.outside} tensors outside the encoder ignored"
        return text


def _best_prefix(stored: Mapping[str, Tensor], own: Mapping[str, Tensor]) -> str:
    """The prefix of the file's keys under which most tensors of the model are found.

    Tensors found with the shape the model has count first, tensors found by name
    second; among equals the shortest prefix wins, so the empty prefix is chosen when
    the file is the encoder itself.
    """
    scores: dict[str, list[int]] = {}
    for key, value in stored.items():
        parts = key.split(".")
        for cut in range(min(len(parts), 4)):
            prefix = "".join(part + "." for part in parts[:cut])
            target = own.get(key[len(prefix) :])
            if target is not None:
                score = scores.setdefault(prefix, [0, 0])
                score[0] += int(target.shape == value.shape)
                score[1] += 1
    if not scores:
        return ""
    return max(scores, key=lambda prefix: (*scores[prefix], -len(prefix)))


def _convert_each(
    stored: Mapping[str, Tensor], convert: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]]
) -> dict[str, Tensor]:
    """Apply a per-tensor re-layout to every tensor of a file, heads included."""
    converted: dict[str, Tensor] = {}
    for key, value in stored.items():
        try:
            converted.update(convert({key: value}))
        except (ValueError, RuntimeError):
            converted[key] = value
    return converted


_CONDITIONED = re.compile(
    r"\.norm\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$"
)


def select_condition(state: Mapping[str, Tensor], index: int, *, user: str) -> dict[str, Tensor]:
    """Keep one set of the normalisation layers of a prompt-driven-normalisation model.

    Pointcept's ``PDNorm`` (Wu et al., CVPR 2024) replaces every normalisation layer by
    one layer per training dataset, stored as ``<layer>.norm.<i>.<tensor>``. Set
    ``index`` is renamed to ``<layer>.<tensor>``, the name the layer has without
    ``PDNorm``; the other sets are left out. All other tensors pass through.
    """
    selected: dict[str, Tensor] = {}
    found: set[int] = set()
    for key, value in state.items():
        match = _CONDITIONED.search(key)
        if match is None:
            selected[key] = value
            continue
        found.add(int(match.group(1)))
        if int(match.group(1)) == index:
            selected[f"{key[: match.start()]}.{match.group(2)}"] = value
    if index not in found:
        have = f"it has the sets {sorted(found)}" if found else "it has none"
        raise PretrainedError(
            f"{user}: pretrained_condition={index}, but the checkpoint holds no such set of "
            f"normalisation layers ({have}; set it to null for a checkpoint without "
            "prompt-driven normalisation)."
        )
    return selected


def load_pretrained(
    module: nn.Module,
    spec: str,
    *,
    mismatch: Mismatch = "error",
    convert: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    condition: int | None = None,
) -> PretrainedReport:
    """Load released weights into ``module`` and report every tensor.

    Args:
        module: The encoder.
        spec: A name of :data:`RELEASED` or a file path.
        mismatch: ``error`` refuses when a tensor of the encoder is absent from the file
            or has another shape there; ``skip`` leaves such tensors as initialised.
        convert: Re-layout of a state dict of the original code base for the class (for
            example :func:`grasppanda.models.backbones.spunet.convert_pointcept_state`).
            It must treat every tensor on its own; it is applied to the whole file, and
            a tensor outside the encoder that it cannot re-lay is left as it is.
        condition: For a checkpoint with prompt-driven normalisation, the index of the
            set of normalisation layers to load (:func:`select_condition`).
    """
    user = type(module).__name__
    if mismatch not in ("error", "skip"):
        raise ValueError(f"{user}: pretrained_mismatch must be `error` or `skip`, got {mismatch!r}")
    path = resolve_pretrained(spec, user=user)
    stored = read_state(path)
    if condition is not None:
        stored = select_condition(stored, condition, user=user)
    if convert is not None:
        stored = _convert_each(stored, convert)
    own = module.state_dict()
    prefix = _best_prefix(stored, own)
    state = {key[len(prefix) :]: value for key, value in stored.items() if key.startswith(prefix)}
    report = PretrainedReport(str(path), prefix, outside=len(stored) - len(state))
    accepted = {}
    for key, value in own.items():
        if key not in state:
            report.missing.append(key)
        elif state[key].shape != value.shape:
            report.mismatched.append(
                f"{key} {tuple(state[key].shape)} in the file, {tuple(value.shape)} here"
            )
        else:
            accepted[key] = state[key]
            report.loaded.append(key)
    report.unexpected = [key for key in state if key not in own]
    if not report.loaded:
        raise PretrainedError(
            f"{user}: no tensor of {path} fits this encoder; it is not a checkpoint of "
            f"this architecture. First keys of the file: {list(stored)[:3]}"
        )
    if mismatch == "error" and (report.missing or report.mismatched):
        raise PretrainedError(
            f"{user}: {report.summary()}.\nThe file does not fit the encoder as configured. "
            "Give the encoder the options of the checkpoint, or set "
            "`pretrained_mismatch: skip` to keep the initialisation of these tensors "
            "(usual when only the number of input channels or the output width differs)."
        )
    module.load_state_dict(accepted, strict=False)
    _LOG.info("%s: %s", user, report.summary())
    return report
