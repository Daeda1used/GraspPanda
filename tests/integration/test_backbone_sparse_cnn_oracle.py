"""The sparse CNNs ported from Pointcept, compared with the official implementation.

``tools/make_backbone_fixtures.py`` ran the official models on the shared tabletop
scene; here the GraspPanda backbones run on the same scene and have to agree. No
checkpoint of OA-CNNs has been released, so both sides fill their state dict with the
same closed-form values, which also proves that the two state dicts have identical keys
and shapes. Two kinds of fixture:

* **Analytic.** A small model that switches on what the published configuration leaves
  alone, with the state of ``analytic_state`` and nothing else.
* **Published configuration.** The official model built from the published config file,
  against the defaults of the backbone. These networks are dozens of residual blocks
  deep, which the sine-wave state does not survive (see :func:`hashed_state`), so the
  state comes from an integer hash and the batch-norm statistics from one training-mode
  pass over the scene on each side (:func:`calibrate_norms`).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from grasppanda.models.backbones.spunet import convert_pointcept_state
from grasppanda.registry import BACKBONES
from grasppanda.structures import PointSet

from .test_backbone_oracle import (
    TOLERANCE,
    analytic_state,
    assert_matches,
    fixture,
    needs_spconv,
    point_features,
    state_keys,
)

pytestmark = pytest.mark.oracle

#: Parameters of the published models, as the official code counts them.
PUBLISHED_PARAMETERS = {"oacnns_base": 51_494_484, "spunet_base": 39_157_844}


# -------------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def scene() -> torch.Tensor:
    return torch.from_numpy(fixture("tabletop")["cloud"])


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def strict_float32():
    """No TF32: it alone moves convolution outputs by more than the tolerance."""
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def hashed_state(model: nn.Module) -> None:
    """The closed-form state the fixture recorder gives a deep official model.

    Values come from a 32-bit integer hash of the tensor's name and flat index: exact on
    every platform, and, unlike the sine wave of ``analytic_state``, full-rank weight
    matrices with singular values of order one.
    """
    mask = np.uint64(0xFFFFFFFF)
    state = model.state_dict()
    for name, tensor in state.items():
        if not tensor.is_floating_point():
            continue
        leaf = name.rsplit(".", 1)[-1]
        seed = np.uint64(zlib.crc32(name.encode()))
        bits = (np.arange(tensor.numel(), dtype=np.uint64) + seed) & mask
        for shift, factor in ((16, 0x7FEB352D), (15, 0x31848BAB), (16, 0x2C1B3C6D)):
            bits = ((bits ^ (bits >> np.uint64(shift))) * np.uint64(factor)) & mask
        bits ^= bits >> np.uint64(15)
        wave = torch.from_numpy(bits.astype(np.float64) / 2.0**31 - 1.0)
        if leaf == "running_var":
            values = 1.0 + 0.5 * wave.pow(2)
        elif leaf == "running_mean":
            values = 0.1 * wave
        elif tensor.ndim >= 2:
            values = wave * (3.0 / (tensor.numel() // tensor.shape[0])) ** 0.5
        elif leaf == "weight":
            values = 1.0 + 0.1 * wave
        else:
            values = 0.1 * wave
        state[name] = values.reshape(tensor.shape).to(tensor.dtype)
    model.load_state_dict(state)


def calibrate_norms(model: nn.Module, run: Callable[[], object]) -> None:
    """Store the batch-norm statistics of one training-mode pass, as the recorder does."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [norm.momentum for norm in norms]
    for norm in norms:
        norm.momentum = 1.0
    model.train()
    try:
        run()
    finally:
        model.eval()
        for norm, momentum in zip(norms, momenta):
            norm.momentum = momentum


def closed_form_output(
    model: nn.Module,
    cloud: torch.Tensor,
    device: torch.device,
    calibrated: bool,
    relay: Callable[[nn.Module], None] | None = None,
) -> np.ndarray:
    """Features of the scene under the state the fixture was recorded with.

    ``relay`` is called once the closed-form values are in place, for a backbone whose
    tensors are laid out differently from the official model's.
    """
    (hashed_state if calibrated else analytic_state)(model)
    if relay is not None:
        relay(model)
    model = model.to(device).eval()
    points = PointSet(cloud.to(device))
    feats = point_features(cloud, model.in_channels).to(device)
    with torch.no_grad():
        if calibrated:
            calibrate_norms(model, lambda: model(points, feats))
        return model(points, feats).feats.cpu().numpy()


def build(kind: str, recorded: dict[str, np.ndarray], **overrides: object) -> nn.Module:
    config = json.loads(str(recorded["config"]))
    return BACKBONES.build(
        {"type": kind, **config, "grid_size": float(recorded["grid_size"]), **overrides}
    )


# ---------------------------------------------------------------------------- tests


@needs_spconv
@pytest.mark.parametrize("name", ["oacnns_analytic", "oacnns_base", "oacnns_notch"])
def test_oacnns_matches_the_official_code(name, scene, device):
    recorded = fixture(name)
    model = build("OACNNs", recorded)
    assert state_keys(model) == str(recorded["state_keys"]), "state dict layout differs"
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == int(recorded["parameters"])
    assert parameters == PUBLISHED_PARAMETERS.get(name, parameters)
    # Odd extents at several levels: the bounding-box rule is in play in every fixture.
    assert (recorded["extent"] % model.total_stride).any()
    # ``oacnns_notch`` brings its own cloud, crafted so that a level does not start at cell 0.
    cloud = torch.from_numpy(recorded["cloud"]) if "cloud" in recorded else scene
    out = closed_form_output(model, cloud, device, bool(recorded["calibrated"]))
    assert_matches(out, recorded, TOLERANCE)


@needs_spconv
def test_oacnns_fixture_is_sensitive_to_the_bounding_box_rule(scene):
    """Keeping the voxels the official model loses moves the output far beyond the tolerance."""
    recorded = fixture("oacnns_analytic")
    model = build("OACNNs", recorded, crop_odd_extent=False)
    out = closed_form_output(model, scene, torch.device("cpu"), calibrated=False)
    rows = recorded["feature_rows"]
    assert np.abs(out[rows] - recorded["features"]).max() > 10 * TOLERANCE


def official_layout(model: nn.Module) -> dict[str, torch.Tensor]:
    """The state of a facade-based ``SpUNet`` as the official model would hold it.

    The two differ in the 1×1 convolutions only: ``(out, in)`` linear weights here,
    ``(out, 1, 1, 1, in)`` kernels there. A closed-form state gives a tensor the same
    flat sequence of values under either shape, so reshaping is all it takes to obtain
    exactly what the recorder put into the official model.
    """
    return {
        key: value.reshape(value.shape[0], 1, 1, 1, value.shape[1]) if value.ndim == 2 else value
        for key, value in model.state_dict().items()
    }


@needs_spconv
@pytest.mark.parametrize("name", ["spunet_analytic", "spunet_base"])
def test_spunet_matches_the_official_code(name, scene, device):
    recorded = fixture(name)
    model = build("SpUNet", recorded)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == int(recorded["parameters"])
    assert parameters == PUBLISHED_PARAMETERS.get(name, parameters)

    def relay(model: nn.Module) -> None:
        """Load the official state through the documented conversion, strictly."""
        state = official_layout(model)
        lines = sorted(f"{key} {tuple(value.shape)}" for key, value in state.items())
        digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        assert digest == str(recorded["state_keys"]), "state dict layout differs"
        model.load_state_dict(convert_pointcept_state(state), strict=True)

    out = closed_form_output(model, scene, device, bool(recorded["calibrated"]), relay)
    assert_matches(out, recorded, TOLERANCE)


@needs_spconv
def test_spunet_fixture_is_sensitive_to_the_layout_of_1x1_kernels(scene):
    """Reading a 1×1 kernel as ``(out, in)``, as its shape suggests, is far off."""
    recorded = fixture("spunet_analytic")
    model = build("SpUNet", recorded)
    out = closed_form_output(model, scene, torch.device("cpu"), calibrated=False)
    rows = recorded["feature_rows"]
    assert np.abs(out[rows] - recorded["features"]).max() > 10 * TOLERANCE
