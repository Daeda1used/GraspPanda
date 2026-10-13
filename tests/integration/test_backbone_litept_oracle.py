"""LitePT compared with the official implementation (``prs-eth/LitePT``).

``tools/make_backbone_fixtures.py litept`` ran the official standalone model on the
shared tabletop scene; here :class:`~grasppanda.models.backbones.litept.LitePT` runs on
the same scene and has to agree. As for the PTv3 family there are two kinds of evidence:
released checkpoints loaded strictly, and analytic weights for the options no released
checkpoint uses.

One thing differs from PTv3. The official attention casts queries, keys and values to
float16; GraspPanda attends in float32. A fixture therefore holds two outputs of the
official model:

* ``features``: the official forward pass with that cast switched off. This is what the
  port computes, and it has to be reproduced to :data:`TOLERANCE`.
* ``half_features`` (released weights only): the official forward pass as released. The
  port has to stay within :data:`HALF_PRECISION` of it; ``half_difference`` records how
  far the two official outputs are apart.

The fixture also names the attention backend and the PointROPE implementation the
official model ran with; the recorder's docstrings say what stands in for flash-attention
where that package is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from grasppanda.registry import BACKBONES

from .test_backbone_oracle import (
    TOLERANCE,
    analytic_state,
    assert_matches,
    dense_output,
    device,  # noqa: F401  (a pytest fixture)
    fixture,
    load_checkpoint,
    needs_spconv,
    released,
    scene,  # noqa: F401  (a pytest fixture)
    state_keys,
    strict_float32,  # noqa: F401  (a pytest fixture, applied to every test)
)

pytestmark = pytest.mark.oracle

#: Fixture name -> released checkpoint below ``$GRASPPANDA_WEIGHTS``, parameters of its
#: backbone. The paper's table: LitePT-S 12.7 M, S* 16.0 M, B 45.1 M, L 85.9 M.
RELEASED = {
    "litept_scannet": ("components/scannet-semseg-litept-small-v1m1.pth", 12_712_932),
    "litept_scannet_insseg": ("components/scannet-insseg-litept-small-v1m2.pth", 15_988_716),
    "litept_structured3d_base": (
        "components/structured3d-semseg-litept-base-v1m1.pth",
        45_112_788,
    ),
    "litept_structured3d_large": (
        "components/structured3d-semseg-litept-large-v1m1.pth",
        85_880_448,
    ),
}
ANALYTIC = ("litept_analytic", "litept_analytic_dense")

#: Largest absolute difference accepted against the official model as released, whose
#: attention runs in float16, on features of magnitude 4 to 6. Measured values are listed
#: in docs/model_zoo_backbones.md: the port is within 9.2e-4 of the released model, and
#: the released model within 2.0e-3 of itself with float32 attention.
HALF_PRECISION = 5e-3


def build(recorded: dict[str, np.ndarray]):
    config = json.loads(str(recorded["config"]))
    return BACKBONES.build({"type": "LitePT", **config, "grid_size": float(recorded["grid_size"])})


@needs_spconv
@pytest.mark.parametrize("name", sorted(RELEASED))
def test_litept_reproduces_released_weights(name, scene, device):  # noqa: F811
    relative, parameters = RELEASED[name]
    recorded = fixture(name)
    path = released(relative, str(recorded["weights_sha256"]))
    model = build(recorded)
    prefix = "module.backbone."  # a Pointcept checkpoint: backbone plus task head
    state = load_checkpoint(path)["state_dict"]
    state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == int(recorded["parameters"]) == parameters

    out = dense_output(model, scene, device)
    assert_matches(out, recorded, TOLERANCE)
    # The official model as released: the same, up to what float16 attention changes.
    assert 0 < float(recorded["half_difference"]) < HALF_PRECISION
    released_output = recorded["half_features"]
    assert np.abs(out[recorded["feature_rows"]] - released_output).max() < HALF_PRECISION


@needs_spconv
@pytest.mark.parametrize("name", ANALYTIC)
def test_litept_matches_the_official_code_with_analytic_weights(name, scene, device):  # noqa: F811
    recorded = fixture(name)
    model = build(recorded)
    assert state_keys(model) == str(recorded["state_keys"]), "state dict layout differs"
    assert sum(p.numel() for p in model.parameters()) == int(recorded["parameters"])
    analytic_state(model)
    assert_matches(dense_output(model, scene, device), recorded, TOLERANCE)
