"""Fixtures for tests that need real data, a GPU or an upstream oracle.

These tests are skipped, not failed, when a resource is absent, so the unit suite stays
runnable anywhere. Locations come from the environment:

    GRASPPANDA_GRASPNET1B_ROOT       GraspNet-1Billion
    GRASPPANDA_GRASPCLUTTER6D_ROOT   GraspClutter6D
    GRASPPANDA_CACHE                 derived assets and caches
    GRASPPANDA_WEIGHTS               released third-party checkpoints, laid out as
                                     <method or 'components'>/<file>
    GRASPPANDA_ORACLE_PYTHON         interpreter of an environment with the official
                                     graspnetAPI (and, for model parity, the upstream
                                     dependencies); used only to record reference values
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from grasppanda.utils.paths import cache_root, dataset_root


def _existing(path: Path, what: str) -> Path:
    if not path.is_dir():
        pytest.skip(f"{what} not found at {path}")
    return path


@pytest.fixture(scope="session")
def graspnet1b_root() -> Path:
    return _existing(dataset_root("graspnet1b"), "GraspNet-1Billion")


@pytest.fixture(scope="session")
def graspclutter6d_root() -> Path:
    return _existing(dataset_root("graspclutter6d"), "GraspClutter6D")


@pytest.fixture(scope="session")
def asset_cache() -> Path:
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def oracle_python() -> str:
    interpreter = os.environ.get("GRASPPANDA_ORACLE_PYTHON", "")
    if not interpreter or not Path(interpreter).is_file():
        pytest.skip("no oracle interpreter configured (GRASPPANDA_ORACLE_PYTHON)")
    return interpreter


@pytest.fixture(scope="session")
def cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda")
