#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
unset PYTHONPATH PYTHONHOME
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This runtime requires Linux x86-64. See docs/INSTALL.md.' >&2
  exit 1
fi
for prerequisite in git python3 g++ cmake pkg-config; do
  if ! command -v "$prerequisite" >/dev/null 2>&1; then
    echo "Missing $prerequisite. Install the system prerequisites in docs/INSTALL.md." >&2
    exit 1
  fi
done
PANDA_NVCC="${GRASPPANDA_CUDA_HOME:-/usr/local/cuda-11.8}/bin/nvcc"
if [[ ! -x "$PANDA_NVCC" ]] || ! "$PANDA_NVCC" --version | grep -q 'release 11.8'; then
  echo 'Set GRASPPANDA_CUDA_HOME to a CUDA 11.8 compiler toolkit. See docs/INSTALL.md.' >&2
  exit 1
fi
mkdir -p logs environments/state environments/wheels
UV_BIN="${UV_BIN:-$PROJECT_ROOT/environments/bootstrap/uv}"
if [[ ! -x "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" ]]; then
  echo 'Install uv following https://docs.astral.sh/uv/getting-started/installation/ and rerun.' >&2
  exit 1
fi
"$UV_BIN" sync --python 3.11 --locked --inexact --no-dev
"$UV_BIN" pip install --python .venv/bin/python --no-build-isolation grasp-nms==1.0.2
python3 tools/clone_upstreams.py
.venv/bin/python tools/build_native.py
.venv/bin/python tools/build_components.py
.venv/bin/python tools/build_extras.py
"$UV_BIN" pip check --python .venv/bin/python
echo 'GraspPanda installed. Run ./panda weights graspness, then ./panda ui.'
