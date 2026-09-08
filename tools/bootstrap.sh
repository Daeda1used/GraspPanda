#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
unset PYTHONPATH PYTHONHOME
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
