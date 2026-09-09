"""Read-only shared runtime and pinned source checks."""
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
modules = ["torch", "gradio", "MinkowskiEngine", "pytorch3d", "pointnet2._ext", "pointnet2_ops._ext", "knn_pytorch", "graspnetAPI", "_grasppanda_openpoints_cuda", "robo_orchard_core", "transformers", "spconv.pytorch", "torch_scatter", "timm", "selective_scan_cuda_oflex", "_grasppanda_deepla_cuda"]
result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "modules": {}}
modules += ['flash_attn', '_grasppanda_sampling_cuda', '_grasppanda_pointrope_cuda', '_grasppanda_pointops', '_grasppanda_pointcept_cuda', 'ocnn', 'dwconv.core', '_grasppanda_gpg', 'fpsample', 'torch_cluster', '_grasppanda_pointmamba_scan', '_grasppanda_causal_conv1d', '_grasppanda_pcm_scan', '_grasppanda_pcm_causal']
for name in modules:
    try:
        mod = importlib.import_module(name)
        result["modules"][name] = {"status": "ok", "file": mod.__file__}
    except Exception as error:
        result["modules"][name] = {"status": "failed", "error": str(error)}
import torch
result["torch"] = torch.__version__
result["cuda_runtime"] = torch.version.cuda
result["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
result["sources"] = []
pins=json.loads((ROOT / "grasppanda/resources/upstreams.lock.json").read_text())
pins+=json.loads((ROOT / "grasppanda/resources/component_sources.lock.json").read_text())
pins += [p for p in json.loads((ROOT/'grasppanda/resources/native_sources.lock.json').read_text()) if p.get('id') == 'gpg']
for entry in {p["path"]:p for p in pins}.values():
    repo = ROOT / entry["path"]
    if not repo.exists():
        result["sources"].append({"id": entry["id"], "status": "missing"})
        continue
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    result["sources"].append({"id": entry["id"], "status": "ok" if sha == (entry.get("pinned_commit") or entry["commit"]) and not changes else "changed"})
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["gpu"] and all(r["status"] == "ok" for r in result["modules"].values()) and all(r["status"] == "ok" for r in result["sources"]) else 1)
