import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import ROOT, Experiment, capabilities, catalogue, default_dataset


def main():
    parser = argparse.ArgumentParser(prog="panda", description="GraspPanda · Modular visual grasping toolbox")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="Install or repair the shared runtime")
    ui = commands.add_parser("ui", help="Open the experiment browser")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    commands.add_parser("list", help="List runnable methods and operations")
    commands.add_parser("fetch", help="Fetch the pinned method implementations")
    commands.add_parser("doctor", help="Check the shared runtime and source revisions")
    fetch_weights = commands.add_parser("weights", help="Download weights for a method and camera")
    fetch_weights.add_argument("method", choices=list(catalogue()))
    fetch_weights.add_argument("--camera", choices=["realsense","kinect"], default="realsense")
    sdf = commands.add_parser("prepare-sdf", help="Prepare object SDF grids for fusion training")
    sdf.add_argument("arguments", nargs=argparse.REMAINDER)
    verify=commands.add_parser("verify", help="Run the documented fixed recipe or single-frame preset")
    verify.add_argument("method", choices=list(catalogue()))
    verify.add_argument("--dataset-root", default=default_dataset())
    run = commands.add_parser("run", help="Run an experiment from YAML or JSON")
    run.add_argument("config", type=Path)
    run.add_argument("--runs-dir", type=Path)
    sweep = commands.add_parser('sweep', help='Preview or run a validated configuration grid')
    sweep.add_argument('config', type=Path)
    sweep.add_argument('--preview', action='store_true', help='Print exact configurations without downloading or running')
    sweep.add_argument('--runs-dir', type=Path)
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare-sdf':
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'grasppanda/runtime/prepare_sdf.py'), *sys.argv[2:]], cwd=ROOT))
    args = parser.parse_args()
    if args.command == "install":
        raise SystemExit(subprocess.call(["bash", str(ROOT / "grasppanda/runtime/bootstrap.sh")], cwd=ROOT))
    elif args.command == "ui":
        from .ui import launch
        launch(args.host, args.port)
    elif args.command == "list":
        for mid, m in catalogue().items():
            if not capabilities(mid): continue
            print(f"{mid:28} {m['group']:28} {','.join(capabilities(mid))}")
    elif args.command in ("fetch", "doctor", "weights"):
        script = {"fetch": "clone_upstreams.py", "doctor": "doctor.py", "weights": "fetch_weights.py"}[args.command]
        command = [sys.executable, str(ROOT / "grasppanda/runtime" / script)]
        if args.command == "weights":
            command += [args.method,"--camera",args.camera]
        raise SystemExit(subprocess.call(command, cwd=ROOT))
    elif args.command == 'sweep':
        import yaml
        from .sweeps import Sweep
        from .jobs import JobManager
        sweep = Sweep.from_dict(yaml.safe_load(args.config.read_text()))
        if args.preview:
            print(json.dumps(sweep.preview(), indent=2))
            return
        manager = JobManager(args.runs_dir, allow_attach=True)
        ids = []
        try:
            ids = manager.submit_sweep(sweep.to_dict())
            print(json.dumps({'jobs': ids}), flush=True)
            while any(manager.get(job)['state'] in ('queued', 'running') for job in ids):
                time.sleep(.5)
            rows = [manager.get(job) for job in ids]
            print(json.dumps(rows, indent=2))
            print(f'Artifacts: {manager.root}')
            raise SystemExit(0 if all(row['state'] == 'succeeded' for row in rows) else 1)
        except KeyboardInterrupt:
            for job in ids:
                manager.cancel(job)
            raise
        finally:
            manager.close()
    elif args.command in ("run","verify"):
        import yaml
        from .jobs import JobManager
        if args.command=='verify':
            from .recipes import preset,RECIPES
            config=preset(args.method,args.dataset_root)
            print(RECIPES.get(args.method,('', 'Configurable single-frame adapter'))[1],flush=True)
            print(json.dumps(config.to_dict(),indent=2),flush=True)
        else: config = Experiment.from_dict(yaml.safe_load(args.config.read_text()))
        manager = JobManager(getattr(args,'runs_dir',None), allow_attach=True)
        try:
            job = manager.submit(config)
            print(job, flush=True)
            while manager.get(job)["state"] in ("queued", "running"):
                time.sleep(0.5)
            row = manager.get(job)
            print(json.dumps(row, indent=2))
            print(f"Artifacts: {manager.root / job}")
            raise SystemExit(0 if row["state"] == "succeeded" else 1)
        except KeyboardInterrupt:
            if "job" in locals():
                manager.cancel(job)
            raise
        finally:
            manager.close()


if __name__ == "__main__":
    main()
