import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import ROOT, Experiment, capabilities, catalogue, default_dataset


def read_configuration(path):
    """Preserve JSON number types, including scientific notation."""
    import yaml
    text = path.read_text()
    return json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)


def main():
    parser = argparse.ArgumentParser(prog="panda", description="GraspPanda · Modular visual grasping toolbox")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="Install or repair the shared runtime")
    ui = commands.add_parser("ui", help="Open the experiment browser")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    commands.add_parser("list", help="List runnable methods and operations")
    init = commands.add_parser('init', help='Create an editable local experiment without downloading or running')
    source = init.add_mutually_exclusive_group()
    source.add_argument('--method', choices=[mid for mid in catalogue() if capabilities(mid)], help='Method preset (default: graspnet_baseline)')
    source.add_argument('--example', help='Curated example name; see --list')
    source.add_argument('--list', action='store_true', help='List curated configuration examples')
    init.add_argument('--dataset-root', help='Set the dataset path in the generated configuration')
    init.add_argument('-o', '--output', type=Path, default=Path('experiment.local.yaml'), help='New YAML file (default: experiment.local.yaml); existing files are preserved')
    commands.add_parser("fetch", help="Fetch the pinned method implementations")
    commands.add_parser("doctor", help="Check the shared runtime and source revisions")
    fetch_weights = commands.add_parser("weights", help="Download weights for a method and camera")
    fetch_weights.add_argument("method", choices=list(catalogue()))
    fetch_weights.add_argument("--camera", choices=["realsense","kinect"], default="realsense")
    sdf = commands.add_parser("prepare-sdf", help="Prepare object SDF grids for fusion training")
    sdf.add_argument("arguments", nargs=argparse.REMAINDER)
    graph = commands.add_parser('prepare-gtg2', help='Prepare reusable labelled candidate graphs for ensemble training')
    graph.add_argument('arguments', nargs=argparse.REMAINDER)
    component_weights = commands.add_parser('component-weights', help='Download verified pretrained backbone weights')
    from .weights import component_records
    component_weights.add_argument('name', choices=list(component_records()))
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
    if len(sys.argv) > 1 and sys.argv[1] in ('prepare-sdf', 'prepare-gtg2'):
        script = {'prepare-sdf': 'prepare_sdf.py', 'prepare-gtg2': 'prepare_gtg2.py'}[sys.argv[1]]
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'grasppanda/runtime' / script), *sys.argv[2:]], cwd=ROOT))
    args = parser.parse_args()
    if args.command == "install":
        raise SystemExit(subprocess.call(["bash", str(ROOT / "grasppanda/runtime/bootstrap.sh")], cwd=ROOT))
    elif args.command == 'component-weights':
        from .weights import fetch_component
        print(fetch_component(args.name))
    elif args.command == "ui":
        from .ui import launch
        launch(args.host, args.port)
    elif args.command == "list":
        for mid, m in catalogue().items():
            if not capabilities(mid): continue
            print(f"{mid:28} {m['group']:28} {','.join(capabilities(mid))}")
    elif args.command == 'init':
        from .templates import configuration, examples, write_configuration
        if args.list:
            for name, entry in examples().items():
                print(f'{name:24} {entry["description"]}')
            return
        try:
            data = configuration(method=args.method, example=args.example, dataset_root=args.dataset_root)
            path = write_configuration(data, args.output)
        except FileExistsError:
            parser.error('Output already exists. Choose a different --output to preserve your configuration.')
        except (ValueError, OSError) as error:
            parser.error(str(error))
        import shlex
        command = 'sweep' if 'base' in data else 'run'
        print(f'Created {path}. Edit its paths and settings, then run:')
        print(f'./panda {command} {shlex.quote(str(path))}')
    elif args.command in ("fetch", "doctor", "weights"):
        script = {"fetch": "clone_upstreams.py", "doctor": "doctor.py", "weights": "fetch_weights.py"}[args.command]
        command = [sys.executable, str(ROOT / "grasppanda/runtime" / script)]
        if args.command == "weights":
            command += [args.method,"--camera",args.camera]
        raise SystemExit(subprocess.call(command, cwd=ROOT))
    elif args.command == 'sweep':
        from .sweeps import Sweep
        from .jobs import JobManager
        sweep = Sweep.from_dict(read_configuration(args.config))
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
        from .jobs import JobManager
        if args.command=='verify':
            from .recipes import preset,RECIPES
            config=preset(args.method,args.dataset_root)
            print(RECIPES.get(args.method,('', 'Configurable single-frame adapter'))[1],flush=True)
            print(json.dumps(config.to_dict(),indent=2),flush=True)
        else: config = Experiment.from_dict(read_configuration(args.config))
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
