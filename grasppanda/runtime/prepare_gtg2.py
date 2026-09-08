"""Prepare reusable, officially scored candidate-graph inputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output-root', type=Path, help='Prepared graph directory; defaults to config.label_root')
    parser.add_argument('--camera', choices=('realsense', 'kinect'))
    parser.add_argument('--scenes', help='Training scene range, end exclusive, or comma-separated IDs; defaults to config.trainer.scenes or 0:100')
    parser.add_argument('--frame', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--config', type=Path, help='Training YAML/JSON; explicit CLI arguments override its preparation settings')
    parser.add_argument('--candidate-limit', type=int, help='Explicitly limit scored candidates per frame; 0 keeps all')
    parser.add_argument('--scoring-batch', type=int, default=64, help='Geometry evaluation batch size; does not cap candidates')
    args = parser.parse_args()
    from grasppanda.gtg2_options import resolved, TRAINER
    modules = {}
    raw = {}
    if args.config:
        import yaml
        raw = yaml.safe_load(args.config.read_text())
        from grasppanda.config import Experiment
        config = Experiment.from_dict(raw)
        if config.method != 'gtg2' or config.action != 'train': parser.error('Expected a GtG2 training configuration')
        raw = config.to_dict()
        modules = config.modules
    args.dataset_root = args.dataset_root or raw.get('dataset_root')
    args.output_root = args.output_root or raw.get('label_root')
    if not args.dataset_root or not args.output_root:
        parser.error('Provide --dataset-root and --output-root, or set dataset_root and label_root in --config')
    args.dataset_root, args.output_root = [Path(p).expanduser() for p in (args.dataset_root, args.output_root)]
    for name, default in (('camera', 'realsense'), ('frame', 0), ('seed', 0)):
        if getattr(args, name) is None: setattr(args, name, raw.get(name, default))
    if args.candidate_limit is not None:
        from grasppanda.module_options import unpack
        kind, opts = unpack(modules.get('crop', 'grasp_graph'))
        modules = {**modules, 'crop': {'type': 'grasp_graph' if kind == 'upstream' else kind, **opts, 'candidate_limit': args.candidate_limit}}
    _, options = resolved(modules)
    try:
        scenes = (list(range(*map(int, args.scenes.split(':')))) if ':' in args.scenes else list(map(int, args.scenes.split(',')))) if args.scenes else raw.get('trainer', {}).get('scenes', TRAINER['scenes'])
    except ValueError: parser.error('Invalid scene range')
    if not scenes or any(not 0 <= scene < 100 for scene in scenes) or len(set(scenes)) != len(scenes):
        parser.error('Use distinct training scene IDs from 0 to 99')
    if not 0 <= args.frame < 256 or not 0 <= args.seed < 2**31 or not 1 <= args.scoring_batch <= 512:
        parser.error('Invalid frame, seed or scoring batch size')
    from grasppanda.compat import legacy_torch
    legacy_torch()
    from grasppanda.gtg2_data import prepare_frame
    for scene in scenes:
        path = prepare_frame(args.dataset_root.resolve(), args.output_root.resolve(), scene, args.camera,
                             args.frame, options, args.seed, args.scoring_batch)
        print(f'Prepared scene {scene:04d}/{args.frame:04d}: {path}', flush=True)


if __name__ == '__main__': main()
