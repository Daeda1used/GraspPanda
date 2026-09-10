"""Prepare camera-aligned CAD observations for Scale-Balanced-Grasp NcM."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def prepare_frame(root, output, camera, scene, frame, voxel_size=.005, distance=.008):
    import fcntl
    root, output = Path(root).resolve(), Path(output).resolve()
    lock_path = output/f'scene_{scene:04d}'/camera/'manifests'/f'{frame:04d}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_frame(root, output, camera, scene, frame, voxel_size, distance)


def _prepare_frame(root, output, camera, scene, frame, voxel_size, distance):
    import numpy as np
    import scipy.io
    import open3d as o3d
    from graspnetAPI.utils.utils import xmlReader, parse_posevector
    from grasppanda.config import Experiment
    from grasppanda.worker import frame_cloud, prepare
    from grasppanda.jobs import digest
    from grasppanda.methods.scale_balanced_data import frame_paths
    config = Experiment(method='scale_balanced_grasp', dataset_root=str(root), label_root=str(output), camera=camera)
    points_path, seg_path, manifest = frame_paths(config, scene, frame)
    folder = root/'scenes'/f'scene_{scene:04d}'/camera
    inputs = [folder/'meta'/f'{frame:04d}.mat', folder/'depth'/f'{frame:04d}.png',
              folder/'label'/f'{frame:04d}.png', folder/'camera_poses.npy', folder/'cam0_wrt_table.npy',
              folder/'annotations'/f'{frame:04d}.xml']
    meta = scipy.io.loadmat(inputs[0]); annotation = xmlReader(str(inputs[-1]))
    objects = [parse_posevector(value) for value in annotation.getposevectorlist()]
    if sorted(obj+1 for obj,_ in objects) != sorted(meta['cls_indexes'].flatten().tolist()):
        raise ValueError('CAD annotation IDs differ from RGB-D frame metadata')
    for obj,_ in objects: inputs.append(root/'models'/f'{obj:03d}'/'nontextured.ply')
    input_hashes = {str(p.relative_to(root)):digest(p) for p in inputs}
    parameters = dict(voxel_size=voxel_size, distance=distance)
    if any(p.exists() for p in (points_path, seg_path, manifest)):
        from grasppanda.methods.scale_balanced_data import verify_frame
        verify_frame(config, scene, frame)
        record = json.loads(manifest.read_text())
        if record['inputs'] != input_hashes or record['parameters'] != parameters:
            raise ValueError('Existing clean cache uses different sources or geometry; choose a new output directory')
        return record
    import os
    previous_directory = Path.cwd()
    try:
        prepare(config.method)
        raw, *_ = frame_cloud(config, scene, frame)
    finally:
        os.chdir(previous_directory)
    clouds, labels = [], []
    for obj, pose in objects:
        model = o3d.io.read_point_cloud(str(root/'models'/f'{obj:03d}'/'nontextured.ply'))
        if not len(model.points): raise ValueError(f'Object {obj} has no CAD points')
        model.transform(pose)
        xyz = np.asarray(model.voxel_down_sample(voxel_size).points)
        clouds.append(xyz); labels.append(np.full(len(xyz), obj+1, dtype=np.int64))
    # Native table grid and alignment; both XML and RGB-D use the selected camera.
    x, y, z = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500), np.linspace(0, .01, 1), indexing='xy')
    table = np.stack([x-.5, y-.5, z], -1).reshape(-1, 3)
    transform = np.linalg.inv(np.load(folder/'cam0_wrt_table.npy') @ np.load(folder/'camera_poses.npy')[frame])
    table = table @ transform[:3,:3].T + transform[:3,3]
    clouds.append(table); labels.append(np.zeros(len(table), dtype=np.int64))
    combined = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(clouds)))
    observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw))
    keep = np.asarray(combined.compute_point_cloud_distance(observed)) < distance
    points, seg = np.asarray(combined.points)[keep], np.concatenate(labels)[keep]
    if not len(points): raise ValueError('No CAD points agree with the observed depth; check camera and object poses')
    for p in (points_path, seg_path, manifest): p.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open('xb') as stream: np.save(stream, points, allow_pickle=False)
    with seg_path.open('xb') as stream: np.save(stream, seg, allow_pickle=False)
    record = dict(format='grasppanda-clean-v1', camera=camera, scene=scene, frame=frame,
                  parameters=parameters, inputs=input_hashes, meta_sha256=digest(inputs[0]),
                  points_sha256=digest(points_path), seg_sha256=digest(seg_path), point_count=len(points))
    with manifest.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
    return record


def _worker_threads():
    import os
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '1')


def _prepare_task(task):
    result = prepare_frame(*task)
    return result['scene'], result['frame'], result['point_count']


def main():
    parser = argparse.ArgumentParser(prog='panda prepare-clean-scenes', description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--camera', choices=('realsense', 'kinect'), default='realsense')
    parser.add_argument('--scenes', type=int, nargs='+', default=list(range(100)))
    parser.add_argument('--frames', type=int, nargs='+', default=list(range(256)))
    parser.add_argument('--voxel-size', type=float, default=.005)
    parser.add_argument('--distance', type=float, default=.008)
    parser.add_argument('--workers', type=int, default=1, help='Parallel preparation processes sharing the same runtime (1–32)')
    args = parser.parse_args()
    if not all(0 <= s < 100 for s in args.scenes) or not all(0 <= f < 256 for f in args.frames): parser.error('Preparation uses training scenes 0..99 and frames 0..255')
    if not .0005 <= args.voxel_size <= .02 or not .001 <= args.distance <= .05: parser.error('Invalid voxel size or agreement distance in metres')
    if not 1 <= args.workers <= 32: parser.error('workers must be between 1 and 32')
    dataset_root, output_root = args.dataset_root.resolve(), args.output_root.resolve()
    tasks = ((dataset_root, output_root, args.camera, scene, frame, args.voxel_size, args.distance)
             for scene in dict.fromkeys(args.scenes) for frame in dict.fromkeys(args.frames))
    if args.workers == 1:
        for scene, frame, count in map(_prepare_task, tasks):
            print(f'{scene:04d}/{frame:04d}: {count} clean points', flush=True)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_worker_threads)
        try:
            for scene, frame, count in pool.map(_prepare_task, tasks):
                print(f'{scene:04d}/{frame:04d}: {count} clean points', flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__': main()
