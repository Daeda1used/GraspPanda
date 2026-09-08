"""Grasp candidate preparation and content-verified, memory-mapped graph inputs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .config import ROOT, catalogue
from .jobs import digest

FORMAT = 'grasppanda_gtg2_samples_v1'
ARRAYS = ('inside', 'outside', 'inside_offsets', 'outside_offsets', 'targets', 'grasps', 'candidate_ids')


def geometry(options):
    return {k: v for k, v in options.items() if k not in
            ('k', 'max_points', 'include_outside', 'encoding', 'min_inside_infer', 'gpg_threads')}


def native_geometry():
    import numpy as np
    import open3d as o3d
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.eval_utils import create_table_points, transform_points, voxel_sample_points
    path = ROOT / catalogue()['gtg2']['path'] / 'Data_Gathering/generate_data.py'
    names = {'process_point_cloud', 'to_gripper_coord', 'get_scene_stuff'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names: raise RuntimeError('GtG2 geometry definitions changed')
    namespace = dict(np=np, o3d=o3d, copy=copy, get_config=get_config,
        create_table_points=create_table_points, transform_points=transform_points, voxel_sample_points=voxel_sample_points)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def observation(g, root, scene, camera, frame, inpaint, options, seed):
    import numpy as np
    import open3d as o3d
    cloud = g.loadScenePointCloud(sceneId=scene, camera=camera, annId=frame,
        use_workspace=True, align=False, use_inpainting=inpaint)
    if camera == 'kinect' and not inpaint:
        _, indices = cloud.hidden_point_removal([0, 0, 0], 50)
        cloud = cloud.select_by_index(indices)
    cloud = cloud.voxel_down_sample(options['voxel_size'])
    folder = Path(root) / 'scenes' / f'scene_{scene:04d}' / camera
    transform = np.load(folder/'cam0_wrt_table.npy') @ np.load(folder/'camera_poses.npy')[frame]
    cloud.transform(transform)
    cloud = native_geometry()['process_point_cloud'](cloud)
    cloud.transform(np.linalg.inv(transform))
    if len(cloud.points) < 5: raise ValueError('Too few workspace points for GPG plane fitting')
    o3d.utility.random.seed(seed)
    plane, indices = cloud.segment_plane(distance_threshold=options['plane_threshold'], ransac_n=5, num_iterations=1000)
    if len(indices) < 5 or abs(plane[2]) < 1e-8: raise ValueError('GPG requires a nonvertical workspace plane')
    inside = cloud.select_by_index(indices)
    outside = cloud.select_by_index(indices, invert=True)
    if not len(outside.points): raise ValueError('Workspace plane contains the entire observation')
    bounds = inside.get_axis_aligned_bounding_box()
    x, y = np.meshgrid(np.linspace(bounds.min_bound[0], bounds.max_bound[0], 100),
                       np.linspace(bounds.min_bound[1], bounds.max_bound[1], 100))
    z = (-plane[0]*x - plane[1]*y - plane[3]) / plane[2]
    plane_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    center = plane_points.mean(0)
    plane_points = (plane_points-center)*2 + center
    return outside + o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points)), outside


def nms(array, translation, rotation):
    import numpy as np
    from graspnetAPI import GraspGroup
    if not len(array): return np.empty((0, 17), dtype=np.float64)
    # grasp-nms uses strict distance comparisons: a zero threshold suppresses
    # nothing. Keep deterministic score ties without quadratic comparisons.
    if translation == 0 or rotation == 0:
        return array[np.argsort(-array[:, 0], kind='stable')]
    return GraspGroup(array).nms(translation, rotation).grasp_group_array


def candidates(g, root, scene, camera, frame, options, seed, training=False):
    import numpy as np
    import _grasppanda_gpg
    batches = []
    for inpaint in ((True, False) if options['dual_cloud'] else (True,)):
        cloud, interest = observation(g, root, scene, camera, frame, inpaint, options, seed)
        rows = _grasppanda_gpg.generate(np.asarray(cloud.points), options['gpg_samples'], options['gpg_threads'], seed)
        bounds = interest.get_axis_aligned_bounding_box()
        keep = np.all(rows[:, 13:16] >= bounds.min_bound-.05, axis=1)
        keep &= np.all(rows[:, 13:16] <= bounds.max_bound+.05, axis=1)
        rows[:, 1] = rows[:, 1]*2 + .02
        keep &= (rows[:, 1] < .085) & (rows[:, 1] <= options['max_width'])
        batches.append(rows[keep])
    result = nms(np.concatenate(batches), options['nms_translation'], np.deg2rad(options['nms_rotation_degrees']))
    initial = len(result)
    if training:
        variants = []
        for width_offset in [0., *options['train_width_offsets']]:
            for depth in options['train_depths']:
                rows = result.copy()
                rows[:, 3] = depth
                rows[:, 1] += width_offset
                mask = rows[:, 1] <= options['max_width']
                if width_offset: mask &= result[:, 1] <= .08
                variants.append(rows[mask])
        result = np.concatenate(variants)
    total = len(result)
    if options['candidate_limit'] and total > options['candidate_limit']:
        result = result[np.random.default_rng(seed).permutation(total)[:options['candidate_limit']]]
    if not np.isfinite(result).all(): raise ValueError('GPG generated nonfinite grasp parameters')
    return result, dict(initial_candidates=initial, expanded_candidates=total, selected_candidates=len(result))


def score_candidates(array, models, dexmodels, poses, config, table, batch_size=64):
    import numpy as np
    from scipy.spatial import cKDTree
    from graspnetAPI.utils import eval_utils as ev
    transformed = [ev.transform_points(m, p) for m, p in zip(models, poses)]
    scene = np.concatenate(transformed)
    object_ids = np.concatenate([np.full(len(m), i) for i, m in enumerate(transformed)])
    distance, nearest = cKDTree(scene).query(array[:, 13:16], k=2)
    closest = nearest[:, 0]
    ambiguous = np.flatnonzero(np.isclose(distance[:, 0], distance[:, 1], rtol=1e-12, atol=1e-12))
    for start in range(0, len(ambiguous), batch_size):
        indices = ambiguous[start:start+batch_size]
        closest[indices] = ev.compute_closest_points(array[indices, 13:16], scene)
    assigned = object_ids[closest]
    scene = np.concatenate([scene, table])
    friction = np.array([1.2, 1., .8, .6, .4, .2])
    quality, config = {}, copy.deepcopy(config)
    for value in friction:
        config['metrics']['force_closure']['friction_coef'] = float(value)
        quality[float(value)] = ev.GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
    scores = np.full(len(array), np.nan)
    collision = np.zeros(len(array), dtype=bool)
    for obj in range(len(models)):
        indices = np.flatnonzero(assigned == obj)
        for start in range(0, len(indices), batch_size):
            subset = indices[start:start+batch_size]
            masks, _, grasps = ev.collision_detection([array[subset]], [transformed[obj]],
                [dexmodels[obj]], [poses[obj]], scene, outlier=.05, return_dexgrasps=True)
            collision[subset] = masks[0]
            for j, index in enumerate(subset):
                scores[index] = (-1. if masks[0][j] or grasps[0][j] is None else
                    ev.get_grasp_score(grasps[0][j], dexmodels[obj], friction, quality))
    if not np.isfinite(scores).all(): raise ValueError('Candidate scoring produced invalid labels')
    targets = 1.2-scores
    targets[collision] = -1.
    targets[(scores == -1) & ~collision] = -.5
    return targets.astype(np.float32)


def input_hashes(root, scene, camera, frame):
    from graspnetAPI.utils.xmlhandler import xmlReader
    from graspnetAPI.utils.utils import parse_posevector
    root = Path(root)
    folder = root / 'scenes' / f'scene_{scene:04d}' / camera
    files = [folder / name for name in ('camera_poses.npy', 'cam0_wrt_table.npy')]
    files += [folder / kind / f'{frame:04d}.{suffix}' for kind, suffix in
              (('rgb', 'png'), ('depth', 'png'), ('label', 'png'), ('meta', 'mat'), ('annotations', 'xml'))]
    objects = {parse_posevector(v)[0] for v in xmlReader(str(files[-1])).getposevectorlist()}
    for obj in sorted(objects):
        files.append(root/'models'/f'{obj:03d}'/'nontextured.ply')
        cache = root/'dex_models'/f'{obj:03d}.pkl'
        files += [cache] if cache.exists() else [root/'models'/f'{obj:03d}'/('textured'+s) for s in ('.obj', '.sdf')]
    return {str(p.relative_to(root)): digest(p) for p in files}


def contract(root, scene, camera, frame, options, seed):
    pins = json.loads((ROOT/'grasppanda/resources/upstreams.lock.json').read_text())
    native_pins = json.loads((ROOT/'grasppanda/resources/native_sources.lock.json').read_text())
    sources = {p['id']: p['pinned_commit'] for p in pins if p['id'] in ('gtg2', 'graspnet_api')}
    sources['gpg'] = next(p['commit'] for p in native_pins if p.get('id') == 'gpg')
    return dict(format=FORMAT, scene=scene, camera=camera, frame=frame, seed=seed,
        geometry=geometry(options), sources=sources, input_hashes=input_hashes(root, scene, camera, frame),
        preparation_sha256=digest(Path(__file__)), binding_sha256=digest(ROOT/'grasppanda/resources/gpg_binding.cpp'))


def cache_path(cache_root, record):
    key = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return Path(cache_root)/record['camera']/f"scene_{record['scene']:04d}"/f"{record['frame']:04d}"/key


def load_cache(path, expected=None, verify=True):
    import numpy as np
    path = Path(path)
    record = json.loads((path/'manifest.json').read_text())
    if record.get('format') != FORMAT or expected is not None and record.get('contract') != expected:
        raise ValueError('Prepared graph data does not match the requested input contract')
    if set(record.get('files', {})) != {name+'.npy' for name in ARRAYS}:
        raise ValueError('Incomplete graph cache manifest')
    if verify and any(digest(path/name) != sha for name, sha in record['files'].items()):
        raise ValueError('Prepared graph data changed or is corrupted')
    data = {name: np.load(path/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in ARRAYS}
    count = len(data['targets'])
    if data['targets'].shape != (count,) or data['grasps'].shape != (count, 17) or data['candidate_ids'].shape != (count,):
        raise ValueError('Invalid candidate/label alignment in graph cache')
    for region in ('inside', 'outside'):
        points, offsets = data[region], data[region+'_offsets']
        if points.ndim != 2 or points.shape[1] != 3 or offsets.shape != (count+1,):
            raise ValueError('Invalid packed graph array shapes')
        if offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != len(points) or (offsets[1:] < offsets[:-1]).any():
            raise ValueError('Invalid packed graph offsets')
        if points.dtype.kind != 'f' or not np.isfinite(points).all(): raise ValueError('Invalid graph cache point coordinates')
    targets = data['targets']
    valid = (targets == -1.) | (targets == -.5) | ((targets >= 0) & (targets <= 1.))
    if type(record['graphs']) is not int or count != record['graphs'] or not valid.all():
        raise ValueError('Invalid graph cache labels')
    if not np.isfinite(data['grasps']).all() or (data['grasps'][:, 1:4] <= 0).any():
        raise ValueError('Invalid cached grasp geometry')
    ids = data['candidate_ids']
    if (ids.dtype.kind not in 'iu' or len(np.unique(ids)) != count or (ids < 0).any() or
            (ids >= record['counts']['selected_candidates']).any()):
        raise ValueError('Graph candidate identifiers must be distinct integers')
    return data, record


def prepare_frame(root, cache_root, scene, camera, frame, options, seed=0, scoring_batch=64):
    import fcntl
    import numpy as np
    from graspnetAPI import GraspNet, GraspNetEval
    seed = (seed + scene*256 + frame) % 2**32
    expected = contract(root, scene, camera, frame, options, seed)
    destination = cache_path(cache_root, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent/(destination.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            load_cache(destination, expected)
            return destination
        g = GraspNet(str(root), camera=camera, split='train')
        ge = GraspNetEval(str(root), camera=camera, split='train')
        rows, counts = candidates(g, root, scene, camera, frame, options, seed, training=True)
        native = native_geometry()
        config, table, models, dexmodels, poses = native['get_scene_stuff'](ge, scene, frame)
        targets = score_candidates(rows, models, dexmodels, poses, config, table, scoring_batch) if len(rows) else np.empty(0, np.float32)
        cloud = g.loadScenePointCloud(sceneId=scene, annId=frame, camera=camera,
            use_workspace=True, align=False, use_inpainting=True).voxel_down_sample(options['voxel_size'])
        inside, outside, selected = [], [], []
        for index, row in enumerate(rows):
            a, b, _, _ = native['to_gripper_coord'](cloud, dict(t=row[13:16], R=row[4:13].reshape(3, 3),
                width=row[1], depth=row[3], score=float(targets[index])),
                gripper_depth=options['gripper_depth'], gripper_height=options['gripper_height'], bound_size=options['bound_size'])
            if len(a) >= options['min_inside_train']:
                inside.append(a); outside.append(b); selected.append(index)
        indices = np.array(selected, dtype=np.int64)
        arrays = dict(inside=np.concatenate(inside) if inside else np.empty((0, 3), np.float32),
            outside=np.concatenate(outside) if outside else np.empty((0, 3), np.float32),
            inside_offsets=np.cumsum([0]+[len(a) for a in inside], dtype=np.int64),
            outside_offsets=np.cumsum([0]+[len(a) for a in outside], dtype=np.int64),
            targets=targets[indices], grasps=rows[indices], candidate_ids=indices)
        if input_hashes(root, scene, camera, frame) != expected['input_hashes']:
            raise ValueError('Dataset files changed during candidate preparation; retry with stable inputs')
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.prepare-') as temporary:
            temporary = Path(temporary)
            for name, array in arrays.items(): np.save(temporary/(name+'.npy'), array, allow_pickle=False)
            record = dict(format=FORMAT, contract=expected, graphs=len(indices), counts=counts,
                files={name+'.npy': digest(temporary/(name+'.npy')) for name in ARRAYS})
            (temporary/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
            load_cache(temporary, expected)
            os.replace(temporary, destination)
        return destination
