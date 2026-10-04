"""Object-level GraspGen contracts, without importing rendering or CUDA."""
import json
from pathlib import Path
import re
from ..config import ROOT


def trainer(config):
    return {'stage':'generator', 'grasps_per_object':40, **config.trainer}


def validate_options(config):
    if config.dataset_options:
        raise ValueError('GraspGen uses the registered isolated-object partial-depth protocol')
    if not isinstance(config.trainer,dict) or set(config.trainer)-{'stage','grasps_per_object'}:
        raise ValueError('GraspGen trainer accepts stage and grasps_per_object')
    if config.trainer and config.action not in ('train','train_short'):
        raise ValueError('GraspGen trainer parameters apply only to training')
    values = trainer(config)
    if values['stage'] not in ('generator','discriminator'):
        raise ValueError('GraspGen stage must be generator or discriminator')
    count = values['grasps_per_object']
    if type(count) is not int or not 20 <= count <= 1000 or count % 20:
        raise ValueError('grasps_per_object must be a multiple of 20 in [20, 1000] for native stratified sampling')
    if config.workspace != 'object_partial' or config.collision_thresh != 0:
        raise ValueError('GraspGen requires object_partial and collision_thresh: 0; outputs use the author gripper convention')
    if config.checkpoint_policy != 'strict' or config.sdf_root:
        raise ValueError('GraspGen requires strict checkpoint loading and an empty sdf_root')
    if config.eval_batch_limit or config.data_workers:
        raise ValueError('GraspGen renders on the job GPU with data_workers: 0; simulation evaluation remains upstream')
    if config.num_points != 2048:
        raise ValueError('The registered GraspGen partial-view protocol uses num_points: 2048')


def contained(root, value):
    path = (root/value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or non-local GraspGen asset: '+str(root/value))
    return path


def layout(config):
    root = Path(config.dataset_root)
    labels = root/'grasp_data/franka_panda'
    if not (labels/'uuid_index.json').is_file(): labels = root/'labels'
    return root, labels, root/'meshes'


def inventory(config, split=None):
    root, labels, meshes = layout(config)
    split = split or config.split
    split_file = root/(split+'.txt')
    if not split_file.is_file():
        raise ValueError('Missing GraspGen '+split+'.txt. Download starter data or follow Guide → Datasets for the full release.')
    ids = split_file.read_text().split()
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch('[0-9a-f]{32}',uid) for uid in ids):
        raise ValueError('GraspGen split must contain unique Objaverse UUIDs')
    other = root/('valid.txt' if split == 'train' else 'train.txt')
    if other.is_file() and set(ids).intersection(other.read_text().split()):
        raise ValueError('GraspGen train and valid splits overlap')
    mesh_map = json.loads((meshes/'map_uuid_to_path.json').read_text())
    sharded = (labels/'uuid_index.json').is_file()
    mapping = json.loads((labels/('uuid_index.json' if sharded else 'map_uuid_to_path.json')).read_text())
    rows = []
    for uid in ids:
        if uid not in mapping or uid not in mesh_map:
            raise ValueError('GraspGen split references an uninstalled object: '+uid)
        value = mapping[uid]
        if sharded and (type(value) is not int or value < 0):
            raise ValueError('Invalid GraspGen shard index')
        label = contained(labels,f'shard_{value:03d}.tar' if sharded else value)
        rows.append({'uid':uid, 'label':label, 'mesh':contained(meshes,mesh_map[uid]), 'sharded':sharded})
    return rows


def selection(config, rows):
    if config.action == 'train' and not config.train_batch_limit: return rows
    if config.scene+config.frames > len(rows):
        raise ValueError(f'This GraspGen {config.split} split has {len(rows)} installed objects; reduce scene or frames')
    return rows[config.scene:config.scene+config.frames]


class GraspLabelReader:
    """Native JSON schema with indexed TAR reads instead of rescanning a shard."""
    def __init__(self,config,rows):
        self.rows = {row['uid']:row for row in rows}
        self.cache = Path(config.label_root) if config.label_root else Path(config.dataset_root)/'.grasppanda/graspgen-index-v1'
        self.indices = {}

    def _index(self,path):
        import fcntl
        import hashlib
        import tarfile
        if path in self.indices: return self.indices[path]
        self.cache.mkdir(parents=True,exist_ok=True)
        name = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        target = self.cache/(name+'.json')
        stat = path.stat()
        identity = [stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino]
        with (self.cache/(name+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try: saved = json.loads(target.read_text())
            except (OSError,ValueError): saved = {}
            if saved.get('identity') == identity and saved.get('version') == 1:
                index = saved['members']
            else:
                index = {}
                with tarfile.open(path,'r:') as archive:
                    for member in archive:
                        match = re.fullmatch(r'([0-9a-f]{32})\.grasps\.json',member.name)
                        if not match: continue
                        if not member.isfile() or match[1] in index:
                            raise ValueError('Invalid or duplicate GraspGen TAR member: '+member.name)
                        index[match[1]] = [member.offset_data,member.size]
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps({'version':1,'identity':identity,'members':index}))
                temporary.replace(target)
        self.indices[path] = index
        return index

    def read_grasps_by_uuid(self,uid):
        if uid not in self.rows: return None
        row = self.rows[uid]
        if not row['sharded']: return json.loads(row['label'].read_text())
        index = self._index(row['label'])
        if uid not in index: raise ValueError('UUID missing from its registered GraspGen shard: '+uid)
        offset,size = index[uid]
        if type(offset) is not int or type(size) is not int or offset < 0 or not 0 < size <= 32*1024*1024 or offset+size > row['label'].stat().st_size:
            raise ValueError('Invalid cached GraspGen TAR range; remove the derived index and retry')
        with row['label'].open('rb') as stream:
            stream.seek(offset); value = stream.read(size)
        if len(value) != size: raise ValueError('Incomplete GraspGen TAR member: '+uid)
        return json.loads(value)


def assets(config):
    """Architecture and secondary weights participate in queued provenance."""
    from ..weights import records
    return {r['path']:ROOT/r['path'] for r in records('graspgen','synthetic-depth') if r.get('role') == 'auxiliary'}


def preflight(config):
    validate_options(config)
    if not config.dataset_root or not Path(config.dataset_root).is_dir():
        raise ValueError('Choose a GraspGen root; Download starter data installs four original objects')
    selection(config,inventory(config))
    for path in assets(config).values():
        if not path.is_file(): raise ValueError('Missing GraspGen model artifact; use Download registered weights: '+path.name)
    if config.checkpoint and not Path(config.checkpoint).is_file():
        raise ValueError('Missing GraspGen checkpoint; use Download registered weights')
    if config.action == 'infer' and not config.checkpoint:
        raise ValueError('GraspGen inference requires a trained generator or paired toolbox checkpoint')


def protocol(config):
    return {'dataset':config.dataset,'method':config.method,'gripper':'franka_panda',
            'split':config.split,'input':'2048 points from one noisy rendered depth view of an isolated object',
            'coordinate_frame':'observation point-cloud frame, centered at its mean',
            'translation_units':'metres','pose_convention':'author Franka Panda gripper base link',
            'output':'SE(3) matrices and discriminator confidence; no width or GraspNet conversion',
            'renderer':'author 256px, 60-degree FOV; prob_object_only=1; seeded depth noise',
            'inference_selection':'visual geometry only; no grasp-label visibility filtering',
            'training_selection':'native visible-positive-grasp filtering; retry invalid views without dropping objects'}


def frame_cloud(config, scene, frame):
    raise ValueError('GraspGen uses its object-centric renderer and native pose runner')


def evaluate(config, out):
    raise ValueError('GraspGen simulation success evaluation requires the upstream simulator workflow; no GraspNet AP conversion is registered')
