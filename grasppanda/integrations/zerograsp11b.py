"""Local, streaming access to the author's ZeroGrasp-11B training shards."""
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from ..datasets import get_dataset

OBSERVATION_FIELDS = frozenset(('rgb.jpg', 'depth.png', 'camera.json', 'mask_visib.json', 'gt_info.json'))
TRAINING_FIELDS = OBSERVATION_FIELDS | {'depth_st.png', 'spc.npz', 'grasp.npz', 'gt.json', 'gt_info.json'}


def shard_path(root, index):
    return Path(root)/'train'/f'shard-{index:06d}.tar.gz'


def selected_ranges(config):
    ranges = {}
    for shard, sample in get_dataset(config.dataset).frame_keys(config.split, config.scene, config.frame, config.frames):
        ranges.setdefault(shard, []).append(sample)
    return ranges


def iter_samples(path, fields=OBSERVATION_FIELDS):
    """Yield contiguous WebDataset samples without extracting archive paths."""
    current, values, seen = None, {}, set()
    try:
        with tarfile.open(path, 'r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if '.' not in name:
                    raise ValueError(f'Unexpected shard member: {name}')
                key, suffix = name.split('.', 1)
                if not re.fullmatch(r'[A-Za-z0-9_-]+', key):
                    raise ValueError('Invalid ZeroGrasp sample key')
                if key != current:
                    if current is not None:
                        missing = set(fields)-set(values)
                        if missing:
                            raise ValueError(f'{path.name}/{current} is missing {sorted(missing)}')
                        yield current, values
                    if key in seen or len(seen) >= 100:
                        raise ValueError('ZeroGrasp shards require at most 100 distinct, contiguous sample groups')
                    seen.add(key)
                    current, values = key, {}
                if suffix in fields:
                    if suffix in values or not 0 < member.size <= 256*1024*1024:
                        raise ValueError(f'Duplicate or oversized ZeroGrasp field: {name}')
                    values[suffix] = archive.extractfile(member).read()
            if current is not None:
                missing = set(fields)-set(values)
                if missing:
                    raise ValueError(f'{path.name}/{current} is missing {sorted(missing)}')
                yield current, values
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError(f'Cannot read {path}; finish or verify the shard download: {error}') from error


def selected_samples(config, fields=OBSERVATION_FIELDS, shard_order=None):
    ranges = selected_ranges(config)
    for shard in ranges if shard_order is None else shard_order:
        wanted = set(ranges[shard])
        for index, (key, raw) in enumerate(iter_samples(shard_path(config.dataset_root, shard), fields)):
            if index in wanted:
                yield shard, index, key, raw
                wanted.remove(index)
            if not wanted:
                break
        if wanted:
            raise ValueError(f'Shard {shard} does not contain selected sample indices {sorted(wanted)}')


def observation(raw):
    import numpy as np
    from PIL import Image
    camera = json.loads(raw['camera.json'])
    intrinsics = np.asarray(camera['cam_K'], dtype=np.float32).reshape(3, 3)
    scale = float(camera['depth_scale'])
    if not np.isfinite(intrinsics).all() or min(intrinsics[0,0], intrinsics[1,1], scale) <= 0 or not np.isfinite(scale):
        raise ValueError('Invalid ZeroGrasp camera calibration or depth scale')
    with Image.open(io.BytesIO(raw['depth.png'])) as image:
        depth = np.asarray(image).astype(np.float32)*scale
    if depth.shape != (480,640) or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError('ZeroGrasp native training images require finite 480×640 depth in millimetres')
    with Image.open(io.BytesIO(raw['rgb.jpg'])) as image:
        if image.height != 480 or image.width < 640:
            raise ValueError('ZeroGrasp stereo RGB does not contain the calibrated left image')
        rgb = image.convert('RGB').crop((0,0,640,480))
    masks = {}
    info = json.loads(raw['gt_info.json'])
    for label, rle in json.loads(raw['mask_visib.json']).items():
        index = int(label)+1
        counts = list(rle['counts'])
        box = info[int(label)]['bbox_visib']
        # Match the author's correction for foreground-first RLE at the image
        # origin. Only visible 2D bounding boxes are read from gt_info.json.
        if len(counts)%2 == 0 and box[0] == box[1] == 0:
            counts.insert(0, 0)
        counts = np.asarray(counts, dtype=np.int64)
        if not 0 < index < 65536 or tuple(rle['size']) != depth.shape or counts.ndim != 1 or (counts < 0).any() or counts.sum() != depth.size:
            raise ValueError('Invalid visible-instance RLE mask')
        decoded = np.repeat(np.arange(len(counts), dtype=np.int64)%2, counts).reshape(depth.shape, order='F') > 0
        decoded &= depth > 10
        if decoded.any():
            masks[index] = decoded
    if not masks:
        raise ValueError('The selected observation has no valid instance pixels')
    # Released visible masks can overlap by a few boundary pixels. Preserve
    # each instance plane; converting model input to a label PNG loses them.
    return rgb, depth, dict(sorted(masks.items())), intrinsics


def write_observation(raw, destination):
    import numpy as np
    import yaml
    from PIL import Image
    rgb, depth, masks, intrinsics = observation(raw)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination/file for name,file in (
        ('rgb','rgb.png'), ('depth','depth.tiff'), ('mask','mask.png'), ('camera','camera.yml'))}
    rgb.save(paths['rgb'])
    Image.fromarray(depth).save(paths['depth'])
    mask = np.zeros(depth.shape,dtype=np.uint16)
    for label, plane in masks.items():
        mask[plane] = label
    Image.fromarray(mask).save(paths['mask'])
    np.savez_compressed(destination/'instance_masks.npz', **{str(label):plane for label,plane in masks.items()})
    paths['camera'].write_text(yaml.safe_dump({'left_p':np.c_[intrinsics,np.zeros(3)].reshape(-1).tolist(),
                                             'height':480,'width':640}))
    return paths, (rgb, depth, masks, intrinsics)


def member_hashes(raw):
    return {name: hashlib.sha256(value).hexdigest() for name,value in raw.items()}


def preflight(config):
    if config.workspace != 'provided_instance_masks':
        raise ValueError('ZeroGrasp requires the provided_instance_masks input protocol')
    if config.collision_thresh not in (0, .01):
        raise ValueError('ZeroGrasp collision_thresh is 0 (disabled) or 0.01 (native geometric filter); it is not a tunable density threshold')
    if config.checkpoint and not Path(config.checkpoint).is_file():
        raise ValueError('Selected ZeroGrasp checkpoint is missing; download registered weights')
    if config.action == 'infer' and not config.checkpoint:
        raise ValueError('ZeroGrasp inference requires a compatible checkpoint')
    if config.action not in ('infer','train','train_short'):
        raise ValueError('The public ZeroGrasp training shards do not define a held-out grasp AP protocol')
    if config.label_root or config.eval_batch_limit:
        raise ValueError('ZeroGrasp labels are inside each shard; there is no separate label root or published validation split in this adapter')
    if config.action in ('train','train_short') and (config.optimizer or config.scheduler or config.loss or config.augmentation):
        raise ValueError('ZeroGrasp currently retains its native AdamW, step scheduler, losses and observation augmentations')
    ranges = selected_ranges(config)
    if config.action in ('train','train_short') and config.frames < config.batch_size:
        raise ValueError('Select at least one complete batch of ZeroGrasp samples')
    for shard in ranges:
        path = shard_path(config.dataset_root, shard)
        if not path.is_file() or Path(str(path)+'.aria2').exists():
            raise ValueError(f'Missing or incomplete shard: {path}; see the ZeroGrasp dataset download guide')
    # Decode one observation before starting an expensive GPU worker. Remaining
    # selected samples are checked as the stream is consumed, with explicit errors.
    first = next(selected_samples(config))
    observation(first[3])


def frame_cloud(config, scene, frame):
    raise ValueError('ZeroGrasp uses its native RGB-D and instance tensors; its current provider does not expose a point-only method adapter')


def evaluate(config, out):
    raise ValueError('No held-out grasp AP is defined for the public ZeroGrasp training-shard workflow')
