"""Native ZeroGrasp reconstruction and grasp decoding on released local shards."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import sys
import time
from types import MethodType

from ..jobs import digest


def build(config, out=None):
    import torch
    import segmentation_models_pytorch as smp
    from ..components import load_checkpoint
    from ..module_options import unpack
    from ..worker import prepare
    repo = prepare('zerograsp')
    from zerograsp.utils.config import parse_config
    argv = sys.argv
    try:
        sys.argv = ['zerograsp']
        with redirect_stdout(io.StringIO()):
            native = parse_config(str(repo/'configs/default.yaml'))
    finally:
        sys.argv = argv
    native.img_height, native.img_width = 480, 640
    native.update_octree = config.action == 'infer'
    native.lr, native.batch_size, native.num_workers = config.learning_rate, config.batch_size, config.data_workers
    choice, options = unpack(config.modules.get('backbone', 'upstream'))
    module_spec = importlib.util.spec_from_file_location('_grasppanda_zerograsp_trainer', repo/'main.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    constructor = smp.Unet
    # A complete grasp checkpoint already supplies the image encoder weights.
    # Avoid a second network download, without changing any saved tensor keys.
    if config.checkpoint or choice != 'upstream':
        smp.Unet = lambda *args, **kwargs: constructor(*args, **{**kwargs, 'encoder_weights': None})
    try:
        model = module.BaseTrainer(native)
    finally:
        smp.Unet = constructor
    prefixes = []
    if choice != 'upstream':
        from ..modules.zerograsp_image import DenseImageFeatures
        model.model.backbone = DenseImageFeatures(choice, **options)
        prefixes = ['model.backbone.']
    checkpoint = {}
    transfer = {'policy': 'random_initialization', 'initialized': sorted(model.state_dict()), 'discarded': []}
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        saved = checkpoint.get('grasppanda', {}).get('config')
        if saved and config.checkpoint_policy == 'strict' and saved.get('modules',{}) != config.modules:
            raise ValueError('Strict ZeroGrasp loading requires the component configuration saved with the checkpoint')
        state = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        transfer = load_checkpoint(model, state, prefixes, config.checkpoint_policy)
    adapt_feature_extractor(model.model)
    if out is not None:
        (out/'component_transfer.json').write_text(json.dumps(transfer, indent=2)+'\n')
    return model, native, checkpoint


def adapt_feature_extractor(model):
    """Supply scene boundaries required by the pinned OFE extension.

    The author network batches octrees by object and repeats depth per object.
    Other objects from a different scene must never enter its occlusion test.
    Intrinsics may also vary between scenes; preserve each scene's calibration.
    """
    import torch
    original_batch = model.process_batch
    original_forward = model.ofe.forward

    def process_batch(self, batch, features):
        counts = [int(torch.unique(points.labels).numel()) for points in batch[3]]
        if any(count == 0 for count in counts):
            raise ValueError('ZeroGrasp cannot construct an octree for an empty instance set')
        if [int(mask[0].shape[0]) for mask in batch[1]] != counts:
            raise ValueError('ZeroGrasp instance masks and visible-point labels disagree')
        self.ofe.scene_counts = counts
        self.ofe.scene_intrinsics = batch[5]
        return original_batch(batch, features)

    def forward(self, points, masks, depth, intrinsics, batch_id, grid_size):
        # The network carries [self, neighbour] channels, while the pinned CUDA
        # kernel indexes one H×W instance mask and computes other-object
        # occlusion itself. Passing the interleaved tensor corrupts pixel indices.
        if masks.ndim != 4 or masks.shape[-1] != 2:
            raise ValueError('ZeroGrasp OFE expects native self/neighbor mask pairs')
        masks = masks[...,0].contiguous()
        counts = self.scene_counts
        count_tensor = torch.tensor(counts, dtype=torch.int32, device=points.device)
        ends = count_tensor.cumsum(0, dtype=torch.int32)
        starts = ends-count_tensor
        object_starts = torch.repeat_interleave(starts, count_tensor).contiguous()
        object_ends = torch.repeat_interleave(ends, count_tensor).contiguous()
        cameras = self.scene_intrinsics
        if torch.allclose(cameras, cameras[:1].expand_as(cameras), rtol=0, atol=1e-6):
            return original_forward(points, masks, depth, cameras[0].contiguous(), batch_id,
                                    object_starts, object_ends, grid_size)
        result = torch.empty((points.shape[0],2),dtype=torch.int32,device=points.device)
        offset = 0
        for index, count in enumerate(counts):
            selected = (batch_id >= offset)&(batch_id < offset+count)
            if selected.any():
                result[selected] = original_forward(points[selected].contiguous(), masks, depth,
                    cameras[index].contiguous(), batch_id[selected].contiguous(), object_starts, object_ends, grid_size)
            offset += count
        return result

    model.process_batch = MethodType(process_batch, model)
    model.ofe.forward = MethodType(forward, model.ofe)


def decode(output, batch, native, collision=True):
    """Keep the author's camera-frame decoder, refinement and per-object NMS."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from graspnetAPI import GraspGroup
    from zerograsp.nets.utils import get_xyz_from_octree
    from zerograsp.utils.math import unnormalize_pts, rotation_6d_to_matrix
    from zerograsp.utils.collision_detector import ModelFreeCollisionDetector
    octree = output['octrees_out']
    cloud, ids = get_xyz_from_octree(octree, native.max_lod, nempty=True, return_batch=True)
    cloud = unnormalize_pts(cloud, batch[-2][0], native.grid_size, 1 << native.min_lod)
    normals = F.normalize(octree.normals[native.max_lod], dim=-1)
    signal = octree.features[native.max_lod]
    cloud = (cloud-normals*signal[:,:1])/1000
    ids = ids.cpu().numpy()
    objects = []
    detector = None
    if collision:
        depth_cloud = batch[4][0].reshape(-1,3)[::5].to(cloud.device)/1000
        detector = ModelFreeCollisionDetector(cloud.float(), normals.float(), depth_cloud.float())
    for index, label in enumerate(torch.unique(batch[3][0].labels, sorted=True)):
        selected = ids == index
        features = signal[selected,1:]
        quality = features[:,:1].cpu().numpy()
        rotation = rotation_6d_to_matrix(torch.cat((-features[:,5:8],features[:,2:5]),dim=-1)).cpu().numpy()
        array = np.concatenate((quality, np.clip(features[:,9:10].cpu().numpy()*.1,0,.1),
            np.full_like(quality,.02), np.clip(features[:,8:9].cpu().numpy()*.04,0,.04),
            rotation.reshape(-1,9), cloud[selected].cpu().numpy(), -np.ones_like(quality)), axis=1)
        if not np.isfinite(array).all():
            raise ValueError('Native ZeroGrasp decoder produced non-finite grasps')
        grasps = GraspGroup(array).sort_by_score()
        before, fallback, colliding = len(grasps), False, 0
        if detector is not None and before:
            mask, delta_width, depth = detector.detect(grasps)
            grasps.grasp_group_array[:,1] += delta_width
            grasps.grasp_group_array[:,3] = depth
            colliding = int(mask.sum())
            if (~mask).sum() > 0:
                grasps = grasps[~mask]
            else:
                # Retain and expose the author's fallback; these predictions
                # must not be described as collision-free.
                fallback = True
        grasps = grasps.nms(.03,30/180*np.pi).sort_by_score()
        objects.append((grasps.grasp_group_array, dict(instance_label=int(label), candidates=before,
            collision_flags=colliding, native_all_colliding_fallback=fallback, grasps=len(grasps))))
    return objects


def observation_batch(observation, native, training_normalization=False):
    """Author demo tensor construction, retaining separate released RLE masks."""
    import cv2
    import numpy as np
    import torch
    from ocnn.octree import Points
    from torchvision import transforms
    from zerograsp.utils.math import get_camera_rays, normalize_pts
    rgb, depth, masks, intrinsics = observation
    mean, std = ((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711)) if training_normalization else ((.485,.456,.406),(.229,.224,.225))
    rgb = transforms.Compose((transforms.Resize((480,640)),transforms.ToTensor(),transforms.Normalize(mean,std)))(rgb)
    rays = torch.from_numpy((get_camera_rays(intrinsics,depth.shape)*depth[:,:,None]).astype(np.float32))
    points, labels, planes, dilated = [], [], [], []
    for label, mask in masks.items():
        selected = rays[mask].reshape(-1,3)
        points.append(selected)
        labels.append(torch.full((len(selected),1),label,dtype=torch.long))
        planes.append(mask)
        dilated.append(cv2.dilate(mask.astype(np.float32),np.ones((5,5),np.uint8),iterations=1)>.5)
    points, labels = torch.cat(points), torch.cat(labels)
    z_min = (points[:,2].min()//native.grid_size)*native.grid_size-5*native.grid_size
    visible = Points(normalize_pts(points,z_min,native.grid_size,1<<native.min_lod), labels=labels)
    visible.clip()
    planes, dilated = np.stack(planes), np.stack(dilated)
    overlap = dilated.reshape(len(planes),-1) @ dilated.reshape(len(planes),-1).T
    np.fill_diagonal(overlap,False)
    neighbors = (overlap @ planes.reshape(len(planes),-1)).reshape(planes.shape)
    paired_masks = torch.from_numpy(np.stack((planes,neighbors),axis=-1)[None,None])
    return (rgb[None].cuda(), paired_masks.cuda(), torch.from_numpy(depth[None]).cuda(), [visible.to('cuda')],
            [rays], torch.from_numpy(intrinsics[None]).cuda(), z_min.reshape(1).cuda(), ['released_observation'])


def infer(config, out):
    import numpy as np
    import torch
    from ..integrations.zerograsp11b import selected_samples, write_observation, member_hashes
    from ..worker import overlay
    model, native, checkpoint = build(config, out)
    trained_here = bool(checkpoint.get('grasppanda'))
    model.cuda().eval()
    records = []
    for shard, sample, key, raw in selected_samples(config):
        print(f'Preparing RGB-D observation {key} from shard {shard}',flush=True)
        paths, observation = write_observation(raw, out/'inputs'/f'{shard:06d}'/key)
        intrinsics = observation[3]
        batch = observation_batch(observation, native, trained_here)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.no_grad():
            print('Reconstructing instance surfaces and grasp fields',flush=True)
            prediction = model.model(batch)
            print('Decoding grasps and applying native geometric filtering',flush=True)
            objects = decode(prediction, batch, native, config.collision_thresh > 0)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        array = np.concatenate([value for value,_ in objects], axis=0) if objects else np.empty((0,17))
        path = out/'predictions'/f'{shard:06d}'/f'{key}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        if not records:
            overlay(paths['rgb'], array, intrinsics, out/'preview.png')
        record = dict(shard=shard, sample=sample, sample_key=key, objects=[meta for _,meta in objects],
            grasps_after_collision=len(array), forward_decode_seconds=elapsed,
            prediction=str(path.relative_to(out)), prediction_sha256=digest(path), input_sha256=member_hashes(raw))
        records.append(record)
        print(json.dumps(record), flush=True)
    checkpoint_sha = digest(config.checkpoint)
    (out/'predictions/manifest.json').write_text(json.dumps(dict(config=config.to_dict(),
        checkpoint_sha256=checkpoint_sha, observations=records), indent=2)+'\n')
    return dict(stage='dataset_inference', dataset=config.dataset, method=config.method, frames=records,
        checkpoint_sha256=checkpoint_sha, torch=torch.__version__, gpu=torch.cuda.get_device_name(), ap=None,
        image_normalization='native_training_clip' if trained_here else 'author_demo_imagenet',
        protocol_note='Checkpoint-associated RGB normalization, native reconstruction, decoding and collision/NMS on '
                      'released training observations. Inputs include supplied visible instance masks and synthetic depth; '
                      'no target surface or grasp annotations are used. Native all-colliding fallbacks are reported per object. '
                      'This is not held-out grasp AP.')
