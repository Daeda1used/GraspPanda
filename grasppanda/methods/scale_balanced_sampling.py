"""Object-balanced inference sampling from the author's independent DSN predictions."""
from pathlib import Path

SCHEMA = {
    'seed_count': ('int', 32, 4096),
    'iterations': ('int', 1, 100),
    'epsilon': ('float', .001, .5),
    'sigma': ('float', .001, .5),
    'cluster_seeds': ('int', 1, 200),
    'subsample_factor': ('int', 1, 20),
    'min_cluster_points': ('int', 1, 1000),
    'foreground_threshold': ('float', .05, .95),
    'empty_policy': ('choice', ('scene_fps', 'error')),
    'segmentation_checkpoint': ('checkpoint_path',),
}


def enabled(config):
    from ..module_options import unpack
    return config.method == 'scale_balanced_grasp' and unpack(config.modules.get('sampling', 'upstream'))[0] == 'object_balanced'


def options(config):
    from ..module_options import unpack
    return unpack(config.modules.get('sampling', 'upstream'))[1]


def validate(config):
    if not enabled(config): return
    if config.action not in ('infer', 'evaluate'):
        raise ValueError('Object-balanced sampling is an inference policy; use sampling: upstream for training')
    if config.camera != 'realsense' and not options(config).get('segmentation_checkpoint'):
        raise ValueError('The registered DSN weights use RealSense; provide segmentation_checkpoint for your camera')


def checkpoint(config):
    from ..config import ROOT
    from ..weights import component_records
    from ..jobs import digest
    value = options(config).get('segmentation_checkpoint')
    record = component_records()['scale_balanced_dsn']
    path = (ROOT/Path(value or record['path']).expanduser()).resolve()
    if not path.is_file():
        raise ValueError('OBS segmentation checkpoint is missing; run ./panda component-weights scale_balanced_dsn or set modules.sampling.segmentation_checkpoint')
    sha = digest(path)
    if not value and sha != record['sha256']: raise ValueError('Registered DSN checkpoint checksum mismatch')
    return path, sha


def balanced(end, *, seed_count=1024, empty_policy='scene_fps', **_):
    import torch
    from pointnet2_utils import furthest_point_sample
    xyz, labels = end['point_clouds'], end['seed_cluster']
    features = end['up_sample_features']
    if labels.shape != xyz.shape[:2] or features.shape != (xyz.shape[0], 256, xyz.shape[1]):
        raise ValueError('OBS requires segmentation and 256-channel features aligned to every input point')
    indices, details = [], []
    for points, ids in zip(xyz, labels):
        objects = torch.unique(ids[ids > 0])
        fallback = not len(objects)
        if fallback:
            if empty_policy == 'error': raise ValueError('DSN found no foreground objects for object-balanced sampling')
            selected = furthest_point_sample(points[None].contiguous(), seed_count)[0].long()
            allocations = []
        else:
            if len(objects) > seed_count: raise ValueError('OBS seed_count must be at least the number of predicted objects')
            allocations = [seed_count//len(objects)]*len(objects)
            allocations[-1] += seed_count % len(objects)
            selected = []
            for label, count in zip(objects, allocations):
                rows = torch.where(ids == label)[0]
                local = furthest_point_sample(points[rows][None].contiguous(), count)[0].long()
                selected.append(rows[local])
            selected = torch.cat(selected)
        indices.append(selected)
        details.append(dict(objects=len(objects), seeds=seed_count, allocation=allocations, scene_fps_fallback=fallback))
    indices = torch.stack(indices)
    end['fp2_inds_fps'] = end['fp2_inds']
    end['fp2_inds'] = indices.int()
    end['fp2_xyz'] = torch.gather(xyz, 1, indices[..., None].expand(-1, -1, 3))
    end['fp2_features'] = torch.gather(features, 2, indices[:, None].expand(-1, 256, -1))
    end['object_sampling'] = details
    return end


def install(stage, parameters):
    from functools import partial
    from types import FunctionType, MethodType
    original = stage.forward.__func__
    namespace = {**original.__globals__, 'ObjectBalanceSampling': partial(balanced, **parameters)}
    stage.forward = MethodType(FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__), stage)
    stage.obs = True


class Segmenter:
    def __init__(self, config):
        import torch
        from models.dsn import DSN
        self.path, self.sha256 = checkpoint(config)
        state = torch.load(self.path, map_location='cpu', weights_only=True)
        self.model = DSN(is_training=False)
        self.model.load_state_dict(state.get('model_state_dict', state), strict=True)
        self.model.cuda().eval()
        self.options = options(config)

    def __call__(self, inputs):
        import torch
        from segmentation_loss import GaussianMeanShift
        xyz = inputs['point_clouds']
        end = self.model({'point_clouds': xyz})
        if not torch.isfinite(end['foreground_logits']).all() or not torch.isfinite(end['center_offsets']).all():
            raise ValueError('DSN produced non-finite segmentation predictions')
        foreground = end['foreground_logits'].softmax(1)[:, 1] > self.options.get('foreground_threshold', .5)
        centers = xyz + end['center_offsets'].transpose(1, 2)
        predictions = []
        for points, mask in zip(centers, foreground):
            labels = torch.zeros(len(points), dtype=torch.long, device=points.device)
            if mask.any():
                factor = self.options.get('subsample_factor', 5)
                available = len(torch.unique(points[mask][::factor], dim=0))
                mean_shift = GaussianMeanShift(max_iters=self.options.get('iterations', 10),
                    epsilon=self.options.get('epsilon', .05), sigma=self.options.get('sigma', .02),
                    num_seeds=min(self.options.get('cluster_seeds', 50), available), subsample_factor=factor)
                clusters = mean_shift.mean_shift_smart_init(points[mask]) + 1
                predicted = torch.zeros_like(labels); predicted[mask] = clusters
                label = 1
                for cluster in mean_shift.uniq_labels + 1:
                    rows = predicted == cluster
                    if int(rows.sum()) >= self.options.get('min_cluster_points', 10):
                        labels[rows] = label; label += 1
            predictions.append(labels)
        inputs['seed_cluster'] = torch.stack(predictions)
