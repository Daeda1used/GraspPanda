"""Native SPGrasp construction and RGB-only temporal prediction boundaries."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import sys

import torch


@dataclass(frozen=True)
class Architecture:
    resolution: int = 512
    memory_frames: int = 7
    memory_layers: int = 4
    memory_heads: int = 1
    memory_dropout: float = .1
    image_drop_path: float = .1
    trainable_backbone_blocks: int = -1

    def __post_init__(self):
        for name, allowed in [('resolution', (256, 512, 768, 1024)), ('memory_heads', (1, 2, 4, 8))]:
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f'Invalid SPGrasp {name}: choose from {allowed}')
        for name, low, high in [('memory_frames', 2, 16), ('memory_layers', 1, 8), ('trainable_backbone_blocks', -1, 24)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'SPGrasp {name} must be an integer in [{low}, {high}]')
        for name in ('memory_dropout', 'image_drop_path'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .5:
                raise ValueError(f'SPGrasp {name} must be finite and in [0, .5]')

    def to_dict(self):
        return asdict(self)


def create_model(source, architecture=Architecture()):
    """Construct the author model from the reviewed source overlay."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    path = source / 'spgrasp/configs/sam2.1_training/sam2.1_hiera_b+_Graspnet_finetune.yaml'
    cfg = OmegaConf.load(path)
    cfg.scratch.resolution = architecture.resolution
    cfg.trainer.model.num_maskmem = architecture.memory_frames
    cfg.trainer.model.memory_attention.num_layers = architecture.memory_layers
    layer = cfg.trainer.model.memory_attention.layer
    layer.dropout = architecture.memory_dropout
    for field in ('self_attention', 'cross_attention'):
        layer[field].num_heads = architecture.memory_heads
        layer[field].dropout = architecture.memory_dropout
    cfg.trainer.model.image_encoder.trunk.drop_path_rate = architecture.image_drop_path
    # Training prompts represent interaction. Ground-truth grasp maps may never
    # bypass the decoder through the author's mask-input shortcut.
    cfg.trainer.model.prob_to_use_pt_input_for_train = 1.
    model = instantiate(cfg.trainer.model)
    origin = Path(sys.modules[type(model).__module__].__file__).resolve()
    if not origin.is_relative_to(source):
        raise RuntimeError('SPGrasp imports conflict with another method; use a separate worker process')
    model._grasppanda_architecture = architecture
    set_training(model, True)
    return model, cfg


def set_training(model, enabled):
    """Freeze selected Hiera blocks, including their stochastic behavior."""
    model.train(enabled)
    architecture = model._grasppanda_architecture
    count = architecture.trainable_backbone_blocks
    if count == -1:
        return model
    encoder = model.image_encoder
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    blocks = encoder.trunk.blocks
    if count:
        for block in blocks[-count:]:
            block.train(enabled)
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    return model


def initialize_sam2(model, checkpoint):
    """Explicit partial initialization from SAM2, not a trained grasp checkpoint."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
        raise ValueError('Expected the registered SAM2 initialization checkpoint')
    source, current = payload['model'], model.state_dict()
    compatible = {key: value for key, value in source.items()
                  if key in current and isinstance(value, torch.Tensor) and value.shape == current[key].shape}
    encoder = [key for key in current if key.startswith('image_encoder.')]
    if not encoder or any(key not in compatible for key in encoder):
        raise ValueError('SAM2 initialization does not match the complete selected Hiera image encoder')
    model.load_state_dict(compatible, strict=False)
    return {'kind': 'SAM2 initialization', 'loaded_keys': sorted(compatible),
            'initialized_keys': sorted(set(current) - set(compatible)),
            'unused_keys': sorted(set(source) - set(compatible))}


def validate_width_calibration(targets, positive_weight=10.):
    """Verify every training frame uses the recorded native width-loss weight.

    The released loss caps its width positive-class weight at ten. Decoding a
    continuous width requires a fixed, known logit correction. A batch that does
    not reach that cap must use a different explicit loss/normalization protocol.
    """
    if targets.ndim != 5 or targets.shape[2] != 5 or not torch.isfinite(targets).all():
        raise ValueError('Expected finite planar training targets [T,O,5,H,W]')
    if type(positive_weight) not in (int, float) or positive_weight != 10:
        raise ValueError('The released SPGrasp width loss uses a maximum positive weight of ten')
    widths = targets[:, :, 3]
    if (widths < 0).any() or (widths > 1).any():
        raise ValueError('Normalized width targets must lie in [0,1]')
    for frame in widths:
        positives = frame.sum().float()
        ratio = ((frame.numel() - positives) / positives.clamp_min(1)).clamp_max(10)
        if float(ratio) != positive_weight:
            raise ValueError('Native width-loss weight varies for this batch; change the explicit width normalization or objective before training')
    return float(positive_weight)


def save_checkpoint(model, path, width_scale, metadata=None, width_positive_weight=10.):
    if type(width_scale) not in (int, float) or not math.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('Checkpoint width scale must be positive and finite')
    if type(width_positive_weight) not in (int, float) or not math.isfinite(width_positive_weight) or width_positive_weight <= 0:
        raise ValueError('Checkpoint width positive-class weight must be positive and finite')
    json.dumps(metadata or {}, allow_nan=False)
    torch.save({'format': 'grasppanda.spgrasp.v1', 'architecture': model._grasppanda_architecture.to_dict(),
                'width_scale_pixels': width_scale, 'width_positive_weight': width_positive_weight,
                'model': model.state_dict(),
                'metadata': metadata or {}}, path)


def load_checkpoint(source, path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != 'grasppanda.spgrasp.v1':
        raise ValueError('Expected a GraspPanda SPGrasp checkpoint with architecture and width units')
    scale = payload.get('width_scale_pixels')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('SPGrasp checkpoint has invalid width units')
    alpha = payload.get('width_positive_weight')
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('SPGrasp checkpoint is missing its width-loss calibration')
    architecture = Architecture(**payload['architecture'])
    model, cfg = create_model(source, architecture)
    model.load_state_dict(payload['model'], strict=True)
    model._grasppanda_width_positive_weight = alpha
    set_training(model, False)
    return model.to(device), cfg, scale


def predict_sequence(model, images, prompts):
    """Stream RGB frames with explicit first-frame prompts; never accept targets.

    Returns raw CPU [T,O,5,R,R] logits. The caller selects inference precision and
    decodes width/angle into original-image coordinates using the recorded units.
    """
    if model.training:
        raise ValueError('Temporal inference requires evaluation mode')
    resolution = model._grasppanda_architecture.resolution
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, resolution, resolution) or len(images) < 1:
        raise ValueError('Expected normalized RGB sequence [T,3,R,R]')
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError('Non-finite RGB sequence')
    if not isinstance(prompts, dict) or set(prompts) != {'point_coords', 'point_labels'}:
        raise ValueError('Temporal prediction requires explicit point/box prompts')
    coords, labels = prompts['point_coords'], prompts['point_labels']
    if coords.ndim != 3 or coords.shape[-1] != 2 or coords.shape[:2] != labels.shape or coords.shape[0] < 1:
        raise ValueError('Invalid native prompt tensor shapes')
    if not coords.is_floating_point() or labels.dtype not in (torch.int32, torch.int64) or not torch.isfinite(coords).all() or not ((labels >= -1) & (labels <= 3)).all():
        raise ValueError('Invalid native prompt coordinates or labels')
    if not (labels >= 0).any(dim=1).all():
        raise ValueError('Every tracked object needs an explicit prompt')
    active = coords[labels >= 0]
    if (active < -.5).any() or (active >= resolution).any():
        raise ValueError('Native prompt coordinates lie outside the resized image')
    if images.device != coords.device or images.device != labels.device:
        raise ValueError('Images and prompts must use the same device')
    objects = len(coords)
    memory = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
    predictions = []
    with torch.no_grad():
        for frame, image in enumerate(images):
            features = model.forward_image(image[None])
            _, feats, positions, sizes = model._prepare_backbone_features(features)
            ids = torch.zeros(objects, device=images.device, dtype=torch.long)
            output = model.track_step(
                frame_idx=frame, is_init_cond_frame=frame == 0,
                current_vision_feats=[value[:, ids] for value in feats],
                current_vision_pos_embeds=[value[:, ids] for value in positions],
                feat_sizes=sizes, point_inputs=prompts if frame == 0 else None,
                mask_inputs=None, gt_masks=None, frames_to_add_correction_pt=[],
                output_dict=memory, num_frames=len(images),
            )
            memory['cond_frame_outputs' if frame == 0 else 'non_cond_frame_outputs'][frame] = output
            predictions.append(output['multistep_pred_multimasks_high_res'][-1].detach().float().cpu())
            # Spatial memories and object pointers have separate native horizons.
            # Retain both, plus the initial conditioning frame, at stride one.
            horizon = max(model.num_maskmem, model.max_obj_ptrs_in_encoder)
            for old in list(memory['non_cond_frame_outputs']):
                if old < frame - horizon:
                    del memory['non_cond_frame_outputs'][old]
    return torch.stack(predictions)
