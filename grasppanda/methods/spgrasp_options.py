"""JSON-safe configuration contracts for prompted planar sequences."""
import math

BACKBONE = {'resolution': ('int', 256, 1024), 'drop_path': ('float', 0, .5), 'trainable_blocks': ('int', -1, 24)}
MEMORY = {'frames': ('int', 2, 16), 'layers': ('int', 1, 8), 'heads': ('int', 1, 8), 'dropout': ('float', 0, .5)}
TRAINER = dict(objects=3, box_probability=.5, correction_clicks=7, conditioning_frames=2, correction_frames=2)
PLANAR = dict(width_scale_pixels=1280., score_threshold=.5, semantic_threshold=.5, max_grasps=10, min_distance=20.)


def finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def architecture(modules):
    from grasppanda.module_options import unpack
    _, backbone = unpack(modules.get('backbone', 'upstream'))
    _, memory = unpack(modules.get('memory', 'upstream'))
    resolution, heads = backbone.get('resolution', 512), memory.get('heads', 1)
    if resolution not in (256, 512, 768, 1024) or heads not in (1, 2, 4, 8):
        raise ValueError('SPGrasp resolution must be 256/512/768/1024 and memory heads must be 1/2/4/8')
    return dict(resolution=resolution, image_drop_path=backbone.get('drop_path', .1),
                trainable_backbone_blocks=backbone.get('trainable_blocks', -1),
                memory_frames=memory.get('frames', 7), memory_layers=memory.get('layers', 4),
                memory_heads=heads, memory_dropout=memory.get('dropout', .1))


def validate(config):
    if not isinstance(config.prompts, list) or not isinstance(config.planar, dict):
        raise ValueError('prompts must be an object list; planar must be a mapping')
    if config.method != 'spgrasp':
        if config.prompts or config.planar:
            raise ValueError('Planar settings and prompts are registered for SPGrasp only')
        return
    if config.workspace != 'native_demo' or config.collision_thresh != 0:
        raise ValueError('SPGrasp uses native_demo and collision_thresh: 0; it has no 3D collision stage')
    if config.checkpoint_policy != 'strict':
        raise ValueError('SPGrasp grasp checkpoints require strict loading; SAM2 initialization is separate')
    if not 1 <= config.frames <= 256 or config.frame + config.frames > 256:
        raise ValueError('A prompted sequence must stay within one scene')
    architecture(config.modules)
    if len(config.prompts) > 8:
        raise ValueError('SPGrasp accepts at most eight prompted objects')
    seen = set()
    for obj in config.prompts:
        if not isinstance(obj, dict) or set(obj) - {'id', 'box', 'points'}:
            raise ValueError('Object prompts accept id, box and points')
        oid = obj.get('id')
        if type(oid) is not int or oid < 0 or oid in seen:
            raise ValueError('Prompt IDs must be unique nonnegative integers')
        seen.add(oid)
        box, points = obj.get('box'), obj.get('points', [])
        if box is None and not points:
            raise ValueError('Each object needs a box or labelled points')
        if 'box' in obj and (not isinstance(box, list) or len(box) != 4 or not all(finite(v, 0, 100000) for v in box) or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('A prompt box is [x0,y0,x1,y1] in original RGB pixels')
        if not isinstance(points, list) or len(points) > 64:
            raise ValueError('Provide at most 64 points per object')
        for point in points:
            if not isinstance(point, list) or len(point) != 3 or not all(finite(v, 0, 100000) for v in point[:2]) or type(point[2]) is not int or point[2] not in (0, 1):
                raise ValueError('Each prompt point is [x,y,label], with label 0 or 1')
    if set(config.planar) - set(PLANAR):
        raise ValueError('Unknown planar output setting')
    for key, value in config.planar.items():
        bounds = (1, 10000) if key == 'width_scale_pixels' else (0, 2000) if key == 'min_distance' else (1, 100) if key == 'max_grasps' else (0, 1)
        if not finite(value, *bounds) or key == 'max_grasps' and type(value) is not int:
            raise ValueError('Invalid planar setting: ' + key)
    if not isinstance(config.trainer, dict) or set(config.trainer) - set(TRAINER):
        raise ValueError('Unknown SPGrasp trainer setting')
    if config.action != 'train_short':
        if config.trainer:
            raise ValueError('SPGrasp trainer settings apply to short training only')
        return
    if config.prompts:
        raise ValueError('Training simulates prompts from instance labels; explicit prompts apply to inference')
    settings = {**TRAINER, **config.trainer}
    for key, low, high in [('objects',1,8),('correction_clicks',0,7),('conditioning_frames',1,4),('correction_frames',1,4)]:
        if type(settings[key]) is not int or not low <= settings[key] <= high:
            raise ValueError('Invalid SPGrasp trainer setting: ' + key)
    if not finite(settings['box_probability'],0,1):
        raise ValueError('box_probability must be in [0,1]')
    if config.frames < 2 or settings['conditioning_frames'] > settings['correction_frames'] or settings['correction_frames'] > config.frames:
        raise ValueError('Training requires at least two frames and conditioning_frames <= correction_frames <= frames')
    if not 1 <= config.batch_size <= 4 or config.frame + config.frames * config.batch_size > 256:
        raise ValueError('Training uses one to four consecutive clips within the selected scene')


def validate_training(config):
    for name in ('loss', 'augmentation'):
        value = getattr(config, name)
        if not isinstance(value, dict) or value and config.action != 'train_short':
            raise ValueError(f'SPGrasp {name} settings apply to short training only')
    terms = {'position', 'angle', 'width', 'semantic'}
    loss = config.loss
    if set(loss) - {'weights'} or not isinstance(loss.get('weights', {}), dict):
        raise ValueError('SPGrasp currently accepts loss.weights for position, angle, width and semantic')
    weights = loss.get('weights', {})
    if set(weights) - terms or not all(finite(v,0,1000) for v in weights.values()):
        raise ValueError('Invalid SPGrasp loss weights')
    if all(weights.get(key, 1) == 0 for key in terms):
        raise ValueError('At least one SPGrasp loss must be active')
    aug = config.augmentation
    if set(aug) - {'mode', 'brightness', 'contrast', 'saturation', 'grayscale', 'consistent'}:
        raise ValueError('SPGrasp exposes photometric augmentation only')
    mode = aug.get('mode', 'native')
    if mode not in ('native','none','custom') or mode != 'custom' and set(aug) - {'mode'}:
        raise ValueError('Custom augmentation fields require mode: custom')
    for key in ('brightness','contrast','saturation','grayscale'):
        if key in aug and not finite(aug[key],0,1):
            raise ValueError('Invalid photometric augmentation: '+key)
    if 'consistent' in aug and type(aug['consistent']) is not bool:
        raise ValueError('consistent must be boolean')
