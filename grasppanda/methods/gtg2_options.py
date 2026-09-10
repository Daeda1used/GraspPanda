"""Candidate-graph geometry, encoder and ensemble training contracts."""
import math

GRAPH = dict(k=5, max_points=70, include_outside=True, encoding='binary',
             min_inside_train=50, min_inside_infer=70, bound_size=.03,
             gripper_depth=.06, gripper_height=.03, voxel_size=.005,
             plane_threshold=.002, dual_cloud=True, gpg_samples=1_000_000,
             gpg_threads=4, max_width=.1, nms_translation=.005,
             nms_rotation_degrees=1., train_depths=[.03, .04, .05],
             train_width_offsets=[.02, .04], candidate_limit=0)
ENCODER = dict(hidden=64, layers=3)
TRAINER = dict(folds=[0, 1, 2, 3, 4], scenes=list(range(100)), sampling='balanced', refresh_epochs=10,
               negative_per_class=50_000)


def schema(slot, choice):
    if slot == 'backbone' and choice in ('gtg_sage', 'gtg_gatv2'):
        common = {'hidden': ('int', 8, 512), 'layers': ('int', 1, 12)}
        return {**common, **({'aggregation': ('choice', ('max', 'mean', 'sum'))}
                if choice == 'gtg_sage' else {
                    'heads': ('int', 1, 16), 'dropout': ('float', 0, .8),
                    'add_self_loops': ('bool',), 'share_weights': ('bool',)})}
    if slot != 'crop' or choice != 'grasp_graph': return {}
    return {'k': ('int', 1, 64), 'max_points': ('int', 8, 512),
            'include_outside': ('bool',), 'encoding': ('choice', ('binary', 'onehot')),
            'min_inside_train': ('int', 2, 512), 'min_inside_infer': ('int', 2, 512),
            'bound_size': ('float', .001, .1), 'gripper_depth': ('float', .02, .1),
            'gripper_height': ('float', .005, .05), 'voxel_size': ('float', .001, .02),
            'plane_threshold': ('float', .0001, .02), 'dual_cloud': ('bool',),
            'gpg_samples': ('int', 1, 1_000_000), 'gpg_threads': ('int', 1, 64),
            'max_width': ('float', .01, .1), 'nms_translation': ('float', 0, .1),
            'nms_rotation_degrees': ('float', 0, 90),
            'train_depths': ('float_list', 1, 8, .005, .08),
            'train_width_offsets': ('float_list', 0, 8, 0, .08),
            'candidate_limit': ('int', 0, 1_000_000)}


def resolved(modules):
    from grasppanda.module_options import unpack, validate_options
    unknown = set(modules) - {'backbone', 'crop'}
    if unknown: raise ValueError('Unknown GtG2 component slots: ' + str(sorted(unknown)))
    encoder, enc = unpack(modules.get('backbone', 'gtg_sage'))
    crop, graph = unpack(modules.get('crop', 'grasp_graph'))
    encoder = 'gtg_sage' if encoder == 'upstream' else encoder
    crop = 'grasp_graph' if crop == 'upstream' else crop
    if encoder not in ('gtg_sage', 'gtg_gatv2') or crop != 'grasp_graph':
        raise ValueError('GtG2 requires a registered graph encoder and grasp_graph crop')
    validate_options('gtg2', 'backbone', encoder, enc)
    validate_options('gtg2', 'crop', crop, graph)
    enc = {**ENCODER, **({'aggregation': 'max'} if encoder == 'gtg_sage' else
           dict(heads=4, dropout=0., add_self_loops=True, share_weights=False)), **enc, 'type': encoder}
    graph = {**GRAPH, **graph}
    if min(graph['max_points'], graph['min_inside_train'], graph['min_inside_infer']) <= graph['k']:
        raise ValueError('Graph point caps and minimum inside counts must exceed k')
    if encoder == 'gtg_gatv2' and enc['hidden'] % enc['heads']:
        raise ValueError('Graph hidden channels must be divisible by GATv2 heads')
    return enc, graph


def validate_training(config):
    options = config.trainer
    if set(options) - set(TRAINER): raise ValueError('Unknown GtG2 trainer parameters')
    folds = options.get('folds', TRAINER['folds'])
    if (not isinstance(folds, list) or not 1 <= len(folds) <= 10 or
            any(type(f) is not int or not 0 <= f <= 9 for f in folds) or len(set(folds)) != len(folds)):
        raise ValueError('GtG2 folds must be distinct scene residues from 0 to 9')
    if options.get('sampling', 'balanced') not in ('all', 'balanced'):
        raise ValueError('GtG2 sampling must be all or balanced')
    scenes = options.get('scenes', TRAINER['scenes'])
    if (not isinstance(scenes, list) or not scenes or
            any(type(s) is not int or not 0 <= s < 100 for s in scenes) or len(set(scenes)) != len(scenes)):
        raise ValueError('GtG2 trainer scenes must be distinct training scene IDs from 0 to 99')
    for fold in folds:
        if not any(s % 10 == fold for s in scenes) or not any(s % 10 != fold for s in scenes):
            raise ValueError('Each GtG2 fold needs separate training and validation scenes')
    for name, maximum in (('refresh_epochs', 10000), ('negative_per_class', 10_000_000)):
        value = options.get(name, TRAINER[name])
        if type(value) is not int or not 1 <= value <= maximum: raise ValueError('Invalid GtG2 ' + name)
    if options and config.action != 'train': raise ValueError('GtG2 trainer settings apply to epoch training')
    if config.action in ('train', 'train_short') and config.batch_size < 2:
        raise ValueError('GtG2 graph-level batch normalization needs batch_size >= 2')


def validate_config(config):
    if not isinstance(config.trainer, dict): raise ValueError('trainer must be a mapping')
    if not isinstance(config.modules, dict): raise ValueError('modules must be a mapping')
    resolved(config.modules)
    validate_training(config)
    if config.checkpoint_policy != 'strict':
        raise ValueError('GtG2 checkpoints require matching graph and encoder settings; train a new composition with an empty checkpoint')
    if config.workspace != 'official_gt_workspace':
        raise ValueError('GtG2 candidate generation requires official_gt_workspace')
    if config.num_points != 15000:
        raise ValueError('GtG2 uses crop.max_points per graph and crop.gpg_samples for proposals; leave num_points at its default')
    if config.action == 'train':
        if config.scene != 0 or config.frames != 1:
            raise ValueError('GtG2 training selects scenes through trainer.scenes; leave scene=0 and frames=1')
        if not config.label_root:
            raise ValueError('GtG2 training requires label_root pointing to prepared graph inputs')


def validate_augmentation(options):
    if set(options) - {'mode', 'half_turn_probability', 'point_dropout'}:
        raise ValueError('GtG2 augmentation accepts mode, half_turn_probability and point_dropout')
    mode = options.get('mode', 'custom')
    if mode not in ('native', 'none', 'custom'): raise ValueError('Unknown graph augmentation mode')
    if mode != 'custom' and set(options) - {'mode'}: raise ValueError('Parameters require custom augmentation')
    for key, limit in (('half_turn_probability', 1.), ('point_dropout', .8)):
        value = options.get(key, 0.)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('Invalid graph augmentation ' + key)
