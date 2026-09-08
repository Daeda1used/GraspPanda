"""Loss substitutions with native target selection, units and reduction contracts."""
import math


CLASSIFICATION = ('cross_entropy', 'focal', 'poly1', 'asl')
REGRESSION = ('l1', 'mse', 'smooth_l1', 'huber', 'charbonnier')


PARAMETERS = {
    'upstream': {}, 'cross_entropy': {'label_smoothing': (0, .5)},
    'focal': {'gamma': (0, 8), 'alpha': (0, 1)}, 'poly1': {'epsilon': (-1, 10)},
    'asl': {'gamma_pos': (0, 8), 'gamma_neg': (0, 8), 'label_smoothing': (0, .5)},
    'l1': {}, 'mse': {}, 'smooth_l1': {'beta': (0, 10)},
    'huber': {'delta': (1e-6, 10)}, 'charbonnier': {'epsilon': (1e-6, 1)},
}

def is_classification(method, term):
    return term in ('objectness', 'angle') or (method == 'finegrasp' and term in ('depth', 'score'))


def choices(term, method=None):
    return ('upstream',) + (CLASSIFICATION if is_classification(method, term) else REGRESSION)


def validate(method, functions):
    from .training_options import LOSS_TERMS
    from .module_options import unpack
    if not isinstance(functions, dict) or set(functions) - set(LOSS_TERMS.get(method, {})):
        raise ValueError('loss.functions must map registered loss terms to formulations')
    for term, value in functions.items():
        kind, options = unpack(value)
        if kind not in choices(term, method) or set(options) - set(PARAMETERS[kind]):
            raise ValueError(f'{method}/{term}: unsupported loss formulation or parameters')
        if 'alpha' in options and term != 'objectness':
            raise ValueError('Focal alpha is registered only for binary objectness')
        for key, value in options.items():
            low, high = PARAMETERS[kind][key]
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid {term}/{kind} loss parameter: {key}')


def classification(logits, target, kind, options):
    """Return one value per item; input class dimension is last."""
    import torch
    import torch.nn.functional as F
    if kind == 'asl':
        # Single-label ASL: preserve the author's softmax/smoothing objective.
        # expm1 and a positive floor keep fractional powers differentiable when
        # finite logits saturate softmax probabilities to exactly zero or one.
        log_probability = F.log_softmax(logits, dim=-1)
        positive = F.one_hot(target, logits.shape[-1]).to(logits.dtype)
        base = torch.where(positive.bool(), -log_probability.expm1(), log_probability.exp())
        gamma = positive * options.get('gamma_pos', 0) + (1 - positive) * options.get('gamma_neg', 4)
        weight = base.clamp_min(torch.finfo(logits.dtype).tiny).pow(gamma)
        smoothing = options.get('label_smoothing', .1)
        target_probability = positive * (1 - smoothing) + smoothing / logits.shape[-1]
        return -(target_probability * log_probability * weight).sum(-1)
    ce = F.cross_entropy(logits, target, reduction='none',
                         label_smoothing=options.get('label_smoothing', 0))
    if kind == 'cross_entropy':
        return ce
    one_minus_probability = -(-ce).expm1()
    if kind == 'poly1':
        return ce + options.get('epsilon', 1.) * one_minus_probability
    if kind == 'focal':
        value = ce * one_minus_probability.clamp_min(torch.finfo(ce.dtype).tiny).pow(options.get('gamma', 2.))
        if 'alpha' in options:
            value = value * (target * options['alpha'] + (1 - target) * (1 - options['alpha']))
        return value
    raise ValueError('Unknown classification loss')


def regression(error, kind, options):
    import torch.nn.functional as F
    if kind == 'l1': return error.abs()
    if kind == 'mse': return error.square()
    if kind == 'smooth_l1':
        return F.smooth_l1_loss(error, error.new_zeros(error.shape), beta=options.get('beta', 1.), reduction='none')
    if kind == 'huber':
        return F.huber_loss(error, error.new_zeros(error.shape), delta=options.get('delta', 1.), reduction='none')
    if kind == 'charbonnier':
        epsilon = options.get('epsilon', .001)
        return (error.square() + epsilon ** 2).sqrt() - epsilon
    raise ValueError('Unknown regression loss')


def targets(end, method, term):
    """Return prediction/target, optional mask, normalization and denominator offset."""
    if method == 'finegrasp':
        if term == 'objectness':
            return end['objectness_score'], end['objectness_label'].long(), None, 1., 0.
        if term == 'graspness':
            return end['graspness_score'].squeeze(1), end['graspness_label'].squeeze(-1), end['objectness_label'].bool(), 1., 0.
        if term == 'view':
            return end['view_score'], end['batch_grasp_view_graspness'], None, 1., 0.
        mask = end['batch_valid_mask']
        if term in ('angle', 'depth', 'score'):
            label = (end['batch_grasp_score'] * 10 / 2).long() if term == 'score' else end['batch_grasp_rotations' if term == 'angle' else 'batch_grasp_depth'].long()
            return end['grasp_' + term + '_pred'], label, mask, 1., 0.
        return end['grasp_width_pred'].squeeze(1), end['batch_grasp_width'] * 10, mask, 1., 0.
    sparse = method == 'graspness'
    if term == 'objectness':
        target = end['objectness_label'].long()
        if not sparse: target = target.gather(1, end['fp2_inds'].long())
        return end['objectness_score'], target, None, 1., 0.
    if sparse:
        if term == 'graspness':
            return end['graspness_score'].squeeze(1), end['graspness_label'].squeeze(-1), end['objectness_label'].bool(), 1., 0.
        if term == 'view':
            return end['view_score'], end['batch_grasp_view_graspness'], None, 1., 0.
        if term == 'score':
            return end['grasp_score_pred'], end['batch_grasp_score'], None, 1., 0.
        if term == 'width':
            return end['grasp_width_pred'], end['batch_grasp_width'] * 10, end['batch_grasp_score'] > 0, 1., 0.
    from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_BAD
    object_mask = end['objectness_label'].gather(1, end['fp2_inds'].long()).bool()
    if term == 'view':
        label = end['batch_grasp_view_label']
        return end['view_score'], label, object_mask[..., None].expand_as(label), 1., 0.
    labels = end['batch_grasp_label']
    angle = labels.argmax(2, keepdim=True)
    quality = labels.gather(2, angle).squeeze(2)
    mask = object_mask[..., None] & (quality > THRESH_BAD)
    if term == 'angle':
        return end['grasp_angle_cls_pred'], angle.squeeze(2), mask, 1., 1e-6
    predicted = end[{'score': 'grasp_score_pred', 'width': 'grasp_width_pred',
                     'tolerance': 'grasp_tolerance_pred'}[term]].gather(1, angle.transpose(1, 2)).squeeze(1)
    if term == 'score':
        return predicted, quality, mask, 1., 1e-6
    if term == 'width':
        label = end['batch_grasp_offset'][..., 2].gather(2, angle).squeeze(2)
        return predicted, label, mask, GRASP_MAX_WIDTH, 1e-6
    label = end['batch_grasp_tolerance'].gather(2, angle).squeeze(2)
    return predicted, label, mask, GRASP_MAX_TOLERANCE, 1e-6


def replace_losses(end_points, config):
    from .module_options import unpack
    from .training_options import LOSS_TERMS
    for term, value in config.loss.get('functions', {}).items():
        kind, options = unpack(value)
        if kind == 'upstream': continue
        prediction, target, mask, scale, offset = targets(end_points, config.method, term)
        if is_classification(config.method, term):
            channels = prediction.shape[1]
            prediction = prediction.movedim(1, -1).reshape(-1, channels)
            target = target.reshape(-1)
            if mask is not None:
                prediction, target = prediction[mask.reshape(-1)], target[mask.reshape(-1)]
            values = classification(prediction, target, kind, options)
        else:
            error = (prediction - target) / scale
            if mask is not None: error = error[mask]
            values = regression(error, kind, options)
        # Native valid-item reductions are retained. An empty selected set has
        # a graph-connected zero instead of an undefined mean.
        # Baseline grasp heads accumulate the validity count in float32,
        # including their epsilon, even if model predictions use float64.
        denominator = mask.float().sum() + offset if offset else max(values.numel(), 1)
        end_points[LOSS_TERMS[config.method][term][0]] = values.sum() / denominator
    return end_points
