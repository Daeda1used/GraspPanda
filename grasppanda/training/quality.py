"""Quality-target objectives with explicit grasp supervision and score mappings."""

METHODS = ('graspnet_baseline', 'pointnet2_upgrade', 'scale_balanced_grasp', 'graspness')
LOSSES = ('quality_bce', 'varifocal', 'mal')
SELECTION = {'supervision': ('choice', 'native', 'all_angles'),
             'normalization': ('choice', 'native', 'positive')}
PARAMETERS = {
    'quality_bce': {**SELECTION},
    'varifocal': {'alpha': (0, 10), 'gamma': (0, 8), **SELECTION},
    'mal': {'alpha': (0, 10), 'gamma': (1e-6, 8), **SELECTION},
}


def elementwise(logits, quality, kind, options):
    """DEIM-style detached focusing weights; quality replaces detection IoU."""
    import torch
    from torch.nn import functional as F
    if logits.dtype in (torch.float16, torch.bfloat16): logits = logits.float()
    quality = quality.detach().to(logits.dtype)
    positive = (quality > 0).to(logits.dtype)
    probability = logits.detach().sigmoid()
    gamma = options.get('gamma', 2.)
    if kind == 'quality_bce':
        weight, target = None, quality
    elif kind == 'varifocal':
        weight = quality + options.get('alpha', .2) * probability.pow(gamma) * (1 - positive)
        target = quality
    elif kind == 'mal':
        weight = positive + options.get('alpha', 1.) * probability.pow(gamma) * (1 - positive)
        target = quality.pow(gamma)
    else:
        raise ValueError('Unknown quality objective')
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction='none')


def replace(end, config, kind, options):
    import torch
    from .options import LOSS_TERMS
    if 'grasp_score_logits' not in end:
        raise ValueError('Quality objectives require the quality_residual prediction head')
    logits = end['grasp_score_logits']
    offset = 0.
    if config.method == 'graspness':
        target = end['batch_grasp_score']
        weights = torch.ones_like(target)
    else:
        from loss_utils import THRESH_BAD
        labels = end['batch_grasp_label']
        object_mask = end['objectness_label'].gather(1, end['fp2_inds'].long()).bool()
        if options.get('supervision', 'native') == 'all_angles':
            target, logits = labels, logits.movedim(1, 2)
            weights = object_mask[..., None, None].expand_as(target).float()
        else:
            angle = labels.argmax(2, keepdim=True)
            target = labels.gather(2, angle).squeeze(2)
            logits = logits.gather(1, angle.transpose(1, 2)).squeeze(1)
            weights = (object_mask[..., None] & (target > THRESH_BAD)).float()
            if config.method == 'scale_balanced_grasp':
                weights = weights.amax(-1, keepdim=True).expand_as(weights)
        if config.method == 'scale_balanced_grasp':
            from ..methods.scale_balanced_losses import scale_prior
            prior = scale_prior(end)
            weights = weights * prior.reshape(*prior.shape, *([1] * (target.ndim - prior.ndim)))
        offset = 1e-6
    selected = weights > 0
    logits, target, weights = logits[selected], target[selected], weights[selected]
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError('Quality supervision requires finite nonnegative native scores')
    quality = target if config.method == 'graspness' else -(-target).expm1()
    if (quality > 1).any():
        raise ValueError('Normalized grasp quality must be in [0, 1]')
    values = elementwise(logits, quality, kind, options)
    if options.get('normalization', 'native') == 'positive':
        denominator = weights[quality > 0].sum().clamp_min(1.)
    else:
        denominator = weights.sum() + offset if offset else weights.sum().clamp_min(1.)
    end[LOSS_TERMS[config.method]['score'][0]] = (values * weights).sum() / denominator
    return end
