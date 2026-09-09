"""SBG loss substitutions retaining graspable targets and scale-weighted reductions."""
import sys


def replace(end, config):
    from grasppanda.module_options import unpack
    from grasppanda.training.options import LOSS_TERMS
    from grasppanda.training.losses import targets, classification, regression, is_classification
    native = next((sys.modules[name] for name in ('models.loss', 'loss')
                   if name in sys.modules and hasattr(sys.modules[name], 'generate_reweight_mask')), None)
    if native is None: raise ValueError('Scale-balanced loss requires the native loss module and scale prior')
    prior = native.generate_reweight_mask(end)
    for term, value in config.loss.get('functions', {}).items():
        kind, options = unpack(value)
        if kind == 'upstream': continue
        if term == 'graspable':
            prediction, target = end['objectness_score'], end['graspable_mask'].long()
            weights, scale, offset = None, 1., 0.
        else:
            prediction, target, mask, scale, _ = targets(end, 'graspnet_baseline', term)
            if term == 'view':
                mask = end['graspable_mask'].bool()[..., None].expand_as(target)
            weights = mask.float() * prior[..., None]
            if term == 'score':
                weights = weights.amax(-1, keepdim=True).expand_as(weights)
            offset = 1e-6
        if is_classification(config.method, term):
            logits = prediction.movedim(1, -1).reshape(-1, prediction.shape[1])
            values = classification(logits, target.reshape(-1), kind, options).reshape_as(target)
        else:
            values = regression((prediction - target) / scale, kind, options)
        loss = values.mean() if weights is None else (values * weights).sum() / (weights.sum() + offset)
        end[LOSS_TERMS[config.method][term][0]] = loss
    return end
