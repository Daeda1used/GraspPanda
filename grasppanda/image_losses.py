"""Loss routing through pinned HGGD/RNG target generators and reductions.

Each invocation gives native functions a private globals dictionary. Only their
classification and elementwise regression primitives change; source modules,
target construction, masks, units and reductions are left intact. The explicit
primitive routes are checked against the pinned source's call contract.
"""
from types import FunctionType
from .module_options import unpack
from .training_options import LOSS_TERMS


def binary_classification(prediction, target, kind, options, threshold, alpha, gamma, suppress, epsilon):
    """Independent sigmoid labels with native positive-count normalization."""
    import torch
    probability = prediction.sigmoid().clamp(epsilon, 1-epsilon)
    positive = (target >= threshold).to(prediction.dtype)
    negative_weight = (1-target).pow(4) if suppress else 1.
    alpha = options.get('alpha', alpha)
    negative = 1-probability
    if kind == 'asl':
        negative = (negative + options.get('clip', .05)).clamp(max=1)
        pos = -probability.log() * (1-probability).pow(options.get('gamma_pos', 0))
        neg = -negative.log() * (1-negative).clamp_min(torch.finfo(prediction.dtype).tiny).pow(options.get('gamma_neg', 4))
        labels = positive
    else:
        pos, neg = -probability.log(), -negative.log()
        labels = positive
        if kind == 'focal':
            pos = pos * (1-probability).pow(options.get('gamma', gamma))
            neg = neg * probability.pow(options.get('gamma', gamma))
        elif kind == 'poly1':
            pos = pos + options.get('epsilon', 1.) * (1-probability)
            neg = neg + options.get('epsilon', 1.) * probability
        elif kind == 'cross_entropy':
            smoothing = options.get('label_smoothing', 0)
            labels = positive * (1-smoothing) + smoothing / 2
        else:
            raise ValueError('Unknown sigmoid classification loss')
    values = alpha * labels * pos + (1-alpha) * (1-labels) * neg * negative_weight
    return values.sum() / positive.sum().clamp_min(1)


class ImageLosses:
    ROUTES = {
        'compute_anchor_loss': (('anchor_location', 'anchor_classification'),
                                ('anchor_theta', 'anchor_depth', 'anchor_width')),
        'compute_theta_width_loss': (('local_theta_classification',), ('local_theta', 'local_width')),
        'compute_multicls_loss': (('local_orientation',), ('local_offset',)),
    }
    HELPERS = ('loc_loss', 'anchor_cls_loss', 'offset_reg_loss', *ROUTES)

    def __init__(self, native, config):
        self.native, self.config = native, config
        self.terms = LOSS_TERMS[config.method]

    def enabled(self, *terms):
        return any(self.config.loss.get('weights', {}).get(name, self.terms[name][1]) > 0 for name in terms)

    def branch_enabled(self, prefix):
        return self.enabled(*(name for name in self.terms if name.startswith(prefix)))

    def _run(self, name, *args, **kwargs):
        if not self.config.loss:
            return getattr(self.native, name)(*args, **kwargs)
        import torch.nn.functional as functional
        from .losses import regression
        classification_terms, regression_terms = self.ROUTES[name]
        used = {'classification': 0, 'regression': 0}

        def select(group):
            terms = classification_terms if group == 'classification' else regression_terms
            index = used[group]
            if index >= len(terms):
                raise ValueError('Pinned image loss primitive route changed: ' + name)
            term = terms[index]; used[group] += 1
            kind, options = unpack(self.config.loss.get('functions', {}).get(term, 'upstream'))
            default = self.terms[term][1]
            ratio = self.config.loss.get('weights', {}).get(term, default) / default
            return kind, options, ratio

        def focal(pred, targets, thres=.99, alpha=.5, gamma=2, neg_suppress=False):
            kind, options, ratio = select('classification')
            if kind == 'upstream':
                loss = self.native.focal_loss(pred, targets, thres, alpha, gamma, neg_suppress)
            else:
                loss = binary_classification(pred, targets, kind, options, thres, alpha, gamma,
                    neg_suppress, 1e-6 if self.config.method == 'hggd' else 1e-4)
            return loss if ratio == 1 else loss * ratio

        def smooth_l1(pred, target, reduction='mean', **options):
            kind, settings, ratio = select('regression')
            if kind == 'upstream':
                loss = functional.smooth_l1_loss(pred, target, reduction=reduction, **options)
            else:
                if options or reduction != 'sum':
                    raise ValueError('Pinned masked image regression contract changed')
                loss = regression(pred-target, kind, settings).sum()
            return loss if ratio == 1 else loss * ratio

        class Functional:
            smooth_l1_loss = staticmethod(smooth_l1)
            def __getattr__(self, key): return getattr(functional, key)

        namespace = dict(vars(self.native), F=Functional(), focal_loss=focal)
        for helper in self.HELPERS:
            original = getattr(self.native, helper, None)
            if original is None: continue
            fn = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
            fn.__kwdefaults__ = original.__kwdefaults__
            namespace[helper] = fn
        result = namespace[name](*args, **kwargs)
        if used != {'classification': len(classification_terms), 'regression': len(regression_terms)}:
            raise ValueError('Pinned image loss did not visit every registered primitive: ' + name)
        return result

    def anchor(self, *args, **kwargs): return self._run('compute_anchor_loss', *args, **kwargs)
    def local(self, *args, **kwargs): return self._run('compute_multicls_loss', *args, **kwargs)
    def theta(self, *args, **kwargs): return self._run('compute_theta_width_loss', *args, **kwargs)
