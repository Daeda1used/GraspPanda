"""Native seed supervision before candidate-dependent grasp training."""
from dataclasses import replace


class SeedWarmup(Exception):
    def __init__(self, end_points):
        self.end_points = end_points


SEED_METHODS = ('economicgrasp', 'graspness', 'finegrasp')


def objective(end_points, config, native=None):
    from ..training.losses import replace_losses
    from ..training.options import LOSS_TERMS
    if native is not None:
        native.compute_objectness_loss(end_points)
        native.compute_graspness_loss(end_points)
    names = ('objectness', 'graspness')
    functions = {key: value for key, value in config.loss.get('functions', {}).items() if key in names}
    if functions:
        replace_losses(end_points, replace(config, loss={'functions': functions}))
    loss = sum(config.loss.get('weights', {}).get(name, LOSS_TERMS[config.method][name][1])
               * end_points[LOSS_TERMS[config.method][name][0]] for name in names)
    end_points['A: Overall Loss'] = end_points['loss/overall_loss'] = loss
    return loss, end_points
