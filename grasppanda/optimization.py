"""Explicit optimizer choices and schedules indexed by completed updates."""
import math


METHODS = ('graspnet_baseline', 'pointnet2_upgrade', 'graspness', 'scale_balanced_grasp', 'hggd', 'region_normalized_grasp')
MUON_METHODS = ('graspnet_baseline', 'pointnet2_upgrade', 'hggd', 'region_normalized_grasp')


def validate(config):
    for name in ('optimizer', 'scheduler'):
        value = getattr(config, name)
        if not isinstance(value, dict): raise ValueError(f'{name} must be a mapping')
        if value and (config.method not in METHODS or config.action not in ('train', 'train_check')):
            raise ValueError(f'{name} overrides require a registered training adapter: {METHODS}')
    options = config.optimizer
    kind = options.get('type')
    fields = {
        'adam': {'weight_decay', 'betas', 'eps'},
        'adamw': {'weight_decay', 'betas', 'eps'},
        'sgd': {'weight_decay', 'momentum', 'nesterov'},
        'lion': {'weight_decay', 'betas'},
        'muon': {'weight_decay', 'momentum', 'nesterov', 'ns_steps', 'fallback_lr_scale', 'betas', 'eps'},
    }
    if options:
        if kind not in fields or set(options) - fields[kind] - {'type'}:
            raise ValueError('Unknown optimizer type or parameters')
        if kind == 'muon' and config.method not in MUON_METHODS:
            raise ValueError('Muon is registered only for dense HGGD/RNG and baseline/PointNet2 models')
        for key, value in options.items():
            if key == 'type': continue
            if key in ('weight_decay', 'momentum', 'eps', 'fallback_lr_scale'):
                bounds = {'weight_decay': (0, 1), 'momentum': (0, .999999), 'eps': (1e-12, .01), 'fallback_lr_scale': (1e-6, 1)}
                low, high = bounds[key]
                valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
            elif key == 'betas':
                valid = isinstance(value, list) and len(value) == 2 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < 1 for v in value)
            elif key == 'nesterov': valid = type(value) is bool
            else: valid = type(value) is int and 1 <= value <= 10
            if not valid: raise ValueError(f'Invalid optimizer parameter: {key}')
        if kind == 'sgd' and options.get('nesterov', False) and options.get('momentum', .9) <= 0:
            raise ValueError('SGD Nesterov requires positive momentum')
    schedule = config.scheduler
    kind = schedule.get('type')
    fields = {'constant': {'warmup_steps'}, 'cosine': {'warmup_steps', 'min_lr_ratio'}, 'multistep': {'milestones', 'gamma'}}
    if schedule:
        if kind not in fields or set(schedule) - fields[kind] - {'type'}:
            raise ValueError('Unknown scheduler type or parameters')
        for key, value in schedule.items():
            if key == 'type': continue
            if key == 'warmup_steps': valid = type(value) is int and 0 <= value <= 100000
            elif key == 'milestones':
                valid = isinstance(value, list) and 1 <= len(value) <= 32 and all(type(v) is int and v > 0 for v in value) and value == sorted(set(value))
            else: valid = type(value) in (int, float) and math.isfinite(value) and (0 if key == 'min_lr_ratio' else .0001) <= value <= 1
            if not valid: raise ValueError(f'Invalid scheduler parameter: {key}')
        if kind == 'multistep' and 'milestones' not in schedule:
            raise ValueError('Multistep scheduling requires update milestones')
        if config.action == 'train_check': validate_horizon(schedule, config.training_steps + config.proposal_warmup_steps)


def validate_horizon(options, total_steps):
    if options.get('warmup_steps', 0) >= total_steps:
        raise ValueError('Warmup must be shorter than the number of optimizer updates')
    if options.get('milestones') and options['milestones'][-1] >= total_steps:
        raise ValueError('Scheduler milestones must precede the final optimizer update')


def build_optimizer(parameters, config):
    import torch
    options = dict(config.optimizer)
    kind = options.pop('type')
    if 'betas' in options: options['betas'] = tuple(options['betas'])
    options.setdefault('weight_decay', 0.)
    if kind in ('adam', 'adamw'):
        constructor = torch.optim.Adam if kind == 'adam' else torch.optim.AdamW
        return constructor(parameters, lr=config.learning_rate, foreach=False, **options)
    if kind == 'sgd':
        options.setdefault('momentum', .9)
        return torch.optim.SGD(parameters, lr=config.learning_rate, foreach=False, **options)
    if kind == 'lion':
        from timm.optim import Lion
        return Lion(parameters, lr=config.learning_rate, foreach=False, **options)
    if kind == 'muon':
        from timm.optim import Muon
        options.setdefault('nesterov', True)
        return Muon(parameters, lr=config.learning_rate, conv_mode='flatten', **options)
    raise ValueError('Unknown optimizer')


class UpdateSchedule:
    """Set the next update's LR; checkpoint the exact schedule position."""
    def __init__(self, optimizer, options, total_steps):
        validate_horizon(options, total_steps)
        self.optimizer, self.options, self.total_steps = optimizer, dict(options), total_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.completed = 0
        self._apply()

    def _apply(self):
        index = min(self.completed, self.total_steps)
        warmup = self.options.get('warmup_steps', 0)
        if index < warmup:
            factor = (index + 1) / warmup
        elif self.options['type'] == 'cosine':
            floor = self.options.get('min_lr_ratio', 0.)
            progress = (index - warmup) / (self.total_steps - warmup)
            factor = floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))
        elif self.options['type'] == 'multistep':
            factor = self.options.get('gamma', .1) ** sum(index >= milestone for milestone in self.options['milestones'])
        else: factor = 1.
        for group, base in zip(self.optimizer.param_groups, self.base_lrs): group['lr'] = base * factor

    def step(self):
        if self.completed >= self.total_steps:
            raise ValueError('Optimizer updates exceeded the configured scheduler horizon')
        self.completed += 1
        self._apply()

    def state_dict(self):
        return dict(options=self.options, total_steps=self.total_steps, base_lrs=self.base_lrs, completed=self.completed)

    def load_state_dict(self, state):
        if state.get('options') != self.options or state.get('total_steps') != self.total_steps:
            raise ValueError('Resume scheduler configuration or update horizon differs')
        completed, rates = state.get('completed'), state.get('base_lrs')
        if type(completed) is not int or not 0 <= completed <= self.total_steps or not isinstance(rates, list) or len(rates) != len(self.optimizer.param_groups):
            raise ValueError('Invalid saved scheduler state')
        if not all(type(rate) in (int,float) and math.isfinite(rate) and rate > 0 for rate in rates):
            raise ValueError('Invalid saved scheduler base learning rates')
        self.completed, self.base_lrs = completed, rates
        self._apply()


class NativeScheduleDisabled:
    def step(self):
        pass


def native_driver(source, path, config, adapt_loader=False):
    """Replace only top-level optimizer/scheduler construction when requested."""
    import ast
    tree = ast.parse(source, filename=str(path))
    counts = dict(optimizer=0, scheduler=0, loader=0)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name): continue
        name = node.targets[0].id
        if name in ('TRAIN_DATALOADER', 'TEST_DATALOADER') and adapt_loader:
            if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != 'DataLoader':
                raise ValueError('Native data loader construction differs from the registered source')
            node.value.func = ast.Name(id='_grasppanda_loader', ctx=ast.Load())
            counts['loader'] += 1
        elif name == 'optimizer' and config.optimizer:
            if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != 'optim.Adam':
                raise ValueError('Native optimizer construction differs from the registered source')
            node.value = ast.parse('_grasppanda_optimizer(net.parameters())', mode='eval').body
            counts[name] += 1
        elif name == 'scheduler' and config.scheduler:
            if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != 'OneCycleLR':
                raise ValueError('Native scheduler construction differs from the registered source')
            node.value = ast.parse('_grasppanda_disabled_schedule()', mode='eval').body
            counts[name] += 1
    if config.optimizer and counts['optimizer'] != 1: raise ValueError('Native optimizer assignment was not found exactly once')
    if config.scheduler and counts['scheduler'] != int(config.method == 'scale_balanced_grasp'):
        raise ValueError('Native scheduler assignments differ from the registered driver')
    if adapt_loader and counts['loader'] != (1 if config.method == 'graspness' else 2):
        raise ValueError('Native data loader assignments differ from the registered driver')
    return compile(ast.fix_missing_locations(tree), str(path), 'exec')
