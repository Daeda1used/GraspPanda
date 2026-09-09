"""Independent SBG cylinder branches with unchanged native fusion and heads."""
import copy
import math

CHOICES = ('upstream', 'multiscale', 'cylinder', 'reslfe_cylinder', 'kpconvx_cylinder')


def validate_branches(branches):
    from ..module_options import unpack, validate_options
    if not isinstance(branches, list) or len(branches) != 4:
        raise ValueError('MSCQ requires four branch policies in increasing native radius order')
    for value in branches:
        kind, options = unpack(value)
        if kind not in CHOICES: raise ValueError(f'MSCQ branch type must be one of {CHOICES}')
        scale = options.pop('radius_scale', 1.)
        if type(scale) not in (int, float) or not math.isfinite(scale) or not .1 <= scale <= 4:
            raise ValueError('MSCQ branch radius_scale must be in [0.1, 4]')
        if kind == 'upstream':
            count = options.pop('nsample', 64)
            if type(count) is not int or not 4 <= count <= 128:
                raise ValueError('Upstream MSCQ branch accepts nsample from 4 to 128 and radius_scale')
            validate_options('graspnet_baseline', 'crop', kind, options)
        else: validate_options('graspnet_baseline', 'crop', kind, options)


def configure(stage, options):
    from ..module_options import unpack, SEED_INTERACTION_FIELDS
    branches = options.get('branches', ['upstream'] * 4)
    validate_branches(branches)
    changed = []
    for index, value in enumerate(branches, 1):
        name = f'crop{index}'
        kind, args = unpack(value)
        if kind == 'upstream' and not args: continue
        interaction = {key: args.pop(key) for key in SEED_INTERACTION_FIELDS if key in args}
        native = copy.deepcopy(getattr(stage, name))
        scale = args.pop('radius_scale', 1.)
        native.cylinder_radius *= scale
        for group in native.groupers: group.radius *= scale
        if kind == 'upstream':
            native.nsample = args.get('nsample', native.nsample)
            for group in native.groupers: group.nsample = native.nsample
            replacement = native
        elif kind == 'multiscale':
            from .multiscale import MultiScaleCrop
            replacement = MultiScaleCrop(native, **args)
        elif kind == 'cylinder':
            from .cylinder import CylindricalAggregation
            replacement = CylindricalAggregation(native, 'baseline', **args)
        elif kind == 'reslfe_cylinder':
            from .deepla import ResLFECylinder
            replacement = ResLFECylinder(native, 'baseline', **args)
        else:
            from .kpconvx_cylinder import KPConvXCylinder
            replacement = KPConvXCylinder(native, 'baseline', **args)
        setattr(stage, name, replacement)
        if kind != 'upstream': changed.append(f'grasp_generator.{name}.')
        if interaction.get('seed_interaction') == 'gaussian':
            from .seed_interaction import install
            install(replacement, **{key.removeprefix('interaction_'): val for key, val in interaction.items()
                                   if key.startswith('interaction_')})
            if kind == 'upstream': changed.append(f'grasp_generator.{name}.seed_interaction.')
    return changed
