"""Serializable controls for HGGD epoch training."""
DEFAULTS = dict(joint_training=True, pre_epochs=0, shift_epochs=5,
                accumulation_steps=2, center_num=128, group_num=512,
                local_grasp_num=500, shift_min_labels=1_000_000)


def validate(config):
    options = config.trainer
    if not isinstance(options, dict): raise ValueError('trainer must be a mapping')
    if options and (config.method != 'hggd' or config.action != 'train'):
        raise ValueError('Trainer parameters are registered for HGGD epoch training only')
    if set(options) - set(DEFAULTS): raise ValueError('Unknown HGGD trainer parameters')
    bounds = dict(pre_epochs=(0, 10000), shift_epochs=(0, 10000), accumulation_steps=(1, 64),
                  center_num=(8, 256), group_num=(32, 2048), local_grasp_num=(1, 2000),
                  shift_min_labels=(2, 10_000_000))
    for name, value in options.items():
        if name == 'joint_training':
            if type(value) is not bool: raise ValueError('joint_training must be boolean')
        elif type(value) is not int or not bounds[name][0] <= value <= bounds[name][1]:
            raise ValueError('Invalid HGGD trainer parameter: ' + name)
    if config.method == 'hggd' and config.action == 'train':
        if config.batch_size < 2: raise ValueError('HGGD epoch training requires batch_size >= 2')
        if config.eval_batch_limit > 256: raise ValueError('Native HGGD validation uses scene 100 (256 frames)')


def resolved(config): return {**DEFAULTS, **config.trainer}
