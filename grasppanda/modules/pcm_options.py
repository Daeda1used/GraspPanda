"""Point Cloud Mamba stage contracts, available without a GPU runtime."""
from copy import deepcopy

CTS_ORDERS = ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')
ORDERS = (*CTS_ORDERS, 'hilbert', 'z', 'z-trans', 'hilbert-trans')
DEFAULTS = dict(
    embed_dim=96, dim_expansion=[1, 2, 2, 2], pre_blocks=[1]*4,
    pos_blocks=[0]*4, mamba_blocks=[1, 2, 2, 4], k_neighbors=[12]*4,
    k_strides=[1]*4, reducers=[4, 4, 2, 2], res_expansion=1.,
    activation='relu', normalize='anchor', use_xyz=True,
    use_order_prompt=True, prompt_num_per_order=6, mamba_pos=True,
    pos_type='share', pos_proj_type='linear', rms_norm=True,
    fused_add_norm=True, residual_in_fp32=True, block_residual=True,
    drop_path=.1, use_windows=True, window_sizes=[1024, 512, 256, 128],
    grid_size=.04, decoder_blocks=[1]*4, decoder_mamba_blocks=[0]*4,
    decoder_rms_norm=True, decoder_fused_add_norm=False,
    decoder_residual_in_fp32=False, gmp_dim=64,
    d_state=16, d_conv=4, expand=2, ssm_bias=False, ssm_conv_bias=True,
    decoder_d_state=16, decoder_d_conv=4, decoder_expand=2,
    decoder_ssm_bias=False, decoder_ssm_conv_bias=True)


def schema():
    fields = dict(embed_dim=('int', 8, 128), dim_expansion=('int_list', 4, 1, 4),
        pre_blocks=('int_list', 4, 0, 12), pos_blocks=('int_list', 4, 0, 12),
        mamba_blocks=('int_list', 4, 0, 12), k_neighbors=('int_list', 4, 4, 128),
        k_strides=('int_list', 4, 1, 16), reducers=('int_list', 4, 1, 8),
        res_expansion=('float', .25, 4), activation=('choice', ('relu', 'gelu', 'silu')),
        normalize=('choice', ('anchor', 'center')), prompt_num_per_order=('int', 1, 32),
        pos_type=('choice', ('share', 'per_layer')), pos_proj_type=('choice', ('linear', 'mlp')),
        orders=('choice_list', 0, 48, ORDERS), drop_path=('float', 0, .8),
        window_sizes=('int_list', 4, 4, 50000), grid_size=('float', .0001, 1),
        decoder_channels=('int_list', 4, 8, 2048), decoder_blocks=('int_list', 4, 0, 12),
        decoder_mamba_blocks=('int_list', 4, 0, 12),
        decoder_orders=('choice_list', 0, 48, (*ORDERS, 'null')), gmp_dim=('int', 8, 256))
    fields.update({key: ('bool',) for key, value in DEFAULTS.items() if type(value) is bool})
    for prefix in ('', 'decoder_'):
        for name, rule in SSM_FIELDS.items():
            fields[prefix + name] = ('per_block', 48, rule)
    return fields


SSM_FIELDS = dict(d_state=('int', 1, 256), d_conv=('int', 2, 4),
                  expand=('int', 1, 4), dt_rank=('int', 1, 128),
                  ssm_bias=('bool',), ssm_conv_bias=('bool',))


def resolve(options):
    result = deepcopy(DEFAULTS)
    result.update(deepcopy(options))
    count = sum(result['mamba_blocks'])
    if not count:
        for key in ('use_order_prompt', 'mamba_pos', 'use_windows'):
            if key not in options:
                result[key] = False
    result.setdefault('orders', [ORDERS[i % 9] for i in range(count)])
    count = sum(result['decoder_mamba_blocks'])
    result.setdefault('decoder_orders', [ORDERS[i % 9] for i in range(count)])
    if count and result['decoder_mamba_blocks'][-1] and 'decoder_orders' not in options:
        result['decoder_orders'][-1] = 'null'
    width = result['embed_dim']
    channels = [width]
    for factor in result['dim_expansion']:
        width *= factor
        channels.append(width)
    result.setdefault('decoder_channels', list(reversed(channels[:-1])))
    return result


def validate(options):
    p = resolve(options)
    width = p['embed_dim']
    for factor in p['dim_expansion']:
        width *= factor
        if width % 8 or width > 2048:
            raise ValueError('PCM stage widths must be multiples of 8 and at most 2048')
    if any(depth == 0 and (p['use_xyz'] or expansion != 2)
           for depth, expansion in zip(p['pre_blocks'], p['dim_expansion'])):
        raise ValueError('PCM zero pre_blocks uses native max pooling: it requires use_xyz: false and dim_expansion: 2 at that stage')
    if any(c % 8 for c in p['decoder_channels']):
        raise ValueError('PCM decoder widths must be multiples of 8')
    if any(k % s for k, s in zip(p['k_neighbors'], p['k_strides'])):
        raise ValueError('Each PCM k_neighbors entry must be divisible by its k_strides entry')
    for depths, orders in (('mamba_blocks', 'orders'), ('decoder_mamba_blocks', 'decoder_orders')):
        if sum(p[depths]) != len(p[orders]):
            raise ValueError(f'PCM {orders} must contain one order per block in {depths}')
    if p['decoder_mamba_blocks'][-1] and p['decoder_orders'][-1] != 'null':
        raise ValueError('PCM final decoder scan must use null to retain the native output convention')
    for prefix, depths in (('', 'mamba_blocks'), ('decoder_', 'decoder_mamba_blocks')):
        count = sum(p[depths])
        for key in SSM_FIELDS:
            name = prefix + key
            if name not in options:
                continue
            if not count:
                raise ValueError(f'PCM {name} requires nonzero {depths}')
            if isinstance(p[name], list) and len(p[name]) != count:
                raise ValueError(f'PCM {name} list must contain one value per block in {depths}')
    if not sum(p['mamba_blocks']):
        if any(p[key] for key in ('use_order_prompt', 'mamba_pos', 'use_windows')):
            raise ValueError('PCM with no encoder scans requires use_order_prompt, mamba_pos and use_windows to be false or omitted')
        if any(key in options for key in ('rms_norm', 'fused_add_norm', 'residual_in_fp32', 'block_residual', 'drop_path')):
            raise ValueError('PCM encoder scan normalization and residual controls require nonzero mamba_blocks')
    if not sum(p['decoder_mamba_blocks']) and any(key in options for key in (
            'decoder_rms_norm', 'decoder_fused_add_norm', 'decoder_residual_in_fp32')):
        raise ValueError('PCM decoder scan normalization requires nonzero decoder_mamba_blocks')
    if not p['use_order_prompt'] and 'prompt_num_per_order' in options:
        raise ValueError('PCM prompt_num_per_order requires use_order_prompt: true')
    if not p['mamba_pos'] and {'pos_type', 'pos_proj_type'} & options.keys():
        raise ValueError('PCM positional projection settings require mamba_pos: true')
    if not p['use_windows'] and 'window_sizes' in options:
        raise ValueError('PCM window_sizes requires use_windows: true')
    return p


def validate_points(p, count):
    sizes = []
    for i, (k, reducer, window) in enumerate(zip(p['k_neighbors'], p['reducers'], p['window_sizes'])):
        if k > count:
            raise ValueError(f'PCM stage {i + 1} needs {k} neighbors but receives {count} points')
        count //= reducer
        if p['use_windows'] and p['mamba_blocks'][i] and count > window:
            count = count // window * window
        if count < 3:
            raise ValueError(f'PCM stage {i + 1} needs at least 3 points for native feature propagation; reduce reducers or increase num_points')
        sizes.append(count)
    return sizes


def selected(config):
    from grasppanda.module_options import unpack
    return unpack(config.modules.get('backbone', 'upstream'))[0] == 'pointcloud_mamba'


def validate_config(config):
    if not selected(config):
        return
    from grasppanda.module_options import unpack
    _, options = unpack(config.modules['backbone'])
    validate_points(resolve(options), config.num_points)
    if config.action in ('train', 'train_short') and config.batch_size < 2:
        raise ValueError('PCM training requires batch_size >= 2 for its native global-context BatchNorm; inference supports batch 1')
    if config.action == 'train_short' and config.frame + config.batch_size > 256:
        raise ValueError('PCM short training uses consecutive frames within one scene; choose an earlier first frame')
    if config.action == 'train' and config.train_batch_limit and config.scene * 256 + config.frame + config.batch_size > 25600:
        raise ValueError('PCM bounded training must include at least one full batch within the training split')


def native_options(p):
    encoder = {key: deepcopy(p[key]) for key in (
        'embed_dim', 'dim_expansion', 'pre_blocks', 'pos_blocks', 'mamba_blocks',
        'k_neighbors', 'k_strides', 'reducers', 'res_expansion', 'activation',
        'normalize', 'use_xyz', 'use_order_prompt', 'prompt_num_per_order',
        'mamba_pos', 'pos_type', 'pos_proj_type', 'rms_norm', 'fused_add_norm',
        'residual_in_fp32', 'block_residual', 'use_windows', 'grid_size')}
    encoder['_grasppanda_ssm'] = block_options(p, '', sum(p['mamba_blocks']))
    encoder.update(in_channels=3, combine_pos=False, bimamba_type='v2',
        drop_path_rate=p['drop_path'], windows_size=p['window_sizes'].copy(),
        mamba_layers_orders=p['orders'].copy())
    channels = [p['embed_dim']]
    for factor in p['dim_expansion']:
        channels.append(channels[-1] * factor)
    decoder = dict(encoder_channel_list=channels,
        decoder_channel_list=p['decoder_channels'].copy(), decoder_blocks=p['decoder_blocks'].copy(),
        mamba_blocks=p['decoder_mamba_blocks'].copy(), mamba_layers_orders=p['decoder_orders'].copy(),
        act_args=p['activation'], gmp_dim=p['gmp_dim'], bimamba_type='v2',
        rms_norm=p['decoder_rms_norm'], fused_add_norm=p['decoder_fused_add_norm'],
        residual_in_fp32=p['decoder_residual_in_fp32'])
    decoder['_grasppanda_ssm'] = block_options(p, 'decoder_', sum(p['decoder_mamba_blocks']))
    return encoder, decoder


def block_options(p, prefix, count):
    result = [{} for _ in range(count)]
    for name in SSM_FIELDS:
        key = prefix + name
        if key not in p:
            continue
        values = p[key] if isinstance(p[key], list) else [p[key]] * count
        native = {'ssm_bias': 'bias', 'ssm_conv_bias': 'conv_bias'}.get(name, name)
        for options, value in zip(result, values):
            options[native] = value
    return result
