"""Serializable, method-specific component arguments; no arbitrary imports."""
import math

SEED_INTERACTION_METHODS = ('graspnet_baseline', 'pointnet2_upgrade', 'graspness', 'economicgrasp')
SEED_INTERACTION_FIELDS = {
    'seed_interaction': ('choice', ('none', 'gaussian')),
    'interaction_heads': ('int', 1, 32),
    'interaction_sigma': ('float', .001, 1),
    'interaction_layers': ('int', 1, 4),
    'interaction_dropout': ('float', 0, .8),
}


def unpack(value):
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, dict) or not isinstance(value.get('type'), str):
        raise ValueError('A component must be a name or a mapping with type and parameters')
    return value['type'], {k: v for k, v in value.items() if k != 'type'}


def schema(method, slot, choice):
    fields = _schema(method, slot, choice)
    if slot == 'crop' and method in SEED_INTERACTION_METHODS:
        fields = {**fields, **SEED_INTERACTION_FIELDS}
    return fields


def _schema(method, slot, choice):
    if method == 'economicgrasp':
        if slot == 'head' and choice == 'native_interactive':
            return {'feature_channels': ('int', 16, 512), 'branch_depths': ('int_list', 4, 1, 4),
                    'activation': ('choice', ('relu', 'gelu', 'silu')),
                    'interaction': ('choice', ('attention', 'none')),
                    'attention_layers': ('int', 1, 8),
                    'attention_heads': ('per_block', 8, ('int', 1, 64)),
                    'attention_dropout': ('per_stage', 8, ('float', 0, .8))}
        if slot == 'backbone' and choice == 'native_tdunet':
            return {'channels': ('int_list', 8, 8, 512), 'blocks': ('int_list', 8, 1, 8),
                    'dilations': ('int_list', 8, 1, 8), 'stem_channels': ('int', 8, 128),
                    'block': ('choice', ('basic', 'bottleneck')), 'bn_momentum': ('float', .001, 1)}
        if slot == 'backbone' and choice == 'pointnet': return {}
        if slot == 'crop' and choice == 'native_cylinder':
            return {'nsample': ('int', 4, 128), 'radius': ('float', .005, .5),
                    'hmin': ('float', -.2, 0), 'hmax': ('float', 0, .2),
                    'attention_heads': ('int', 1, 37), 'attention_dropout': ('float', 0, .8),
                    'local_attention': ('bool',)}
    if method == 'gtg2':
        from grasppanda.methods.gtg2_options import schema as graph_schema
        return graph_schema(slot, choice)
    fusion = {'fusion_layers': ('int', 1, 8), 'fusion_heads': ('int', 1, 32),
              'fusion_ffn_dim': ('int', 64, 4096), 'fusion_dropout': ('float', 0, .8),
              'fusion_activation': ('choice', ('relu', 'gelu')), 'fusion_pre_norm': ('bool',)}
    common = {'activation': ('choice', ('relu', 'gelu', 'silu')),
              'normalization': ('choice', ('batch', 'group', 'none'))}
    if slot == 'backbone' and choice == 'litept':
        from grasppanda.modules.litept_options import schema as litept_schema
        return litept_schema()
    if slot == 'backbone' and choice == 'point_transformer_v2':
        from grasppanda.modules.ptv2_options import schema as ptv2_schema
        return ptv2_schema()
    if slot == 'backbone' and choice == 'sonata_ptv3':
        fields = {}
        for prefix, count in (('enc', 5), ('dec', 4)):
            fields.update({prefix + '_depths': ('int_list', count, 1, 24),
                           prefix + '_channels': ('int_list', count, 8, 1024),
                           prefix + '_num_head': ('int_list', count, 1, 64),
                           prefix + '_patch_size': ('int_list', count, 8, 1024)})
        fields.update(stride=('int_list', 4, 1, 8),
            order=('choice', ('z', 'z-trans', 'hilbert', 'hilbert-trans',
                'z+z-trans', 'hilbert+hilbert-trans', 'z+z-trans+hilbert+hilbert-trans')),
            pooling=('choice', ('max', 'mean', 'sum', 'min')),
            mlp_ratio=('float', 1, 8), drop_path=('float', 0, .8),
            attn_drop=('float', 0, .8), proj_drop=('float', 0, .8),
            layer_scale=('float', 1e-8, 1))
        fields.update({key: ('bool',) for key in ('qkv_bias', 'pre_norm', 'shuffle_orders',
            'enable_rpe', 'upcast_attention', 'upcast_softmax')})
        return fields
    if method == 'finegrasp' and slot == 'crop' and choice == 'native_cylinder':
        return {**fusion, 'nsample': ('int', 4, 128), 'radius': ('float', .005, .5), 'radius_factors': ('radii',)}
    if method in ('hggd','region_normalized_grasp') and slot == 'backbone':
        if choice == 'vmamba':
            return {'stage_channels': ('int_list',4,16,1024), 'stage_depths': ('int_list',4,1,24),
                    'state_dim': ('int',1,64), 'ssm_ratio': ('float',.5,4), 'dt_rank': ('int',1,64),
                    'scan': ('choice',('cross2d','unidirectional','bidirectional','cascade2d')),
                    'ssm_conv': ('int',1,9), 'ssm_conv_bias': ('bool',),
                    'ssm_activation': common['activation'], 'mlp_activation': common['activation'],
                    'ssm_dropout': ('float',0,.5), 'mlp_dropout': ('float',0,.5),
                    'mlp_ratio': ('float',1,8), 'drop_path': ('float',0,.5),
                    'gradient_checkpointing': ('bool',), 'projection_norm': common['normalization']}
        if choice in ('dinov2','dinov3'):
            return {'variant': ('choice', ('small','base')), 'pretrained': ('bool',),
                    'out_indices': ('int_list',4,0,11), 'trainable_blocks': ('int',0,12),
                    'drop_path': ('float',0,.5), 'gradient_checkpointing': ('bool',),
                    'projection_norm': common['normalization']}
        if choice == 'native_resnet':
            return {'variant': ('choice', ('18', '34', '50')), 'stage_depths': ('int_list', 4, 1, 32)}
        variants = {'convnextv2': ('atto', 'tiny'), 'repvit': ('m0_9', 'm1_1'), 'mobilenetv4': ('small', 'medium')}
        if choice in variants:
            fields = {'variant': ('choice', variants[choice]), 'projection_norm': common['normalization']}
            if choice != 'repvit': fields['drop_path'] = ('float', 0, .5)
            if choice != 'mobilenetv4':
                fields.update(stage_channels=('int_list', 4, 16, 1024), stage_depths=('int_list', 4, 1, 32))
            return fields
    if slot == 'backbone' and choice == 'octformer':
        from grasppanda.modules.octformer_options import schema as octformer_schema
        return octformer_schema()
    if slot == 'backbone' and choice == 'pointcloud_mamba':
        from grasppanda.modules.pcm_options import schema as pcm_schema
        return pcm_schema()
    if slot == 'backbone' and choice == 'pointmamba':
        return {'dim': ('int',8,768), 'depth': ('int',1,48),
                'num_group': ('int',4,2048), 'group_size': ('int',4,256),
                'grid_size': ('float',.0001,1), 'd_state': ('int',4,256),
                'd_conv': ('int',2,4), 'expand': ('int',1,4), 'dt_rank': ('int',1,128),
                'rms_norm': ('bool',), 'drop_path': ('float',0,.8),
                'dropout': ('float',0,.8), 'order_fusion': ('choice',('mean','concat')),
                'gradient_checkpointing': ('bool',)}
    if slot == 'backbone' and choice == 'pointmlp':
        return {'embed_dim': ('int', 8, 128), 'dim_expansion': ('int_list', 4, 1, 4),
                'pre_blocks': ('int_list', 4, 1, 12), 'pos_blocks': ('int_list', 4, 1, 12),
                'k_neighbors': ('int_list', 4, 4, 128), 'stage_points': ('int_list', 4, 4, 2048),
                'decoder_channels': ('int_list', 4, 8, 2048), 'decoder_blocks': ('int_list', 4, 1, 12),
                'res_expansion': ('float', .25, 4), 'activation': common['activation'],
                'normalize': ('choice', ('anchor', 'center'))}
    if slot == 'backbone' and choice == 'pointmeta':
        return {'width': ('int',8,128), 'blocks': ('blocks',), 'nsample': ('int',4,128),
                'radius': ('float',.005,.5), 'radius_scaling': ('float',1,4),
                'expansion': ('int',1,8), 'normalize_dp': ('bool',),
                'local_reduction': ('choice',('max','mean','sum')),
                'activation': common['activation'], 'use_res': ('bool',),
                'sa_layers': ('int',1,4), 'sa_use_res': ('bool',), 'decoder_layers': ('int',1,4)}
    if slot == 'backbone' and choice == 'pointvector':
        return {'width': ('int',8,128), 'blocks': ('blocks',),
                'nsample': ('int',4,128), 'local_nsample': ('int',4,128),
                'radius': ('float',.005,.5), 'radius_scaling': ('float',1,4),
                'normalize_dp': ('bool',), 'sa_layers': ('int',1,4),
                'sa_use_res': ('bool',), 'decoder_layers': ('int',1,4)}
    if slot == 'backbone' and choice == 'pointnext':
        return {'width': ('int', 8, 128), 'blocks': ('blocks',),
                'nsample': ('int', 4, 128), 'radius': ('float', .005, .5),
                'radius_scaling': ('float', 1, 4), 'expansion': ('int', 1, 8),
                'activation': ('choice', ('relu', 'gelu', 'silu')),
                'reduction': ('choice', ('max', 'mean', 'sum')),
                'decoder_layers': ('int', 1, 4)}
    if slot == 'backbone' and choice == 'pointnet' and method != 'graspness':
        return {**common, 'local_channels': ('channels',), 'global_channels': ('channels',),
                'fusion_channels': ('channels',), 'dropout': ('float', 0, .8)}
    if slot == 'crop' and choice == 'multiscale':
        return {'radius_factors': ('radii',)}
    if method == 'graspness' and slot == 'crop' and choice == 'finegrasp':
        return {**fusion, 'nsample': ('int', 4, 128), 'radius_factors': ('radii',)}
    if slot == 'crop' and choice == 'reslfe_cylinder':
        return {**common, 'width': ('int',8,512), 'depth': ('int',1,60),
                'local_neighbors': ('int',1,64), 'nsample': ('int',4,128),
                'radius_factors': ('radii',), 'mlp_ratio': ('float',.5,8),
                'drop_path': ('float',0,.8), 'bn_momentum': ('float',.001,1),
                'pooling': ('choice',('max','mean','attention'))}
    if slot == 'crop' and choice == 'cylinder':
        return {**common, 'hidden_channels': ('channels',), 'radius_factors': ('radii',),
                'nsample': ('int', 4, 256), 'pooling': ('choice', ('max', 'mean', 'attention'))}
    return {}


def validate_options(method, slot, choice, options):
    fields = schema(method, slot, choice)
    if set(options) - set(fields):
        raise ValueError(f'{method}/{slot}/{choice}: unknown parameters {sorted(set(options)-set(fields))}')
    for key, value in options.items():
        rule = fields[key]
        valid = False
        if rule[0] == 'choice':
            valid = isinstance(value, str) and value in rule[1]
        elif rule[0] == 'per_stage':
            values = value if isinstance(value, list) else [value]
            scalar = rule[2]
            valid = 1 <= len(values) <= rule[1] and all(
                type(v) is bool if scalar[0] == 'bool' else
                type(v) in (int, float) and math.isfinite(v) and scalar[1] <= v <= scalar[2] for v in values)
        elif rule[0] == 'per_block':
            values = value if isinstance(value, list) else [value]
            scalar = rule[2]
            valid = len(values) <= rule[1] and all(
                type(v) is bool if scalar[0] == 'bool' else
                type(v) is int and scalar[1] <= v <= scalar[2] for v in values)
        elif rule[0] == 'choice_list':
            valid = isinstance(value, list) and rule[1] <= len(value) <= rule[2] and all(isinstance(v, str) and v in rule[3] for v in value)
        elif rule[0] == 'bool':
            valid = type(value) is bool
        elif rule[0] in ('int', 'float'):
            valid = (type(value) in ((int,) if rule[0] == 'int' else (int, float))
                     and math.isfinite(value) and rule[1] <= value <= rule[2])
        elif rule[0] == 'channels':
            valid = isinstance(value, list) and 1 <= len(value) <= 8 and all(type(v) == int and 8 <= v <= 2048 for v in value)
        elif rule[0] == 'radii':
            valid = isinstance(value, list) and 1 <= len(value) <= 8 and all(type(v) in (int, float) and math.isfinite(v) and .1 <= v <= 4 for v in value)
        elif rule[0] == 'blocks':
            valid = isinstance(value, list) and len(value) == 5 and all(type(v) == int and 1 <= v <= 12 for v in value)
        elif rule[0] == 'int_sequence':
            valid = isinstance(value, list) and rule[1] <= len(value) <= rule[2] and all(type(v) is int and rule[3] <= v <= rule[4] for v in value)
        elif rule[0] == 'float_list':
            valid = isinstance(value, list) and rule[1] <= len(value) <= rule[2] and all(type(v) in (int, float) and math.isfinite(v) and rule[3] <= v <= rule[4] for v in value)
        elif rule[0] == 'int_list':
            valid = isinstance(value, list) and len(value) == rule[1] and all(type(v) == int and rule[2] <= v <= rule[3] for v in value)
        if not valid:
            raise ValueError(f'Invalid {method}/{slot}/{choice} parameter {key}: expected {rule}')
    if 'fusion_heads' in options and 256 % options['fusion_heads']:
        raise ValueError('FineGrasp fusion heads must divide the 256-channel features')
    if method == 'economicgrasp' and slot == 'head' and choice == 'native_interactive':
        if 'activation' in options and max(options.get('branch_depths', [1]*4)) == 1:
            raise ValueError('Head activation applies only to branches with depth greater than one')
        if options.get('interaction', 'attention') == 'none' and any(k.startswith('attention_') for k in options):
            raise ValueError('Attention parameters require interaction: attention')
        count = options.get('attention_layers', 1)
        for key in ('attention_heads', 'attention_dropout'):
            value = options.get(key, 1 if key == 'attention_heads' else .05)
            values = value if isinstance(value, list) else [value]*count
            if len(values) != count:
                raise ValueError(f'{key} needs one value per attention layer, or a scalar for all layers')
            if key == 'attention_heads' and any(options.get('feature_channels', 64) % n for n in values):
                raise ValueError('Each head attention count must divide feature_channels')
    if method == 'economicgrasp' and choice == 'native_cylinder':
        if 259 % options.get('attention_heads', 1):
            raise ValueError('EconomicGrasp cylinder attention uses 256 features plus XYZ; heads must divide 259 (1, 7 or 37)')
        if options.get('hmin', -.02) >= options.get('hmax', .04):
            raise ValueError('Cylinder hmin must be below hmax')
        if not options.get('local_attention', True) and {'attention_heads', 'attention_dropout'} & set(options):
            raise ValueError('Attention parameters require local_attention: true')
    if any(key.startswith('interaction_') for key in options):
        if options.get('seed_interaction', 'none') != 'gaussian':
            raise ValueError('Interaction parameters require seed_interaction: gaussian')
    if 'interaction_heads' in options and 256 % options['interaction_heads']:
        raise ValueError('Seed interaction heads must divide the 256-channel features')
    if choice in ('dinov2','dinov3'):
        indices = options.get('out_indices', [2,5,8,11])
        if indices[-1] != 11 or any(a >= b for a,b in zip(indices, indices[1:])):
            raise ValueError('DINO output indices must increase strictly and end at block 11')
    if choice == 'reslfe_cylinder':
        if options.get('width',64) % 8:
            raise ValueError('DeepLA cylinder width must be a multiple of 8')
        if options.get('local_neighbors',8) > options.get('nsample',16):
            raise ValueError('DeepLA local neighbors must not exceed the cylinder sample count')
    if choice == 'litept':
        from grasppanda.modules.litept_options import validate as validate_litept
        validate_litept(options)
    if choice == 'point_transformer_v2':
        from grasppanda.modules.ptv2_options import validate as validate_ptv2
        validate_ptv2(options)
    if choice == 'octformer':
        from grasppanda.modules.octformer_options import validate as validate_octformer
        validate_octformer(options)
    if choice == 'pointcloud_mamba':
        from grasppanda.modules.pcm_options import validate as validate_pcm
        validate_pcm(options)
    if choice == 'pointmamba' and options.get('dim',384) % 8:
        raise ValueError('PointMamba token width must be a multiple of 8')
    if choice == 'pointmeta' and options.get('blocks',[1,3,5,3,3])[0] != 1:
        raise ValueError('PointMetaBase requires a single stem block in blocks[0]')
    if choice == 'pointmlp':
        sizes = options.get('stage_points', [1024, 256, 64, 16])
        neighbors = options.get('k_neighbors', [32, 32, 32, 16])
        if any(b > a or k > a for a, b, k in zip(sizes, sizes[1:], neighbors[1:])):
            raise ValueError('PointMLP stage points must decrease; neighbors must fit the preceding stage')
        width = options.get('embed_dim', 64)
        for factor in options.get('dim_expansion', [2, 2, 2, 2]):
            width *= factor
            if width > 2048:
                raise ValueError('PointMLP expanded stage width must not exceed 2048')
    if choice == 'repvit' and any(width % 8 for width in options.get('stage_channels', [])):
        raise ValueError('RepViT stage channels must be multiples of 8')
    if choice == 'vmamba':
        if any(width % 8 for width in options.get('stage_channels', [])):
            raise ValueError('VMamba stage channels must be multiples of 8')
        if options.get('ssm_conv', 3) % 2 != 1:
            raise ValueError('VMamba local convolution requires an odd kernel size')
    if choice == 'sonata_ptv3':
        for prefix, widths, heads in (('enc', [48,96,192,384,512], [3,6,12,24,32]),
                                     ('dec', [96,96,192,384], [6,6,12,32])):
            for width, head in zip(options.get(prefix+'_channels', widths), options.get(prefix+'_num_head', heads)):
                if width % head or width % 8:
                    raise ValueError('PTv3 stage channels must be divisible by their attention heads and by 8')
        if any(value not in (1,2,4,8) for value in options.get('stride', [])):
            raise ValueError('PTv3 pooling strides must be 1, 2, 4 or 8')
    return options
