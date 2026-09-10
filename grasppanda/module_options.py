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
    if slot == 'backbone' and choice in ('sonata_ptv3', 'concerto', 'utonia'):
        fields = {**fields, 'adaptation': ('pointtpa',)}
    if method in ('graspnet_baseline', 'pointnet2_upgrade', 'scale_balanced_grasp') and slot == 'backbone':
        from .modules.sampling_options import DENSE_BACKBONES, HIERARCHIES
        if choice in DENSE_BACKBONES: fields = {**fields, 'seed_sampling': ('sampler',)}
        if choice in HIERARCHIES: fields = {**fields, 'stage_sampling': ('samplers', 4)}
    if slot == 'crop' and method in SEED_INTERACTION_METHODS:
        fields = {**fields, **SEED_INTERACTION_FIELDS}
    return fields


def _schema(method, slot, choice):
    if slot == 'refinement' and choice == 'contact_score':
        from .refinement import SCHEMA
        return SCHEMA
    if method == 'scale_balanced_grasp' and slot == 'sampling':
        from .methods.scale_balanced_sampling import SCHEMA
        return SCHEMA if choice == 'object_balanced' else {}
    if slot == 'head' and choice == 'quality_residual':
        return {'hidden_channels': ('channels',), 'activation': ('choice', ('relu', 'gelu', 'silu')),
                'normalization': ('choice', ('batch', 'group', 'none')),
                'initial_probability': ('float', .001, .999)}
    if method == 'spgrasp':
        from .methods.spgrasp_options import BACKBONE, MEMORY
        return BACKBONE if (slot, choice) == ('backbone', 'hiera') else MEMORY if (slot, choice) == ('memory', 'temporal') else {}
    if method == 'scale_balanced_grasp' and slot == 'crop' and choice == 'native_mscq':
        return {'branches': ('mscq_branches', 4)}
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
    if slot == 'backbone' and choice in ('utonia', 'concerto'):
        return {'variant': ('choice', ('base',) if choice == 'utonia' else ('tiny', 'small', 'base', 'large')),
                'pretrained': ('bool',), 'train_embedding': ('bool',), 'projection_norm': ('choice', ('none', 'layer')),
                'trainable_blocks': ('int_list', 5, 0, 12), 'feature_levels': ('int_sequence', 1, 5, 0, 4),
                'input_scale': ('float', .1, 16), 'grid_size': ('float', .001, .1),
                'enc_patch_size': ('int_list', 5, 16, 4096), 'drop_path': ('float', 0, .8),
                'attn_drop': ('float', 0, .8), 'proj_drop': ('float', 0, .8), 'shuffle_orders': ('bool',)}
    if slot == 'backbone' and choice == 'kpconvx':
        from .modules.kpconvx_options import schema as kpconvx_schema
        return kpconvx_schema()
    if slot == 'backbone' and choice == 'flash3d':
        from .modules.flash3d_options import schema as flash3d_schema
        return flash3d_schema()
    if slot == 'backbone' and choice == 'oacnns':
        from .modules.oacnns_options import schema as oacnns_schema
        return oacnns_schema()
    if slot == 'backbone' and choice == 'sp2t':
        from .modules.sp2t_options import schema as sp2t_schema
        return sp2t_schema()
    if slot == 'backbone' and choice == 'swin3d':
        from .modules.swin3d_options import schema as swin3d_schema
        fields = swin3d_schema()
        if method != 'finegrasp': fields['rpe_features'] = ('choice', ('xyz',))
        return fields
    if slot == 'backbone' and choice == 'pointrwkv_released':
        from .modules.pointrwkv_options import schema as pointrwkv_schema
        return pointrwkv_schema()
    if slot == 'backbone' and choice == 'pointhr':
        from .modules.pointhr_options import schema as pointhr_schema
        return pointhr_schema()
    if slot == 'backbone' and choice == 'pointcnnpp':
        from .modules.pointcnnpp_options import schema as pointcnnpp_schema
        return pointcnnpp_schema()
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
        if choice == 'mambavision':
            from .modules.mambavision_options import schema as mambavision_schema
            return mambavision_schema()
        if choice == 'rala':
            from .modules.rala_options import schema as rala_schema
            return rala_schema()
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
    if slot == 'crop' and choice == 'kpconvx_cylinder':
        from .modules.kpconvx_cylinder_options import schema as kernel_schema
        return {**kernel_schema(), **(dict(fusion, radius=('float',.005,.5)) if method=='finegrasp' else {})}
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
        if rule[0] == 'pointtpa':
            if not isinstance(value, dict):
                raise ValueError('adaptation must be a mapping with type: pointtpa')
            continue
        valid = False
        if rule[0] == 'checkpoint_path':
            valid = isinstance(value, str) and bool(value.strip()) and '\x00' not in value
        elif rule[0] == 'flash3d_pooling':
            values = value if isinstance(value, list) else [value]
            valid = 1 <= len(values) <= 4 and all(isinstance(v, str) and v in ('mean', 'sum', 'min', 'max') for v in values)
        elif rule[0] == 'mscq_branches':
            from .modules.mscq import validate_branches
            validate_branches(value)
            valid = True
        elif rule[0] in ('sampler', 'samplers'):
            from .modules.sampling_options import normalize
            if rule[0] == 'sampler':
                normalize(value)
                valid = True
            elif isinstance(value, list) and len(value) == rule[1]:
                for item in value: normalize(item)
                valid = True
        elif rule[0] == 'choice':
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
        elif rule[0] == 'float_matrix':
            valid = isinstance(value, list) and rule[1] <= len(value) <= rule[2] and all(
                isinstance(row, list) and rule[3] <= len(row) <= rule[4] and all(
                    type(v) in (int, float) and math.isfinite(v) and rule[5] <= v <= rule[6] for v in row) for row in value)
        elif rule[0] == 'float_list':
            valid = isinstance(value, list) and rule[1] <= len(value) <= rule[2] and all(type(v) in (int, float) and math.isfinite(v) and rule[3] <= v <= rule[4] for v in value)
        elif rule[0] == 'int_list':
            valid = isinstance(value, list) and len(value) == rule[1] and all(type(v) == int and rule[2] <= v <= rule[3] for v in value)
        if not valid:
            raise ValueError(f'Invalid {method}/{slot}/{choice} parameter {key}: expected {rule}')
    options = {k: v for k, v in options.items() if k not in ('seed_sampling', 'stage_sampling')}
    if choice in ('utonia', 'concerto'):
        from .weights import component_records
        weight_id = 'utonia' if choice == 'utonia' else 'concerto_' + options.get('variant', 'base')
        depth = component_records()[weight_id]['model_config']['enc_depths']
        if any(n > d for n, d in zip(options.get('trainable_blocks', [0]*5), depth)):
            raise ValueError('trainable_blocks exceeds the selected pretrained encoder depths')
        levels = options.get('feature_levels', list(range(5)))
        if sorted(set(levels)) != levels:
            raise ValueError('feature_levels must be distinct increasing stage indices')
        if any(options.get('trainable_blocks', [0]*5)[max(levels)+1:]):
            raise ValueError('Trainable blocks must contribute to a selected feature level')
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
    if choice == 'kpconvx_cylinder':
        from .modules.kpconvx_cylinder_options import validate as validate_kernel_cylinder
        validate_kernel_cylinder(options)
    if choice == 'reslfe_cylinder':
        if options.get('width',64) % 8:
            raise ValueError('DeepLA cylinder width must be a multiple of 8')
        if options.get('local_neighbors',8) > options.get('nsample',16):
            raise ValueError('DeepLA local neighbors must not exceed the cylinder sample count')
    if choice == 'kpconvx':
        from .modules.kpconvx_options import validate as validate_kpconvx
        validate_kpconvx(options)
    if choice == 'flash3d':
        from .modules.flash3d_options import validate as validate_flash3d
        validate_flash3d(options)
    if choice == 'oacnns':
        from .modules.oacnns_options import validate as validate_oacnns
        validate_oacnns(options)
    if choice == 'sp2t':
        from .modules.sp2t_options import validate as validate_sp2t
        validate_sp2t(options)
    if choice == 'swin3d':
        from .modules.swin3d_options import validate as validate_swin3d
        validate_swin3d(options)
    if choice == 'pointrwkv_released':
        from .modules.pointrwkv_options import validate as validate_pointrwkv
        validate_pointrwkv(options)
    if choice == 'pointhr':
        from .modules.pointhr_options import validate as validate_pointhr
        validate_pointhr(options)
    if choice == 'pointcnnpp':
        from .modules.pointcnnpp_options import validate as validate_pointcnnpp
        validate_pointcnnpp(options)
    if choice == 'mambavision':
        from .modules.mambavision_options import validate as validate_mambavision
        validate_mambavision(options)
    if choice == 'rala':
        from .modules.rala_options import validate as validate_rala
        validate_rala(options)
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
    if 'adaptation' in options:
        from .modules.pointtpa_options import resolve
        if choice == 'sonata_ptv3':
            depths, max_level = options.get('enc_depths', [3,3,3,12,3]), 4
        else:
            depths, max_level = depth, max(options.get('feature_levels', list(range(5))))
        resolve(options['adaptation'], depths, max_level)
    return options
