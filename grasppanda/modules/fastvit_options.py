"""Configuration contracts for native FastViT and FastViTHD image stages."""
from copy import deepcopy

VARIANTS = {
    't8': ([48,96,192,384], [2,2,4,2], 3),
    't12': ([64,128,256,512], [2,2,6,2], 3),
    's12': ([64,128,256,512], [2,2,6,2], 4),
    'sa12': ([64,128,256,512], [2,2,6,2], 4),
    'sa24': ([64,128,256,512], [4,4,12,4], 4),
    'sa36': ([64,128,256,512], [6,6,18,6], 4),
    'ma36': ([76,152,304,608], [6,6,18,6], 4),
    'hd': ([96,192,384,768,1536], [2,12,24,4,2], 4),
}


def schema():
    return dict(variant=('choice',tuple(VARIANTS)),pretrained=('bool',),
        parameterization=('choice',('branches','fused')),
        stage_channels=('int_sequence',4,5,8,2048),stage_depths=('int_sequence',4,5,1,36),
        stage_mixers=('choice_list',4,5,('repmixer','attention')),
        stage_mlp_ratios=('float_list',4,5,1,8),cpe_kernels=('int_sequence',4,5,0,15),
        downsample_kernel=('int',3,15),
        block_mixers=('choice_list',4,180,('repmixer','attention')),
        block_mlp_ratios=('per_stage',180,('float',1,8)),
        block_dropout=('per_stage',180,('float',0,.8)),
        block_drop_path=('per_stage',180,('float',0,.8)),drop_path=('float',0,.8),
        block_layer_scale=('per_block',180,('bool',)),
        block_layer_scale_init=('per_stage',180,('float',1e-8,1)),
        repmixer_kernels=('per_block',180,('int',3,15)),
        attention_head_dims=('per_block',180,('int',1,256)),
        attention_qkv_bias=('per_block',180,('bool',)),
        attention_dropout=('per_stage',180,('float',0,.8)),
        attention_projection_dropout=('per_stage',180,('float',0,.8)),
        activation=('choice',('gelu','relu','silu')),
        gradient_checkpointing=('bool',),freeze_norm_stats=('bool',),
        trainable_stages=('int',0,5),projection_norm=('choice',('batch','group','none')))


def _resolve(options):
    variant=options.get('variant','t8')
    if variant not in VARIANTS:raise ValueError('Unknown FastViT variant')
    widths,depths,ratio=deepcopy(VARIANTS[variant]);hd=variant=='hd';count=len(widths)
    mixers=['repmixer']*count
    if hd:mixers[-2:]=['attention']*2
    elif variant.startswith(('sa','ma')):mixers[-1]='attention'
    p=dict(variant=variant,pretrained=True,parameterization='fused' if hd else 'branches',
        stage_channels=widths,stage_depths=depths,stage_mixers=mixers,stage_mlp_ratios=[ratio]*count,
        cpe_kernels=[7 if m=='attention' else 0 for m in mixers],downsample_kernel=7,
        drop_path=0.,activation='gelu',gradient_checkpointing=False,freeze_norm_stats=False,
        trainable_stages=count,projection_norm='batch')
    p.update(deepcopy(options))
    for key in ('stage_channels','stage_depths','stage_mixers','stage_mlp_ratios','cpe_kernels'):
        if len(p[key])!=count:raise ValueError(key+' needs '+str(count)+' entries for '+variant)
    if p['trainable_stages']>count:raise ValueError('trainable_stages exceeds the FastViT stage count')
    if any(b%a for a,b in zip(p['stage_channels'],p['stage_channels'][1:])):
        raise ValueError('Native FastViT grouped downsampling requires each width to be a multiple of the previous width')
    if p['downsample_kernel']%2!=1 or any(k and (k<3 or k%2!=1) for k in p['cpe_kernels']):
        raise ValueError('Downsampling/CPE kernels must be odd; CPE 0 disables position convolution')
    n=sum(p['stage_depths'])
    def expand(key,default,size):
        value=p.get(key,default)
        p[key]=deepcopy(value) if isinstance(value,list) else [value]*size
        if len(p[key])!=size:raise ValueError(key+' needs a scalar or exactly '+str(size)+' entries')
    expand('block_mixers',[m for m,d in zip(p['stage_mixers'],p['stage_depths']) for _ in range(d)],n)
    expand('block_mlp_ratios',[r for r,d in zip(p['stage_mlp_ratios'],p['stage_depths']) for _ in range(d)],n)
    expand('block_dropout',0.,n)
    expand('block_drop_path',[p['drop_path']*i/max(1,n-1) for i in range(n)],n)
    expand('block_layer_scale',True,n)
    expand('block_layer_scale_init',1e-6 if variant in ('sa36','ma36') else 1e-5,n)
    nr=p['block_mixers'].count('repmixer');na=n-nr
    expand('repmixer_kernels',3,nr)
    if any(k%2!=1 for k in p['repmixer_kernels']):raise ValueError('RepMixer kernels must be odd')
    for key,default in [('attention_head_dims',32),('attention_qkv_bias',False),
                        ('attention_dropout',0.),('attention_projection_dropout',0.)]:expand(key,default,na)
    block_widths=[w for w,d in zip(p['stage_channels'],p['stage_depths']) for _ in range(d)]
    attention_widths=[w for w,m in zip(block_widths,p['block_mixers']) if m=='attention']
    if any(w%d for w,d in zip(attention_widths,p['attention_head_dims'])):
        raise ValueError('Each attention head dimension must divide its stage width')
    if not na and any(key.startswith('attention_') and value!=[] for key,value in options.items()):raise ValueError('Attention options require at least one attention block')
    if not nr and options.get('repmixer_kernels',[])!=[]:raise ValueError('repmixer_kernels requires RepMixer blocks')
    return p


def resolve(options):
    p=_resolve(options)
    if p['pretrained']:
        if p['variant']=='hd' and p['parameterization']!='fused':
            raise ValueError('Released FastViTHD weights use fused convolutions; use parameterization: fused or pretrained: false')
        native=_resolve({'variant':p['variant']})
        structural=('stage_channels','stage_depths','cpe_kernels','downsample_kernel','block_mixers',
                    'block_mlp_ratios','block_layer_scale','repmixer_kernels','attention_head_dims','attention_qkv_bias')
        changed=[key for key in structural if p[key]!=native[key]]
        if changed:raise ValueError('FastViT pretrained weights require the native structure; set pretrained: false for '+', '.join(changed))
    return p


def weight_id(options):
    p=resolve(options)
    return 'fastvit_'+p['variant']+('_fused' if p['parameterization']=='fused' and p['variant']!='hd' else '')
