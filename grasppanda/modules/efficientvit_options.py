"""Architecture contracts for the MIT EfficientViT B/L image hierarchies."""
from copy import deepcopy

VARIANTS = {
    'b0': ([8, 16, 32, 64, 128], [1, 2, 2, 2, 2], 16),
    'b1': ([16, 32, 64, 128, 256], [1, 2, 3, 3, 4], 16),
    'b2': ([24, 48, 96, 192, 384], [1, 3, 4, 4, 6], 32),
    'b3': ([32, 64, 128, 256, 512], [1, 4, 6, 6, 9], 32),
    'l0': ([32, 64, 128, 256, 512], [1, 1, 1, 4, 4], 32),
    'l1': ([32, 64, 128, 256, 512], [1, 1, 1, 6, 6], 32),
    'l2': ([32, 64, 128, 256, 512], [1, 2, 2, 8, 8], 32),
    'l3': ([64, 128, 256, 512, 1024], [1, 2, 2, 8, 8], 32),
}


def schema():
    return dict(variant=('choice', tuple(VARIANTS)), pretrained=('bool',),
        stage_channels=('int_list', 5, 8, 1024), stage_depths=('int_list', 5, 1, 24),
        expand_ratio=('float', 1, 8), stage_expansions=('float_list', 5, 5, 1, 8),
        stage_blocks=('choice_list', 5, 5, ('res', 'fmb', 'mb', 'att', 'att@3')),
        fewer_norm=('per_block', 5, ('bool',)),
        attention_dims=('per_block', 96, ('int', 1, 128)),
        attention_heads=('per_block', 96, ('int', 1, 128)),
        attention_scales=('float_matrix', 1, 96, 0, 4, 1, 15),
        attention_bias=('per_block', 96, ('bool',)),
        attention_kernel=('choice', ('relu', 'relu6')),
        attention_epsilon=('float', 1e-15, 1e-3),
        norm=('choice', ('bn2d', 'ln2d')), norm_epsilon=('float', 1e-8, 1e-2),
        activation=('choice', ('relu', 'relu6', 'hswish', 'silu', 'gelu')),
        gradient_checkpointing=('bool',), freeze_norm_stats=('bool',),
        trainable_stages=('int', 0, 5), projection_norm=('choice', ('batch', 'group', 'none')))


def _resolve(options):
    variant=options.get('variant', 'b0')
    if variant not in VARIANTS:raise ValueError('Unknown EfficientViT variant')
    widths, depths, dim=deepcopy(VARIANTS[variant]);large=variant.startswith('l')
    invalid=set(options)&({'expand_ratio'} if large else {'stage_expansions','stage_blocks','fewer_norm'})
    if invalid:raise ValueError('Parameters do not apply to this EfficientViT family: '+str(sorted(invalid)))
    p=dict(variant=variant,pretrained=variant!='l0',stage_channels=widths,stage_depths=depths,
        norm='bn2d',norm_epsilon=1e-7 if large else 1e-5,activation='gelu' if large else 'hswish',
        attention_kernel='relu',attention_epsilon=1e-15,gradient_checkpointing=False,
        freeze_norm_stats=False,trainable_stages=5,projection_norm='batch')
    if large:p.update(stage_expansions=[1,4,4,4,6],stage_blocks=['res','fmb','fmb','mb','att'],fewer_norm=[False,False,False,True,True])
    else:p['expand_ratio']=4
    p.update(deepcopy(options))
    if large:
        if p['stage_blocks'][0].startswith('att'):raise ValueError('The EfficientViT stem requires a local block')
        if type(p['fewer_norm']) is bool:p['fewer_norm']=[p['fewer_norm']]*5
        if len(p['fewer_norm'])!=5:raise ValueError('fewer_norm needs one value per stage or a scalar')
        active=[i for i,kind in enumerate(p['stage_blocks']) if kind.startswith('att')]
    else:active=[3,4]
    block_widths=[p['stage_channels'][i] for i in active for _ in range(p['stage_depths'][i])]
    count=len(block_widths)
    for key,default in [('attention_dims',dim),('attention_bias',False)]:
        value=p.get(key,default);p[key]=value if isinstance(value,list) else [value]*count
        if len(p[key])!=count:raise ValueError(key+' must match the number of attention blocks')
    heads=p.get('attention_heads',[width//d for width,d in zip(block_widths,p['attention_dims'])])
    p['attention_heads']=heads if isinstance(heads,list) else [heads]*count
    scales=p.get('attention_scales',[[3 if large and p['stage_blocks'][i]=='att@3' else 5] for i in active for _ in range(p['stage_depths'][i])])
    if len(scales)==1 and count>1:scales=deepcopy(scales)*count
    p['attention_scales']=scales
    if len(p['attention_heads'])!=count or len(scales)!=count:raise ValueError('Attention heads and scale lists must match attention blocks')
    if any(head<1 for head in p['attention_heads']):raise ValueError('Each attention block needs at least one head; decrease attention_dims or set attention_heads')
    if any(type(k) is not int or k%2!=1 for row in scales for k in row):raise ValueError('Attention aggregation kernels must be odd integers')
    if any(len(set(row))!=len(row) for row in scales):raise ValueError('Attention scales must be distinct within a block')
    if count==0 and any(key.startswith('attention_') for key in options):raise ValueError('Attention parameters need an attention stage')
    return p


def resolve(options):
    p=_resolve(options)
    if p['pretrained']:
        if p['variant']=='l0':raise ValueError('No registered ImageNet checkpoint for EfficientViT-L0; use pretrained: false')
        native=_resolve({'variant':p['variant']})
        structural=('stage_channels','stage_depths','expand_ratio','stage_expansions','stage_blocks',
                    'fewer_norm','attention_dims','attention_heads','attention_scales','attention_bias','norm')
        changed=[key for key in structural if p.get(key)!=native.get(key)]
        if changed:raise ValueError('EfficientViT pretrained weights require native parameter shapes; set pretrained: false for '+', '.join(changed))
    return p
