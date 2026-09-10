"""Validated native SP2T hierarchy and sparse proxy controls."""
from copy import deepcopy
import math

DEFAULTS = dict(enc_channels=[32,64,128,256,512], enc_depths=[2,2,2,6,2],
    enc_num_head=[2,4,8,16,32], enc_patch_size=[1024]*5,
    dec_channels=[64,64,128,256], dec_depths=[2]*4, dec_num_head=[4,4,8,16],
    dec_patch_size=[1024]*4, grid_size=.005, serialization_depth=16,
    order=['z','z-trans','hilbert','hilbert-trans'], mlp_ratio=4., drop_path=.3,
    attn_drop=0., proj_drop=0., qkv_bias=True, pre_norm=True, shuffle_orders=True,
    enable_flash=True, enable_rpe=False, upcast_attention=False, upcast_softmax=False,
    enable_checkpoint=True, pooling='max', proxy_operator='attention', proxy_start_stage=1,
    proxy_initializer='square', proxy_target=160, proxy_search_range=[0.,1.],
    proxy_search_iterations=10, proxy_grid_shape=[4,4,4], proxy_cell_size=.1,
    proxy_norm='pre', proxy_local_attention=True, proxy_reuse_features=True,
    proxy_reuse_bias=True, proxy_decoder_reuse=False, proxy_mask_empty=False,
    proxy_query_reduction='none', proxy_projection=True, proxy_same_kv=True,
    proxy_self_attention=False, proxy_projection_norm=False, proxy_similarity_scale=1.,
    proxy_pe_layers=2, proxy_pe_temperature=10., proxy_bias=True,
    proxy_bias_scale=2.5, proxy_bias_table_size=16, proxy_bias_split=True,
    proxy_fuser='attention', proxy_fuser_ffn=True, proxy_fuser_rpe=True,
    proxy_fuser_rpe_scale=.4, proxy_fuser_rpe_table_size=8,
    proxy_se_pool='mean', proxy_se_layers=2)


def schema():
    result={}
    for side,minimum,maximum in [('enc',1,5),('dec',0,4)]:
        for key,low,high in [('channels',16,1024),('depths',1,24),('num_head',1,64),('patch_size',8,4096)]:
            result[f'{side}_{key}']=('int_sequence',minimum,maximum,low,high)
    result.update(stride=('int_sequence',0,4,2,8), grid_size=('float',.001,.1),
        serialization_depth=('int',8,16), order=('choice_list',1,4,('z','z-trans','hilbert','hilbert-trans')),
        pooling=('choice',('sum','mean','min','max')), mlp_ratio=('float',1,8),
        proxy_operator=('choice',('attention','pooling','trb_conv')), proxy_start_stage=('int',0,4),
        proxy_end_stage=('int',0,4), proxy_initializer=('choice',('square','fixed_grid','fixed_size')),
        proxy_target=('int',8,1024), proxy_search_range=('float_list',2,2,0,100),
        proxy_search_iterations=('int',1,30), proxy_grid_shape=('int_list',3,1,16),
        proxy_cell_size=('float',.01,2), proxy_norm=('choice',('pre','post')),
        proxy_query_reduction=('choice',('none','mean','min','max')),
        proxy_similarity_scale=('float',.01,10), proxy_pe_layers=('int',1,4),
        proxy_pe_temperature=('float',.1,100), proxy_bias_scale=('float',.01,100),
        proxy_bias_table_size=('int',2,32), proxy_fuser=('choice',('attention','se')),
        proxy_fuser_rpe_scale=('float',.01,100), proxy_fuser_rpe_table_size=('int',2,32),
        proxy_se_pool=('choice',('mean','min','max')), proxy_se_layers=('int',1,4))
    for key,value in DEFAULTS.items():
        if type(value) is bool:result[key]=('bool',)
    for key in ('drop_path','attn_drop','proj_drop'):result[key]=('float',0,.8)
    for key,lo,hi in [('mlp_ratios',1,8),('drop_path',0,.8),('attn_drop',0,.8),('proj_drop',0,.8)]:
        result['block_'+key]=('float_list',1,216,lo,hi)
    result['block_patch_sizes']=('int_sequence',1,216,8,4096)
    result['block_checkpoint']=('per_block',216,('bool',))
    return result


def block_layout(p):
    return [(side,s,i) for side,stages in [('enc',range(len(p['enc_channels']))),('dec',reversed(range(len(p['dec_channels']))))]
            for s in stages for i in range(p[side+'_depths'][s])]


def resolve(options):
    if not isinstance(options,dict):raise ValueError('SP2T parameters must be a mapping')
    unknown=set(options)-set(schema())
    if unknown:raise ValueError(f'Unknown SP2T parameters: {sorted(unknown)}')
    p={**deepcopy(DEFAULTS),**deepcopy(options)}
    if not isinstance(p['enc_channels'],list) or not 1<=len(p['enc_channels'])<=5:
        raise ValueError('SP2T requires one to five encoder stages')
    levels=len(p['enc_channels'])
    for side,count in [('enc',levels),('dec',levels-1)]:
        for key in ('channels','depths','num_head','patch_size'):
            field=side+'_'+key
            if field not in options:p[field]=p[field][:count]
    if 'proxy_start_stage' not in options:p['proxy_start_stage']=min(1,levels-1)
    p.setdefault('proxy_end_stage',levels-1);p.setdefault('stride',[2]*(levels-1))
    for side,count in [('enc',levels),('dec',levels-1)]:
        depths=p[side+'_depths']
        if not isinstance(depths,list) or len(depths)!=count or any(type(d) is not int or not 1<=d<=24 for d in depths):
            raise ValueError(f'SP2T {side}_depths needs {count} positive stage depths')
    for key,rule in schema().items():
        if key not in p:continue
        value=p[key];kind=rule[0]
        if kind in ('int_sequence','float_list'):
            types=(int,) if kind=='int_sequence' else (int,float)
            valid=isinstance(value,list) and rule[1]<=len(value)<=rule[2] and all(type(v) in types and math.isfinite(v) and rule[3]<=v<=rule[4] for v in value)
        elif kind=='int_list':valid=isinstance(value,list) and len(value)==rule[1] and all(type(v) is int and rule[2]<=v<=rule[3] for v in value)
        elif kind=='choice_list':valid=isinstance(value,list) and rule[1]<=len(value)<=rule[2] and all(isinstance(v,str) and v in rule[3] for v in value)
        elif kind=='per_block':
            values=value if isinstance(value,list) else [value]*len(block_layout(p))
            valid=len(values)==len(block_layout(p)) and all(type(v) is bool for v in values);p[key]=values
        elif kind=='bool':valid=type(value) is bool
        elif kind in ('int','float'):valid=type(value) in ((int,) if kind=='int' else (int,float)) and math.isfinite(value) and rule[1]<=value<=rule[2]
        else:valid=isinstance(value,str) and value in rule[1]
        if not valid:raise ValueError(f'Invalid SP2T {key}: expected {rule}')
    for side,count in [('enc',levels),('dec',levels-1)]:
        for key in ('channels','depths','num_head','patch_size'):
            if len(p[side+'_'+key])!=count:raise ValueError(f'SP2T {side}_{key} needs {count} values, fine to coarse')
        for c,h in zip(p[side+'_channels'],p[side+'_num_head']):
            if c%h:raise ValueError('SP2T heads must divide each stage width')
            if p['enable_flash'] and (c//h>256 or (c//h)%8):raise ValueError('SP2T FlashAttention requires head widths divisible by eight and at most 256')
    if len(p['stride'])!=levels-1 or any(s not in (2,4,8) for s in p['stride']):raise ValueError('SP2T stride needs one power of two per transition')
    if sum(int(math.log2(s)) for s in p['stride'])>p['serialization_depth']:raise ValueError('SP2T strides exceed serialization depth')
    if len(set(p['order']))!=len(p['order']):raise ValueError('SP2T serialization orders must be distinct')
    if not 0<=p['proxy_start_stage']<=p['proxy_end_stage']<levels:raise ValueError('SP2T proxy stage interval must lie within the encoder')
    if p['enable_flash'] and (p['enable_rpe'] or p['upcast_attention'] or p['upcast_softmax']):raise ValueError('SP2T local RPE/upcasting requires enable_flash: false')
    if p['proxy_self_attention'] and not p['proxy_projection']:raise ValueError('SP2T proxy self attention requires projections')
    if p['proxy_projection_norm'] and not p['proxy_projection']:raise ValueError('SP2T projection normalization requires projections')
    if p['proxy_operator']!='attention':
        for key in ('proxy_query_reduction','proxy_same_kv','proxy_self_attention','proxy_similarity_scale'):
            if key in options and p[key]!=DEFAULTS[key]:raise ValueError(f'SP2T {key} requires proxy_operator: attention')
    if p['proxy_operator']=='pooling':
        if options.get('proxy_bias',False):raise ValueError('SP2T sparse pooling does not consume relative bias')
        p['proxy_bias']=False
    if p['proxy_decoder_reuse'] and not p['proxy_reuse_features']:
        raise ValueError('SP2T decoder association reuse requires shared proxy geometry: proxy_reuse_features: true')
    if p['proxy_mask_empty'] and p['proxy_fuser']!='attention':
        raise ValueError('SP2T empty-proxy masking requires the attention fuser')
    if p['proxy_decoder_reuse'] and p['proxy_bias']:
        for s in range(p['proxy_start_stage'],min(p['proxy_end_stage']+1,levels-1)):
            if p['enc_num_head'][s]!=p['dec_num_head'][s]:raise ValueError('SP2T decoder bias reuse requires matching encoder/decoder heads')
    if p['proxy_search_range'][0]>=p['proxy_search_range'][1]:raise ValueError('SP2T proxy search bounds must increase')
    for key in ('proxy_grid_shape','proxy_cell_size','proxy_target','proxy_search_range','proxy_search_iterations'):
        expected='fixed_grid' if key=='proxy_grid_shape' else 'fixed_size' if key=='proxy_cell_size' else 'square'
        if key in options and p['proxy_initializer']!=expected:raise ValueError(f'SP2T {key} requires proxy_initializer: {expected}')
    for key in ('proxy_fuser_ffn','proxy_fuser_rpe','proxy_fuser_rpe_scale','proxy_fuser_rpe_table_size','proxy_se_pool','proxy_se_layers'):
        expected='se' if key.startswith('proxy_se_') else 'attention'
        if key in options and p['proxy_fuser']!=expected:raise ValueError(f'SP2T {key} requires proxy_fuser: {expected}')
    layout=block_layout(p)
    for key in schema():
        if key.startswith('block_') and key in p and len(p[key])!=len(layout):raise ValueError(f'SP2T {key} requires {len(layout)} values in encoder then decoder execution order')
    ratios=p.get('block_mlp_ratios',[p['mlp_ratio']]*len(layout))
    if any(not float(p[side+'_channels'][s]*ratio).is_integer() for (side,s,i),ratio in zip(layout,ratios)):raise ValueError('SP2T MLP ratios must produce integral widths')
    return p


def native_config(p):
    init=dict(type='square',target_proxy_range=p['proxy_target'],proxy_search_range=p['proxy_search_range'],max_search_iter=p['proxy_search_iterations'])
    if p['proxy_initializer']=='fixed_grid':init=dict(type='fix grid',grid_shape=p['proxy_grid_shape'])
    if p['proxy_initializer']=='fixed_size':init=dict(type='fix size',grid_size=p['proxy_cell_size'])
    initializer=dict(init_cfg=init,asso_cfg=dict(type='fgrid',dim=3),reuse_proxy_feat=p['proxy_reuse_features'],
        reuse_rel_bias=p['proxy_reuse_bias'],lvl_wise_reuse=p['proxy_decoder_reuse'],mask_empty=p['proxy_mask_empty'])
    if p['proxy_bias']:initializer['rel_bias_cfg']=dict(input_scale=p['proxy_bias_scale'],table_size=p['proxy_bias_table_size'],temperature=[.5,2.5],strength=1.,split=p['proxy_bias_split'],norm=False)
    fuser=dict(mode='attn',with_ffn=p['proxy_fuser_ffn'])
    if p['proxy_fuser_rpe']:fuser['rpe_cfg']=dict(input_scale=p['proxy_fuser_rpe_scale'],table_size=p['proxy_fuser_rpe_table_size'],temperature=[.5,2.5],strength=1.,norm=False)
    if p['proxy_fuser']=='se':fuser=dict(mode='se',pool_mode=p['proxy_se_pool'],num_layers=p['proxy_se_layers'])
    ca=dict(with_proj=p['proxy_projection'],norm_after_proj=p['proxy_projection_norm'])
    if p['proxy_operator']=='attention':ca.update(reduce_query={'min':'amin','max':'amax'}.get(p['proxy_query_reduction'],p['proxy_query_reduction']),same_kv=p['proxy_same_kv'],include_self=p['proxy_self_attention'],sim_scale=p['proxy_similarity_scale'])
    spa=dict(norm_loc=p['proxy_norm'],with_ptv3=p['proxy_local_attention'],pe_cfg=dict(num_layers=p['proxy_pe_layers'],temperature=p['proxy_pe_temperature']),initializer_cfg=initializer,ca_cfg=ca,fuser_cfg=fuser)
    keys=('order','stride','enc_depths','enc_channels','enc_num_head','enc_patch_size','dec_depths','dec_channels','dec_num_head','dec_patch_size','mlp_ratio','qkv_bias','attn_drop','proj_drop','drop_path','pre_norm','shuffle_orders','enable_rpe','enable_flash','upcast_attention','upcast_softmax','enable_checkpoint')
    cfg={key:deepcopy(p[key]) for key in keys}
    category={'attention':'SparseAttention','pooling':'SparsePooling','trb_conv':'SparseTRBConv'}[p['proxy_operator']]
    cfg.update(spa_cfg=spa,spa_skip_layer=[p['proxy_start_stage'],p['proxy_end_stage']],cls_mode=False,debug_interval=-1,debug_save=False,
        enc_block_category_list=[category]*len(p['enc_channels']),dec_block_category_list=[category]*len(p['dec_channels']))
    return cfg


def validate(options):
    resolve(options)
