"""Scoped compatibility for the available GraspFast source and author weights."""
import importlib
import sys
import types


def prepare():
    from .worker import prepare as prepare_source
    repo=prepare_source('graspfast')
    sys.path[:0]=[str(repo/'GraspFastModel'),str(repo/'PointNetTool/_ext_src')]
    package=types.ModuleType('PointNetTool');package.__path__=[str(repo/'PointNetTool/_ext_src')]
    sys.modules['PointNetTool']=package
    import torch
    from knn_pytorch import knn_pytorch
    knn=types.ModuleType('knn_modules')
    def myknn(ref,query,k=1):
        if k!=1:raise ValueError('This native GraspFast label path uses k=1')
        indices=torch.empty(query.shape[0],1,query.shape[2],dtype=torch.long,device=query.device)
        knn_pytorch.knn(ref.float(),query.float(),indices)
        return indices
    knn.myknn=myknn;sys.modules['knn_modules']=knn
    import loss_utils
    # The implementation exists under a misspelled name in the source mirror.
    loss_utils.generate_grasp_directions=loss_utils.generate_grasp_directins
    return importlib.import_module('GraspFastModel.graspfast')


def checkpoint_state(state):
    """Rename native modules and split a linear 3-output head without loss."""
    state=state.get('model_state_dict',state)
    if any(k.startswith('mle.') for k in state):return state
    prefixes={'backbone':'mle','rotation':'lgd','swad':'wadnet',**{f'crop{i}':f'crr{i}' for i in range(1,5)}}
    mapped={}
    for key,value in state.items():
        if key.startswith('graspable.conv_graspable.'):
            if value.shape[0]!=3:raise ValueError('Unexpected original GraspFast joint-head shape')
            suffix=key.rsplit('.',1)[-1]
            mapped['pcs.conv_object.'+suffix]=value[:2].clone()
            mapped['gce.conv_graspable.'+suffix]=value[2:3].clone()
        else:
            prefix,separator,suffix=key.partition('.')
            mapped[prefixes.get(prefix,prefix)+separator+suffix]=value
    return mapped


def guard(model,module):
    def check(_module,_inputs,end):
        mask=(end['object_criteria'].argmax(1)==1)&(end['graspable_score'].squeeze(1)>module.GRASPNESS_THRESHOLD)
        if (mask.sum(1)==0).any():raise ValueError('Native GraspFast sampler has no graspable points; verify input and checkpoint')
    model.pcs.register_forward_hook(check)
