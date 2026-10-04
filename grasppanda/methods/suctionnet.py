"""Pinned SuctionNet RGB-D model, author label projection and suction decoder."""
import ast
import fcntl
from functools import lru_cache
import inspect
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import types

from ..config import ROOT, catalogue
from ..integrations.suctionnet1b import selection, files_for, label_inputs, cache_root, protocol
from ..jobs import digest


def prepare():
    import torch
    from ..compat import legacy_torch
    legacy_torch()
    alias = types.ModuleType('torchvision.models.utils')
    alias.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules.setdefault(alias.__name__, alias)
    repo = ROOT/catalogue()['suctionnet_rgbd']['path']/'neural_network'
    if str(repo) not in sys.path:sys.path.insert(0, str(repo))
    return repo


def definitions(filename, *, single_frame=False):
    """Load author helpers without executing their command-line/model startup.

    The only label-script change restricts its outer 256-view loop to one view.
    Projection, Gaussian targets, model geometry and decoding remain upstream.
    """
    import cv2
    import numpy as np
    import open3d as o3d
    import os
    import scipy.io as scio
    import time
    from PIL import Image
    from transforms3d.euler import euler2mat
    from utils.xmlhandler import xmlReader
    path = prepare()/filename
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    changes = 0
    if single_frame:
        for node in nodes:
            for loop in ast.walk(node):
                if isinstance(loop, ast.For) and isinstance(loop.target, ast.Name) and loop.target.id == 'anno_idx':
                    if ast.unparse(loop.iter) != 'range(256)':raise ValueError('Unexpected SuctionNet label loop')
                    loop.iter = ast.Name(id='_frames', ctx=ast.Load()); changes += 1
        if changes != 1:raise ValueError('Pinned SuctionNet label script has changed')
    namespace = dict(cv2=cv2, np=np, o3d=o3d, os=os, scio=scio, time=time,
                     Image=Image, euler2mat=euler2mat, xmlReader=xmlReader)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def build(config):
    import torch
    prepare()
    from DeepLabV3Plus.network import deeplabv3plus_resnet101
    model = deeplabv3plus_resnet101(num_classes=2, output_stride=16, pretrained_backbone=False)
    payload = {}
    if config.checkpoint:
        payload = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or 'model_state_dict' not in payload:
            raise ValueError('Expected an author SuctionNet or GraspPanda training checkpoint')
        if payload.get('method',config.method) != config.method or payload.get('camera',config.camera) != config.camera:
            raise ValueError('SuctionNet checkpoint method or camera differs')
        model.load_state_dict({k.removeprefix('module.'):v for k,v in payload['model_state_dict'].items()}, strict=True)
    return model.cuda(), payload


def observation(config, scene, frame):
    import cv2
    import numpy as np
    files = files_for(config,scene,frame)
    bgr = cv2.imread(str(files['rgb']))
    depth = cv2.imread(str(files['depth']), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None or bgr.shape != (720,1280,3) or depth.shape != (720,1280):
        raise ValueError('SuctionNet requires readable native 1280×720 RGB and depth images')
    return bgr.astype(np.float32)/255., depth.astype(np.float32)/1000.


@lru_cache(maxsize=4096)
def _digest_stat(path, size, mtime, ctime):
    return digest(path)


def fingerprint(path):
    stat = path.stat()
    return _digest_stat(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def prepare_label(config, scene, frame):
    """Hash-verified per-view cache; never precomputes the entire dataset."""
    import numpy as np
    root = cache_root(config)/f'scene_{scene:04d}'/config.camera
    root.mkdir(parents=True,exist_ok=True)
    target, stamp = root/f'{frame:04d}.npz', root/f'{frame:04d}.json'
    repo = prepare()
    source_files = label_inputs(config,scene,frame)+[files_for(config,scene,frame)['rgb']]
    source_files += [repo/name for name in ('score_mapping.py','cal_center_bbox.py')]
    identity = {str(p.resolve()):fingerprint(p) for p in source_files}
    with (root/f'{frame:04d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if stamp.is_file() and target.is_file():
            saved = json.loads(stamp.read_text())
            if saved.get('inputs') == identity and saved.get('sha256') == digest(target):return target
        with tempfile.TemporaryDirectory(prefix='prepare-',dir=root) as temporary:
            temp = Path(temporary)
            shared = dict(DATASET_ROOT=config.dataset_root,
                scenedir=config.dataset_root+'/scenes/scene_{}/{}', _frames=[frame],
                FLAGS=types.SimpleNamespace(sigma=4,save_visu=False))
            centers = definitions('cal_center_bbox.py',single_frame=True)
            centers.update(shared,bbox_saveroot=str(temp/'bbox_anno'),center_saveroot=str(temp/'center_anno'),visu_saveroot=str(temp/'visu'))
            centers['get_center_bbox'](scene,config.camera)
            scores = definitions('score_mapping.py',single_frame=True)
            scores.update(shared,labeldir=config.dataset_root+'/seal_label',
                          colli_root=config.dataset_root+'/suction_collision_label',saveroot=str(temp/'score_maps'))
            scores['score_mapping'](scene,config.camera)
            suffix = f'scene_{scene}/{config.camera}/{frame:04d}.npz'
            with np.load(temp/'center_anno'/suffix,allow_pickle=False) as z: center = z['arr_0'][0]
            with np.load(temp/'bbox_anno'/suffix,allow_pickle=False) as z: bbox = z['arr_0'][0]
            with np.load(temp/f'score_maps/scene_{scene}/{config.camera}/numpy/{frame:04d}.npz',allow_pickle=False) as z: score = z['arr_0']
            if score.shape != (720,1280) or not all(np.isfinite(a).all() for a in (center,bbox,score)):
                raise ValueError('Invalid native SuctionNet training labels')
            staged = temp/'labels.npz'
            np.savez_compressed(staged,center=center,bbox=bbox,score=score)
            info = {'inputs':identity,'sha256':digest(staged),'scene':scene,'frame':frame}
            staged.replace(target)
            (temp/'stamp.json').write_text(json.dumps(info,indent=2)+'\n'); (temp/'stamp.json').replace(stamp)
    return target


def training_dataset(config, chosen):
    """Reuse the native crop, photometric/depth augmentation and target builder."""
    import numpy as np
    prepare()
    from suctionnet import suctionnet_dataset as source
    # Replace only three disk reads in the native __getitem__. Its transforms
    # and two supervised targets are executed directly from the pinned source.
    tree = ast.parse(textwrap.dedent(inspect.getsource(source.SuctionNetDataset.__getitem__)))
    names = {'center_dump':'center','bbox_dump':'bbox','score':'score'}
    changed = set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
            if name in names and isinstance(node.value,ast.Subscript) and 'np.load' in ast.unparse(node.value):
                node.value = ast.parse(f"self._labels(scene_idx, anno_idx)[{names[name]!r}]", mode='eval').body
                changed.add(name)
    if changed != set(names):raise ValueError('Pinned SuctionNet data loader has changed')
    scope = dict(vars(source))
    exec(compile(ast.fix_missing_locations(tree),source.__file__,'exec'),scope)
    class SelectedFrames(source.SuctionNetDataset):
        __getitem__ = scope['__getitem__']
        @lru_cache(maxsize=1)
        def _labels(self,scene,frame):
            with np.load(prepare_label(config,scene,frame),allow_pickle=False) as z:
                return {key:z[key] for key in ('center','bbox','score')}
    native = SelectedFrames(config.dataset_root,str(cache_root(config)),camera=config.camera,split='train')
    native.data_list = chosen
    return native


def infer(config, out):
    import cv2
    import numpy as np
    import scipy.io as scio
    import torch
    import time
    model, _ = build(config); model.eval()
    native = definitions('inference.py')
    destination = out/'predictions'; destination.mkdir(exist_ok=True)
    records = []
    for scene,frame in selection(config):
        bgr,depth = observation(config,scene,frame); depth = np.clip(depth,0,1)
        x = torch.from_numpy(np.concatenate([bgr,depth[...,None]],axis=-1)).permute(2,0,1)[None].cuda()
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.inference_mode():pred = model(x).clamp(0,1).cpu()
        if not torch.isfinite(pred).all():raise ValueError('Non-finite SuctionNet prediction')
        kernel = torch.from_numpy(native['uniform_kernel'](15))[None,None]
        heatmap = torch.nn.functional.conv2d((pred[:,0]*pred[:,1])[:,None],kernel,padding=7)[0,0].numpy()
        k = scio.loadmat(files_for(config,scene,frame)['meta'])['intrinsic_matrix']
        camera = native['CameraInfo'](1280,720,k[0,0],k[1,1],k[0,2],k[1,2],1000.)
        suctions,ys,xs = native['get_suction_from_heatmap'](depth,heatmap,camera)
        if suctions.shape != (1024,7) or not np.isfinite(suctions).all():
            raise ValueError('Native suction decoder returned invalid candidates')
        # Match the author archive layout consumed by SuctionNetAPI.
        target = destination/config.split/f'scene_{scene:04d}'/config.camera/'suction'/f'{frame:04d}.npz'
        target.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(target,suctions)
        records.append({'scene':scene,'frame':frame,'candidates':len(suctions),
            'prediction':str(target.relative_to(destination)),'sha256':digest(target),
            'forward_decode_seconds':time.monotonic()-start,
            'inputs_sha256':{key:digest(path) for key,path in files_for(config,scene,frame).items() if key!='annotations'}})
        if len(records)==1:
            overlay = cv2.addWeighted((bgr*255).astype('uint8'),.6,
                cv2.applyColorMap((heatmap*255).clip(0,255).astype('uint8'),cv2.COLORMAP_VIRIDIS),.4,0)
            for y,xp in zip(ys[:20],xs[:20]):cv2.circle(overlay,(int(xp),int(y)),5,(0,255,180),1)
            cv2.imwrite(str(out/'preview.png'),overlay)
    manifest = {'protocol':protocol(config),'checkpoint_sha256':digest(config.checkpoint),'records':records}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'stage':'dataset_inference','dataset':config.dataset,'method':config.method,
            'frames':records,'prediction_dir':'predictions','protocol':protocol(config),'ap':None,
            'note':'Native suction candidates from visual inputs only. No parallel-jaw conversion or suction AP claim.'}


def signature(config, chosen):
    data = config.to_dict()
    for key in ('checkpoint','epochs','train_checkpoint_mode','timeout_minutes','gpu'):data.pop(key,None)
    files = set()
    for scene,frame in chosen:
        files.update(files_for(config,scene,frame).values())
        files.update(label_inputs(config,scene,frame))
    entries = [[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(files)]
    repo = prepare()
    return {'config':data,'files':entries,'sources':{str(p.relative_to(repo)):fingerprint(p)
            for p in sorted(repo.rglob('*.py'))}}


def train(config,out,short=False):
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ..training.state import (same_state, capture_rng_state as rng_state,
                                  restore_rng_state as restore_rng, write_result as snapshot)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Torch selects its deterministic interpolation backward; the default CUDA
    # bilinear kernel otherwise prevents exact resume despite identical RNGs.
    torch.use_deterministic_algorithms(True)
    model,payload = build(config)
    chosen = selection(config)
    native = training_dataset(config,chosen)
    expected = signature(config,chosen)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=.0005)
    start_epoch,step = 0,0
    resume = config.train_checkpoint_mode == 'resume'
    if resume:
        if not payload.get('resume_supported') or payload.get('signature') != expected:
            raise ValueError('Resume requires a SuctionNet epoch checkpoint with unchanged data and training settings')
        start_epoch,step = payload['completed_epochs'],payload['completed_steps']
        if config.epochs<=start_epoch:raise ValueError('Set epochs above the checkpoint completed epoch count')
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if not same_state(optimizer.state_dict(),payload['optimizer_state_dict']):raise ValueError('Optimizer restoration failed')
        restore_rng(payload['rng'])
    result = {'stage':'short_training' if short else 'epoch_training','dataset':config.dataset,'method':config.method,
              'losses':[],'training_samples':len(chosen),'resumed':resume,'protocol':protocol(config),'ap':None,
              'incomplete_batch_frames_per_epoch':len(chosen)%config.batch_size,
              'note':'Native seal-map and center-map MSE. Finite selected-frame epochs; no held-out validation or suction AP.'}
    epochs = config.training_steps if short else config.epochs
    for epoch in range(start_epoch,epochs):
        # Epoch-local augmentation state is reproducible with and without data workers.
        native.data_rng = np.random.RandomState(config.seed+epoch)
        for group in optimizer.param_groups:group['lr']=config.learning_rate*.7**sum(epoch>=e for e in (20,40,60))
        loader = DataLoader(native,batch_size=config.batch_size,shuffle=True,num_workers=config.data_workers,
            drop_last=True,generator=torch.Generator().manual_seed(config.seed+epoch))
        model.train(); updates = 0
        for batch,(bgr,depth,seal,center,_) in enumerate(loader):
            if config.train_batch_limit and batch>=config.train_batch_limit:break
            x = torch.cat([bgr,depth.clamp(0,1)[...,None]],dim=-1).permute(0,3,1,2).cuda()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            seal_loss = torch.nn.functional.mse_loss(pred[:,0],seal.cuda())
            center_loss = torch.nn.functional.mse_loss(pred[:,1],center.cuda())
            loss = seal_loss+center_loss
            if not torch.isfinite(loss):raise ValueError('Non-finite SuctionNet loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite SuctionNet gradient')
            optimizer.step(); step+=1; updates+=1
            result['losses'].append({'stage':'Training','epoch':epoch+1,'step':step,'total':loss.item(),
                                    'seal':seal_loss.item(),'center':center_loss.item()})
            snapshot(out,result);print(json.dumps(result['losses'][-1]),flush=True)
            if short and step>=config.training_steps:break
        if not updates:raise ValueError('No SuctionNet optimizer update was completed; select at least two frames')
        state = {'method':config.method,'camera':config.camera,'model_state_dict':model.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),'completed_epochs':epoch+1,'completed_steps':step,
                 'signature':expected,'rng':rng_state(),'resume_supported':not short}
        temp=out/'checkpoint.pt.tmp';torch.save(state,temp);temp.replace(out/'checkpoint.pt')
        result.update(checkpoint='checkpoint.pt',completed_epochs=epoch+1,completed_steps=step)
        snapshot(out,result)
        if short and step>=config.training_steps:break
    return result


def run(config,out):
    prepare()
    if config.action=='infer':return infer(config,out)
    return train(config,out,short=config.action=='train_short')
