"""Author GR-ConvNet models with native preprocessing, losses and planar decoding."""
import json
from pathlib import Path
import random
import sys
import time

from ..config import ROOT, catalogue
from ..integrations.jacquard import inventory, selection, files_for, options, protocol, check_manifest
from ..jobs import digest


def prepare():
    repo = ROOT/catalogue()['grconvnet_rgbd']['path']
    sys.path.insert(0, str(repo))
    # Pillow decodes the author's LZW float TIFFs without optional imagecodecs.
    # Pixel values, normalisation, resize and augmentation remain native.
    from utils.dataset_processing import image
    import numpy as np
    from PIL import Image
    def read_tiff(path):
        with Image.open(path) as value:
            return np.asarray(value).copy()
    image.imread = read_tiff
    return repo


def build(config):
    import torch
    prepare()
    from inference.models.grconvnet3 import GenerativeResnet
    from inference.models.grasp_model import ResidualBlock
    model = GenerativeResnet(input_channels=4 if config.method == 'grconvnet_rgbd' else 1,
                             channel_size=32, dropout=False)
    payload = {}
    if config.checkpoint:
        # The author release pickles a whole network. Only its known model/layer
        # classes are permitted; arbitrary checkpoint globals remain rejected.
        allowed = [GenerativeResnet, ResidualBlock, set, torch.nn.Conv2d,
                   torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.Dropout]
        with torch.serialization.safe_globals(allowed):
            loaded = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        if isinstance(loaded, GenerativeResnet):
            state = loaded.state_dict()
        elif isinstance(loaded, dict) and 'model_state_dict' in loaded:
            payload, state = loaded, loaded['model_state_dict']
            if payload.get('method', config.method) != config.method:
                raise ValueError('Checkpoint belongs to a different GR-ConvNet input modality')
        else:
            raise ValueError('Expected an author GR-ConvNet model or a GraspPanda training checkpoint')
        model.load_state_dict(state, strict=True)
    return model.cuda(), payload


def dataset(config, split=None, augment=False):
    prepare()
    from utils.data.jacquard_data import JacquardDataset
    files = inventory(config, split)
    native = JacquardDataset(str(files[0].parent.parent), output_size=300,
                            include_depth=True, include_rgb=config.method == 'grconvnet_rgbd',
                            random_rotate=augment, random_zoom=augment)
    # The same author transforms work for one archive and multiple extracted shards.
    native.grasp_files = [str(p) for p in files]
    native.depth_files = [str(files_for(p)['depth']) for p in files]
    native.rgb_files = [str(files_for(p)['rgb']) for p in files]
    native.length = len(files)
    return native, files


def observation(native, index):
    """Load visual inputs alone; prediction never rasterizes target rectangles."""
    import numpy as np
    depth = native.get_depth(index)
    value = np.expand_dims(depth, 0)
    if native.include_rgb:
        value = np.concatenate((value, native.get_rgb(index)), axis=0)
    tensor = native.numpy_to_torch(value)
    if not np.isfinite(value).all() or tensor.shape[1:] != (300, 300):
        raise ValueError('Invalid Jacquard observation: expected finite 300×300 visual inputs')
    return tensor


def snapshot(out, result):
    temp = out/'result.json.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temp.replace(out/'result.json')


def preview(native, index, q, angle, width, destination):
    import numpy as np
    from PIL import Image, ImageDraw
    from utils.dataset_processing.grasp import detect_grasps
    rgb = native.get_rgb(index, normalise=False)
    canvas = Image.fromarray(np.clip(rgb, 0, 255).astype('uint8')).resize((600, 600))
    draw = ImageDraw.Draw(canvas)
    for grasp in detect_grasps(q, angle, width_img=width, no_grasps=5):
        points = grasp.as_gr.points
        points = [(float(x)*2, float(y)*2) for y, x in points]
        draw.line(points+[points[0]], fill=(20, 220, 155), width=3)
    canvas.save(destination)


def infer(config, out):
    import numpy as np
    import torch
    model, _ = build(config)
    from inference.post_process import post_process_output
    model.eval()
    native, files = dataset(config)
    target = out/'predictions'; target.mkdir(exist_ok=True)
    records = []
    for index in selection(config, files):
        x = observation(native, index).unsqueeze(0).cuda()
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.inference_mode():
            q, angle, width = post_process_output(*model(x))
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        if not all(np.isfinite(v).all() and v.shape == (300,300) for v in (q,angle,width)):
            raise ValueError('The native decoder returned invalid planar maps')
        name = files[index].name.removesuffix('_grasps.txt')+'.npz'
        np.savez_compressed(target/name, quality=q, angle_radians=angle, width_pixels=width)
        inputs = {key: digest(p) for key, p in files_for(files[index]).items() if key != 'grasps'}
        records.append({'sample': files[index].name, 'sample_index': index, 'prediction': name,
                        'sha256': digest(target/name), 'inputs_sha256': inputs,
                        'forward_decode_seconds': elapsed})
        if len(records) == 1:
            preview(native, index, q, angle, width, out/'preview.png')
    manifest = {'protocol': protocol(config), 'checkpoint_sha256': digest(config.checkpoint), 'records': records}
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {'stage': 'dataset_inference', 'dataset': config.dataset, 'method': config.method,
            'frames': records, 'split': config.split, 'prediction_dir': 'predictions',
            'checkpoint_sha256': digest(config.checkpoint), 'ap': None,
            'protocol': protocol(config), 'note': 'Planar visual predictions; no labels were used by the predictor.'}


def evaluate(config, out):
    import numpy as np
    native, files = dataset(config)
    from utils.dataset_processing.evaluation import calculate_iou_match
    manifest = json.loads((Path(config.prediction_dir)/'manifest.json').read_text())
    rows = check_manifest(config, manifest, files)
    by_name = {row['sample']: row for row in rows}
    outcomes = []
    for index, label in enumerate(files):
        row = by_name[label.name]
        for role in ('depth', 'rgb'):
            if digest(files_for(label)[role]) != row['inputs_sha256'][role]:
                raise ValueError('Observation changed after inference: '+label.name)
        with np.load(Path(config.prediction_dir)/row['prediction'], allow_pickle=False) as maps:
            values = [maps[key] for key in ('quality', 'angle_radians', 'width_pixels')]
        if not all(v.shape == (300,300) and np.isfinite(v).all() for v in values):
            raise ValueError('Invalid saved Jacquard prediction maps')
        q, angle, width = values
        match = calculate_iou_match(q, angle, native.get_gtbb(index), no_grasps=1,
                                    grasp_width=width, threshold=options(config)['iou_threshold'])
        outcomes.append({'sample': label.name, 'matched': bool(match), 'label_sha256': digest(label)})
    score = sum(row['matched'] for row in outcomes)/len(outcomes)
    return {'stage': 'planar_evaluation', 'dataset': config.dataset, 'method': config.method,
            'metrics': {'planar_iou_success': score}, 'evaluated_samples': len(outcomes),
            'outcomes': outcomes, 'protocol': protocol(config), 'ap': None,
            'note': 'Top-1 rectangle IoU with the native 30-degree angle criterion. '
                    'This is not simulated grasp success or GraspNet AP. '
                    'Author pretrained weights may overlap these validation observations.'}


def signature(config, files):
    values = config.to_dict()
    for key in ('checkpoint', 'epochs', 'train_checkpoint_mode', 'timeout_minutes', 'gpu'):
        values.pop(key, None)
    root = Path(config.dataset_root)
    entries = []
    for label in files:
        for path in files_for(label).values():
            stat = path.stat()
            entries.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    return {'config': values, 'files': entries}


def rng_state():
    import numpy as np
    import torch
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1],dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def train(config, out, short=False):
    import torch
    # CuDNN convolution backward may otherwise vary across worker restarts.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from torch.utils.data import DataLoader, Subset
    from ..training.state import same_state
    model, payload = build(config)
    native, files = dataset(config, 'train', augment=True)
    chosen = selection(config, files)
    validation, val_files = dataset(config, 'val')
    expected = signature(config, [files[i] for i in chosen]+val_files)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch, completed_steps = 0, 0
    resume = config.train_checkpoint_mode == 'resume'
    if resume:
        if not payload.get('resume_supported') or payload.get('signature') != expected:
            raise ValueError('Resume requires an epoch checkpoint with unchanged data, split and training settings')
        start_epoch = payload['completed_epochs']; completed_steps = payload['completed_steps']
        if config.epochs <= start_epoch:
            raise ValueError('Set epochs above the checkpoint completed epoch count')
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if not same_state(optimizer.state_dict(), payload['optimizer_state_dict']):
            raise ValueError('Optimizer state was not restored exactly')
        restore_rng(payload['rng'])
    result = {'stage': 'short_training' if short else 'epoch_training', 'dataset': config.dataset,
              'method': config.method, 'losses': [], 'validation': [], 'ap': None,
              'training_samples': len(chosen), 'protocol': protocol(config), 'resumed': resume}
    epochs = max(1, config.training_steps) if short else config.epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        loader = DataLoader(Subset(native, chosen), batch_size=config.batch_size, shuffle=True,
                            num_workers=config.data_workers,
                            generator=torch.Generator().manual_seed(config.seed+epoch))
        for batch, (x, targets, *_rest) in enumerate(loader):
            if config.train_batch_limit and batch >= config.train_batch_limit:
                break
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_loss(x.cuda(), [target.cuda() for target in targets])
            loss = losses['loss']
            if not torch.isfinite(loss):
                raise ValueError('Non-finite native GR-ConvNet loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite native GR-ConvNet gradient')
            optimizer.step(); completed_steps += 1
            result['losses'].append({'stage': 'Training', 'epoch': epoch+1, 'step': completed_steps,
                                    'total': loss.item(), **{k: v.item() for k,v in losses['losses'].items()}})
            snapshot(out, result)
            print(json.dumps(result['losses'][-1]), flush=True)
            if short and completed_steps >= config.training_steps:
                break
        if not short:
            model.eval(); values = []
            with torch.inference_mode():
                for batch, (x, targets, *_rest) in enumerate(DataLoader(validation, batch_size=config.batch_size,
                        num_workers=config.data_workers)):
                    if config.eval_batch_limit and batch >= config.eval_batch_limit:
                        break
                    value = model.compute_loss(x.cuda(), [t.cuda() for t in targets])['loss']
                    if not torch.isfinite(value):raise ValueError('Non-finite validation loss')
                    values.append((value.item(),len(x)))
            result['validation'].append({'epoch':epoch+1,'samples':sum(n for _,n in values),
                'loss':sum(v*n for v,n in values)/sum(n for _,n in values)})
        state = {'method':config.method, 'model_state_dict':model.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(), 'completed_epochs':epoch+1,
                 'completed_steps':completed_steps, 'signature':expected, 'rng':rng_state(),
                 'resume_supported':not short}
        temp = out/'checkpoint.pt.tmp'; torch.save(state,temp); temp.replace(out/'checkpoint.pt')
        result.update(checkpoint='checkpoint.pt', completed_epochs=epoch+1, completed_steps=completed_steps)
        snapshot(out, result)
        if short and completed_steps >= config.training_steps:
            break
    return result


def run(config, out):
    prepare()
    if config.action == 'infer':return infer(config,out)
    if config.action == 'evaluate':return evaluate(config,out)
    return train(config,out,short=config.action=='train_short')
