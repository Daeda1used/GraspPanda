"""Pinned native DexGraspNet hand models, shared sparse encoders and geometry export."""
import json
from pathlib import Path
import time

VARIANTS = {'dexgraspnet2':'ours', 'dexgraspnet2_isa':'isagrasp', 'dexgraspnet2_cvae':'grasptta'}


def prepare(method):
    from ..worker import prepare as prepare_source
    return prepare_source(method)


def build(config, out=None):
    import torch
    from ..components import load_checkpoint
    from ..module_options import unpack
    repo = prepare(config.method)
    from src.utils.config import load_config
    from src.network.graspness_sample import GraspnessSample
    native = load_config(repo/'configs/network'/f'train_dex_{VARIANTS[config.method]}.yaml')
    native.model.voxel_size = config.voxel_size
    model = GraspnessSample(native.model)
    choice, options = unpack(config.modules.get('backbone', 'upstream'))
    changes = []
    if choice == 'pointnet':
        from ..modules.sparse_pointnet import SparsePointNet
        model.backbone = SparsePointNet(native.model.feature_dim, config.voxel_size)
    elif choice == 'sparse_unet18':
        from src.network.backbones.conv_unet import MinkUNet18D
        model.backbone = MinkUNet18D(in_channels=3, out_channels=native.model.feature_dim, D=3)
    elif choice == 'sonata_ptv3':
        from ..modules.sonata import SparseSonataBackbone
        model.backbone = SparseSonataBackbone(native.model.feature_dim, config.voxel_size, **options)
    if choice != 'upstream': changes = ['backbone.']
    transfer = dict(policy='random_initialization', initialized=sorted(model.state_dict()), discarded=[])
    checkpoint = {}
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        saved = checkpoint.get('grasppanda', {}).get('config')
        if saved and saved.get('method') != config.method:
            raise ValueError('This checkpoint belongs to another native dexterous model.')
        if saved and config.checkpoint_policy == 'strict' and saved.get('modules', {}) != config.modules:
            raise ValueError('Strict checkpoint reuse requires the saved component configuration.')
        transfer = load_checkpoint(model, checkpoint['model'], changes, config.checkpoint_policy)
    if out is not None:
        (out/'component_transfer.json').write_text(json.dumps(transfer, indent=2)+'\n')
    return model.cuda(), native, checkpoint


def geometry(data, arrays, out):
    """Export the actual native hand mesh and its calibrated depth projection."""
    import numpy as np
    import torch
    import trimesh
    from PIL import Image, ImageDraw
    from src.utils.robot_model import RobotModel
    from ..integrations.dexgraspnet2 import JOINT_NAMES
    robot = RobotModel('robot_models/urdf/leap_hand.urdf', 'robot_models/meta/leap_hand/meta.yaml')
    if tuple(robot.joint_names) != JOINT_NAMES:
        raise ValueError('The pinned hand model has an unexpected joint order.')
    index = int(arrays['score'].argmax())
    qpos = torch.from_numpy(arrays['qpos'][index:index+1]).float()
    translations, rotations = robot.forward_kinematics({name:qpos[:,i] for i,name in enumerate(JOINT_NAMES)})
    hand = []
    for link in robot.link_names:
        vertices, faces = robot.get_link_mesh(link, 'visual')
        vertices = (vertices.float()@rotations[link][0].T+translations[link][0]).numpy()
        vertices = vertices@arrays['rotation'][index].T+arrays['translation'][index]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces.numpy(), process=False)
        mesh.visual.vertex_colors = [40,190,150,255]
        hand.append(mesh)
    depth = data['depth'].astype(np.float32)
    valid = depth>0
    gray = np.zeros_like(depth,dtype=np.uint8)
    low, high = np.percentile(depth[valid],[2,98])
    gray[valid] = (45+160*(1-np.clip((depth[valid]-low)/max(high-low,1),0,1))).astype(np.uint8)
    preview = Image.fromarray(gray).convert('RGB')
    draw = ImageDraw.Draw(preview,'RGBA')
    intrinsics = data['intrinsics']
    triangles = np.concatenate([mesh.vertices[mesh.faces] for mesh in hand])
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
    light = np.array([-.3,-.4,-1.]); light /= np.linalg.norm(light)
    shades = .55+.45*np.abs(normals@light)
    image_points = triangles@intrinsics.T
    image_points = image_points[:,:,:2]/np.maximum(image_points[:,:,2:],1e-6)
    for face in np.argsort(triangles[:,:,2].mean(1))[::-1]:
        if (triangles[face,:,2]>0).all():
            color = tuple((np.array([42,208,163])*shades[face]).astype(int))+(230,)
            draw.polygon([tuple(point) for point in image_points[face]],fill=color)
    preview.save(out/'preview.png')
    scene = trimesh.Scene(hand)
    stride = max(1,len(data['raw'])//15000)
    scene.add_geometry(trimesh.points.PointCloud(data['raw'][::stride],colors=[145,150,165,255]))
    scene.export(out/'hand_scene.glb')


def infer(config, out):
    import numpy as np
    import torch
    from ..integrations.dexgraspnet2 import observation, pairs, JOINT_NAMES
    from ..jobs import digest
    model, native, _ = build(config, out)
    from src.utils.dataset import get_sparse_tensor
    from src.utils.edge import detect_edge
    model.eval()
    records = []
    for scene, frame in pairs(config):
        print(f'Preparing rendered depth: scene {scene}, view {frame}', flush=True)
        data = observation(config, scene, frame)
        edges = detect_edge((data['depth']/data['depth'].max()*200).astype(np.uint8))
        batch = get_sparse_tensor(torch.from_numpy(data['points'][None]), config.voxel_size)
        batch = {key:value.cuda() for key,value in batch.items()}
        batch['seg'] = torch.from_numpy(data['labels'][None]).cuda()
        edge = torch.from_numpy(edges[data['mask']][data['indices']][None]).cuda()
        torch.cuda.synchronize(); start = time.monotonic()
        # The native diffusion sampler re-enables autograd for its exact
        # log-density Jacobian, so inference_mode must not be used here.
        with torch.no_grad():
            outputs = model.sample(batch, 1024, cate=False, allow_fail=False,
                                   graspness_scale=5, edge=edge, with_score_parts=True)
        torch.cuda.synchronize(); elapsed = time.monotonic()-start
        arrays = {name:value[0].detach().cpu().numpy() for name,value in zip(
            ('rotation','translation','qpos','score','object_id','graspness','log_probability'), outputs)}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('The native hand decoder returned non-finite predictions.')
        rotation = arrays['rotation']
        if arrays['qpos'].shape != (1024,16) or not np.allclose(rotation@rotation.transpose(0,2,1),np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(rotation),1,atol=1e-4):
            raise ValueError('The native hand decoder violated the pose or joint-shape contract.')
        path = out/'predictions'/f'scene_{scene:04d}'/config.camera/f'{frame:04d}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path, **arrays, joint_names=np.array(JOINT_NAMES),
            camera_to_world=data['pose'], camera_to_table=data['table_pose'],
            coordinate_frame=np.array('camera'), translation_units=np.array('metres'), joint_units=np.array('radians'))
        if not records: geometry(data, arrays, out)
        record = dict(scene=scene, frame=frame, raw_points=len(data['raw']), sampled_points=config.num_points,
            grasps=len(arrays['score']), forward_decode_seconds=elapsed, prediction=str(path.relative_to(out)),
            prediction_sha256=digest(path), input_sha256=data['evidence'])
        records.append(record); print(json.dumps(record),flush=True)
    checkpoint_sha = digest(config.checkpoint)
    (out/'predictions/manifest.json').write_text(json.dumps(dict(config=config.to_dict(),
        checkpoint_sha256=checkpoint_sha, frames=records, joint_names=JOINT_NAMES),indent=2)+'\n')
    return dict(stage='dataset_inference',dataset=config.dataset,method=config.method,frames=records,
        camera=config.camera,split=config.split,checkpoint_sha256=checkpoint_sha,ap=None,
        output_type='leap_hand',joint_names=JOINT_NAMES,coordinate_frame='camera',
        translation_units='metres',joint_units='radians',collision_checked=False,simulation_success=None,
        protocol_note='Native rendered-depth workspace, Canny edge suppression, 1024 proposals, '
            'graspness score scale 5 and native hand decoder. Joint angles are not clamped. '
            'Predicted hand geometry is not a collision or simulated-success result.')
