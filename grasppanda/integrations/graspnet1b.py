"""GraspNet-1B observation, target-readiness and official evaluation provider."""
import json
from pathlib import Path
from ..config import HEATMAP
from ..datasets import get_dataset
from ..jobs import digest


def preflight(config):
    if config.action in ("infer", "train", "train_smoke", "train_check", "evaluate"):
        if not config.dataset_root or not (Path(config.dataset_root) / "scenes").is_dir():
            raise ValueError("Select a dataset root containing scenes/")
    if config.action == 'train_check':
        root=Path(config.dataset_root)
        if config.method=='generalizing_grasp':
            sdf_root=Path(config.sdf_root or config.dataset_root)
            missing=[i for i in range(88) if not (sdf_root/'models'/f'{i:03d}'/'grid_sampled_sdf.npz').is_file()]
            if missing:raise ValueError('Native fusion contact loss requires all 88 object SDF grids. Run ./panda prepare-sdf and set sdf_root; missing IDs: '+str(missing))
        if config.method=='motiongrasp':
            from ..weights import primary
            if not primary('graspnet_baseline',config.camera):raise ValueError('MotionGrasp also requires the author detector: run ./panda weights motiongrasp')
        if config.method=='centergrasp':
            from ..weights import records
            from ..config import ROOT
            missing=[r['role'] for r in records(config.method,config.camera) if r.get('role')!='primary' and not (ROOT/r['path']).is_file()]
            if missing:raise ValueError('CenterGrasp requires its paired SGDF checkpoint: run ./panda weights centergrasp --camera kinect')
        if config.method not in ('graspbalance','granet','finegrasp') and (not config.checkpoint or not Path(config.checkpoint).is_file()):
            raise ValueError('Training checks require the registered checkpoint; replaced components may be initialized explicitly.')
        labels={'finegrasp':['economic_grasp_label_300views'], 'hggd':[], 'region_normalized_grasp':[], 'economicgrasp':['economic_grasp_label_300views','graspness'],
                'dograspnet':['grasp_label_simplified','collision_label'],
                'fgc_graspnet':['grasp_label','FGC_label','collision_label'],
                'graspness_modern':['grasp_label_simplified','collision_label','graspness'],
                'contact_graspnet_g1b':['grasp_label','collision_label'],
                'granet':['grasp_label','collision_label'],
                'rgb_matters':['grasp_label','collision_label'],
                'centergrasp':['grasp_label','models'],
                'gfla':['grasp_label','models','collision_label'],
                'motiongrasp':[],
                'spahybgen':['grasp_label','collision_label'],
                'graspfast':['grasp_label','grasp_label_simplified','collision_label'],
                'generalizing_grasp':['grasp_label','collision_label','tolerance','fusion_scenes'],
                'graspness':['grasp_label_simplified','collision_label','graspness']}.get(config.method,['grasp_label','collision_label','tolerance'])
        for label in labels:
            if not (root/label).is_dir():raise ValueError('Required training targets missing: '+label)
        if config.method in ('hggd','region_normalized_grasp'):
            labelroot=Path(config.label_root) if config.label_root else root/'HGGD_Preprocessed'/f'6dto2drefine_{config.camera}'
            target=labelroot/'6d_dataset'/f'scene_{config.scene}'/'grasp_labels'/f'{config.frame}_view.npz'
            if not target.is_file():raise ValueError('HGGD training labels missing: '+str(target))
    if config.action == "pipeline_smoke":
        from ..recipes import preflight
        preflight(config)
    if config.action == "infer":
        if not config.checkpoint or not Path(config.checkpoint).is_file():
            raise ValueError("Inference requires a local checkpoint for this method/camera")
        if config.method=='finegrasp' and not (Path(config.checkpoint).parent/'model.config.json').is_file():
            raise ValueError('FineGrasp requires model.config.json alongside the selected checkpoint')
        root = Path(config.dataset_root)
        for index in range(config.scene * 256 + config.frame, config.scene * 256 + config.frame + config.frames):
            scene, frame = divmod(index, 256)
            directory = root / "scenes" / f"scene_{scene:04d}" / config.camera
            needed = [directory / "depth" / f"{frame:04d}.png", directory / "meta" / f"{frame:04d}.mat"]
            if config.method in ('finegrasp', 'gtg2'):needed += [directory/'rgb'/f'{frame:04d}.png']
            if config.method in HEATMAP:
                needed = [directory / 'depth' / f'{frame:04d}.png', directory / 'rgb' / f'{frame:04d}.png']
            if config.workspace == "official_gt_workspace":
                needed += [directory / "label" / f"{frame:04d}.png", directory / "camera_poses.npy", directory / "cam0_wrt_table.npy"]
            if any(not p.is_file() for p in needed):
                raise ValueError(f"Missing frame input: {next(p for p in needed if not p.is_file())}")
    if config.method == 'gtg2' and config.action == 'train':
        if not Path(config.label_root).is_dir():
            raise ValueError('GtG2 graph inputs are missing: run ./panda prepare-gtg2 --config with your training configuration')
        if config.checkpoint and not Path(config.checkpoint).is_file():
            raise ValueError('GtG2 initialization or resume checkpoint is missing')
    if config.action in ("train", "train_smoke") and config.method != 'gtg2':
        labels = [] if config.method == 'hggd' else (["economic_grasp_label_300views"] if config.method == "finegrasp" else ["grasp_label", "collision_label"])
        if config.method == 'economicgrasp':labels=['economic_grasp_label_300views','graspness']
        if config.method not in ("finegrasp", 'hggd', 'economicgrasp'):
            labels += ["graspness", "grasp_label_simplified"] if config.method == "graspness" else ["tolerance"]
        for label in labels:
            if not (Path(config.dataset_root) / label).is_dir():
                raise ValueError(f"Required preprocessing directory missing: {label}")
        if config.method == 'economicgrasp' and config.checkpoint and not Path(config.checkpoint).is_file():
            raise ValueError('EconomicGrasp initialization or resume checkpoint is missing')
        if config.method == 'hggd':
            labelroot=Path(config.label_root or Path(config.dataset_root)/'HGGD_Preprocessed'/f'6dto2drefine_{config.camera}')
            scene,frame=(config.scene,config.frame) if config.train_batch_limit else (0,0)
            for scene,frame in ((scene,frame),(100,0)):
                if not (labelroot/'6d_dataset'/f'scene_{scene}'/'grasp_labels'/f'{frame}_view.npz').is_file():
                    raise ValueError('HGGD epoch training needs preprocessed training and scene-100 validation labels')
            if config.checkpoint and not Path(config.checkpoint).is_file(): raise ValueError('HGGD initialization checkpoint is missing')
    if config.method == 'finegrasp' and config.action in ('train', 'train_check', 'train_smoke'):
        root = Path(config.dataset_root)
        if not (root/'instance_norm_graspness').is_dir() and not (root/'graspness').is_dir():
            raise ValueError('FineGrasp training needs instance_norm_graspness or Graspness maps for local preparation')
        if config.checkpoint and not (Path(config.checkpoint).parent/'model.config.json').is_file():
            raise ValueError('FineGrasp training initialization needs model.config.json alongside its checkpoint')
    if config.action == "evaluate":
        # Refuse partial predictions: they must never appear as benchmark AP.
        directory = Path(config.prediction_dir)
        low, high = get_dataset(config.dataset).splits[config.split]
        if config.split == "train":
            raise ValueError("Official evaluation supports test splits only")
        for scene in range(low, high):
            for frame in range(256):
                file = directory / f"scene_{scene:04d}" / config.camera / f"{frame:04d}.npy"
                if not file.is_file():
                    raise ValueError(f"Incomplete prediction split: missing {file}")
        manifest_path = directory/'manifest.json'
        if not manifest_path.exists():
            raise ValueError('Prediction provenance missing: manifest.json is required for toolbox evaluation')
        manifest = json.loads(manifest_path.read_text())
        for key in ('dataset','modules','method','camera','split','workspace','collision_thresh','voxel_size','num_points'):
            if manifest['config'].get(key) != getattr(config,key):
                raise ValueError(f'Prediction protocol mismatch: {key}')
        from ..jobs import digest
        if config.method=='finegrasp' and manifest.get('model_config_sha256')!=digest(Path(config.checkpoint).parent/'model.config.json'):
            raise ValueError('FineGrasp prediction architecture configuration differs from the selected checkpoint')
        hashes = manifest.get('files',{})
        for scene in range(low,high):
            for frame in range(256):
                relative = f'scene_{scene:04d}/{config.camera}/{frame:04d}.npy'
                if hashes.get(relative) != digest(directory/relative):
                    raise ValueError(f'Prediction hash mismatch or missing manifest entry: {relative}')

def frame_cloud(config, scene, frame):
    import numpy as np
    from PIL import Image
    import scipy.io
    from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
    directory = Path(config.dataset_root) / "scenes" / f"scene_{scene:04d}" / config.camera
    depth_path = directory / "depth" / f"{frame:04d}.png"
    meta_path = directory / "meta" / f"{frame:04d}.mat"
    depth = np.asarray(Image.open(depth_path))
    meta = scipy.io.loadmat(meta_path)
    intr = meta["intrinsic_matrix"]
    camera = CameraInfo(depth.shape[1], depth.shape[0], intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], float(meta["factor_depth"].item()))
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = depth > 0
    evidence = {"depth_sha256": digest(depth_path), "meta_sha256": digest(meta_path)}
    if config.workspace == "official_gt_workspace":
        label_path = directory / "label" / f"{frame:04d}.png"
        poses_path, table_path = directory / "camera_poses.npy", directory / "cam0_wrt_table.npy"
        seg = np.asarray(Image.open(label_path))
        trans = np.load(table_path) @ np.load(poses_path)[frame]
        mask &= get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        evidence.update(label_sha256=digest(label_path), poses_sha256=digest(poses_path), table_sha256=digest(table_path))
    raw = cloud[mask].astype(np.float32)
    if not len(raw):
        raise ValueError("Frame contains no valid points after workspace filtering")
    # Upstream sampling policy: include all points before padding if undersized.
    if len(raw) >= config.num_points:
        indexes = np.random.choice(len(raw), config.num_points, replace=False)
    else:
        indexes = np.concatenate([np.arange(len(raw)), np.random.choice(len(raw), config.num_points-len(raw), replace=True)])
    return raw, raw[indexes], intr, directory / "rgb" / f"{frame:04d}.png", evidence


def evaluate(config, out):
    import numpy as np
    from graspnetAPI import GraspNetEval
    evaluator = GraspNetEval(root=config.dataset_root, camera=config.camera, split=config.split)
    fn = getattr(evaluator, "eval_" + config.split.removeprefix("test_"))
    result, ap = fn(dump_folder=config.prediction_dir, proc=4)
    np.save(out / "evaluation.npy", result)
    return {"stage": "official_evaluation", "split": config.split, "camera": config.camera,
            "method":config.method,"ap": np.asarray(ap).tolist(), "prediction_dir": config.prediction_dir,
            "prediction_manifest_sha256":digest(Path(config.prediction_dir)/'manifest.json')}
