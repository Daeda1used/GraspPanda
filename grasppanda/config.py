"""Validated experiment contracts; no user supplied shell commands."""
from dataclasses import asdict, dataclass, fields, field, replace
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ("graspness", "graspnet_baseline", "scale_balanced_grasp", "fgc_graspnet")
TRAIN = (*CORE[:3], 'finegrasp')
TRAIN_CHECK = (*CORE, 'finegrasp', 'hggd', 'economicgrasp', 'dograspnet', 'pointnet2_upgrade', 'region_normalized_grasp','graspness_modern','generalizing_grasp','graspbalance','contact_graspnet_g1b','granet','rgb_matters','centergrasp','gfla','motiongrasp','spahybgen','graspfast')
HEATMAP = ('hggd', 'region_normalized_grasp')
NATIVE_POINTS = ('economicgrasp', 'dograspnet')
SPLITS = {"train": (0, 100), "test_seen": (100, 130), "test_similar": (130, 160), "test_novel": (160, 190)}


def catalogue():
    return {m["id"]: m for m in json.loads((ROOT / "grasppanda/resources/methods.json").read_text())}


def capabilities(method):
    if method not in catalogue() or method == 'graspnet_api':
        return []
    actions = []
    if method == 'gtg2': return ['infer', 'evaluate', 'train']
    if method in CORE:
        actions += ["infer", "evaluate"]
    if method in (*HEATMAP,'finegrasp'):
        actions += ['infer', 'evaluate']
    if method in NATIVE_POINTS or method in ('pointnet2_upgrade','graspfast','graspbalance','granet','graspness_modern'):
        actions += ['infer', 'evaluate']
    if method in TRAIN_CHECK:
        actions += ["train_check"]
    if method in TRAIN:
        actions += ["train_smoke", "train"]
    if method in ('hggd','economicgrasp'): actions += ['train']
    from .recipes import RECIPES
    if method in RECIPES: actions += ["pipeline_smoke"]
    return actions


@dataclass(frozen=True)
class Experiment:
    dataset: str = "graspnet1b"
    modules: dict = field(default_factory=dict)
    loss: dict = field(default_factory=dict)
    augmentation: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)
    trainer: dict = field(default_factory=dict)
    checkpoint_policy: str = "strict"
    method: str = "graspness"
    action: str = "infer"
    dataset_root: str = ""
    checkpoint: str = ""
    camera: str = "realsense"
    split: str = "test_seen"
    scene: int = 100
    frame: int = 0
    frames: int = 1
    num_points: int = 15000
    seed: int = 0
    collision_thresh: float = 0.01
    voxel_size: float = 0.005
    workspace: str = "official_gt_workspace"
    batch_size: int = 2
    epochs: int = 1
    training_steps: int = 3
    proposal_warmup_steps: int = 0
    train_checkpoint_mode: str = 'initialize'
    train_batch_limit: int = 0
    eval_batch_limit: int = 0
    data_workers: int = 0
    label_root: str = ""
    sdf_root: str = ""
    learning_rate: float = 0.001
    prediction_dir: str = ""
    gpu: int = 0
    timeout_minutes: int = 60

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a mapping")
        extra = set(data) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown configuration keys: {sorted(extra)}")
        obj = cls(**data)
        obj.validate()
        return obj

    def validate(self):
        if type(self.proposal_warmup_steps) is not int or not 0 <= self.proposal_warmup_steps <= 10000:
            raise ValueError('proposal_warmup_steps must be an integer in [0, 10000]')
        if self.proposal_warmup_steps and (self.method != 'region_normalized_grasp' or self.action != 'train_check'):
            raise ValueError('Proposal warmup is registered for RNG short training only')
        from grasppanda.training.optimization import validate as validate_optimization
        validate_optimization(self)
        from grasppanda.training.options import validate_training_options
        validate_training_options(self)
        if self.method == 'gtg2':
            from grasppanda.methods.gtg2_options import validate_config as validate_trainer
        else:
            from grasppanda.methods.hggd_options import validate as validate_trainer
        validate_trainer(self)
        from .datasets import get_dataset
        from .components import validate_selection
        spec=get_dataset(self.dataset)
        validate_selection(self.method,self.modules,self.checkpoint_policy)
        if self.dataset not in catalogue().get(self.method,{}).get('datasets',['graspnet1b']):
            raise ValueError('This method has no adapter for the selected dataset')
        if self.modules and self.action not in ('infer','evaluate','train_smoke','train_check','train'):
            raise ValueError('Component overrides require an inference or training action')
        if self.train_checkpoint_mode not in ('initialize','resume'):
            raise ValueError('train_checkpoint_mode must be initialize or resume')
        if self.train_checkpoint_mode == 'resume' and self.action != 'train':
            raise ValueError('Checkpoint resume requires native epoch training; use initialize for inference or short training')
        if self.action=='train' and self.train_checkpoint_mode=='resume':
            if not self.checkpoint:raise ValueError('Resume requires a checkpoint')
            if self.checkpoint_policy!='strict':raise ValueError('Resume requires strict checkpoint loading; use initialize for component transfer')
        if self.method not in catalogue():
            raise ValueError("Unknown method")
        if self.action not in capabilities(self.method):
            raise ValueError(f"{self.method}: no implemented {self.action} adapter; consult method card")
        if self.method == 'finegrasp' and self.eval_batch_limit:
            raise ValueError('FineGrasp training has no validation loop; evaluate complete split predictions separately')
        if self.method == 'economicgrasp' and self.eval_batch_limit:
            raise ValueError('EconomicGrasp training has no validation loop; evaluate complete split predictions separately')
        if self.camera not in spec.cameras or self.split not in spec.splits:
            raise ValueError("Unknown camera or split")
        if self.workspace not in ("official_gt_workspace", "depth_only", "native_demo", "fused_gt_workspace"):
            raise ValueError("Unknown workspace policy")
        if self.action in ('infer', 'evaluate'):
            if (self.method in (*HEATMAP,'finegrasp')) != (self.workspace == 'native_demo'):
                raise ValueError('HGGD / RegionNormalizedGrasp / FineGrasp require workspace: native_demo; point methods require a point-cloud workspace policy')
            if self.method in NATIVE_POINTS and self.workspace != 'official_gt_workspace':
                raise ValueError('This adapter retains the upstream official_gt_workspace preprocessing')
        last_scene=max(stop for _,stop in spec.splits.values())
        for key, low, high in (("scene", 0, last_scene-1), ("frame", 0, spec.frames_per_scene-1), ("frames", 1, last_scene*spec.frames_per_scene),
                               ("num_points", 2048, 50000), ("seed", 0, 2**31-1),
                               ("batch_size", 1, 256 if self.method == 'gtg2' else 64), ("training_steps", 1, 1000), ("epochs", 1, 10000), ("gpu", 0, 127),
                               ("train_batch_limit",0,25600),("eval_batch_limit",0,7680),("data_workers",0,32),
                               ("timeout_minutes", 1, 43200)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer in [{low}, {high}]")
        for key, low, high in (("collision_thresh", 0, 1), ("voxel_size", 0.001, 0.05), ("learning_rate", 1e-8, 1)):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {key}")
        if self.action == "infer":
            low, high = spec.splits[self.split]
            if not low <= self.scene < high or self.scene * spec.frames_per_scene + self.frame + self.frames > high * spec.frames_per_scene:
                raise ValueError("Requested frame range crosses the selected split")
        if self.action in ("train", "train_smoke", "train_check") and self.split != "train":
            raise ValueError("Training requires split: train")
        if self.action in ('train','train_smoke'):
            expected = 'native_demo' if self.method == 'hggd' else 'official_gt_workspace'
            if self.workspace != expected: raise ValueError('This training adapter requires workspace: '+expected)
        if self.action=='train_check':
            if self.method=='motiongrasp' and self.training_steps>6:raise ValueError('MotionGrasp checks support 1–6 temporal updates per native seven-frame sequence')
            if self.method=='centergrasp' and self.camera!='kinect':raise ValueError('The native CenterGrasp training adapter requires camera: kinect')
            expected='native_demo' if self.method in HEATMAP else ('fused_gt_workspace' if self.method=='generalizing_grasp' else 'official_gt_workspace')
            if self.method=='generalizing_grasp' and (self.scene>=30 or self.frame!=0):raise ValueError('The native fusion trainer uses scenes 0–29, one fused sample per scene (frame=0)')
            if self.workspace!=expected:raise ValueError(f'This training check requires workspace: {expected}')
        if self.action in ('train_smoke','train_check') and not spec.splits['train'][0] <= self.scene < spec.splits['train'][1]:
            raise ValueError('Training checks require a scene in the training split')
        if self.action=='train' and self.train_batch_limit and not spec.splits['train'][0] <= self.scene < spec.splits['train'][1]:
            raise ValueError('Bounded native training requires a scene in the training split')
        from grasppanda.modules.pcm_options import validate_config as validate_pcm_config
        validate_pcm_config(self)
        from grasppanda.modules.ptv2_options import validate_config as validate_ptv2_config
        validate_ptv2_config(self)
        from grasppanda.modules.litept_options import validate_config as validate_litept_config
        validate_litept_config(self)
        from .modules.oacnns_options import validate_config as validate_oacnns_config
        validate_oacnns_config(self)
        from .modules.kpconvx_options import validate_config as validate_kpconvx_config
        validate_kpconvx_config(self)
        return self

    def preflight(self):
        self.validate()
        paths = {}
        for name in ('dataset_root', 'checkpoint', 'label_root', 'sdf_root', 'prediction_dir'):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f'{name} must be a path string')
            if value:
                paths[name] = os.path.abspath(ROOT / Path(value).expanduser())
        config = replace(self, **paths)
        repo = ROOT / catalogue()[self.method]["path"]
        if not (repo / ".git").exists():
            raise ValueError("Source missing: run ./panda fetch")
        from .datasets import get_provider
        get_provider(config.dataset).preflight(config)
        return config

    def to_dict(self):
        return asdict(self)


def default_dataset():
    return os.environ.get("GRASPPANDA_DATASET_ROOT", "")
