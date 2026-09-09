"""Bounded upstream recipes: explicit inputs, stages and limits, never AP claims."""
from pathlib import Path
from .config import ROOT

# Fixed recipes are intentionally distinct from configurable frame adapters.
RECIPES = {
 'generalizing_grasp': ('realsense','First test fused scene → MSCQ checkpoint → decoder. Requires fusion_scenes and upstream segmentation; excludes C-SJO.'),
 'contact_graspnet_g1b': ('realsense','Scene 0100/0000 → native GT bounding box → checkpoint → decoder → collision.'),
 'rngnet_sdk': ('realsense','Scene 0100/0000 RGB-D → bundled author weights → native SDK inference.'),
 'zerograsp': ('realsense','Author supplied RGB-D + instance masks → synthetic-trained checkpoint → reconstruction → grasps. Not a GraspNet frame.'),
 'gfla': ('realsense','Scene 0100/0000 → FastSAM + learned SDF → analytic force closure → collision/NMS. Requires dataset table pose.'),
 'active_ngf': ('realsense','Scene 0100 → two mapped views, 150/50 optimization iterations, next-view selection and mesh. Not the complete active-view benchmark.'),
 'centergrasp': ('kinect','Scene 0100/0000 → native crop → RGB/SGDF checkpoints → ICP → FCL collision → grasp conversion.'),
 'motiongrasp': ('realsense','First three test frames → baseline checkpoint → MotionTracker checkpoint → trajectories. All detector candidates; no GT-ranked query or tracking metric.'),
 'graspness_modern': ('realsense','Scene 0100/0000 → pretrained ResUNet14 with spconv → decoder. Other backbones are not validated by this recipe.'),
 'rgb_matters': ('realsense','Scene 0100/0000 RGB + locally generated normals → checkpoint → native decoding/collision/NMS.'),
 'asgrasp': ('realsense','Author RGB + left/right IR sample → RAFT stereo + GSNet checkpoints → top grasp. Standard GraspNet RGB-D alone does not supply these inputs.'),
 'dreds': ('realsense','Scene 0100/0000 → pretrained SwinDRNet depth restoration. Auxiliary network, not a grasp detector.'),
 'spahybgen': ('realsense','Author supplied voxel observation → checkpoint → Robotiq2f optimization, four starts and three steps. No converged robot result.'),
}
NO_DATA = {'zerograsp','asgrasp','spahybgen'}
NO_WEIGHTS = {'rngnet_sdk','spahybgen'}
CHECKPOINT_RECIPES = {'generalizing_grasp','contact_graspnet_g1b','gfla','centergrasp','motiongrasp','rgb_matters','spahybgen'}


def preset(method, dataset_root=''):
    from .config import Experiment, HEATMAP, capabilities
    from .weights import primary, records
    if method == 'spgrasp':
        return Experiment(method=method, action='train_check', dataset_root=dataset_root, split='train', scene=0,
            frames=8, batch_size=1, learning_rate=5e-6, workspace='native_demo', collision_thresh=0,
            checkpoint=primary(method, 'realsense') or 'checkpoints/spgrasp/sam2.1_hiera_base_plus.pt')
    if method == 'gtg2':
        return Experiment(method=method, action='train', dataset_root=dataset_root, split='train', scene=0,
            label_root='outputs/prepared/gtg2', epochs=500, batch_size=128, learning_rate=.01, timeout_minutes=43200)
    if method in RECIPES and 'infer' not in capabilities(method):
        camera=RECIPES[method][0]
        return Experiment(method=method,action='pipeline_smoke',dataset_root=dataset_root,camera=camera,timeout_minutes=15)
    cameras=[r['camera'] for r in records(method,'realsense')]
    camera='realsense' if cameras else ('kinect' if records(method,'kinect') else 'realsense')
    if 'infer' not in capabilities(method):
        raise ValueError('No runnable preset for this entry; see docs/METHODS.md')
    action='infer'
    return Experiment(method=method,action=action,dataset_root=dataset_root,camera=camera,checkpoint=primary(method,camera),
                      workspace='native_demo' if method in (*HEATMAP,'finegrasp') else 'official_gt_workspace',
                      num_points=25600 if method in HEATMAP else 15000)


def preflight(config):
    from .weights import records
    from .config import Experiment
    defaults=Experiment()
    if config.checkpoint:
        if config.method not in CHECKPOINT_RECIPES:
            raise ValueError('This fixed recipe does not accept a checkpoint override')
        if not Path(config.checkpoint).is_file():raise ValueError('Selected recipe checkpoint is missing')
        if config.checkpoint_policy!='strict':raise ValueError('Fixed recipes require strict checkpoint loading')
    for name in ('scene','frame','frames','num_points','seed','split','workspace','collision_thresh','voxel_size','batch_size','epochs','learning_rate','prediction_dir'):
        if getattr(config,name)!=getattr(defaults,name):
            raise ValueError(f'Fixed recipe does not accept {name}; apply the tested preset. Use infer for configurable frames.')
    if config.camera != RECIPES[config.method][0]:
        raise ValueError('This fixed recipe uses '+RECIPES[config.method][0]+'; apply the tested preset.')
    if config.method not in NO_DATA and not (Path(config.dataset_root)/'scenes/scene_0100'/config.camera/'depth/0000.png').is_file():
        raise ValueError('Recipe needs GraspNet scene_0100. Select a dataset root containing scenes/.')
    if config.method not in NO_WEIGHTS:
        for r in records(config.method,config.camera):
            if config.checkpoint and r.get('role')=='primary':continue
            if not (ROOT/r['path']).is_file():raise ValueError('Missing recipe weight: '+r['path']+'. Click Download registered weights.')
    if config.method=='generalizing_grasp' and not (Path(config.dataset_root)/'fusion_scenes').is_dir():
        raise ValueError('Generalizing-Grasp needs fusion_scenes; raw single-view scenes are insufficient. See upstream preprocessing instructions.')
    if config.method=='asgrasp' and not (ROOT/'upstream/auxiliary/stereo/asgrasp/gsnet/models').is_dir():
        # Actual repository location comes from the catalogue (not a fixed layout).
        from .config import catalogue
        if not (ROOT/catalogue()['asgrasp']['path']/'gsnet/models').is_dir():
            raise ValueError('ASGrasp gsnet submodule missing. Run ./panda fetch.')
