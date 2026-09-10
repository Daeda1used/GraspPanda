"""Predicted-instance contact-score refinement in isolated native namespaces."""
from pathlib import Path
import json
import subprocess
import sys
import time


def frame_input(config, row):
    import numpy as np
    import scipy.io
    from PIL import Image
    from ..jobs import digest
    if config.method=='generalizing_grasp':return fused_input(config,row)
    folder=Path(config.dataset_root)/'scenes'/f"scene_{row['scene']:04d}"/config.camera
    depth_path=folder/'depth'/f"{row['frame']:04d}.png"
    meta_path=folder/'meta'/f"{row['frame']:04d}.mat"
    evidence=dict(depth_sha256=digest(depth_path),meta_sha256=digest(meta_path))
    for key,sha in evidence.items():
        if row.get(key) and row[key]!=sha:raise ValueError('Frame changed between prediction and refinement: '+key)
    meta=scipy.io.loadmat(meta_path);intr=meta['intrinsic_matrix']
    scale=float(meta['factor_depth'].reshape(-1)[0])
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid depth scale')
    depth=np.asarray(Image.open(depth_path),dtype=np.float32)/scale
    y,x=np.indices(depth.shape)
    raw=np.stack(((x-intr[0,2])*depth/intr[0,0],(y-intr[1,2])*depth/intr[1,1],depth),axis=-1)
    raw=raw[(depth>0)&np.isfinite(raw).all(-1)].astype(np.float32)
    if not len(raw):raise ValueError('No valid visual depth points for refinement')
    seed=(config.seed+row['scene']*256+row['frame'])%2147483646
    rng=np.random.default_rng(seed);n=config.refinement.get('scene_points',20000)
    indices=rng.choice(len(raw),n,replace=len(raw)<n)
    transform=np.eye(4)
    if config.refinement.get('coordinate_frame','table')=='table':
        table=folder/'cam0_wrt_table.npy';poses=folder/'camera_poses.npy'
        transform=np.load(table,allow_pickle=False)@np.load(poses,allow_pickle=False)[row['frame']]
        evidence.update(table_sha256=digest(table),poses_sha256=digest(poses))
    if not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1],atol=1e-6) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-4):
        raise ValueError('Refinement camera-to-table transform must be rigid')
    return raw,raw[indices],transform,intr,folder/'rgb'/f"{row['frame']:04d}.png",evidence,seed,np.eye(4)


def fused_input(config,row):
    import numpy as np
    import scipy.io
    from ..jobs import digest
    from ..methods.generalizing import cloud,source_path
    folder=Path(config.dataset_root)/'scenes'/f"scene_{row['scene']:04d}"/config.camera
    table_path=folder/'cam0_wrt_table.npy'
    output_from_camera=np.load(table_path,allow_pickle=False)
    if not np.isfinite(output_from_camera).all() or not np.allclose(output_from_camera[3],[0,0,0,1],atol=1e-6) or not np.allclose(output_from_camera[:3,:3].T@output_from_camera[:3,:3],np.eye(3),atol=1e-4):raise ValueError('Fused camera calibration must be rigid')
    camera_from_output=np.linalg.inv(output_from_camera)
    xyz=cloud(config)['xyz']
    raw=(xyz@camera_from_output[:3,:3].T+camera_from_output[:3,3]).astype(np.float32)
    evidence=dict(fusion_points_sha256=digest(source_path(config)),table_sha256=digest(table_path))
    if evidence['fusion_points_sha256']!=row['fusion_points_sha256']:raise ValueError('Fused observations changed after prediction')
    seed=(config.seed+row['scene']*256)%2147483646
    n=config.refinement.get('scene_points',20000)
    selected=np.random.default_rng(seed).choice(len(raw),n,replace=len(raw)<n)
    transform=output_from_camera if config.refinement.get('coordinate_frame','table')=='table' else np.eye(4)
    meta=folder/'meta/0000.mat';intr=scipy.io.loadmat(meta)['intrinsic_matrix'] if meta.is_file() else None
    if meta.is_file():evidence['meta_sha256']=digest(meta)
    return raw,raw[selected],transform,intr,folder/'rgb/0000.png',evidence,seed,output_from_camera


def segment(config, requests):
    import torch
    import numpy as np
    from dataclasses import replace
    from ..worker import prepare
    from ..methods.scale_balanced_sampling import Segmenter
    from . import weight_paths
    prepare('scale_balanced_grasp');torch.set_num_threads(4)
    opts={**config.refinement.get('segmentation',{}), 'type':'object_balanced',
          'segmentation_checkpoint':weight_paths(config)['segmentation_checkpoint']}
    dsn=Segmenter(replace(config,method='scale_balanced_grasp',modules={'sampling':opts},refinement={}))
    for request in requests:
        torch.manual_seed(request['seed']);np.random.seed(request['seed'])
        cloud=np.load(request['points'],allow_pickle=False)
        inputs={'point_clouds':torch.from_numpy(cloud)[None].cuda()}
        with torch.inference_mode():dsn(inputs)
        labels=inputs['seed_cluster'][0].cpu().numpy()
        np.save(request['labels'],labels)
        print('Predicted refinement instances for '+Path(request['points']).name,flush=True)


def transform_grasps(values, transform):
    result=values.copy()
    result[:,13:16]=values[:,13:16]@transform[:3,:3].T+transform[:3,3]
    result[:,4:13]=(transform[:3,:3]@values[:,4:13].reshape(-1,3,3)).reshape(-1,9)
    return result


def select_candidates(grasps, points, labels, options):
    import numpy as np
    from scipy.spatial import cKDTree
    from graspnetAPI import GraspGroup
    if not len(grasps):return [], np.zeros(0,dtype=np.int64)
    distance,index=cKDTree(points).query(grasps[:,13:16])
    assignments=labels[index].copy();assignments[distance>options.get('max_assignment_distance',.03)]=0
    for label in np.unique(assignments):
        if label and (labels==label).sum()<options.get('min_object_points',64):assignments[assignments==label]=0
    if options.get('unassigned','keep')=='error' and (assignments==0).any():
        raise ValueError('Some grasps have no sufficiently large predicted object within max_assignment_distance')
    suppressed=GraspGroup(grasps).nms(options.get('nms_translation',.03),options.get('nms_angle',30.)*np.pi/180).grasp_group_array
    # Match the complete candidate, not just its center: different orientations can share a center.
    lookup={}
    for i,row in enumerate(grasps):lookup.setdefault(tuple(row.tolist()),[]).append(i)
    remaining=[]
    for row in suppressed:
        indices=lookup.get(tuple(row.tolist()))
        if not indices:raise ValueError('Native NMS returned a candidate outside its input')
        remaining.append(indices.pop(0))
    selected=[]
    for label in sorted(set(assignments)-{0}):
        ids=[i for i in remaining if assignments[i]==label]
        ids.sort(key=lambda i:(-grasps[i,0],i))
        selected.extend(ids[:options.get('top_per_instance',5)])
    return selected,assignments


def run(config, out):
    import numpy as np
    from ..config import ROOT
    from ..jobs import digest
    from . import manifest_artifacts, weight_paths
    result=json.loads((out/'refinement-input.json').read_text())
    cache=out/'prepared/refinement';cache.mkdir(parents=True,exist_ok=True)
    frames=[];requests=[]
    for row in result['frames']:
        raw,points,transform,intr,rgb,evidence,seed,output_from_camera=frame_input(config,row)
        folder=cache/f"scene_{row['scene']:04d}"/config.camera;folder.mkdir(parents=True,exist_ok=True)
        cloud=folder/f"{row['frame']:04d}-points.npy";np.save(cloud,points)
        label=folder/f"{row['frame']:04d}-instances.npy"
        requests.append(dict(points=str(cloud),labels=str(label),seed=seed))
        frames.append((row,transform,intr,rgb,evidence,seed,folder,label,output_from_camera))
    request_path=cache/'segmentation.json';request_path.write_text(json.dumps(requests,indent=2))
    subprocess.run([sys.executable,'-m','grasppanda.refinement.pipeline',str(out/'config.json'),str(out),'--segment',str(request_path)],cwd=ROOT,check=True)
    from ..worker import prepare,overlay
    source=prepare('generalizing_grasp')
    import torch
    torch.set_num_threads(4);torch.manual_seed(config.seed)
    from .contact_score import ContactScore
    from collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup
    paths=weight_paths(config)
    refiner=ContactScore(source,paths['contact_checkpoint'],paths['score_checkpoint'],config.refinement)
    for number,(row,transform,intr,rgb,evidence,seed,folder,label_path,output_from_camera) in enumerate(frames):
        raw,_,current_transform,_,_,current_evidence,_,_=frame_input(config,row)
        if current_evidence!=evidence or not np.array_equal(current_transform,transform):raise ValueError('Visual inputs changed while refinement was running')
        points=np.load(requests[number]['points'],allow_pickle=False)
        start=time.monotonic();path=out/row['prediction']
        original=np.load(path,allow_pickle=False)
        if digest(path)!=row['prediction_sha256']:raise ValueError('Base predictions changed before refinement')
        if original.ndim!=2 or original.shape[1]!=17 or not np.isfinite(original).all():raise ValueError('Refinement expects finite GraspNet [N,17] predictions')
        labels=np.load(label_path,allow_pickle=False)
        if labels.shape!=(len(points),) or labels.dtype.kind not in 'iu' or (labels<0).any():raise ValueError('DSN label rows do not match the visual input')
        original_output=original.copy()
        original=transform_grasps(original,np.linalg.inv(output_from_camera))
        selected,assignments=select_candidates(original,points,labels,config.refinement)
        transformed=transform_grasps(original,transform)
        target_points=points@transform[:3,:3].T+transform[:3,3]
        refined=original.copy();refined_output=original_output.copy();records=[]
        for index in selected:
            cloud=target_points[labels==assignments[index]]
            updated,record=refiner.refine(transformed[index],cloud,(seed+index)%2147483646)
            if record['accepted_step']:
                refined[index]=transform_grasps(updated[None],np.linalg.inv(transform))[0]
                refined_output[index]=transform_grasps(refined[index:index+1],output_from_camera)[0]
            records.append(dict(candidate=index,instance=int(assignments[index]),**record))
            print(f"Contact-score refinement: frame {number+1}, candidate {len(records)}/{len(selected)}",flush=True)
        if config.refinement.get('output','all')=='selected':
            keep=set(selected)
            if config.refinement.get('unassigned','keep')=='keep':keep.update(np.where(assignments==0)[0].tolist())
            refined=refined[sorted(keep)];refined_output=refined_output[sorted(keep)]
        elif config.refinement.get('unassigned','keep')=='drop':
            refined=refined[assignments>0];refined_output=refined_output[assignments>0]
        before_collision=len(refined)
        if config.collision_thresh>0 and len(refined):
            group=GraspGroup(refined)
            detector=ModelFreeCollisionDetector(raw,voxel_size=.01)
            keep=~detector.detect(group,approach_dist=.05,collision_thresh=config.collision_thresh)
            refined=refined[keep];refined_output=refined_output[keep]
        np.save(folder/f"{row['frame']:04d}-original.npy",original_output)
        details=folder/f"{row['frame']:04d}-optimization.json";details.write_text(json.dumps(records,indent=2)+'\n')
        np.save(path,refined_output)
        row['prediction_sha256']=digest(path);row['grasps_after_collision']=len(refined)
        row['refinement']=dict(selected=len(selected),accepted=sum(r['accepted_step']>0 for r in records),
            before_collision=before_collision,after_collision=len(refined),seconds=time.monotonic()-start,
            coordinate_frame=config.refinement.get('coordinate_frame','table'),object_source='predicted_dsn',
            points_sha256=digest(requests[number]['points']),instances_sha256=digest(label_path),
            optimization=str(details.relative_to(out)),optimization_sha256=digest(details),**evidence)
        if not number and intr is not None:overlay(rgb,refined,intr,out/'preview.png')
    manifest_path=out/'predictions/manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['files']={str(Path(row['prediction']).relative_to('predictions')):row['prediction_sha256'] for row in result['frames']}
    manifest['refinement_artifacts']=manifest_artifacts(config)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    result['refinement']=dict(configuration=config.refinement,artifacts=manifest['refinement_artifacts'],
        protocol='Predicted-instance contact-score adaptation; observed XYZ and calibration only. Native method scores are retained. Collision filtering is repeated after refinement.')
    (out/'refinement-result.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    from ..config import Experiment
    config=Experiment.from_dict(json.loads(Path(sys.argv[1]).read_text()));out=Path(sys.argv[2]).resolve()
    if len(sys.argv)>3 and sys.argv[3]=='--segment':segment(config,json.loads(Path(sys.argv[4]).read_text()))
    else:run(config,out)


if __name__=='__main__':main()
