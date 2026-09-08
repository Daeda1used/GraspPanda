"""Generate Generalizing-Grasp SDF targets with the pinned native sampler."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grasppanda.config import catalogue
from grasppanda.jobs import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True,help='Writes models/ID/grid_sampled_sdf.npz below this directory')
    parser.add_argument('--objects',type=int,nargs='+',default=list(range(88)))
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if any(i<0 or i>87 for i in args.objects):parser.error('Object IDs must be in [0,87]')
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import numpy as np
    import trimesh
    source=ROOT/catalogue()['generalizing_grasp']['path']/'dataset/grid_sample.py'
    spec=importlib.util.spec_from_file_location('native_sdf_preparation',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for obj in args.objects:
        mesh_path=args.dataset_root/'models'/f'{obj:03d}'/'nontextured.ply'
        target=args.output_root/'models'/f'{obj:03d}';target.mkdir(parents=True,exist_ok=True)
        output=target/'grid_sampled_sdf.npz'
        if output.exists():raise ValueError(f'Output already exists; choose a fresh output directory: {output}')
        np.random.seed(args.seed+obj)
        start=time.monotonic()
        mesh=trimesh.load(mesh_path,process=False)
        module.grid_sample_sdf(mesh,str(target))
        with np.load(output) as data:
            points,sdf=data['points'],data['sdf']
            if points.shape!=sdf.shape+(3,) or not np.isfinite(points).all() or not np.isfinite(sdf).all():
                raise ValueError('Native sampler returned invalid SDF arrays')
            if not (sdf<0).any() or not (sdf>0).any():raise ValueError('SDF lacks both interior and exterior values; inspect the mesh')
            record=dict(object_id=obj,seed=args.seed+obj,shape=list(sdf.shape),minimum=float(sdf.min()),maximum=float(sdf.max()),
                mesh_sha256=digest(mesh_path),sdf_sha256=digest(output),generator_sha256=digest(source),
                seconds=time.monotonic()-start,protocol='Verbatim native grid: 3 cm padding, longest padded side / 128 spacing, sample-based mesh_to_sdf defaults.')
        (target/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record);print(json.dumps(record),flush=True)
    (args.output_root/'sdf_preparation.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
