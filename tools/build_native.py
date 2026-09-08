"""Build pinned native dependencies into a local wheelhouse; keep upstream source intact."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'logs').mkdir(exist_ok=True)
(ROOT/'environments/state').mkdir(parents=True,exist_ok=True)
wheelhouse=ROOT/'environments/wheels';wheelhouse.mkdir(exist_ok=True)
uv=ROOT/'environments/bootstrap/uv'
if not uv.exists():
    import shutil
    uv=shutil.which('uv')
python=ROOT/'.venv/bin/python'
env={**os.environ,'CUDA_HOME':os.environ.get('GRASPPANDA_CUDA_HOME','/usr/local/cuda-11.8'),'MAX_JOBS':os.environ.get('MAX_JOBS','4')}
nvcc_version=subprocess.check_output([str(Path(env['CUDA_HOME'])/'bin/nvcc'),'--version'],text=True)
if 'release 11.8' not in nvcc_version:raise SystemExit('This runtime lock is validated with the CUDA 11.8 toolkit. Set GRASPPANDA_CUDA_HOME accordingly.')
# Auto-select the local GPU unless the caller requests a distribution architecture list.
if 'TORCH_CUDA_ARCH_LIST' not in env:
    import torch
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU unavailable; set TORCH_CUDA_ARCH_LIST explicitly for cross compilation.')
    major,minor=torch.cuda.get_device_capability();env['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
specs=[
 ('pointnet2','upstream/single_view/pointcloud/graspnet_baseline/pointnet2',[]),
 ('knn_pytorch','upstream/single_view/pointcloud/graspnet_baseline/knn',[]),
 ('pointnet2_cuda','upstream/single_view/pointcloud/graspbalance/pointnet2_batch',[]),
 ('KNN','upstream/single_view/pointcloud/graspbalance/KNN',[]),
 ('MinkowskiEngine','environments/sources/MinkowskiEngine',['--blas=openblas','--force_cuda']),
 ('pytorch3d','environments/sources/pytorch3d',[]),
]
lockpath=ROOT/'grasppanda/resources/native_sources.lock.json'
sources=json.loads(lockpath.read_text()) if lockpath.exists() else []
for src in sources:
    dest=ROOT/src['path']
    if not dest.exists():
        subprocess.run(['git','clone',src['repository'],str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout',src['commit']],check=True)
    actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if actual!=src['commit']:raise SystemExit(f'Native source revision mismatch: {dest}')
previous_path=ROOT/'environments/state/native_builds.json'
previous={r['name']:r for r in json.loads(previous_path.read_text())} if previous_path.exists() else {}
results=[]
for name,path,flags in specs:
    repo=ROOT/path;log=ROOT/'logs'/f'wheel_{name}.log'
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    key={'python_abi':sys.implementation.cache_tag,'torch':__import__('torch').__version__,'source_commit':commit,'arch':env['TORCH_CUDA_ARCH_LIST'],'cuda_home':env['CUDA_HOME']}
    old=previous.get(name,{})
    cached=ROOT/old.get('wheel','__missing__')
    if all(old.get(k)==v for k,v in key.items()) and cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest()==old.get('sha256'):
        print('Reusing verified wheel',name,flush=True)
        subprocess.run([str(uv),'pip','install','--python',str(python),'--no-deps',str(cached)],check=True)
        results.append(old)
        continue
    cmd=[str(python),'setup.py','bdist_wheel','--dist-dir',str(wheelhouse),*flags]
    print('Building',name,flush=True)
    with log.open('w') as f:
        p=subprocess.run(cmd,cwd=repo,env={**env,'FORCE_CUDA':'1'},stdout=f,stderr=subprocess.STDOUT)
    result={'name':name,'source':path,'command':cmd,'exit_code':p.returncode,**key,'nvcc_version':nvcc_version,'log':str(log.relative_to(ROOT))}
    if p.returncode==0:
        wheels=[w for w in wheelhouse.glob('*.whl') if w.name.lower().startswith(name.lower()+'-')]
        if not wheels:raise RuntimeError(f'No wheel found for {name}')
        wheel=max(wheels,key=lambda w:w.stat().st_mtime)
        subprocess.run([str(uv),'pip','install','--python',str(python),'--no-deps',str(wheel)],check=True)
        result.update(wheel=str(wheel.relative_to(ROOT)),sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    results.append(result)
    (ROOT/'environments/state/native_builds.json').write_text(json.dumps(results,indent=2)+'\n')
    if p.returncode:raise SystemExit(f'Native build failed: {name}; see {log}')
subprocess.run([str(uv),'pip','install','--python',str(python),'--no-deps',str(ROOT/'upstream/infrastructure/graspnet_api')],check=True)
(ROOT/'environments/state/native_builds.json').write_text(json.dumps(results,indent=2)+'\n')
