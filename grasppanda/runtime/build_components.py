"""Fetch pinned CV components and build their isolated native operators."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    uv=ROOT/'environments/bootstrap/uv'
    uv=str(uv) if uv.exists() else shutil.which('uv')
    if not uv:raise SystemExit('Install uv before building components')
    env={**os.environ,'CUDA_HOME':os.environ.get('GRASPPANDA_CUDA_HOME','/usr/local/cuda-11.8'),
         'MAX_JOBS':os.environ.get('MAX_JOBS','4')}
    if 'TORCH_CUDA_ARCH_LIST' not in env:
        import torch
        env['TORCH_CUDA_ARCH_LIST']='.'.join(map(str,torch.cuda.get_device_capability()))
    pins={r['id']:r for r in json.loads((ROOT/'grasppanda/resources/component_sources.lock.json').read_text())}
    for record in pins.values():
        dest=ROOT/record['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not (dest/'.git').exists():
            subprocess.run(['git','clone','--depth','1',record['repository'],str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',record['commit']],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',record['commit']],check=True)
        actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        if actual!=record['commit']:raise SystemExit(f'Component source revision mismatch: {record["id"]}')
    # Native packaging generates the version module used by source imports.
    # The broader robotics application dependencies are outside this adapter.
    subprocess.run([uv,'build',str(ROOT/pins['finegrasp']['path']),'--wheel',
                    '--out-dir',str(ROOT/'environments/wheels')],env=env,check=True)
    source=ROOT/pins['openpoints']['path']/'cpp/pointnet2_batch'
    build=ROOT/'environments/build/openpoints-ops'
    shutil.copytree(source,build,dirs_exist_ok=True,ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__'))
    setup=build/'setup.py'
    text=setup.read_text()
    for original in ("name='pointnet2_cuda'", "CUDAExtension('pointnet2_batch_cuda'"):
        if original not in text:raise RuntimeError('OpenPoints operator packaging differs from the registered source')
    text=text.replace("name='pointnet2_cuda'","name='grasppanda-openpoints-ops'").replace("CUDAExtension('pointnet2_batch_cuda'","CUDAExtension('_grasppanda_openpoints_cuda'")
    setup.write_text(text)
    # The official OpenPoints operator and GraspBalance operator have the same
    # package/module names but different source. Give this build its own name.
    subprocess.run([uv,'pip','install','--python',sys.executable,'--no-deps','--no-build-isolation',str(build)],env=env,check=True)
    source=ROOT/pins['pointmlp']['path']/'pointnet2_ops_lib'
    build=ROOT/'environments/build/pointmlp-ops'
    shutil.copytree(source,build,dirs_exist_ok=True,ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__'))
    setup=build/'setup.py'
    text=setup.read_text()
    original='os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
    if original not in text:raise RuntimeError('PointMLP operator packaging differs from the registered source')
    setup.write_text(text.replace(original, '# GPU architecture is supplied by the shared runtime installer.'))
    subprocess.run([uv,'pip','install','--python',sys.executable,'--no-deps','--no-build-isolation',str(build)],env=env,check=True)


if __name__=='__main__':main()
