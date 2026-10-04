"""Build TARGO's native operators without editing the pinned author checkout."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
COMMIT = 'e71d00e6c081aa39a164a6a402375aa75173ff80'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(allow_build=True):
    import torch
    source = ROOT/'upstream/target_driven/targo'
    if not source.is_dir(): raise ValueError('TARGO author source is missing; run ./panda install')
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if commit != COMMIT or dirty: raise ValueError('TARGO source differs from its pinned author revision')
    patch = ROOT/'grasppanda/resources/patches/targo/shared-runtime.patch'
    identity = dict(source=COMMIT,patch=digest(patch),builder=digest(__file__),
                    python=sys.implementation.cache_tag,torch=torch.__version__,cuda=torch.version.cuda,
                    architecture=list(torch.cuda.get_device_capability()))
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    cache = ROOT/'environments/artifact-cache/targo'/key
    if not allow_build and not cache.is_dir(): raise ValueError('Native TARGO runtime is not prepared; run ./panda install')
    if allow_build: cache.mkdir(parents=True,exist_ok=True)
    target = cache/'source'
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record = cache/'build.json'
        try: saved = json.loads(record.read_text())
        except (OSError,ValueError): saved = {}
        if saved.get('identity') == identity:
            if all((target/name).is_file() and digest(target/name)==sha for name,sha in saved['files'].items()):
                return target
            raise ValueError('Prepared TARGO files changed; move its generated cache aside and rerun ./panda install')
        if not allow_build: raise ValueError('Native TARGO runtime is not prepared; run ./panda install')
        if target.exists(): shutil.rmtree(target)
        target.mkdir()
        for name in ('src','setup','scripts'):
            shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns('__pycache__','build','*.so','*.egg-info'))
        for name in ('LICENSE','train_targo.py'):
            shutil.copy2(source/name,target/name)
        env = {**os.environ,'CUDA_HOME':os.environ.get('GRASPPANDA_CUDA_HOME','/usr/local/cuda-11.8'),
               'TORCH_CUDA_ARCH_LIST':'.'.join(map(str,identity['architecture'])),
               'MAX_JOBS':os.environ.get('MAX_JOBS','4'),'PYTHONDONTWRITEBYTECODE':'1',
               'GIT_CEILING_DIRECTORIES':str(cache.resolve())}
        for key in ('CONDA_PREFIX','CONDA_DEFAULT_ENV','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
        subprocess.run(['git','apply','--check',str(patch)],cwd=target,env=env,check=True)
        subprocess.run(['git','apply',str(patch)],cwd=target,env=env,check=True)
        for name,directory,script in (
            ('mesh',target,'scripts/convonet_setup.py'),
            ('chamfer',target/'src/shape_completion/chamfer_dist','setup.py')):
            print('Preparing native TARGO '+name+' operator',flush=True)
            with (cache/(name+'-build.log')).open('w') as log:
                subprocess.run([sys.executable,script,'build_ext','--inplace'],cwd=directory,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        files = {str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()
                 and 'build' not in p.relative_to(target).parts and '__pycache__' not in p.parts}
        temporary = record.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(identity=identity,files=files),indent=2)+'\n')
        temporary.replace(record)
    return target


if __name__ == '__main__':
    build()
