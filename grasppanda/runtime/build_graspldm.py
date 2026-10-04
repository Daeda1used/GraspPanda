"""Compile pinned GraspLDM PVCNN operators into a runtime-specific artifact cache."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '747687402eaebd8fb3ca8d954ed0cf8b772d070e'
NAME = '_pvcnn_backend'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(allow_build=True):
    import torch
    source = ROOT/'upstream/object_centric/graspldm'
    if not source.is_dir(): raise ValueError('GraspLDM source is missing; run ./panda install')
    sha = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if sha!=COMMIT or dirty: raise ValueError('GraspLDM source differs from its pinned author revision')
    identity = dict(source=COMMIT,builder=digest(__file__),python=sys.implementation.cache_tag,
                    torch=torch.__version__,cuda=torch.version.cuda,architecture=list(torch.cuda.get_device_capability()))
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    cache = ROOT/'environments/artifact-cache/graspldm'/key
    if not allow_build and not cache.is_dir(): raise ValueError('GraspLDM operators are not prepared; run ./panda install')
    if allow_build: cache.mkdir(parents=True,exist_ok=True)
    extension = cache/(NAME+sysconfig.get_config_var('EXT_SUFFIX'))
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record = cache/'build.json'
        try: saved = json.loads(record.read_text())
        except (OSError,ValueError): saved = {}
        if saved.get('identity') == identity:
            if not extension.is_file() or digest(extension)!=saved.get('sha256'):
                raise ValueError('GraspLDM cached operator changed; move its cache aside and rerun ./panda install')
        else:
            if not allow_build: raise ValueError('GraspLDM operators are not prepared; run ./panda install')
            from torch.utils import cpp_extension
            cpp_extension.CUDA_HOME = os.environ.get('GRASPPANDA_CUDA_HOME','/usr/local/cuda-11.8')
            os.environ['TORCH_CUDA_ARCH_LIST'] = '.'.join(map(str,identity['architecture']))
            os.environ.setdefault('MAX_JOBS','4')
            base = source/'grasp_ldm/models/modules/ext/pvcnn/modules/functional/src'
            sources = [base/(name+suffix) for name in ('ball_query/ball_query','grouping/grouping',
                       'interpolate/neighbor_interpolate','interpolate/trilinear_devox','sampling/sampling','voxelization/vox')
                       for suffix in ('.cpp','.cu')]+[base/'bindings.cpp']
            print('Preparing native GraspLDM PVCNN operators',flush=True)
            module = cpp_extension.load(name=NAME,sources=list(map(str,sources)),extra_cflags=['-O3','-std=c++17'],
                                        build_directory=str(cache),verbose=False)
            built = Path(module.__file__)
            if built != extension:
                import shutil
                shutil.copy2(built,extension)
            temporary = record.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(identity=identity,sha256=digest(extension)),indent=2)+'\n')
            temporary.replace(record)
            return module
        if NAME in sys.modules:
            if Path(sys.modules[NAME].__file__).resolve()!=extension.resolve():
                raise ValueError('A different PVCNN operator is loaded; launch GraspLDM in its isolated experiment worker')
            return sys.modules[NAME]
        spec = importlib.util.spec_from_file_location(NAME,extension)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sys.modules[NAME] = module
        return module


if __name__ == '__main__':
    build()
