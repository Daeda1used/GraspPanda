"""Build method-specific dependencies in the same environment; original clones stay clean."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
UV = str(ROOT / 'environments/bootstrap/uv') if (ROOT / 'environments/bootstrap/uv').exists() else shutil.which('uv')
ENV = {**os.environ, 'CUDA_HOME': os.environ.get('GRASPPANDA_CUDA_HOME', '/usr/local/cuda-11.8'), 'MAX_JOBS': os.environ.get('MAX_JOBS', '4')}
for key in ('CONDA_PREFIX','CONDA_DEFAULT_ENV','PYTHONPATH','PYTHONHOME'):
    ENV.pop(key,None)
if 'TORCH_CUDA_ARCH_LIST' not in ENV:
    import torch
    ENV['TORCH_CUDA_ARCH_LIST'] = '.'.join(map(str, torch.cuda.get_device_capability()))


def run(name, command, cwd=ROOT):
    print('Building/installing', name, flush=True)
    with (ROOT / 'logs' / (name + '-install.log')).open('w') as log:
        subprocess.run(command, cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)


def build_center_dependencies():
    import pybind11
    source=ROOT/'environments/sources/manifold_python'
    run('manifold-fetch',['git','submodule','update','--init','--recursive'],cwd=source)
    build=ROOT/'environments/build/manifold_python'
    # CMake records absolute source, build and interpreter paths. Recreate this
    # generated tree so moving the checkout or replacing its runtime is safe.
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(source,build,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','build','*.egg-info'))
    # Keep the native watertight-mesh algorithm; use the shared Python 3.11
    # compatible pybind11 headers instead of the old bundled bindings.
    cmake=build/'CMakeLists.txt'
    text=cmake.read_text().replace('add_subdirectory(3rdparty/pybind11)','find_package(pybind11 CONFIG REQUIRED)')
    cmake.write_text(text)
    ENV['CMAKE_PREFIX_PATH']=pybind11.get_cmake_dir()
    ENV['CMAKE_BUILD_PARALLEL_LEVEL']=ENV['MAX_JOBS']
    run('manifold',[UV,'pip','install','--python',sys.executable,'--no-deps','--no-build-isolation',str(build)])


def main():
    build_center_dependencies()
    run('pyg-extensions', [UV, 'pip', 'install', '--python', sys.executable, '--no-deps',
                          'torch-scatter==2.1.2+pt25cu118', 'torch-cluster==1.6.3+pt25cu118',
                          '-f', 'https://data.pyg.org/whl/torch-2.5.1+cu118.html'])
    zero = ROOT / 'upstream/single_view/rgbd/zerograsp'
    ofe = zero / 'submodules/octree_feature_extractor'
    run('ofe-fetch', ['git', '-c', 'url.https://github.com/.insteadOf=git@github.com:',
                      'submodule', 'update', '--init', '--recursive', 'submodules/octree_feature_extractor'], cwd=zero)
    run('ofe', [UV, 'pip', 'install', '--python', sys.executable, '--no-deps', '--no-build-isolation', str(ofe)])
    # Compile GFLA's exact C++ source into a separate namespace directory.
    import pybind11
    extension = ROOT / 'environments/extensions'
    extension.mkdir(exist_ok=True)
    source = ROOT / 'upstream/single_view/rgbd/gfla/models/my_grasp_nms/grasp_nms.cpp'
    output = extension / ('grasp_nms_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))
    run('gfla-nms', ['c++', '-O2', '-shared', '-std=c++17', '-fPIC', '-I'+pybind11.get_include(),
                     '-I'+sysconfig.get_path('include'), '-I/usr/include/eigen3', str(source), '-o', str(output)])
    original = ROOT / 'environments/sources/scikit-geometry'
    build = ROOT / 'environments/build/scikit-geometry'
    shutil.copytree(original, build, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git', 'build', '*.egg-info'))
    patch = ROOT / 'grasppanda/resources/patches/scikit-geometry/pybind11-property-lifetime.patch'
    run('skgeom-patch', ['git', 'apply', str(patch)], cwd=build)
    record_path = ROOT/'environments/state/extra_builds.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    cached = ROOT/previous.get('skgeom_wheel','__missing__')
    patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    geometry_commit = subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
    reuse = (previous.get('skgeom_patch_sha256')==patch_hash and cached.is_file()
             and previous.get('skgeom_source_commit')==geometry_commit
             and previous.get('python_abi')==sys.implementation.cache_tag
             and hashlib.sha256(cached.read_bytes()).hexdigest()==previous.get('skgeom_wheel_sha256'))
    if not reuse:
        run('skgeom-build', [sys.executable,str(ROOT/'grasppanda/runtime/build_geometry.py')])
    wheels = list((ROOT/'environments/wheels').glob('skgeom-*-cp311-*.whl'))
    if not wheels: raise RuntimeError('Missing geometry wheel')
    geometry_wheel = cached if reuse else max(wheels,key=lambda p:p.stat().st_mtime)
    run('skgeom', [UV, 'pip', 'install', '--python', sys.executable, '--no-deps', str(geometry_wheel)])
    result = {'python': sys.executable, 'cuda': ENV['CUDA_HOME'], 'arch': ENV['TORCH_CUDA_ARCH_LIST'],
              'python_abi':sys.implementation.cache_tag,'skgeom_source_commit':geometry_commit,
              'gfla_nms_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'skgeom_patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'skgeom_wheel':str(geometry_wheel.relative_to(ROOT)),
              'skgeom_wheel_sha256':hashlib.sha256(geometry_wheel.read_bytes()).hexdigest(),
              'ofe_commit': subprocess.check_output(['git', '-C', str(ofe), 'rev-parse', 'HEAD'], text=True).strip()}
    (ROOT / 'environments/state/extra_builds.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
