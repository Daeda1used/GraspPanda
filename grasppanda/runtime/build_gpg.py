"""Build the pinned GPG implementation into the shared runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import platform

ROOT = Path(__file__).resolve().parents[2]


def main():
    import fcntl
    build = ROOT / 'environments/build/gpg'
    build.mkdir(parents=True, exist_ok=True)
    with (build/'install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build_extension(build)


def build_extension(build):
    import pybind11
    pins = json.loads((ROOT / 'grasppanda/resources/native_sources.lock.json').read_text())
    pin = next(p for p in pins if p.get('id') == 'gpg')
    source = ROOT / pin['path']
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin['commit']: raise RuntimeError('GPG source revision differs from its lock')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True):
        raise RuntimeError('Restore the pinned GPG source before building')
    packages = subprocess.check_output(['pkg-config', '--list-all'], text=True).splitlines()
    pcl = next((line.split()[0] for line in packages if line.split()[0].split('-')[0] == 'pcl_common'), None)
    if not pcl: raise RuntimeError('Install libpcl-dev using docs/INSTALL.md before building GPG')
    fingerprint = dict(source_commit=actual, python_abi=sysconfig.get_config_var('SOABI'),
        machine=platform.machine(), pybind11=pybind11.__version__,
        pcl_version=subprocess.check_output(['pkg-config', '--modversion', pcl], text=True).strip(),
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binding_sha256=hashlib.sha256((ROOT/'grasppanda/resources/gpg_binding.cpp').read_bytes()).hexdigest())
    extension = '_grasppanda_gpg' + sysconfig.get_config_var('EXT_SUFFIX')
    binary = build/'binary'
    state = ROOT/'environments/state/gpg-build.json'
    destination = Path(sysconfig.get_path('platlib'))/extension
    old = json.loads(state.read_text()) if state.exists() else {}

    def install():
        temporary = destination.with_suffix(destination.suffix+'.tmp')
        shutil.copy2(binary/extension, temporary)
        os.replace(temporary, destination)

    if (old.get('fingerprint') == fingerprint and (binary/extension).is_file() and
            hashlib.sha256((binary/extension).read_bytes()).hexdigest() == old.get('binary_sha256')):
        install()
        print('Reusing verified GPG extension in the shared runtime.')
        return
    original = build / 'gpg'
    if original.exists(): shutil.rmtree(original)
    shutil.copytree(source, original, ignore=shutil.ignore_patterns('.git', 'build'))
    cmake = original / 'CMakeLists.txt'
    text = cmake.read_text()
    old = 'add_library(${PROJECT_NAME}_grasp_candidates_generator SHARED'
    if text.count(old) != 1: raise RuntimeError('Unexpected GPG build layout')
    cmake.write_text(text.replace(old, 'add_library(${PROJECT_NAME}_grasp_candidates_generator STATIC'))
    sample = original / 'src/gpg/cloud_camera.cpp'
    text = sample.read_text()
    old = 'random_sample.setSample(num_samples);'
    if text.count(old) != 1: raise RuntimeError('Unexpected GPG sample implementation')
    sample.write_text(text.replace(old, old + '\n  random_sample.setSeed(static_cast<unsigned int>(std::rand()));'))
    shutil.copy2(ROOT / 'grasppanda/resources/gpg_binding.cpp', build / 'binding.cpp')
    (build / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.20)
project(grasppanda_gpg LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(gpg)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_grasppanda_gpg MODULE binding.cpp)
target_link_libraries(_grasppanda_gpg PRIVATE gpg_grasp_candidates_generator)
''')
    (ROOT / 'logs').mkdir(exist_ok=True)
    env = {**os.environ, 'CMAKE_BUILD_PARALLEL_LEVEL': os.environ.get('MAX_JOBS', '4')}
    with (ROOT / 'logs/gpg-install.log').open('w') as log:
        for command in ([
            'cmake', '-S', str(build), '-B', str(binary), '-DCMAKE_BUILD_TYPE=Release',
            '-DPYTHON_EXECUTABLE=' + sys.executable, '-Dpybind11_DIR=' + pybind11.get_cmake_dir(), '-DCMAKE_SKIP_RPATH=TRUE'],
            ['cmake', '--build', str(binary), '--target', '_grasppanda_gpg']):
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    install()
    state.parent.mkdir(exist_ok=True)
    state.write_text(json.dumps(dict(fingerprint=fingerprint,
        binary_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()), indent=2) + '\n')
    print('GPG installed in the shared runtime.')


if __name__ == '__main__': main()
