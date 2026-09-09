"""Build pinned Flash3D operators and Transformer Engine for the shared interpreter."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT/'grasppanda/resources/flash3d'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, **kwargs):
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def checkout(record):
    source = ROOT/record['path']
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        run(['git', 'clone', '--no-checkout', '--filter=blob:none', record['repository'], source])
        run(['git', '-C', source, 'checkout', '--detach', record['commit']])
    head = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    if head != record['commit']:
        raise ValueError('Flash3D dependency source revision differs from its pin')
    run(['git', '-C', source, 'submodule', 'update', '--init', '--recursive'])
    if subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True).strip():
        raise ValueError('Flash3D dependency source has local changes; preserve them before rebuilding')
    submodules = subprocess.check_output(['git', '-C', source, 'submodule', 'status', '--recursive'], text=True)
    if any(line and line[0] != ' ' for line in submodules.splitlines()):
        raise ValueError('Flash3D dependency submodule differs from its recorded commit')
    return source


def compiler(work):
    cache = ROOT/'environments/downloads/flash3d'
    cache.mkdir(parents=True, exist_ok=True)
    target = work/'cuda-12.2'
    target.mkdir()
    records = json.loads((RESOURCES/'toolchain.lock.json').read_text())
    for name, record in records.items():
        archive = cache/Path(record['relative_path']).name
        if not archive.is_file() or digest(archive) != record['sha256']:
            pending = archive.with_suffix('.partial')
            url = 'https://developer.download.nvidia.com/compute/cuda/redist/'+record['relative_path']
            with urllib.request.urlopen(url, timeout=60) as response, pending.open('wb') as output:
                shutil.copyfileobj(response, output)
            if digest(pending) != record['sha256']:
                raise ValueError('CUDA compiler download checksum mismatch')
            pending.replace(archive)
        extracted = work/name
        extracted.mkdir()
        with tarfile.open(archive) as package:
            package.extractall(extracted, filter='data')
        entries = list(extracted.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise ValueError('Unexpected CUDA compiler archive layout')
        for child in entries[0].iterdir():
            if child.is_dir():
                shutil.copytree(child, target/child.name, dirs_exist_ok=True)
            elif child.name.lower().startswith(('license', 'eula')):
                destination = target/'licenses'/name
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, destination/child.name)
            else:
                shutil.copy2(child, target/child.name)
    (target/'lib64').symlink_to('lib', target_is_directory=True)
    return target


def verify(record):
    """Exercise the installed hierarchy and checkpoint format without a dataset."""
    import io
    import torch
    from grasppanda.modules.flash3d import Flash3DFeatures, native_module
    torch.manual_seed(0)
    native = native_module(record)
    model = Flash3DFeatures(6, 32, _native=native).cuda().train()
    xyz = torch.rand(4099, 3, device='cuda')
    features = torch.cat([xyz, torch.ones_like(xyz)], 1).requires_grad_(True)
    batch = torch.zeros(len(xyz), device='cuda', dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = model(xyz, features, batch)
    if output.shape != (len(xyz), 32) or not torch.isfinite(output).all():
        raise ValueError('Flash3D native installation produced invalid features')
    output.square().mean().backward()
    if features.grad is None or not torch.isfinite(features.grad).all() or not torch.count_nonzero(features.grad):
        raise ValueError('Flash3D native input gradients are missing or invalid')
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Flash3D native parameter gradients are missing or invalid')
    before = model.projection.weight.detach().clone()
    optimizer.step()
    if torch.equal(before, model.projection.weight) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Flash3D native optimizer update is invalid')
    buffer = io.BytesIO()
    state = model.state_dict()
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Flash3D checkpoints must contain tensor-only model state')
    torch.save(state, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu', weights_only=True)
    model.load_state_dict(restored, strict=True)
    if any(not torch.equal(value.detach().cpu(), restored[name]) for name, value in model.state_dict().items()):
        raise ValueError('Flash3D checkpoint values changed during restoration')
    with torch.no_grad():
        if not torch.isfinite(model.eval()(xyz, features.detach(), batch)).all():
            raise ValueError('Flash3D restored model produced invalid features')
    torch.cuda.synchronize()
    print('Flash3D native computation and checkpoint loading verified.', flush=True)


def activate(state, installed, environment):
    """Validate an immutable candidate before atomically replacing the pointer."""
    import tempfile
    candidate = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.candidate-',
                                         suffix='.json', dir=installed, delete=False) as stream:
            candidate = Path(stream.name)
            json.dump(state, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        run([sys.executable, '-m', 'grasppanda.runtime.build_flash3d', '--verify', candidate],
            cwd=ROOT, env=environment)
        os.replace(candidate, installed/'state.json')
    finally:
        if candidate is not None:
            candidate.unlink(missing_ok=True)


def build(uv, pins, environment):
    import fcntl
    installed = ROOT/'environments/native/flash3d'
    installed.mkdir(parents=True, exist_ok=True)
    with (installed/'.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _build(uv, pins, environment)


def _build(uv, pins, environment):
    import importlib.metadata
    import pybind11
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    capability = torch.cuda.get_device_capability()
    if capability not in ((8, 0), (8, 6), (8, 9), (9, 0)):
        raise ValueError('The pinned Flash3D compiler supports Ampere, Ada and Hopper GPUs')
    profile = 'KITTENS_A100' if capability == (8, 0) else ('KITTENS_HOPPER' if capability == (9, 0) else 'KITTENS_4090')
    selected = {key: pins[key] for key in ('flash3d', 'flash3d-transformer-engine', 'flash3d-glog')}
    sources = {key: checkout(record) for key, record in selected.items()}
    cuda = Path(environment.get('CUDA_HOME', '/usr/local/cuda-11.8'))
    cuda_version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
    if 'release 11.8' not in cuda_version:
        raise ValueError('Transformer Engine must build against the shared CUDA 11.8 toolkit')
    cudnn = Path(importlib.metadata.distribution('nvidia-cudnn-cu11').locate_file('nvidia/cudnn'))
    conditions = dict(torch=torch.__version__, python=list(sys.version_info[:2]),
        abi=int(torch._C._GLIBCXX_USE_CXX11_ABI), capabilities=[list(capability)],
        cuda=cuda_version, compiler=subprocess.check_output(['g++', '--version'], text=True),
        cudnn=importlib.metadata.version('nvidia-cudnn-cu11'),
        pybind11=pybind11.__version__, sources=selected,
        resources={p.name:digest(p) for p in RESOURCES.iterdir() if p.is_file()},
        builder=digest(__file__), tools={'cmake':'3.31.6','scikit-build':'0.18.1'},
        compiler_environment={key:environment[key] for key in (
            'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CUDAHOSTCXX', 'CUDACXX',
            'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'TORCH_CUDA_ARCH_LIST') if key in environment})
    fingerprint = hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()
    installed = ROOT/'environments/native/flash3d'
    installed.mkdir(parents=True, exist_ok=True)
    state_file = installed/'state.json'
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        if state.get('fingerprint') == fingerprint and state.get('files') and all(
            (installed/name).is_file() and digest(installed/name) == expected for name, expected in state['files'].items()):
            run([sys.executable, '-m', 'grasppanda.runtime.build_flash3d', '--verify', state_file],
                cwd=ROOT, env=environment)
            print('Flash3D native build matches the shared runtime.', flush=True)
            return
    # Every invocation uses a new directory; previously loaded native files are
    # never overwritten, and a failed build cannot replace the active pointer.
    import tempfile
    work_root = ROOT/'environments/build/flash3d'
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=fingerprint[:12]+'-', dir=work_root))
    tools = work/'build-tools'
    run([uv, 'pip', 'install', '--python', sys.executable, '--target', tools,
         '--no-deps', 'cmake==3.31.6', 'scikit-build==0.18.1'])
    cmake = tools/'cmake/data/bin/cmake'
    env = {**environment, 'PYTHONPATH':str(tools), 'CUDA_HOME':str(cuda),
           'CUDNN_PATH':str(cudnn), 'NVTE_FRAMEWORK':'pytorch', 'NVTE_NO_LOCAL_VERSION':'1',
           'MAX_JOBS':environment.get('MAX_JOBS', '2'),
           'CMAKE_BUILD_PARALLEL_LEVEL':environment.get('MAX_JOBS', '2'),
           'PATH':str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')}
    env.pop('PYTHONHOME', None)
    env.pop('NVTE_RELEASE_BUILD', None)
    toolchain = compiler(work)
    flash = work/'source'
    engine = work/'transformer-engine'
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'build', 'dist', '*.egg-info')
    shutil.copytree(sources['flash3d'], flash, ignore=ignore)
    shutil.copytree(sources['flash3d-transformer-engine'], engine, ignore=ignore)
    for patch in ('attention-backward.patch', 'native-runtime.patch'):
        run(['git', 'apply', RESOURCES/patch], cwd=flash)
    run(['git', 'apply', RESOURCES/'te-cuda118.patch'], cwd=engine)
    setup = engine/'setup.py'
    source = setup.read_text()
    if source.count('cmake_flags=[],') != 1:
        raise ValueError('Transformer Engine CMake entry differs from the pinned source')
    flags = [f'-DCMAKE_CUDA_ARCHITECTURES={capability[0]}{capability[1]}',
             '-DCMAKE_CUDA_COMPILER='+str(cuda/'bin/nvcc'), '-DCUDNN_PATH='+str(cudnn)]
    setup.write_text(source.replace('cmake_flags=[],', 'cmake_flags='+repr(flags)+','))
    engine_wheels = work/'wheels'
    run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', engine_wheels], cwd=engine, env=env)
    wheels = list(engine_wheels.glob('transformer_engine-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Transformer Engine must produce exactly one native wheel')
    prefix = work/'prefix'
    run([cmake, '-S', sources['flash3d-glog'], '-B', work/'glog', '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF',
         '-DWITH_GFLAGS=OFF', '-DWITH_GTEST=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
         '-DCMAKE_INSTALL_PREFIX='+str(prefix)], env=env)
    run([cmake, '--build', work/'glog'], env=env)
    run([cmake, '--install', work/'glog'], env=env)
    native_build = work/'native'
    run([cmake, '-S', RESOURCES, '-B', native_build, '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_ARCHITECTURES={capability[0]}{capability[1]}',
         '-DTK_PROFILE='+profile, '-DCMAKE_CUDA_COMPILER='+str(toolchain/'bin/nvcc'),
         '-DCUDAToolkit_ROOT='+str(toolchain), '-DPython_EXECUTABLE='+sys.executable,
         '-Dpybind11_DIR='+pybind11.get_cmake_dir(), '-Dglog_DIR='+str(prefix/'lib/cmake/glog'),
         '-DFLASH3D_SOURCE='+str(flash), '-DTORCH_INCLUDE_DIRS='+';'.join(include_paths()),
         '-DTORCH_LIBRARY_DIR='+library_paths()[0], '-DTORCH_ABI='+str(conditions['abi'])], env=env)
    run([cmake, '--build', native_build], env=env)
    versions = installed/'builds'
    versions.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=fingerprint[:12]+'-', dir=versions))
    python = destination/'python'
    with zipfile.ZipFile(wheels[0]) as archive:
        for name in archive.namelist():
            if not (python/name).resolve().is_relative_to(python.resolve()):
                raise ValueError('Unsafe Transformer Engine wheel path')
        archive.extractall(python)
    shutil.copytree(flash/'flash3dxfmr', destination/'source/flash3dxfmr', ignore=ignore)
    shutil.copy2(flash/'LICENSE', destination/'source/LICENSE')
    shutil.copytree(toolchain/'licenses', destination/'licenses/cuda')
    shutil.copy2(sources['flash3d-glog']/'COPYING', destination/'licenses/glog.txt')
    shutil.copy2(flash/'third_party/ThunderKittens/LICENSE', destination/'licenses/thunderkittens.txt')
    binary = next(native_build.glob('pshattn*.so'))
    shutil.copy2(binary, destination/binary.name)
    shutil.copy2(native_build/'libcudart.so.12', destination/'libcudart.so.12')
    relative = destination.relative_to(installed)
    state = {**conditions, 'fingerprint':fingerprint,
             'source':str(relative/'source/flash3dxfmr'), 'engine_path':str(relative/'python'),
             'extension':str(relative/binary.name),
             'files':{str(p.relative_to(installed)):digest(p) for p in destination.rglob('*') if p.is_file()}}
    (work/'candidate-state.json').write_text(json.dumps(state, indent=2)+'\n')
    activate(state, installed, env)
    print('Flash3D native components prepared for the shared interpreter.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--verify':
        verify(Path(sys.argv[2]))
        raise SystemExit(0)
    if len(sys.argv) != 1:
        raise SystemExit('Usage: python -m grasppanda.runtime.build_flash3d')
    uv = os.environ.get('UV_BIN') or shutil.which('uv')
    if not uv:
        raise SystemExit('Install uv or set UV_BIN to its executable')
    records = json.loads((ROOT/'grasppanda/resources/component_sources.lock.json').read_text())
    build(uv, {r['id']:r for r in records}, {**os.environ,
        'CUDA_HOME':os.environ.get('GRASPPANDA_CUDA_HOME', '/usr/local/cuda-11.8')})
