"""Build and verify pinned Swin3D operators in the shared Python runtime."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = '57d513771d5777767518cfe0f42c3707f3880fb7'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args, **kwargs):
    subprocess.run([str(v) for v in args], check=True, **kwargs)


def runtime_info():
    import torch
    return dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                capability=list(torch.cuda.get_device_capability()))


def verify(record):
    import torch
    from grasppanda.modules.swin3d import Swin3DFeatures, native_module
    torch.manual_seed(17)
    torch.set_num_threads(4)
    model = Swin3DFeatures(3, 32, _native=native_module(record), channels=[16, 32, 64],
                          heads=[2, 4, 8], depths=[2, 2, 2]).cuda().train()
    xyz = torch.rand(4096, 3, device='cuda') * .5
    attributes = torch.randn(4096, 3, device='cuda', requires_grad=True)
    batch = torch.arange(4096, device='cuda') % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    output = model(xyz, attributes, batch)
    assert output.shape == (4096, 32) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.all(attributes.grad.abs().sum(0) > 0)
    optimizer.step()
    assert not torch.equal(before, parameter)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    model.eval()
    attributes.grad = None
    model(xyz, attributes, batch)[batch == 0].square().mean().backward()
    assert torch.count_nonzero(attributes.grad[batch == 1]) == 0
    print('Swin3D native artifact validation passed.', flush=True)


def build(uv=None, pins=None, env=None):
    env = dict(os.environ if env is None else env)
    env['CUDA_HOME'] = env.get('CUDA_HOME', env.get('GRASPPANDA_CUDA_HOME', '/usr/local/cuda-11.8'))
    env.setdefault('MAX_JOBS', '4')
    info = runtime_info()
    env.setdefault('TORCH_CUDA_ARCH_LIST', '.'.join(map(str, info['capability'])))
    if pins is None:
        pins = {v['id']: v for v in json.loads((ROOT/'grasppanda/resources/component_sources.lock.json').read_text())}
    pin = pins['swin3d']
    source = ROOT/pin['path']
    source.parent.mkdir(parents=True, exist_ok=True)
    native = ROOT/'environments/native/swin3d'
    native.mkdir(parents=True, exist_ok=True)
    patch = ROOT/'grasppanda/resources/patches/swin3d/native-runtime.patch'
    with (native/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not source.exists():
            run(['git', 'clone', '--no-checkout', '--filter=blob:none', pin['repository'], source])
            run(['git', '-C', source, 'checkout', '--detach', pin['commit']])
        head = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
        if head != SOURCE_COMMIT or pin['commit'] != SOURCE_COMMIT:
            raise ValueError('Swin3D source revision differs from its pin')
        if subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True).strip():
            raise ValueError('Swin3D source has local changes; preserve them before rebuilding')
        identity = dict(runtime=info, source=head, patch=digest(patch), builder=digest(__file__),
                        adapter=digest(ROOT/'grasppanda/modules/swin3d.py'),
                        options=digest(ROOT/'grasppanda/modules/swin3d_options.py'),
                        compiler=subprocess.check_output([str(Path(env['CUDA_HOME'])/'bin/nvcc'), '--version'], text=True),
                        cxx=subprocess.check_output([env.get('CXX', 'c++'), '--version'], text=True),
                        architectures=env['TORCH_CUDA_ARCH_LIST'])
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        state = native/'state.json'
        if state.is_file() and json.loads(state.read_text()).get('fingerprint') == fingerprint:
            run([sys.executable, '-m', 'grasppanda.runtime.build_swin3d', '--verify', state], cwd=ROOT, env=env)
            return
        builds = native/'builds'
        builds.mkdir(exist_ok=True)
        artifact = Path(tempfile.mkdtemp(prefix=fingerprint[:12]+'-', dir=builds))
        workspace = ROOT/'environments/build/swin3d'
        workspace.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=fingerprint[:12]+'-', dir=workspace))
        archive = subprocess.check_output(['git', '-C', source, 'archive', SOURCE_COMMIT])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(work, filter='data')
        run(['git', 'apply', '--check', patch], cwd=work)
        run(['git', 'apply', patch], cwd=work)
        run([sys.executable, 'setup.py', 'bdist_wheel'], cwd=work, env=env)
        wheels = list((work/'dist').glob('*.whl'))
        if len(wheels) != 1:
            raise ValueError('Swin3D build did not produce exactly one wheel')
        with zipfile.ZipFile(wheels[0]) as wheel:
            for member in wheel.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid Swin3D wheel path')
            wheel.extractall(artifact/'python')
        shutil.copy2(source/'LICENSE', artifact/'LICENSE')
        hashes = {str(p.relative_to(artifact)): digest(p) for p in artifact.rglob('*') if p.is_file()}
        record = dict(root=str(artifact.resolve()), fingerprint=fingerprint, runtime=info,
                      source=head, patch=digest(patch), hashes=hashes)
        pending = artifact/'state.json'
        pending.write_text(json.dumps(record, indent=2)+'\n')
        run([sys.executable, '-m', 'grasppanda.runtime.build_swin3d', '--verify', pending], cwd=ROOT, env=env)
        with tempfile.NamedTemporaryFile(mode='w', dir=native, delete=False) as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, state)
        print('Swin3D installed in the shared runtime.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--verify':
        verify(json.loads(Path(sys.argv[2]).read_text()))
    else:
        build()
