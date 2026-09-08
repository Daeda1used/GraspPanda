"""Fetch pinned CV components and build their isolated native operators."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def build_vmamba(uv, source, env):
    build=ROOT/'environments/build/vmamba-scan'
    shutil.copytree(source/'kernels/selective_scan',build,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__','dist'))
    subprocess.run([uv,'pip','install','--python',sys.executable,'--no-deps',
                    '--no-build-isolation',str(build)],env=env,check=True)


def build_deepla(uv, source, env):
    build = ROOT/'environments/build/deepla-ops'
    shutil.copytree(source/'utils/cutils/srcs', build/'srcs', dirs_exist_ok=True)
    # Native launches use the default stream. Preserve the operator arithmetic
    # while honoring PyTorch's selected device and current CUDA stream.
    for path in (build/'srcs').glob('*.cu'):
        text = path.read_text()
        if text.count('<<<grid, block>>>') != 3:
            raise RuntimeError('DeepLA CUDA launch layout differs from the pinned source')
        text = '#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>\n' + text
        text = text.replace('    const uint64_t grid =',
            '    const c10::cuda::CUDAGuard device_guard(output.device());\n    const uint64_t grid =')
        text = text.replace('<<<grid, block>>>',
            '<<<grid, block, 0, at::cuda::getCurrentCUDAStream(output.get_device())>>>')
        path.write_text(text)
    (build/'setup.py').write_text("""from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
sources = sorted(str(p) for p in Path('srcs').glob('*') if p.suffix in ('.cpp', '.cu'))
setup(name='grasppanda-deepla-ops', version='0.1.0',
      ext_modules=[CUDAExtension('_grasppanda_deepla_cuda', sources,
                   extra_compile_args={'cxx':['-O3'], 'nvcc':['-O3']})],
      cmdclass={'build_ext': BuildExtension})
""")
    subprocess.run([uv, 'pip', 'install', '--python', sys.executable, '--no-deps',
                    '--no-build-isolation', str(build)], env=env, check=True)


def verify_pointmeta_operators(source, openpoints):
    native = source/'openpoints/cpp/pointnet2_batch/src'
    shared = openpoints/'cpp/pointnet2_batch/src'
    files = {p.name for p in native.iterdir() if p.is_file()}
    if not files or files != {p.name for p in shared.iterdir() if p.is_file()}:
        raise RuntimeError('PointMetaBase and shared OpenPoints operator files differ')
    if any((native/name).read_bytes() != (shared/name).read_bytes() for name in files):
        raise RuntimeError('PointMetaBase requires a different native operator build')


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
    verify_pointmeta_operators(ROOT/pins['pointmetabase']['path'], ROOT/pins['openpoints']['path'])
    build_vmamba(uv, ROOT/pins['vmamba']['path'], env)
    build_deepla(uv, ROOT/pins['deepla']['path'], env)
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
