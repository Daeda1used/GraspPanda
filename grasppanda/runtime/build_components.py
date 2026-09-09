"""Fetch pinned CV components and build their isolated native operators."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def build_pointcept(uv, source, env):
    import re
    build = ROOT/'environments/build/pointcept-ops'
    shutil.copytree(source/'libs/pointops', build, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__'))
    launches = guards = 0
    for path in (build/'src').rglob('*.cu'):
        text = path.read_text()
        def current_stream(match):
            fields = match.group(1).split(',')
            if len(fields) != 3:
                raise RuntimeError('Pointcept CUDA launch differs from the pinned source')
            return '<<<' + match.group(1) + ', at::cuda::getCurrentCUDAStream()>>>'
        text, count = re.subn(r'<<<([^>]+)>>>', current_stream, text)
        launches += count
        path.write_text('#include <ATen/cuda/CUDAContext.h>\n' + text)
    for path in (build/'src').rglob('*.cpp'):
        text, count = re.subn(r'(void\s+\w+\s*\([^)]*\bat::Tensor\s+(\w+)[^)]*\)\s*\{)',
            lambda m: m[1] + '\n    const c10::cuda::CUDAGuard device_guard(' + m[2] + '.device());', path.read_text())
        guards += count
        path.write_text('#include <c10/cuda/CUDAGuard.h>\n' + text)
    if (launches, guards) != (27, 16):
        raise RuntimeError('Pointcept native CUDA entry points were not found')
    for path in (build/'functions').glob('*.py'):
        path.write_text(path.read_text().replace('from pointops._C import', 'from _grasppanda_pointcept_cuda import')
                        .replace('from pointops import', 'from _grasppanda_pointops import'))
    (build/'setup.py').write_text("""from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
sources = sorted(str(p) for p in Path('src').rglob('*') if p.suffix in ('.cpp', '.cu'))
setup(name='grasppanda-pointcept-ops', version='0.1.0',
      packages=['_grasppanda_pointops'], package_dir={'_grasppanda_pointops':'functions'},
      ext_modules=[CUDAExtension('_grasppanda_pointcept_cuda', sources,
                   extra_compile_args={'cxx':['-O2'], 'nvcc':['-O2']})],
      cmdclass={'build_ext': BuildExtension})
""")
    subprocess.run([uv, 'pip', 'install', '--python', sys.executable, '--no-deps',
                    '--no-build-isolation', str(build)], env=env, check=True)


def build_mamba_operators(uv, operators, env):
    flags = ['-O3', '-std=c++17', '--expt-relaxed-constexpr',
             '--expt-extended-lambda', '--use_fast_math',
             '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
             '-U__CUDA_NO_BFLOAT16_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
             '-U__CUDA_NO_BFLOAT162_OPERATORS__', '-U__CUDA_NO_BFLOAT162_CONVERSIONS__']
    for name, directory in operators:
        build = ROOT/'environments/build'/name
        shutil.copytree(directory, build/'csrc', dirs_exist_ok=True)
        files = sorted(str(p.relative_to(build)) for p in (build/'csrc').iterdir()
                       if p.suffix in ('.cpp', '.cu'))
        if not files:
            raise RuntimeError(f'{name}: native operator sources are missing')
        # Build every native precision kernel against this interpreter/torch,
        # without installing a global mamba_ssm or downloading a stock wheel.
        (build/'setup.py').write_text(
            'from setuptools import setup\n'
            'from torch.utils.cpp_extension import CUDAExtension, BuildExtension\n'
            f'setup(name={name.lstrip("_").replace("_", "-")!r},version="0.1.0",'
            f'ext_modules=[CUDAExtension({name!r},{files!r},include_dirs=[{str(build/"csrc")!r}],'
            f'extra_compile_args={{"cxx":["-O3","-std=c++17"],"nvcc":{flags!r}}})],'
            'cmdclass={"build_ext":BuildExtension})\n')
        subprocess.run([uv, 'pip', 'install', '--python', sys.executable, '--no-deps',
                        '--no-build-isolation', str(build)], env=env, check=True)


def build_pointmamba(uv, source, causal, env):
    build_mamba_operators(uv, (
        ('_grasppanda_pointmamba_scan', source/'mamba/csrc/selective_scan'),
        ('_grasppanda_causal_conv1d', causal/'csrc')), env)


def build_pcm(uv, source, env):
    native = source/'openpoints/models/PCM'
    build_mamba_operators(uv, (
        ('_grasppanda_pcm_scan', native/'mamba/csrc/selective_scan'),
        ('_grasppanda_pcm_causal', native/'causal-conv1d/csrc')), env)


def build_octree(uv, source, env):
    build = ROOT/'environments/build/octree-dwconv'
    shutil.copytree(source, build, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('.git', 'test', 'build', 'dist', '*.egg-info', '__pycache__'))
    subprocess.run([uv, 'pip', 'install', '--python', sys.executable, '--no-deps',
                    '--no-build-isolation', str(build)], env=env, check=True)


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


def verify_shared_operators(source, openpoints):
    native = source/'openpoints/cpp/pointnet2_batch/src'
    shared = openpoints/'cpp/pointnet2_batch/src'
    files = {p.name for p in native.iterdir() if p.is_file()}
    if not files or files != {p.name for p in shared.iterdir() if p.is_file()}:
        raise RuntimeError('Component and shared OpenPoints operator files differ')
    if any((native/name).read_bytes() != (shared/name).read_bytes() for name in files):
        raise RuntimeError('Component requires a different native point operator build')


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
    for component in ('pointmetabase', 'pointcloudmamba'):
        verify_shared_operators(ROOT/pins[component]['path'], ROOT/pins['openpoints']['path'])
    build_pointcept(uv, ROOT/pins['pointcept']['path'], env)
    build_octree(uv, ROOT/pins['octree-dwconv']['path'], env)
    build_pcm(uv, ROOT/pins['pointcloudmamba']['path'], env)
    build_pointmamba(uv, ROOT/pins['pointmamba']['path'], ROOT/pins['causal-conv1d']['path'], env)
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
