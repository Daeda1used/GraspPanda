"""Build a pinned PointCNN++ artifact in the shared Python runtime."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
SOURCE_COMMIT='e8a078e73cb38a1995bd8389f8729caa27d9eee4'
CUTLASS_COMMIT='7127592069c2fe01b041e174ba4345ef9b279671'
PACKAGE='_grasppanda_pointcnnpp'


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def run(args, **kwargs):
    subprocess.run([str(v) for v in args],check=True,**kwargs)


class Namespace(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module and node.module.split('.')[0] in ('layers','internals','sparse_engines','models'):
            node.module=PACKAGE+'.'+node.module
        return node


class BlockGeometry(ast.NodeTransformer):
    def visit_keyword(self,node):
        if node.arg=='receptive_field_scaler' and isinstance(node.value,ast.Constant) and node.value.value==2.5:
            node.value=ast.parse('self.receptive_field_scaler',mode='eval').body
        if node.arg=='kernel_size' and isinstance(node.value,ast.Constant) and node.value.value==3:
            node.value=ast.parse('self.conv2.kernel_size_3[0]',mode='eval').body
        return node


def prepare(source, target):
    """Copy only runtime Python, preserving author kernels and local source pins."""
    target.mkdir(parents=True)
    for folder in ('layers','internals','sparse_engines','models'):
        for path in (source/folder).rglob('*.py'):
            relative=path.relative_to(source)
            if '__pycache__' in relative.parts: continue
            text=path.read_text()
            text=text.replace('from torch.library import triton_op, wrap_triton',
                'from torch._library.triton import triton_op, capture_triton as wrap_triton')
            tree=Namespace().visit(ast.parse(text))
            destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    path=source/'overlays/Pointcept/add/pointcept/models/sparse_unet/unet_pointcnnpp.py'
    tree=ast.parse(path.read_text())
    body=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module in ('pointcept.models.builder','pointcept.models.utils'):continue
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='sys.path.insert':continue
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_'):continue
        if isinstance(node,ast.If) and '__name__' in ast.unparse(node.test):continue
        if isinstance(node,ast.ClassDef):
            node.decorator_list=[v for v in node.decorator_list if 'MODELS.' not in ast.unparse(v)]
            if node.name=='BasicBlock':
                for method in node.body:
                    if isinstance(method,ast.FunctionDef) and method.name=='forward': BlockGeometry().visit(method)
        body.append(node)
    tree.body=body+ast.parse('def offset2bincount(offset):\n    return torch.diff(offset, prepend=offset.new_zeros(1))\n').body
    tree=Namespace().visit(tree)
    (target/'native_unet.py').write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    (target/'__init__.py').write_text('')


def runtime_info():
    import torch
    return dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
        abi=torch._C._GLIBCXX_USE_CXX11_ABI, capability=list(torch.cuda.get_device_capability()))


def verify(record):
    import io
    import torch
    from grasppanda.modules.pointcnnpp import PointCNNFeatures, native_module
    torch.manual_seed(17)
    model=PointCNNFeatures(6,32,base_channels=16,channels=[16,32,48,64,64,48,32,16],
        depths=[1]*8,_native=native_module(record)).cuda().train()
    xyz=torch.rand(2053,3,device='cuda')*.3
    feat=torch.randn(2053,6,device='cuda',requires_grad=True)
    batch=torch.arange(2053,device='cuda')%2
    before=model.network.conv1.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    output=model(xyz,feat,batch)
    assert output.shape==(2053,32) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.all(feat.grad.abs().sum(0)>0)
    optimizer.step(); assert not torch.equal(before,model.network.conv1.weight)
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    model.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    print('PointCNN++ native artifact validation passed.',flush=True)


def build(uv=None,pins=None,env=None):
    env=dict(os.environ if env is None else env)
    env['CUDA_HOME']=env.get('CUDA_HOME',env.get('GRASPPANDA_CUDA_HOME','/usr/local/cuda-11.8'))
    env.setdefault('MAX_JOBS','4')
    info=runtime_info();env.setdefault('TORCH_CUDA_ARCH_LIST','.'.join(map(str,info['capability'])))
    if not pins: pins={v['id']:v for v in json.loads((ROOT/'grasppanda/resources/component_sources.lock.json').read_text())}
    pin=pins['pointcnnpp'];source=ROOT/pin['path']
    source.parent.mkdir(parents=True,exist_ok=True)
    native=ROOT/'environments/native/pointcnnpp';native.mkdir(parents=True,exist_ok=True)
    with (native/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not source.exists():
            run(['git','clone','--no-checkout','--filter=blob:none',pin['repository'],source])
            run(['git','-C',source,'checkout','--detach',pin['commit']])
        head=subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()
        if head!=SOURCE_COMMIT or pin['commit']!=SOURCE_COMMIT:raise ValueError('PointCNN++ source revision differs from its pin')
        run(['git','-C',source,'submodule','update','--init','third_party/cutlass'])
        cutlass=source/'third_party/cutlass'
        if subprocess.check_output(['git','-C',cutlass,'rev-parse','HEAD'],text=True).strip()!=CUTLASS_COMMIT:
            raise ValueError('PointCNN++ CUTLASS source differs from its pin')
        for path in (source,cutlass):
            if subprocess.check_output(['git','-C',path,'status','--porcelain'],text=True).strip():
                raise ValueError('PointCNN++ source has local changes; preserve them before rebuilding')
        fingerprint=hashlib.sha256(json.dumps(dict(runtime=info,source=head,cutlass=CUTLASS_COMMIT,
            builder=digest(__file__),adapter=digest(ROOT/'grasppanda/modules/pointcnnpp.py'),
            options=digest(ROOT/'grasppanda/modules/pointcnnpp_options.py'),
            compiler=subprocess.check_output([str(Path(env['CUDA_HOME'])/'bin/nvcc'),'--version'],text=True),
            architectures=env['TORCH_CUDA_ARCH_LIST']),sort_keys=True).encode()).hexdigest()
        state=native/'state.json'
        if state.is_file():
            old=json.loads(state.read_text())
            if old.get('fingerprint')==fingerprint:
                run([sys.executable,'-m','grasppanda.runtime.build_pointcnnpp','--verify',state],cwd=ROOT,env=env)
                return
        builds=native/'builds';builds.mkdir(exist_ok=True)
        artifact=Path(tempfile.mkdtemp(prefix=fingerprint[:12]+'-',dir=builds))
        workspace=ROOT/'environments/build/pointcnnpp';workspace.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix=fingerprint[:12]+'-',dir=workspace))
        prepare(source,artifact/'python'/PACKAGE)
        shutil.copytree(source/'extensions',work/'extensions',ignore=shutil.ignore_patterns('build','dist','*.egg-info','__pycache__'))
        (work/'third_party').mkdir();(work/'third_party/cutlass').symlink_to(cutlass,target_is_directory=True)
        run([sys.executable,'setup.py','bdist_wheel'],cwd=work/'extensions',env=env)
        wheels=list((work/'extensions/dist').glob('*.whl'))
        if len(wheels)!=1:raise ValueError('PointCNN++ build did not produce exactly one wheel')
        with zipfile.ZipFile(wheels[0]) as wheel:
            for member in wheel.infolist():
                relative=Path(member.filename)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid native wheel path')
            wheel.extractall(artifact/'python')
        notices=artifact/'licenses';notices.mkdir()
        for name in ('LICENSE','LEGAL.md'):shutil.copy2(source/name,notices/name)
        shutil.copy2(cutlass/'LICENSE.txt',notices/'CUTLASS-LICENSE.txt')
        hashes={str(p.relative_to(artifact)):digest(p) for p in artifact.rglob('*') if p.is_file()}
        record=dict(root=str(artifact.resolve()),fingerprint=fingerprint,runtime=info,source=head,
            cutlass=CUTLASS_COMMIT,hashes=hashes)
        pending=artifact/'state.json';pending.write_text(json.dumps(record,indent=2)+'\n')
        run([sys.executable,'-m','grasppanda.runtime.build_pointcnnpp','--verify',pending],cwd=ROOT,env=env)
        with tempfile.NamedTemporaryFile(mode='w',dir=native,delete=False) as stream:
            json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno());temp=Path(stream.name)
        os.replace(temp,state)
        print('PointCNN++ installed in the shared runtime.',flush=True)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--verify':verify(json.loads(Path(sys.argv[2]).read_text()))
    else:build()
