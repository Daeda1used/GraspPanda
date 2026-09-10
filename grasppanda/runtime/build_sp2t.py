"""Prepare checksum-pinned SP2T modules in an isolated shared-runtime namespace."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = 'e355eead80df1fc75297f55d164d64151efeaab5'
SOURCE_HASHES = {
    "models/builder.py": "9f2f79a242987167aeb0f8197bf4ebbb48094326d26c0ba295ca2069cd3e926c",
    "models/modules.py": "b50f5713b2a4a10af225bafdbace8a32c74821e6820fc545fb28ba6d0d16a1ed",
    "models/point_prompt_training/prompt_driven_normalization.py": "3b9180c8dc4820e62f3e159839666c5a304438f6d4b5f0dc69941ca684aec041",
    "models/sparse_proxy_point_transformer/__init__.py": "57a54daac7a378efffe1297f57c114c787b7b370b86e62a50212a9f828255e06",
    "models/sparse_proxy_point_transformer/basic_blocks.py": "ad51e8317af5b677c773c624f07c7bf8591979cb814e5c3438fc191848ee0a2f",
    "models/sparse_proxy_point_transformer/proxy_fuser.py": "12dd0187d27ab4de19f9bba16a22c95c3a081c20138209fe7c705714bc3493f0",
    "models/sparse_proxy_point_transformer/proxy_initializer.py": "faec62d2e0a221fd862a10efedb270001fcff9ed564af4af508ba2485733274a",
    "models/sparse_proxy_point_transformer/sparse_attention.py": "446ac53f771135d97c4536ee9e09cb03157cb1ce5dd691a8f37e1851ccd20060",
    "models/sparse_proxy_point_transformer/sparse_proxy_ptv3.py": "db38f5dc0a596e1d289686cb744c15ffba9f154b16f2c78aac37abd9b0140b46",
    "models/sparse_proxy_point_transformer/taichi_sparse_attention.py": "c6f3d20699f85914367588dfb4eca7c68d544ee3ecceb8ca3cd3eae5b8845f83",
    "models/sparse_proxy_point_transformer/torch_obb.py": "d656f4edd68db53e61744a3e2e2950b89aea2d751d0a1384eaa58926cbafca48",
    "models/sparse_proxy_point_transformer/warp_sparse_attention.py": "de2999ed220f8a41ab203e0ad2f06982080f8ee9a9981dcb7ccdcc7c247c3805",
    "models/utils/__init__.py": "0ad96d4ee47bae83b87962e2195e3354c0d6cd4ef1c83de48d750bad668b51a2",
    "models/utils/checkpoint.py": "d75d3b2d225f4b81fa894b8aaff5b831fa17cd32443352b16039f35d886b2f98",
    "models/utils/misc.py": "26ef0b7d6ffbb7290435a1463593a9ac46b8ae9e70a08a064b110379b221a1bf",
    "models/utils/serialization/__init__.py": "e6461b37ccb5dcb24725943271259c4f74dac89d27b22becb97e48ce04c8a9b1",
    "models/utils/serialization/default.py": "886b8f3f0bbfaccb96b629fe2021d0bc9a700887c2ff8ec13b347f62ae60d964",
    "models/utils/serialization/hilbert.py": "60cb8365656312bb49a7a4c45859d35c60d764b613fab53bb0d97178a9c2f25a",
    "models/utils/serialization/z_order.py": "81101b623cf80aedf376202d4b4cb9c5f8e74a320390bd33ee67af662e3e05a4",
    "models/utils/structure.py": "cebe08382e12a3f4f82d4e6fc0f3ccb0f500b782b6316a417766703f484ab429",
    "utils/comm.py": "7ffce47290ad321934de20c8bd78fade5207d144a765b4562621ef8f30e3e3ee",
    "utils/misc.py": "8471773ed6878eca9601845539aa812b893ff6d709732b7f414d234048c2c378",
    "utils/registry.py": "ed38b85057632145f4b7269f40321688ccf80edd8528314712c01b78832c6e4e"
}

PREPARED_HASHES = {
    "LICENSE": "4cbd9de5f0da0d2f06aee41c6e46cfa2180bdb145668ef0211b062ce1cf2d468",
    "sp2t_native/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "sp2t_native/models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "sp2t_native/models/builder.py": "8403e075e0126ab8f19329f38aed2e8903eff52afcbb69acd23d0917017caa2e",
    "sp2t_native/models/modules.py": "b4d462dd11ea17b4fed3a02aea40108769613c2d55895b37d5afcb9998f70158",
    "sp2t_native/models/point_prompt_training/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "sp2t_native/models/point_prompt_training/prompt_driven_normalization.py": "3f40e872b2c0cfe21ef85784906dba900e679768d4970eaeaf07bcad78f3667b",
    "sp2t_native/models/sparse_proxy_point_transformer/__init__.py": "57a54daac7a378efffe1297f57c114c787b7b370b86e62a50212a9f828255e06",
    "sp2t_native/models/sparse_proxy_point_transformer/basic_blocks.py": "5958d9a32f37a418f8ac443cdfae5c51c4ca3635aeabd823e34a0e4c16e6c3ce",
    "sp2t_native/models/sparse_proxy_point_transformer/proxy_fuser.py": "590fd3842342e09a5480644738092094a03b003b9f46e3a31bed31349b1edcb5",
    "sp2t_native/models/sparse_proxy_point_transformer/proxy_initializer.py": "dda797c12ebe3dcd984252a404f55ec2c7a15a21d4e6bb5b7aa84f2aa49661ce",
    "sp2t_native/models/sparse_proxy_point_transformer/sparse_attention.py": "74c589fabac1e55bc2a16998d1226a8ca65972db10419437aabd594e8f3b9d0f",
    "sp2t_native/models/sparse_proxy_point_transformer/sparse_proxy_ptv3.py": "11645de3471bee95f65604a596d3785e40aa19244eab8ad82c9d87ccf8c10e68",
    "sp2t_native/models/sparse_proxy_point_transformer/taichi_sparse_attention.py": "c6f3d20699f85914367588dfb4eca7c68d544ee3ecceb8ca3cd3eae5b8845f83",
    "sp2t_native/models/sparse_proxy_point_transformer/torch_obb.py": "d656f4edd68db53e61744a3e2e2950b89aea2d751d0a1384eaa58926cbafca48",
    "sp2t_native/models/sparse_proxy_point_transformer/warp_sparse_attention.py": "de2999ed220f8a41ab203e0ad2f06982080f8ee9a9981dcb7ccdcc7c247c3805",
    "sp2t_native/models/utils/__init__.py": "0ad96d4ee47bae83b87962e2195e3354c0d6cd4ef1c83de48d750bad668b51a2",
    "sp2t_native/models/utils/checkpoint.py": "d75d3b2d225f4b81fa894b8aaff5b831fa17cd32443352b16039f35d886b2f98",
    "sp2t_native/models/utils/misc.py": "26ef0b7d6ffbb7290435a1463593a9ac46b8ae9e70a08a064b110379b221a1bf",
    "sp2t_native/models/utils/serialization/__init__.py": "e6461b37ccb5dcb24725943271259c4f74dac89d27b22becb97e48ce04c8a9b1",
    "sp2t_native/models/utils/serialization/default.py": "886b8f3f0bbfaccb96b629fe2021d0bc9a700887c2ff8ec13b347f62ae60d964",
    "sp2t_native/models/utils/serialization/hilbert.py": "60cb8365656312bb49a7a4c45859d35c60d764b613fab53bb0d97178a9c2f25a",
    "sp2t_native/models/utils/serialization/z_order.py": "81101b623cf80aedf376202d4b4cb9c5f8e74a320390bd33ee67af662e3e05a4",
    "sp2t_native/models/utils/structure.py": "6dcfd6d21abe7ec2d99185397052f62638972bb2f5adacb9c4ddd3413ff08493",
    "sp2t_native/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "sp2t_native/utils/comm.py": "7ffce47290ad321934de20c8bd78fade5207d144a765b4562621ef8f30e3e3ee",
    "sp2t_native/utils/misc.py": "8471773ed6878eca9601845539aa812b893ff6d709732b7f414d234048c2c378",
    "sp2t_native/utils/registry.py": "ed38b85057632145f4b7269f40321688ccf80edd8528314712c01b78832c6e4e"
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    for name, expected in PREPARED_HASHES.items():
        path = directory/name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('SP2T prepared source is missing or modified: '+name)


def prepare_sources(source, target):
    package = target/'sp2t_native'
    for name, expected in SOURCE_HASHES.items():
        path = source/'pointcept'/name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('SP2T source differs from its pin: '+name)
        text = path.read_text()
        tree = ast.parse(text)
        loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        lines = text.splitlines(keepends=True)
        for node in reversed(tree.body):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in ('mmcv', 'mmdet3d'):
                used = {a.asname or a.name for a in node.names} & loads
                if used and not (name.endswith('proxy_initializer.py') and used == {'knn', 'furthest_point_sample'}):
                    raise ValueError('SP2T optional imports differ from the pinned interfaces')
                del lines[node.lineno-1:node.end_lineno]
        text = ''.join(lines)
        if name.endswith('proxy_initializer.py'):
            text = text.replace('        inds = furthest_point_sample(pos, self.num_proxies)[0]',
                '        from mmcv.ops import furthest_point_sample\n        inds = furthest_point_sample(pos, self.num_proxies)[0]')
            text = text.replace('        px_ids = knn(A, px_pos.unsqueeze(0), pt_pos.unsqueeze(0))',
                '        from mmcv.ops import knn\n        px_ids = knn(A, px_pos.unsqueeze(0), pt_pos.unsqueeze(0))')
        text = text.replace('from pointcept.models.point_prompt_training import PDNorm',
            'from pointcept.models.point_prompt_training.prompt_driven_normalization import PDNorm')
        text = text.replace('from pointcept.', 'from sp2t_native.').replace('import pointcept.', 'import sp2t_native.')
        destination = package/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    for folder in (package, package/'models', package/'utils', package/'models/point_prompt_training'):
        (folder/'__init__.py').write_text('')
    shutil.copyfile(source/'LICENSE', target/'LICENSE')
    patch = ROOT/'grasppanda/resources/patches/sp2t/native-runtime.patch'
    env = {**os.environ, 'GIT_CEILING_DIRECTORIES': str(target.parent)}
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, env=env, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=target, env=env, check=True)
    verify(target)


def build(pins=None):
    if pins is None:
        pins = {v['id']:v for v in json.loads((ROOT/'grasppanda/resources/component_sources.lock.json').read_text())}
    pin = pins['sp2t']
    source = ROOT/pin['path']
    if pin['commit'] != SOURCE_COMMIT:
        raise ValueError('SP2T source revision differs from the registered implementation')
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise ValueError('Restore the pinned SP2T source before preparing the runtime')
    patch = ROOT/'grasppanda/resources/patches/sp2t/native-runtime.patch'
    identity = dict(source=SOURCE_COMMIT, builder=digest(__file__), patch=digest(patch))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    base = ROOT/'environments/native/sp2t'
    base.mkdir(parents=True, exist_ok=True)
    with (base/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        artifact = base/signature
        if artifact.exists():
            verify(artifact)
        else:
            pending = Path(tempfile.mkdtemp(prefix='pending-', dir=base))
            try:
                prepare_sources(source, pending)
                pending.rename(artifact)
            finally:
                if pending.exists():shutil.rmtree(pending)
        record = dict(identity, directory=str(artifact.resolve()))
        with tempfile.NamedTemporaryFile(mode='w', dir=base, delete=False) as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
            temporary = stream.name
        os.replace(temporary, base/'state.json')
    return artifact


if __name__ == '__main__':
    print(build())
