"""Prepare reviewable compatibility overlays without touching upstream clones."""
import hashlib
from pathlib import Path
import shutil
import subprocess

from .config import ROOT


def prepare_overlay(method):
    if method == 'spgrasp':
        return spgrasp_overlay()
    if method == 'gfla':
        source = ROOT/'environments/sources/LauncherTemplate'
        directory = 'LauncherTemplate'
        patch = ROOT/'grasppanda/resources/patches/LauncherTemplate/linux.patch'
    elif method == 'centergrasp':
        source = ROOT/'upstream/single_view/rgbd/centergrasp'
        directory = 'simnet'
        patch = ROOT/'grasppanda/resources/patches/centergrasp/python311.patch'
    elif method == 'gfla_source':
        source = ROOT/'upstream/single_view/rgbd/gfla'
        directory = 'models'
        patch = ROOT/'grasppanda/resources/patches/gfla_source/pipeline.patch'
    else:
        raise ValueError(method)
    source_commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    patches = [patch] + ([ROOT/'grasppanda/resources/patches/centergrasp/pipeline.patch'] if method == 'centergrasp' else [])
    key = hashlib.sha256(b''.join(p.read_bytes() for p in patches)+source_commit.encode()).hexdigest()[:16]
    destination = ROOT/'environments/overlays'/f'{method}-{key}'
    if not (destination/'.complete').exists():
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source/directory,destination/directory,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if method == 'centergrasp':
            for part in ('centergrasp', 'configs'):
                shutil.copytree(source/part,destination/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for patch in patches:
            subprocess.run(['git','apply','--unidiff-zero',str(patch)],cwd=destination,check=True)
        (destination/'.complete').write_text(source_commit+'\n'+key)
    return destination


def spgrasp_overlay():
    """Apply the reviewed planar-label fixes to pinned, unmodified source."""
    import fcntl
    import io
    import json
    import tarfile
    import tempfile
    from .config import catalogue
    from .jobs import digest
    source = ROOT / catalogue()['spgrasp']['path']
    pin = next(p['pinned_commit'] for p in json.loads((ROOT/'grasppanda/resources/upstreams.lock.json').read_text()) if p['id'] == 'spgrasp')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if commit != pin or changed.strip():
        raise ValueError('SPGrasp requires the unmodified pinned source checkout')
    patch = ROOT/'grasppanda/resources/patches/spgrasp/continuous-targets.patch'
    key = hashlib.sha256(commit.encode() + patch.read_bytes()).hexdigest()[:16]
    parent = ROOT/'environments/overlays'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent/f'spgrasp-{key}'
    with (parent/'.spgrasp.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                work = Path(temporary)/'source'
                work.mkdir()
                archive = subprocess.check_output(['git', '-C', str(source), 'archive', commit, 'spgrasp', 'training', 'LICENSE', 'LICENSE-sam2.txt'])
                with tarfile.open(fileobj=io.BytesIO(archive)) as package:
                    package.extractall(work, filter='data')
                for path in work.rglob('*'):
                    if path.is_symlink():
                        if not path.resolve().is_relative_to(work.resolve()) or not path.is_file():
                            raise ValueError('Unexpected external link in SPGrasp source')
                        content = path.read_bytes()
                        path.unlink()
                        path.write_bytes(content)
                subprocess.run(['git', 'apply', '--unidiff-zero', str(patch)], cwd=work, check=True)
                files = {str(p.relative_to(work)): digest(p) for p in work.rglob('*') if p.is_file()}
                (work/'.complete').write_text(json.dumps(files, sort_keys=True))
                work.rename(destination)
        manifest = destination/'.complete'
        if not manifest.is_file():
            raise ValueError('Incomplete SPGrasp overlay; remove its local cache and retry')
        for name, expected in json.loads(manifest.read_text()).items():
            path = destination/name
            if not path.is_file() or path.is_symlink() or digest(path) != expected:
                raise ValueError('SPGrasp overlay changed; remove its local cache and retry')
    return destination
