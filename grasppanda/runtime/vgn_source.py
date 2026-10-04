"""Verified VGN compatibility overlay; author checkout remains unchanged."""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from ..config import ROOT
from ..jobs import digest

COMMIT = 'd7af0622433f52ae88ebe81533f12b46b33e951a'


def source():
    original = ROOT/'upstream/volumetric/vgn'
    if not original.is_dir(): raise ValueError('VGN source is missing; run ./panda install')
    commit = subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(original),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if commit != COMMIT or dirty: raise ValueError('VGN requires its unmodified pinned author checkout')
    patch = ROOT/'grasppanda/resources/patches/vgn/shared-runtime.patch'
    key = hashlib.sha256(commit.encode()+patch.read_bytes()).hexdigest()[:20]
    parent = ROOT/'environments/artifact-cache/vgn';parent.mkdir(parents=True,exist_ok=True)
    target = parent/key
    with (parent/'overlay.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                work=Path(temporary)/'source';work.mkdir()
                archive=subprocess.check_output(['git','-C',str(original),'archive',commit])
                with tarfile.open(fileobj=io.BytesIO(archive)) as package:package.extractall(work,filter='data')
                if any(p.is_symlink() for p in work.rglob('*')):raise ValueError('Unexpected symbolic VGN source')
                subprocess.run(['git','apply',str(patch)],cwd=work,check=True)
                files={str(p.relative_to(work)):digest(p) for p in work.rglob('*') if p.is_file()}
                (work/'.complete').write_text(json.dumps(files,sort_keys=True));work.rename(target)
        if not (target/'.complete').is_file():raise ValueError('Incomplete VGN overlay; move its cache aside and retry')
        for name,sha in json.loads((target/'.complete').read_text()).items():
            path=target/name
            if path.is_symlink() or not path.is_file() or digest(path)!=sha:raise ValueError('VGN overlay changed; move its cache aside and retry')
    return target
