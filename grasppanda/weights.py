"""Content-verified author weights shared by the CLI, UI and recipes."""
import fcntl
import json
import shutil
import zipfile
from .config import ROOT
from .jobs import digest


def records(method, camera='realsense'):
    method = {'pointnet2_upgrade': 'graspnet_baseline'}.get(method, method)
    rows = [r for r in json.loads((ROOT/'grasppanda/resources/checkpoints.json').read_text())
            if r['method'] == method and r['camera'] in (camera, 'any')]
    if method == 'motiongrasp': rows += records('graspnet_baseline', camera)
    return rows


def primary(method, camera):
    return next((str(ROOT/r['path']) for r in records(method, camera)
                 if r.get('role', 'primary') == 'primary' and (ROOT/r['path']).is_file()), '')


def fetch(method, camera='realsense', progress=print):
    rows = records(method, camera)
    if not rows:
        raise ValueError(f'No registered weights for {method}/{camera}. Check the method card; weights for another camera are not substituted.')
    for record in rows:
        path = ROOT/record['path']; path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if digest(path) != record['sha256']: raise ValueError(f'Checksum mismatch: {path}. Move this file aside, then retry.')
                progress(f'Verified {path.name}'); continue
            temporary = path.with_suffix(path.suffix+'.download')
            source = record['source']; progress(f'Downloading {path.name} ({record["bytes"] / 1e6:.1f} MB extracted)')
            if 'drive.google.com/file/d/' in source:
                import gdown
                if not gdown.download(id=source.split('/d/')[1].split('/')[0], output=str(temporary), resume=True):
                    raise ValueError(f'Author download unavailable: {source}. Retry or place the file at {path}.')
            else:
                import requests
                with requests.get(source, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(1024*1024): stream.write(chunk)
            payload = temporary
            if record.get('archive_member'):
                payload = temporary.with_suffix('.extracted')
                with zipfile.ZipFile(temporary) as archive, archive.open(record['archive_member']) as src, payload.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            if payload.stat().st_size != record['bytes'] or digest(payload) != record['sha256']:
                raise ValueError(f'Download differs from the tested checkpoint: {path.name}. It was not installed.')
            payload.replace(path)
            if temporary.exists(): temporary.unlink()
            progress(f'Installed and verified {path.name}')
    return primary(method, camera)
