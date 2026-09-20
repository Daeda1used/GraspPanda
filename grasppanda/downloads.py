"""Resumable, content-verified public artifacts with a shared local cache."""
import fcntl
import json
from pathlib import Path
import time


def download_file(url, destination, size, sha256, progress=print, *, byte_offset=None):
    import requests
    from .jobs import digest
    path = Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        stamp = path.with_suffix(path.suffix+'.verified.json')
        def identity():
            stat = path.stat()
            return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,
                        inode=stat.st_ino,sha256=sha256)
        if path.is_file():
            try: cached = json.loads(stamp.read_text())
            except (OSError,ValueError): cached = None
            if cached != identity():
                if path.stat().st_size != size or digest(path) != sha256:
                    raise ValueError(f'Existing artifact has a different checksum: {path}. Move it aside before retrying.')
                stamp.write_text(json.dumps(identity()))
            progress('Verified '+path.name)
            return path
        partial = path.with_suffix(path.suffix+'.download')
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size: raise ValueError('Oversized partial download: '+str(partial))
            if offset == size: break
            start = offset+(byte_offset or 0)
            headers = ({'Range':f'bytes={start}-{byte_offset+size-1}'} if byte_offset is not None
                       else {'Range':f'bytes={offset}-'} if offset else {})
            try:
                with requests.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        expected = f'bytes {start}-'
                        if not response.headers.get('Content-Range','').startswith(expected):
                            raise ValueError('Server returned an unexpected download range.')
                    elif response.status_code == 200:
                        if byte_offset is not None:
                            raise ValueError('This source did not honor the registered byte range. Use the full author archive in Guide → Datasets.')
                        offset = 0
                    else: raise ValueError('Unexpected artifact response: '+str(response.status_code))
                    received, reported = offset, offset
                    with partial.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(1024*1024):
                            received += len(chunk)
                            if received > size: raise ValueError('Artifact exceeds the registered size.')
                            stream.write(chunk)
                            if received-reported >= 64*1024*1024:
                                progress(f'{path.name}: {received/size:.0%}'); reported = received
                if received == size: break
            except requests.RequestException:
                if attempt == 3: raise
            if attempt < 3: time.sleep(2**attempt)
        if not partial.is_file() or partial.stat().st_size != size:
            raise ValueError('Download is incomplete; rerun to resume: '+str(partial))
        progress('Checking SHA-256: '+path.name)
        if digest(partial) != sha256:
            raise ValueError('Artifact checksum mismatch; it was not installed. Move the partial file aside and retry: '+str(partial))
        partial.replace(path)
        stamp.write_text(json.dumps(identity()))
        progress('Installed and verified '+path.name)
        return path
