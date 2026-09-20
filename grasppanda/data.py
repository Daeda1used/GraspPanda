"""Small verified starter sets and resumable author-archive downloads."""
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from .config import ROOT
from .datasets import get_dataset


def records():
    return json.loads((ROOT/'grasppanda/resources/dataset_downloads.json').read_text())


def _relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Invalid registered dataset path')
    return path


def plan(dataset, root='', profile='starter', include=None):
    spec = get_dataset(dataset)
    record = records().get(dataset, {})
    if profile not in ('starter', 'archives'):
        raise ValueError('Choose starter or archives')
    if include and profile != 'archives':
        raise ValueError('--include only filters archive downloads')
    if profile == 'starter':
        selection = record.get('starter', {})
        files, bundles = selection.get('files', []), selection.get('bundles', [])
        description = selection.get('description',
            'Use the official download links in Guide → Datasets. No small starter set is registered for this dataset.')
    else:
        files, bundles = record.get('archives', []), []
        if include:
            files = [r for r in files if any(fnmatch.fnmatchcase(PurePosixPath(r['path']).name,p) for p in include)]
            if not files: raise ValueError('No registered archives match --include')
        description = 'Pinned author archives. Extraction is a separate step; see Guide → Datasets.'
        if not files: description = 'Use the author full-download instructions in Guide → Datasets.'
    members = [m for b in bundles for m in b['members']]
    for row in [*files, *members]: _relative(row['path'])
    return dict(dataset=dataset,title=spec.title,profile=profile,
        root=str(Path(root).expanduser().resolve()) if root else '',
        available=bool(files or bundles),description=description,
        source=record.get('repository','https://graspnet.net/datasets.html'),
        terms=record.get('terms','See the official dataset download terms.'),
        download_bytes=sum(r['bytes'] for r in files),
        stream_limit_bytes=sum(b['max_download_bytes'] for b in bundles),
        installed_bytes=sum(r['bytes'] for r in [*files,*members]),
        files=files,bundles=bundles)


def _verified(path, row):
    if path.is_symlink(): raise ValueError('Refusing to replace a symbolic link: '+str(path))
    if not path.exists(): return False
    from .jobs import digest
    if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
        raise ValueError('Existing dataset file has a different checksum; preserve or move it before retrying: '+str(path))
    return True


def _target(root, name):
    path = root.joinpath(*_relative(name).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Dataset path resolves outside the selected root: '+str(path))
    return path


class _BoundedStream:
    def __init__(self, raw, limit):
        self.raw, self.limit, self.read_bytes = raw, limit, 0
    def read(self, size=-1):
        size = min(size if size >= 0 else self.limit+1, self.limit-self.read_bytes+1)
        result = self.raw.read(size)
        self.read_bytes += len(result)
        if self.read_bytes > self.limit:
            raise ValueError('The requested starter members exceed the registered streaming limit. Use the pinned full archive in Guide → Datasets.')
        return result


def _extract_starter(root, bundle, progress):
    import requests
    from urllib3.exceptions import HTTPError
    pending = {r['path']:r for r in bundle['members'] if not _verified(_target(root,r['path']),r)}
    if not pending:
        progress('Verified starter members: '+bundle['url'].rsplit('/',1)[-1]); return
    # A gzip member cannot be range-resumed. Verified completed members are
    # retained, and only the bounded archive prefix is re-read on a retry.
    for attempt in range(4):
        progress('Reading starter members: '+bundle['url'].rsplit('/',1)[-1])
        try:
            with requests.get(bundle['url'],stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                if response.status_code != 200: raise ValueError('Expected a complete archive stream')
                stream = _BoundedStream(response.raw,bundle['max_download_bytes'])
                with tarfile.open(fileobj=stream,mode='r|gz') as archive:
                    for member in archive:
                        name = member.name.removeprefix('./')
                        if name not in pending: continue
                        row = pending[name]
                        if not member.isfile() or member.size != row['bytes']:
                            raise ValueError('Unexpected registered archive member: '+name)
                        value = archive.extractfile(member).read()
                        if hashlib.sha256(value).hexdigest() != row['sha256']:
                            raise ValueError('Starter member checksum mismatch: '+name)
                        target = _target(root,name); target.parent.mkdir(parents=True,exist_ok=True)
                        temporary = target.with_name(target.name+'.download')
                        if temporary.is_symlink(): raise ValueError('Refusing a symbolic partial file: '+str(temporary))
                        temporary.write_bytes(value); temporary.replace(target)
                        del pending[name]; progress('Installed and verified '+name)
                        if not pending: return
                raise ValueError('Pinned archive lacks starter members: '+', '.join(sorted(pending)))
        except (requests.RequestException, HTTPError, OSError, EOFError, tarfile.ReadError):
            if attempt == 3: raise
            time.sleep(2**attempt)


def fetch(dataset, root, profile='starter', include=None, progress=print):
    import fcntl
    from .downloads import download_file
    if not root or not str(root).strip(): raise ValueError('Choose a dataset root on the server before downloading')
    selection = plan(dataset,root,profile,include)
    if not selection['available']:
        raise ValueError(selection['description']+' '+selection['source'])
    root = Path(selection['root']); root.mkdir(parents=True,exist_ok=True)
    with (root/'.grasppanda-download.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A dataset download is already active for this root. Wait for it or resume after it stops.') from None
        for row in selection['files']:
            download_file(row['url'],_target(root,row['path']),row['bytes'],row['sha256'],progress)
        for bundle in selection['bundles']:
            _extract_starter(root,bundle,progress)
        progress('Data verified. '+('Extract the archives using Guide → Datasets.' if profile=='archives' else 'Load the dataset preset, download its weights, then Check current form.'))
    return selection


def description(dataset, root=''):
    value = plan(dataset,root)
    size = value['download_bytes']/1e6
    text = f"**{value['title']}** — {value['description']}\n\n"
    if value['available']:
        text += (f"Download: {size:.1f} MB. " if size else
                 f"Stream at most {value['stream_limit_bytes']/1e6:.1f} MB from pinned archives; keep {value['installed_bytes']/1e6:.1f} MB. ")
        text += 'Completed files are content-verified and reused. Weights are downloaded separately.\n\n'
    elif dataset == 'graspclutter6d':
        text += 'The scene archive spans five volumes (about 203 GB); all five are required. See Guide → Datasets for download and extraction.\n\n'
    return text+f"[Original data]({value['source']}) · {value['terms']}"
