"""Place generated runtime directories on a user-selected storage volume."""
import fcntl
from pathlib import Path

from .config import ROOT


DIRECTORIES = {'.venv': 'venv', 'upstream': 'upstream', 'environments': 'environments',
               'checkpoints': 'checkpoints', 'outputs': 'outputs', 'logs': 'logs'}


def status(project=ROOT):
    project = Path(project).resolve()
    return {'directories': {name: {'destination': str((project/name).resolve()),
                                  'linked': (project/name).is_symlink(),
                                  'exists': (project/name).exists()}
                            for name in DIRECTORIES},
            'cache': str((project/'environments/artifact-cache').resolve())}


def configure(root, project=ROOT):
    """Link a fresh checkout without moving or replacing any existing data."""
    project = Path(project).resolve()
    destination = Path(root).expanduser().resolve()
    if destination == project or destination.is_relative_to(project):
        raise ValueError('Choose a storage root outside the source checkout')
    links = [(project/name, destination/folder) for name,folder in DIRECTORIES.items()]
    # Check every path before creating any links. Existing populated directories
    # belong to the user and must never be moved by an installation command.
    for local, target in links:
        if target == project or target.is_relative_to(project):
            raise ValueError('Storage layout would overlap the source checkout')
        if target.is_symlink() or target.exists() and not target.is_dir():
            raise ValueError('Storage destinations must be regular directories: '+str(target))
        if local.is_symlink():
            if local.resolve() != target:
                raise ValueError('Existing runtime link uses another location: '+str(local)+'. Use a fresh checkout to configure a new storage root.')
        elif local.exists():
            raise ValueError('Existing runtime data will not be replaced: '+str(local)+'. Configure storage in a fresh checkout before installation.')
    destination.mkdir(parents=True,exist_ok=True)
    lock_path = destination/'.grasppanda-storage.lock'
    if lock_path.is_symlink(): raise ValueError('Refusing a symbolic storage lock')
    created = []
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            for local,target in links:
                target.mkdir(exist_ok=True)
                if local.is_symlink() and local.resolve() == target: continue
                local.symlink_to(target,target_is_directory=True)
                created.append(local)
        except Exception:
            for local in reversed(created): local.unlink()
            raise
    return status(project)
