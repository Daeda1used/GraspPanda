"""Prepare reviewable compatibility overlays without touching upstream clones."""
import hashlib
from pathlib import Path
import shutil
import subprocess

from .config import ROOT


def prepare_overlay(method):
    if method == 'gfla':
        source = ROOT/'environments/sources/LauncherTemplate'
        directory = 'LauncherTemplate'
        patch = ROOT/'patches/LauncherTemplate/linux.patch'
    elif method == 'centergrasp':
        source = ROOT/'upstream/single_view/rgbd/centergrasp'
        directory = 'simnet'
        patch = ROOT/'patches/centergrasp/python311.patch'
    elif method == 'gfla_source':
        source = ROOT/'upstream/single_view/rgbd/gfla'
        directory = 'models'
        patch = ROOT/'patches/gfla_source/pipeline.patch'
    else:
        raise ValueError(method)
    source_commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    patches = [patch] + ([ROOT/'patches/centergrasp/pipeline.patch'] if method == 'centergrasp' else [])
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
