"""Fetch upstream repositories without changing existing checkouts."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT/'grasppanda/resources/upstreams.lock.json'
PINS = {m['id']:m.get('pinned_commit') or m.get('commit') for m in json.loads(LOCK.read_text())} if LOCK.exists() else {}

def clone(m):
    start = time.monotonic()
    dest = ROOT / m['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    log = ROOT / 'logs' / (m['id'] + '_clone.log')
    result = {'id': m['id'], 'repository': m['repository'], 'path': m['path'], 'pinned_commit':PINS.get(m['id'])}
    try:
        if not (dest / '.git').exists():
            with log.open('w') as f:
                subprocess.run(['git', 'clone', '--depth', '1', m['repository'], str(dest)],
                               stdout=f, stderr=subprocess.STDOUT, check=True, timeout=360,
                               env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1', 'GIT_TERMINAL_PROMPT': '0'})
            if PINS.get(m['id']):
                subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',PINS[m['id']]],check=True,timeout=120,capture_output=True)
                subprocess.run(['git','-C',str(dest),'checkout','--detach',PINS[m['id']]],check=True,timeout=30,capture_output=True)
        if m['id']=='asgrasp':
            subprocess.run(['git','-C',str(dest),'submodule','update','--init','--depth','1','gsnet'],check=True,timeout=180,capture_output=True)
        def git(*args):
            return subprocess.check_output(['git', '-C', str(dest), *args], text=True, timeout=30).strip()
        result.update(status='cloned', commit=git('rev-parse', 'HEAD'),
                      commit_date=git('show','-s','--format=%cI','HEAD'),
                      origin=git('remote', 'get-url', 'origin'),
                      branch=git('branch', '--show-current'),
                      submodules=subprocess.run(['git', '-C', str(dest), 'submodule', 'status'], text=True, capture_output=True).stdout,
                      submodule_error=subprocess.run(['git', '-C', str(dest), 'submodule', 'status'], text=True, capture_output=True).stderr,
                      shallow=git('rev-parse', '--is-shallow-repository'))
        result['tracked_changes']=git('status','--porcelain','--untracked-files=no')
        result['pinned_commit']=result['pinned_commit'] or result['commit']
        if PINS.get(m['id']) and result['commit'] != PINS[m['id']]:
            result.update(status='revision_mismatch',expected_commit=PINS[m['id']])
    except Exception as e:
        result.update(status='clone_failed', error=str(e))
    result['seconds'] = round(time.monotonic() - start, 2)
    result['checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result

if __name__ == '__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    methods = json.loads((ROOT/'grasppanda/resources/methods.json').read_text())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(clone, methods):
            results.append(r)
            print(r['id'], r['status'], r.get('commit', '')[:12], flush=True)
    # Installation state belongs to the local cache; source pins are immutable.
    (ROOT/'environments').mkdir(exist_ok=True)
    (ROOT/'environments/source-state.json').write_text(json.dumps(results, indent=2)+'\n')
    if any(r['status'] != 'cloned' for r in results):
        raise SystemExit('Some upstreams could not be fetched at their pinned revisions; consult clone logs.')
