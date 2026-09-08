"""Persistent, single-GPU queue with process-group cancellation."""
import atexit
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import socketserver
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

from .config import ROOT, Experiment, catalogue


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class JobManager:
    def __init__(self, directory=None, allow_attach=False):
        self.root = Path(directory or os.environ.get("GRASPPANDA_RUNS", ROOT / "outputs/runs")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = False
        self._socket_path = str(self.root / '.queue.sock')
        self._lease = (self.root / ".worker.lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lease.close()
            if allow_attach:
                self._client = True
                self._rpc('ping')
                return
            raise RuntimeError("This run directory already has a worker. Use the existing UI.")
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "jobs.sqlite3", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created TEXT, state TEXT, config TEXT, detail TEXT)")
        self._db.execute("UPDATE jobs SET state='interrupted', detail='Worker restarted; resubmit explicitly' WHERE state IN ('queued','running')")
        self._db.commit()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._process = None
        self._active = None
        self._cancel = set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._start_rpc()
        atexit.register(self.close)

    def _rpc(self, operation, payload=None):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(180)
            try:
                connection.connect(self._socket_path)
            except OSError as error:
                raise RuntimeError('The existing worker does not accept queue connections. Restart the UI with the current GraspPanda version.') from error
            connection.sendall((json.dumps({'operation': operation, 'payload': payload})+'\n').encode())
            with connection.makefile('rb') as stream:
                response = json.loads(stream.readline())
        if 'error' in response: raise RuntimeError(response['error'])
        return response['result']

    def _start_rpc(self):
        manager = self
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    self.connection.settimeout(180)
                    request = json.loads(self.rfile.readline(1024 * 1024))
                    operation, payload = request['operation'], request.get('payload')
                    if operation == 'ping': result = 'ready'
                    elif operation == 'submit': result = manager.submit(Experiment.from_dict(payload))
                    elif operation == 'sweep': result = manager.submit_sweep(payload)
                    elif operation == 'list': result = manager.list()
                    elif operation == 'get': result = manager.get(payload)
                    elif operation == 'cancel': result = manager.cancel(payload)
                    else: raise ValueError('Unknown queue operation')
                    response = {'result': result}
                except Exception as error:
                    response = {'error': str(error)}
                self.wfile.write((json.dumps(response)+'\n').encode())
        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
        Path(self._socket_path).unlink(missing_ok=True)
        self._server = Server(self._socket_path, Handler)
        os.chmod(self._socket_path, 0o600)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

    def submit(self, config):
        if self._client: return self._rpc('submit', config.to_dict())
        return self._submit_many([config])[0]

    def submit_sweep(self, specification):
        from .sweeps import Sweep
        sweep = Sweep.from_dict(specification)
        if self._client: return self._rpc('sweep', sweep.to_dict())
        return self._submit_many(sweep.expand(), sweep.to_dict())

    def _submit_many(self, configs, sweep=None):
        # Reject every invalid input before any experiment becomes runnable.
        configs = [config.preflight() for config in configs]
        directories = []
        manifest = None
        group = uuid.uuid4().hex if sweep else None
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError('Worker is shutting down')
            try:
                for index, config in enumerate(configs):
                    job_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
                    directory = self.root / job_id
                    directory.mkdir()
                    directories.append(directory)
                    data = self._prepare(config, directory)
                    if sweep:
                        (directory/'sweep.json').write_text(json.dumps(dict(id=group, index=index, **sweep), indent=2)+'\n')
                    self._db.execute('INSERT INTO jobs VALUES (?,?,?,?,?)', (job_id, now(), 'queued', json.dumps(data), ''))
                ids = [directory.name for directory in directories]
                if sweep:
                    manifest = self.root/'sweeps'/f'{group}.json'
                    manifest.parent.mkdir(exist_ok=True)
                    manifest.write_text(json.dumps(dict(id=group, jobs=ids, **sweep), indent=2)+'\n')
                self._db.commit()
            except Exception:
                self._db.rollback()
                for directory in directories:
                    shutil.rmtree(directory)
                if manifest:
                    manifest.unlink(missing_ok=True)
                raise
        self._wake.set()
        return ids

    def _prepare(self, config, directory):
        data = config.to_dict()
        (directory / "config.json").write_text(json.dumps(data, indent=2) + "\n")
        (directory / "experiment.log").touch()
        repo = ROOT / catalogue()[config.method]["path"]
        provenance = {"created": now(), "python": sys.executable, "config_sha256": digest(directory / "config.json"),
                      "upstream_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                      "upstream_tracked_changes": subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True),
                      "checkpoint_sha256": digest(config.checkpoint) if config.checkpoint and Path(config.checkpoint).is_file() else None}
        provenance['compatibility_patches'] = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'patches').rglob('*.patch')}
        provenance['runtime_lock_sha256'] = digest(ROOT/'uv.lock')
        if config.method=='finegrasp':
            provenance['model_config_sha256']=digest(Path(config.checkpoint).parent/'model.config.json')
        provenance['native_source_lock_sha256']=digest(ROOT/'grasppanda/resources/native_sources.lock.json')
        component_lock=ROOT/'grasppanda/resources/component_sources.lock.json'
        if component_lock.exists():
            provenance['component_source_lock_sha256']=digest(component_lock)
            provenance['component_sources']={row['id']:subprocess.check_output(['git','-C',str(ROOT/row['path']),'rev-parse','HEAD'],text=True).strip() for row in json.loads(component_lock.read_text()) if (ROOT/row['path']/'.git').exists()}
        provenance['native_sources']={row['path']:subprocess.check_output(['git','-C',str(ROOT/row['path']),'rev-parse','HEAD'],text=True).strip()
            for row in json.loads((ROOT/'grasppanda/resources/native_sources.lock.json').read_text()) if (ROOT/row['path']/'.git').exists()}
        provenance['workbench_sources'] = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'grasppanda').rglob('*.py')}
        if config.action in ('pipeline_smoke','train_check'):
            from .weights import records
            if config.action=='pipeline_smoke':provenance['workbench_sources']['tools/run_recipe.py']=digest(ROOT/'tools/run_recipe.py')
            provenance['recipe_weights']={r['path']:digest(ROOT/r['path']) for r in records(config.method,config.camera) if (ROOT/r['path']).is_file()}
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return data

    def list(self):
        if self._client: return self._rpc('list')
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created DESC").fetchall()
        return [dict(id=r[0], created=r[1], state=r[2], config=json.loads(r[3]), detail=r[4]) for r in rows]

    def get(self, job_id):
        if self._client: return self._rpc('get', job_id)
        return next((r for r in self.list() if r["id"] == job_id), None)

    def _state(self, job_id, state, detail=""):
        with self._lock:
            self._db.execute("UPDATE jobs SET state=?,detail=? WHERE id=?", (state, detail, job_id))
            self._db.commit()
        (self.root / job_id / "status.json").write_text(json.dumps({"state": state, "detail": detail, "updated": now()}, indent=2))

    def cancel(self, job_id):
        if self._client: return self._rpc('cancel', job_id)
        with self._lock:
            row = self.get(job_id)
            if row is None:
                raise ValueError("Unknown job")
            if row["state"] not in ("queued", "running"):
                return row["state"]
            self._cancel.add(job_id)
            if row["state"] == "queued":
                self._state(job_id, "cancelled", "Cancelled before execution")
            if self._active == job_id and self._process:
                self._terminate(self._process)
        self._wake.set()
        return "Cancellation requested"

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                row = next((r for r in reversed(self.list()) if r["state"] == "queued"), None)
                if row:
                    self._active = row["id"]
                    self._state(row["id"], "running")
            if not row:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            job_id = row["id"]
            directory = self.root / job_id
            started = time.monotonic()
            try:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(row["config"]["gpu"]), "PYTHONUNBUFFERED": "1",
                       "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
                command = [sys.executable, "-m", "grasppanda.worker", str(directory / "config.json"), str(directory)]
                (directory / "command.json").write_text(json.dumps(command, indent=2))
                with (directory / "experiment.log").open("a") as log:
                    with self._lock:
                        if job_id in self._cancel or self._stop.is_set():
                            self._state(job_id, "cancelled")
                            continue
                        self._process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = started + row["config"]["timeout_minutes"] * 60
                    cancelled_at = None
                    timed_out = False
                    while self._process.poll() is None:
                        stopping = job_id in self._cancel or self._stop.is_set()
                        timed_out = timed_out or time.monotonic() > deadline
                        if stopping or timed_out:
                            if cancelled_at is None:
                                cancelled_at = time.monotonic()
                                self._terminate(self._process)
                            elif time.monotonic() - cancelled_at > 3:
                                try:
                                    os.killpg(self._process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                        self._stop.wait(0.15) if not self._stop.is_set() else time.sleep(0.15)
                    code = self._process.returncode
                    # The leader may exit before a descendant that ignores TERM.
                    # No process in this experiment session may outlive its job.
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                state = "cancelled" if job_id in self._cancel or self._stop.is_set() else "timeout" if timed_out else "succeeded" if code == 0 else "failed"
                self._state(job_id, state, f"exit={code}; wall_seconds={time.monotonic()-started:.3f}")
            except Exception as error:
                self._state(job_id, "failed", str(error))
            finally:
                with self._lock:
                    self._process = None
                    self._active = None

    def close(self):
        if self._client: return
        if self._stop.is_set():
            return
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()
        Path(self._socket_path).unlink(missing_ok=True)
        self._wake.set()
        self._thread.join(timeout=6)
        with self._lock:
            self._db.execute("UPDATE jobs SET state='interrupted',detail='Worker stopped' WHERE state='queued'")
            self._db.commit()
        self._db.close()
        self._lease.close()
        atexit.unregister(self.close)
