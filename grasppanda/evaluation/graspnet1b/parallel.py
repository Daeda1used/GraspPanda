"""Run independent tasks in short-lived worker processes.

The reference tool hands scenes to ``multiprocessing.Pool`` and joins it. Three things
go wrong there: a worker's exception is only seen after every other scene has
finished; a worker killed by the kernel (out of memory) is replaced silently and its
task never completes, so ``join()`` blocks forever; and workers live for the whole run,
so whatever a scene leaks stays allocated.

This runner starts one process per task, so memory is returned to the system after
every scene by construction, and it watches each worker's pipe *and* its process
handle:

* a result or an exception is delivered as soon as its task ends;
* the first failure terminates the remaining workers and is raised immediately;
* a worker that dies without reporting, or runs longer than ``timeout``, is an error
  that names its task.

Workers are created with the ``forkserver`` start method by default: they are forked
from a clean single-threaded helper that has already imported the worker module, which
is as cheap as ``fork`` without inheriting the parent's threads and locks. As with any
non-fork start method, a script that starts an evaluation must guard its entry point
with ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.connection import Connection, wait
from typing import Any

__all__ = ["WorkerError", "run_in_processes"]

_POLL_SECONDS = 1.0


class WorkerError(RuntimeError):
    """A task failed, died or timed out in a worker process.

    Attributes:
        task_index: Position of the failed task in the submitted sequence.
        remote_traceback: Traceback text from the worker, when it could report one.
    """

    def __init__(self, message: str, task_index: int, remote_traceback: str = "") -> None:
        super().__init__(message)
        self.task_index = task_index
        self.remote_traceback = remote_traceback


def _entry(function: Callable[[Any], Any], task: Any, connection: Connection) -> None:
    try:
        payload = ("ok", function(task))
    except BaseException as error:
        payload = ("error", f"{type(error).__name__}: {error}", traceback.format_exc())
    try:
        connection.send(payload)
    finally:
        connection.close()


@contextmanager
def _importable_main() -> Iterator[None]:
    """Hide a main module that has no file while a worker starts.

    ``forkserver`` and ``spawn`` workers re-import the parent's main module by its path.
    A program read from standard input reports the path ``<stdin>``, which does not
    exist, and every worker would die on start-up.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if path is None or os.path.exists(path):
        yield
        return
    del main.__file__
    try:
        yield
    finally:
        main.__file__ = path


def run_in_processes(
    function: Callable[[Any], Any],
    tasks: Sequence[Any],
    *,
    num_workers: int,
    timeout: float,
    start_method: str = "forkserver",
    preload: Sequence[str] = (),
) -> Iterator[tuple[int, Any]]:
    """Yield ``(task index, function(task))`` as tasks finish, in completion order.

    Args:
        function: Picklable callable defined at module level.
        tasks: Picklable arguments, one process each.
        num_workers: Maximum number of processes alive at once.
        timeout: Seconds one task may run.
        start_method: ``forkserver``, ``spawn`` or ``fork``.
        preload: Modules the fork server imports once, so workers start with them loaded.

    Raises:
        WorkerError: on the first task that raises, dies or exceeds ``timeout``.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")
    context = multiprocessing.get_context(start_method)
    if start_method == "forkserver" and preload:
        context.set_forkserver_preload(list(preload))

    pending = list(range(len(tasks)))[::-1]
    running: dict[Connection, tuple[int, Any, float]] = {}

    def stop_all() -> None:
        for connection, (_, process, _) in running.items():
            process.terminate()
            connection.close()
        for _, process, _ in running.values():
            process.join()
        running.clear()

    try:
        while pending or running:
            while pending and len(running) < num_workers:
                index = pending.pop()
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_entry, args=(function, tasks[index], sender), daemon=True
                )
                with _importable_main():
                    process.start()
                sender.close()  # the parent keeps only the reading end
                running[receiver] = (index, process, time.monotonic())

            sentinels = {process.sentinel: conn for conn, (_, process, _) in running.items()}
            ready = wait([*running, *sentinels], timeout=_POLL_SECONDS)
            finished = {item if item in running else sentinels[item] for item in ready}
            for connection in finished:
                index, process, _ = running.pop(connection)
                try:
                    status, *payload = connection.recv()
                except (EOFError, OSError):
                    process.join()
                    raise WorkerError(
                        f"the worker of task {index} died without reporting "
                        f"(exit code {process.exitcode}); a negative code is the signal that "
                        "killed it, e.g. -9 when the kernel reclaims memory",
                        index,
                    ) from None
                finally:
                    connection.close()
                process.join()
                if status == "error":
                    raise WorkerError(payload[0], index, payload[1])
                yield index, payload[0]

            now = time.monotonic()
            for index, _, started in running.values():
                if now - started > timeout:
                    raise WorkerError(
                        f"task {index} exceeded the time limit of {timeout:.0f} s and was stopped",
                        index,
                    )
    finally:
        stop_all()
