"""The process runner: results, immediate errors, dead workers, time-outs, recycling."""

from __future__ import annotations

import os
import time

import pytest

from grasppanda.evaluation.graspnet1b.parallel import WorkerError, run_in_processes


def square(value: int) -> tuple[int, int]:
    return value * value, os.getpid()


def fail_on_three(value: int) -> int:
    if value == 3:
        raise ValueError("three is not allowed")
    time.sleep(0.2 if value < 3 else 30.0)
    return value


def die(value: int) -> int:
    if value == 1:
        os._exit(7)  # no exception, no report: what an out-of-memory kill looks like
    return value


def stall(value: int) -> int:
    time.sleep(60.0 if value == 0 else 0.0)
    return value


def run(function, tasks, **kwargs):
    kwargs.setdefault("num_workers", 2)
    kwargs.setdefault("timeout", 60.0)
    kwargs.setdefault("preload", (__name__,))
    return dict(run_in_processes(function, tasks, **kwargs))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize(
    "start_method", ["fork", "forkserver", pytest.param("spawn", marks=pytest.mark.slow)]
)
def test_results_arrive_for_every_task(start_method):
    results = run(square, list(range(5)), start_method=start_method)
    assert {index: value for index, (value, _) in results.items()} == {i: i * i for i in range(5)}
    pids = {pid for _, pid in results.values()}
    assert len(pids) == 5 and os.getpid() not in pids  # one fresh process per task


def test_no_tasks_and_bad_arguments():
    assert run(square, []) == {}
    with pytest.raises(ValueError, match="at least 1"):
        run(square, [1], num_workers=0)


def test_first_error_is_raised_at_once_with_the_worker_traceback():
    started = time.monotonic()
    with pytest.raises(WorkerError, match="ValueError: three is not allowed") as caught:
        run(fail_on_three, [0, 1, 2, 3, 4, 5], num_workers=3)
    assert time.monotonic() - started < 15.0  # did not wait for the 30 s tasks
    assert caught.value.task_index == 3
    assert "fail_on_three" in caught.value.remote_traceback


def test_a_worker_that_dies_is_an_error_not_a_hang():
    with pytest.raises(WorkerError, match=r"died without reporting \(exit code 7\)") as caught:
        run(die, [0, 1, 2], num_workers=1)
    assert caught.value.task_index == 1


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_time_limit():
    # Forked workers start at once. Under forkserver a worker's start-up counts against the
    # limit, and it depends on whether an earlier test started the fork server with this
    # module preloaded: run first, task 0 took more than a second on a busy machine.
    started = time.monotonic()
    with pytest.raises(WorkerError, match="exceeded the time limit of 3 s") as caught:
        run(stall, [1, 0, 1], timeout=3.0, start_method="fork")
    assert caught.value.task_index == 1
    assert time.monotonic() - started < 15.0


def test_results_are_yielded_as_they_finish():
    seen = []
    with pytest.raises(WorkerError):
        for index, _ in run_in_processes(fail_on_three, [0, 1, 3], num_workers=1, timeout=60.0):
            seen.append(index)
    assert seen == [0, 1]  # delivered before the failure, not lost with it
