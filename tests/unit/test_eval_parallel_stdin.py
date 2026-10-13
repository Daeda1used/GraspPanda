"""Evaluator workers must start even when the calling program has no file."""

from __future__ import annotations

import subprocess
import sys

PROGRAM = """
from grasppanda.evaluation.graspnet1b.parallel import run_in_processes

if __name__ == "__main__":
    results = dict(run_in_processes(abs, [-1, -2, -3], num_workers=2, timeout=60))
    print(sorted(results.items()))
"""


def test_workers_start_for_a_program_read_from_stdin():
    done = subprocess.run(
        [sys.executable, "-"],
        input=PROGRAM,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert "[(0, 1), (1, 2), (2, 3)]" in done.stdout
