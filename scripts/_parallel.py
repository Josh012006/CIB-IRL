"""Run independent jobs (each one a command line) several at a time, each in its own process.

Shared by run_suite.py and run_toy_suite.py. Subprocesses (not multiprocessing) so that it behaves the
same on Windows, macOS and Linux. Python threads here only WAIT for their subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


@dataclass
class Job:
    name: str
    cmd: list[str]
    log_path: Path


def thread_env(cpu_count: int, threads_per_run: int | None) -> tuple[dict, int]:
    """Environment for the children: every process limited to (cores // cpu_count) torch threads, so that N
    processes do not fight for the same cores."""
    machine = os.cpu_count() or 1
    if cpu_count > machine:
        say(f"warning: --cpu-count {cpu_count} > {machine} cores on this machine")
    threads = threads_per_run or max(1, machine // cpu_count)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUTF8="1")
    return env, threads


def _run_one(job: Job, cwd: Path, env: dict) -> tuple[str, int]:
    t0 = time.time()
    say(f"start  {job.name}")
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(job.log_path, "w", encoding="utf-8") as log:
        code = subprocess.run(job.cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    dt = (time.time() - t0) / 60
    say(f"{'done ' if code == 0 else 'FAIL '}  {job.name}  ({dt:.1f} min)" + ("" if code == 0 else f"  -> see {job.log_path}"))
    return job.name, code


def run_jobs(jobs: list[Job], cwd: Path, cpu_count: int, threads_per_run: int | None = None) -> list[str]:
    """Runs all jobs, `cpu_count` at a time. Returns the names of the failed ones (the others go on)."""
    env, threads = thread_env(cpu_count, threads_per_run)
    say(f"{len(jobs)} runs, {cpu_count} in parallel, {threads} torch thread(s) each ({os.cpu_count()} cores)")
    failed: list[str] = []
    pool = ThreadPoolExecutor(max_workers=cpu_count)
    try:
        futures = [pool.submit(_run_one, j, cwd, env) for j in jobs]
        for fut in as_completed(futures):
            name, code = fut.result()
            if code != 0:
                failed.append(name)
    except KeyboardInterrupt:
        say("interrupted: pending runs cancelled, running ones stop with Ctrl-C (resume with --skip-existing)")
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)
    pool.shutdown()
    return failed
