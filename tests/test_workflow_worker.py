"""Worker lifetime locks protect checkpoints even outside the notebook process."""

import fcntl
from pathlib import Path
import select
import subprocess
import sys

import pytest

from dingo import _workflow_worker as worker


@pytest.mark.parametrize("module", sorted(worker.SUPPORTED_WORKERS))
def test_worker_runs_in_same_process_with_unchanged_arguments_under_lock(
        tmp_path, monkeypatch, module):
    lock = tmp_path / "_worker_locks" / "posterior" / "worker.lock"
    original_argv = sys.argv
    captured = {}

    def fake_worker(name, *, run_name, alter_sys):
        captured.update(module=name, run_name=run_name, alter_sys=alter_sys,
                        argv=sys.argv.copy())
        # A second file description cannot acquire this process's held lock.
        with lock.open("a+") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)

    monkeypatch.setattr(worker.runpy, "run_module", fake_worker)
    arguments = ["--parent-run", "a parent with spaces", "--draws", "250", "--prepare-only"]
    worker.main(["--lock", str(lock), "--", "-m", module, *arguments])
    assert captured == dict(module=module, run_name="__main__", alter_sys=True,
                            argv=[module, *arguments])
    assert sys.argv is original_argv
    assert f"module={module}" in lock.read_text()
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)


@pytest.mark.parametrize("error", [RuntimeError("Worker failed"), SystemExit(7), KeyboardInterrupt()])
def test_worker_error_releases_lock_and_restores_argv(tmp_path, monkeypatch, error):
    lock = tmp_path / "worker.lock"
    original_argv = sys.argv

    def fail_worker(*args, **kwargs):
        raise error

    monkeypatch.setattr(worker.runpy, "run_module", fail_worker)
    with pytest.raises(type(error)) as raised:
        worker.main(["--lock", str(lock), "--", "-m", "dingo.three_file_search"])
    assert raised.value is error
    assert sys.argv is original_argv
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)


@pytest.mark.parametrize("command", [[], ["-c", "print('not allowed')"],
                                      ["-m", "os"], ["some_script.py"],
                                      ["-m", "dingo.imaging_fit"]])
def test_unsupported_command_fails_before_creating_lock_or_importing_worker(
        tmp_path, monkeypatch, command):
    lock = tmp_path / "worker.lock"
    monkeypatch.setattr(worker.runpy, "run_module", lambda *args, **kwargs: pytest.fail("Imported a worker"))
    with pytest.raises(SystemExit) as raised:
        worker.main(["--lock", str(lock), "--", *command])
    assert raised.value.code == 2
    assert not lock.exists()


def test_second_subprocess_cannot_enter_a_live_workers_lock(tmp_path):
    """The wrapper owns the lock in the child, independent of a notebook lock."""
    lock = tmp_path / "_worker_locks" / "posterior" / "worker.lock"
    repository = Path(__file__).resolve().parents[1]
    # Substitute only the scientific body inside a real separate process. No
    # grism model, GPU initialization or sampling is involved in this test.
    owner_code = """
import sys
from dingo import _workflow_worker as worker
def fake_scientific_body(*args, **kwargs):
    print('locked', flush=True)
    sys.stdin.readline()
worker.runpy.run_module = fake_scientific_body
worker.main(['--lock', sys.argv[1], '--', '-m', 'dingo.three_file_search'])
"""
    owner = subprocess.Popen(
        [sys.executable, "-u", "-c", owner_code, str(lock)],
        cwd=repository, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        ready, _, _ = select.select([owner.stdout], [], [], 10)
        assert ready, "The subprocess did not acquire its lock in time"
        assert owner.stdout.readline().strip() == "locked"
        contender = subprocess.run(
            [sys.executable, "-m", "dingo._workflow_worker", "--lock", str(lock),
             "--", "-m", "dingo.three_file_search"],
            cwd=repository, capture_output=True, text=True, timeout=10,
        )
        assert contender.returncode != 0
        assert "A workflow worker is already active" in contender.stderr
        assert owner.poll() is None
        _, stderr = owner.communicate("release\n", timeout=10)
        assert owner.returncode == 0, stderr
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.communicate(timeout=10)
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)


def test_symlink_lock_cannot_overwrite_an_unrelated_file(tmp_path, monkeypatch):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("Keep these notes")
    lock = tmp_path / "worker.lock"
    lock.symlink_to(unrelated)
    monkeypatch.setattr(worker.runpy, "run_module", lambda *args, **kwargs: pytest.fail("Imported a worker"))
    with pytest.raises(ValueError, match="symlink"):
        worker.main(["--lock", str(lock), "--", "-m", "dingo.three_file_search"])
    assert unrelated.read_text() == "Keep these notes"
