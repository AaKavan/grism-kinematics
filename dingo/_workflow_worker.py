"""Hold a workflow worker lock for the lifetime of a scientific subprocess.

The notebook launcher runs this module instead of the scientific entry point
directly. The requested worker then executes in this same process under an
exclusive file lock. If the notebook kernel disappears, the running worker
continues holding its lock, so a replacement notebook cannot safely overwrite
or resume the same output until that worker exits.

Example::

    python -m dingo._workflow_worker --lock /run/_worker_locks/posterior/worker.lock \
        -- -m dingo.three_file_fixed_i_fit [worker arguments]

Keep the lock outside the scientific worker's output directory. Existing
workers require a fresh output directory to contain only their own files.
No scientific backend is imported until the command is validated and locked.
"""

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import runpy
import sys


SUPPORTED_WORKERS = frozenset({
    "dingo.three_file_search",
    "dingo.three_file_manual_fit",
    "dingo.three_file_fixed_i_fit",
})


@contextmanager
def _worker_lock(path, module):
    """Acquire without waiting; preserve the lock file after normal/error exit."""
    path = Path(path).expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Worker lock paths must not be symlinked: {component}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A workflow worker is already active: {path}") from exc
        try:
            # Presence of this file is not evidence of a live worker. The OS
            # lock is authoritative; the text only aids inspection of logs.
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\nmodule={module}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    """Validate ``-- -m MODULE ...`` and execute one allowed worker under lock.

    Worker arguments are forwarded unchanged. ``runpy`` provides normal
    ``python -m`` main-module behavior without creating an unlocked grandchild.
    The original ``sys.argv`` is restored when called from a Python process,
    including when the worker raises or exits. Exceptions and exit statuses
    propagate to the existing progress/logging launcher.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if len(command) < 2 or command[0] != "-m" or command[1] not in SUPPORTED_WORKERS:
        choices = ", ".join(sorted(SUPPORTED_WORKERS))
        parser.error(f"Use -- -m followed by one supported workflow worker: {choices}")

    module, worker_arguments = command[1], command[2:]
    with _worker_lock(arguments.lock, module):
        previous_argv = sys.argv
        try:
            sys.argv = [module, *worker_arguments]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        finally:
            sys.argv = previous_argv


if __name__ == "__main__":
    main()
