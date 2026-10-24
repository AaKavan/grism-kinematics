"""Serialize notebook jobs and reset only explicitly selected output folders."""
from contextlib import ExitStack, contextmanager
import fcntl
import os
from pathlib import Path
import re
import shutil


def _lock(stream, path):
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f'Another job is still running for this source, possibly in another notebook or kernel. '
            f'Wait for it to finish before starting {path}. '
            'To view saved results without starting a new job, set RUN_SEARCH=False and RUN_MCMC=False.'
        ) from exc


@contextmanager
def output_run(output, source, *, kind, override=False):
    """Hold an outside-folder lock through reset, preparation, and worker exit.

    Reset is restricted to one named run for this source. Raw inputs, other
    runs, and historical imaging are never selected by this helper. Existing
    worker locks are checked before removal, including both imaging variants.
    """
    if kind not in ('rotation', 'imaging') or not isinstance(override, bool):
        raise ValueError('Use kind="rotation" or "imaging" and a boolean override.')
    output = Path(os.path.abspath(Path(output).expanduser()))
    source_dir = source.directory
    expected_parent = source_dir if kind == 'rotation' else source_dir/'imaging_runs'
    if override:
        if (output.parent != expected_parent
                or not re.fullmatch(r'[A-Za-z0-9_-]+', output.name)
                or (kind == 'rotation' and output.name in ('imaging', 'imaging_runs', 'data', 'calibration'))):
            raise ValueError('Override only resets a named output run inside this source\'s results directory.')
        for path in (source.results/'real_sources', source_dir, expected_parent, output):
            if path.is_symlink():
                raise ValueError(f'Override cannot delete a symlinked output directory: {path}')
        if output.exists() and not output.is_dir():
            raise ValueError(f'The selected output must be a directory: {output}')
    locks = source_dir/'.run_locks'
    if locks.is_symlink():
        raise ValueError('The run lock directory must not be a symlink.')
    locks.mkdir(parents=True, exist_ok=True)
    # A grism worker depends on this source's imaging inputs. Serializing by
    # source also prevents another notebook from replacing them mid-fit.
    guard_path = locks/'source.lock'
    if guard_path.is_symlink():
        raise ValueError('The run lock must not be a symlink.')
    with guard_path.open('a') as guard:
        _lock(guard, output)
        if override and output.exists():
            worker_locks = ([output/'worker.lock'] if kind == 'rotation' else
                [output/name/'.fit.lock' for name in ('full_pixel', 'core_pixel_clean')]
                + sorted(source_dir.glob('*/worker.lock')))
            with ExitStack() as stack:
                for lock_path in worker_locks:
                    if lock_path.parent.is_symlink() or lock_path.is_symlink():
                        raise ValueError(f'Cannot reset a symlinked worker directory or lock: {lock_path}')
                    if lock_path.exists():
                        stream = stack.enter_context(lock_path.open('r+'))
                        _lock(stream, output)
                print(f'OVERRIDE=True: deleting saved outputs, checkpoints and logs in {output}', flush=True)
                shutil.rmtree(output)
        elif override:
            print(f'OVERRIDE=True: starting a fresh run in {output}', flush=True)
        yield output
