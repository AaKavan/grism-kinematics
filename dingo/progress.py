"""Notebook and terminal progress bars for the existing subprocess workers.

The workers still write their original messages to ``run.log``. This module
changes only their display in the calling notebook, after those messages have
been logged. Reported counts are never estimates of scientific convergence.
"""

from __future__ import annotations

import contextlib
import re
import sys
from typing import Callable, TextIO


class ProgressStream:
    """Turn complete worker log lines into bars, keeping other output visible.

    Parameters
    ----------
    output
        Original stdout, captured before redirecting the worker launcher.
    bar_factory
        Optional ``tqdm``-compatible constructor, mainly for testing. Otherwise
        ``tqdm.auto.tqdm`` is imported on the first progress event, selecting
        notebook widgets when available and a text bar otherwise.
    warmup, draws
        Requested MCMC iterations per walker. The MCMC bar counts their sum,
        without multiplying by the number of walkers. The worker's settings
        message can also supply these values.
    imaging_warmup, imaging_draws
        PySersic iterations per chain. Imaging has a separate bar for each
        phase because its worker reports phase-relative counts.

    Notes
    -----
    Only reported completed iterations advance a bar. Starting a PySersic
    block displays its current count while compilation or sampling proceeds.
    Closing a bar never fills in unreported iterations, including on failure.
    """

    def __init__(
        self,
        output: TextIO,
        *,
        bar_factory: Callable | None = None,
        warmup: int | None = None,
        draws: int | None = None,
        imaging_warmup: int | None = None,
        imaging_draws: int | None = None,
    ) -> None:
        self.output = output
        self.bar_factory = bar_factory
        self.warmup = warmup
        self.draws = draws
        self.imaging_totals = {
            "warmup": imaging_warmup,
            "draws": imaging_draws,
        }
        self.bar = None
        self.stage = None
        self._buffer = ""
        self._closed = False
        self._bars_available = True

    @property
    def encoding(self):
        """Preserve the wrapped stream's encoding for stream consumers."""
        return getattr(self.output, "encoding", None)

    def isatty(self) -> bool:
        """Report the original stream's terminal status."""
        return bool(getattr(self.output, "isatty", lambda: False)())

    def write(self, text: str) -> int:
        """Buffer split ``print`` writes until a complete line is available."""
        if self._closed:
            return self.output.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.feed_line(line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        """Flush displayed output without treating a partial write as a line."""
        self.output.flush()

    def close(self) -> None:
        """Display any final partial line and close the bar at its actual count."""
        if self._closed:
            return
        try:
            if self._buffer:
                line, self._buffer = self._buffer, ""
                self.feed_line(line)
        finally:
            self._close_bar()
            self._closed = True
            self.output.flush()

    def _close_bar(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.stage = None

    def _message(self, line: str) -> None:
        """Keep warnings and ordinary messages readable beside an active bar."""
        writer = getattr(self.bar, "write", None)
        if writer is not None:
            writer(line, file=self.output)
        else:
            self.output.write(line + "\n")

    def _progress(
        self,
        stage: str,
        current: int,
        total: int | None,
        description: str,
        *,
        postfix: str = "",
        unit: str = "step",
    ) -> bool:
        """Create or update one stage; return False for plain-text fallback."""
        if not self._bars_available:
            return False
        if current < 0 or (total is not None and (total <= 0 or current > total)):
            return False

        # Profile refinements can repeat with the same label. A decreasing
        # counter starts a new stage rather than moving the old bar backwards.
        if self.bar is not None and (
            self.stage != stage or current < self.bar.n
        ):
            self._close_bar()

        if self.bar is None:
            if self.bar_factory is None:
                try:
                    from tqdm.auto import tqdm
                except ImportError:
                    self._bars_available = False
                    return False
                self.bar_factory = tqdm
            try:
                self.bar = self.bar_factory(
                    total=total,
                    initial=current,
                    desc=description,
                    unit=unit,
                    file=self.output,
                    dynamic_ncols=True,
                    leave=True,
                )
            except ImportError:
                # Missing notebook-widget dependencies must not prevent a fit.
                self._bars_available = False
                return False
            self.stage = stage
        else:
            if total is not None:
                self.bar.total = total
            self.bar.set_description(description, refresh=False)
            self.bar.update(current - self.bar.n)

        self.bar.set_postfix_str(postfix, refresh=False)
        refresh = getattr(self.bar, "refresh", None)
        if refresh is not None:
            refresh()
        return True

    def _mcmc_description(self, current: int) -> str:
        if self.warmup is None:
            phase = "steps"
        else:
            phase = "warmup" if current < self.warmup else "retained"
        return f"MCMC {phase} (total incl. warmup)"

    def _mcmc_total(self) -> int | None:
        if self.warmup is None or self.draws is None:
            return None
        return self.warmup + self.draws

    def _mcmc_progress(self, current, total=None, *, postfix="") -> bool:
        if total is None:
            total = self._mcmc_total()
        # A progress line supplies the total even when no settings were printed.
        # Retain it for the final checkpoint, which prints only the count.
        if total is None and self.stage == "mcmc" and self.bar is not None:
            total = self.bar.total
        return self._progress(
            "mcmc", current, total, self._mcmc_description(current),
            postfix=postfix,
        )

    def feed_line(self, line: str) -> None:
        """Handle one complete worker line, retaining unrecognized messages."""
        line = line.rstrip("\r\n")
        # Worker logging uses ``HH:MM:SS | message``; raw print messages have
        # no prefix. Keep the untouched line when displaying ordinary output.
        message = re.sub(r"^\d{2}:\d{2}:\d{2}\s*\|\s*", "", line).strip()

        settings = re.search(r"\[MCMC\].*\bnburn=(\d+)\s+nsteps=(\d+)", message)
        if settings:
            self.warmup, self.draws = map(int, settings.groups())
            self._message(line)
            return

        match = re.search(r"\bMCMC step (\d+)/(\d+)(?:,\s*(.*))?$", message)
        if match and self._mcmc_progress(
            int(match[1]), int(match[2]), postfix=match[3] or ""
        ):
            return

        match = re.search(
            r"\[MCMC\] resuming (?:checkpoint|backend) at step (\d+)/(\d+):",
            message,
        )
        if match:
            self._message(line)
            self._mcmc_progress(int(match[1]), int(match[2]), postfix="resumed")
            return

        match = re.search(
            r"\[MCMC\] (?:checkpoint|resumable backend) at step (\d+) ->",
            message,
        )
        if match and self._mcmc_progress(int(match[1]), postfix="checkpoint saved"):
            return

        match = re.search(r"\[MCMC\] done: (\d+) steps,", message)
        if match:
            # A cached target can emit this summary after its bar was closed.
            # Do not create another apparent sampling phase for that summary.
            if self.stage == "mcmc" and self.bar is not None:
                self._mcmc_progress(int(match[1]))
                self._close_bar()
            self._message(line)
            return

        match = re.search(r"\[MCMC\] target of (\d+) steps already exists", message)
        if match:
            if self.stage == "mcmc" and self.bar is not None:
                self._mcmc_progress(int(match[1]), int(match[1]), postfix="cached")
                self._close_bar()
            self._message(line)
            return

        match = re.search(r"KeyboardInterrupt, stopping MCMC at step (\d+)", message)
        if match:
            if self.stage == "mcmc" and self.bar is not None:
                self._mcmc_progress(int(match[1]), postfix="interrupted")
            self._message(line)
            return

        match = re.fullmatch(
            r"PySersic (warmup|draws): starting (\d+) \+ (\d+) per chain", message
        )
        if match:
            phase, current, chunk = match[1], int(match[2]), int(match[3])
            if self._progress(
                f"imaging-{phase}", current, self.imaging_totals[phase],
                f"PySersic {phase} / chain",
                postfix=f"computing next {chunk} (first block may compile)",
            ):
                return

        match = re.fullmatch(
            r"PySersic (warmup|draws): (\d+)/(\d+) per chain; (.*)", message
        )
        if match:
            phase, current, total = match[1], int(match[2]), int(match[3])
            self.imaging_totals[phase] = total
            if self._progress(
                f"imaging-{phase}", current, total,
                f"PySersic {phase} / chain", postfix=match[4],
            ):
                return

        match = re.fullmatch(
            r"PySersic transition block, including compilation on first use \.\.\. "
            r"(\d+)s elapsed; the next completed block will report its step count\.",
            message,
        )
        if match and self.bar is not None and str(self.stage).startswith("imaging-"):
            # A heartbeat confirms activity, but does not certify more draws.
            self.bar.set_postfix_str(f"current block running: {match[1]}s", refresh=True)
            return

        match = re.fullmatch(r"Geometry grid (\d+)/(\d+): (.*)", message)
        if match and self._progress(
            "geometry", int(match[1]), int(match[2]), "Candidate geometry grid",
            postfix=match[3], unit="group",
        ):
            return

        match = re.fullmatch(
            r"(.+): iteration (\d+)/(\d+), (\d+) starts active, best lnP (.+)",
            message,
        )
        if match and self._progress(
            f"search-{match[1]}", int(match[2]), int(match[3]),
            f"{match[1]} (max iterations)",
            postfix=f"{match[4]} starts active, best lnP {match[5]}", unit="iteration",
        ):
            return

        self._message(line)


def _integer_option(arguments, name: str) -> int | None:
    """Read a display hint from a CLI argument without validating the worker."""
    value = None
    for index, argument in enumerate(arguments):
        argument = str(argument)
        if argument == name and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif argument.startswith(name + "="):
            value = argument.split("=", 1)[1]
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def run_command(arguments, log_path, device="cuda", *, progress=True):
    """Run an existing DINGO worker with live bars and its full original log.

    Set ``progress=False`` for the previous line-by-line text display. This
    wrapper preserves the original launcher's subprocess handling, device
    setup, log writing and interruption behavior. It changes neither worker
    code nor any configuration used to identify a resumable science run.
    """
    from .real_workflow import run_command as legacy_run_command

    if not progress:
        return legacy_run_command(arguments, log_path, device)

    arguments = list(arguments)
    warmup = _integer_option(arguments, "--warmup")
    stream = ProgressStream(
        sys.stdout,
        warmup=warmup,
        draws=_integer_option(arguments, "--draws"),
        imaging_warmup=warmup,
        imaging_draws=_integer_option(arguments, "--samples"),
    )
    try:
        with contextlib.redirect_stdout(stream):
            return legacy_run_command(arguments, log_path, device)
    finally:
        stream.close()
