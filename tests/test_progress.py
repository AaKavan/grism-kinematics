"""Progress presentation must reflect worker counts without changing worker logs.

The parser tests use a small stand-in for tqdm. The launcher tests run only a
tiny Python child that prints representative messages, never a scientific fit.
"""

import io

import pytest

from dingo import progress


class RecordingBar:
    """Record the public tqdm operations needed by the progress adapter."""

    def __init__(self, **options):
        self.options = options
        self.n = options.get("initial", 0)
        self.total = options.get("total")
        self.desc = options.get("desc", "")
        self.descriptions = [self.desc]
        self.counts = [self.n]
        self.postfixes = []
        self.closed = False

    def update(self, amount=1):
        self.n += amount
        self.counts.append(self.n)

    def set_description(self, description, refresh=True):
        self.desc = description
        self.descriptions.append(description)

    def set_postfix_str(self, value, refresh=True):
        self.postfixes.append(value)

    def close(self):
        self.closed = True

    def refresh(self):
        pass

    @staticmethod
    def write(text, file=None, **kwargs):
        print(text, file=file)


class RecordingBars(list):
    """A factory that retains every phase bar for assertions."""

    def __call__(self, **options):
        bar = RecordingBar(**options)
        self.append(bar)
        return bar


@pytest.fixture
def stream():
    output = io.StringIO()
    bars = RecordingBars()
    adapter = progress.ProgressStream(output, bar_factory=bars)
    yield adapter, output, bars
    adapter.close()


def mcmc_header(adapter, warmup=25, draws=78):
    adapter.feed_line(
        f"09:20:00 | [MCMC] ndim=9 nwalkers=64 nburn={warmup} "
        f"nsteps={draws}; params=['V_rot', 'inc_v']"
    )


def test_mcmc_absolute_counts_warmup_switch_and_final_checkpoint(stream):
    adapter, output, bars = stream
    mcmc_header(adapter)
    adapter.feed_line("09:20:01 | MCMC step 1/103, acc=0.500, max lnP=-25.5")
    adapter.feed_line("09:20:03 | MCMC step 25/103, acc=0.450, max lnP=-24.5")
    adapter.feed_line("09:20:05 | MCMC step 50/103, acc=0.440, max lnP=-24.0")
    adapter.feed_line("09:20:05 | [MCMC] checkpoint at step 50 -> /tmp/checkpoint.npz")
    adapter.feed_line("09:20:10 | MCMC step 100/103, acc=0.420, max lnP=-23.0")
    adapter.feed_line("09:20:11 | [MCMC] checkpoint at step 103 -> /tmp/checkpoint.npz")
    adapter.close()

    assert len(bars) == 1
    bar = bars[0]
    assert bar.total == 103
    assert bar.n == 103
    assert bar.counts == sorted(bar.counts)
    assert all(0 <= count <= 103 for count in bar.counts)
    assert any("warmup" in label.lower() for label in bar.descriptions)
    assert any(
        "retained" in label.lower() or "sampling" in label.lower()
        for label in bar.descriptions
    )
    assert bar.closed
    assert "MCMC step" not in output.getvalue()


@pytest.mark.parametrize("storage", ["backend", "checkpoint"])
def test_resumed_run_starts_at_saved_absolute_count(stream, storage):
    adapter, _, bars = stream
    mcmc_header(adapter, warmup=25, draws=75)
    adapter.feed_line(f"[MCMC] resuming {storage} at step 60/100: /tmp/saved")
    assert len(bars) == 1
    assert bars[0].n == 60

    adapter.feed_line("MCMC step 61/100, acc=0.550, max lnP=-20")
    adapter.feed_line("[MCMC] checkpoint at step 61 -> /tmp/checkpoint.npz")
    adapter.feed_line("[MCMC] checkpoint at step 61 -> /tmp/checkpoint.npz")
    assert len(bars) == 1
    assert bars[0].n == 61
    assert bars[0].total == 100


def test_cached_target_does_not_invent_sampling(stream):
    adapter, output, bars = stream
    adapter.feed_line(
        "Requested 100 retained steps already saved in /tmp/run; "
        "loading without further sampling."
    )
    adapter.feed_line("[MCMC] target of 125 steps already exists; no sampling needed")
    adapter.feed_line("[MCMC] done: 125 steps, 6400 post-burn-in samples, acc=nan")
    adapter.close()

    assert not bars
    assert "no sampling needed" in output.getvalue()
    assert "loading without further sampling" in output.getvalue()


def test_closing_interrupted_stream_preserves_partial_count_and_warning(stream):
    adapter, output, bars = stream
    mcmc_header(adapter, warmup=25, draws=75)
    adapter.feed_line("MCMC step 25/100, acc=0.400, max lnP=-20")
    adapter.feed_line("KeyboardInterrupt, stopping MCMC at step 31")
    adapter.feed_line("[MCMC] checkpoint at step 31 -> /tmp/checkpoint.npz")
    adapter.close()

    assert bars[0].n == 31
    assert bars[0].closed
    assert bars[0].n < bars[0].total
    assert "KeyboardInterrupt" in output.getvalue()


def test_worker_done_message_does_not_turn_partial_run_into_complete_bar(stream):
    adapter, output, bars = stream
    mcmc_header(adapter, warmup=25, draws=75)
    adapter.feed_line("MCMC step 25/100, acc=0.400, max lnP=-20")
    adapter.feed_line("[MCMC] done: 31 steps, 384 post-burn-in samples, acc=0.400")

    assert bars[0].n == 31
    assert bars[0].total == 100
    assert bars[0].closed
    assert "done: 31 steps" in output.getvalue()


def test_unknown_warnings_and_failure_details_stay_visible(stream):
    adapter, output, bars = stream
    adapter.feed_line("MCMC step 25/100, acc=0.400, max lnP=-20")
    warning = "09:22:00 | [MCMC] checkpoint failed at step 50: Permission denied"
    traceback = "RuntimeError: likelihood returned nonfinite values"
    adapter.feed_line(warning)
    adapter.feed_line(traceback)

    assert warning in output.getvalue()
    assert traceback in output.getvalue()
    assert bars[0].n == 25


def test_partial_print_writes_are_buffered_and_last_diagnostic_survives(stream):
    adapter, output, bars = stream
    adapter.write("MCMC step 2")
    adapter.flush()
    assert not bars
    adapter.write("5/100, acc=0.4, max lnP=-20\nWARNING: partial diagnostic")
    assert bars[0].n == 25
    adapter.close()

    assert "WARNING: partial diagnostic" in output.getvalue()
    assert "MCMC step" not in output.getvalue()


def test_pysersic_bars_count_per_chain_and_reset_between_phases():
    output = io.StringIO()
    bars = RecordingBars()
    adapter = progress.ProgressStream(
        output, bar_factory=bars, imaging_warmup=100, imaging_draws=200,
    )
    adapter.feed_line("PySersic warmup: starting 0 + 25 per chain")
    adapter.feed_line("PySersic warmup: 25/100 per chain; 5.0s; checkpoint saved")
    adapter.feed_line("PySersic warmup: 100/100 per chain; 12.0s; checkpoint saved")
    adapter.feed_line("PySersic draws: starting 0 + 25 per chain")
    adapter.feed_line("PySersic draws: 25/200 per chain; 5.0s; checkpoint saved")
    adapter.feed_line("PySersic draws: 200/200 per chain; 35.0s; checkpoint saved")
    adapter.close()

    assert len(bars) == 2
    assert [(bar.n, bar.total) for bar in bars] == [(100, 100), (200, 200)]
    assert all(bar.closed for bar in bars)
    assert "warmup" in bars[0].desc.lower()
    assert "draw" in bars[1].desc.lower()


def test_repeated_imaging_variant_starts_a_new_bar():
    bars = RecordingBars()
    adapter = progress.ProgressStream(
        io.StringIO(), bar_factory=bars, imaging_warmup=50,
    )
    adapter.feed_line("PySersic warmup: starting 0 + 25 per chain")
    adapter.feed_line("PySersic warmup: 50/50 per chain; 5.0s; checkpoint saved")
    adapter.feed_line("PySersic warmup: starting 0 + 25 per chain")
    adapter.feed_line("PySersic warmup: 25/50 per chain; 5.0s; checkpoint saved")
    adapter.close()

    assert [(bar.n, bar.total) for bar in bars] == [(50, 50), (25, 50)]
    assert all(bar.closed for bar in bars)


def test_imaging_compilation_heartbeat_does_not_claim_completed_draws():
    bars = RecordingBars()
    adapter = progress.ProgressStream(
        io.StringIO(), bar_factory=bars, imaging_warmup=50,
    )
    adapter.feed_line("PySersic warmup: starting 0 + 25 per chain")
    adapter.feed_line(
        "PySersic transition block, including compilation on first use ... "
        "60s elapsed; the next completed block will report its step count."
    )
    assert bars[0].n == 0
    assert bars[0].total == 50
    assert "60s" in bars[0].postfixes[-1]
    adapter.close()


def test_unavailable_bar_dependency_falls_back_to_original_messages():
    output = io.StringIO()

    def unavailable_factory(**options):
        raise ImportError("Notebook widgets are not installed")

    adapter = progress.ProgressStream(output, bar_factory=unavailable_factory)
    lines = [
        "MCMC step 25/100, acc=0.400, max lnP=-20",
        "MCMC step 50/100, acc=0.400, max lnP=-20",
        "WARNING: fit needs inspection",
    ]
    for line in lines:
        adapter.feed_line(line)
    adapter.close()

    assert output.getvalue() == "\n".join(lines) + "\n"


def test_search_grid_and_repeated_optimizer_stages_get_separate_bars(stream):
    adapter, _, bars = stream
    adapter.feed_line("Geometry grid 1/15: inclination 30.0 deg, turnover 2.0 pixels")
    adapter.feed_line("Geometry grid 15/15: inclination 60.0 deg, turnover 6.0 pixels")
    adapter.feed_line("Refine: iteration 1/100, 30 starts active, best lnP -20")
    adapter.feed_line("Refine: iteration 100/100, 30 starts active, best lnP -10")
    adapter.feed_line("Refine: iteration 1/100, 30 starts active, best lnP -15")
    adapter.close()

    assert [(bar.n, bar.total) for bar in bars] == [(15, 15), (100, 100), (1, 100)]
    assert all(bar.closed for bar in bars)


def install_recording_bars(monkeypatch):
    """Use real launcher behavior while keeping terminal rendering deterministic."""
    bars = RecordingBars()
    original = progress.ProgressStream

    def recording_stream(output, **options):
        return original(output, bar_factory=bars, **options)

    monkeypatch.setattr(progress, "ProgressStream", recording_stream)
    return bars


def test_launcher_keeps_full_raw_log_and_appends_on_reuse(tmp_path, monkeypatch, capsys):
    bars = install_recording_bars(monkeypatch)
    log = tmp_path / "run.log"
    log.write_text("Earlier run\n")
    lines = [
        "[MCMC] ndim=9 nwalkers=64 nburn=25 nsteps=75; params=['V_rot']",
        "MCMC step 25/100, acc=0.4, max lnP=-20",
        "[MCMC] checkpoint at step 100 -> /tmp/checkpoint.npz",
        "WARNING: keep this scientific diagnostic",
    ]
    raw = "\n".join(lines) + "\n"
    program = f"import sys; sys.stdout.write({raw!r}); sys.stdout.flush()"

    progress.run_command(["-c", program], log, device="cpu")

    assert log.read_text() == "Earlier run\n" + raw
    captured = capsys.readouterr()
    assert "WARNING: keep this scientific diagnostic" in captured.out
    assert "MCMC step" not in captured.out
    assert bars[0].n == 100
    assert bars[0].closed


def test_nonzero_child_exit_preserves_diagnostics_and_partial_progress(
    tmp_path, monkeypatch, capsys,
):
    bars = install_recording_bars(monkeypatch)
    log = tmp_path / "run.log"
    raw = "MCMC step 25/100, acc=0.4, max lnP=-20\nRuntimeError: example failure\n"
    program = f"import sys; sys.stdout.write({raw!r}); sys.stdout.flush(); sys.exit(7)"

    with pytest.raises(RuntimeError, match="code 7"):
        progress.run_command(["-c", program], log, device="cpu")

    assert log.read_text() == raw
    assert "RuntimeError: example failure" in capsys.readouterr().out
    assert bars[0].n == 25
    assert bars[0].closed


@pytest.mark.parametrize("exception", [RuntimeError("worker failed"), KeyboardInterrupt()])
def test_launcher_propagates_failure_or_interrupt_and_closes_partial_bar(
    tmp_path, monkeypatch, exception,
):
    from dingo import real_workflow

    bars = install_recording_bars(monkeypatch)

    def failing_worker(*args, **kwargs):
        print("MCMC step 25/100, acc=0.4, max lnP=-20", flush=True)
        raise exception

    monkeypatch.setattr(real_workflow, "run_command", failing_worker)
    with pytest.raises(type(exception)):
        progress.run_command(["-m", "dingo.three_file_search"], tmp_path / "run.log")

    assert bars[0].n == 25
    assert bars[0].closed


def test_plain_output_option_keeps_original_launcher(tmp_path, monkeypatch, capsys):
    from dingo import real_workflow

    calls = []

    def worker(arguments, log_path, device="cuda"):
        calls.append((arguments, log_path, device))
        print("MCMC step 25/100, acc=0.4, max lnP=-20")

    monkeypatch.setattr(real_workflow, "run_command", worker)
    monkeypatch.setattr(
        progress, "ProgressStream",
        lambda *args, **kwargs: pytest.fail("Plain output created a progress bar"),
    )
    arguments = ["-m", "dingo.three_file_search"]
    log_path = tmp_path / "run.log"
    progress.run_command(arguments, log_path, device="cpu", progress=False)

    assert calls == [(arguments, log_path, "cpu")]
    assert "MCMC step 25/100" in capsys.readouterr().out
