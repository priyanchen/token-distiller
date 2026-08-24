import json
from argparse import Namespace

from token_distiller import cli, storage


def _isolate(monkeypatch, tmp_path, name="report.db"):
    monkeypatch.setattr("token_distiller.storage.DB_PATH", tmp_path / name)


def test_savings_summary_aggregates_across_runs(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    storage.insert_run(
        source_path="a.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=1000, distilled_tokens_est=250,
    )
    storage.insert_run(
        source_path="b.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=2000, distilled_tokens_est=500,
    )
    summary = storage.savings_summary()
    assert summary["run_count"] == 2
    assert summary["raw_tokens_est"] == 3000
    assert summary["distilled_tokens_est"] == 750
    assert summary["tokens_saved_est"] == 2250
    assert summary["compression_ratio"] == 4.0


def test_savings_summary_excludes_failed_runs(monkeypatch, tmp_path):
    """A run that errored produced no usable distillation. Counting its tokens toward
    savings would overstate them -- this matters once the number is used to bill a
    savings-share license, not just as an internal metric."""
    _isolate(monkeypatch, tmp_path)
    storage.insert_run(
        source_path="a.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=1000, distilled_tokens_est=250,
    )
    storage.insert_run(
        source_path="b.pdf", source_type="pdf", trigger="file", status="error",
        raw_tokens_est=5000, distilled_tokens_est=0, error_message="boom",
    )
    summary = storage.savings_summary()
    assert summary["run_count"] == 1
    assert summary["raw_tokens_est"] == 1000


def test_savings_summary_since_filters_by_timestamp(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    storage.insert_run(
        source_path="old.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=1000, distilled_tokens_est=250, ts="2020-01-01T00:00:00+00:00",
    )
    storage.insert_run(
        source_path="new.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=2000, distilled_tokens_est=500, ts="2026-08-01T00:00:00+00:00",
    )
    summary = storage.savings_summary(since="2026-01-01")
    assert summary["run_count"] == 1
    assert summary["raw_tokens_est"] == 2000


def test_empty_report_does_not_crash(monkeypatch, tmp_path):
    """No runs yet must not divide by zero computing compression_ratio."""
    _isolate(monkeypatch, tmp_path)
    summary = storage.savings_summary()
    assert summary == {
        "run_count": 0,
        "raw_tokens_est": 0,
        "distilled_tokens_est": 0,
        "tokens_saved_est": 0,
        "compression_ratio": 0.0,
    }


def test_cmd_report_json_includes_dollar_estimate_when_rate_given(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    storage.insert_run(
        source_path="a.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=1_000_000, distilled_tokens_est=250_000,
    )
    code = cli.cmd_report(Namespace(since=None, json=True, rate=3.0))
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["dollars_saved_est"] == 2.25  # 750,000 tokens saved / 1e6 * $3


def test_cmd_report_text_output_shows_dollar_line_with_rate(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    storage.insert_run(
        source_path="a.pdf", source_type="pdf", trigger="file", status="ok",
        raw_tokens_est=1_000_000, distilled_tokens_est=250_000,
    )
    code = cli.cmd_report(Namespace(since=None, json=False, rate=3.0))
    out = capsys.readouterr().out
    assert code == 0
    assert "$2.25" in out


def test_cmd_report_omits_dollar_line_without_rate(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    code = cli.cmd_report(Namespace(since=None, json=False, rate=None))
    out = capsys.readouterr().out
    assert code == 0
    assert "$" not in out
