from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.integrations.market_store.cli import (
    EXIT_EMBARGO,
    EXIT_FAILURE,
    EXIT_OK,
    run,
)

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-5"


def invoke(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(list(argv), out=out, err=err, clock=lambda: NOW)
    return code, out.getvalue(), err.getvalue()


def test_default_plan_stops_at_the_fs1_warmup_start() -> None:
    code, out, _ = invoke("plan", "--bases", "BTC", "--start", "2026-06-01")
    assert code == EXIT_OK
    plan = json.loads(out)
    assert plan["end"] == "2026-08-01T00:00:00+00:00"
    assert plan["fs1_warmup_allowed"] is False
    assert plan["archives_planned"] == len(plan["archives"]) > 0
    assert all(
        item["period_end"] <= "2026-08-01T00:00:00+00:00" for item in plan["archives"]
    )


def test_ingest_without_allow_network_is_a_dry_run() -> None:
    code, out, err = invoke("ingest", "--bases", "BTC", "--start", "2026-06-01")
    assert code == EXIT_OK
    assert json.loads(out)["dry_run"] is True
    assert "nothing was fetched" in err


@pytest.mark.parametrize("command", ["plan", "ingest"])
@pytest.mark.parametrize("end", ["2026-08-02", "2026-10-30", "2026-11-01"])
def test_default_refuses_any_end_inside_the_warmup_span(command: str, end: str) -> None:
    extra = ("--allow-network",) if command == "ingest" else ()
    code, out, err = invoke(command, "--end", end, *extra)
    assert code == EXIT_EMBARGO
    assert out == ""
    assert "warm-up" in err and "--allow-fs1-warmup-span" in err


@pytest.mark.parametrize("command", ["plan", "ingest"])
@pytest.mark.parametrize("flags", [(), ("--allow-fs1-warmup-span",)])
def test_the_sealed_window_is_refused_with_or_without_the_warmup_flags(
    command: str, flags: tuple[str, ...]
) -> None:
    extra = ("--allow-network",) if command == "ingest" else ()
    auth = ("--fs1-warmup-authorization-url", URL) if flags else ()
    code, out, err = invoke(command, "--end", "2026-11-02", *flags, *auth, *extra)
    assert code == EXIT_EMBARGO and out == ""
    assert "sealed window" in err or "warm-up" in err


def test_warmup_flag_and_url_must_be_given_together() -> None:
    code, _, err = invoke("plan", "--allow-fs1-warmup-span")
    assert code == EXIT_FAILURE and "together" in err
    code, _, err = invoke("plan", "--fs1-warmup-authorization-url", URL)
    assert code == EXIT_FAILURE and "together" in err
    code, _, err = invoke(
        "plan",
        "--allow-fs1-warmup-span",
        "--fs1-warmup-authorization-url",
        "https://example.com/not-the-issue",
    )
    assert code == EXIT_FAILURE and "Issue #810" in err


def test_there_is_no_release_option() -> None:
    for flag in ("--release-sealed-window", "--release-authorization-url"):
        with pytest.raises(SystemExit):
            invoke("plan", flag, URL)


def test_warmup_authorization_warns_and_plans_up_to_the_sealed_window() -> None:
    code, out, err = invoke(
        "plan",
        "--bases",
        "BTC",
        "--markets",
        "spot",
        "--start",
        "2026-10-01",
        "--allow-fs1-warmup-span",
        "--fs1-warmup-authorization-url",
        URL,
    )
    assert code == EXIT_OK
    assert "ADMITTED" in err and URL in err and "re-acquires" in err
    plan = json.loads(out)
    assert plan["fs1_warmup_allowed"] is True
    assert plan["end"] == "2026-10-30T00:00:00+00:00"
    assert all(
        item["period_end"] <= "2026-11-01T00:00:00+00:00" for item in plan["archives"]
    )


def test_export_end_inside_a_closed_span_is_refused_before_touching_a_database() -> (
    None
):
    for end in ("2026-09-01", "2026-11-03"):
        code, _, err = invoke(
            "export",
            "--market",
            "spot",
            "--start",
            "2026-01-01",
            "--end",
            end,
            "--output-dir",
            "unused",
        )
        assert code in (EXIT_EMBARGO, EXIT_FAILURE)
        assert "psycopg" not in err


def test_offline_commands_import_neither_psycopg_nor_the_network_client() -> None:
    script = (
        "import sys, io\n"
        "from datetime import datetime, UTC\n"
        "from trade_rl.integrations.market_store.cli import run\n"
        "now = datetime(2026, 10, 30, tzinfo=UTC)\n"
        "out = io.StringIO()\n"
        "assert run(['plan', '--bases', 'BTC', '--start', '2026-06-01'],"
        " out=out, err=io.StringIO(), clock=lambda: now) == 0\n"
        "assert run(['ingest', '--bases', 'BTC', '--start', '2026-06-01'],"
        " out=out, err=io.StringIO(), clock=lambda: now) == 0\n"
        "assert 'psycopg' not in sys.modules\n"
        "assert 'trade_rl.integrations.market_store.store' not in sys.modules\n"
        "assert 'trade_rl.integrations.market_store.fetch' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
