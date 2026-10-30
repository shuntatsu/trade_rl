from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.forward_study.registry import Fs1Registry
from trade_rl.integrations.market_store.embargo import (
    SEALED_WINDOW_START,
    SEALED_WINDOW_START_MS,
    EmbargoPolicy,
    SealedWindowEmbargoError,
)

ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    ROOT
    / "trade_rl"
    / "integrations"
    / "market_store"
    / "migrations"
    / "0001_market_store.sql"
)
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-123456789"


def test_boundary_is_the_fs1_registry_scoring_start() -> None:
    scoring_start = datetime.fromisoformat(Fs1Registry().scoring_start)
    assert SEALED_WINDOW_START == scoring_start.replace(tzinfo=UTC)
    assert SEALED_WINDOW_START_MS == int(SEALED_WINDOW_START.timestamp() * 1_000)


def test_sql_migration_boundary_equals_the_python_constant() -> None:
    text = MIGRATION.read_text(encoding="utf-8")
    match = re.search(r"TIMESTAMPTZ '([^']+)'", text)
    assert match is not None
    assert datetime.fromisoformat(match.group(1)) == SEALED_WINDOW_START


def test_check_ms_allows_the_last_millisecond_and_refuses_the_boundary() -> None:
    policy = EmbargoPolicy.enforced()
    policy.check_ms(SEALED_WINDOW_START_MS - 1, context="x")
    for offset in (0, 1, 3_600_000, 10**10):
        with pytest.raises(SealedWindowEmbargoError):
            policy.check_ms(SEALED_WINDOW_START_MS + offset, context="x")


def test_archive_end_may_touch_but_not_pass_the_boundary() -> None:
    policy = EmbargoPolicy.enforced()
    policy.check_archive_end(SEALED_WINDOW_START, url="october")
    with pytest.raises(SealedWindowEmbargoError):
        policy.check_archive_end(SEALED_WINDOW_START + timedelta(days=1), url="x")


def test_effective_end_clips_unless_released() -> None:
    later = SEALED_WINDOW_START + timedelta(days=30)
    assert EmbargoPolicy.enforced().effective_end(later) == SEALED_WINDOW_START
    assert EmbargoPolicy.release(URL).effective_end(later) == later


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/shuntatsu/trade_rl/issues/810",
        "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
        "http://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        "https://github.com/other/trade_rl/issues/810#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1 ",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-x",
        "",
    ],
)
def test_release_requires_an_issue_810_comment_url(url: str) -> None:
    with pytest.raises(ValueError):
        EmbargoPolicy.release(url)


def test_release_and_url_must_travel_together() -> None:
    with pytest.raises(ValueError):
        EmbargoPolicy(released=True)
    with pytest.raises(ValueError):
        EmbargoPolicy(released=False, authorization_url=URL)


def test_released_policy_admits_sealed_timestamps() -> None:
    policy = EmbargoPolicy.release(URL)
    policy.check_ms(SEALED_WINDOW_START_MS + 1, context="x")
    policy.check_archive_end(SEALED_WINDOW_START + timedelta(days=40), url="x")
