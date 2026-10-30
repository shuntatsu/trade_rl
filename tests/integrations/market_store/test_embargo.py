from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.forward_study.registry import Fs1Registry
from trade_rl.integrations.market_store import embargo as embargo_module
from trade_rl.integrations.market_store.embargo import (
    FS1_WARMUP_START,
    FS1_WARMUP_START_MS,
    SEALED_WINDOW_START,
    SEALED_WINDOW_START_MS,
    EmbargoPolicy,
    Fs1WarmupSpanError,
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


def test_boundaries_are_the_fs1_registry_values() -> None:
    registry = Fs1Registry()
    assert FS1_WARMUP_START == datetime.fromisoformat(
        registry.warmup_only_start
    ).replace(tzinfo=UTC)
    assert SEALED_WINDOW_START == datetime.fromisoformat(
        registry.scoring_start
    ).replace(tzinfo=UTC)
    assert FS1_WARMUP_START_MS == int(FS1_WARMUP_START.timestamp() * 1_000)
    assert SEALED_WINDOW_START_MS == int(SEALED_WINDOW_START.timestamp() * 1_000)


def test_sql_boundaries_equal_the_python_constants() -> None:
    text = MIGRATION.read_text(encoding="utf-8")
    sealed = re.search(
        r"market_sealed_from\(\)[^$]*\$\$\s*SELECT TIMESTAMPTZ '([^']+)'", text
    )
    warmup = re.search(
        r"market_fs1_warmup_start\(\)[^$]*\$\$\s*SELECT TIMESTAMPTZ '([^']+)'", text
    )
    assert sealed is not None and warmup is not None
    assert datetime.fromisoformat(sealed.group(1)) == SEALED_WINDOW_START
    assert datetime.fromisoformat(warmup.group(1)) == FS1_WARMUP_START


def test_there_is_no_release_path_anywhere() -> None:
    assert not hasattr(EmbargoPolicy, "release")
    assert not hasattr(EmbargoPolicy, "released")
    assert not hasattr(EmbargoPolicy(), "released")
    assert not hasattr(embargo_module, "RELEASE_AUTHORIZATION_URL_PATTERN")
    package = ROOT / "trade_rl" / "integrations" / "market_store"
    for path in [*package.glob("*.py"), *package.glob("migrations/*.sql")]:
        text = path.read_text(encoding="utf-8").lower()
        assert "released_at" not in text, path.name
        assert "market_embargo" not in text, path.name
        assert "release-sealed-window" not in text, path.name
    # even with the warm-up authorization the sealed window stays closed
    policy = EmbargoPolicy.with_warmup(URL)
    with pytest.raises(SealedWindowEmbargoError):
        policy.check_ms(SEALED_WINDOW_START_MS, context="x")
    with pytest.raises(SealedWindowEmbargoError):
        policy.check_archive_end(SEALED_WINDOW_START + timedelta(days=1), url="x")
    with pytest.raises(SealedWindowEmbargoError):
        policy.check_range_end(SEALED_WINDOW_START + timedelta(days=1), what="x")


def test_default_policy_closes_the_warmup_span_and_the_sealed_window() -> None:
    policy = EmbargoPolicy.enforced()
    assert policy.start == FS1_WARMUP_START and not policy.warmup_allowed
    policy.check_ms(FS1_WARMUP_START_MS - 1, context="x")
    with pytest.raises(Fs1WarmupSpanError, match="warm-up"):
        policy.check_ms(FS1_WARMUP_START_MS, context="x")
    with pytest.raises(Fs1WarmupSpanError):
        policy.check_ms(SEALED_WINDOW_START_MS - 1, context="x")
    for offset in (0, 1, 3_600_000, 10**10):
        with pytest.raises(SealedWindowEmbargoError) as excinfo:
            policy.check_ms(SEALED_WINDOW_START_MS + offset, context="x")
        assert not isinstance(excinfo.value, Fs1WarmupSpanError)


def test_archive_end_may_touch_but_not_pass_the_default_boundary() -> None:
    policy = EmbargoPolicy.enforced()
    policy.check_archive_end(FS1_WARMUP_START, url="july")
    with pytest.raises(Fs1WarmupSpanError):
        policy.check_archive_end(FS1_WARMUP_START + timedelta(days=1), url="aug")
    with pytest.raises(SealedWindowEmbargoError):
        policy.check_range_end(SEALED_WINDOW_START + timedelta(days=1), what="x")


def test_warmup_authorization_admits_only_the_warmup_span() -> None:
    policy = EmbargoPolicy.with_warmup(URL)
    assert policy.warmup_allowed and policy.warmup_authorization_url == URL
    assert policy.start == SEALED_WINDOW_START
    policy.check_ms(FS1_WARMUP_START_MS, context="x")
    policy.check_ms(SEALED_WINDOW_START_MS - 1, context="x")
    policy.check_archive_end(SEALED_WINDOW_START, url="october")


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
def test_warmup_requires_an_issue_810_comment_url(url: str) -> None:
    with pytest.raises(ValueError):
        EmbargoPolicy.with_warmup(url)


def test_boundaries_must_be_ordered_and_aware() -> None:
    with pytest.raises(ValueError):
        EmbargoPolicy(warmup_start=SEALED_WINDOW_START + timedelta(days=1))
    with pytest.raises(ValueError):
        EmbargoPolicy(sealed_start=datetime(2026, 11, 1))
