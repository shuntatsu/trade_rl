"""Pre-registered attempt-3 collection-failure policy: classes, budgets and constants.

The values come from the pre-registered packet (Issue #810 comment 6715501420); a test
here fails if the code and the registered numbers diverge. Nothing here uses the
network: failures are built from the standard-library exception types the transport
chains.
"""

from __future__ import annotations

import math
import socket
import urllib.error

import pytest

from trade_rl.evaluation.paper import collection_policy as policy
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.attempt2 import SCREEN_SLOTS
from trade_rl.integrations.binance.forward import ForwardFreshnessError
from trade_rl.integrations.binance.types import BinanceTransportError


def _transport(cause: BaseException | None) -> BinanceTransportError:
    error = BinanceTransportError("Binance request failed for https://example.test")
    error.__cause__ = cause
    return error


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test", code, "x", {}, None)  # type: ignore[arg-type]


def test_registered_constants_equal_the_pre_registered_values() -> None:
    assert policy.RETRIES_PER_SLOT == 2
    assert policy.RETRY_DELAY_SECONDS == 2
    assert policy.RETRY_START_WINDOW_SECONDS == 30
    assert policy.MAX_CONSECUTIVE_SKIPPED_SLOTS == 1
    assert policy.MAX_SKIPPED_SLOTS_PER_HOUR == 2
    assert policy.MAX_SKIPPED_SLOTS_TOTAL == 129
    assert policy.CADENCE_SECONDS == 60
    assert policy.SKIP_WINDOW_SLOTS == 3600 // policy.CADENCE_SECONDS == 60
    assert policy.RULES_HOLD_SECONDS == 3600
    assert policy.RULES_REFRESH_SECONDS == 3300
    default_gap = PaperSettings.__dataclass_fields__["maximum_gap_seconds"].default
    assert default_gap == 180
    assert policy.MAX_SKIPPED_SLOTS_TOTAL == math.floor(SCREEN_SLOTS / 1000)
    # One skipped slot stays inside the frozen gap limit, two consecutive ones reach it.
    assert (policy.MAX_CONSECUTIVE_SKIPPED_SLOTS + 1) * policy.CADENCE_SECONDS < (
        default_gap
    )
    assert default_gap <= (policy.MAX_CONSECUTIVE_SKIPPED_SLOTS + 2) * (
        policy.CADENCE_SECONDS
    )


@pytest.mark.parametrize("code", [401, 403, 418, 429, 451])
def test_denial_statuses_are_class_d(code: int) -> None:
    assert policy.classify_failure(_transport(_http(code))) == "D"
    assert policy.classify_failure(_http(code)) == "D"


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_listed_server_statuses_are_class_t(code: int) -> None:
    assert policy.classify_failure(_transport(_http(code))) == "T"


@pytest.mark.parametrize("code", [400, 404, 405, 408, 501, 505])
def test_every_other_status_is_class_i(code: int) -> None:
    assert policy.classify_failure(_transport(_http(code))) == "I"


@pytest.mark.parametrize(
    "cause",
    [
        TimeoutError("timed out"),
        urllib.error.URLError(socket.gaierror(-2, "name resolution")),
        urllib.error.URLError(ConnectionRefusedError()),
        urllib.error.URLError(TimeoutError()),
    ],
)
def test_timeout_connection_and_dns_errors_are_class_t(cause: BaseException) -> None:
    assert policy.classify_failure(_transport(cause)) == "T"


@pytest.mark.parametrize(
    "error", [ConnectionResetError(), TimeoutError(), socket.gaierror(-2, "dns")]
)
def test_raw_connection_errors_are_class_t(error: BaseException) -> None:
    assert policy.classify_failure(error) == "T"


def test_a_freshness_rejection_of_one_capture_is_class_t() -> None:
    assert policy.classify_failure(ForwardFreshnessError("stale")) == "T"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("source unavailable"),
        ValueError("depth book is crossed or locked"),
        ValueError("depth levels must be strictly ordered"),
        ValueError("local clock moved backwards"),
        ValueError("snapshot consumed too late after capture"),
        BinanceTransportError("Binance response exceeds configured maximum"),
        _transport(None),
        PermissionError("journal"),
        OSError("disk"),
        KeyError("x"),
        KeyboardInterrupt(),
    ],
)
def test_integrity_malformed_and_unknown_errors_are_class_i(
    error: BaseException,
) -> None:
    assert policy.classify_failure(error) == "I"


def test_a_denial_anywhere_in_the_cause_chain_wins_over_a_transient_type() -> None:
    outer = _transport(_http(429))
    outer.__cause__.__cause__ = TimeoutError()  # type: ignore[union-attr]
    assert policy.classify_failure(outer) == "D"


def test_budgets_allow_isolated_skips_and_name_each_breach() -> None:
    assert policy.first_budget_breach([]) is None
    assert policy.first_budget_breach([5]) is None
    # Two skips in an hour are allowed when not adjacent; the third is not.
    assert policy.first_budget_breach([5, 7]) is None
    assert policy.first_budget_breach([5, 7, 9]) == (9, "per_hour")
    # Two adjacent slots breach the consecutive budget first.
    assert policy.first_budget_breach([5, 6]) == (6, "consecutive")
    # The window is 60 slots: a skip exactly 60 slots later leaves the window.
    assert policy.first_budget_breach([0, 2, 60]) is None
    assert policy.first_budget_breach([0, 2, 59]) == (59, "per_hour")
    # Total budget: 129 spaced skips pass, the 130th breaches.
    spaced = list(range(0, 130 * 30, 30))
    assert policy.first_budget_breach(spaced[:129]) is None
    assert policy.first_budget_breach(spaced) == (spaced[-1], "total")


def test_budget_scan_reports_the_first_breach_in_slot_order() -> None:
    assert policy.first_budget_breach([10, 11, 12]) == (11, "consecutive")
    assert policy.first_budget_breach([1, 3, 5, 6]) == (5, "per_hour")


def test_skip_budget_error_carries_its_budget_and_slot() -> None:
    error = policy.SkipBudgetExceeded("per_hour", 9)
    assert error.budget == "per_hour" and error.slot == 9
    assert "per_hour" in str(error) and "9" in str(error)


def test_registered_skip_reasons_and_purposes_equal_what_the_collector_registers() -> (
    None
):
    """The evaluator accepts only the classes the collector can register.

    The literals in ``supervisor`` are read from the source so a new reason cannot be
    registered without the audit (and therefore the sealed screen) naming it.
    """
    import ast
    from pathlib import Path

    import trade_rl.evaluation.paper.supervisor as supervisor_module

    tree = ast.parse(Path(supervisor_module.__file__).read_text(encoding="utf-8"))
    reasons: set[str] = set()
    purposes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name == "_skip" and node.args and isinstance(node.args[0], ast.Constant):
            reasons.add(str(node.args[0].value))
        if name == "register_skips" and isinstance(node.args[-1], ast.Constant):
            reasons.add(str(node.args[-1].value))
        for keyword in node.keywords:
            if keyword.arg == "purpose" and isinstance(keyword.value, ast.Constant):
                purposes.add(str(keyword.value.value))
    assert reasons == set(policy.SKIP_REASONS)
    assert purposes == set(policy.ATTEMPT_PURPOSES)
    assert policy.SKIP_REASONS == tuple(sorted(policy.SKIP_REASONS))


def test_the_failed_capture_bound_per_slot_follows_the_retry_allowance() -> None:
    """Each retried failure uses the shared allowance; at most two acquisitions end a
    slot with a failure (a held rules refresh and a snapshot or execution capture)."""
    assert policy.MAX_FAILED_CAPTURES_PER_SLOT == policy.RETRIES_PER_SLOT + 2
