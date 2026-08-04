from __future__ import annotations

from trade_rl.integrations.oracle_bellman_torch import (
    _resolve_compile_fallback_reason,
)


def test_compile_setup_failure_reason_is_preserved() -> None:
    assert (
        _resolve_compile_fallback_reason(
            setup_reason="compile_setup_failed:RuntimeError",
            execution_reason=None,
        )
        == "compile_setup_failed:RuntimeError"
    )


def test_compile_execution_failure_takes_precedence() -> None:
    assert (
        _resolve_compile_fallback_reason(
            setup_reason=None,
            execution_reason="compile_failed:BackendCompilerFailed",
        )
        == "compile_failed:BackendCompilerFailed"
    )
