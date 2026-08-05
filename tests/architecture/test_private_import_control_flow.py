from __future__ import annotations

from pathlib import Path

from tests.architecture.import_references import (
    cross_package_private_usage_violations,
)


def _violations(tmp_path: Path, source: str) -> tuple[str, ...]:
    path = tmp_path / "private_control_flow.py"
    path.write_text(source, encoding="utf-8")
    return cross_package_private_usage_violations(
        path,
        module_name="trade_rl.rl.consumer",
        root_package="trade_rl",
    )


def _targets(violations: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item.rsplit(":", 1)[-1] for item in violations)


def test_while_body_shadow_preserves_possible_import_after_zero_iterations(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics

while condition:
    metrics = object()

metrics._private_after_zero_iterations
""",
    )

    assert _targets(violations) == (
        "trade_rl.evaluation.metrics._private_after_zero_iterations",
    )


def test_try_handler_shadow_preserves_normal_path_import(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics

try:
    operation()
except Exception:
    metrics = object()

metrics._private_after_success
""",
    )

    assert _targets(violations) == (
        "trade_rl.evaluation.metrics._private_after_success",
    )


def test_try_handlers_are_analyzed_from_the_same_pre_try_scope(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics

try:
    operation()
except ValueError:
    metrics = object()
except Exception:
    metrics._private_in_second_handler
""",
    )

    assert _targets(violations) == (
        "trade_rl.evaluation.metrics._private_in_second_handler",
    )


def test_try_else_is_not_shadowed_by_an_unrelated_handler(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics

try:
    operation()
except Exception:
    metrics = object()
else:
    metrics._private_on_success
""",
    )

    assert _targets(violations) == ("trade_rl.evaluation.metrics._private_on_success",)


def test_finally_shadow_is_definite_after_try(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics

try:
    operation()
finally:
    metrics = object()

metrics._local_after_finally
""",
    )

    assert violations == ()
