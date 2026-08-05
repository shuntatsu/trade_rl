from __future__ import annotations

from pathlib import Path

from tests.architecture.import_references import (
    cross_package_private_usage_violations,
)


def _violations(
    tmp_path: Path,
    source: str,
    *,
    module_name: str = "trade_rl.rl.consumer",
) -> tuple[str, ...]:
    path = tmp_path / "private_usage.py"
    path.write_text(source, encoding="utf-8")
    return cross_package_private_usage_violations(
        path,
        module_name=module_name,
        root_package="trade_rl",
    )


def test_detects_direct_and_module_alias_private_access(tmp_path: Path) -> None:
    removed_private_name = "_" + "portfolio_states"
    violations = _violations(
        tmp_path,
        f"""
from trade_rl.evaluation.metrics import _hidden
import trade_rl.evaluation.metrics as metrics
from trade_rl.evaluation import series as series_module
import trade_rl.learning.oracle_teacher

metrics._evaluate()
series_module._coerce()
trade_rl.learning.oracle_teacher.{removed_private_name}()
""",
    )

    assert tuple(item.rsplit(":", 1)[-1] for item in violations) == (
        "trade_rl.evaluation.metrics._hidden",
        "trade_rl.evaluation.metrics._evaluate",
        "trade_rl.evaluation.series._coerce",
        f"trade_rl.learning.oracle_teacher.{removed_private_name}",
    )


def test_ignores_public_same_package_dunder_and_shadowed_access(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.rl.training as training
import trade_rl.evaluation.metrics as metrics

training._internal_helper()
metrics.evaluate_performance()
metrics.__name__


def consume(metrics: object) -> object:
    return metrics._local_attribute

metrics = object()
metrics._local_attribute
""",
    )

    assert violations == ()


def test_definition_names_shadow_imported_aliases(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics
import trade_rl.evaluation.series as series_module


def metrics() -> None:
    return None


class series_module:
    _local_attribute = object()

metrics._local_attribute
series_module._local_attribute
""",
    )

    assert violations == ()


def test_class_name_is_visible_when_method_body_executes(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as Metrics


class Metrics:
    def read(self) -> object:
        return Metrics._local_attribute
""",
    )

    assert violations == ()


def test_deleted_and_augmented_names_no_longer_retain_import_targets(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as deleted_metrics
import trade_rl.evaluation.series as updated_series

del deleted_metrics
updated_series += object()

deleted_metrics._local_attribute
updated_series._local_attribute
""",
    )

    assert violations == ()


def test_conditional_shadowing_preserves_possible_import_targets(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as if_metrics
import trade_rl.evaluation.series as match_series
import trade_rl.learning.oracle_teacher as loop_teacher

if condition:
    if_metrics = object()

match subject:
    case {"value": match_series}:
        pass

for loop_teacher in iterable:
    pass

if_metrics._conditional_private
match_series._conditional_private
loop_teacher._conditional_private
""",
    )

    assert tuple(item.rsplit(":", 1)[-1] for item in violations) == (
        "trade_rl.evaluation.metrics._conditional_private",
        "trade_rl.evaluation.series._conditional_private",
        "trade_rl.learning.oracle_teacher._conditional_private",
    )


def test_match_pattern_names_shadow_imported_aliases(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        """
import trade_rl.evaluation.metrics as metrics

match {"value": object()}:
    case {"value": metrics}:
        metrics._local_attribute
""",
    )

    assert violations == ()


def test_nested_class_body_sees_its_alias_but_method_does_not(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
def build() -> type[object]:
    class Local:
        import trade_rl.evaluation.metrics as metrics

        value = metrics._class_private

        def method(self) -> object:
            return metrics._not_a_lexical_binding

    return Local
""",
    )

    assert tuple(item.rsplit(":", 1)[-1] for item in violations) == (
        "trade_rl.evaluation.metrics._class_private",
    )


def test_nested_class_and_method_do_not_close_over_outer_class_alias(
    tmp_path: Path,
) -> None:
    violations = _violations(
        tmp_path,
        """
class Outer:
    import trade_rl.evaluation.metrics as outer_metrics

    class Nested:
        value = outer_metrics._not_a_nested_class_binding

    def build(self) -> type[object]:
        class Local:
            value = outer_metrics._not_a_method_binding

        return Local
""",
    )

    assert violations == ()
