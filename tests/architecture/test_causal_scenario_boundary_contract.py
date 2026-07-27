from __future__ import annotations

from pathlib import Path

from tests.architecture.import_references import (
    causal_scenario_dependency_violations,
    forbidden_json_key_paths,
)

_PROHIBITED = "trade_rl.workflows.causal_scenario"


def _source(tmp_path: Path, relative: str, source: str) -> Path:
    path = tmp_path / "trade_rl" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _violations(tmp_path: Path) -> tuple[str, ...]:
    package_root = tmp_path / "trade_rl"
    return causal_scenario_dependency_violations(
        protected_roots=(
            package_root / "rl",
            package_root / "serving",
            package_root / "release",
            package_root / "workflows",
            package_root / "integrations",
        ),
        excluded_root=package_root / "workflows" / "causal_scenario",
        package_root=package_root,
        root_package="trade_rl",
        prohibited_prefix=_PROHIBITED,
    )


def test_comments_docstrings_and_strings_do_not_create_boundary_edges(
    tmp_path: Path,
) -> None:
    _source(
        tmp_path,
        "rl/comment_only.py",
        '''
"""trade_rl.workflows.causal_scenario.replay"""
# from trade_rl.workflows.causal_scenario import replay
MESSAGE = "import trade_rl.workflows.causal_scenario.library"
''',
    )

    assert _violations(tmp_path) == ()


def test_lazy_and_literal_dynamic_imports_are_reported(tmp_path: Path) -> None:
    lazy = _source(
        tmp_path,
        "rl/lazy.py",
        """
def load() -> None:
    from trade_rl.workflows.causal_scenario import replay
""",
    )
    dynamic = _source(
        tmp_path,
        "serving/dynamic.py",
        """
import importlib as modules
modules.import_module("trade_rl.workflows.causal_scenario.library")
""",
    )

    assert _violations(tmp_path) == (
        f"{lazy}:3:from:{_PROHIBITED}.replay",
        f"{dynamic}:3:dynamic:{_PROHIBITED}.library",
    )


def test_unresolved_recognized_dynamic_import_fails_closed(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        "integrations/unresolved.py",
        """
import importlib
module_name = resolve_module_name()
importlib.import_module(module_name)
""",
    )

    assert _violations(tmp_path) == (f"{source}:4:dynamic:<unresolved>",)


def test_causal_scenario_package_is_excluded_from_runtime_boundary(
    tmp_path: Path,
) -> None:
    _source(
        tmp_path,
        "workflows/causal_scenario/internal.py",
        "from trade_rl.workflows.causal_scenario import replay\n",
    )

    assert _violations(tmp_path) == ()


def test_json_values_are_allowed_but_exact_mapping_keys_are_rejected() -> None:
    payload = {
        "description": "causal_scenario_library is discussed here",
        "nested": [
            {"safe": "causal_scenario_library"},
            {"causal_scenario_library": {"enabled": True}},
        ],
        "causal_scenario_library_suffix": "allowed",
    }

    assert forbidden_json_key_paths(
        payload,
        key="causal_scenario_library",
    ) == ("$.nested[1].causal_scenario_library",)


def test_json_key_detection_reports_all_nested_paths_deterministically() -> None:
    payload = {
        "causal_scenario_library": None,
        "z": {"causal_scenario_library": {}},
        "a": [{"causal_scenario_library": False}],
    }

    assert forbidden_json_key_paths(
        payload,
        key="causal_scenario_library",
    ) == (
        "$.causal_scenario_library",
        "$.a[0].causal_scenario_library",
        "$.z.causal_scenario_library",
    )
