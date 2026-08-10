from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"
RUNNER_PATH = EXAMPLE_ROOT / "run_full_research_state.py"
EXPECTED_DEFAULT_CANDIDATES = (
    (
        "target-weight-growth-gamma-one-ppo",
        "training-target-weight-growth-ppo.json",
    ),
    (
        "target-weight-constrained-growth-gamma-one",
        "training-target-weight-constrained-growth.json",
    ),
    (
        "target-weight-constrained-growth-discounted-168h",
        "training-target-weight-constrained-growth-discounted.json",
    ),
)


def _runner_namespace() -> dict[str, Any]:
    return runpy.run_path(str(RUNNER_PATH))


def _candidate_rows(path: Path) -> tuple[tuple[str, str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        (str(candidate["name"]), str(candidate["run_file"]))
        for candidate in payload["candidates"]
    )


def test_default_full_research_uses_target_weight_growth_catalog() -> None:
    namespace = _runner_namespace()

    assert "_DEFAULT_WALK_FORWARD_TEMPLATE" in namespace
    template = namespace["_DEFAULT_WALK_FORWARD_TEMPLATE"]
    assert isinstance(template, Path)
    assert template == (
        EXAMPLE_ROOT / "walk-forward-target-weight-constrained-growth.json"
    ).resolve()
    assert _candidate_rows(template) == EXPECTED_DEFAULT_CANDIDATES


def test_training_full_is_available_only_through_explicit_template_selection() -> None:
    namespace = _runner_namespace()
    template = namespace["_DEFAULT_WALK_FORWARD_TEMPLATE"]
    example_template = namespace["_example_template"]

    assert "training-full.json" not in {
        run_file for _, run_file in _candidate_rows(template)
    }
    assert example_template(
        "training-full.json",
        field="training template",
    ) == (EXAMPLE_ROOT / "training-full.json").resolve()
