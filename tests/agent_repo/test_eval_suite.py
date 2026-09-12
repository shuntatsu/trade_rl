from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.agent_repo.eval_suite import (
    DEFAULT_SUITE,
    EvalSuite,
    load_eval_suite,
    score_eval,
)

DIMENSIONS = [
    "authority_discovery",
    "authority_reuse",
    "boundary_compliance",
    "scope_discipline",
    "test_discovery",
    "verification_selection",
    "compatibility_awareness",
    "context_efficiency",
]


def _payload() -> dict[str, object]:
    return {
        "schema_version": "agent_repo_eval_suite_v1",
        "rubric_version": "agent_repo_rubric_v1",
        "rubric": {
            "dimensions": DIMENSIONS,
            "max_score_per_dimension": 2,
            "critical_dimensions": [
                "authority_reuse",
                "boundary_compliance",
                "scope_discipline",
            ],
        },
        "tasks": [
            {
                "task_id": "example",
                "prompt": "Change one maintained semantic contract safely.",
            }
        ],
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_eval_suite_accepts_exact_contract(tmp_path: Path) -> None:
    suite = load_eval_suite(_write(tmp_path / "suite.json", _payload()))

    assert isinstance(suite, EvalSuite)
    assert suite.schema_version == "agent_repo_eval_suite_v1"
    assert suite.rubric_version == "agent_repo_rubric_v1"
    assert tuple(task.task_id for task in suite.tasks) == ("example",)
    assert suite.rubric.dimensions == tuple(DIMENSIONS)
    assert suite.rubric.max_score_per_dimension == 2


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("unknown", True),
        lambda value: value["tasks"].append(dict(value["tasks"][0])),
        lambda value: value["tasks"][0].__setitem__("prompt", ""),
        lambda value: value["tasks"][0].__setitem__(
            "semantic_goal", "evaluator-only answer key"
        ),
        lambda value: value["rubric"]["critical_dimensions"].append("unknown"),
        lambda value: value["rubric"].__setitem__("max_score_per_dimension", 0),
        lambda value: value.__setitem__("tasks", []),
    ],
)
def test_load_eval_suite_rejects_invalid_contract(
    tmp_path: Path,
    mutator,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValueError):
        load_eval_suite(_write(tmp_path / "suite.json", payload))


def test_checked_in_suite_does_not_expose_task_specific_evaluator_answer_key() -> None:
    raw = json.loads(DEFAULT_SUITE.read_text(encoding="utf-8"))

    assert raw["tasks"]
    assert all(set(task) == {"task_id", "prompt"} for task in raw["tasks"])


def test_score_eval_requires_exact_dimensions_and_evidence(tmp_path: Path) -> None:
    suite = load_eval_suite(_write(tmp_path / "suite.json", _payload()))
    scores = {dimension: (2, f"evidence for {dimension}") for dimension in DIMENSIONS}

    result = score_eval(suite, task_id="example", scores=scores)

    assert result.total == 16
    assert result.maximum == 16
    assert result.critical_failure is False
    assert tuple(item.dimension for item in result.dimensions) == tuple(DIMENSIONS)

    missing = dict(scores)
    del missing[DIMENSIONS[0]]
    with pytest.raises(ValueError):
        score_eval(suite, task_id="example", scores=missing)

    extra = dict(scores)
    extra["unknown"] = (1, "evidence")
    with pytest.raises(ValueError):
        score_eval(suite, task_id="example", scores=extra)

    out_of_range = dict(scores)
    out_of_range[DIMENSIONS[0]] = (3, "evidence")
    with pytest.raises(ValueError):
        score_eval(suite, task_id="example", scores=out_of_range)

    empty_evidence = dict(scores)
    empty_evidence[DIMENSIONS[0]] = (1, "")
    with pytest.raises(ValueError):
        score_eval(suite, task_id="example", scores=empty_evidence)


def test_score_eval_marks_zero_in_critical_dimension(tmp_path: Path) -> None:
    suite = load_eval_suite(_write(tmp_path / "suite.json", _payload()))
    scores = {dimension: (2, f"evidence for {dimension}") for dimension in DIMENSIONS}
    scores["authority_reuse"] = (0, "parallel authority introduced")

    result = score_eval(suite, task_id="example", scores=scores)

    assert result.critical_failure is True
