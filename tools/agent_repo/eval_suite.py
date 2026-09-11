"""Strict loading and deterministic scoring for Agent-UX evaluation tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SUITE_SCHEMA = "agent_repo_eval_suite_v1"
RUBRIC_VERSION = "agent_repo_rubric_v1"
DEFAULT_SUITE = Path(__file__).with_name("evals") / "v1.json"


@dataclass(frozen=True, slots=True)
class EvalTask:
    task_id: str
    prompt: str
    semantic_goal: str
    critical_failures: tuple[str, ...]
    review_questions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalRubric:
    dimensions: tuple[str, ...]
    max_score_per_dimension: int
    critical_dimensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalSuite:
    schema_version: str
    rubric_version: str
    tasks: tuple[EvalTask, ...]
    rubric: EvalRubric


@dataclass(frozen=True, slots=True)
class DimensionScore:
    dimension: str
    score: int
    evidence: str


@dataclass(frozen=True, slots=True)
class EvalScore:
    task_id: str
    rubric_version: str
    dimensions: tuple[DimensionScore, ...]
    total: int
    maximum: int
    critical_failure: bool


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, field: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} keys differ from contract")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strings(
    value: object, *, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(_string(item, field=f"{field} item") for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _task(value: object) -> EvalTask:
    raw = _object(value, field="task")
    _exact_keys(
        raw,
        frozenset(
            {
                "task_id",
                "prompt",
                "semantic_goal",
                "critical_failures",
                "review_questions",
            }
        ),
        field="task",
    )
    return EvalTask(
        task_id=_string(raw["task_id"], field="task_id"),
        prompt=_string(raw["prompt"], field="prompt"),
        semantic_goal=_string(raw["semantic_goal"], field="semantic_goal"),
        critical_failures=_strings(raw["critical_failures"], field="critical_failures"),
        review_questions=_strings(raw["review_questions"], field="review_questions"),
    )


def _rubric(value: object) -> EvalRubric:
    raw = _object(value, field="rubric")
    _exact_keys(
        raw,
        frozenset({"dimensions", "max_score_per_dimension", "critical_dimensions"}),
        field="rubric",
    )
    dimensions = _strings(raw["dimensions"], field="rubric dimensions")
    maximum = raw["max_score_per_dimension"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_score_per_dimension must be a positive integer")
    critical = _strings(raw["critical_dimensions"], field="critical_dimensions")
    if not set(critical) <= set(dimensions):
        raise ValueError("critical_dimensions must be rubric dimensions")
    return EvalRubric(
        dimensions=dimensions,
        max_score_per_dimension=maximum,
        critical_dimensions=critical,
    )


def load_eval_suite(path: Path) -> EvalSuite:
    """Load one exact versioned eval suite without permissive defaults."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Agent eval suite must be valid JSON") from error
    raw = _object(decoded, field="suite")
    _exact_keys(
        raw,
        frozenset({"schema_version", "rubric_version", "rubric", "tasks"}),
        field="suite",
    )
    schema_version = _string(raw["schema_version"], field="schema_version")
    rubric_version = _string(raw["rubric_version"], field="rubric_version")
    if schema_version != SUITE_SCHEMA:
        raise ValueError("unsupported Agent eval suite schema")
    if rubric_version != RUBRIC_VERSION:
        raise ValueError("unsupported Agent eval rubric version")
    rubric = _rubric(raw["rubric"])
    tasks_value = raw["tasks"]
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ValueError("tasks must be a non-empty array")
    tasks = tuple(_task(item) for item in tasks_value)
    identifiers = tuple(task.task_id for task in tasks)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("task_id values must be unique")
    return EvalSuite(
        schema_version=schema_version,
        rubric_version=rubric_version,
        tasks=tasks,
        rubric=rubric,
    )


def score_eval(
    suite: EvalSuite,
    *,
    task_id: str,
    scores: Mapping[str, tuple[int, str]],
) -> EvalScore:
    """Score evaluator-authored evidence against the exact suite rubric."""

    if task_id not in {task.task_id for task in suite.tasks}:
        raise ValueError(f"unknown eval task: {task_id}")
    expected = set(suite.rubric.dimensions)
    if set(scores) != expected:
        raise ValueError("score dimensions differ from rubric")
    dimension_scores: list[DimensionScore] = []
    for dimension in suite.rubric.dimensions:
        score, evidence = scores[dimension]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"score for {dimension} must be an integer")
        if not 0 <= score <= suite.rubric.max_score_per_dimension:
            raise ValueError(f"score for {dimension} is outside rubric range")
        evidence_value = _string(evidence, field=f"evidence for {dimension}")
        dimension_scores.append(
            DimensionScore(dimension=dimension, score=score, evidence=evidence_value)
        )
    maximum = len(suite.rubric.dimensions) * suite.rubric.max_score_per_dimension
    total = sum(item.score for item in dimension_scores)
    by_dimension = {item.dimension: item.score for item in dimension_scores}
    critical_failure = any(
        by_dimension[dimension] == 0 for dimension in suite.rubric.critical_dimensions
    )
    return EvalScore(
        task_id=task_id,
        rubric_version=suite.rubric_version,
        dimensions=tuple(dimension_scores),
        total=total,
        maximum=maximum,
        critical_failure=critical_failure,
    )


__all__ = [
    "DEFAULT_SUITE",
    "DimensionScore",
    "EvalRubric",
    "EvalScore",
    "EvalSuite",
    "EvalTask",
    "load_eval_suite",
    "score_eval",
]
