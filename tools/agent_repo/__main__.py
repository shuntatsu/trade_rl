"""Network-free CLI for source-derived repository-agent inspection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from tools.agent_repo.eval_suite import (
    DEFAULT_SUITE,
    EvalTask,
    load_eval_suite,
    score_eval,
)
from tools.agent_repo.git_state import read_git_state
from tools.agent_repo.semantic_diff import semantic_diff
from tools.agent_repo.source_index import SourceIndex
from tools.agent_repo.verification import plan_verification


def _write_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--base", dest="base_ref")

    context = subparsers.add_parser("context")
    context.add_argument("path")

    impact = subparsers.add_parser("impact")
    impact.add_argument("paths", nargs="+")

    diff = subparsers.add_parser("diff")
    diff.add_argument("--base", dest="base_ref", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--base", dest="base_ref", required=True)

    subparsers.add_parser("eval-list")
    eval_show = subparsers.add_parser("eval-show")
    eval_show.add_argument("task_id")
    eval_score = subparsers.add_parser("eval-score")
    eval_score.add_argument("task_id")
    eval_score.add_argument("score_json")
    return parser


def _eval_task(task_id: str) -> EvalTask:
    suite = load_eval_suite(DEFAULT_SUITE)
    for task in suite.tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown eval task: {task_id}")


def _score_input(path: Path) -> dict[str, tuple[int, str]]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("eval score input must be valid JSON") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("eval score input must be a JSON object")
    result: dict[str, tuple[int, str]] = {}
    for dimension, value in decoded.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("each eval score must be [integer, evidence]")
        score, evidence = value
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError("eval score value must be an integer")
        if not isinstance(evidence, str):
            raise ValueError("eval score evidence must be a string")
        result[dimension] = (score, evidence)
    return result


def _dispatch(args: argparse.Namespace, repository: Path) -> object:
    command = str(args.command)
    if command == "preflight":
        return asdict(read_git_state(repository, base_ref=args.base_ref))
    if command == "diff":
        return {
            "signals": [
                asdict(signal)
                for signal in semantic_diff(repository, base_ref=str(args.base_ref))
            ]
        }
    if command == "verify":
        return {
            "steps": [
                asdict(step)
                for step in plan_verification(repository, base_ref=str(args.base_ref))
            ]
        }
    if command == "eval-list":
        suite = load_eval_suite(DEFAULT_SUITE)
        return {
            "rubric_version": suite.rubric_version,
            "task_ids": [task.task_id for task in suite.tasks],
        }
    if command == "eval-show":
        return asdict(_eval_task(str(args.task_id)))
    if command == "eval-score":
        suite = load_eval_suite(DEFAULT_SUITE)
        score_path = Path(str(args.score_json))
        if not score_path.is_absolute():
            score_path = repository / score_path
        result = score_eval(
            suite,
            task_id=str(args.task_id),
            scores=cast(dict[str, tuple[int, str]], _score_input(score_path)),
        )
        return asdict(result)

    index = SourceIndex.build(repository)
    if command == "context":
        return asdict(index.context(str(args.path)))
    if command == "impact":
        return {
            "contexts": [asdict(value) for value in index.impact(tuple(args.paths))]
        }
    raise ValueError(f"unsupported command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload: Any = _dispatch(args, Path.cwd())
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        SyntaxError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    _write_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
