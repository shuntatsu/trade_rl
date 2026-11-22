"""Driver for the learning-lab harness (``anchor_rl_execution_dev_v1``).

``python -m tools.learning_lab plan [--methods ...] [--seeds 0 1 2 3 4]``
``python -m tools.learning_lab train --method L1-PPO --seed 0 --registry R --output-dir D``
``python -m tools.learning_lab evaluate --method L1-PPO --seed 0 --split validate ...``
``python -m tools.learning_lab checkpoint --registry R``
``python -m tools.learning_lab registry --registry R [--deflated-for-last]``
``python -m tools.learning_lab registry --registry R --expected-trial-count N --expected-head-sha256 H``

``plan`` and ``registry`` read no data. ``train`` and ``evaluate`` default to ``--data
synthetic`` (a seeded generator; nothing is read or fetched). ``--data real`` is refused
unless ``--authorization-url`` is an owner G4 comment on this repository that quotes the
exact study id and protocol digest and was not edited between two fetches (the same
verifier as the other drivers, :mod:`tools.core7_g4_comment`); the check runs before any
file or row is read. Even then this version has no real-data loader wired: an authorized
real-data request ends with a distinct exit code and runs nothing. The driver authorizes
no economic claim, no orders and no data at or after 2026-08-01.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from tools import core7_g4_comment as g4
from tools import core7_mtf_tierd as tierd_driver
from trade_rl.evaluation.agy_anchor_oos import protocol as anchor_protocol
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.evaluation.learning_lab import agents, runner, splits
from trade_rl.evaluation.learning_lab.registry import RegistryCheckpoint, TrialRegistry
from trade_rl.evaluation.learning_lab.synthetic import synthetic_stage

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_UNAUTHENTICATED = 5
EXIT_NOT_WIRED = 6

DATA_SYNTHETIC = "synthetic"
DATA_REAL = "real"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
GitState = Callable[[], tuple[str, bool]]
SetupFactory = Callable[[argparse.Namespace], runner.LabSetup]


def _synthetic_setup(args: argparse.Namespace) -> runner.LabSetup:
    stage = synthetic_stage(n_bars=args.synthetic_bars, seed=args.data_seed)
    return runner.LabSetup(
        dataset=stage.dataset,
        signals=stage.signals,
        execution_cost=anchor_protocol.scenario_cost("BASE"),
        risk_config=anchor_protocol.risk_config(),
        splits=runner.synthetic_splits(args.synthetic_bars),
        data_kind=DATA_SYNTHETIC,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools.learning_lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print the runs and the protocol digest")
    plan.add_argument("--methods", nargs="+", default=list(runner.method_ids()))
    plan.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--method", required=True, choices=runner.method_ids())
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--registry", type=Path, required=True)
        sub.add_argument(
            "--data", choices=(DATA_SYNTHETIC, DATA_REAL), default=DATA_SYNTHETIC
        )
        sub.add_argument(
            "--authorization-url",
            default=None,
            help="owner G4 comment (#810) quoting the study id and protocol digest; "
            "required for --data real",
        )
        sub.add_argument("--timesteps", type=int, default=2048)
        sub.add_argument("--synthetic-bars", type=int, default=400)
        sub.add_argument("--data-seed", type=int, default=0)

    train = commands.add_parser("train", help="train one agent on the train split")
    common(train)
    train.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate", help="evaluate one method on validate/test"
    )
    common(evaluate)
    evaluate.add_argument(
        "--split", choices=(splits.VALIDATE, splits.TEST), required=True
    )
    evaluate.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="directory of the registered checkpoint (learned methods)",
    )

    checkpoint = commands.add_parser(
        "checkpoint", help="print a verified head/count for independent storage"
    )
    checkpoint.add_argument("--registry", type=Path, required=True)

    registry = commands.add_parser("registry", help="verify and summarize the registry")
    registry.add_argument("--registry", type=Path, required=True)
    registry.add_argument(
        "--deflated-for-last",
        action="store_true",
        help="deflated Sharpe of the last record against every trial",
    )
    registry.add_argument(
        "--expected-trial-count",
        type=int,
        default=None,
        help="authenticated external checkpoint count; requires head sha256",
    )
    registry.add_argument(
        "--expected-head-sha256",
        default=None,
        help="authenticated external checkpoint terminal line SHA-256; requires count",
    )
    registry.add_argument(
        "--allow-checkpoint-extensions",
        action="store_true",
        help="permit later appended trials if the pinned prefix is identical",
    )
    return parser


def _authenticate(url: str | None, gh: g4.GhRunner) -> tuple[g4.CommentSnapshot, str]:
    """Fetch and check the owner comment; nothing is read before this succeeds."""
    if url is None:
        raise g4.G4AuthError("--data real needs --authorization-url (an #810 comment)")
    try:
        core7_protocol.validate_authorization_url(url)
    except (TypeError, ValueError) as error:
        raise g4.G4AuthError(str(error)) from error
    digest = runner.protocol_digest()
    first = g4.fetch_comment(url, gh)
    g4.require_protocol_digest(first, digest)
    g4.require_protocol_id(first, runner.STUDY_ID)
    return first, digest


def _agent_config(args: argparse.Namespace) -> agents.AgentConfig:
    return agents.AgentConfig(
        algorithm=agents.algorithm_for_method(args.method),
        total_timesteps=args.timesteps,
        seed=args.seed,
    )


def _run(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    state: GitState,
    gh: g4.GhRunner,
    setup_factory: SetupFactory | None,
) -> int:
    receipt: dict[str, Any] | None = None
    first: g4.CommentSnapshot | None = None
    digest = runner.protocol_digest()
    try:
        if args.data == DATA_REAL:
            # authenticate the owner comment before any file, row or data is touched
            first, digest = _authenticate(args.authorization_url, gh)
            if setup_factory is None:
                err.write(
                    "authorized, but no real-data loader is wired in this version; "
                    "nothing was read or run\n"
                )
                return EXIT_NOT_WIRED
        commit, clean = state()
        if args.data == DATA_REAL and not clean:
            raise ValueError("the working tree must be clean so the commit is exact")
        code_commit = commit if clean else f"{commit}-dirty"
        factory = setup_factory if args.data == DATA_REAL else _synthetic_setup
        assert factory is not None
        setup = factory(args)
        if args.data == DATA_REAL:
            assert first is not None
            confirmed = g4.confirm_unedited(
                args.authorization_url, first, gh, protocol_digest=digest
            )
            receipt = confirmed.to_dict()
        registry = TrialRegistry(args.registry)
        agent = None if args.method in _baseline_ids() else _agent_config(args)
        if args.command == "train":
            if agent is None:
                raise ValueError("a baseline is not trained")
            trained, record = runner.train_method(
                setup,
                args.method,
                args.seed,
                agent_config=agent,
                output_dir=args.output_dir,
                registry=registry,
                code_commit=code_commit,
            )
            document: dict[str, Any] = {
                "command": "train",
                "checkpoint": str(trained.checkpoint),
                "checkpoint_sha256": trained.checkpoint_sha256,
                "record": record.to_dict(),
            }
        else:
            checkpoint: Path | None = None
            sha: str | None = None
            if agent is not None:
                checkpoint, sha = _registered_checkpoint(registry, setup, args, agent)
            record = runner.evaluate_method(
                setup,
                args.method,
                args.seed,
                args.split,
                registry=registry,
                code_commit=code_commit,
                agent_config=agent,
                checkpoint=checkpoint,
                checkpoint_sha256=sha,
            )
            document = {"command": "evaluate", "record": record.to_dict()}
        document["data"] = args.data
        document["authorization_receipt"] = receipt
        out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    except g4.G4AuthError as error:
        err.write(f"authorization failed: {error}\n")
        return EXIT_UNAUTHENTICATED
    except (ValueError, RuntimeError, OSError, agents.AgentError) as error:
        err.write(f"refused: {error}\n")
        return EXIT_REFUSED


def _baseline_ids() -> set[str]:
    from trade_rl.evaluation.learning_lab import baselines

    return set(baselines.baseline_methods())


def _registered_checkpoint(
    registry: TrialRegistry,
    setup: runner.LabSetup,
    args: argparse.Namespace,
    agent: agents.AgentConfig,
) -> tuple[Path, str]:
    if args.checkpoint_dir is None:
        raise ValueError("a learned method needs --checkpoint-dir")
    digest = runner.config_digest(setup, agent, args.method)
    for record in reversed(registry.records()):
        if (
            record.method_id == args.method
            and record.seed == args.seed
            and record.config_digest == digest
            and record.split == splits.TRAIN
            and record.status == "completed"
        ):
            sha = str(record.metrics["checkpoint_sha256"])
            path = args.checkpoint_dir / f"{agent.algorithm}-seed{agent.seed}.zip"
            return path, sha
    raise ValueError("no completed train trial of this method, seed and configuration")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state: GitState = tierd_driver.git_state,
    gh: g4.GhRunner = g4.default_gh,
    setup_factory: SetupFactory | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        try:
            runs = runner.plan(args.methods, args.seeds)
        except runner.RunnerError as error:
            err.write(f"refused: {error}\n")
            return EXIT_REFUSED
        document: dict[str, Any] = {
            "study_id": runner.STUDY_ID,
            "protocol_digest": runner.protocol_digest(),
            "protocol": runner.protocol_payload(),
            "runs": runs,
        }
        out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    if args.command == "checkpoint":
        try:
            checkpoint = TrialRegistry(args.registry).checkpoint()
        except ValueError as error:
            err.write(f"refused: {error}\n")
            return EXIT_REFUSED
        out.write(json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    if args.command == "registry":
        registry = TrialRegistry(args.registry)
        try:
            supplied_count = args.expected_trial_count is not None
            supplied_hash = args.expected_head_sha256 is not None
            if supplied_count != supplied_hash:
                raise ValueError(
                    "checkpoint needs both expected count and head SHA-256"
                )
            if args.allow_checkpoint_extensions and not supplied_count:
                raise ValueError("checkpoint extensions need an external checkpoint")
            expected = (
                RegistryCheckpoint(
                    trial_count=args.expected_trial_count,
                    head_sha256=args.expected_head_sha256,
                )
                if supplied_count
                else None
            )
            if expected is None:
                records = registry.records()
            else:
                records, _ = registry.snapshot(
                    expected_checkpoint=expected,
                    allow_extensions=args.allow_checkpoint_extensions,
                )
            document = {
                "trials": len(records),
                "chain_verified": True,
                "records": [r.to_dict() for r in records],
            }
            if expected is not None:
                document["checkpoint_verified"] = True
            if args.deflated_for_last and records:
                document["deflated_sharpe"] = runner.deflated_for(records)
        except ValueError as error:
            err.write(f"refused: {error}\n")
            return EXIT_REFUSED
        out.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    return _run(args, out=out, err=err, state=state, gh=gh, setup_factory=setup_factory)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
