#!/usr/bin/env python3
"""Run one externally approved phase of the maintained full research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.run_manifest import (
    validate_training_run_directory,
    validate_walk_forward_run_directory,
)
from trade_rl.data import load_market_dataset_artifact
from trade_rl.integrations.binance import BinancePublicTransport
from trade_rl.release.asymmetric import load_public_verification_keys
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.rl.sequence_observations import SequenceObservationBuilder
from trade_rl.workflows.binance_metadata_modes import BinanceMetadataMode
from trade_rl.workflows.full_research_state import (
    FullResearchStatus,
    ResearchPhase,
    ResearchPhaseOutcome,
    require_separate_cache_root,
    run_research_phase,
)
from trade_rl.workflows.selection_authorization import (
    SelectionProposal,
    load_selection_authorization,
    load_selection_proposal,
    write_selection_proposal,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    normalize_training_run_config,
)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import full_research_pipeline as pipeline  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SUPERVISED_BOOTSTRAP_ARTIFACTS = frozenset(
    {"cuda-preflight.json", "entrypoint-provenance.json", "heartbeat.json"}
)


def _require_fresh_develop_root(work_root: Path) -> None:
    unexpected = tuple(
        entry
        for entry in work_root.iterdir()
        if entry.name not in _SUPERVISED_BOOTSTRAP_ARTIFACTS or not entry.is_file()
    )
    if unexpected:
        raise FileExistsError(f"research generation already exists: {work_root}")


def _lockfile_digest() -> str:
    return hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest()


def _required_path(value: str | None, *, field: str) -> Path:
    if not value:
        raise ValueError(f"{field} is required")
    path = Path(value)
    if not path.is_absolute():
        path = _ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{field} is missing: {path}")
    return path


def _absolute_artifact_path(raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} is missing")
    path = Path(raw)
    return path if path.is_absolute() else _ROOT / path


def _normalize_selected_config(path: Path) -> TrainingRunConfig:
    payload = pipeline.load_json(path)
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("selected training config environment is missing")
    environment["liquidate_on_end"] = True
    payload["environment"] = environment
    payload["resume_checkpoints"] = {}
    pipeline.write_json(path, payload)
    config = normalize_training_run_config(TrainingRunConfig.from_json(path))
    if config.resume_checkpoints:
        raise ValueError("selected-final configuration must not resume checkpoints")
    return config


def _base_summary(
    *,
    dataset_a: dict[str, object],
    dataset_b: dict[str, object],
    resolution: object,
    metadata_report: dict[str, Any],
    dataset: object,
    flat_observation_count: int,
    sequence_observation_count: int,
    policy_observation_count: int,
    walk_forward: dict[str, Any],
) -> dict[str, object]:
    return {
        "dataset": dataset_a,
        "dataset_repeat": dataset_b,
        "decision_hours": 0.25,
        "end_time": pipeline._END,
        "flat_observation_count": flat_observation_count,
        "metadata": metadata_report,
        "metadata_evidence_digest": getattr(resolution, "evidence_digest"),
        "metadata_mode": getattr(resolution, "mode").value,
        "metadata_source": getattr(resolution, "source_uri"),
        "native_timeframes": list(pipeline._NATIVE_TIMEFRAMES),
        "policy_observation_count": policy_observation_count,
        "production_status": "NO-GO",
        "raw_feature_count": getattr(dataset, "n_features"),
        "schema": "binance_multitimeframe_full_research_state_v1",
        "sequence_observation_count": sequence_observation_count,
        "start_time": pipeline._START,
        "training": None,
        "walk_forward": walk_forward,
    }


class BinanceFullResearchStages:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self, phase: ResearchPhase, work_root: Path) -> ResearchPhaseOutcome:
        if phase is ResearchPhase.DEVELOP:
            return self._develop(work_root)
        if phase is ResearchPhase.TRAIN_SELECTED:
            return self._train_selected(work_root)
        return self._finalize(work_root)

    def _develop(self, work_root: Path) -> ResearchPhaseOutcome:
        _require_fresh_develop_root(work_root)
        cache_root = self.args.cache_root
        if not cache_root.is_absolute():
            cache_root = _ROOT / cache_root
        # run_research_phase creates work_root before dispatch; validate separation here.
        cache_root = require_separate_cache_root(cache_root, work_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        transport = BinancePublicTransport(
            timeout_seconds=60.0,
            max_attempts=4,
            retry_backoff_seconds=0.5,
            cache_root=cache_root,
        )
        resolution = pipeline.resolve_metadata(
            mode=BinanceMetadataMode(self.args.metadata_mode),
            transport=transport,
            conservative_static_path=self.args.conservative_static_path,
        )
        resolution.write_artifacts(work_root)
        metadata_report = pipeline.load_json(work_root / "exchange-info.json")

        dataset_a_path = work_root / "dataset-a"
        dataset_b_path = work_root / "dataset-b"
        dataset_a = pipeline.build_dataset(
            output=dataset_a_path,
            transport=transport,
            metadata=resolution.metadata,
            execution_rule_histories=resolution.execution_rule_histories,
            metadata_evidence=resolution.identity_evidence,
            metadata_mode=resolution.mode,
            metadata_evidence_digest=resolution.evidence_digest,
        )
        dataset_b = pipeline.build_dataset(
            output=dataset_b_path,
            transport=transport,
            metadata=resolution.metadata,
            execution_rule_histories=resolution.execution_rule_histories,
            metadata_evidence=resolution.identity_evidence,
            metadata_mode=resolution.mode,
            metadata_evidence_digest=resolution.evidence_digest,
        )
        if dataset_a["dataset_id"] != dataset_b["dataset_id"]:
            raise RuntimeError("repeated dataset builds produced different dataset IDs")
        if dataset_a["artifact_digest"] != dataset_b["artifact_digest"]:
            raise RuntimeError(
                "repeated dataset builds produced different artifact digests"
            )

        dataset = load_market_dataset_artifact(dataset_a_path)
        flat_observation_count = (
            ObservationBuilder(action_size=3, n_factors=0, finite_horizon=True)
            .layout(dataset)
            .size
        )
        sequence_payload = SequenceObservationBuilder().schema_payload(dataset)
        raw_windows = sequence_payload.get("windows")
        if not isinstance(raw_windows, (tuple, list)):
            raise RuntimeError("sequence schema windows must be ordered")
        sequence_observation_count = sum(
            dataset.n_symbols
            * int(dict(window)["length"])
            * len(tuple(dict(window)["feature_names"]))
            * 3
            for window in raw_windows
        )
        policy_observation_count = flat_observation_count + sequence_observation_count
        if policy_observation_count != pipeline._EXPECTED_POLICY_OBSERVATIONS:
            raise RuntimeError(
                f"expected {pipeline._EXPECTED_POLICY_OBSERVATIONS:,} policy observations, "
                f"observed {policy_observation_count:,}"
            )

        workflow_config = pipeline.write_run_config(
            template_path=_EXAMPLE_DIR / "walk-forward-full.json",
            output_path=work_root / "walk-forward-full.json",
        )
        artifact_root = work_root / "artifacts"
        walk_forward = pipeline.run_cli(
            [
                *pipeline._WALK_FORWARD_RUN_COMMAND,
                "--config",
                str(workflow_config),
                "--dataset",
                str(dataset_a_path),
                "--output",
                str(artifact_root),
                "--run-id",
                "binance-multitimeframe-full-walk-forward",
            ],
            root=_ROOT,
            log_path=work_root / "walk-forward.log",
        )
        walk_forward_path = _absolute_artifact_path(
            walk_forward.get("artifact_path"), field="walk-forward artifact_path"
        )
        walk_forward_manifest = validate_walk_forward_run_directory(walk_forward_path)
        if walk_forward_manifest.dataset_id != dataset.dataset_id:
            raise ValueError("walk-forward manifest dataset identity mismatch")

        summary = _base_summary(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            resolution=resolution,
            metadata_report=metadata_report,
            dataset=dataset,
            flat_observation_count=flat_observation_count,
            sequence_observation_count=sequence_observation_count,
            policy_observation_count=policy_observation_count,
            walk_forward=walk_forward,
        )
        gate = pipeline.evaluate_walk_forward_research_gate(
            walk_forward_path, strict=True
        )
        sensitivity_passed, sensitivity = pipeline.execution_sensitivity_gate(
            walk_forward_path
        )
        summary["execution_sensitivity"] = sensitivity
        summary["research_gate"] = asdict(gate)
        if not gate.passed or not sensitivity_passed:
            return ResearchPhaseOutcome(
                status=FullResearchStatus.COMPLETE_NO_GO,
                summary=summary,
            )

        selected_name, selected_seeds, selected_path = (
            pipeline.selected_walk_forward_recipe(
                walk_forward_path,
                workflow_config,
                work_root / "training-selected.json",
            )
        )
        selected_config = _normalize_selected_config(selected_path)
        sensitivity_digest = pipeline.load_json(
            walk_forward_path / "walk-forward.json"
        ).get("execution_sensitivity_digest")
        if not isinstance(sensitivity_digest, str) or len(sensitivity_digest) != 64:
            raise ValueError(
                "strict full research requires execution sensitivity evidence"
            )
        gate_evidence = {
            "execution_sensitivity": sensitivity,
            "research_gate": asdict(gate),
            "schema_version": "selection_gate_evidence_v1",
        }
        pipeline.write_json(work_root / "selection-gate-evidence.json", gate_evidence)
        if selected_config.git_commit is None:
            raise ValueError("selected training config lacks git commit provenance")
        proposal = SelectionProposal.create(
            walk_forward_run_digest=walk_forward_manifest.digest,
            gate_evidence_digest=content_digest(gate_evidence),
            execution_sensitivity_digest=sensitivity_digest,
            dataset_id=dataset.dataset_id,
            selected_configuration=selected_name,
            candidate_config_digest=content_digest(
                selected_config.candidate_digest_payload()
            ),
            seeds=selected_seeds,
            git_commit=selected_config.git_commit,
            dependency_digest=_lockfile_digest(),
            resume_checkpoint_digests=(),
        )
        proposal_path = write_selection_proposal(
            work_root / "selection-proposal.json", proposal
        )
        summary.update(
            {
                "selected_training_configuration": selected_name,
                "selected_training_seeds": list(selected_seeds),
                "selected_training_config_path": str(selected_path),
                "selection_gate_evidence_digest": proposal.gate_evidence_digest,
                "selection_proposal_digest": proposal.digest,
                "selection_proposal_path": str(proposal_path),
                "walk_forward_artifact_path": str(walk_forward_path),
                "walk_forward_run_digest": walk_forward_manifest.digest,
            }
        )
        return ResearchPhaseOutcome(
            status=FullResearchStatus.AWAITING_SELECTION_AUTHORIZATION,
            summary=summary,
        )

    def _train_selected(self, work_root: Path) -> ResearchPhaseOutcome:
        summary = pipeline.load_json(work_root / "summary.json")
        if (
            summary.get("status")
            != FullResearchStatus.AWAITING_SELECTION_AUTHORIZATION.value
        ):
            raise ValueError("generation is not awaiting selection authorization")
        proposal_path = work_root / "selection-proposal.json"
        proposal = load_selection_proposal(proposal_path)
        authorization_path = _required_path(
            self.args.selection_authorization, field="selection authorization"
        )
        public_keys_path = _required_path(
            self.args.selection_public_keys, field="selection public-key store"
        )
        authorization = load_selection_authorization(authorization_path)
        authorization.verify(
            proposal,
            trusted_keys=load_public_verification_keys(public_keys_path),
            trusted_at=self.args.trusted_now or datetime.now(UTC),
        )
        selected_config_path = _required_path(
            str(summary.get("selected_training_config_path", "")),
            field="selected training config",
        )
        dataset_path = work_root / "dataset-a"
        artifact_root = work_root / "artifacts"
        training = pipeline.run_cli(
            [
                *pipeline._TRAIN_RUN_COMMAND,
                "--config",
                str(selected_config_path),
                "--dataset",
                str(dataset_path),
                "--output",
                str(artifact_root),
                "--run-id",
                "binance-multitimeframe-selected-training",
                "--selection-proposal",
                str(proposal_path),
                "--selection-authorization",
                str(authorization_path),
                "--selection-public-keys",
                str(public_keys_path),
                "--require-selection-authorization",
            ],
            root=_ROOT,
            log_path=work_root / "training.log",
        )
        training_path = _absolute_artifact_path(
            training.get("artifact_path"), field="training artifact_path"
        )
        training_manifest = validate_training_run_directory(training_path)
        if training_manifest.run_kind != "research_selected_final":
            raise RuntimeError(
                "final training did not retain selected-final authorization"
            )
        summary.update(
            {
                "confirmation_required_from": training_manifest.completed_at.isoformat(),
                "selection_authorization_digest": authorization.authorization_digest,
                "selection_authorization_path": str(authorization_path),
                "training": training,
                "training_artifact_path": str(training_path),
                "training_run_digest": training_manifest.digest,
            }
        )
        return ResearchPhaseOutcome(
            status=FullResearchStatus.AWAITING_FRESH_CONFIRMATION,
            summary=summary,
        )

    def _finalize(self, work_root: Path) -> ResearchPhaseOutcome:
        summary = pipeline.load_json(work_root / "summary.json")
        if (
            summary.get("status")
            != FullResearchStatus.AWAITING_FRESH_CONFIRMATION.value
        ):
            raise ValueError("generation is not awaiting fresh confirmation")
        confirmation_path = _required_path(
            self.args.confirmation, field="fresh confirmation evidence"
        )
        confirmation_keys_path = _required_path(
            self.args.confirmation_public_keys,
            field="confirmation public-key store",
        )
        trusted_now = self.args.trusted_now
        if trusted_now is None:
            raise ValueError("--trusted-now is required for finalization")
        retained_confirmation = work_root / "confirmation-evidence.json"
        if retained_confirmation.exists():
            if retained_confirmation.read_bytes() != confirmation_path.read_bytes():
                raise FileExistsError("retained confirmation evidence differs")
        else:
            shutil.copy2(confirmation_path, retained_confirmation)

        training_path = _required_path(
            str(summary.get("training_artifact_path", "")),
            field="training artifact",
        )
        training_manifest = validate_training_run_directory(training_path)
        ensemble = pipeline.load_json(training_path / "ensemble.json")
        walk_forward_path = _required_path(
            str(summary.get("walk_forward_artifact_path", "")),
            field="walk-forward artifact",
        )
        gate_exit = pipeline.finalize_research_run(
            work_root=work_root,
            walk_forward_path=walk_forward_path,
            summary=summary,
            strict=True,
            require_confirmation=True,
            expected_policy_digest=training_manifest.ensemble_digest,
            expected_dataset_id=training_manifest.dataset_id,
            expected_environment_digest=str(ensemble["environment_digest"]),
            expected_training_run_digest=training_manifest.digest,
            expected_required_after=training_manifest.completed_at,
            trusted_now=trusted_now,
            trusted_confirmation_keys=load_public_verification_keys(
                confirmation_keys_path
            ),
        )
        if gate_exit != 0:
            return ResearchPhaseOutcome(
                status=FullResearchStatus.COMPLETE_NO_GO,
                summary=summary,
            )
        summary["confirmation_evidence_path"] = str(retained_confirmation)
        return ResearchPhaseOutcome(
            status=FullResearchStatus.AWAITING_RELEASE_APPROVAL,
            summary=summary,
        )


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone")
    return parsed.astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=tuple(phase.value for phase in ResearchPhase),
        default=ResearchPhase.DEVELOP.value,
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("var/cache/binance-vision")
    )
    parser.add_argument(
        "--metadata-mode",
        choices=tuple(mode.value for mode in BinanceMetadataMode),
        default=os.environ.get(
            "TRADE_RL_METADATA_MODE", BinanceMetadataMode.FROZEN_SNAPSHOT.value
        ),
    )
    parser.add_argument("--conservative-static-path", type=Path)
    parser.add_argument("--selection-authorization")
    parser.add_argument("--selection-public-keys")
    parser.add_argument("--confirmation")
    parser.add_argument("--confirmation-public-keys")
    parser.add_argument("--trusted-now", type=_parse_datetime)
    args = parser.parse_args(argv)
    work_root = (
        args.work_root if args.work_root.is_absolute() else _ROOT / args.work_root
    )
    result = run_research_phase(
        phase=ResearchPhase(args.phase),
        work_root=work_root,
        stages=BinanceFullResearchStages(args),
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
