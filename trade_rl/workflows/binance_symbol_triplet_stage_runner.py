"""Run one Binance USDS-M symbol-triplet stage from PostgreSQL evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from trade_rl.data import publish_market_dataset_artifact
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.postgres_indicator_artifacts import (
    IndicatorArtifactConnection,
)
from trade_rl.integrations.postgres_market_dataset import (
    build_postgres_market_dataset,
)
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    SymbolTripletStageRequest,
    build_symbol_triplet_stage_request,
    load_symbol_triplet_stage_completion,
)
from trade_rl.workflows.symbol_triplet_stage_training import (
    SymbolTripletStageTrainingResult,
    build_symbol_triplet_stage_dataset_binding,
    execute_symbol_triplet_stage_training,
    load_symbol_triplet_stage_dataset_binding,
    write_symbol_triplet_stage_dataset_binding,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    SymbolTripletTrainingPlan,
    SymbolTripletTrainingStage,
    current_symbol_triplet_training_stage,
    load_symbol_triplet_training_cursor,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    normalize_training_run_config,
)

_PROVENANCE_SCHEMA = "binance_symbol_triplet_stage_provenance_v1"


def _training_seeds(path: str | Path) -> tuple[int, ...]:
    config = normalize_training_run_config(TrainingRunConfig.from_json(Path(path)))
    return tuple(config.training.seeds)


def binance_symbol_triplet_stage_root(
    work_root: str | Path,
    stage: SymbolTripletStageRequest | SymbolTripletTrainingStage,
) -> Path:
    """Return the deterministic immutable root for one Plan stage."""

    root = Path(work_root).resolve()
    return root / "stages" / f"stage-{stage.stage_index:04d}-{stage.stage_id[:16]}"


def _previous_completion_path(
    plan: SymbolTripletTrainingPlan,
    work_root: str | Path,
    *,
    stage_index: int,
) -> Path | None:
    if stage_index == 0:
        return None
    previous_stage = plan.stages[stage_index - 1]
    return binance_symbol_triplet_stage_root(work_root, previous_stage) / "completion.json"


def current_binance_symbol_triplet_stage_request(
    plan: SymbolTripletTrainingPlan,
    cursor_path: str | Path,
    base_config_path: str | Path,
    work_root: str | Path,
) -> SymbolTripletStageRequest | None:
    """Resolve the current request and only the immediately prior completion."""

    cursor = load_symbol_triplet_training_cursor(cursor_path, plan=plan)
    stage = current_symbol_triplet_training_stage(plan, cursor)
    if stage is None:
        return None
    previous_path = _previous_completion_path(
        plan,
        work_root,
        stage_index=stage.stage_index,
    )
    previous_completion = (
        None
        if previous_path is None
        else load_symbol_triplet_stage_completion(previous_path, plan=plan)
    )
    return build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_training_seeds(base_config_path),
        previous_completion=previous_completion,
    )


def _stage_provenance(request: SymbolTripletStageRequest) -> dict[str, object]:
    return {
        "cycle_index": request.cycle_index,
        "plan_digest": request.plan_digest,
        "request_digest": request.digest,
        "schema_version": _PROVENANCE_SCHEMA,
        "selected_symbols": request.symbols,
        "slot_symbols": request.slot_symbols,
        "stage_id": request.stage_id,
        "stage_index": request.stage_index,
        "train_split_slot": request.train_split_slot,
    }


def _prepare_bound_dataset(
    *,
    connection: IndicatorArtifactConnection,
    request: SymbolTripletStageRequest,
    dataset_path: Path,
    binding_path: Path,
    symbol_vocabulary: Sequence[str],
    start_time: datetime,
    end_time: datetime,
    metadata: Mapping[str, Mapping[str, object]],
    metadata_evidence_digest: str,
    execution_rule_histories: Mapping[str, Sequence[InstrumentExecutionRule]] | None,
) -> None:
    dataset_exists = dataset_path.exists()
    binding_exists = binding_path.exists()
    if dataset_exists:
        if not dataset_path.is_dir() or not binding_path.is_file():
            raise FileExistsError(
                "symbol-triplet stage dataset exists without an immutable binding"
            )
        load_symbol_triplet_stage_dataset_binding(
            binding_path,
            request=request,
            dataset_path=dataset_path,
        )
        return
    if binding_exists:
        raise FileExistsError(
            "symbol-triplet stage binding exists without its dataset artifact"
        )

    dataset = build_postgres_market_dataset(
        connection,
        symbols=request.symbols,
        symbol_vocabulary=symbol_vocabulary,
        slot_symbols=request.slot_symbols,
        start_time=start_time,
        end_time=end_time,
        metadata=metadata,
        metadata_evidence_digest=metadata_evidence_digest,
        execution_rule_histories=execution_rule_histories,
        symbol_triplet_provenance=_stage_provenance(request),
    )
    if dataset.symbols != request.slot_symbols:
        raise ValueError("PostgreSQL stage dataset slot symbols differ from the request")
    publish_market_dataset_artifact(dataset_path, dataset)
    binding = build_symbol_triplet_stage_dataset_binding(
        request,
        dataset_path=dataset_path,
        selected_symbols=request.symbols,
    )
    write_symbol_triplet_stage_dataset_binding(binding_path, binding)


def execute_binance_symbol_triplet_postgres_stage(
    *,
    connection: IndicatorArtifactConnection,
    plan: SymbolTripletTrainingPlan,
    cursor_path: str | Path,
    base_config_path: str | Path,
    work_root: str | Path,
    symbol_vocabulary: Sequence[str],
    start_time: datetime,
    end_time: datetime,
    metadata: Mapping[str, Mapping[str, object]],
    metadata_evidence_digest: str,
    execution_rule_histories: Mapping[str, Sequence[InstrumentExecutionRule]]
    | None = None,
) -> SymbolTripletStageTrainingResult | None:
    """Build/reuse the current PostgreSQL dataset and execute one training stage."""

    require_sha256(metadata_evidence_digest, field="metadata_evidence_digest")
    request = current_binance_symbol_triplet_stage_request(
        plan,
        cursor_path,
        base_config_path,
        work_root,
    )
    if request is None:
        return None

    stage_root = binance_symbol_triplet_stage_root(work_root, request)
    dataset_path = stage_root / "dataset"
    binding_path = stage_root / "dataset-binding.json"
    _prepare_bound_dataset(
        connection=connection,
        request=request,
        dataset_path=dataset_path,
        binding_path=binding_path,
        symbol_vocabulary=symbol_vocabulary,
        start_time=start_time,
        end_time=end_time,
        metadata=metadata,
        metadata_evidence_digest=metadata_evidence_digest,
        execution_rule_histories=execution_rule_histories,
    )
    previous_completion_path = _previous_completion_path(
        plan,
        work_root,
        stage_index=request.stage_index,
    )
    run_id = f"binance-triplet-stage-{request.stage_index:04d}-{request.stage_id[:16]}"
    return execute_symbol_triplet_stage_training(
        plan=plan,
        cursor_path=Path(cursor_path),
        previous_completion_path=previous_completion_path,
        dataset_path=dataset_path,
        dataset_binding_path=binding_path,
        base_config_path=Path(base_config_path),
        stage_config_path=stage_root / "training-config.json",
        store_root=stage_root / "artifacts",
        run_id=run_id,
        completion_path=stage_root / "completion.json",
    )


__all__ = [
    "binance_symbol_triplet_stage_root",
    "current_binance_symbol_triplet_stage_request",
    "execute_binance_symbol_triplet_postgres_stage",
]
