from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_between(path: str, start_marker: str, end_marker: str, replacement: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    target.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_between(
    "trade_rl/artifacts/signals.py",
    "@dataclass(frozen=True, slots=True)\nclass SignalArrays:",
    "\n\ndef _sha256",
    '''@dataclass(frozen=True, slots=True)
class SignalArrays:
    values: np.ndarray
    valid: np.ndarray
    available_at: np.ndarray
    knowledge_cutoff: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype[np.generic]:
        return self.values.dtype

    @property
    def valid_mask(self) -> np.ndarray:
        axes = tuple(range(1, self.valid.ndim))
        complete = self.valid if not axes else np.all(self.valid, axis=axes)
        return complete & (self.knowledge_cutoff >= 0)

    def __getitem__(self, item: object) -> np.ndarray:
        return self.values[item]

    def __array__(self, dtype: np.dtype[np.generic] | None = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


SignalArrayPayload = SignalArrays
''',
)
replace_between(
    "trade_rl/artifacts/signals.py",
    "def _validate_arrays(",
    "\n\ndef write_signal_artifact(",
    '''def _validate_arrays(
    *,
    kind: SignalKind,
    names: tuple[str, ...],
    values: np.ndarray,
    fit_stop: int,
    prediction_start: int,
    valid: np.ndarray | None,
    valid_mask: np.ndarray | None,
    available_at: np.ndarray | None,
    knowledge_cutoff: np.ndarray | None,
) -> SignalArrays:
    array = np.asarray(values)
    if array.ndim not in {2, 3} or not np.issubdtype(array.dtype, np.number):
        raise ValueError("signal values must be a numeric rank-2 or rank-3 array")
    if not np.isfinite(array).all():
        raise ValueError("signal values must be finite")
    if kind == "alpha" and (array.ndim != 2 or array.shape[1] != len(names)):
        raise ValueError("alpha values must have shape (bars, symbols)")
    if kind == "factor" and (array.ndim != 3 or array.shape[1] != len(names)):
        raise ValueError("factor values must have shape (bars, factors, symbols)")
    if valid is not None and valid_mask is not None:
        raise ValueError("provide only one of valid or valid_mask")
    if valid is not None:
        validity = np.asarray(valid, dtype=np.bool_)
        if validity.shape != array.shape:
            raise ValueError("signal validity shape must match values")
    elif valid_mask is not None:
        row_valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if row_valid.shape != (array.shape[0],):
            raise ValueError("signal valid_mask must match the bar count")
        shape = (array.shape[0],) + (1,) * (array.ndim - 1)
        validity = np.broadcast_to(row_valid.reshape(shape), array.shape).copy()
    else:
        validity = np.ones(array.shape, dtype=np.bool_)

    availability = (
        np.arange(array.shape[0], dtype=np.int64)
        if available_at is None
        else np.asarray(available_at)
    )
    if availability.shape != (array.shape[0],):
        raise ValueError("signal available_at must have one value per bar")
    if np.issubdtype(availability.dtype, np.datetime64):
        availability = availability.astype("datetime64[ns]")
        if np.any(availability.astype(np.int64) == np.iinfo(np.int64).min):
            raise ValueError("signal available_at must not contain NaT")
    elif np.issubdtype(availability.dtype, np.integer):
        availability = availability.astype(np.int64)
        if np.any(availability < 0):
            raise ValueError("signal available_at indices must be non-negative")
    else:
        raise ValueError("signal available_at must use datetime64 or integer indices")

    axes = tuple(range(1, validity.ndim))
    complete = validity if not axes else np.all(validity, axis=axes)
    if knowledge_cutoff is None:
        cutoff = np.full(array.shape[0], -1, dtype=np.int64)
        consumable = complete & (
            np.arange(array.shape[0], dtype=np.int64) >= prediction_start
        )
        cutoff[consumable] = fit_stop - 1
    else:
        cutoff = np.asarray(knowledge_cutoff, dtype=np.int64).reshape(-1)
    if cutoff.shape != (array.shape[0],):
        raise ValueError("signal knowledge_cutoff must match the bar count")
    indices = np.arange(array.shape[0], dtype=np.int64)
    if np.any(cutoff[~complete] != -1):
        raise ValueError("invalid signal rows must use knowledge cutoff -1")
    consumable = cutoff >= 0
    if np.any(cutoff[consumable] >= indices[consumable]):
        raise ValueError(
            "signal knowledge cutoff must strictly precede each prediction"
        )
    return SignalArrays(
        array.copy(),
        validity.copy(),
        availability.copy(),
        cutoff.copy(),
    )
''',
)
replace_between(
    "trade_rl/artifacts/signals.py",
    "def write_signal_artifact(",
    "\n\ndef load_signal_artifact(",
    '''def write_signal_artifact(
    root: str | Path,
    *,
    kind: SignalKind,
    dataset_id: str,
    fit_start: int,
    fit_stop: int,
    names: tuple[str, ...],
    values: np.ndarray,
    prediction_start: int | None = None,
    prediction_stop: int | None = None,
    generator_config_digest: str = _LEGACY_GENERATOR_CONFIG,
    generator_code_digest: str = _LEGACY_GENERATOR_CODE,
    valid: np.ndarray | None = None,
    available_at: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    knowledge_cutoff: np.ndarray | None = None,
    generator_digest: str | None = None,
) -> str:
    array = np.asarray(values)
    resolved_prediction_start = fit_stop if prediction_start is None else prediction_start
    resolved_prediction_stop = array.shape[0] if prediction_stop is None else prediction_stop
    if generator_digest is not None:
        require_sha256(generator_digest, field="generator_digest")
        if generator_config_digest == _LEGACY_GENERATOR_CONFIG:
            generator_config_digest = generator_digest
        if generator_code_digest == _LEGACY_GENERATOR_CODE:
            generator_code_digest = generator_digest
    arrays = _validate_arrays(
        kind=kind,
        names=names,
        values=array,
        fit_stop=fit_stop,
        prediction_start=resolved_prediction_start,
        valid=valid,
        valid_mask=valid_mask,
        available_at=available_at,
        knowledge_cutoff=knowledge_cutoff,
    )
    output = Path(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"signal artifact destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    payload = _deterministic_npz(
        {
            "available_at": arrays.available_at,
            "knowledge_cutoff": arrays.knowledge_cutoff,
            "valid": arrays.valid,
            "values": arrays.values,
        }
    )
    arrays_digest = _sha256(payload)
    base = {
        "arrays_digest": arrays_digest,
        "arrays_file": SIGNAL_ARRAYS_NAME,
        "available_at_dtype": arrays.available_at.dtype.str,
        "dataset_id": dataset_id,
        "dtype": arrays.values.dtype.str,
        "fit_start": fit_start,
        "fit_stop": fit_stop,
        "generator_code_digest": generator_code_digest,
        "generator_config_digest": generator_config_digest,
        "kind": kind,
        "names": names,
        "prediction_start": resolved_prediction_start,
        "prediction_stop": resolved_prediction_stop,
        "schema_version": SIGNAL_ARTIFACT_SCHEMA,
        "shape": tuple(int(size) for size in arrays.shape),
    }
    manifest = SignalArrayManifest(
        artifact_digest=content_digest(base),
        arrays_digest=arrays_digest,
        dataset_id=dataset_id,
        fit_start=fit_start,
        fit_stop=fit_stop,
        prediction_start=resolved_prediction_start,
        prediction_stop=resolved_prediction_stop,
        generator_config_digest=generator_config_digest,
        generator_code_digest=generator_code_digest,
        kind=kind,
        names=names,
        shape=tuple(int(size) for size in arrays.shape),
        dtype=arrays.values.dtype.str,
        available_at_dtype=arrays.available_at.dtype.str,
    )
    _atomic_write(output / SIGNAL_ARRAYS_NAME, payload)
    _atomic_write(output / SIGNAL_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest.artifact_digest
''',
)
replace_between(
    "trade_rl/artifacts/signals.py",
    "def load_signal_artifact(",
    "\n\n__all__ =",
    '''def load_signal_artifact(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, SignalArrays]:
    path = Path(root)
    _verify_file_closure(path)
    raw = json.loads((path / SIGNAL_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("signal manifest must be a mapping")
    try:
        manifest = SignalArrayManifest(
            artifact_digest=str(raw["artifact_digest"]),
            arrays_digest=str(raw["arrays_digest"]),
            dataset_id=str(raw["dataset_id"]),
            fit_start=int(raw["fit_start"]),
            fit_stop=int(raw["fit_stop"]),
            prediction_start=int(raw["prediction_start"]),
            prediction_stop=int(raw["prediction_stop"]),
            generator_config_digest=str(raw["generator_config_digest"]),
            generator_code_digest=str(raw["generator_code_digest"]),
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            names=tuple(str(value) for value in raw["names"]),
            shape=tuple(int(value) for value in raw["shape"]),
            dtype=str(raw["dtype"]),
            available_at_dtype=str(raw["available_at_dtype"]),
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("signal manifest is invalid") from error
    if content_digest(manifest.digest_payload()) != manifest.artifact_digest:
        raise ValueError("signal manifest digest mismatch")
    if expected_kind is not None and manifest.kind != expected_kind:
        raise ValueError("signal artifact kind mismatch")
    payload = (path / SIGNAL_ARRAYS_NAME).read_bytes()
    if _sha256(payload) != manifest.arrays_digest:
        raise ValueError("signal arrays digest mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "available_at",
                "knowledge_cutoff",
                "valid",
                "values",
            }:
                raise ValueError("signal array allow-list mismatch")
            arrays = SignalArrays(
                values=np.asarray(archive["values"]),
                valid=np.asarray(archive["valid"], dtype=np.bool_),
                available_at=np.asarray(archive["available_at"]),
                knowledge_cutoff=np.asarray(
                    archive["knowledge_cutoff"], dtype=np.int64
                ),
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("signal arrays are invalid") from error
    validated = _validate_arrays(
        kind=manifest.kind,
        names=manifest.names,
        values=arrays.values,
        fit_stop=manifest.fit_stop,
        prediction_start=manifest.prediction_start,
        valid=arrays.valid,
        valid_mask=None,
        available_at=arrays.available_at,
        knowledge_cutoff=arrays.knowledge_cutoff,
    )
    if validated.shape != manifest.shape or validated.dtype.str != manifest.dtype:
        raise ValueError("signal array shape or dtype mismatch")
    if validated.available_at.dtype.str != manifest.available_at_dtype:
        raise ValueError("signal availability dtype mismatch")
    return manifest, validated


def load_signal_artifact_payload(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, SignalArrays]:
    return load_signal_artifact(root, expected_kind=expected_kind)
''',
)
replace_once(
    "trade_rl/artifacts/signals.py",
    '''    "SignalArrayManifest",
    "SignalArrays",
    "load_signal_artifact",
''',
    '''    "SignalArrayManifest",
    "SignalArrayPayload",
    "SignalArrays",
    "load_signal_artifact",
    "load_signal_artifact_payload",
''',
)

replace_between(
    "trade_rl/integrations/signal_artifacts.py",
    "    @property\n    def minimum_index(self) -> int:\n        return max(0, self.manifest.prediction_start - self.offset)",
    "\n\n    @property\n    def dataset_id",
    '''    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.arrays.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
''',
)
replace_between(
    "trade_rl/integrations/signal_artifacts.py",
    "    @property\n    def minimum_index(self) -> int:\n        return max(0, self.manifest.prediction_start - self.offset)",
    "\n\n    @property\n    def dataset_id",
    '''    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.arrays.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
''',
)
replace_once(
    "trade_rl/integrations/signal_artifacts.py",
    "\n\ndef _decision_value(\n",
    '''\n\ndef _minimum_index(
    valid_mask: np.ndarray,
    *,
    offset: int,
    bound_bars: int | None,
) -> int:
    stop = valid_mask.shape[0] if bound_bars is None else offset + bound_bars
    available = np.flatnonzero(valid_mask[offset:stop])
    if available.size == 0:
        raise ValueError("signal artifact has no valid predictions in the bound range")
    return int(available[0])


def _decision_value(
''',
)
replace_between(
    "trade_rl/integrations/signal_artifacts.py",
    "def _validate_dataset_and_index(",
    "\n\ndef _validate_common(",
    '''def _validate_dataset_and_index(
    manifest: SignalArrayManifest,
    arrays: SignalArrays,
    dataset: MarketDataset,
    index: int,
    *,
    dataset_id: str,
    offset: int,
    bound_bars: int | None,
) -> int:
    if dataset.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    expected_bars = arrays.shape[0] if bound_bars is None else bound_bars
    if dataset.n_bars != expected_bars:
        raise ValueError("signal artifact bar count does not match dataset")
    if not 0 <= index < expected_bars:
        raise ValueError("signal artifact is unavailable at the requested index")
    source_index = index + offset
    if not np.all(arrays.valid[source_index]):
        if np.any(arrays.valid[source_index]):
            raise ValueError("signal prediction is invalid at the requested index")
        raise ValueError("signal artifact is unavailable at the requested index")
    if arrays.knowledge_cutoff[source_index] < 0:
        raise ValueError("signal artifact is unavailable at the requested index")
    decision = _decision_value(dataset, index, source_index, arrays.available_at)
    if arrays.available_at[source_index] > decision:
        raise ValueError("signal prediction is not available at the decision timestamp")
    if arrays.knowledge_cutoff[source_index] >= source_index:
        raise ValueError("signal artifact violates its point-in-time knowledge cutoff")
    return source_index
''',
)
replace_between(
    "trade_rl/integrations/signal_artifacts.py",
    "def _validate_common(",
    "\n\ndef load_alpha_artifact(",
    '''def _validate_common(
    manifest: SignalArrayManifest,
    arrays: SignalArrays,
    *,
    dataset_id: str,
    evaluation_start: int | None,
) -> None:
    if manifest.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    if evaluation_start is not None:
        if evaluation_start < 0 or evaluation_start >= arrays.shape[0]:
            raise ValueError("signal evaluation start is outside the artifact")
        if not np.any(arrays.valid_mask[evaluation_start:]):
            raise ValueError("signal artifact has no valid predictions for evaluation")
''',
)
replace_once(
    "trade_rl/integrations/signal_artifacts.py",
    "    _validate_common(manifest, dataset_id=dataset_id, evaluation_start=evaluation_start)\n",
    '''    _validate_common(
        manifest,
        arrays,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
''',
)
replace_once(
    "trade_rl/integrations/signal_artifacts.py",
    "    _validate_common(manifest, dataset_id=dataset_id, evaluation_start=evaluation_start)\n",
    '''    _validate_common(
        manifest,
        arrays,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
''',
)

replace_once(
    "trade_rl/workflows/walk_forward.py",
    '''            "schema_version": "walk_forward_execution_v1",
            "stitch_mode": config.stitch_mode.value,
''',
    '''            "baseline_diagnostics": baseline_stitched.diagnostics.digest_payload(),
            "schema_version": "walk_forward_execution_v2",
            "selected_diagnostics": selected_stitched.diagnostics.digest_payload(),
            "stitch_mode": config.stitch_mode.value,
''',
)
replace_once(
    "trade_rl/workflows/walk_forward.py",
    '''            None if independent else evaluate_performance(selected_stitched.returns)
''',
    '''            None
            if independent
            else evaluate_performance(
                selected_stitched.returns,
                turnover_total=selected_stitched.diagnostics.turnover_total,
                total_cost=selected_stitched.diagnostics.total_cost,
                funding_pnl=selected_stitched.diagnostics.funding_pnl,
                borrow_cost=selected_stitched.diagnostics.borrow_cost,
                n_trades=selected_stitched.diagnostics.n_trades,
                rebalance_events=selected_stitched.diagnostics.rebalance_events,
                termination_count=selected_stitched.diagnostics.termination_count,
            )
''',
)
replace_once(
    "trade_rl/workflows/walk_forward.py",
    '''            None if independent else evaluate_performance(baseline_stitched.returns)
''',
    '''            None
            if independent
            else evaluate_performance(
                baseline_stitched.returns,
                turnover_total=baseline_stitched.diagnostics.turnover_total,
                total_cost=baseline_stitched.diagnostics.total_cost,
                funding_pnl=baseline_stitched.diagnostics.funding_pnl,
                borrow_cost=baseline_stitched.diagnostics.borrow_cost,
                n_trades=baseline_stitched.diagnostics.n_trades,
                rebalance_events=baseline_stitched.diagnostics.rebalance_events,
                termination_count=baseline_stitched.diagnostics.termination_count,
            )
''',
)
replace_once(
    "trade_rl/workflows/market_walk_forward.py",
    '''        walk_forward_payload = {
            "baseline_metrics": asdict(result.baseline_metrics),
            "dataset_id": dataset.dataset_id,
            "evaluation_digest": result.evaluation_digest,
            "folds": tuple(folds_payload),
            "production_status": "NO-GO",
            "schema_version": "market_walk_forward_run_v1",
            "selected_metrics": asdict(result.selected_metrics),
        }
''',
    '''        walk_forward_payload = {
            "baseline_metrics": (
                None if result.baseline_metrics is None else asdict(result.baseline_metrics)
            ),
            "baseline_independent_summary": (
                None
                if result.baseline_independent_summary is None
                else asdict(result.baseline_independent_summary)
            ),
            "dataset_id": dataset.dataset_id,
            "evaluation_digest": result.evaluation_digest,
            "folds": tuple(folds_payload),
            "production_status": "NO-GO",
            "schema_version": "market_walk_forward_run_v2",
            "selected_metrics": (
                None if result.selected_metrics is None else asdict(result.selected_metrics)
            ),
            "selected_independent_summary": (
                None
                if result.selected_independent_summary is None
                else asdict(result.selected_independent_summary)
            ),
            "stitch_mode": config.workflow.stitch_mode.value,
        }
''',
)
replace_once(
    "tests/workflows/test_fold_runner.py",
    "    from trade_rl.evaluation.walk_forward.stitching import ExecutionEvidence\n",
    "    from trade_rl.evaluation.evidence import ExecutionDiagnostics\n",
)
replace_once(
    "tests/workflows/test_fold_runner.py",
    '''                evidence=ExecutionEvidence(
                    total_cost=4.0, turnover_total=0.5, n_trades=3
                ),
''',
    '''                diagnostics=ExecutionDiagnostics(
                    total_cost=4.0, turnover_total=0.5, n_trades=3
                ),
''',
)
replace_once(
    "tests/workflows/test_fold_runner.py",
    "    assert result.selected_oos.evidence.total_cost == 4.0\n",
    "    assert result.selected_oos.diagnostics.total_cost == 4.0\n",
)
replace_once(
    "tests/e2e/test_research_to_serving_v2.py",
    '''                "reward": {"scale": 1.0, "baseline_window_hours": 4.0},
''',
    '''                "reward": {
                    "scale": 1.0,
                    "baseline_window_hours": 4.0,
                    "baseline_minimum_history_hours": 1.0,
                },
''',
)
