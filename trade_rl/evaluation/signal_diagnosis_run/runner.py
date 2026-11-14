"""Single-run orchestration of the signal diagnosis (Issue #1146, card SD2).

``run_signal_diagnosis`` applies ``signal_diagnosis_protocol_v4`` to one admitted
``SignalPanel`` and classifies the saved cell evidence with the merged verdict code.
It refuses, before any statistic is computed, a real run without a G4 authorization
comment URL, from a dirty Git working tree, on a protocol that is not the frozen digest,
with a non-default plan, or on a panel outside the protocol window or roster. Every
seed comes from the protocol; the output holds no clock value and is a pure function of
the panel, the protocol and the code.

The protocol assigns D0, D1, D2 and part of D5 to card SD2 and D3, D4 to card SD3, but
its decisive cells are family-by-model cells that need the prequential fits, and its
budget allows one run. This runner therefore performs all layers in that single run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.signal_diagnosis.protocol import (
    SCOPE_REAL,
    SCOPE_SYNTHETIC,
    SignalDiagnosisProtocol,
    SignalDiagnosisProtocolError,
    SignalDiagnosisRunRequest,
    admit_run,
    protocol_digest,
)
from trade_rl.evaluation.signal_diagnosis.verdict import (
    DecisionCellEvidence,
    ValidityFlags,
    bh_adjusted_p_values,
    classify_signal_diagnosis,
    decision_cell_ids,
)
from trade_rl.evaluation.signal_diagnosis_run.fitting import (
    RunPlan,
    build_labels,
    build_vintages,
    drift_forecast,
    forecast_family,
)
from trade_rl.evaluation.signal_diagnosis_run.leakage import (
    BuilderProbe,
    label_copy_screen,
    maintained_builder_probe,
)
from trade_rl.evaluation.signal_diagnosis_run.measure import (
    CellMeasurement,
    RowSet,
    d0_profile,
    d1_beta,
    d2_features,
    d4_record,
    decision_rows,
    measure_cell,
    regime_signal,
)
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    SignalPanel,
    admit_panel,
    parse_instant_ns,
)

FROZEN_PROTOCOL_DIGEST: Final = (
    "1f48b065b03bc38299a751bef89456b57d5afc6a7d8b917668b1b461f852d0b3"
)
G4_AUTHORIZATION_URL_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/(issues|pull)/[0-9]+#issuecomment-[0-9]+$"
)
_AUTHORIZATION_DOMAIN: Final = "signal_diagnosis_g4_authorization_v1"
_BETA_MIN_ROWS: Final = 720
_PROFILE_MIN_TRAIN_ROWS: Final = 720


class DirtyWorkingTreeError(SignalDiagnosisProtocolError):
    """The working tree has uncommitted or untracked files: refuse to run."""


def validate_authorization_url(url: object) -> str:
    """A well-formed owner G4 comment URL; the runner cannot authenticate its author."""

    if not isinstance(url, str) or G4_AUTHORIZATION_URL_PATTERN.fullmatch(url) is None:
        raise SignalDiagnosisProtocolError(
            "a G4 authorization is required: pass the owner's issue or pull-request "
            "comment URL (https://github.com/shuntatsu/trade_rl/issues/<n>#issuecomment-<id>); "
            "the runner cannot authenticate its author"
        )
    return url


def authorization_digest(url: str, digest_of_protocol: str) -> str:
    """The lowercase SHA-256 the protocol's run request asks for, bound to the URL."""

    text = f"{_AUTHORIZATION_DOMAIN}\n{validate_authorization_url(url)}\n{digest_of_protocol}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------- code identity


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """The code a result was produced by (Git HEAD, ``trade_rl/`` tree, dirty flag)."""

    git_head: str
    trade_rl_tree: str
    dirty: bool
    module_digest: str
    module_count: int

    def require_clean(self) -> None:
        if self.dirty:
            raise DirtyWorkingTreeError(
                "refusing to run from a dirty working tree: commit or remove every "
                "change and untracked file so the result names the exact code"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "git_head": self.git_head,
            "trade_rl_tree": self.trade_rl_tree,
            "dirty": self.dirty,
            "module_digest": self.module_digest,
            "module_count": self.module_count,
        }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SignalDiagnosisProtocolError(
            f"cannot read the Git state of {root}: run from a clean Git checkout"
        ) from error
    return done.stdout


def loaded_module_digest(
    root: Path, modules: Mapping[str, Any] | None = None
) -> tuple[str, int]:
    """SHA-256 over the loaded ``trade_rl`` module files; each must lie under ``root``."""

    loaded = sys.modules if modules is None else modules
    entries: list[list[str]] = []
    resolved = root.resolve()
    for name in sorted(loaded):
        if name != "trade_rl" and not name.startswith("trade_rl."):
            continue
        file = getattr(loaded[name], "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if not path.is_relative_to(resolved):
            raise SignalDiagnosisProtocolError(
                f"module {name} is loaded from {path}, outside the checkout {resolved}"
            )
        entries.append([name, hashlib.sha256(path.read_bytes()).hexdigest()])
    if not entries:
        raise SignalDiagnosisProtocolError("no trade_rl module is loaded")
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest(), len(entries)


def capture_code_provenance(
    root: Path | None = None, modules: Mapping[str, Any] | None = None
) -> CodeProvenance:
    base = repository_root() if root is None else root
    head = _git(base, "rev-parse", "HEAD").strip()
    tree = _git(base, "rev-parse", "HEAD:trade_rl").strip()
    dirty = bool(_git(base, "status", "--porcelain", "--untracked-files=all").strip())
    digest, count = loaded_module_digest(base, modules)
    return CodeProvenance(head, tree, dirty, digest, count)


# ------------------------------------------------------------------- the run


@dataclass(frozen=True, slots=True)
class SignalDiagnosisRun:
    """Every file of one run: JSON-able records by relative path plus forecast arrays."""

    label: str | None
    records: dict[str, Any]
    forecasts: dict[str, np.ndarray]


def _instant(text: str) -> dt.datetime:
    return dt.datetime.fromtimestamp(parse_instant_ns(text) / 1e9, dt.UTC)


def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to ``None``."""

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        out["lightgbm"] = importlib.metadata.version("lightgbm")
    except importlib.metadata.PackageNotFoundError:
        out["lightgbm"] = None
    return out


def _available_at_check(panel: SignalPanel) -> dict[str, Any]:
    """Cells whose ``available_at`` is later than the bar while a feature is flagged
    available are a dataset contradiction (a violation); cells the dataset itself marks
    unavailable are an ordinary absence and are never used."""

    late = panel.available_at_ns > panel.timestamps_ns[:, None]
    declared = panel.feature_available.any(axis=2)
    return {
        "late_cells": int(late.sum()),
        "late_cells_with_available_features": int((late & declared).sum()),
        "violation": bool((late & declared).any()),
    }


def _validity_only(
    protocol: SignalDiagnosisProtocol, flags: ValidityFlags, record: dict[str, Any]
) -> SignalDiagnosisRun:
    result = classify_signal_diagnosis(protocol, validity=flags, cells=())
    verdict = {
        "verdict": result.verdict,
        "reasons": list(result.reasons),
        "claim_ceiling": result.claim_ceiling,
        "validity_flags_raised": list(flags.raised()),
    }
    return SignalDiagnosisRun(
        result.verdict, {"run_record.json": record, "verdict.json": verdict}, {}
    )


def _precursor_cells(
    cells: list[DecisionCellEvidence], protocol: SignalDiagnosisProtocol
) -> list[str]:
    """Cells that clear both BH gates and the year rule: the only ones D4 evaluates."""

    q = protocol.multiple_testing.bh_fdr_q
    raw = bh_adjusted_p_values([c.rank_ic_p_value for c in cells])
    dem = bh_adjusted_p_values([c.demeaned_p_value for c in cells])
    floor = protocol.stability.min_covered_months_per_full_year
    need = protocol.decision_rules.min_full_years_same_sign
    out = []
    for cell, a, b in zip(cells, raw, dem, strict=True):
        years = sum(
            1
            for y in cell.year_evidence
            if y.covered_months >= floor and y.increment_positive
        )
        if (
            cell.rank_ic_vs_drift > 0.0
            and a <= q
            and cell.demeaned_rank_ic > 0.0
            and b <= q
            and years >= need
        ):
            out.append(cell.cell_id)
    return out


def run_signal_diagnosis(
    panel: SignalPanel,
    protocol: SignalDiagnosisProtocol,
    *,
    scope: str,
    code_provenance: CodeProvenance,
    g4_authorization_url: str | None = None,
    expected_protocol_digest: str = FROZEN_PROTOCOL_DIGEST,
    plan: RunPlan | None = None,
    builder_probe: BuilderProbe | None = None,
    progress: Callable[[str], None] | None = None,
) -> SignalDiagnosisRun:
    """One diagnosis. See the module docstring for what is refused before computing."""

    say = progress or (lambda message: None)
    digest = protocol_digest(protocol)
    default_plan = RunPlan.from_protocol(protocol)
    plan = default_plan if plan is None else plan
    authorization: str | None = None
    if scope == SCOPE_REAL:
        url = validate_authorization_url(g4_authorization_url)
        code_provenance.require_clean()
        if plan != default_plan:
            raise SignalDiagnosisProtocolError(
                "a real run uses the frozen plan; only a synthetic run may differ"
            )
        authorization = authorization_digest(url, digest)
    elif scope == SCOPE_SYNTHETIC:
        if g4_authorization_url is not None:
            raise SignalDiagnosisProtocolError("a synthetic run carries no G4 URL")
        if not panel.identity.mismatches(protocol) == (
            "dataset_id",
            "artifact_digest",
            "arrays_digest",
            "feature_config_digest",
        ):
            raise SignalDiagnosisProtocolError(
                "a panel with any development Dataset identity is a real run"
            )
    else:
        raise SignalDiagnosisProtocolError(
            "scope must be synthetic_only or real_development"
        )

    record: dict[str, Any] = {
        "schema": "signal_diagnosis_run_record_v1",
        "protocol_schema": protocol.schema_version,
        "protocol_digest": digest,
        "data_scope": scope,
        "claim_ceiling": protocol.claim_ceiling,
        "g4_authorization_url": g4_authorization_url,
        "g4_authorization_digest": authorization,
        "code": code_provenance.to_mapping(),
        "versions": _versions(),
        "plan": {
            "refit_interval_days": plan.refit_interval_days,
            "reported_layers": plan.reported_layers,
        },
        "panel_identity": {
            "dataset_id": panel.identity.dataset_id,
            "artifact_digest": panel.identity.artifact_digest,
            "arrays_digest": panel.identity.arrays_digest,
            "feature_config_digest": panel.identity.feature_config_digest,
        },
    }
    mismatches = panel.identity.mismatches(protocol) if scope == SCOPE_REAL else ()
    digest_mismatch = digest != expected_protocol_digest
    if digest_mismatch or mismatches:
        flags = ValidityFlags(
            leakage_detected=False,
            dataset_identity_mismatch=bool(mismatches),
            protocol_digest_mismatch=digest_mismatch,
            available_at_violation=False,
            fs1_or_warmup_contact=False,
        )
        record["identity_mismatches"] = list(mismatches)
        return _validity_only(protocol, flags, record)

    request = SignalDiagnosisRunRequest(
        data_scope=scope,
        protocol_digest=digest,
        evaluation_start=None,
        evaluation_end=None,
        real_scope_authorization_digest=None,
    )
    if scope == SCOPE_REAL:
        request = SignalDiagnosisRunRequest(
            data_scope=scope,
            protocol_digest=digest,
            evaluation_start=_instant(protocol.data.evaluation_start),
            evaluation_end=_instant(protocol.data.evaluation_end),
            real_scope_authorization_digest=authorization,
        )
    admit_run(request, protocol)
    admit_panel(panel, protocol)

    horizons = (protocol.horizons.primary_bars, *protocol.horizons.secondary_bars)
    primary = protocol.horizons.primary_bars
    labels_by_h = {h: build_labels(panel, h) for h in horizons}
    labels = labels_by_h[primary]
    vintages = build_vintages(panel, protocol, plan)
    stamps = panel.timestamps_ns
    drift = drift_forecast(panel, labels, vintages)
    regime = regime_signal(panel.close, protocol.stability.regime_lookback_days)
    clip = next(t.clip_abs for t in protocol.transforms if t.standardise)

    say("D0: availability, leakage screens, non-stationarity profile")
    available_at = _available_at_check(panel)
    probe = (builder_probe or maintained_builder_probe)(protocol)
    copies = label_copy_screen(panel, labels_by_h)
    record["d0_available_at"] = available_at
    record["d0_leakage"] = {"builder_probe": probe, "label_copy": copies}
    record["vintage_cutoffs_ns"] = [v.cutoff_ns for v in vintages]
    flags = ValidityFlags(
        leakage_detected=bool(probe["leaks"] or copies["leaks"]),
        dataset_identity_mismatch=False,
        protocol_digest_mismatch=False,
        available_at_violation=bool(available_at["violation"]),
        fs1_or_warmup_contact=False,
    )
    if flags.raised():
        # A validity flag makes every result unusable: nothing is fitted or saved but
        # the evidence of the flag, so no leaky cell can be read by mistake.
        record["status"] = "invalid_before_fits"
        return _validity_only(protocol, flags, record)
    record["panel_rows"] = panel.n_rows
    record["models_secondary_horizons"] = "not_run_reported_horizons_cover_d2_only"
    records: dict[str, Any] = {}
    if plan.reported_layers:
        records["d0_profile.json"] = d0_profile(
            panel, clip_abs=clip, min_train_rows=_PROFILE_MIN_TRAIN_ROWS
        )
        records["d1_baselines.json"] = d1_beta(
            labels, vintages, panel.symbols, stamps, _BETA_MIN_ROWS
        )
        say("D2: feature-by-horizon-by-transform rank and linear IC")
        records["d2_features.json"] = d2_features(
            panel, protocol, labels_by_h, vintages
        )
    record["reported_layers_run"] = plan.reported_layers

    say("D3/D5: prequential fits of the decision cells")
    measured: dict[str, CellMeasurement] = {}
    rows_by_cell: dict[str, RowSet] = {}
    forecasts: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    by_family = {family.family_id: family for family in protocol.families}
    for family in protocol.families:
        try:
            fitted = forecast_family(
                panel,
                protocol,
                by_family[family.family_id],
                labels,
                vintages,
                progress=None,
            )
            for model_id, fit in fitted.items():
                cell_id = f"{family.family_id}:{model_id}"
                rows = decision_rows(fit.forecast, drift, labels, stamps)
                if rows.day.size == 0:
                    raise SignalDiagnosisProtocolError("the cell has no decision rows")
                measurement = measure_cell(
                    family.family_id,
                    model_id,
                    rows,
                    protocol,
                    close_symbols=panel.symbols,
                    regime=regime,
                )
                measurement.record["vintages"] = [
                    {
                        "cutoff_ns": v.cutoff_ns,
                        "n_train": v.n_train,
                        "n_decided": v.n_decided,
                        "ridge_penalty": v.ridge_penalty,
                        "ridge_cv_scores": v.ridge_cv_scores,
                    }
                    for v in fit.vintages
                ]
                measured[cell_id] = measurement
                rows_by_cell[cell_id] = rows
                forecasts[cell_id.replace(":", "__")] = fit.forecast.astype(np.float32)
        except Exception as error:  # a failing cell is recorded, never dropped
            for model_id in protocol.models.model_ids:
                failures[f"{family.family_id}:{model_id}"] = (
                    f"{type(error).__name__}: {error}"
                )
            measured = {
                k: v
                for k, v in measured.items()
                if not k.startswith(f"{family.family_id}:")
            }
        say(f"{family.family_id}: done")

    cell_files: dict[str, Any] = {}
    for cell_id in decision_cell_ids(protocol):
        name = f"cells/{cell_id.replace(':', '__')}.json"
        if cell_id in failures:
            cell_files[name] = {
                "cell_id": cell_id,
                "status": "failed",
                "error": failures[cell_id],
            }
        else:
            cell_files[name] = measured[cell_id].record
    records.update(cell_files)
    if failures:
        records["run_record.json"] = {
            **record,
            "status": "halted_cell_failure",
            "failed_cells": failures,
        }
        records["verdict.json"] = {
            "verdict": None,
            "reasons": ["a decision cell failed; no verdict is produced"],
        }
        return SignalDiagnosisRun(None, _clean(records), forecasts)

    ordered = [measured[cell_id] for cell_id in decision_cell_ids(protocol)]
    evidence = [m.evidence for m in ordered]
    precursors = set(_precursor_cells(evidence, protocol))
    final: list[DecisionCellEvidence] = []
    for item in ordered:
        cell = item.evidence
        if cell.cell_id in precursors:
            say(f"D4: {cell.cell_id}")
            d4 = d4_record(rows_by_cell[cell.cell_id], protocol)
            item.record["d4"] = d4
            decisive = d4["decisive"]
            cell = replace(
                cell,
                drift_excess_ci_lower_cost_x1=decisive["ci_lower_cost_x1"],
                drift_excess_ci_upper_cost_x1=decisive["ci_upper_cost_x1"],
                drift_excess_point_cost_x2=decisive["point_cost_x2"],
            )
        final.append(cell)
    for cell_id in decision_cell_ids(protocol):
        records[f"cells/{cell_id.replace(':', '__')}.json"] = measured[cell_id].record
    result = classify_signal_diagnosis(protocol, validity=flags, cells=final)
    q = protocol.multiple_testing.bh_fdr_q
    raw_adj = bh_adjusted_p_values([c.rank_ic_p_value for c in final])
    dem_adj = bh_adjusted_p_values([c.demeaned_p_value for c in final])
    records["verdict.json"] = {
        "verdict": result.verdict,
        "signal_cell_ids": list(result.signal_cell_ids),
        "candidate_cell_ids": list(result.candidate_cell_ids),
        "reasons": list(result.reasons),
        "claim_ceiling": result.claim_ceiling,
        "bh_fdr_q": q,
        "validity_flags_raised": list(flags.raised()),
        "d4_evaluated_cells": sorted(precursors),
        "cells": [
            {
                "cell_id": c.cell_id,
                "rank_ic_vs_drift": c.rank_ic_vs_drift,
                "rank_ic_p_value": c.rank_ic_p_value,
                "rank_ic_p_bh_adjusted": float(a),
                "demeaned_rank_ic": c.demeaned_rank_ic,
                "demeaned_p_value": c.demeaned_p_value,
                "demeaned_p_bh_adjusted": float(b),
                "raw_n_days": c.p_value_provenance.raw_n_days,
                "demeaned_n_days": c.p_value_provenance.demeaned_n_days,
            }
            for c, a, b in zip(final, raw_adj, dem_adj, strict=True)
        ],
    }
    records["run_record.json"] = {**record, "status": "completed"}
    return SignalDiagnosisRun(result.verdict, _clean(records), forecasts)


def _json_bytes(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
    return (text + chr(10)).encode("utf-8")


def write_outputs(output_dir: Path, run: SignalDiagnosisRun) -> list[Path]:
    """Write every file of the run into a new, empty directory; never overwrite."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite")
    written: list[Path] = []
    for name in sorted(run.records):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(_json_bytes(run.records[name]))
        written.append(path)
    if run.forecasts:
        path = output_dir / "forecasts.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **run.forecasts)
        written.append(path)
    return written


__all__ = [
    "FROZEN_PROTOCOL_DIGEST",
    "CodeProvenance",
    "DirtyWorkingTreeError",
    "SignalDiagnosisRun",
    "authorization_digest",
    "capture_code_provenance",
    "run_signal_diagnosis",
    "validate_authorization_url",
    "write_outputs",
]
