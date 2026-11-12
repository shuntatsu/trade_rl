"""Orchestration of the multi-symbol carry development screen.

The runner is data-source agnostic: it consumes a ``ScreenData`` object, so tests
drive it with synthetic rows and the command line binds checksum-verified Vision
archives or a populated database. It owns no I/O except the output directory. It
refuses any window except the registered one, refuses to run without a G4
authorization comment URL, and writes every cell, passing or not, before it writes
the decision.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.carry import replay_carry
from trade_rl.evaluation.carry_multisymbol import protocol
from trade_rl.evaluation.carry_multisymbol.decision import (
    INCONCLUSIVE,
    CellSummary,
    decide,
)
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryScreenHalt,
    assemble_carry_dataset,
    assemble_multisymbol_carry_dataset,
)

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ScreenData(Protocol):
    """What the screen needs from a data source; coverage failures are exclusions."""

    @property
    def name(self) -> str: ...

    def lookback_median_volumes(self, symbol: str) -> Mapping[str, float]:
        """Median hourly quote turnover per market over the pre-window lookback."""
        ...

    def pair(self, symbol: str) -> tuple[RawMarketSeries, RawMarketSeries]:
        """The (spot, perpetual) series over the registered window."""
        ...

    def evidence(self) -> list[dict[str, object]]:
        """Per-archive verification evidence for the data manifest, if any."""
        ...


@dataclass(frozen=True, slots=True)
class ScreenRequest:
    start: datetime
    end: datetime
    g4_authorization_url: str
    output_dir: Path
    code_commit: str
    coverage_only: bool = False


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    symbols: tuple[str, ...]
    capital: float
    legacy_contract: bool
    decision: bool


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _gzip_json(value: object) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return gzip.compress(payload, compresslevel=6, mtime=0)


def _prepare_output(request: ScreenRequest) -> None:
    root = request.output_dir
    document = {
        "protocol": protocol.protocol_document(),
        "protocol_digest": protocol.protocol_digest(),
        "code_commit": request.code_commit,
    }
    authorization = {
        "g4_authorization_url": request.g4_authorization_url,
        "verified": False,
        "note": "attestation of the owner's authorization comment; not verified here",
    }
    existing = root / "protocol.json"
    if root.exists() and any(root.iterdir()):
        if not existing.is_file():
            raise FileExistsError(f"{root} is not an empty or resumable screen root")
        if json.loads(existing.read_text(encoding="utf-8")) != document:
            raise FileExistsError(f"{root} belongs to a different protocol or commit")
        previous = json.loads((root / "authorization.json").read_text("utf-8"))
        if previous != authorization:
            raise FileExistsError(f"{root} was authorized by a different comment")
        return
    root.mkdir(parents=True, exist_ok=True)
    _write_json(existing, document)
    _write_json(root / "authorization.json", authorization)


def _refuse_after_terminal_decision(root: Path) -> None:
    """``INCONCLUSIVE_UNIVERSE`` ends this protocol; the run is never reopened."""
    path = root / "decision.json"
    if not path.is_file():
        return
    if json.loads(path.read_text(encoding="utf-8")).get("outcome") == INCONCLUSIVE:
        raise FileExistsError(
            f"{root} already holds an {INCONCLUSIVE} decision, which is terminal for "
            f"this protocol: a different universe is a new protocol version, not a "
            "rerun"
        )


@dataclass(slots=True)
class _Loaded:
    symbol: str
    pair: tuple[RawMarketSeries, RawMarketSeries] | None
    facts: dict[str, object]


def _load_symbol(data: ScreenData, symbol: str) -> _Loaded:
    facts: dict[str, object] = {"symbol": symbol}

    def excluded(
        code: str, detail: str, evidence: Mapping[str, object] | None = None
    ) -> _Loaded:
        facts.update(
            {
                "admitted": False,
                "reason_code": code,
                "detail": detail[:400],
                "exclusion_evidence": None if evidence is None else dict(evidence),
            }
        )
        return _Loaded(symbol, None, facts)

    try:
        volumes = dict(data.lookback_median_volumes(symbol))
    except CarryCoverageError as error:
        return excluded("lookback_incomplete", str(error), error.evidence)
    facts["lookback_median_hourly_quote_volume"] = volumes
    if min(volumes.values()) < protocol.MIN_MEDIAN_HOURLY_QUOTE_VOLUME:
        return excluded(
            "liquidity_below_threshold",
            "median hourly quote volume",
            {"kind": "liquidity", "medians": volumes},
        )
    try:
        pair = data.pair(symbol)
    except CarryCoverageError as error:
        return excluded("window_coverage", str(error), error.evidence)
    probe = (
        assemble_carry_dataset
        if symbol in protocol.REFERENCE
        else _multisymbol_assembler
    )
    try:
        probe({symbol: pair})
    except ValueError as error:
        # gaps were classified as coverage facts already; a failure here is not one
        raise CarryIntegrityHalt(
            f"{symbol}: assembly refused the loaded series: {error}",
            {"kind": "assembly_failure", "symbol": symbol, "detail": str(error)},
        ) from error
    facts.update(
        {
            "admitted": True,
            "reason_code": None,
            "bars": int(len(pair[0].timestamps)),
            "halted_spot_bars": int((~pair[0].tradable).sum()),
            "funding_events": int(np.asarray(pair[1].funding_event_count).sum()),
        }
    )
    return _Loaded(symbol, pair, facts)


def _multisymbol_assembler(
    pairs: Mapping[str, tuple[RawMarketSeries, RawMarketSeries]],
    *,
    cost_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
) -> MarketDataset:
    return assemble_multisymbol_carry_dataset(
        pairs,
        universe=protocol.CANDIDATES,
        cost_multiplier=cost_multiplier,
        capacity_multiplier=capacity_multiplier,
    )


def _arms(admitted: Sequence[str], references: Sequence[str]) -> list[_Arm]:
    count = len(admitted)
    arms = [
        _Arm(
            protocol.PORTFOLIO_ARM,
            tuple(admitted),
            protocol.INITIAL_CAPITAL,
            False,
            True,
        )
    ]
    arms.extend(
        _Arm(symbol, (symbol,), protocol.INITIAL_CAPITAL / count, False, True)
        for symbol in admitted
    )
    if len(references) == len(protocol.REFERENCE):
        arms.append(
            _Arm(
                protocol.REFERENCE_SHARED_ARM,
                tuple(references),
                protocol.INITIAL_CAPITAL,
                True,
                False,
            )
        )
    arms.extend(
        _Arm(
            protocol.reference_arm(symbol),
            (symbol,),
            protocol.REFERENCE_PAIR_CAPITAL,
            True,
            False,
        )
        for symbol in references
    )
    return arms


def _cell(
    root: Path,
    arm: _Arm,
    treatment: protocol.Treatment,
    dataset: MarketDataset,
) -> CellSummary:
    summary_path = root / "cells" / f"{arm.name}__{treatment.name}.summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["dataset_id"] != dataset.dataset_id:
            raise FileExistsError(f"{summary_path} was produced from other data")
        return CellSummary.from_dict(summary)
    started = time.monotonic()
    replay = replay_carry(
        dataset,
        start_index=0,
        stop_index=dataset.n_bars - 1,
        initial_capital=arm.capital,
        entry_delay_bars=treatment.entry_delay_bars,
    )
    summary = CellSummary.from_replay(arm.name, treatment.name, replay)
    artifact = _gzip_json(replay)
    detail = root / "cells" / f"{arm.name}__{treatment.name}.json.gz"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_bytes(artifact)
    _write_json(
        summary_path,
        summary.to_dict()
        | {
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
            "artifact_bytes": len(artifact),
            "capital": arm.capital,
            "symbols": list(arm.symbols),
            "decision_arm": arm.decision,
            "replay_seconds": round(time.monotonic() - started, 3),
        },
    )
    return summary


def _publish_once(path: Path, value: object) -> None:
    """Write a stage record once; on resume it must reproduce exactly.

    A resumed run recomputes the inclusion and the data manifest. If they differ from
    the first run's record the data or the source changed under the run, which is an
    integrity failure: halt rather than overwrite the evidence.
    """
    if not path.is_file():
        _write_json(path, value)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != json.loads(json.dumps(value)):
        raise CarryIntegrityHalt(
            f"{path.name} differs from the record of the first run",
            {"kind": "resume_record_mismatch", "file": path.name},
        )


def _archive_stale_halt(root: Path) -> None:
    stale = root / "halt.json"
    if stale.is_file():
        folder = root / "halts"
        folder.mkdir(exist_ok=True)
        stale.replace(
            folder / f"halt-{len(list(folder.glob('halt-*.json'))) + 1:03d}.json"
        )


def _write_halt(
    root: Path,
    error: BaseException,
    *,
    symbol: str | None,
    request: ScreenRequest,
    at: datetime,
) -> None:
    """Leave a marker that the run halted instead of finishing or excluding."""
    if isinstance(error, CarryScreenHalt):
        kind, resumable, evidence = error.kind, error.resumable, error.evidence
    else:
        kind, resumable, evidence = "unexpected_error", False, None
    _write_json(
        root / "halt.json",
        {
            "kind": kind,
            "resumable": resumable,
            "symbol": symbol,
            "detail": f"{type(error).__name__}: {error}"[:500],
            "evidence": evidence,
            "protocol_digest": protocol.protocol_digest(),
            "code_commit": request.code_commit,
            "halted_at": at.isoformat(),
            "note": (
                "a halt is not an exclusion and not a result: no decision was written"
            ),
        },
    )


def run_screen(
    request: ScreenRequest,
    data: ScreenData,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = lambda _message: None,
) -> dict[str, Any]:
    """Run the registered screen; returns the decision document it also writes."""
    protocol.validate_window(request.start, request.end)
    protocol.validate_g4_authorization_url(request.g4_authorization_url)
    if COMMIT_PATTERN.fullmatch(request.code_commit) is None:
        raise ValueError("code_commit must be a full lowercase 40-hex commit")
    started_at = now()
    root = request.output_dir
    _prepare_output(request)
    _refuse_after_terminal_decision(root)
    _archive_stale_halt(root)
    current: dict[str, str | None] = {"symbol": None}
    try:
        loaded: dict[str, _Loaded] = {}
        for symbol in (*protocol.CANDIDATES, *protocol.REFERENCE):
            current["symbol"] = symbol
            loaded[symbol] = _load_symbol(data, symbol)
            log(f"{symbol}: admitted={loaded[symbol].facts['admitted']}")
        admitted = tuple(s for s in protocol.CANDIDATES if loaded[s].pair is not None)
        references = tuple(s for s in protocol.REFERENCE if loaded[s].pair is not None)
        excluded = {
            s: str(loaded[s].facts["reason_code"])
            for s in protocol.CANDIDATES
            if loaded[s].pair is None
        }
        _publish_once(
            root / "inclusion.json",
            {
                "admitted": list(admitted),
                "excluded": excluded,
                "references_loaded": list(references),
                "symbols": {s: loaded[s].facts for s in sorted(loaded)},
            },
        )
        _publish_once(
            root / "data-manifest.json",
            {
                "archives": data.evidence(),
                "source": data.name,
            },
        )
        if request.coverage_only:
            return {
                "coverage_only": True,
                "admitted": list(admitted),
                "excluded": excluded,
            }
        summaries: dict[tuple[str, str], CellSummary] = {}
        if len(admitted) >= protocol.MIN_ADMITTED:
            for arm in _arms(admitted, references):
                pairs: dict[str, tuple[RawMarketSeries, RawMarketSeries]] = {}
                for member in arm.symbols:
                    pair = loaded[member].pair
                    if pair is None:
                        raise ValueError(
                            f"arm {arm.name} member {member} is not loaded"
                        )
                    pairs[member] = pair
                assemble = (
                    assemble_carry_dataset
                    if arm.legacy_contract
                    else _multisymbol_assembler
                )
                datasets: dict[tuple[float, float], MarketDataset] = {}
                for treatment in protocol.TREATMENTS:
                    key = (treatment.cost_multiplier, treatment.capacity_multiplier)
                    if key not in datasets:
                        datasets[key] = assemble(
                            pairs,
                            cost_multiplier=treatment.cost_multiplier,
                            capacity_multiplier=treatment.capacity_multiplier,
                        )
                    summaries[(arm.name, treatment.name)] = _cell(
                        root, arm, treatment, datasets[key]
                    )
                    log(f"cell {arm.name}__{treatment.name} done")
        decision = decide(
            admitted,
            excluded,
            summaries,
            reference_arms=[
                name
                for name in (
                    protocol.REFERENCE_SHARED_ARM,
                    *(protocol.reference_arm(s) for s in references),
                )
                if (name, "base") in summaries
            ],
        )
        decision = decision | {
            "protocol_digest": protocol.protocol_digest(),
            "protocol_id": protocol.PROTOCOL_ID,
            "code_commit": request.code_commit,
            "started_at": started_at.isoformat(),
            "finished_at": now().isoformat(),
            "cells": {
                f"{arm}__{treatment}": cell.to_dict()
                for (arm, treatment), cell in sorted(summaries.items())
            },
        }
        _write_json(root / "decision.json", decision)
        return decision
    except Exception as error:
        _write_halt(root, error, symbol=current["symbol"], request=request, at=now())
        raise


__all__ = ["ScreenData", "ScreenRequest", "run_screen"]
