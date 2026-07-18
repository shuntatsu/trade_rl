from __future__ import annotations

import hashlib
import importlib
import json
import sys
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.binance import BinanceExchangeInfoSnapshot
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.offline_signing import public_key_bytes, sign_payload
from trade_rl.workflows.binance_metadata_modes import (
    BinanceMetadataMode,
    load_verified_binance_rule_history,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")


def _namespace() -> dict[str, Any]:
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return vars(importlib.import_module("full_research_pipeline"))


def _state_module() -> Any:
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return importlib.import_module("run_full_research_state")


def _snapshot() -> BinanceExchangeInfoSnapshot:
    payload = {
        "symbols": [
            {
                "symbol": symbol,
                "contractStatus": "TRADING",
                "onboardDate": 1_600_000_000_000,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
            for symbol in SYMBOLS
        ]
    }
    raw = json.dumps(payload).encode("utf-8")
    return BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 7, 17, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


class _Transport:
    def __init__(self) -> None:
        self.calls = 0

    def load_exchange_information_snapshot(
        self, **_: object
    ) -> BinanceExchangeInfoSnapshot:
        self.calls += 1
        return _snapshot()


def _verified_history():
    now = datetime(2026, 7, 17, tzinfo=UTC)
    start = datetime(2024, 12, 1, tzinfo=UTC)
    end = datetime(2026, 7, 1, tzinfo=UTC)
    payload = {
        "schema_version": "binance_instrument_rule_history_v4",
        "policy_version": "binance_metadata_modes_v2",
        "market": "usds-m",
        "symbol_order": list(SYMBOLS),
        "coverage": {"start_time": start.isoformat(), "end_time": end.isoformat()},
        "issued_at": now.isoformat(),
        "source_uri": "operator://signed-binance-rules",
        "symbols": {
            symbol: {
                "listed_at": "2020-01-01T00:00:00+00:00",
                "tick_size": 0.1,
                "lot_size": 0.001,
                "minimum_notional": 5.0,
                "execution_rules": [
                    {
                        "effective_at": start.isoformat(),
                        "tick_size": 0.1,
                        "lot_size": 0.001,
                        "minimum_notional": 5.0,
                    }
                ],
            }
            for symbol in SYMBOLS
        },
    }
    private_key = Ed25519PrivateKey.from_private_bytes(b"\x44" * 32)
    public_key = PublicVerificationKey(
        key_id="metadata-2026",
        public_key=public_key_bytes(private_key),
        purpose="binance-rule-history",
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=365),
    )
    envelope = sign_payload(
        payload,
        key_id=public_key.key_id,
        purpose=public_key.purpose,
        private_key=private_key,
        signed_at=now,
    )
    envelope_mapping = envelope.to_mapping()
    envelope_mapping["signed_at"] = envelope.signed_at.isoformat()
    return load_verified_binance_rule_history(
        {"payload": payload, "envelope": envelope_mapping},
        trusted_keys={public_key.key_id: public_key},
        trusted_now=now,
    )


def test_runner_frozen_mode_does_not_require_signed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    resolve = namespace["_resolve_metadata"]
    monkeypatch.setitem(
        resolve.__globals__,
        "_load_rule_history",
        lambda: pytest.fail("frozen mode must not read signed history"),
    )
    transport = _Transport()

    result = resolve(
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        transport=transport,
        conservative_static_path=None,
    )

    assert transport.calls == 1
    assert result.mode is BinanceMetadataMode.FROZEN_SNAPSHOT
    assert result.execution_rule_histories is None


def test_runner_historical_mode_accepts_only_verified_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    resolve = namespace["_resolve_metadata"]
    verified = _verified_history()
    calls = 0

    def load() -> object:
        nonlocal calls
        calls += 1
        return verified

    monkeypatch.setitem(resolve.__globals__, "_load_rule_history", load)
    result = resolve(
        mode=BinanceMetadataMode.HISTORICAL_SIGNED,
        transport=_Transport(),
        conservative_static_path=None,
    )

    assert calls == 1
    assert result.mode is BinanceMetadataMode.HISTORICAL_SIGNED
    assert result.execution_rule_histories == verified.execution_rule_histories
    assert result.identity_evidence["point_in_time"] is True
    assert result.raw_payload == verified.signed_document


def test_runner_conservative_mode_requires_explicit_payload() -> None:
    resolve = _namespace()["_resolve_metadata"]
    with pytest.raises(ValueError, match="conservative-static-path"):
        resolve(
            mode=BinanceMetadataMode.CONSERVATIVE_STATIC,
            transport=_Transport(),
            conservative_static_path=None,
        )


def test_develop_accepts_supervised_bootstrap_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _state_module()
    work_root = tmp_path / "generation"
    work_root.mkdir()
    for name in (
        "cuda-preflight.json",
        "entrypoint-provenance.json",
        "heartbeat.json",
    ):
        (work_root / name).write_text("{}\n", encoding="utf-8")

    class ReachedMetadataResolution(Exception):
        pass

    def reached_resolution(**_: object) -> None:
        raise ReachedMetadataResolution

    monkeypatch.setattr(module.pipeline, "resolve_metadata", reached_resolution)
    stages = module.BinanceFullResearchStages(
        Namespace(
            cache_root=tmp_path / "cache",
            conservative_static_path=None,
            metadata_mode="frozen_snapshot",
        )
    )

    with pytest.raises(ReachedMetadataResolution):
        stages._develop(work_root)


def test_develop_rejects_prior_research_artifacts_without_deleting_them(
    tmp_path: Path,
) -> None:
    module = _state_module()
    work_root = tmp_path / "generation"
    work_root.mkdir()
    prior_summary = work_root / "summary.json"
    prior_summary.write_text('{"status":"infrastructure_error"}\n', encoding="utf-8")
    stages = module.BinanceFullResearchStages(
        Namespace(
            cache_root=tmp_path / "cache",
            conservative_static_path=None,
            metadata_mode="frozen_snapshot",
        )
    )

    with pytest.raises(FileExistsError, match="research generation already exists"):
        stages._develop(work_root)

    assert prior_summary.read_text(encoding="utf-8") == (
        '{"status":"infrastructure_error"}\n'
    )


def test_runner_source_uses_typed_state_and_public_key_verification() -> None:
    state = (EXAMPLE_ROOT / "run_full_research_state.py").read_text(encoding="utf-8")
    pipeline = (EXAMPLE_ROOT / "full_research_pipeline.py").read_text(encoding="utf-8")
    launchers = "\n".join(
        (EXAMPLE_ROOT / name).read_text(encoding="utf-8")
        for name in ("run_full_research.py", "run_full_research_hardened.py")
    )

    assert "SelectionProposal.create" in state
    assert "SelectionAuthorization.authorize" not in state
    assert "load_verified_binance_rule_history" in pipeline
    assert "TRADE_RL_METADATA_PUBLIC_KEYS" in pipeline
    assert "runpy" not in launchers


def test_runner_requires_identity_verified_execution_sensitivity_gate(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    load_gate = namespace["_execution_sensitivity_gate"]
    payload = {
        "dataset_id": "a" * 64,
        "experiment_plan_digest": "b" * 64,
        "folds": [],
        "gate": {
            "passed": True,
            "required_scenario": "joint_2x",
            "selected_total_return": 0.01,
            "baseline_uplift": 0.001,
            "maximum_fold_drawdown": 0.1,
        },
        "production_status": "NO-GO",
        "scenario_pack_digest": "c" * 64,
        "schema_version": "execution_sensitivity_v1",
    }
    payload["artifact_digest"] = content_digest(payload)
    (tmp_path / "execution-sensitivity.json").write_text(json.dumps(payload))
    (tmp_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "experiment_plan_digest": payload["experiment_plan_digest"],
                "execution_sensitivity_digest": payload["artifact_digest"],
            }
        )
    )

    passed, gate = load_gate(tmp_path)
    assert passed is True
    assert gate["required_scenario"] == "joint_2x"

    payload["gate"]["selected_total_return"] = -0.1
    (tmp_path / "execution-sensitivity.json").write_text(json.dumps(payload))
    passed, gate = load_gate(tmp_path)
    assert passed is False
    assert "digest" in gate["reason"]


def test_runner_skips_execution_sensitivity_when_not_configured(tmp_path: Path) -> None:
    load_gate = _namespace()["_execution_sensitivity_gate"]
    (tmp_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "dataset_id": "a" * 64,
                "experiment_plan_digest": "b" * 64,
                "execution_sensitivity_digest": None,
            }
        )
    )
    passed, gate = load_gate(tmp_path)
    assert passed is True
    assert gate["required"] is False


def test_runner_rejects_execution_sensitivity_identity_mismatch(tmp_path: Path) -> None:
    namespace = _namespace()
    load_gate = namespace["_execution_sensitivity_gate"]
    payload = {
        "dataset_id": "a" * 64,
        "experiment_plan_digest": "b" * 64,
        "folds": [],
        "gate": {"passed": True},
        "production_status": "NO-GO",
        "scenario_pack_digest": "c" * 64,
        "schema_version": "execution_sensitivity_v1",
    }
    payload["artifact_digest"] = content_digest(payload)
    (tmp_path / "execution-sensitivity.json").write_text(json.dumps(payload))
    (tmp_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "dataset_id": "d" * 64,
                "experiment_plan_digest": payload["experiment_plan_digest"],
                "execution_sensitivity_digest": payload["artifact_digest"],
            }
        )
    )
    passed, gate = load_gate(tmp_path)
    assert passed is False
    assert "dataset_id" in gate["reason"]
