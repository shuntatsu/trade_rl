from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows import universal_trade_rl_universe_runner as runner
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_universe_config_v1",
        "train_symbols": ["BTCUSDT", "ETHUSDT"],
        "development_symbols": ["LINKUSDT"],
        "admission_symbols": ["AVAXUSDT"],
        "excluded_symbols": [
            {"symbol": "LUNA2USDT", "reason": "insufficient_contiguous_history"}
        ],
    }


def _source(symbol: str, char: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "dataset_digest": char * 64,
        "first_timestamp_ns": 1,
        "last_timestamp_ns": 100,
        "row_count": 100,
    }


def _catalog_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_source_catalog_v1",
        "symbols": [
            _source("AVAXUSDT", "a"),
            _source("BTCUSDT", "b"),
            _source("ETHUSDT", "c"),
            _source("LINKUSDT", "d"),
            _source("LUNA2USDT", "e"),
        ],
    }


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_json(tmp_path / "universe-input.json", _config_payload()),
        _write_json(tmp_path / "source-catalog.json", _catalog_payload()),
    )


def _argv(config: Path, catalog: Path, output: Path) -> list[str]:
    return [
        "--config",
        str(config),
        "--source-catalog",
        str(catalog),
        "--output-root",
        str(output),
    ]


def _load_artifacts(
    output: Path,
) -> tuple[UniversalTradeRLUniverseManifest, UniversalTradeRLRunIdentity]:
    universe = UniversalTradeRLUniverseManifest.from_payload(
        json.loads((output / "universe.json").read_text(encoding="utf-8"))
    )
    identity = UniversalTradeRLRunIdentity.from_payload(
        json.loads((output / "identity.json").read_text(encoding="utf-8"))
    )
    return universe, identity


def test_cli_materializes_bound_artifacts(tmp_path: Path, capsys) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"

    result = runner.cli_main(_argv(config, catalog, output))

    assert result == 0
    universe, identity = _load_artifacts(output)
    assert identity.stage is UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION
    assert identity.universe_manifest_digest == universe.digest
    assert identity.model_config_digest is None
    assert identity.fit_provenance_digests == ()
    terminal = json.loads(capsys.readouterr().out)
    assert terminal == {
        "identity_digest": identity.digest,
        "production_status": "NO-GO",
        "status": "materialized",
        "universe_manifest_digest": universe.digest,
    }


def test_materialize_returns_valid_bound_contracts(tmp_path: Path) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"

    universe, identity = runner.materialize_universal_trade_rl_universe(
        config_path=config,
        source_catalog_path=catalog,
        output_root=output,
    )

    assert universe == _load_artifacts(output)[0]
    assert identity == _load_artifacts(output)[1]
    assert identity.universe_manifest_digest == universe.digest


def test_unassigned_source_failure_leaves_no_output(tmp_path: Path, capsys) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    payload = _catalog_payload()
    payload["symbols"] = [*payload["symbols"], _source("XRPUSDT", "f")]  # type: ignore[index]
    _write_json(catalog, payload)
    output = tmp_path / "output"

    result = runner.cli_main(_argv(config, catalog, output))

    assert result == 5
    assert not output.exists()
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["status"] == "rejected"
    assert terminal["production_status"] == "NO-GO"


def test_second_staged_write_failure_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    original = runner._write_canonical_json
    calls = 0

    def fail_on_second_write(path: Path, payload: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-write failure")
        original(path, payload)

    monkeypatch.setattr(runner, "_write_canonical_json", fail_on_second_write)

    with pytest.raises(OSError, match="second-write"):
        runner.materialize_universal_trade_rl_universe(
            config_path=config,
            source_catalog_path=catalog,
            output_root=output,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".output.staging-*"))


def test_identical_rerun_is_byte_identical_idempotent_success(tmp_path: Path) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    first = runner.materialize_universal_trade_rl_universe(
        config_path=config,
        source_catalog_path=catalog,
        output_root=output,
    )
    before = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir(), key=lambda item: item.name)
    }

    second = runner.materialize_universal_trade_rl_universe(
        config_path=config,
        source_catalog_path=catalog,
        output_root=output,
    )
    after = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir(), key=lambda item: item.name)
    }

    assert second == first
    assert after == before
    assert set(after) == {"identity.json", "universe.json"}


def test_edited_existing_artifact_is_rejected_without_repair(tmp_path: Path) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    runner.materialize_universal_trade_rl_universe(
        config_path=config,
        source_catalog_path=catalog,
        output_root=output,
    )
    identity_path = output / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["universe_manifest_digest"] = "f" * 64
    _write_json(identity_path, identity)
    tampered = identity_path.read_bytes()

    with pytest.raises(ValueError, match="existing|drift|digest"):
        runner.materialize_universal_trade_rl_universe(
            config_path=config,
            source_catalog_path=catalog,
            output_root=output,
        )

    assert identity_path.read_bytes() == tampered


def test_role_config_drift_is_rejected_against_existing_output(tmp_path: Path) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    runner.materialize_universal_trade_rl_universe(
        config_path=config,
        source_catalog_path=catalog,
        output_root=output,
    )
    payload = _config_payload()
    payload["development_symbols"] = ["AVAXUSDT"]
    payload["admission_symbols"] = ["LINKUSDT"]
    _write_json(config, payload)

    with pytest.raises(ValueError, match="existing|drift"):
        runner.materialize_universal_trade_rl_universe(
            config_path=config,
            source_catalog_path=catalog,
            output_root=output,
        )


def test_source_digest_drift_is_rejected_against_existing_output(tmp_path: Path) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    runner.materialize_universal_trade_rl_universe(
        config_path=config,
        source_catalog_path=catalog,
        output_root=output,
    )
    payload = _catalog_payload()
    records = [dict(item) for item in payload["symbols"]]  # type: ignore[index]
    records[1]["dataset_digest"] = "f" * 64
    payload["symbols"] = records
    _write_json(catalog, payload)

    with pytest.raises(ValueError, match="existing|drift"):
        runner.materialize_universal_trade_rl_universe(
            config_path=config,
            source_catalog_path=catalog,
            output_root=output,
        )


def test_preexisting_empty_output_directory_is_rejected(tmp_path: Path) -> None:
    config, catalog = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError, match="existing"):
        runner.materialize_universal_trade_rl_universe(
            config_path=config,
            source_catalog_path=catalog,
            output_root=output,
        )


def test_cli_rejects_unknown_arguments() -> None:
    with pytest.raises(SystemExit):
        runner.cli_main(["--unexpected", "value"])
