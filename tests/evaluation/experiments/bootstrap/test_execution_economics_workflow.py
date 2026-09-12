from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.evaluation.experiments.bootstrap.test_binance import _config
from trade_rl.data.build import ExecutionEconomicsProfile
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
)


class _StopAfterBuildObservation(RuntimeError):
    pass


def _profile() -> ExecutionEconomicsProfile:
    return ExecutionEconomicsProfile(
        name="canonical_m2_research_assumption_v1",
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )


def _write_config(tmp_path: Path, *, v2: bool) -> tuple[Path, object]:
    config = _config()
    if v2:
        config = replace(
            config,
            schema_version="canonical_m2_bootstrap_config_v2",
            execution_economics=_profile(),
        )
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(config.to_payload()), encoding="utf-8")
    return path, config


@pytest.mark.parametrize("v2", [False, True])
def test_bootstrap_passes_versioned_execution_economics_to_binance_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v2: bool,
) -> None:
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    config_path, config = _write_config(tmp_path, v2=v2)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        workflow_module,
        "_freeze_binance_source",
        lambda *_args, **_kwargs: SimpleNamespace(
            composite_transport=object(),
            metadata_evidence={"source": "fixture"},
        ),
    )

    def observe_build(**kwargs: object) -> None:
        observed.update(kwargs)
        raise _StopAfterBuildObservation

    monkeypatch.setattr(workflow_module, "build_binance_market_dataset", observe_build)
    output = tmp_path / "canonical-m2"

    with pytest.raises(_StopAfterBuildObservation):
        bootstrap_canonical_m2_study(config_path, output)

    expected = _profile() if v2 else None
    assert observed.get("execution_economics") == expected
    assert not output.exists()
    assert not any(
        path.name.startswith(".canonical-m2.staging-") for path in tmp_path.iterdir()
    )
    assert config.execution_economics == expected


def test_v2_bootstrap_rejects_dataset_that_ignores_execution_economics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.evaluation.experiments.bootstrap.test_workflow import _install_fakes

    _install_fakes(monkeypatch)
    config_path, _config_value = _write_config(tmp_path, v2=True)
    output = tmp_path / "canonical-m2"

    with pytest.raises(ValueError, match="execution economics"):
        bootstrap_canonical_m2_study(config_path, output)

    assert not output.exists()
    assert not any(
        path.name.startswith(".canonical-m2.staging-") for path in tmp_path.iterdir()
    )
