from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import run_universal_causal_alpha_v3_research as module
from trade_rl.workflows.universal_causal_alpha_v3_runner import (
    CausalAlphaV3AdmissionRejected,
    CausalAlphaV3SignalRejected,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
)


def _argv(tmp_path) -> list[str]:
    return [
        "--config",
        str(tmp_path / "v3.json"),
        "--run-config",
        str(tmp_path / "run.json"),
        "--runtime-manifest",
        str(tmp_path / "manifest.json"),
        "--frozen-metadata-root",
        str(tmp_path / "metadata"),
        "--output-root",
        str(tmp_path / "output"),
    ]


def _wire(monkeypatch, *, outcome) -> None:
    config = SimpleNamespace(digest="1" * 64)
    run_config = SimpleNamespace()
    context = SimpleNamespace(manifest=SimpleNamespace(fold_train_range=(10, 20)))
    runtime = SimpleNamespace()
    prepared = SimpleNamespace()

    monkeypatch.setattr(
        module.CausalAlphaV3ResearchConfig,
        "from_json",
        lambda path: config,
    )
    monkeypatch.setattr(
        module.TrainingRunConfig,
        "from_json",
        lambda path: run_config,
    )
    monkeypatch.setattr(module, "UniversalRuntimeFactoryContext", lambda **kwargs: context)
    monkeypatch.setattr(
        module,
        "load_universal_runtime_factory",
        lambda spec: (lambda **kwargs: runtime),
    )
    monkeypatch.setattr(
        module,
        "prepare_causal_alpha_v3_research_data",
        lambda **kwargs: prepared,
    )

    def run(**kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(module, "run_universal_causal_alpha_v3_research", run)


def test_cli_success_reports_research_only_package(monkeypatch, tmp_path, capsys) -> None:
    package = SimpleNamespace(
        digest="a" * 64,
        selected_candidate_digest="b" * 64,
        promotion_eligible=False,
        research_only=True,
    )
    _wire(monkeypatch, outcome=package)

    exit_code = module.main(_argv(tmp_path))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "package_digest": "a" * 64,
        "promotion_eligible": False,
        "research_only": True,
        "selected_candidate_digest": "b" * 64,
        "status": "admitted",
    }


@pytest.mark.parametrize(
    ("outcome", "expected_code", "expected_status"),
    [
        (
            CausalAlphaV3SignalRejected(({"fit_config_digest": "1" * 64},)),
            2,
            "signal_rejected",
        ),
        (CausalAlphaV3SelectionRejected(()), 3, "selection_rejected"),
        (
            CausalAlphaV3AdmissionRejected(
                admission_digest="2" * 64,
                selected_candidate_digest="3" * 64,
            ),
            4,
            "admission_rejected",
        ),
    ],
)
def test_cli_maps_terminal_research_rejections_to_distinct_exit_codes(
    monkeypatch,
    tmp_path,
    capsys,
    outcome,
    expected_code,
    expected_status,
) -> None:
    _wire(monkeypatch, outcome=outcome)

    exit_code = module.main(_argv(tmp_path))

    assert exit_code == expected_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status
    assert payload["promotion_eligible"] is False
