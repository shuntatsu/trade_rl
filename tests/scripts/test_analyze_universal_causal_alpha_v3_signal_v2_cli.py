from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import analyze_universal_causal_alpha_v3_signal as module


def _report(schema: str) -> SimpleNamespace:
    payload = {
        "artifact_digest": "b" * 64,
        "promotion_eligible": False,
        "research_only": True,
        "schema_version": schema,
    }
    return SimpleNamespace(to_payload=lambda: payload)


def test_cli_schema_v1_remains_explicitly_selectable(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    observed: list[tuple[str, Path]] = []

    def load_v1(root: Path):
        observed.append(("v1", root))
        return _report("causal_alpha_v3_signal_forensics_v1")

    def load_v2(root: Path):
        observed.append(("v2", root))
        return _report("causal_alpha_v3_signal_forensics_v2")

    monkeypatch.setattr(module, "load_causal_alpha_v3_signal_forensics", load_v1)
    monkeypatch.setattr(
        module,
        "load_causal_alpha_v3_signal_forensics_v2",
        load_v2,
        raising=False,
    )

    assert module.main([str(run_root), "--schema", "v1"]) == 0
    assert observed == [("v1", run_root)]
    assert "causal_alpha_v3_signal_forensics_v1" in capsys.readouterr().out


def test_cli_schema_v2_uses_only_v2_loader(tmp_path: Path, capsys, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    observed: list[tuple[str, Path]] = []

    def load_v1(root: Path):
        observed.append(("v1", root))
        return _report("causal_alpha_v3_signal_forensics_v1")

    def load_v2(root: Path):
        observed.append(("v2", root))
        return _report("causal_alpha_v3_signal_forensics_v2")

    monkeypatch.setattr(module, "load_causal_alpha_v3_signal_forensics", load_v1)
    monkeypatch.setattr(
        module,
        "load_causal_alpha_v3_signal_forensics_v2",
        load_v2,
        raising=False,
    )

    assert module.main([str(run_root), "--schema", "v2"]) == 0
    assert observed == [("v2", run_root)]
    assert "causal_alpha_v3_signal_forensics_v2" in capsys.readouterr().out
