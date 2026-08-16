from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_universal_causal_alpha_v3_signal as module


def _report() -> SimpleNamespace:
    payload = {
        "artifact_digest": "a" * 64,
        "promotion_eligible": False,
        "raw_scope_count": 72,
        "research_only": True,
        "schema_version": "causal_alpha_v3_signal_forensics_v1",
    }
    return SimpleNamespace(to_payload=lambda: payload)


def test_cli_emits_canonical_forensics_json_to_stdout(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    marker = run_root / "source-marker.txt"
    marker.write_bytes(b"unchanged")
    observed: list[Path] = []

    def fake_load(root: Path):
        observed.append(root)
        return _report()

    monkeypatch.setattr(module, "load_causal_alpha_v3_signal_forensics", fake_load)

    exit_code = module.main([str(run_root)])

    assert exit_code == 0
    assert observed == [run_root]
    assert marker.read_bytes() == b"unchanged"
    expected = json.dumps(
        _report().to_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert capsys.readouterr().out == expected + "\n"


def test_cli_writes_same_json_only_to_external_output(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    marker = run_root / "source-marker.txt"
    marker.write_bytes(b"unchanged")
    output = tmp_path / "reports" / "signal-forensics.json"
    monkeypatch.setattr(
        module,
        "load_causal_alpha_v3_signal_forensics",
        lambda root: _report(),
    )

    exit_code = module.main([str(run_root), "--output", str(output)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert marker.read_bytes() == b"unchanged"
    assert json.loads(output.read_text(encoding="utf-8")) == _report().to_payload()
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_cli_rejects_output_inside_source_run(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(
        module,
        "load_causal_alpha_v3_signal_forensics",
        lambda root: _report(),
    )

    with pytest.raises(ValueError, match="outside"):
        module.main(
            [
                str(run_root),
                "--output",
                str(run_root / "signal" / "forensics.json"),
            ]
        )
