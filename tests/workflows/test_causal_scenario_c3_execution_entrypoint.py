from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.workflows.causal_scenario import c3_execution


def _request(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_text(
        '{"schema_version":"causal_scenario_c3_request_v1"}\n', encoding="utf-8"
    )
    return path


def test_execute_c3_evaluation_request_publishes_evidence_from_injected_backend(
    tmp_path: Path,
    c3_reporting,
) -> None:
    request = _request(tmp_path)
    output = tmp_path / "output"

    def backend(request_path: Path, *, output_root: Path) -> Path:
        assert request_path == request
        path = output_root / "summary.json"
        c3_reporting.write_summary(path)
        return path

    result = c3_execution.execute_c3_evaluation_request(
        request,
        output_root=output,
        backend=backend,
    )

    assert result.production_status == "NO-GO"
    assert result.gate.passed is True
    assert result.report_artifact_root == output / "report"
    assert result.gate_artifact_root == output / "gate"
    assert result.report_artifact_root.is_dir()
    assert result.gate_artifact_root.is_dir()
    assert result.source_summary_path == output / "core" / "summary.json"
    assert len(result.report_artifact_digest) == 64
    assert len(result.gate_artifact_digest) == 64


def test_execute_c3_evaluation_request_resolves_relative_backend_path(
    tmp_path: Path,
    c3_reporting,
) -> None:
    request = _request(tmp_path)

    def backend(request_path: Path, *, output_root: Path) -> Path:
        c3_reporting.write_summary(output_root / "nested" / "summary.json")
        return Path("nested/summary.json")

    result = c3_execution.execute_c3_evaluation_request(
        request,
        output_root=tmp_path / "output",
        backend=backend,
    )

    assert result.source_summary_path.name == "summary.json"
    assert result.source_summary_path.parent.name == "nested"


def test_execute_c3_evaluation_request_rejects_backend_output_escape(
    tmp_path: Path,
    c3_reporting,
) -> None:
    request = _request(tmp_path)
    outside = tmp_path / "outside.json"
    c3_reporting.write_summary(outside)

    def backend(request_path: Path, *, output_root: Path) -> Path:
        return outside

    with pytest.raises(ValueError, match="outside the core output root"):
        c3_execution.execute_c3_evaluation_request(
            request,
            output_root=tmp_path / "output",
            backend=backend,
        )


def test_execute_c3_evaluation_request_rejects_symlinked_backend_summary(
    tmp_path: Path,
    c3_reporting,
) -> None:
    request = _request(tmp_path)

    def backend(request_path: Path, *, output_root: Path) -> Path:
        target = output_root / "summary-target.json"
        link = output_root / "summary.json"
        c3_reporting.write_summary(target)
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable on this platform")
        return link

    with pytest.raises(ValueError, match="symbolic link"):
        c3_execution.execute_c3_evaluation_request(
            request,
            output_root=tmp_path / "output",
            backend=backend,
        )


def test_execute_c3_evaluation_request_rejects_malformed_backend_summary(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    def backend(request_path: Path, *, output_root: Path) -> Path:
        path = output_root / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    with pytest.raises(ValueError, match="field closure"):
        c3_execution.execute_c3_evaluation_request(
            request,
            output_root=tmp_path / "output",
            backend=backend,
        )


def test_execute_c3_evaluation_request_fails_closed_when_backend_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)

    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(c3_execution.importlib, "import_module", missing)

    with pytest.raises(c3_execution.C3CoreBackendUnavailable, match="lane B"):
        c3_execution.execute_c3_evaluation_request(
            request,
            output_root=tmp_path / "output",
        )


def test_execute_c3_evaluation_request_rejects_missing_request(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="request file"):
        c3_execution.execute_c3_evaluation_request(
            tmp_path / "missing.json",
            output_root=tmp_path / "output",
            backend=lambda request_path, *, output_root: output_root / "summary.json",
        )


def test_execute_c3_evaluation_request_never_authorizes_production(
    tmp_path: Path,
    c3_reporting,
) -> None:
    request = _request(tmp_path)

    def backend(request_path: Path, *, output_root: Path) -> Path:
        path = output_root / "summary.json"
        c3_reporting.write_summary(path)
        return path

    result = c3_execution.execute_c3_evaluation_request(
        request,
        output_root=tmp_path / "output",
        backend=backend,
    )

    assert result.gate.passed is True
    assert result.production_status == "NO-GO"
