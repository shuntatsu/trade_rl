"""Read-only resume preflight with temporary JSON/gzip and stubbed forecast IO."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.tools.tier_d_nonrl.plan_fixtures import costs_payload, matrix_payload
from trade_rl.artifacts import canonical_json_bytes, content_digest


def api():
    return importlib.import_module("tools.tier_d_nonrl.resume")


def test_windows_reparse_resolution_must_stay_inside_source(tmp_path, monkeypatch):
    module = api()
    source = tmp_path / "source"
    source.mkdir()
    logical = source / "retained.json"
    logical.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    original = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, strict=False: (
            outside if path == logical else original(path, strict=strict)
        ),
    )
    with pytest.raises(ValueError, match="inside|escape|contain"):
        module._file(source, "retained.json")


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "status,complete,reader_accepts,accepted",
    [
        ("completed", True, True, True),
        ("COMPLETED", True, True, False),
        ("economic_stop", True, True, False),
        ("integrity_failure", True, True, False),
        ("completed", False, True, False),
        ("completed", True, False, False),
    ],
)
def test_completed_receipt_status_uses_the_public_lowercase_contract(
    tmp_path, monkeypatch, status, complete, reader_accepts, accepted
):
    module, records, source, protocol, provenance = source_fixture(
        tmp_path, monkeypatch
    )
    state = records.read_json(source / "attempt.json")
    frozen = records.read_json(source / "freeze.json")
    matrix = module.restore_matrix(
        records.load_gzip(source / frozen["matrix_file"]["path"]),
        expected_digest=state["matrix_digest"],
    )
    plan = matrix.controls[0].nonrl_plan
    # This tiny JSON stubs the financial reader only to isolate its status seam.
    raw = {
        "scope": "ARTIFICIAL_RECEIPT_STATUS_STUB_ONLY",
        "status": status,
        "settlement_complete": complete,
    }
    relative = "receipts/A_BASEL0_nonrl.json.gz"
    metadata = records.save_gzip(source, relative, raw)
    state["arms"][0].update(
        state="COMPLETED",
        receipt=metadata,
        plan_digest=plan.digest,
        receipt_digest=content_digest(raw),
    )
    records.write_json(source / "attempt.json", state, replace=True)
    records.write_json(
        source / "output-manifest.json", records.output_manifest(source), replace=True
    )
    calls = []

    def read_receipt(value, *, expected_plan):
        calls.append(expected_plan.digest)
        assert value == raw
        if not reader_accepts:
            raise ValueError("public reader rejected this manufactured receipt")
        return SimpleNamespace(digest=content_digest(value))

    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(from_payload=read_receipt),
    )
    before, destination = snapshot(source), tmp_path / "next"
    if accepted:
        reused = module.preflight_resume(source, destination, protocol, provenance)
        assert relative in reused.files
    else:
        with pytest.raises(ValueError):
            module.preflight_resume(source, destination, protocol, provenance)
    assert calls == [plan.digest]
    assert snapshot(source) == before
    assert not destination.exists()


def source_fixture(tmp_path, monkeypatch):
    module = api()
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    restore = importlib.import_module("tools.tier_d_nonrl.restore")
    protocol = {
        "scope": "ARTIFICIAL_RESUME_IO_ONLY",
        "source_head": "a" * 40,
        "accounts": ["A"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "configurations": [{"id": "fixture", "horizon_hours": 24}],
        "dataset": {"dataset_id": "a" * 64, "fit_symbols": ["ARTIFICIAL"]},
    }
    provenance = {
        "source_head": "a" * 40,
        "driver_sha256": {"fixture.py": "b" * 64},
        "native": {
            "implementation_digest": "a" * 64,
            "runtime_environment_digest": "a" * 64,
        },
    }
    source = tmp_path / "source"
    state = records.initialize_attempt(source, protocol)
    records.write_json(source / "provenance.json", provenance)
    path = source / "forecasts/fixture.json"
    path.parent.mkdir()
    path.write_bytes(
        canonical_json_bytes({"scope": "ARTIFICIAL_FORECAST_LOADER_STUB_ONLY"})
    )
    digest = content_digest({"scope": "ARTIFICIAL_FORECAST_LOADER_STUB_ONLY"})
    stream = SimpleNamespace(digest=digest, dataset_id="a" * 64)
    prepared = {
        "fixture": SimpleNamespace(
            stream=stream,
            artifact=SimpleNamespace(
                path=path, digest=digest, dataset_id=stream.dataset_id
            ),
        )
    }
    monkeypatch.setattr(
        module, "load_simple_return_stream_artifact", lambda *args, **kwargs: stream
    )
    costs = costs_payload()
    tables = restore.restore_costs(costs, expected_digest=content_digest(costs))
    raw = matrix_payload(
        {
            "forecast_context_digest": digest,
            "cost_beliefs_digest": content_digest(costs["tables"][0]["rows"]),
        }
    )
    matrix = restore.restore_matrix(raw, expected_digest=content_digest(raw))
    pair = SimpleNamespace(
        configuration="fixture",
        account="A",
        scenario="BASEL0",
        symbol_index=0,
        controls=matrix.controls[0],
    )
    frozen = module.freeze_inputs(
        source, (pair,), prepared, tables, matrix, protocol, provenance
    )
    state.update(
        status="FAILED",
        matrix_digest=matrix.digest,
        freeze_digest=content_digest(frozen),
    )
    records.write_json(source / "attempt.json", state, replace=True)
    records.write_json(source / "output-manifest.json", records.output_manifest(source))
    return module, records, source, protocol, provenance


def test_resume_preflight_is_read_only_and_materializes_into_new_root(
    tmp_path, monkeypatch
):
    module, records, source, protocol, provenance = source_fixture(
        tmp_path, monkeypatch
    )
    before = snapshot(source)
    destination = tmp_path / "next"
    reused = module.preflight_resume(source, destination, protocol, provenance)
    assert not destination.exists()
    assert snapshot(source) == before
    records.initialize_attempt(destination, protocol)
    reused.materialize(destination)
    assert snapshot(source) == before
    assert (destination / "native-plan.json.gz").read_bytes() == (
        source / "native-plan.json.gz"
    ).read_bytes()
    assert (
        reused.matrix.digest
        == records.read_json(source / "attempt.json")["matrix_digest"]
    )


@pytest.mark.parametrize(
    "change", ["source", "protocol", "corrupt", "missing_matrix", "nested_destination"]
)
def test_invalid_resume_never_creates_new_outputs_or_changes_source(
    tmp_path, monkeypatch, change
):
    module, records, source, protocol, provenance = source_fixture(
        tmp_path, monkeypatch
    )
    destination = tmp_path / "next"
    if change == "source":
        provenance = copy.deepcopy(provenance)
        provenance["native"]["implementation_digest"] = "c" * 64
    elif change == "protocol":
        protocol = dict(protocol, scope="different")
    elif change == "corrupt":
        (source / "native-plan.json.gz").write_bytes(b"preserved corrupt artifact")
    elif change == "missing_matrix":
        state = records.read_json(source / "attempt.json")
        state.pop("matrix_digest")
        records.write_json(source / "attempt.json", state, replace=True)
        records.write_json(
            source / "output-manifest.json",
            records.output_manifest(source),
            replace=True,
        )
    else:
        destination = source / "next"
    before = snapshot(source)
    with pytest.raises(ValueError):
        module.preflight_resume(source, destination, protocol, provenance)
    assert not destination.exists()
    assert snapshot(source) == before
