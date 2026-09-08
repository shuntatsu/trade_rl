from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _provenance(
    implementation: str = "a",
    runtime: str = "b",
    *,
    context: str | None = None,
) -> dict[str, object]:
    return {
        "implementation_digest": implementation * 64,
        "runtime_environment_digest": runtime * 64,
        "research_context_digest": context,
    }


def test_run_candidate_artifact_delegates_through_run_core(monkeypatch, tmp_path) -> None:
    from trade_rl.evaluation.runs import candidate

    dataset = SimpleNamespace(dataset_id="d" * 64)
    dataset_artifact = SimpleNamespace(
        schema_version="market_dataset_artifact_v3",
        artifact_digest="e" * 64,
    )
    config = object()
    spec = object()
    result = object()
    published = SimpleNamespace(root=tmp_path / "published")
    order: list[str] = []

    def inspect(path):
        order.append("inspect")
        return dataset_artifact

    def load_dataset(path):
        order.append("load_dataset")
        return dataset

    def load_config(path):
        order.append("load_config")
        return config

    def resolve(loaded, *, dataset_artifact_schema, dataset_artifact_digest, config):
        order.append("resolve")
        assert loaded is dataset
        assert dataset_artifact_schema == "market_dataset_artifact_v3"
        assert dataset_artifact_digest == "e" * 64
        return spec

    provenance_calls = 0

    def provenance(*, research_context_digest=None):
        nonlocal provenance_calls
        provenance_calls += 1
        order.append(f"provenance_{provenance_calls}")
        return _provenance(context=research_context_digest)

    def execute(loaded, resolved):
        order.append("execute")
        assert loaded is dataset
        assert resolved is spec
        return result

    def publish(output_root, run_result, run_provenance):
        order.append("publish")
        assert Path(output_root) == tmp_path / "result"
        assert run_result is result
        assert run_provenance["research_context_digest"] == "f" * 64
        return published

    monkeypatch.setattr(candidate, "inspect_published_market_dataset_artifact", inspect)
    monkeypatch.setattr(candidate, "load_market_dataset_artifact", load_dataset)
    monkeypatch.setattr(candidate, "load_candidate_run_config", load_config)
    monkeypatch.setattr(candidate, "resolve_candidate_run_spec", resolve)
    monkeypatch.setattr(candidate, "build_candidate_run_provenance", provenance)
    monkeypatch.setattr(candidate, "execute_candidate_run", execute)
    monkeypatch.setattr(candidate, "publish_candidate_run", publish)

    actual = candidate.run_candidate_artifact(
        dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "config.json",
        output_root=tmp_path / "result",
        research_context_digest="f" * 64,
    )

    assert actual is published
    assert order == [
        "inspect",
        "load_dataset",
        "load_config",
        "resolve",
        "provenance_1",
        "execute",
        "provenance_2",
        "publish",
    ]


def test_run_candidate_artifact_rejects_provenance_drift_before_publish(
    monkeypatch,
    tmp_path,
) -> None:
    from trade_rl.evaluation.runs import candidate

    monkeypatch.setattr(
        candidate,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="e" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate,
        "load_market_dataset_artifact",
        lambda path: SimpleNamespace(dataset_id="d" * 64),
    )
    monkeypatch.setattr(candidate, "load_candidate_run_config", lambda path: object())
    monkeypatch.setattr(
        candidate,
        "resolve_candidate_run_spec",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(candidate, "execute_candidate_run", lambda *args: object())
    provenances = iter((_provenance("a", "b"), _provenance("c", "b")))
    monkeypatch.setattr(
        candidate,
        "build_candidate_run_provenance",
        lambda **kwargs: next(provenances),
    )
    published = False

    def publish(*args, **kwargs):
        nonlocal published
        published = True
        return object()

    monkeypatch.setattr(candidate, "publish_candidate_run", publish)

    with pytest.raises(RuntimeError, match="provenance changed during execution"):
        candidate.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=tmp_path / "config.json",
            output_root=tmp_path / "result",
        )

    assert published is False
    assert not (tmp_path / "result").exists()


def test_candidate_cli_keeps_existing_arguments(monkeypatch, capsys) -> None:
    from trade_rl.evaluation.runs import candidate

    calls: dict[str, object] = {}

    def run_candidate_artifact(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(root=Path("result"))

    monkeypatch.setattr(candidate, "run_candidate_artifact", run_candidate_artifact)

    assert (
        candidate.main(
            [
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--output",
                "result",
            ]
        )
        == 0
    )
    assert calls == {
        "dataset_root": "dataset",
        "config_path": "config.json",
        "output_root": "result",
        "research_context_digest": None,
    }
    assert capsys.readouterr().out == "result\n"
