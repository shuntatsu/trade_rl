from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from trade_rl.artifacts.hashing import content_digest


def _digest(label: str) -> str:
    return content_digest(label)


def test_build_universal_bindings_is_train_only_and_metadata_bound() -> None:
    from trade_rl.workflows.universal_training_runner import build_universal_bindings

    datasets = {
        "AAAUSDT": SimpleNamespace(dataset_id=_digest("dataset:A")),
        "BBBUSDT": SimpleNamespace(dataset_id=_digest("dataset:B")),
    }
    contracts = {
        "AAAUSDT": SimpleNamespace(
            canonical_payload=lambda: {"symbol": "AAAUSDT", "version": 1}
        ),
        "BBBUSDT": SimpleNamespace(
            canonical_payload=lambda: {"symbol": "BBBUSDT", "version": 1}
        ),
    }
    catalog = SimpleNamespace(
        per_symbol_metadata_digests=(
            ("AAAUSDT", _digest("metadata:A")),
            ("BBBUSDT", _digest("metadata:B")),
            ("VALIDUSDT", _digest("metadata:V")),
        )
    )

    bindings = build_universal_bindings(
        datasets=datasets,
        contracts=contracts,
        catalog=catalog,
        train_symbols=("AAAUSDT", "BBBUSDT"),
    )

    assert tuple(item.concrete_symbol for item in bindings) == (
        "AAAUSDT",
        "BBBUSDT",
    )
    assert all(item.split == "train" for item in bindings)
    assert bindings[0].execution_metadata_digest == _digest("metadata:A")
    assert bindings[1].execution_metadata_digest == _digest("metadata:B")


def test_train_universal_seeds_writes_manifest_from_exact_backend_outputs(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.universal_training_runner import train_universal_seeds

    calls: list[tuple[int, Path]] = []

    class Backend:
        def train(self, *, seed: int, config: object, output_path: Path) -> object:
            del config
            calls.append((seed, output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(f"policy:{seed}".encode())
            return SimpleNamespace(
                checkpoint_path=output_path,
                environment_digest=_digest("universal-environment"),
                architecture_digest=_digest(f"architecture:{seed}"),
                actual_timesteps=123,
            )

    runtime = SimpleNamespace(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        instrument_context_schema_digest=_digest("context-schema"),
        training_contract_digest=_digest("training-contract"),
        pretraining_artifact_digest=_digest("pretraining"),
        routed_environment_factory=object(),
    )
    training = SimpleNamespace(
        seeds=(3, 5),
        digest_payload=lambda: {"schema_version": "training-test-v1"},
    )

    manifest = train_universal_seeds(
        runtime=runtime,
        training=training,
        backend=Backend(),
        output_root=tmp_path,
        architecture_name="u_medium_direct",
    )

    assert calls == [
        (3, tmp_path / "seed-3" / "policy.zip"),
        (5, tmp_path / "seed-5" / "policy.zip"),
    ]
    assert (tmp_path / "universal-training.json").is_file()
    assert manifest["train_symbols"] == ["AAAUSDT", "BBBUSDT"]
    assert manifest["research_success"] is False
    assert len(manifest["run_digest"]) == 64
    assert [item["seed"] for item in manifest["members"]] == [3, 5]
