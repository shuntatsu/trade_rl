from __future__ import annotations

from scripts.materialize_universal_runtime import _parser


def test_runtime_cli_requires_explicit_nonsecret_inputs() -> None:
    args = _parser().parse_args(
        [
            "--postgres-url",
            "postgresql://db",
            "--frozen-metadata-root",
            "metadata",
            "--instrument-artifact-root",
            "artifacts/instruments",
            "--dataset-artifact-root",
            "artifacts/datasets",
            "--normalizer-artifact-root",
            "artifacts/normalizer",
            "--runtime-manifest",
            "artifacts/runtime.json",
        ]
    )

    assert args.postgres_url == "postgresql://db"
    assert args.runtime_manifest.name == "runtime.json"
