import json
from pathlib import Path

import pytest

from mars_lite.serving.bundle import build_manifest, load_bundle


def _release_eligibility() -> dict[str, object]:
    return {
        "eligible": True,
        "forced": False,
        "skipped_gates": [],
        "optimization_steps_skipped": [],
        "sealed_holdout_used": True,
        "required_gates": {
            "p0": "passed",
            "walk_forward": "passed",
            "gate2": "passed",
            "significance": "not_required",
        },
    }


def _release_risk() -> dict[str, object]:
    return {
        "guardrails": {},
        "pre_trade": {
            "max_leverage": 1.0,
            "max_single_weight": 0.5,
            "max_net_exposure": 1.0,
            "max_worst_case_notional": 100_000.0,
            "min_order_notional": 10.0,
            "symbol_liquidity_caps": {"BTCUSDT": 50_000.0},
            "forbidden_symbols": [],
        },
    }


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "model.zip").write_bytes(b"model-v1")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "v1",
                "git_sha": "a" * 40,
                "model_kind": "single",
                "symbols": ["BTCUSDT"],
                "observation_schema_version": 1,
                "observation_progress_mode": "zero",
                "observation_dim": 5,
                "run_config": {},
                "release_eligibility": _release_eligibility(),
            }
        ),
        encoding="utf-8",
    )
    (root / "preprocessing.json").write_text(
        '{"feature_names":["ret"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true],"post_mask_dim":1}',
        encoding="utf-8",
    )
    (root / "risk.json").write_text(json.dumps(_release_risk()), encoding="utf-8")
    return root


def _rewrite_json(root: Path, filename: str, mutate) -> None:
    path = root / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_metadata(root: Path, **changes) -> None:
    _rewrite_json(root, "metadata.json", lambda metadata: metadata.update(changes))


def test_bundle_digest_is_deterministic_and_loadable(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    first = build_manifest(root)
    second = build_manifest(root)
    assert first.bundle_digest == second.bundle_digest
    loaded = load_bundle(root)
    assert loaded.version == "v1"
    assert loaded.git_sha == "a" * 40
    assert loaded.release_eligibility["eligible"] is True
    assert loaded.bundle_digest == first.bundle_digest
    assert loaded.model_path == root / "model.zip"


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    build_manifest(root)
    (root / "model.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(root)


def test_feature_mask_dimension_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "preprocessing.json").write_text(
        '{"feature_names":["a","b"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true,false],"post_mask_dim":1}',
        encoding="utf-8",
    )
    build_manifest(root)
    with pytest.raises(ValueError, match="post_mask_dim"):
        load_bundle(root)


def test_zero_mask_preserves_feature_dimension(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "preprocessing.json").write_text(
        '{"feature_names":["a","b"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true,false],"post_mask_dim":2}',
        encoding="utf-8",
    )
    _rewrite_metadata(root, observation_dim=6)
    build_manifest(root)
    loaded = load_bundle(root)
    assert loaded.preprocessing["post_mask_dim"] == 2


def test_episode_progress_bundle_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, observation_progress_mode="episode")
    build_manifest(root)
    with pytest.raises(ValueError, match="observation_progress_mode"):
        load_bundle(root)


def test_observation_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, observation_dim=6)
    build_manifest(root)
    with pytest.raises(ValueError, match="observation_dim"):
        load_bundle(root)


def test_missing_or_mismatched_model_kind_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, model_kind="ensemble")
    build_manifest(root)
    with pytest.raises(ValueError, match="ensemble model_kind"):
        load_bundle(root)


def test_ensemble_requires_only_seed_zip_members(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "model.zip").unlink()
    ensemble = root / "ensemble"
    ensemble.mkdir()
    (ensemble / "seed_0.zip").write_bytes(b"seed")
    (ensemble / "notes.txt").write_text("unexpected", encoding="utf-8")
    _rewrite_metadata(root, model_kind="ensemble")
    build_manifest(root)
    with pytest.raises(ValueError, match="ensemble model_kind"):
        load_bundle(root)


def test_invalid_git_sha_is_rejected_before_manifest_build(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, git_sha="abc123")
    with pytest.raises(ValueError, match="git_sha"):
        build_manifest(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligible", False),
        ("forced", True),
        ("skipped_gates", ["p0"]),
        ("sealed_holdout_used", False),
    ],
)
def test_bundle_rejects_ineligible_release_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    root = _candidate(tmp_path)

    def mutate(metadata: dict[str, object]) -> None:
        eligibility = metadata["release_eligibility"]
        assert isinstance(eligibility, dict)
        eligibility[field] = value

    _rewrite_json(root, "metadata.json", mutate)
    build_manifest(root)
    with pytest.raises(ValueError, match="release eligibility|sealed holdout|forced"):
        load_bundle(root)


def test_bundle_rejects_failed_required_gate(tmp_path: Path) -> None:
    root = _candidate(tmp_path)

    def mutate(metadata: dict[str, object]) -> None:
        eligibility = metadata["release_eligibility"]
        assert isinstance(eligibility, dict)
        gates = eligibility["required_gates"]
        assert isinstance(gates, dict)
        gates["gate2"] = "failed"

    _rewrite_json(root, "metadata.json", mutate)
    build_manifest(root)
    with pytest.raises(ValueError, match="gate2"):
        load_bundle(root)


def test_bundle_rejects_incomplete_release_risk(tmp_path: Path) -> None:
    root = _candidate(tmp_path)

    def mutate(risk: dict[str, object]) -> None:
        pre_trade = risk["pre_trade"]
        assert isinstance(pre_trade, dict)
        del pre_trade["max_leverage"]

    _rewrite_json(root, "risk.json", mutate)
    build_manifest(root)
    with pytest.raises(ValueError, match="missing required release fields"):
        load_bundle(root)
