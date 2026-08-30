from __future__ import annotations

from copy import deepcopy

import pytest

from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)


def _digest(char: str) -> str:
    return char * 64


def test_materialization_identity_has_no_model_or_fit_identity() -> None:
    identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
        universe_manifest_digest=_digest("a"),
        model_config_digest=None,
        fit_provenance_digests=(),
    )

    assert identity.model_config_digest is None
    assert identity.fit_provenance_digests == ()
    assert identity.admission_authorization_digest is None
    assert len(identity.digest) == 64


def test_materialization_forbids_model_fit_and_admission_identity() -> None:
    with pytest.raises(ValueError, match="materialization.*model"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
            universe_manifest_digest=_digest("a"),
            model_config_digest=_digest("b"),
            fit_provenance_digests=(),
        )
    with pytest.raises(ValueError, match="materialization.*fit"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
            universe_manifest_digest=_digest("a"),
            model_config_digest=None,
            fit_provenance_digests=(_digest("c"),),
        )
    with pytest.raises(ValueError, match="materialization.*authorization"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
            universe_manifest_digest=_digest("a"),
            model_config_digest=None,
            fit_provenance_digests=(),
            admission_authorization_digest=_digest("d"),
        )


@pytest.mark.parametrize(
    "stage",
    (
        UniversalTradeRLRunStage.BASE_TRAINING,
        UniversalTradeRLRunStage.DEVELOPMENT_SELECTION,
        UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION,
    ),
)
def test_post_materialization_stages_require_model_config(
    stage: UniversalTradeRLRunStage,
) -> None:
    with pytest.raises(ValueError, match="model config"):
        UniversalTradeRLRunIdentity(
            stage=stage,
            universe_manifest_digest=_digest("a"),
            model_config_digest=None,
            fit_provenance_digests=(_digest("c"),),
            admission_authorization_digest=(
                _digest("d")
                if stage is UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION
                else None
            ),
        )


@pytest.mark.parametrize(
    "stage",
    (
        UniversalTradeRLRunStage.BASE_TRAINING,
        UniversalTradeRLRunStage.DEVELOPMENT_SELECTION,
        UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION,
    ),
)
def test_post_materialization_stages_require_fit_provenance(
    stage: UniversalTradeRLRunStage,
) -> None:
    with pytest.raises(ValueError, match="fit provenance"):
        UniversalTradeRLRunIdentity(
            stage=stage,
            universe_manifest_digest=_digest("a"),
            model_config_digest=_digest("b"),
            fit_provenance_digests=(),
            admission_authorization_digest=(
                _digest("d")
                if stage is UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION
                else None
            ),
        )


def test_admission_identity_requires_authorization() -> None:
    with pytest.raises(ValueError, match="authorization"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION,
            universe_manifest_digest=_digest("a"),
            model_config_digest=_digest("b"),
            fit_provenance_digests=(_digest("c"),),
        )


@pytest.mark.parametrize(
    "stage",
    (
        UniversalTradeRLRunStage.BASE_TRAINING,
        UniversalTradeRLRunStage.DEVELOPMENT_SELECTION,
    ),
)
def test_pre_admission_run_identity_forbids_authorization(
    stage: UniversalTradeRLRunStage,
) -> None:
    with pytest.raises(ValueError, match="forbid"):
        UniversalTradeRLRunIdentity(
            stage=stage,
            universe_manifest_digest=_digest("a"),
            model_config_digest=_digest("b"),
            fit_provenance_digests=(_digest("c"),),
            admission_authorization_digest=_digest("d"),
        )


def test_fit_provenance_digests_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.BASE_TRAINING,
            universe_manifest_digest=_digest("a"),
            model_config_digest=_digest("b"),
            fit_provenance_digests=(_digest("d"), _digest("c")),
        )
    with pytest.raises(ValueError, match="unique"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.BASE_TRAINING,
            universe_manifest_digest=_digest("a"),
            model_config_digest=_digest("b"),
            fit_provenance_digests=(_digest("c"), _digest("c")),
        )


def test_run_identity_round_trips_strict_payload() -> None:
    identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION,
        universe_manifest_digest=_digest("a"),
        model_config_digest=_digest("b"),
        fit_provenance_digests=(_digest("c"), _digest("e")),
        admission_authorization_digest=_digest("d"),
    )

    restored = UniversalTradeRLRunIdentity.from_payload(identity.to_payload())

    assert restored == identity
    assert restored.digest == identity.digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("universe_manifest_digest", _digest("f")),
        ("model_config_digest", _digest("f")),
        ("fit_provenance_digests", (_digest("c"), _digest("f"))),
        ("admission_authorization_digest", _digest("f")),
    ),
)
def test_run_identity_rejects_payload_tampering(field: str, value: object) -> None:
    identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION,
        universe_manifest_digest=_digest("a"),
        model_config_digest=_digest("b"),
        fit_provenance_digests=(_digest("c"), _digest("e")),
        admission_authorization_digest=_digest("d"),
    )
    payload = deepcopy(identity.to_payload())
    payload[field] = value

    with pytest.raises(ValueError, match="digest|identity|contract"):
        UniversalTradeRLRunIdentity.from_payload(payload)


def test_run_identity_rejects_unknown_payload_keys() -> None:
    identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.BASE_TRAINING,
        universe_manifest_digest=_digest("a"),
        model_config_digest=_digest("b"),
        fit_provenance_digests=(_digest("c"),),
    )
    payload = dict(identity.to_payload())
    payload["transfer_source"] = _digest("f")

    with pytest.raises(ValueError, match="exact keys"):
        UniversalTradeRLRunIdentity.from_payload(payload)
