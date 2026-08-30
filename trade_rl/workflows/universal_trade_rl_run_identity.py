"""Versioned future RL run identity for Universal Trade RL U0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

UNIVERSAL_TRADE_RL_RUN_IDENTITY_SCHEMA: Final = "universal_trade_rl_run_identity_v1"
_RUN_IDENTITY_KEYS: Final = (
    "schema_version",
    "stage",
    "universe_manifest_digest",
    "model_config_digest",
    "fit_provenance_digests",
    "admission_authorization_digest",
    "artifact_digest",
)


class UniversalTradeRLRunStage(str, Enum):
    """Versioned U0 execution stages whose inputs must remain identity-bound."""

    UNIVERSE_MATERIALIZATION = "universe_materialization"
    BASE_TRAINING = "base_training"
    DEVELOPMENT_SELECTION = "development_selection"
    ZERO_SHOT_ADMISSION = "zero_shot_admission"


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object with exact keys")
    result = {str(key): item for key, item in value.items()}
    if set(result) != set(_RUN_IDENTITY_KEYS) or len(result) != len(
        _RUN_IDENTITY_KEYS
    ):
        raise ValueError(f"{field} must use exact keys")
    return result


def _fit_digests(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError("run identity fit provenance digests must be a tuple")
    if len(set(values)) != len(values):
        raise ValueError("run identity fit provenance digests must be unique")
    if values != tuple(sorted(values)):
        raise ValueError("run identity fit provenance digests must be sorted")
    for digest in values:
        require_sha256(digest, field="run identity fit provenance digest")
    return values


@dataclass(frozen=True, slots=True)
class UniversalTradeRLRunIdentity:
    """Immutable stage identity for U0 materialization and later RL research runs."""

    stage: UniversalTradeRLRunStage
    universe_manifest_digest: str
    model_config_digest: str | None
    fit_provenance_digests: tuple[str, ...]
    admission_authorization_digest: str | None = None
    schema_version: str = UNIVERSAL_TRADE_RL_RUN_IDENTITY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stage, UniversalTradeRLRunStage):
            raise TypeError("run identity stage is invalid")
        if self.schema_version != UNIVERSAL_TRADE_RL_RUN_IDENTITY_SCHEMA:
            raise ValueError("unsupported Universal Trade RL run identity schema")
        require_sha256(
            self.universe_manifest_digest,
            field="run identity universe manifest digest",
        )
        fit_digests = _fit_digests(self.fit_provenance_digests)
        object.__setattr__(self, "fit_provenance_digests", fit_digests)

        if self.stage is UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION:
            if self.model_config_digest is not None:
                raise ValueError("materialization must not bind a model config")
            if fit_digests:
                raise ValueError("materialization must not bind fit provenance")
            if self.admission_authorization_digest is not None:
                raise ValueError(
                    "materialization must not bind Admission authorization"
                )
        else:
            if self.model_config_digest is None:
                raise ValueError("post-materialization run identity requires model config")
            require_sha256(
                self.model_config_digest,
                field="run identity model config digest",
            )
            if not fit_digests:
                raise ValueError(
                    "post-materialization run identity requires fit provenance"
                )
            if self.stage is UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION:
                if self.admission_authorization_digest is None:
                    raise ValueError("zero-shot Admission identity requires authorization")
                require_sha256(
                    self.admission_authorization_digest,
                    field="run identity Admission authorization digest",
                )
            elif self.admission_authorization_digest is not None:
                raise ValueError(
                    "pre-Admission run identity must forbid Admission authorization"
                )

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="run identity artifact digest")
            if self.digest != expected:
                raise ValueError("Universal Trade RL run identity digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "universe_manifest_digest": self.universe_manifest_digest,
            "model_config_digest": self.model_config_digest,
            "fit_provenance_digests": self.fit_provenance_digests,
            "admission_authorization_digest": self.admission_authorization_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLRunIdentity:
        values = _mapping(payload, field="run identity")
        if values["schema_version"] != UNIVERSAL_TRADE_RL_RUN_IDENTITY_SCHEMA:
            raise ValueError("unsupported Universal Trade RL run identity schema")
        stage_value = values["stage"]
        if not isinstance(stage_value, str):
            raise ValueError("run identity stage contract is invalid")
        try:
            stage = UniversalTradeRLRunStage(stage_value)
        except ValueError as error:
            raise ValueError("run identity stage contract is invalid") from error

        universe_digest = values["universe_manifest_digest"]
        model_digest = values["model_config_digest"]
        authorization_digest = values["admission_authorization_digest"]
        artifact_digest = values["artifact_digest"]
        if not isinstance(universe_digest, str) or not isinstance(artifact_digest, str):
            raise ValueError("run identity digest contract is invalid")
        if model_digest is not None and not isinstance(model_digest, str):
            raise ValueError("run identity model config contract is invalid")
        if authorization_digest is not None and not isinstance(
            authorization_digest, str
        ):
            raise ValueError("run identity authorization contract is invalid")

        raw_fit = values["fit_provenance_digests"]
        if not isinstance(raw_fit, Sequence) or isinstance(raw_fit, (str, bytes)):
            raise ValueError("run identity fit provenance contract is invalid")
        fit_values = tuple(raw_fit)
        if any(not isinstance(item, str) for item in fit_values):
            raise ValueError("run identity fit provenance contract is invalid")

        return cls(
            stage=stage,
            universe_manifest_digest=universe_digest,
            model_config_digest=model_digest,
            fit_provenance_digests=fit_values,
            admission_authorization_digest=authorization_digest,
            schema_version=UNIVERSAL_TRADE_RL_RUN_IDENTITY_SCHEMA,
            digest=artifact_digest,
        )


__all__ = [
    "UNIVERSAL_TRADE_RL_RUN_IDENTITY_SCHEMA",
    "UniversalTradeRLRunIdentity",
    "UniversalTradeRLRunStage",
]
