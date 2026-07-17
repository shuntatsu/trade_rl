"""Immutable authorization for one walk-forward-selected final training run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256

SELECTION_AUTHORIZATION_SCHEMA = "selection_authorization_v1"


def _seeds(value: tuple[int, ...]) -> tuple[int, ...]:
    seeds = tuple(value)
    if (
        len(seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("selection authorization requires unique non-negative seeds")
    return seeds


@dataclass(frozen=True, slots=True)
class SelectionAuthorization:
    authorization_digest: str
    walk_forward_run_digest: str
    gate_evidence_digest: str
    dataset_id: str
    selected_configuration: str
    candidate_config_digest: str
    seeds: tuple[int, ...]
    schema_version: str = SELECTION_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("authorization_digest", self.authorization_digest),
            ("walk_forward_run_digest", self.walk_forward_run_digest),
            ("gate_evidence_digest", self.gate_evidence_digest),
            ("dataset_id", self.dataset_id),
            ("candidate_config_digest", self.candidate_config_digest),
        ):
            require_sha256(value, field=name)
        require_non_empty(self.selected_configuration, field="selected_configuration")
        object.__setattr__(self, "seeds", _seeds(self.seeds))
        if self.schema_version != SELECTION_AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported selection authorization schema")
        if self.authorization_digest != content_digest(self.digest_payload()):
            raise ValueError("selection authorization digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "dataset_id": self.dataset_id,
            "gate_evidence_digest": self.gate_evidence_digest,
            "schema_version": self.schema_version,
            "seeds": self.seeds,
            "selected_configuration": self.selected_configuration,
            "walk_forward_run_digest": self.walk_forward_run_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        walk_forward_run_digest: str,
        gate_evidence_digest: str,
        dataset_id: str,
        selected_configuration: str,
        candidate_config_digest: str,
        seeds: tuple[int, ...],
    ) -> SelectionAuthorization:
        resolved_seeds = _seeds(seeds)
        payload = {
            "candidate_config_digest": candidate_config_digest,
            "dataset_id": dataset_id,
            "gate_evidence_digest": gate_evidence_digest,
            "schema_version": SELECTION_AUTHORIZATION_SCHEMA,
            "seeds": resolved_seeds,
            "selected_configuration": selected_configuration,
            "walk_forward_run_digest": walk_forward_run_digest,
        }
        return cls(
            authorization_digest=content_digest(payload),
            walk_forward_run_digest=walk_forward_run_digest,
            gate_evidence_digest=gate_evidence_digest,
            dataset_id=dataset_id,
            selected_configuration=selected_configuration,
            candidate_config_digest=candidate_config_digest,
            seeds=resolved_seeds,
        )

    def verify(
        self,
        *,
        dataset_id: str,
        candidate_config_digest: str,
        seeds: tuple[int, ...],
    ) -> None:
        if self.dataset_id != dataset_id:
            raise ValueError("selection authorization dataset identity mismatch")
        if self.candidate_config_digest != candidate_config_digest:
            raise ValueError("selection authorization candidate identity mismatch")
        if self.seeds != _seeds(seeds):
            raise ValueError("selection authorization seed set mismatch")


def write_selection_authorization(
    path: str | Path,
    authorization: SelectionAuthorization,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(asdict(authorization)))
    temporary.replace(output)
    return output


def load_selection_authorization(path: str | Path) -> SelectionAuthorization:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("selection authorization must be an object")
    try:
        raw_seeds = raw["seeds"]
        if not isinstance(raw_seeds, list):
            raise ValueError("selection authorization seeds must be a list")
        seeds = tuple(int(seed) for seed in raw_seeds)
        return SelectionAuthorization(
            authorization_digest=str(raw["authorization_digest"]),
            walk_forward_run_digest=str(raw["walk_forward_run_digest"]),
            gate_evidence_digest=str(raw["gate_evidence_digest"]),
            dataset_id=str(raw["dataset_id"]),
            selected_configuration=str(raw["selected_configuration"]),
            candidate_config_digest=str(raw["candidate_config_digest"]),
            seeds=seeds,
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection authorization is invalid") from error


__all__ = [
    "SELECTION_AUTHORIZATION_SCHEMA",
    "SelectionAuthorization",
    "load_selection_authorization",
    "write_selection_authorization",
]
