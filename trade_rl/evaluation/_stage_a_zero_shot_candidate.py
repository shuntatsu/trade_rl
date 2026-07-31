"""Candidate identity contract for Stage A zero-shot evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation._stage_a_zero_shot_contract_helpers import (
    STAGE_A_CANDIDATE_SCHEMA,
    _non_negative_int,
)


@dataclass(frozen=True, slots=True)
class StageACandidate:
    """One exact trained candidate and its retained per-seed checkpoints."""

    candidate_id: str
    candidate_config_digest: str
    final_training_completion_digest: str
    policy_identity: str
    checkpoint_digests: tuple[tuple[int, str], ...]
    schema_version: str = STAGE_A_CANDIDATE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_CANDIDATE_SCHEMA:
            raise ValueError("unsupported Stage A candidate schema")
        candidate_id = require_non_empty(
            self.candidate_id, field="stage_a_candidate.candidate_id"
        )
        for field, value in (
            ("candidate_config_digest", self.candidate_config_digest),
            ("final_training_completion_digest", self.final_training_completion_digest),
            ("policy_identity", self.policy_identity),
        ):
            require_sha256(value, field=f"stage_a_candidate.{field}")
        if not self.checkpoint_digests:
            raise ValueError("Stage A candidate checkpoints must not be empty")
        normalized: list[tuple[int, str]] = []
        seen: set[int] = set()
        for seed, digest in self.checkpoint_digests:
            resolved_seed = _non_negative_int(
                seed, field="stage_a_candidate.checkpoint_seed"
            )
            if resolved_seed in seen:
                raise ValueError("Stage A candidate checkpoint seeds must be unique")
            require_sha256(digest, field="stage_a_candidate.checkpoint_digest")
            seen.add(resolved_seed)
            normalized.append((resolved_seed, digest))
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "checkpoint_digests", tuple(sorted(normalized)))
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A candidate digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        candidate_config_digest: str,
        final_training_completion_digest: str,
        policy_identity: str,
        checkpoint_digests: tuple[tuple[int, str], ...],
    ) -> StageACandidate:
        return cls(
            candidate_id=candidate_id,
            candidate_config_digest=candidate_config_digest,
            final_training_completion_digest=final_training_completion_digest,
            policy_identity=policy_identity,
            checkpoint_digests=checkpoint_digests,
        )

    def checkpoint_digest(self, seed: int) -> str:
        resolved = _non_negative_int(seed, field="stage_a_candidate.seed")
        try:
            return dict(self.checkpoint_digests)[resolved]
        except KeyError as error:
            raise ValueError(
                "Stage A candidate checkpoint seed is not declared"
            ) from error

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "candidate_id": self.candidate_id,
            "checkpoint_digests": self.checkpoint_digests,
            "final_training_completion_digest": self.final_training_completion_digest,
            "policy_identity": self.policy_identity,
            "schema_version": self.schema_version,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}
