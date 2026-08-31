"""Durable hash-chained intermediate diagnostics for Causal Alpha V7."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

_HEADER_SCHEMA: Final = "causal_alpha_v7_checkpoint_header_v1"
_EVENT_SCHEMA: Final = "causal_alpha_v7_checkpoint_event_v1"
_STAGES: Final = ("signal", "selection", "admission")


class CausalAlphaV7CheckpointWriter:
    """Append fsynced diagnostic events with one tamper-evident hash chain."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_manifest_digest: str,
        config_digest: str,
        generator_code_digest: str,
    ) -> None:
        for name, value in (
            ("run_manifest_digest", run_manifest_digest),
            ("config_digest", config_digest),
            ("generator_code_digest", generator_code_digest),
        ):
            require_sha256(value, field=f"V7 checkpoint {name}")
        self.path = Path(root) / "causal-alpha-v7-checkpoint.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header: dict[str, object] = {
            "config_digest": config_digest,
            "generator_code_digest": generator_code_digest,
            "promotion_eligible": False,
            "research_only": True,
            "run_manifest_digest": run_manifest_digest,
            "schema_version": _HEADER_SCHEMA,
        }
        header["artifact_digest"] = content_digest(header)
        with self.path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(header, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._previous_digest = str(header["artifact_digest"])
        self._sequence = 0
        self._last_cutoff: dict[str, int] = {}

    def append(
        self,
        *,
        stage: str,
        cutoff: int,
        diagnostics: Mapping[str, object],
    ) -> str:
        if stage not in _STAGES:
            raise ValueError("V7 checkpoint stage is invalid")
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
            raise ValueError("V7 checkpoint cutoff is invalid")
        previous_cutoff = self._last_cutoff.get(stage)
        if previous_cutoff is not None and cutoff <= previous_cutoff:
            raise ValueError("V7 checkpoint cutoffs must strictly increase per stage")
        diagnostic_payload = dict(diagnostics)
        self._sequence += 1
        body: dict[str, object] = {
            "cutoff": cutoff,
            "diagnostics": diagnostic_payload,
            "previous_digest": self._previous_digest,
            "promotion_eligible": False,
            "schema_version": _EVENT_SCHEMA,
            "sequence": self._sequence,
            "stage": stage,
        }
        digest = content_digest(body)
        payload = {**body, "artifact_digest": digest}
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            self._sequence -= 1
            raise ValueError(
                "V7 checkpoint diagnostics are not JSON-serializable"
            ) from error
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._previous_digest = digest
        self._last_cutoff[stage] = cutoff
        return digest


__all__ = ["CausalAlphaV7CheckpointWriter"]
