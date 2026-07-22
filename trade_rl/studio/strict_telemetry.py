"""Fail-closed Studio telemetry stream discovery."""

from __future__ import annotations

from pathlib import Path

from trade_rl.studio.errors import ArtifactInvalid
from trade_rl.studio.telemetry import StudioTelemetryReader as _BaseTelemetryReader

_TELEMETRY_NAME = "training-telemetry.jsonl"


class StrictStudioTelemetryReader(_BaseTelemetryReader):
    """Reject multiple distinct telemetry files that claim one seed identity."""

    def _paths(self, job: object) -> dict[int, Path]:
        root = self._artifact_root(job)  # type: ignore[arg-type]
        streams: dict[int, Path] = {}
        for namespace in (".staging", "runs", "failed"):
            run_root = (root / namespace / job.run_id).resolve()  # type: ignore[attr-defined]
            try:
                run_root.relative_to(root)
            except ValueError as error:
                raise ArtifactInvalid(
                    "telemetry run path escapes artifact root"
                ) from error
            if not run_root.is_dir():
                continue
            for candidate in sorted(run_root.rglob(_TELEMETRY_NAME)):
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(run_root)
                except ValueError as error:
                    raise ArtifactInvalid(
                        "telemetry file escapes artifact root"
                    ) from error
                if not resolved.is_file() or candidate.is_symlink():
                    continue
                seed = self._seed_from_path(resolved, run_root=run_root)
                if seed is None:
                    continue
                previous = streams.get(seed)
                if previous is not None and previous != resolved:
                    raise ArtifactInvalid(
                        f"multiple telemetry streams claim seed {seed}"
                    )
                streams[seed] = resolved
        return streams


__all__ = ["StrictStudioTelemetryReader"]
