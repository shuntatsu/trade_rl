from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "trade_rl/telemetry/training.py",
    "    terminated: bool\n    truncated: bool\n    schema_version: str = TELEMETRY_SCHEMA_VERSION\n",
    "    terminated: bool\n    truncated: bool\n    episode_id: int | None = None\n    schema_version: str = TELEMETRY_SCHEMA_VERSION\n",
)
replace_once(
    "trade_rl/telemetry/training.py",
    "            raise ValueError(\"market_index is invalid\")\n        try:\n",
    "            raise ValueError(\"market_index is invalid\")\n"
    "        if self.episode_id is not None and (\n"
    "            isinstance(self.episode_id, bool)\n"
    "            or not isinstance(self.episode_id, int)\n"
    "            or self.episode_id < 0\n"
    "        ):\n"
    "            raise ValueError(\"episode_id is invalid\")\n"
    "        try:\n",
)
replace_once(
    "trade_rl/telemetry/training.py",
    '            "environment_id": self.environment_id,\n            "event_type": self.event_type,\n',
    '            "environment_id": self.environment_id,\n'
    '            "episode_id": self.episode_id,\n'
    '            "event_type": self.event_type,\n',
)
replace_once(
    "trade_rl/telemetry/training.py",
    '            environment_id=_required_int(raw, "environment_id"),\n'
    '            event_type=cast(TelemetryEventType, event_type),\n',
    '            environment_id=_required_int(raw, "environment_id"),\n'
    '            episode_id=_optional_int(raw, "episode_id"),\n'
    '            event_type=cast(TelemetryEventType, event_type),\n',
)

replace_once(
    "trade_rl/rl/training_telemetry.py",
    "        self.sequence = training_telemetry_status(path).last_sequence\n"
    "        self.disabled = False\n"
    "        self._previous_weights: dict[int, tuple[float, ...]] = {}\n"
    "        self._previous_close: dict[int, float] = {}\n\n"
    "    def _weights(\n",
    "        self.sequence = training_telemetry_status(path).last_sequence\n"
    "        self.disabled = False\n"
    "        self._previous_weights: dict[int, tuple[float, ...]] = {}\n"
    "        self._previous_close: dict[int, float] = {}\n"
    "        self._episode_ids: dict[int, int] = {}\n"
    "        self._next_episode_id = self.sequence + 1\n\n"
    "    def _episode_id(self, environment_id: int) -> int:\n"
    "        current = self._episode_ids.get(environment_id)\n"
    "        if current is not None:\n"
    "            return current\n"
    "        assigned = self._next_episode_id\n"
    "        self._next_episode_id += 1\n"
    "        self._episode_ids[environment_id] = assigned\n"
    "        return assigned\n\n"
    "    def _finish_episode(self, environment_id: int) -> None:\n"
    "        self._episode_ids.pop(environment_id, None)\n"
    "        self._previous_weights.pop(environment_id, None)\n"
    "        self._previous_close.pop(environment_id, None)\n\n"
    "    def _weights(\n",
)
replace_once(
    "trade_rl/rl/training_telemetry.py",
    "                reward_fallback = (\n"
    "                    float(reward_rows[environment_id])\n"
    "                    if environment_id < reward_rows.size\n"
    "                    else None\n"
    "                )\n"
    "                self.writer.append(\n",
    "                reward_fallback = (\n"
    "                    float(reward_rows[environment_id])\n"
    "                    if environment_id < reward_rows.size\n"
    "                    else None\n"
    "                )\n"
    "                episode_id = self._episode_id(environment_id)\n"
    "                self.writer.append(\n",
)
replace_once(
    "trade_rl/rl/training_telemetry.py",
    "                        emergency_deleverage=bool(info.get(\"emergency_deleverage\")),\n"
    "                        terminated=bool(info.get(\"hybrid_terminated\")) or done,\n"
    "                        truncated=bool(info.get(\"TimeLimit.truncated\")),\n"
    "                    )\n"
    "                )\n"
    "                emitted += 1\n",
    "                        emergency_deleverage=bool(info.get(\"emergency_deleverage\")),\n"
    "                        terminated=bool(info.get(\"hybrid_terminated\")) or done,\n"
    "                        truncated=bool(info.get(\"TimeLimit.truncated\")),\n"
    "                        episode_id=episode_id,\n"
    "                    )\n"
    "                )\n"
    "                emitted += 1\n"
    "                if (\n"
    "                    done\n"
    "                    or bool(info.get(\"hybrid_terminated\"))\n"
    "                    or bool(info.get(\"TimeLimit.truncated\"))\n"
    "                ):\n"
    "                    self._finish_episode(environment_id)\n",
)

replace_once(
    "trade_rl/studio/telemetry.py",
    "    seed: int = Field(ge=0)\n    environment_id: int = Field(ge=0)\n    event_type: Literal[\n",
    "    seed: int = Field(ge=0)\n"
    "    environment_id: int = Field(ge=0)\n"
    "    episode_id: int | None = Field(default=None, ge=0)\n"
    "    event_type: Literal[\n",
)

replace_once(
    "studio/src/data/types.ts",
    "  seed: number\n  environmentId: number\n  eventType: TelemetryEventType\n",
    "  seed: number\n  environmentId: number\n  episodeId: number | null\n  eventType: TelemetryEventType\n",
)
replace_once(
    "studio/src/live/telemetryGuards.ts",
    "    && isNonNegativeInteger(value.environmentId)\n"
    "    && typeof value.eventType === 'string' && eventTypes.has(value.eventType)\n",
    "    && isNonNegativeInteger(value.environmentId)\n"
    "    && (value.episodeId === null || isNonNegativeInteger(value.episodeId))\n"
    "    && typeof value.eventType === 'string' && eventTypes.has(value.eventType)\n",
)
replace_once(
    "studio/src/live/useTrainingTelemetry.test.tsx",
    "    seed: 7,\n    environmentId: 0,\n    eventType: 'rollout',\n",
    "    seed: 7,\n    environmentId: 0,\n    episodeId: 1,\n    eventType: 'rollout',\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.test.tsx",
    "    seed,\n    environmentId: 0,\n    eventType: sequence === 2 ? 'position' : 'rollout',\n",
    "    seed,\n    environmentId: 0,\n    episodeId: 1,\n    eventType: sequence === 2 ? 'position' : 'rollout',\n",
)

Path("studio/src/live/telemetryTracks.ts").write_text(
    """import type { TrainingTelemetryRecord } from '../data/types'\n\n"
    "export interface TelemetryTrack {\n"
    "  key: string\n"
    "  environmentId: number\n"
    "  episodeId: number | null\n"
    "  legacyOrdinal: number | null\n"
    "  records: TrainingTelemetryRecord[]\n"
    "  firstSequence: number\n"
    "  lastSequence: number\n"
    "  ended: boolean\n"
    "  inferred: boolean\n"
    "}\n\n"
    "interface MutableTelemetryTrack {\n"
    "  key: string\n"
    "  environmentId: number\n"
    "  episodeId: number | null\n"
    "  legacyOrdinal: number | null\n"
    "  records: TrainingTelemetryRecord[]\n"
    "  inferred: boolean\n"
    "}\n\n"
    "interface LegacyState {\n"
    "  key: string\n"
    "  ordinal: number\n"
    "  previous: TrainingTelemetryRecord\n"
    "}\n\n"
    "function ended(record: TrainingTelemetryRecord): boolean {\n"
    "  return record.eventType === 'episode_end' || record.terminated || record.truncated\n"
    "}\n\n"
    "function legacyBoundary(\n"
    "  previous: TrainingTelemetryRecord,\n"
    "  current: TrainingTelemetryRecord,\n"
    "): boolean {\n"
    "  if (ended(previous)) return true\n"
    "  if (current.environmentStep < previous.environmentStep) return true\n"
    "  return previous.marketIndex !== null\n"
    "    && current.marketIndex !== null\n"
    "    && current.marketIndex < previous.marketIndex\n"
    "}\n\n"
    "export function deriveTelemetryTracks(\n"
    "  records: TrainingTelemetryRecord[],\n"
    "): TelemetryTrack[] {\n"
    "  const ordered = [...records].sort((left, right) => left.sequence - right.sequence)\n"
    "  const tracks = new Map<string, MutableTelemetryTrack>()\n"
    "  const activeLegacy = new Map<number, LegacyState>()\n"
    "  const nextLegacyOrdinal = new Map<number, number>()\n\n"
    "  for (const record of ordered) {\n"
    "    let key: string\n"
    "    let legacyOrdinal: number | null = null\n"
    "    let inferred = false\n\n"
    "    if (record.episodeId !== null) {\n"
    "      activeLegacy.delete(record.environmentId)\n"
    "      key = `explicit:${record.environmentId}:${record.episodeId}`\n"
    "    } else {\n"
    "      inferred = true\n"
    "      const active = activeLegacy.get(record.environmentId)\n"
    "      if (!active || legacyBoundary(active.previous, record)) {\n"
    "        legacyOrdinal = nextLegacyOrdinal.get(record.environmentId) ?? 0\n"
    "        nextLegacyOrdinal.set(record.environmentId, legacyOrdinal + 1)\n"
    "        key = `legacy:${record.environmentId}:${legacyOrdinal}`\n"
    "      } else {\n"
    "        key = active.key\n"
    "        legacyOrdinal = active.ordinal\n"
    "      }\n"
    "      activeLegacy.set(record.environmentId, {\n"
    "        key,\n"
    "        ordinal: legacyOrdinal,\n"
    "        previous: record,\n"
    "      })\n"
    "    }\n\n"
    "    const existing = tracks.get(key)\n"
    "    if (existing) {\n"
    "      existing.records.push(record)\n"
    "    } else {\n"
    "      tracks.set(key, {\n"
    "        key,\n"
    "        environmentId: record.environmentId,\n"
    "        episodeId: record.episodeId,\n"
    "        legacyOrdinal,\n"
    "        records: [record],\n"
    "        inferred,\n"
    "      })\n"
    "    }\n"
    "  }\n\n"
    "  return [...tracks.values()]\n"
    "    .map((track) => ({\n"
    "      ...track,\n"
    "      firstSequence: track.records[0].sequence,\n"
    "      lastSequence: track.records.at(-1)?.sequence ?? track.records[0].sequence,\n"
    "      ended: ended(track.records.at(-1) ?? track.records[0]),\n"
    "    }))\n"
    "    .sort((left, right) => left.lastSequence - right.lastSequence)\n"
    "}\n\n"
    "export function selectTelemetryTrack(\n"
    "  records: TrainingTelemetryRecord[],\n"
    "  cursorSequence: number | null,\n"
    "): TelemetryTrack | null {\n"
    "  const tracks = deriveTelemetryTracks(records)\n"
    "  if (cursorSequence !== null) {\n"
    "    const containing = tracks.find((track) =>\n"
    "      track.records.some((record) => record.sequence === cursorSequence))\n"
    "    if (containing) return containing\n"
    "  }\n"
    "  return tracks.at(-1) ?? null\n"
    "}\n",
    encoding="utf-8",
)
