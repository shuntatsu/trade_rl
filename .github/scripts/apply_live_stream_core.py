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
    """import type { TrainingTelemetryRecord } from '../data/types'

export interface TelemetryTrack {
  key: string
  environmentId: number
  episodeId: number | null
  legacyOrdinal: number | null
  records: TrainingTelemetryRecord[]
  firstSequence: number
  lastSequence: number
  ended: boolean
  inferred: boolean
}

interface MutableTelemetryTrack {
  key: string
  environmentId: number
  episodeId: number | null
  legacyOrdinal: number | null
  records: TrainingTelemetryRecord[]
  inferred: boolean
}

interface LegacyState {
  key: string
  ordinal: number
  previous: TrainingTelemetryRecord
}

function ended(record: TrainingTelemetryRecord): boolean {
  return record.eventType === 'episode_end' || record.terminated || record.truncated
}

function legacyBoundary(
  previous: TrainingTelemetryRecord,
  current: TrainingTelemetryRecord,
): boolean {
  if (ended(previous)) return true
  if (current.environmentStep < previous.environmentStep) return true
  return previous.marketIndex !== null
    && current.marketIndex !== null
    && current.marketIndex < previous.marketIndex
}

export function deriveTelemetryTracks(
  records: TrainingTelemetryRecord[],
): TelemetryTrack[] {
  const ordered = [...records].sort((left, right) => left.sequence - right.sequence)
  const tracks = new Map<string, MutableTelemetryTrack>()
  const activeLegacy = new Map<number, LegacyState>()
  const nextLegacyOrdinal = new Map<number, number>()

  for (const record of ordered) {
    let key: string
    let legacyOrdinal: number | null = null
    let inferred = false

    if (record.episodeId !== null) {
      activeLegacy.delete(record.environmentId)
      key = `explicit:${record.environmentId}:${record.episodeId}`
    } else {
      inferred = true
      const active = activeLegacy.get(record.environmentId)
      if (!active || legacyBoundary(active.previous, record)) {
        const ordinal = nextLegacyOrdinal.get(record.environmentId) ?? 0
        nextLegacyOrdinal.set(record.environmentId, ordinal + 1)
        legacyOrdinal = ordinal
        key = `legacy:${record.environmentId}:${ordinal}`
      } else {
        key = active.key
        legacyOrdinal = active.ordinal
      }
      activeLegacy.set(record.environmentId, {
        key,
        ordinal: legacyOrdinal,
        previous: record,
      })
    }

    const existing = tracks.get(key)
    if (existing) {
      existing.records.push(record)
    } else {
      tracks.set(key, {
        key,
        environmentId: record.environmentId,
        episodeId: record.episodeId,
        legacyOrdinal,
        records: [record],
        inferred,
      })
    }
  }

  return [...tracks.values()]
    .map((track) => ({
      ...track,
      firstSequence: track.records[0].sequence,
      lastSequence: track.records.at(-1)?.sequence ?? track.records[0].sequence,
      ended: ended(track.records.at(-1) ?? track.records[0]),
    }))
    .sort((left, right) => left.lastSequence - right.lastSequence)
}

export function selectTelemetryTrack(
  records: TrainingTelemetryRecord[],
  cursorSequence: number | null,
): TelemetryTrack | null {
  const tracks = deriveTelemetryTracks(records)
  if (cursorSequence !== null) {
    const containing = tracks.find((track) =>
      track.records.some((record) => record.sequence === cursorSequence))
    if (containing) return containing
  }
  return tracks.at(-1) ?? null
}
""",
    encoding="utf-8",
)
