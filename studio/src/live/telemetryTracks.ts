import type { TrainingTelemetryRecord } from '../data/types'

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
