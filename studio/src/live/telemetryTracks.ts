import type { TrainingTelemetryRecord } from '../data/types'

export interface TelemetryTrack {
  key: string
  environmentId: number
  episodeOrdinal: number
  records: TrainingTelemetryRecord[]
  firstSequence: number
  lastSequence: number
  ended: boolean
}

interface ActiveTrack {
  key: string
  ordinal: number
  previous: TrainingTelemetryRecord
}

interface MutableTrack {
  key: string
  environmentId: number
  episodeOrdinal: number
  records: TrainingTelemetryRecord[]
}

function ended(record: TrainingTelemetryRecord): boolean {
  return record.eventType === 'episode_end' || record.terminated || record.truncated
}

function crossesEpisodeBoundary(
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
  const tracks = new Map<string, MutableTrack>()
  const activeByEnvironment = new Map<number, ActiveTrack>()
  const nextOrdinalByEnvironment = new Map<number, number>()

  for (const record of ordered) {
    const active = activeByEnvironment.get(record.environmentId)
    let key = active?.key ?? ''
    let ordinal = active?.ordinal ?? 0

    if (!active || crossesEpisodeBoundary(active.previous, record)) {
      ordinal = nextOrdinalByEnvironment.get(record.environmentId) ?? 0
      nextOrdinalByEnvironment.set(record.environmentId, ordinal + 1)
      key = `${record.environmentId}:${ordinal}`
    }

    const existing = tracks.get(key)
    if (existing) {
      existing.records.push(record)
    } else {
      tracks.set(key, {
        key,
        environmentId: record.environmentId,
        episodeOrdinal: ordinal,
        records: [record],
      })
    }

    activeByEnvironment.set(record.environmentId, {
      key,
      ordinal,
      previous: record,
    })
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
