import type { TrainingTelemetryRecord } from '../data/types'

function closesEpisode(record: TrainingTelemetryRecord): boolean {
  return record.eventType === 'episode_end' || record.terminated || record.truncated
}

function crossesEpisodeBoundary(
  previous: TrainingTelemetryRecord,
  current: TrainingTelemetryRecord,
): boolean {
  if (closesEpisode(previous)) return true
  if (current.environmentStep < previous.environmentStep) return true
  return previous.marketIndex !== null
    && current.marketIndex !== null
    && current.marketIndex < previous.marketIndex
}

export function telemetryEnvironmentIds(records: TrainingTelemetryRecord[]): number[] {
  return [...new Set(records.map((record) => record.environmentId))].sort((left, right) => left - right)
}

export function currentEnvironmentEpisode(
  records: TrainingTelemetryRecord[],
  environmentId: number | null,
): TrainingTelemetryRecord[] {
  if (environmentId === null) return []
  const environmentRecords = records
    .filter((record) => record.environmentId === environmentId)
    .sort((left, right) => left.sequence - right.sequence)
  if (environmentRecords.length === 0) return []

  const latestEpisodeId = environmentRecords.at(-1)?.episodeId ?? null
  if (latestEpisodeId !== null) {
    return environmentRecords.filter((record) => record.episodeId === latestEpisodeId)
  }

  let episodeStart = 0
  for (let index = 1; index < environmentRecords.length; index += 1) {
    if (crossesEpisodeBoundary(environmentRecords[index - 1], environmentRecords[index])) {
      episodeStart = index
    }
  }
  return environmentRecords.slice(episodeStart)
}
