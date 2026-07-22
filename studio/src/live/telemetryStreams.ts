import type { TrainingTelemetryRecord } from '../data/types'

function closesEpisode(record: TrainingTelemetryRecord): boolean {
  return record.eventType === 'episode_end' || record.terminated || record.truncated
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

  let episodeStart = 0
  for (let index = 0; index < environmentRecords.length - 1; index += 1) {
    if (closesEpisode(environmentRecords[index])) episodeStart = index + 1
  }
  return environmentRecords.slice(episodeStart)
}
