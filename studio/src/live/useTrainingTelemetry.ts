import { useCallback, useEffect, useRef, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { TelemetryStatusResponse, TrainingTelemetryRecord } from '../data/types'

const MAX_BUFFERED_RECORDS = 2_048
const POLL_INTERVAL_MS = 1_000

export type TelemetryConnection = 'connecting' | 'live' | 'delayed' | 'offline'

export interface TrainingTelemetryState {
  records: TrainingTelemetryRecord[]
  status: TelemetryStatusResponse | null
  connection: TelemetryConnection
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

function mergeRecords(
  current: TrainingTelemetryRecord[],
  incoming: TrainingTelemetryRecord[],
): TrainingTelemetryRecord[] {
  if (incoming.length === 0) return current
  const bySequence = new Map(current.map((item) => [item.sequence, item]))
  for (const item of incoming) bySequence.set(item.sequence, item)
  return [...bySequence.values()]
    .sort((left, right) => left.sequence - right.sequence)
    .slice(-MAX_BUFFERED_RECORDS)
}

export function useTrainingTelemetry(
  jobId: string | null,
  api: StudioApi = studioApi,
  seed: number | null = null,
): TrainingTelemetryState {
  const [records, setRecords] = useState<TrainingTelemetryRecord[]>([])
  const [status, setStatus] = useState<TelemetryStatusResponse | null>(null)
  const [connection, setConnection] = useState<TelemetryConnection>('connecting')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const sequence = useRef(0)
  const generation = useRef<string | null>(null)
  const request = useRef(0)

  const refresh = useCallback(async () => {
    if (!jobId) {
      setLoading(false)
      setConnection('offline')
      return
    }
    const resolvedJobId = jobId
    const currentRequest = ++request.current
    setLoading(true)

    async function loadSnapshot(allowRetry: boolean): Promise<void> {
      const expectedGeneration = generation.current
      const [nextStatus, page] = await Promise.all([
        api.loadTelemetryStatus(resolvedJobId, seed),
        api.loadTelemetryEvents(
          resolvedJobId,
          sequence.current,
          512,
          seed,
          expectedGeneration,
        ),
      ])
      if (currentRequest !== request.current) return

      const retry = async (
        nextGeneration: string | null,
        reason: string,
      ): Promise<void> => {
        setRecords([])
        setStatus(null)
        setConnection('connecting')
        sequence.current = 0
        generation.current = nextGeneration
        if (!allowRetry) {
          throw new Error(`Telemetry stream changed repeatedly: ${reason}`)
        }
        await loadSnapshot(false)
      }

      if (page.resetRequired) {
        if (page.streamGeneration === null) {
          throw new Error('Telemetry reset response has no stream generation')
        }
        await retry(page.streamGeneration, 'cursor generation is stale')
        return
      }

      if (
        nextStatus.streamGeneration !== null
        && page.streamGeneration !== null
        && nextStatus.streamGeneration !== page.streamGeneration
      ) {
        await retry(null, 'status and events generations differ')
        return
      }

      if (
        nextStatus.available
        && (
          nextStatus.streamGeneration === null
          || page.streamGeneration === null
        )
      ) {
        throw new Error('Available telemetry response has no stream generation')
      }

      const resolvedGeneration = page.streamGeneration
        ?? nextStatus.streamGeneration
      if (generation.current === null) {
        generation.current = resolvedGeneration
      } else if (
        resolvedGeneration !== null
        && generation.current !== resolvedGeneration
      ) {
        throw new Error('Telemetry generation changed without a reset response')
      }

      setStatus(nextStatus)
      if (page.items.length > 0) {
        const accepted = page.items.filter(
          (item) => item.sequence > sequence.current,
        )
        if (accepted.length > 0) {
          setRecords((current) => mergeRecords(current, accepted))
        }
      }
      sequence.current = Math.max(sequence.current, page.nextSequence)
      setError(null)
      setConnection(
        nextStatus.available ? (page.truncated ? 'delayed' : 'live') : 'delayed',
      )
    }

    try {
      await loadSnapshot(true)
    } catch (reason) {
      if (currentRequest !== request.current) return
      setError(
        reason instanceof Error
          ? reason.message
          : 'テレメトリを取得できませんでした。',
      )
      setConnection('offline')
    } finally {
      if (currentRequest === request.current) setLoading(false)
    }
  }, [api, jobId, seed])

  useEffect(() => {
    request.current += 1
    sequence.current = 0
    generation.current = null
    setRecords([])
    setStatus(null)
    setConnection(jobId ? 'connecting' : 'offline')
    setLoading(true)
    setError(null)
    void refresh()
    if (!jobId) return undefined
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => {
      window.clearInterval(timer)
      request.current += 1
    }
  }, [jobId, refresh, seed])

  return { records, status, connection, loading, error, refresh }
}
