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
  const request = useRef(0)

  const refresh = useCallback(async () => {
    if (!jobId) {
      setLoading(false)
      setConnection('offline')
      return
    }
    const currentRequest = ++request.current
    try {
      const [nextStatus, page] = await Promise.all([
        api.loadTelemetryStatus(jobId, seed),
        api.loadTelemetryEvents(jobId, sequence.current, 512, seed),
      ])
      if (currentRequest !== request.current) return
      setStatus(nextStatus)
      if (page.items.length > 0) {
        const accepted = page.items.filter((item) => item.sequence > sequence.current)
        if (accepted.length > 0) {
          sequence.current = Math.max(sequence.current, page.nextSequence)
          setRecords((current) => {
            const bySequence = new Map(current.map((item) => [item.sequence, item]))
            for (const item of accepted) bySequence.set(item.sequence, item)
            return [...bySequence.values()]
              .sort((left, right) => left.sequence - right.sequence)
              .slice(-MAX_BUFFERED_RECORDS)
          })
        }
      }
      setError(null)
      setConnection(nextStatus.available ? (page.truncated ? 'delayed' : 'live') : 'delayed')
    } catch (reason) {
      if (currentRequest !== request.current) return
      setError(reason instanceof Error ? reason.message : 'テレメトリを取得できませんでした。')
      setConnection('offline')
    } finally {
      if (currentRequest === request.current) setLoading(false)
    }
  }, [api, jobId, seed])

  useEffect(() => {
    request.current += 1
    sequence.current = 0
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
