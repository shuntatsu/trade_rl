import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { TrainingMetricSeries, TrainingMetricsStatusResponse } from '../data/types'

const MAX_POINTS = 2_048
const MAX_TAGS_PER_REQUEST = 8
const POLL_INTERVAL_MS = 2_000

export interface TrainingMetricsState {
  status: TrainingMetricsStatusResponse | null
  series: TrainingMetricSeries[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

function mergeSeries(current: TrainingMetricSeries[], incoming: TrainingMetricSeries[]): TrainingMetricSeries[] {
  const byTag = new Map(current.map((series) => [series.tag, series]))
  for (const series of incoming) {
    const previous = byTag.get(series.tag)
    const byStep = new Map(previous?.points.map((point) => [point.step, point]) ?? [])
    for (const point of series.points) byStep.set(point.step, point)
    byTag.set(series.tag, {
      ...series,
      points: [...byStep.values()].sort((left, right) => left.step - right.step).slice(-MAX_POINTS),
    })
  }
  return [...byTag.values()]
}

function tagBatches(tags: string[]): string[][] {
  const batches: string[][] = []
  for (let index = 0; index < tags.length; index += MAX_TAGS_PER_REQUEST) {
    batches.push(tags.slice(index, index + MAX_TAGS_PER_REQUEST))
  }
  return batches
}

export function useTrainingMetrics(
  jobId: string | null,
  seed: number | null,
  tags: string[],
  api: StudioApi = studioApi,
): TrainingMetricsState {
  const [status, setStatus] = useState<TrainingMetricsStatusResponse | null>(null)
  const [series, setSeries] = useState<TrainingMetricSeries[]>([])
  const [loading, setLoading] = useState(Boolean(jobId))
  const [error, setError] = useState<string | null>(null)
  const cursors = useRef(new Map<string, number>())
  const generation = useRef<string | null>(null)
  const request = useRef(0)
  const tagKey = useMemo(() => JSON.stringify([...new Set(tags)].sort()), [tags])
  const stableTags = useMemo(() => JSON.parse(tagKey) as string[], [tagKey])

  const refresh = useCallback(async () => {
    const loadStatus = api.loadTrainingMetricsStatus
    const loadScalars = api.loadTrainingMetricScalars
    if (!jobId || !loadStatus || !loadScalars) {
      setLoading(false)
      setStatus(null)
      setSeries([])
      return
    }
    const requestId = ++request.current
    try {
      const nextStatus = await loadStatus(jobId, seed)
      if (requestId !== request.current) return
      setStatus(nextStatus)
      if (!nextStatus.available || stableTags.length === 0) {
        setError(null)
        return
      }

      const incoming: TrainingMetricSeries[] = []
      let expectedGeneration = generation.current
      for (const batch of tagBatches(stableTags)) {
        const batchKey = batch.join('|')
        const page = await loadScalars(
          jobId,
          batch,
          cursors.current.get(batchKey) ?? 0,
          512,
          seed,
          expectedGeneration,
        )
        if (requestId !== request.current) return
        if (page.resetRequired) {
          cursors.current.clear()
          generation.current = page.generation
          setSeries([])
          return
        }
        if (seed !== null && page.seed !== seed) {
          throw new Error('学習指標のseed identityが一致しません。')
        }
        if (expectedGeneration !== null && page.generation !== expectedGeneration) {
          throw new Error('学習指標のgenerationがresetなしで変化しました。')
        }
        expectedGeneration = page.generation
        cursors.current.set(batchKey, Math.max(cursors.current.get(batchKey) ?? 0, page.nextStep))
        incoming.push(...page.series)
      }

      generation.current = expectedGeneration
      setSeries((current) => mergeSeries(current, incoming))
      setError(null)
    } catch (reason) {
      if (requestId !== request.current) return
      setError(reason instanceof Error ? reason.message : '学習指標を取得できませんでした。')
    } finally {
      if (requestId === request.current) setLoading(false)
    }
  }, [api, jobId, seed, stableTags])

  useEffect(() => {
    request.current += 1
    cursors.current.clear()
    generation.current = null
    setStatus(null)
    setSeries([])
    setError(null)
    setLoading(Boolean(jobId))

    let active = true
    let timer: number | undefined
    const poll = async () => {
      await refresh()
      if (active && jobId) {
        timer = window.setTimeout(() => void poll(), POLL_INTERVAL_MS)
      }
    }
    void poll()

    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
      request.current += 1
    }
  }, [jobId, refresh, seed, tagKey])

  return { status, series, loading, error, refresh }
}
