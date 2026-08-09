import { useEffect, useState } from 'react'

import { loadStudioOverview } from '../api/studioApi'
import type { StudioOverviewResult } from '../data/types'
import { applyPollingFailure, applyPollingResult, initialPollingState, type StudioOverviewPollingState } from './studioOverviewPollingState'

const defaultLoader = () => loadStudioOverview()
const defaultNow = () => Date.now()

interface StudioOverviewPollingOptions {
  loader?: () => Promise<StudioOverviewResult>
  intervalMs?: number
  now?: () => number
}

export function useStudioOverviewPolling(
  initial: StudioOverviewResult,
  options: StudioOverviewPollingOptions = {},
): StudioOverviewPollingState {
  const loader = options.loader ?? defaultLoader
  const intervalMs = options.intervalMs ?? 500
  const now = options.now ?? defaultNow
  const [state, setState] = useState<StudioOverviewPollingState>(() => initialPollingState(initial, now()))

  useEffect(() => {
    if (initial.source === 'demo') return undefined
    let active = true
    let timer: number | undefined
    let generation = 0

    const schedule = () => {
      if (active) timer = window.setTimeout(() => void refresh(), intervalMs)
    }
    const refresh = async () => {
      const currentGeneration = ++generation
      try {
        const result = await loader()
        if (!active || currentGeneration !== generation) return
        setState((previous) => applyPollingResult(previous, result, now()))
      } catch (reason) {
        if (!active || currentGeneration !== generation) return
        const message = reason instanceof Error ? reason.message : 'Studio API refresh failed'
        setState((previous) => applyPollingFailure(previous, message))
      } finally {
        if (active && currentGeneration === generation) schedule()
      }
    }

    schedule()
    return () => {
      active = false
      generation += 1
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [initial.source, intervalMs, loader, now])

  return state
}
