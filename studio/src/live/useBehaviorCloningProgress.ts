import { useEffect, useState } from 'react'

import type { StudioApi } from '../api/studioApi'
import type { BehaviorCloningProgressResponse } from '../data/types'

export function useBehaviorCloningProgress(jobId: string | null, api: StudioApi) {
  const [progress, setProgress] = useState<BehaviorCloningProgressResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setProgress(null)
    setError(null)
    if (!jobId || !api.loadBehaviorCloningProgress) return undefined
    let active = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const next = await api.loadBehaviorCloningProgress?.(jobId)
        if (!active || !next) return
        setProgress(next)
        setError(null)
      } catch (reason) {
        if (!active) return
        setError(reason instanceof Error ? reason.message : 'BC進捗を取得できませんでした。')
      } finally {
        if (active) timer = window.setTimeout(() => { void poll() }, 200)
      }
    }
    void poll()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [api, jobId])

  return { progress, error }
}
