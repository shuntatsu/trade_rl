import { useCallback, useEffect, useRef, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { RunComparison, RunSummary } from '../data/types'
import { readParam, replaceParams } from '../state/urlState'
import type { ComparisonRangeSelection } from './InteractiveComparisonWorkspace'

type CompareApi = Pick<StudioApi, 'loadRuns' | 'loadRunComparison'>

function integerParam(search: string, key: string): number | null {
  const raw = readParam(search, key)
  if (raw === null || raw.trim() === '') return null
  const value = Number(raw)
  return Number.isInteger(value) && value >= 0 ? value : null
}

function rangeFromUrl(search: string): ComparisonRangeSelection | null {
  const start = integerParam(search, 'compareStart')
  const end = integerParam(search, 'compareEnd')
  return start === null || end === null ? null : { start, end }
}

export function useRunComparisonWorkspace(api: CompareApi = studioApi) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [leftRunId, setLeftRunIdState] = useState(
    () => readParam(window.location.search, 'left') ?? '',
  )
  const [rightRunId, setRightRunIdState] = useState(
    () => readParam(window.location.search, 'right') ?? '',
  )
  const [comparison, setComparison] = useState<RunComparison | null>(null)
  const [committedPoint, setCommittedPointState] = useState<number | null>(
    () => integerParam(window.location.search, 'comparePoint'),
  )
  const [committedRange, setCommittedRangeState] = useState<ComparisonRangeSelection | null>(
    () => rangeFromUrl(window.location.search),
  )
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const requestSequence = useRef(0)
  const pairRef = useRef({ left: leftRunId, right: rightRunId })

  const setLeftRunId = useCallback((value: string) => {
    pairRef.current.left = value
    setLeftRunIdState(value)
  }, [])

  const setRightRunId = useCallback((value: string) => {
    pairRef.current.right = value
    setRightRunIdState(value)
  }, [])

  const replaceSelection = useCallback((
    point: number | null,
    range: ComparisonRangeSelection | null,
  ) => {
    setCommittedPointState(point)
    setCommittedRangeState(range)
    replaceParams({
      comparePoint: point === null ? null : String(point),
      compareStart: range === null ? null : String(range.start),
      compareEnd: range === null ? null : String(range.end),
    })
  }, [])

  const setCommittedPoint = useCallback(
    (index: number) => replaceSelection(index, null),
    [replaceSelection],
  )
  const setCommittedRange = useCallback(
    (range: ComparisonRangeSelection | null) => replaceSelection(null, range),
    [replaceSelection],
  )
  const clearSelection = useCallback(
    () => replaceSelection(null, null),
    [replaceSelection],
  )

  const loadComparison = useCallback(async (
    left: string,
    right: string,
    preserveSelection = false,
  ) => {
    if (!left || !right) {
      setComparison(null)
      setLoading(false)
      replaceParams({ left: null, right: null })
      clearSelection()
      return
    }
    const sequence = ++requestSequence.current
    setLoading(true)
    setError(null)
    setComparison((current) => (
      current?.leftResourceId === left && current.rightResourceId === right
        ? current
        : null
    ))
    replaceParams({ left, right })
    if (!preserveSelection) clearSelection()
    try {
      const value = await api.loadRunComparison(left, right)
      if (sequence === requestSequence.current) setComparison(value)
    } catch (reason) {
      if (sequence === requestSequence.current) {
        setComparison(null)
        setError(
          reason instanceof Error ? reason.message : 'run比較を取得できませんでした。',
        )
      }
    } finally {
      if (sequence === requestSequence.current) setLoading(false)
    }
  }, [api, clearSelection])

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.loadRuns()
      const valid = response.items.filter((item) => item.status === 'VALID')
      setRuns(valid)
      const current = pairRef.current
      const left = valid.some((item) => item.id === current.left)
        ? current.left
        : valid[0]?.id ?? ''
      const right = valid.some((item) => item.id === current.right)
        ? current.right
        : valid[1]?.id ?? valid[0]?.id ?? ''
      setLeftRunId(left)
      setRightRunId(right)
      await loadComparison(left, right, true)
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : 'run一覧を取得できませんでした。',
      )
      setLoading(false)
    }
  }, [api, loadComparison, setLeftRunId, setRightRunId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    const restore = () => {
      const search = window.location.search
      const nextLeft = readParam(search, 'left') ?? pairRef.current.left
      const nextRight = readParam(search, 'right') ?? pairRef.current.right
      setCommittedPointState(integerParam(search, 'comparePoint'))
      setCommittedRangeState(rangeFromUrl(search))
      if (nextLeft !== pairRef.current.left || nextRight !== pairRef.current.right) {
        setLeftRunId(nextLeft)
        setRightRunId(nextRight)
        void loadComparison(nextLeft, nextRight, true)
      }
    }
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [loadComparison, setLeftRunId, setRightRunId])

  return {
    runs,
    leftRunId,
    rightRunId,
    comparison,
    committedPoint,
    committedRange,
    loading,
    error,
    setLeftRunId,
    setRightRunId,
    setCommittedPoint,
    setCommittedRange,
    clearSelection,
    loadComparison,
    refresh,
  }
}

export type { CompareApi }
