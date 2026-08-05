import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

import type { TrainingTelemetryRecord } from '../data/types'
import {
  buildResearchChartData,
  type ResearchChartLayers,
  type ResearchTimeframe,
  type ResearchTradeBand,
} from './researchChartModel'

export type ResearchRangePreset = '1h' | '24h' | '7d' | 'all'
export type ResearchEmphasisPreset = 'overview' | 'trade' | 'risk'

export interface SynchronizedResearchChartWorkspaceProps {
  records: TrainingTelemetryRecord[]
  symbol: string
  timeframe: ResearchTimeframe
  rangePreset: ResearchRangePreset
  layers: ResearchChartLayers
  followLatest: boolean
  committedSequence: number | null
  resetToken: number
  onSymbolChange: (symbol: string) => void
  onTimeframeChange: (timeframe: ResearchTimeframe) => void
  onRangePresetChange: (preset: ResearchRangePreset) => void
  onPreviewRecord: (record: TrainingTelemetryRecord | null) => void
  onCommitRecord: (record: TrainingTelemetryRecord) => void
  onManualNavigation: () => void
}

interface SeriesRefs {
  candles: ISeriesApi<'Candlestick'>
  targetWeight: ISeriesApi<'Line'>
  executedWeight: ISeriesApi<'Line'>
  equity: ISeriesApi<'Line'>
  baseline: ISeriesApi<'Line'>
  drawdown: ISeriesApi<'Line'>
  grossExposure: ISeriesApi<'Line'>
  reward: ISeriesApi<'Line'>
  cost: ISeriesApi<'Line'>
}

interface LatestLineRef {
  series: ISeriesApi<'Candlestick'> | ISeriesApi<'Line'>
  line: IPriceLine
}

interface RangeSelection {
  start: number
  end: number
}

interface TradeBandRect extends ResearchTradeBand {
  left: number
  width: number
  height: number
}

type ResearchChartData = ReturnType<typeof buildResearchChartData>

const TIMEFRAME_OPTIONS: ResearchTimeframe[] = ['15m', '1h', '4h', '1d']
const RANGE_OPTIONS: Array<{ value: ResearchRangePreset; label: string }> = [
  { value: '1h', label: '1H' },
  { value: '24h', label: '24H' },
  { value: '7d', label: '7D' },
  { value: 'all', label: '全期間' },
]
const PRESETS: Array<{ value: ResearchEmphasisPreset; label: string }> = [
  { value: 'overview', label: 'Overview' },
  { value: 'trade', label: 'Trade focus' },
  { value: 'risk', label: 'Risk focus' },
]
const MANUAL_DRAG_THRESHOLD_PX = 5

function timeNumber(time: Time | null | undefined): number | null {
  return typeof time === 'number' && Number.isFinite(time) ? time : null
}

function rangeSeconds(preset: ResearchRangePreset): number | null {
  if (preset === '1h') return 60 * 60
  if (preset === '24h') return 24 * 60 * 60
  if (preset === '7d') return 7 * 24 * 60 * 60
  return null
}

function setSeriesData(series: SeriesRefs, data: ResearchChartData) {
  series.candles.setData(data.candles.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.targetWeight.setData(data.targetWeight.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.executedWeight.setData(data.executedWeight.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.equity.setData(data.equity.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.baseline.setData(data.baseline.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.drawdown.setData(data.drawdown.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.grossExposure.setData(data.grossExposure.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.reward.setData(data.reward.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.cost.setData(data.cost.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
}

function nearestRecord(data: ResearchChartData, time: number): TrainingTelemetryRecord | null {
  const exact = data.recordByTime.get(time)
  if (exact) return exact
  let nearest: TrainingTelemetryRecord | null = null
  let nearestDistance = Number.POSITIVE_INFINITY
  for (const [candidateTime, record] of data.recordByTime) {
    const distance = Math.abs(candidateTime - time)
    if (distance < nearestDistance) {
      nearest = record
      nearestDistance = distance
    }
  }
  return nearest
}

function recordAtLogicalIndex(data: ResearchChartData, logical: number): TrainingTelemetryRecord | null {
  if (data.candles.length === 0 || !Number.isFinite(logical)) return null
  const index = Math.min(data.candles.length - 1, Math.max(0, Math.round(logical)))
  return data.recordByTime.get(data.candles[index]!.time) ?? null
}

function interactionRecord(
  chart: IChartApi,
  params: MouseEventParams<Time>,
  data: ResearchChartData,
): TrainingTelemetryRecord | null {
  if (typeof params.hoveredObjectId === 'string') {
    const match = /^telemetry-(\d+)$/.exec(params.hoveredObjectId)
    if (match) {
      const markerRecord = data.recordBySequence.get(Number(match[1]))
      if (markerRecord) return markerRecord
    }
  }
  const directTime = timeNumber(params.time)
  if (directTime !== null) return nearestRecord(data, directTime)
  if (params.point === undefined) return null
  const timeScale = chart.timeScale()
  const coordinateTime = timeNumber(timeScale.coordinateToTime(params.point.x))
  if (coordinateTime !== null) return nearestRecord(data, coordinateTime)
  const logical = timeScale.coordinateToLogical(params.point.x)
  return logical === null ? null : recordAtLogicalIndex(data, logical)
}

function latestValue(points: Array<{ value: number }>): number | null {
  return points.at(-1)?.value ?? null
}

function durationLabel(seconds: number): string {
  const minutes = Math.max(0, Math.round(seconds / 60))
  const hours = Math.floor(minutes / 60)
  const remainder = minutes % 60
  return hours > 0 ? `${hours}h ${remainder}m` : `${remainder}m`
}

function rangeSummary(data: ResearchChartData, selection: RangeSelection | null) {
  if (selection === null) return null
  const start = Math.min(selection.start, selection.end)
  const end = Math.max(selection.start, selection.end)
  const records = [...data.recordByTime.entries()]
    .filter(([time]) => time >= start && time <= end)
    .sort(([left], [right]) => left - right)
    .map(([, record]) => record)
  if (records.length === 0) return { duration: durationLabel(end - start), returnValue: null, events: 0, maxDrawdown: null }
  const firstValue = records.find((record) => record.portfolioValue !== null)?.portfolioValue ?? null
  const lastValue = [...records].reverse().find((record) => record.portfolioValue !== null)?.portfolioValue ?? null
  const returnValue = firstValue !== null && lastValue !== null && firstValue !== 0
    ? ((lastValue / firstValue) - 1) * 100
    : null
  const drawdowns = records.map((record) => record.drawdown).filter((value): value is number => value !== null && Number.isFinite(value))
  return {
    duration: durationLabel(end - start),
    returnValue,
    events: records.filter((record) => record.eventType !== 'rollout').length,
    maxDrawdown: drawdowns.length === 0 ? null : Math.max(...drawdowns) * 100,
  }
}

function presetMarker(marker: ResearchChartData['markers'][number], preset: ResearchEmphasisPreset) {
  if (preset === 'trade' && marker.kind === 'risk') return { ...marker, color: 'rgba(243, 179, 61, 0.38)' }
  if (preset === 'risk' && marker.kind === 'position') return { ...marker, color: 'rgba(148, 168, 183, 0.38)' }
  return marker
}

export function SynchronizedResearchChartWorkspace({
  records,
  symbol,
  timeframe,
  rangePreset,
  layers,
  followLatest,
  committedSequence,
  resetToken,
  onSymbolChange,
  onTimeframeChange,
  onRangePresetChange,
  onPreviewRecord,
  onCommitRecord,
  onManualNavigation,
}: SynchronizedResearchChartWorkspaceProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<SeriesRefs | null>(null)
  const markerRef = useRef<{ setMarkers: (markers: SeriesMarker<Time>[]) => void } | null>(null)
  const dataRef = useRef<ResearchChartData | null>(null)
  const callbacksRef = useRef({ onPreviewRecord, onCommitRecord, onManualNavigation })
  const latestLinesRef = useRef<LatestLineRef[]>([])
  const appliedRangeKey = useRef<string | null>(null)
  const rangeDragRef = useRef<{ pointerId: number; start: number } | null>(null)
  const rangeModeRef = useRef(false)
  const [preset, setPreset] = useState<ResearchEmphasisPreset>('overview')
  const [rangeMode, setRangeMode] = useState(false)
  const [selection, setSelection] = useState<RangeSelection | null>(null)
  const [overlayVersion, setOverlayVersion] = useState(0)
  const [paneHeights, setPaneHeights] = useState([0, 0, 0])

  callbacksRef.current = { onPreviewRecord, onCommitRecord, onManualNavigation }
  rangeModeRef.current = rangeMode
  const data = useMemo(() => buildResearchChartData(records, symbol, timeframe), [records, symbol, timeframe])
  dataRef.current = data

  const selectedRecord = committedSequence === null
    ? null
    : data.recordBySequence.get(committedSequence) ?? null
  const latestSequence = data.recordByTime.size === 0
    ? null
    : [...data.recordByTime.values()].at(-1)?.sequence ?? null
  const showSelectionCard = selectedRecord !== null && (!followLatest || selectedRecord.sequence !== latestSequence)
  const summary = rangeSummary(data, selection)
  const refreshOverlay = () => setOverlayVersion((current) => current + 1)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return undefined

    const chart = createChart(container, {
      autoSize: typeof ResizeObserver !== 'undefined',
      width: Math.max(320, container.clientWidth || 900),
      height: Math.max(420, container.clientHeight || 620),
      layout: {
        background: { type: ColorType.Solid, color: '#07111a' },
        textColor: '#91a3b0',
        attributionLogo: true,
        panes: {
          enableResize: true,
          separatorColor: '#223543',
          separatorHoverColor: 'rgba(64, 152, 255, 0.34)',
        },
      },
      grid: {
        vertLines: { color: '#13222e' },
        horzLines: { color: '#13222e' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      handleScroll: true,
      handleScale: true,
      timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 4 },
    })

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: '#36e37d', downColor: '#ff5b63', wickUpColor: '#36e37d', wickDownColor: '#ff5b63', borderVisible: false,
    }, 0)
    const targetWeight = chart.addSeries(LineSeries, { color: '#4098ff', lineWidth: 2, title: 'Target', priceScaleId: 'left' }, 0)
    const executedWeight = chart.addSeries(LineSeries, { color: '#b2c2cc', lineWidth: 1, lineStyle: LineStyle.Dashed, title: 'Executed', priceScaleId: 'left' }, 0)
    const equity = chart.addSeries(LineSeries, { color: '#56c2e6', lineWidth: 2, title: 'Portfolio' }, 1)
    const baseline = chart.addSeries(LineSeries, { color: '#a6b6c0', lineWidth: 2, lineStyle: LineStyle.Dashed, title: 'Baseline' }, 1)
    const drawdown = chart.addSeries(LineSeries, { color: '#ff756d', lineWidth: 2, title: 'Drawdown', priceScaleId: 'left' }, 1)
    const grossExposure = chart.addSeries(LineSeries, { color: '#f2b84b', lineWidth: 2, title: 'Gross exposure' }, 2)
    const reward = chart.addSeries(LineSeries, { color: '#7fd37a', lineWidth: 1, title: 'Reward', priceScaleId: 'left' }, 2)
    const cost = chart.addSeries(LineSeries, { color: '#ff8a65', lineWidth: 1, title: 'Cost', priceScaleId: 'left' }, 2)
    const series = { candles, targetWeight, executedWeight, equity, baseline, drawdown, grossExposure, reward, cost }

    chart.panes()[0]?.setStretchFactor(4.8)
    chart.panes()[1]?.setStretchFactor(2)
    chart.panes()[2]?.setStretchFactor(1.4)

    const updatePaneHeights = () => {
      const heights = chart.panes().slice(0, 3).map((pane) => pane.getHeight())
      setPaneHeights([heights[0] ?? 0, heights[1] ?? 0, heights[2] ?? 0])
      refreshOverlay()
    }
    const crosshairHandler = (params: MouseEventParams<Time>) => {
      const currentData = dataRef.current
      callbacksRef.current.onPreviewRecord(currentData === null ? null : interactionRecord(chart, params, currentData))
    }
    const clickHandler = (params: MouseEventParams<Time>) => {
      if (rangeModeRef.current) return
      const currentData = dataRef.current
      if (currentData === null) return
      const record = interactionRecord(chart, params, currentData)
      if (record) callbacksRef.current.onCommitRecord(record)
    }
    const visibleRangeHandler = () => refreshOverlay()

    let pointerOrigin: { x: number; y: number } | null = null
    let manualNavigationSignaled = false
    const pointerDownHandler = (event: PointerEvent) => {
      if (event.button !== 0 || rangeModeRef.current) return
      pointerOrigin = { x: event.clientX, y: event.clientY }
      manualNavigationSignaled = false
    }
    const pointerMoveHandler = (event: PointerEvent) => {
      if (pointerOrigin === null || manualNavigationSignaled) return
      const distance = Math.hypot(event.clientX - pointerOrigin.x, event.clientY - pointerOrigin.y)
      if (distance < MANUAL_DRAG_THRESHOLD_PX) return
      manualNavigationSignaled = true
      callbacksRef.current.onManualNavigation()
    }
    const pointerEndHandler = () => {
      pointerOrigin = null
      manualNavigationSignaled = false
    }
    const wheelHandler = () => callbacksRef.current.onManualNavigation()

    chart.subscribeCrosshairMove(crosshairHandler)
    chart.subscribeClick(clickHandler)
    chart.timeScale().subscribeVisibleLogicalRangeChange(visibleRangeHandler)
    container.addEventListener('pointerdown', pointerDownHandler, { passive: true })
    window.addEventListener('pointermove', pointerMoveHandler, { capture: true, passive: true })
    window.addEventListener('pointerup', pointerEndHandler, { capture: true, passive: true })
    window.addEventListener('pointercancel', pointerEndHandler, { capture: true, passive: true })
    container.addEventListener('wheel', wheelHandler, { passive: true })
    const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(updatePaneHeights)
    resizeObserver?.observe(container)

    chartRef.current = chart
    seriesRef.current = series
    markerRef.current = createSeriesMarkers(candles, [])
    requestAnimationFrame(updatePaneHeights)

    return () => {
      chart.unsubscribeCrosshairMove(crosshairHandler)
      chart.unsubscribeClick(clickHandler)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(visibleRangeHandler)
      container.removeEventListener('pointerdown', pointerDownHandler)
      window.removeEventListener('pointermove', pointerMoveHandler, true)
      window.removeEventListener('pointerup', pointerEndHandler, true)
      window.removeEventListener('pointercancel', pointerEndHandler, true)
      container.removeEventListener('wheel', wheelHandler)
      resizeObserver?.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      markerRef.current = null
    }
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series) return
    setSeriesData(series, data)

    const visibleMarkers = data.markers.filter((marker) => marker.kind === 'position'
      ? layers.positionEvents
      : marker.kind === 'risk' ? layers.riskEvents : true)
    markerRef.current?.setMarkers(visibleMarkers.map((marker) => {
      const adjusted = presetMarker(marker, preset)
      const { sequence: _sequence, kind: _kind, ...chartMarker } = adjusted
      return { ...chartMarker, time: chartMarker.time as UTCTimestamp }
    }) as SeriesMarker<Time>[])

    latestLinesRef.current.forEach(({ series: owner, line }) => owner.removePriceLine(line))
    latestLinesRef.current = []
    const addLabel = (
      owner: ISeriesApi<'Candlestick'> | ISeriesApi<'Line'>,
      price: number | null,
      title: string,
      color: string,
    ) => {
      if (price === null) return
      latestLinesRef.current.push({
        series: owner,
        line: owner.createPriceLine({
          price,
          title,
          color,
          lineVisible: false,
          axisLabelVisible: true,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
        }),
      })
    }
    addLabel(series.candles, data.candles.at(-1)?.close ?? null, symbol, '#d7e4ea')
    addLabel(series.equity, latestValue(data.equity), 'Portfolio', '#56c2e6')
    if (layers.baseline) addLabel(series.baseline, latestValue(data.baseline), 'Baseline', '#a6b6c0')
    if (layers.drawdown) addLabel(series.drawdown, latestValue(data.drawdown), 'Drawdown', '#ff756d')
    addLabel(series.grossExposure, latestValue(data.grossExposure), 'Gross exposure', '#f2b84b')
    requestAnimationFrame(refreshOverlay)
  }, [data, layers, preset, symbol])

  useEffect(() => {
    const series = seriesRef.current
    const chart = chartRef.current
    if (!series || !chart) return

    series.executedWeight.applyOptions({ visible: layers.executedWeight })
    series.baseline.applyOptions({ visible: layers.baseline })
    series.drawdown.applyOptions({ visible: layers.drawdown })
    series.reward.applyOptions({ visible: layers.rewardCost })
    series.cost.applyOptions({ visible: layers.rewardCost })

    if (preset === 'overview') {
      series.candles.applyOptions({ upColor: '#36e37d', downColor: '#ff5b63', wickUpColor: '#36e37d', wickDownColor: '#ff5b63' })
      series.targetWeight.applyOptions({ color: '#4098ff', lineWidth: 2 })
      series.executedWeight.applyOptions({ color: '#b2c2cc', lineWidth: 1 })
      series.equity.applyOptions({ color: '#56c2e6', lineWidth: 2 })
      series.baseline.applyOptions({ color: '#a6b6c0', lineWidth: 2 })
      series.drawdown.applyOptions({ color: '#ff756d', lineWidth: 2 })
      series.grossExposure.applyOptions({ color: '#f2b84b', lineWidth: 2 })
      series.reward.applyOptions({ color: '#7fd37a', lineWidth: 1 })
      series.cost.applyOptions({ color: '#ff8a65', lineWidth: 1 })
      chart.panes()[0]?.setStretchFactor(4.8)
      chart.panes()[1]?.setStretchFactor(2)
      chart.panes()[2]?.setStretchFactor(1.4)
    } else if (preset === 'trade') {
      series.candles.applyOptions({ upColor: '#42eb86', downColor: '#ff6670', wickUpColor: '#42eb86', wickDownColor: '#ff6670' })
      series.targetWeight.applyOptions({ color: '#56b0ff', lineWidth: 3 })
      series.executedWeight.applyOptions({ color: '#d3dde3', lineWidth: 2 })
      series.equity.applyOptions({ color: 'rgba(86,194,230,.42)', lineWidth: 1 })
      series.baseline.applyOptions({ color: 'rgba(166,182,192,.32)', lineWidth: 1 })
      series.drawdown.applyOptions({ color: 'rgba(255,117,109,.38)', lineWidth: 1 })
      series.grossExposure.applyOptions({ color: '#f2b84b', lineWidth: 2 })
      series.reward.applyOptions({ color: 'rgba(127,211,122,.34)', lineWidth: 1 })
      series.cost.applyOptions({ color: 'rgba(255,138,101,.44)', lineWidth: 1 })
      chart.panes()[0]?.setStretchFactor(5.4)
      chart.panes()[1]?.setStretchFactor(1.35)
      chart.panes()[2]?.setStretchFactor(1.25)
    } else {
      series.candles.applyOptions({ upColor: '#477462', downColor: '#7b4b51', wickUpColor: '#477462', wickDownColor: '#7b4b51' })
      series.targetWeight.applyOptions({ color: 'rgba(64,152,255,.28)', lineWidth: 1 })
      series.executedWeight.applyOptions({ color: 'rgba(178,194,204,.24)', lineWidth: 1 })
      series.equity.applyOptions({ color: 'rgba(86,194,230,.44)', lineWidth: 1 })
      series.baseline.applyOptions({ color: 'rgba(166,182,192,.32)', lineWidth: 1 })
      series.drawdown.applyOptions({ color: '#ff756d', lineWidth: 3 })
      series.grossExposure.applyOptions({ color: '#f2b84b', lineWidth: 3 })
      series.reward.applyOptions({ color: 'rgba(127,211,122,.36)', lineWidth: 1 })
      series.cost.applyOptions({ color: '#ff8a65', lineWidth: 2 })
      chart.panes()[0]?.setStretchFactor(2.9)
      chart.panes()[1]?.setStretchFactor(2.25)
      chart.panes()[2]?.setStretchFactor(2.7)
    }
    requestAnimationFrame(refreshOverlay)
  }, [layers, preset])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || data.candles.length === 0) return
    const rangeKey = `${symbol}|${timeframe}|${rangePreset}|${resetToken}`
    if (appliedRangeKey.current === rangeKey) return
    appliedRangeKey.current = rangeKey
    const latest = data.candles.at(-1)!.time
    const seconds = rangeSeconds(rangePreset)
    if (seconds === null) chart.timeScale().fitContent()
    else chart.timeScale().setVisibleRange({
      from: Math.max(data.candles[0]!.time, latest - seconds) as UTCTimestamp,
      to: latest as UTCTimestamp,
    })
    requestAnimationFrame(refreshOverlay)
  }, [data.candles.length, rangePreset, resetToken, symbol, timeframe])

  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series || committedSequence === null) return
    const time = data.timeBySequence.get(committedSequence)
    const record = time === undefined ? null : data.recordByTime.get(time) ?? null
    if (time === undefined || !record || record.close === null) return
    chart.setCrosshairPosition(record.close, time as UTCTimestamp, series.candles)
    if (followLatest && time === data.candles.at(-1)?.time) chart.timeScale().scrollToRealTime()
  }, [committedSequence, data, followLatest])

  const chart = chartRef.current
  const containerWidth = containerRef.current?.clientWidth ?? 0
  const pricePaneHeight = paneHeights[0] || Math.round((containerRef.current?.clientHeight ?? 0) * 0.58)
  const tradeBandRects: TradeBandRect[] = data.tradeBands.flatMap((band) => {
    if (!chart || containerWidth <= 0) return []
    const start = chart.timeScale().timeToCoordinate(band.startTime as UTCTimestamp)
    const end = chart.timeScale().timeToCoordinate(band.endTime as UTCTimestamp)
    if (start === null || end === null) return []
    const left = Math.max(0, Math.min(start, end))
    const right = Math.min(containerWidth, Math.max(start, end))
    if (right <= 0 || left >= containerWidth) return []
    return [{ ...band, left, width: Math.max(4, right - left), height: Math.max(24, pricePaneHeight - 12) }]
  })
  const rangeRect = (() => {
    if (!chart || !selection || containerWidth <= 0) return null
    const start = chart.timeScale().timeToCoordinate(Math.min(selection.start, selection.end) as UTCTimestamp)
    const end = chart.timeScale().timeToCoordinate(Math.max(selection.start, selection.end) as UTCTimestamp)
    if (start === null || end === null) return null
    return { left: Math.max(0, start), width: Math.max(2, Math.min(containerWidth, end) - Math.max(0, start)) }
  })()
  void overlayVersion

  const timeAtPointer = (clientX: number): number | null => {
    const currentChart = chartRef.current
    const container = containerRef.current
    if (!currentChart || !container) return null
    const bounds = container.getBoundingClientRect()
    return timeNumber(currentChart.timeScale().coordinateToTime(clientX - bounds.left))
  }

  const startRange = (event: ReactPointerEvent<HTMLDivElement>) => {
    const time = timeAtPointer(event.clientX)
    if (time === null) return
    event.currentTarget.setPointerCapture(event.pointerId)
    rangeDragRef.current = { pointerId: event.pointerId, start: time }
    setSelection({ start: time, end: time })
  }
  const moveRange = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = rangeDragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    const time = timeAtPointer(event.clientX)
    if (time !== null) setSelection({ start: drag.start, end: time })
  }
  const endRange = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (rangeDragRef.current?.pointerId !== event.pointerId) return
    rangeDragRef.current = null
    event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const symbols = data.symbols.length > 0 ? data.symbols : [symbol]
  const exposure = selectedRecord?.weightsAfter.reduce((total, weight) => total + Math.abs(weight), 0) ?? null

  return (
    <div className="synchronized-chart-column" data-preset={preset}>
      <div className="synchronized-chart-header">
        <div className="synchronized-chart-context">
          <select aria-label="Chart symbol" value={symbol} onChange={(event) => onSymbolChange(event.target.value)}>
            {symbols.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
          <div className="synchronized-chart-segments" aria-label="時間足">
            {TIMEFRAME_OPTIONS.map((value) => (
              <button type="button" key={value} aria-pressed={timeframe === value} onClick={() => onTimeframeChange(value)}>{value}</button>
            ))}
          </div>
        </div>
        <div className="synchronized-chart-presets" aria-label="分析強調">
          {PRESETS.map((option) => (
            <button type="button" key={option.value} aria-pressed={preset === option.value} onClick={() => setPreset(option.value)}>{option.label}</button>
          ))}
        </div>
        <div className="synchronized-chart-context synchronized-chart-context--right">
          <div className="synchronized-chart-segments" aria-label="表示期間">
            {RANGE_OPTIONS.map((option) => (
              <button type="button" key={option.value} aria-pressed={rangePreset === option.value} onClick={() => onRangePresetChange(option.value)}>{option.label}</button>
            ))}
          </div>
          <button
            type="button"
            className="synchronized-range-toggle"
            aria-pressed={rangeMode}
            onClick={() => setRangeMode((current) => !current)}
          >
            Range
          </button>
        </div>
      </div>

      <div className="synchronized-chart-stage">
        {data.candles.length === 0 ? <div className="research-chart-empty">選択したSymbol・時間足で表示可能なテレメトリがありません。</div> : null}
        <div
          ref={containerRef}
          className="synchronized-chart-canvas"
          role="img"
          aria-label={`${symbol} ${timeframe} 市場・売買・資産・リスクの3Pane同期チャート`}
        />
        <div className="synchronized-pane-labels" aria-hidden="true">
          <span style={{ top: 8 }}>Price &amp; Execution</span>
          <span style={{ top: paneHeights[0] + 8 }}>Portfolio</span>
          <span style={{ top: paneHeights[0] + paneHeights[1] + 8 }}>State &amp; Risk</span>
        </div>
        <div className="synchronized-trade-bands" aria-hidden="true">
          {tradeBandRects.map((band, index) => (
            <span
              key={band.id}
              className={`synchronized-trade-band synchronized-trade-band--${band.direction} synchronized-trade-band--${band.status}`}
              style={{ left: band.left, width: band.width, height: band.height }}
            >
              <b>#{index + 1}</b>
            </span>
          ))}
        </div>
        {rangeRect ? (
          <div className="synchronized-range-selection" style={{ left: rangeRect.left, width: rangeRect.width }}>
            {summary ? (
              <span aria-live="polite">
                {summary.duration} · NAV {summary.returnValue === null ? '—' : `${summary.returnValue >= 0 ? '+' : ''}${summary.returnValue.toFixed(2)}%`} · {summary.events} events · DD {summary.maxDrawdown === null ? '—' : `${summary.maxDrawdown.toFixed(2)}%`}
              </span>
            ) : null}
          </div>
        ) : null}
        {rangeMode ? (
          <div
            className="synchronized-range-interaction"
            aria-label="チャート範囲選択"
            onPointerDown={startRange}
            onPointerMove={moveRange}
            onPointerUp={endRange}
            onPointerCancel={endRange}
          />
        ) : null}
        {showSelectionCard && selectedRecord ? (
          <aside className="synchronized-selection-card" aria-label="選択時点のチャート要約">
            <strong>Step {selectedRecord.globalStep.toLocaleString('ja-JP')}</strong>
            <time>{selectedRecord.marketTime?.replace('T', ' ').slice(0, 19) ?? '—'}</time>
            <dl>
              <div><dt>Price</dt><dd>{selectedRecord.close?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'}</dd></div>
              <div><dt>Portfolio</dt><dd>{selectedRecord.portfolioValue?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'}</dd></div>
              <div><dt>Gross exposure</dt><dd>{exposure === null ? '—' : `${(exposure * 100).toFixed(1)}%`}</dd></div>
              <div><dt>Drawdown</dt><dd>{selectedRecord.drawdown === null ? '—' : `${(selectedRecord.drawdown * 100).toFixed(2)}%`}</dd></div>
              <div><dt>Cost</dt><dd>{selectedRecord.intervalCost?.toLocaleString('ja-JP', { maximumFractionDigits: 4 }) ?? '—'}</dd></div>
            </dl>
          </aside>
        ) : null}
        {selection ? (
          <button type="button" className="synchronized-range-clear" onClick={() => setSelection(null)}>範囲解除</button>
        ) : null}
      </div>
    </div>
  )
}