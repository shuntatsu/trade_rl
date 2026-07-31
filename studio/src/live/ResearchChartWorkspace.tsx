import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef } from 'react'

import type { TrainingTelemetryRecord } from '../data/types'
import {
  buildResearchChartData,
  type ResearchChartLayers,
  type ResearchTimeframe,
} from './researchChartModel'

export type ResearchRangePreset = '1h' | '24h' | '7d' | 'all'

export interface ResearchChartWorkspaceProps {
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
  reward: ISeriesApi<'Line'>
  cost: ISeriesApi<'Line'>
  equity: ISeriesApi<'Line'>
  baseline: ISeriesApi<'Line'>
  drawdown: ISeriesApi<'Line'>
}

type ResearchChartData = ReturnType<typeof buildResearchChartData>

const TIMEFRAME_OPTIONS: ResearchTimeframe[] = ['15m', '1h', '4h', '1d']
const RANGE_OPTIONS: Array<{ value: ResearchRangePreset; label: string }> = [
  { value: '1h', label: '1H' },
  { value: '24h', label: '24H' },
  { value: '7d', label: '7D' },
  { value: 'all', label: '全期間' },
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
  series.reward.setData(data.reward.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.cost.setData(data.cost.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.equity.setData(data.equity.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.baseline.setData(data.baseline.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
  series.drawdown.setData(data.drawdown.map((point) => ({ ...point, time: point.time as UTCTimestamp })))
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

function interactionRecord(
  chart: IChartApi,
  params: MouseEventParams<Time>,
  data: ResearchChartData,
): TrainingTelemetryRecord | null {
  const directTime = timeNumber(params.time)
  const resolvedTime = directTime ?? (
    params.point === undefined
      ? null
      : timeNumber(chart.timeScale().coordinateToTime(params.point.x))
  )
  return resolvedTime === null ? null : nearestRecord(data, resolvedTime)
}

export function ResearchChartWorkspace({
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
}: ResearchChartWorkspaceProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<SeriesRefs | null>(null)
  const markerRef = useRef<{ setMarkers: (markers: SeriesMarker<Time>[]) => void } | null>(null)
  const dataRef = useRef<ResearchChartData | null>(null)
  const appliedRangeKey = useRef<string | null>(null)
  const callbacksRef = useRef({ onPreviewRecord, onCommitRecord, onManualNavigation })
  callbacksRef.current = { onPreviewRecord, onCommitRecord, onManualNavigation }

  const data = useMemo(
    () => buildResearchChartData(records, symbol, timeframe),
    [records, symbol, timeframe],
  )
  dataRef.current = data

  useEffect(() => {
    const container = containerRef.current
    if (!container) return undefined

    const chart = createChart(container, {
      autoSize: typeof ResizeObserver !== 'undefined',
      width: Math.max(320, container.clientWidth || 900),
      height: Math.max(420, container.clientHeight || 620),
      layout: {
        background: { type: ColorType.Solid, color: '#07111a' },
        textColor: '#8193a2',
        attributionLogo: true,
        panes: {
          enableResize: true,
          separatorColor: '#1a2a36',
          separatorHoverColor: 'rgba(64, 152, 255, 0.25)',
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
      upColor: '#36e37d',
      downColor: '#ff5b63',
      wickUpColor: '#36e37d',
      wickDownColor: '#ff5b63',
      borderVisible: false,
    }, 0)
    const targetWeight = chart.addSeries(LineSeries, { color: '#4098ff', lineWidth: 2, title: 'Target weight' }, 1)
    const executedWeight = chart.addSeries(LineSeries, { color: '#94a8b7', lineWidth: 1, lineStyle: LineStyle.Dashed, title: 'Executed' }, 1)
    const reward = chart.addSeries(LineSeries, { color: '#36e37d', lineWidth: 2, title: 'Reward' }, 2)
    const cost = chart.addSeries(LineSeries, { color: '#f3b33d', lineWidth: 2, title: 'Cost', priceScaleId: 'left' }, 2)
    const equity = chart.addSeries(LineSeries, { color: '#4098ff', lineWidth: 2, title: 'RL equity' }, 3)
    const baseline = chart.addSeries(LineSeries, { color: '#94a8b7', lineWidth: 2, lineStyle: LineStyle.Dashed, title: 'Baseline' }, 3)
    const drawdown = chart.addSeries(LineSeries, { color: '#ff5b63', lineWidth: 1, title: 'Drawdown', priceScaleId: 'left' }, 3)
    const series = { candles, targetWeight, executedWeight, reward, cost, equity, baseline, drawdown }

    chart.panes()[0]?.setStretchFactor(4)
    chart.panes()[1]?.setStretchFactor(1.15)
    chart.panes()[2]?.setStretchFactor(1.15)
    chart.panes()[3]?.setStretchFactor(1.4)

    const crosshairHandler = (params: MouseEventParams<Time>) => {
      const currentData = dataRef.current
      callbacksRef.current.onPreviewRecord(
        currentData === null ? null : interactionRecord(chart, params, currentData),
      )
    }
    const clickHandler = (params: MouseEventParams<Time>) => {
      const currentData = dataRef.current
      if (currentData === null) return
      const record = interactionRecord(chart, params, currentData)
      if (record) callbacksRef.current.onCommitRecord(record)
    }

    let pointerOrigin: { x: number; y: number } | null = null
    let manualNavigationSignaled = false
    const pointerDownHandler = (event: PointerEvent) => {
      if (event.button !== 0) return
      pointerOrigin = { x: event.clientX, y: event.clientY }
      manualNavigationSignaled = false
    }
    const pointerMoveHandler = (event: PointerEvent) => {
      if (pointerOrigin === null || manualNavigationSignaled) return
      const distance = Math.hypot(
        event.clientX - pointerOrigin.x,
        event.clientY - pointerOrigin.y,
      )
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
    container.addEventListener('pointerdown', pointerDownHandler, { passive: true })
    window.addEventListener('pointermove', pointerMoveHandler, { capture: true, passive: true })
    window.addEventListener('pointerup', pointerEndHandler, { capture: true, passive: true })
    window.addEventListener('pointercancel', pointerEndHandler, { capture: true, passive: true })
    container.addEventListener('wheel', wheelHandler, { passive: true })
    chartRef.current = chart
    seriesRef.current = series
    markerRef.current = createSeriesMarkers(candles, [])

    return () => {
      chart.unsubscribeCrosshairMove(crosshairHandler)
      chart.unsubscribeClick(clickHandler)
      container.removeEventListener('pointerdown', pointerDownHandler)
      window.removeEventListener('pointermove', pointerMoveHandler, true)
      window.removeEventListener('pointerup', pointerEndHandler, true)
      window.removeEventListener('pointercancel', pointerEndHandler, true)
      container.removeEventListener('wheel', wheelHandler)
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

    const visibleMarkers = data.markers.filter((marker) =>
      marker.text === 'BUY' || marker.text === 'SELL'
        ? layers.positionEvents
        : marker.text === 'RISK' ? layers.riskEvents : true)
    markerRef.current?.setMarkers(visibleMarkers.map(({ sequence: _sequence, ...marker }) => ({
      ...marker,
      time: marker.time as UTCTimestamp,
    })) as SeriesMarker<Time>[])

    series.executedWeight.applyOptions({ visible: layers.executedWeight })
    series.reward.applyOptions({ visible: layers.rewardCost })
    series.cost.applyOptions({ visible: layers.rewardCost })
    series.baseline.applyOptions({ visible: layers.baseline })
    series.drawdown.applyOptions({ visible: layers.drawdown })
  }, [data, layers])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || data.candles.length === 0) return

    const rangeKey = `${symbol}|${timeframe}|${rangePreset}|${resetToken}`
    if (appliedRangeKey.current === rangeKey) return
    appliedRangeKey.current = rangeKey

    const latest = data.candles.at(-1)!.time
    const seconds = rangeSeconds(rangePreset)
    if (seconds === null) {
      chart.timeScale().fitContent()
    } else {
      chart.timeScale().setVisibleRange({
        from: Math.max(data.candles[0]!.time, latest - seconds) as UTCTimestamp,
        to: latest as UTCTimestamp,
      })
    }
  }, [data.candles.length, rangePreset, resetToken, symbol, timeframe])

  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series || committedSequence === null) return
    const time = data.timeBySequence.get(committedSequence)
    const record = time === undefined ? null : data.recordByTime.get(time) ?? null
    if (time === undefined || !record || record.close === null) return
    chart.setCrosshairPosition(record.close, time as UTCTimestamp, series.candles)
    if (followLatest && time === data.candles.at(-1)?.time) {
      chart.timeScale().scrollToRealTime()
    }
  }, [committedSequence, data, followLatest])

  const committedTime = committedSequence === null ? null : data.timeBySequence.get(committedSequence) ?? null
  const committedRecord = committedTime === null ? null : data.recordByTime.get(committedTime) ?? null
  const symbols = data.symbols.length > 0 ? data.symbols : [symbol]

  return (
    <div className="research-chart-column">
      <div className="research-chart-header">
        <div className="research-chart-identity">
          <select aria-label="Chart symbol" value={symbol} onChange={(event) => onSymbolChange(event.target.value)}>
            {symbols.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
          <div className="research-chart-segments" aria-label="時間足">
            {TIMEFRAME_OPTIONS.map((value) => (
              <button type="button" key={value} aria-pressed={timeframe === value} onClick={() => onTimeframeChange(value)}>{value}</button>
            ))}
          </div>
          <div className="research-chart-ohlc" aria-label="選択時点OHLC">
            <span>O {committedRecord?.open?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'}</span>
            <span>H {committedRecord?.high?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'}</span>
            <span>L {committedRecord?.low?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'}</span>
            <span>C {committedRecord?.close?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'}</span>
          </div>
        </div>
        <div className="research-chart-segments" aria-label="表示期間">
          {RANGE_OPTIONS.map((option) => (
            <button type="button" key={option.value} aria-pressed={rangePreset === option.value} onClick={() => onRangePresetChange(option.value)}>{option.label}</button>
          ))}
        </div>
      </div>

      {data.candles.length === 0 ? (
        <div className="research-chart-empty">選択したSymbol・時間足で表示可能なテレメトリがありません。</div>
      ) : null}
      <div
        ref={containerRef}
        className="research-chart-canvas"
        role="img"
        aria-label={`${symbol} ${timeframe} 市場・方策・学習・成績の同期チャート`}
      />
    </div>
  )
}
