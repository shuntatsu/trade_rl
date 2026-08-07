import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { DEFAULT_RESEARCH_CHART_LAYERS } from './researchChartModel'

const runtime = vi.hoisted(() => {
  const baseTime = Math.floor(Date.parse('2026-05-27T08:15:00Z') / 1_000 / 900) * 900
  const handlers: {
    crosshair?: (value: unknown) => void
    click?: (value: unknown) => void
    range?: () => void
  } = {}
  const paneHeights = [360, 150, 110]
  const panes = paneHeights.map((height) => ({
    setStretchFactor: vi.fn(),
    getHeight: vi.fn(() => height),
  }))
  const timeScale = {
    fitContent: vi.fn(),
    setVisibleRange: vi.fn(),
    scrollToRealTime: vi.fn(),
    coordinateToTime: vi.fn((coordinate: number) => baseTime + Math.round((Number.isFinite(coordinate) ? coordinate : 0) * 9)),
    coordinateToLogical: vi.fn(() => 1),
    timeToCoordinate: vi.fn((time: number) => (time - baseTime) / 9),
    subscribeVisibleLogicalRangeChange: vi.fn((handler: () => void) => { handlers.range = handler }),
    unsubscribeVisibleLogicalRangeChange: vi.fn(),
  }
  const series: Array<{
    setData: ReturnType<typeof vi.fn>
    applyOptions: ReturnType<typeof vi.fn>
    createPriceLine: ReturnType<typeof vi.fn>
    removePriceLine: ReturnType<typeof vi.fn>
  }> = []
  const chart = {
    addSeries: vi.fn((_definition: unknown, _options: unknown, _paneIndex: number) => {
      const next = {
        setData: vi.fn(),
        applyOptions: vi.fn(),
        createPriceLine: vi.fn((options: unknown) => ({ options })),
        removePriceLine: vi.fn(),
      }
      series.push(next)
      return next
    }),
    panes: vi.fn(() => panes),
    subscribeCrosshairMove: vi.fn((handler: (value: unknown) => void) => { handlers.crosshair = handler }),
    unsubscribeCrosshairMove: vi.fn(),
    subscribeClick: vi.fn((handler: (value: unknown) => void) => { handlers.click = handler }),
    unsubscribeClick: vi.fn(),
    timeScale: vi.fn(() => timeScale),
    setCrosshairPosition: vi.fn(),
    remove: vi.fn(),
  }
  const markerApi = { setMarkers: vi.fn() }
  return { handlers, panes, timeScale, series, chart, markerApi }
})

vi.mock('lightweight-charts', () => ({
  CandlestickSeries: { type: 'Candlestick' },
  LineSeries: { type: 'Line' },
  ColorType: { Solid: 'solid' },
  CrosshairMode: { Normal: 0 },
  LineStyle: { Dashed: 2 },
  createChart: vi.fn(() => runtime.chart),
  createSeriesMarkers: vi.fn(() => runtime.markerApi),
}))

import {
  SynchronizedResearchChartWorkspace,
  type SynchronizedResearchChartWorkspaceProps,
} from './SynchronizedResearchChartWorkspace'

function telemetry(
  sequence: number,
  overrides: Partial<TrainingTelemetryRecord> = {},
): TrainingTelemetryRecord {
  const instant = new Date(Date.parse('2026-05-27T08:00:00Z') + sequence * 15 * 60 * 1_000)
  const marketTime = instant.toISOString()
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: instant.toISOString(),
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType: 'rollout',
    marketIndex: 100 + sequence,
    marketTime,
    symbol: 'BTCUSDT',
    open: 100 + sequence,
    high: 110 + sequence,
    low: 90 + sequence,
    close: 105 + sequence,
    action: [0.2],
    executedTarget: [0.18],
    weightsBefore: [0],
    weightsAfter: [0],
    portfolioValue: 1_000 + sequence * 10,
    baselinePortfolioValue: 995 + sequence * 8,
    reward: 0.1 * sequence,
    drawdown: 0.01 * sequence,
    intervalCost: 1.5,
    intervalReturn: 0.001,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
    ...overrides,
  }
}

function records(): TrainingTelemetryRecord[] {
  return [
    telemetry(1),
    telemetry(2, { eventType: 'position', weightsBefore: [0], weightsAfter: [0.4] }),
    telemetry(3, { weightsBefore: [0.4], weightsAfter: [0.4] }),
    telemetry(4, { eventType: 'position', weightsBefore: [0.4], weightsAfter: [0] }),
  ]
}

function renderWorkspace(overrides: Partial<SynchronizedResearchChartWorkspaceProps> = {}) {
  const props: SynchronizedResearchChartWorkspaceProps = {
    records: records(),
    symbol: 'BTCUSDT',
    timeframe: '15m',
    rangePreset: '24h',
    layers: DEFAULT_RESEARCH_CHART_LAYERS,
    followLatest: true,
    committedSequence: 4,
    resetToken: 0,
    onSymbolChange: vi.fn(),
    onTimeframeChange: vi.fn(),
    onRangePresetChange: vi.fn(),
    onPreviewRecord: vi.fn(),
    onCommitRecord: vi.fn(),
    onManualNavigation: vi.fn(),
    ...overrides,
  }
  return { ...render(<SynchronizedResearchChartWorkspace {...props} />), props }
}

beforeEach(() => {
  vi.clearAllMocks()
  runtime.series.length = 0
  runtime.handlers.crosshair = undefined
  runtime.handlers.click = undefined
  runtime.handlers.range = undefined
  Object.defineProperty(globalThis, 'requestAnimationFrame', {
    configurable: true,
    value: (callback: FrameRequestCallback) => {
      callback(0)
      return 1
    },
  })
  Object.defineProperty(globalThis, 'ResizeObserver', {
    configurable: true,
    value: class {
      observe() {}
      disconnect() {}
    },
  })
})

describe('SynchronizedResearchChartWorkspace', () => {
  it('creates one Lightweight Charts instance with exactly three synchronized pane indices', async () => {
    renderWorkspace()

    await waitFor(() => expect(runtime.chart.addSeries).toHaveBeenCalledTimes(9))
    expect(runtime.chart.addSeries.mock.calls.map((call) => call[2])).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2])
    expect(runtime.panes[0]?.setStretchFactor).toHaveBeenCalledWith(4.8)
    expect(runtime.panes[1]?.setStretchFactor).toHaveBeenCalledWith(2)
    expect(runtime.panes[2]?.setStretchFactor).toHaveBeenCalledWith(1.4)
  })

  it('creates direct latest-value labels without leaving the built-in labels visible', async () => {
    renderWorkspace()

    await waitFor(() => expect(runtime.series[0]?.createPriceLine).toHaveBeenCalled())
    const titles = runtime.series.flatMap((series) => series.createPriceLine.mock.calls.map((call) => call[0]?.title))
    expect(titles).toEqual(expect.arrayContaining(['BTCUSDT', 'Portfolio', 'Baseline', 'Drawdown', 'Gross exposure']))
    for (const index of [0, 3, 4, 5, 6]) {
      expect(runtime.chart.addSeries.mock.calls[index]?.[1]).toEqual(expect.objectContaining({
        lastValueVisible: false,
        priceLineVisible: false,
      }))
    }
  })

  it('changes emphasis without replacing panes or hiding context', async () => {
    const user = userEvent.setup()
    renderWorkspace()
    await waitFor(() => expect(runtime.chart.addSeries).toHaveBeenCalledTimes(9))

    await user.click(screen.getByRole('button', { name: 'Trade focus' }))
    expect(runtime.panes[0]?.setStretchFactor).toHaveBeenLastCalledWith(5.4)
    expect(runtime.series[1]?.applyOptions).toHaveBeenCalledWith(expect.objectContaining({ lineWidth: 3 }))
    expect(runtime.series[3]?.applyOptions).toHaveBeenCalledWith(expect.objectContaining({ lineWidth: 1 }))

    await user.click(screen.getByRole('button', { name: 'Risk focus' }))
    expect(runtime.panes[2]?.setStretchFactor).toHaveBeenLastCalledWith(2.7)
    expect(runtime.series[5]?.applyOptions).toHaveBeenCalledWith(expect.objectContaining({ lineWidth: 3 }))
    expect(runtime.series[6]?.applyOptions).toHaveBeenCalledWith(expect.objectContaining({ lineWidth: 3 }))
    expect(runtime.chart.addSeries).toHaveBeenCalledTimes(9)
  })

  it('previews on the crosshair and commits on click', async () => {
    const onPreviewRecord = vi.fn()
    const onCommitRecord = vi.fn()
    renderWorkspace({ onPreviewRecord, onCommitRecord })
    await waitFor(() => expect(runtime.handlers.crosshair).toBeDefined())
    const bucketTime = Math.floor(Date.parse('2026-05-27T08:15:00Z') / 1_000 / 900) * 900

    act(() => runtime.handlers.crosshair?.({ time: bucketTime }))
    act(() => runtime.handlers.click?.({ time: bucketTime }))

    expect(onPreviewRecord).toHaveBeenCalledWith(expect.objectContaining({ sequence: 1 }))
    expect(onCommitRecord).toHaveBeenCalledWith(expect.objectContaining({ sequence: 1 }))
  })

  it('draws a contract-safe trade lifecycle band without replacing the chart', async () => {
    const view = renderWorkspace()
    const canvas = view.container.querySelector('.synchronized-chart-canvas') as HTMLElement
    Object.defineProperty(canvas, 'clientWidth', { configurable: true, value: 900 })
    Object.defineProperty(canvas, 'clientHeight', { configurable: true, value: 620 })

    act(() => runtime.handlers.range?.())

    await waitFor(() => expect(view.container.querySelectorAll('.synchronized-trade-band')).toHaveLength(1))
    expect(view.container.querySelector('.synchronized-trade-band--long')).toBeInTheDocument()
    expect(view.container.querySelector('.synchronized-trade-band--closed')).toBeInTheDocument()
  })

  it('creates an opt-in range selection while normal chart navigation stays available otherwise', async () => {
    const user = userEvent.setup()
    const view = renderWorkspace()
    const canvas = view.container.querySelector('.synchronized-chart-canvas') as HTMLElement
    Object.defineProperty(canvas, 'clientWidth', { configurable: true, value: 900 })
    Object.defineProperty(canvas, 'clientHeight', { configurable: true, value: 620 })
    canvas.getBoundingClientRect = () => ({
      x: 0, y: 0, top: 0, left: 0, right: 900, bottom: 620, width: 900, height: 620,
      toJSON: () => ({}),
    })

    expect(view.container.querySelector('.synchronized-range-interaction')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Range' }))
    const surface = view.container.querySelector('.synchronized-range-interaction') as HTMLElement
    surface.setPointerCapture = vi.fn()
    surface.releasePointerCapture = vi.fn()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 7, clientX: 20, clientY: 100 })
    fireEvent.pointerMove(surface, { pointerId: 7, clientX: 100, clientY: 100 })
    fireEvent.pointerUp(surface, { pointerId: 7, clientX: 100, clientY: 100 })

    expect(await screen.findByRole('button', { name: '範囲解除' })).toBeInTheDocument()
    expect(view.container.querySelector('.synchronized-range-selection')).toBeInTheDocument()
    expect(runtime.chart.timeScale).toHaveBeenCalled()
  })

  it('removes the chart and visible-range subscription on unmount', async () => {
    const view = renderWorkspace()
    await waitFor(() => expect(runtime.chart.subscribeClick).toHaveBeenCalled())

    view.unmount()

    expect(runtime.timeScale.unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1)
    expect(runtime.chart.remove).toHaveBeenCalledTimes(1)
  })
})