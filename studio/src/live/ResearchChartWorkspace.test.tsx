import { act, render, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { DEFAULT_RESEARCH_CHART_LAYERS } from './researchChartModel'

const runtime = vi.hoisted(() => {
  const handlers: { crosshair?: (value: unknown) => void; click?: (value: unknown) => void } = {}
  const pane = { setStretchFactor: vi.fn() }
  const timeScale = {
    fitContent: vi.fn(),
    setVisibleRange: vi.fn(),
    scrollToRealTime: vi.fn(),
    subscribeVisibleLogicalRangeChange: vi.fn(),
    unsubscribeVisibleLogicalRangeChange: vi.fn(),
  }
  const series = Array.from({ length: 8 }, () => ({
    setData: vi.fn(),
    applyOptions: vi.fn(),
  }))
  const chart = {
    addSeries: vi.fn((_definition: unknown, _options: unknown, _paneIndex: number) => series.shift()),
    panes: vi.fn(() => [pane, pane, pane, pane]),
    subscribeCrosshairMove: vi.fn((handler: (value: unknown) => void) => { handlers.crosshair = handler }),
    unsubscribeCrosshairMove: vi.fn(),
    subscribeClick: vi.fn((handler: (value: unknown) => void) => { handlers.click = handler }),
    unsubscribeClick: vi.fn(),
    timeScale: vi.fn(() => timeScale),
    setCrosshairPosition: vi.fn(),
    remove: vi.fn(),
  }
  const markerApi = { setMarkers: vi.fn() }
  return { handlers, pane, timeScale, chart, markerApi }
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

import { ResearchChartWorkspace } from './ResearchChartWorkspace'

function telemetry(sequence: number, eventType: TrainingTelemetryRecord['eventType'] = 'rollout'): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1', sequence, recordedAt: `2026-07-31T08:${String(sequence).padStart(2, '0')}:00+00:00`,
    globalStep: sequence * 32, environmentStep: sequence, seed: 7, environmentId: 0, episodeId: 1,
    eventType, marketIndex: 100 + sequence, marketTime: `2026-07-31T08:${String(sequence).padStart(2, '0')}:00.000000000`,
    symbol: 'BTCUSDT', open: 100, high: 110, low: 90, close: 105 + sequence,
    action: [0.2], executedTarget: [0.18], weightsBefore: [0.1], weightsAfter: [0.2],
    portfolioValue: 1_000 + sequence, baselinePortfolioValue: 995 + sequence, reward: 0.1,
    drawdown: 0.01, intervalCost: 1.5, intervalReturn: 0.001, riskReasons: [],
    emergencyDeleverage: false, terminated: false, truncated: false,
  }
}

function renderWorkspace(overrides: Record<string, unknown> = {}) {
  const records = [telemetry(1), telemetry(2, 'position')]
  const props = {
    records, symbol: 'BTCUSDT', timeframe: '15m' as const, rangePreset: '24h' as const,
    layers: DEFAULT_RESEARCH_CHART_LAYERS, followLatest: true, committedSequence: 2,
    resetToken: 0, onSymbolChange: vi.fn(), onTimeframeChange: vi.fn(), onRangePresetChange: vi.fn(),
    onPreviewRecord: vi.fn(), onCommitRecord: vi.fn(), onManualNavigation: vi.fn(),
    ...overrides,
  }
  return { ...render(<ResearchChartWorkspace {...props} />), props }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('ResearchChartWorkspace', () => {
  it('creates one chart with four synchronized pane indices', async () => {
    renderWorkspace()
    await waitFor(() => expect(runtime.chart.addSeries).toHaveBeenCalledTimes(8))
    expect(runtime.chart.addSeries.mock.calls.map((call) => call[2])).toEqual([0, 1, 1, 2, 2, 3, 3, 3])
    expect(runtime.pane.setStretchFactor).toHaveBeenCalledTimes(4)
  })

  it('previews on crosshair and commits on click', async () => {
    const onPreviewRecord = vi.fn()
    const onCommitRecord = vi.fn()
    renderWorkspace({ onPreviewRecord, onCommitRecord })
    await waitFor(() => expect(runtime.handlers.crosshair).toBeDefined())

    act(() => runtime.handlers.crosshair?.({ time: 1_775_117_700 }))
    act(() => runtime.handlers.click?.({ time: 1_775_117_700 }))

    expect(onPreviewRecord).toHaveBeenCalled()
    expect(onCommitRecord).toHaveBeenCalled()
  })

  it('removes the chart on unmount', async () => {
    const view = renderWorkspace()
    await waitFor(() => expect(runtime.chart.subscribeClick).toHaveBeenCalled())
    view.unmount()
    expect(runtime.chart.remove).toHaveBeenCalledTimes(1)
  })
})
