import { act, fireEvent, render, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { DEFAULT_RESEARCH_CHART_LAYERS } from './researchChartModel'

const runtime = vi.hoisted(() => {
  const handlers: {
    crosshair?: (value: unknown) => void
    click?: (value: unknown) => void
    range?: () => void
  } = {}
  const emitRangeAsync = () => setTimeout(() => handlers.range?.(), 0)
  const pane = { setStretchFactor: vi.fn() }
  const timeScale = {
    fitContent: vi.fn(emitRangeAsync),
    setVisibleRange: vi.fn(emitRangeAsync),
    scrollToRealTime: vi.fn(emitRangeAsync),
    subscribeVisibleLogicalRangeChange: vi.fn((handler: () => void) => { handlers.range = handler }),
    unsubscribeVisibleLogicalRangeChange: vi.fn(),
  }
  const series: Array<{ setData: ReturnType<typeof vi.fn>; applyOptions: ReturnType<typeof vi.fn> }> = []
  const chart = {
    addSeries: vi.fn((_definition: unknown, _options: unknown, _paneIndex: number) => {
      const next = {
        setData: vi.fn(emitRangeAsync),
        applyOptions: vi.fn(),
      }
      series.push(next)
      return next
    }),
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
  return { handlers, pane, timeScale, series, chart, markerApi }
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
  ResearchChartWorkspace,
  type ResearchChartWorkspaceProps,
} from './ResearchChartWorkspace'

function telemetry(
  sequence: number,
  eventType: TrainingTelemetryRecord['eventType'] = 'rollout',
): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: `2026-07-31T08:${String(sequence).padStart(2, '0')}:00+00:00`,
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType,
    marketIndex: 100 + sequence,
    marketTime: `2026-07-31T08:${String(sequence).padStart(2, '0')}:00.000000000`,
    symbol: 'BTCUSDT',
    open: 100,
    high: 110,
    low: 90,
    close: 105 + sequence,
    action: [0.2],
    executedTarget: [0.18],
    weightsBefore: [0.1],
    weightsAfter: [0.2],
    portfolioValue: 1_000 + sequence,
    baselinePortfolioValue: 995 + sequence,
    reward: 0.1,
    drawdown: 0.01,
    intervalCost: 1.5,
    intervalReturn: 0.001,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

function renderWorkspace(overrides: Partial<ResearchChartWorkspaceProps> = {}) {
  const records = [telemetry(1), telemetry(2, 'position')]
  const props: ResearchChartWorkspaceProps = {
    records,
    symbol: 'BTCUSDT',
    timeframe: '15m',
    rangePreset: '24h',
    layers: DEFAULT_RESEARCH_CHART_LAYERS,
    followLatest: true,
    committedSequence: 2,
    resetToken: 0,
    onSymbolChange: vi.fn(),
    onTimeframeChange: vi.fn(),
    onRangePresetChange: vi.fn(),
    onPreviewRecord: vi.fn(),
    onCommitRecord: vi.fn(),
    onManualNavigation: vi.fn(),
    ...overrides,
  }
  return { ...render(<ResearchChartWorkspace {...props} />), props }
}

async function flushRangeNotifications() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  runtime.series.length = 0
  runtime.handlers.crosshair = undefined
  runtime.handlers.click = undefined
  runtime.handlers.range = undefined
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
    const bucketTime = Math.floor(Date.parse('2026-07-31T08:00:00Z') / 1_000 / 900) * 900

    act(() => runtime.handlers.crosshair?.({ time: bucketTime }))
    act(() => runtime.handlers.click?.({ time: bucketTime }))

    expect(onPreviewRecord).toHaveBeenCalledWith(expect.objectContaining({ sequence: 2 }))
    expect(onCommitRecord).toHaveBeenCalledWith(expect.objectContaining({ sequence: 2 }))
  })

  it('does not classify delayed programmatic updates as manual navigation', async () => {
    const onManualNavigation = vi.fn()
    renderWorkspace({ onManualNavigation })

    await waitFor(() => expect(runtime.series[0]?.setData).toHaveBeenCalled())
    await flushRangeNotifications()

    expect(runtime.timeScale.subscribeVisibleLogicalRangeChange).not.toHaveBeenCalled()
    expect(onManualNavigation).not.toHaveBeenCalled()
  })

  it('classifies drag and wheel gestures as manual navigation', async () => {
    const onManualNavigation = vi.fn()
    const view = renderWorkspace({ onManualNavigation })
    const chartSurface = view.container.querySelector('.research-chart-canvas')
    expect(chartSurface).toBeInstanceOf(HTMLElement)

    fireEvent.pointerDown(chartSurface!, { button: 0, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(window, { clientX: 103, clientY: 103 })
    expect(onManualNavigation).not.toHaveBeenCalled()

    fireEvent.pointerMove(window, { clientX: 112, clientY: 100 })
    fireEvent.pointerMove(window, { clientX: 140, clientY: 100 })
    expect(onManualNavigation).toHaveBeenCalledTimes(1)

    fireEvent.pointerUp(window)
    fireEvent.wheel(chartSurface!, { deltaY: 100 })
    expect(onManualNavigation).toHaveBeenCalledTimes(2)
  })

  it('does not reset a manual viewport when new records arrive', async () => {
    const view = renderWorkspace({ followLatest: false })
    await waitFor(() => expect(runtime.timeScale.setVisibleRange).toHaveBeenCalled())
    await flushRangeNotifications()
    vi.clearAllMocks()

    view.rerender(
      <ResearchChartWorkspace
        {...view.props}
        followLatest={false}
        records={[...view.props.records, telemetry(3)]}
      />,
    )

    await waitFor(() => expect(runtime.series[0]?.setData).toHaveBeenCalled())
    await flushRangeNotifications()
    expect(runtime.timeScale.setVisibleRange).not.toHaveBeenCalled()
    expect(runtime.timeScale.fitContent).not.toHaveBeenCalled()
    expect(runtime.timeScale.scrollToRealTime).not.toHaveBeenCalled()
  })

  it('removes the chart on unmount', async () => {
    const view = renderWorkspace()
    await waitFor(() => expect(runtime.chart.subscribeClick).toHaveBeenCalled())
    view.unmount()
    expect(runtime.chart.remove).toHaveBeenCalledTimes(1)
  })
})
