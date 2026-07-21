import type { TrainingTelemetryRecord } from '../data/types'

interface MarketReplayChartProps {
  records: TrainingTelemetryRecord[]
  cursorSequence: number | null
  compressed: boolean
}

const WIDTH = 1_000
const HEIGHT = 380
const PADDING = { top: 28, right: 62, bottom: 42, left: 18 }

function weightDelta(record: TrainingTelemetryRecord): number {
  const length = Math.max(record.weightsBefore.length, record.weightsAfter.length)
  let delta = 0
  for (let index = 0; index < length; index += 1) {
    delta += (record.weightsAfter[index] ?? 0) - (record.weightsBefore[index] ?? 0)
  }
  return delta
}

function formatPrice(value: number): string {
  return value.toLocaleString('ja-JP', { maximumFractionDigits: 1 })
}

export function MarketReplayChart({ records, cursorSequence, compressed }: MarketReplayChartProps) {
  const selected = (compressed ? records.filter((record) => record.eventType !== 'rollout') : records)
    .filter((record) => record.close !== null)
    .slice(-96)
  if (selected.length === 0) {
    return (
      <div className="live-chart live-chart--empty" role="img" aria-label="市場リプレイ データ待機中">
        <strong>市場リプレイを準備しています</strong>
        <span>学習テレメトリが到着すると価格と探索行動を表示します。</span>
      </div>
    )
  }

  const lows = selected.map((record) => record.low ?? record.close ?? 0)
  const highs = selected.map((record) => record.high ?? record.close ?? 0)
  const minimum = Math.min(...lows)
  const maximum = Math.max(...highs)
  const spread = Math.max(maximum - minimum, Math.abs(maximum) * 0.001, 1)
  const low = minimum - spread * 0.08
  const high = maximum + spread * 0.08
  const chartWidth = WIDTH - PADDING.left - PADDING.right
  const chartHeight = HEIGHT - PADDING.top - PADDING.bottom
  const x = (index: number) => PADDING.left + (selected.length === 1 ? chartWidth / 2 : index * chartWidth / (selected.length - 1))
  const y = (value: number) => PADDING.top + (high - value) / (high - low) * chartHeight
  const candleWidth = Math.max(3, Math.min(10, chartWidth / Math.max(selected.length, 1) * 0.55))
  const cursorIndex = Math.max(0, selected.findIndex((record) => record.sequence === cursorSequence))
  const resolvedCursor = cursorSequence === null || cursorIndex < 0 ? selected.length - 1 : cursorIndex
  const cursorRecord = selected[resolvedCursor]
  const navigatorPoints = selected.map((record, index) => `${x(index)},${HEIGHT - 12 - ((record.close ?? low) - low) / (high - low) * 18}`).join(' ')

  return (
    <div className="live-chart" role="img" aria-label={`${selected.at(-1)?.symbol ?? '市場'} 市場リプレイ`}> 
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="replay-area" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--live-cyan)" stopOpacity="0.18" />
            <stop offset="100%" stopColor="var(--live-cyan)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const gridY = PADDING.top + chartHeight * ratio
          const label = high - (high - low) * ratio
          return (
            <g key={ratio}>
              <line x1={PADDING.left} x2={WIDTH - PADDING.right} y1={gridY} y2={gridY} className="live-chart__grid" />
              <text x={WIDTH - PADDING.right + 8} y={gridY + 4} className="live-chart__axis">{formatPrice(label)}</text>
            </g>
          )
        })}
        {selected.map((record, index) => {
          const open = record.open ?? record.close ?? 0
          const close = record.close ?? open
          const recordHigh = record.high ?? Math.max(open, close)
          const recordLow = record.low ?? Math.min(open, close)
          const rising = close >= open
          const candleX = x(index)
          const bodyTop = y(Math.max(open, close))
          const bodyHeight = Math.max(1.5, Math.abs(y(open) - y(close)))
          const delta = weightDelta(record)
          const markerY = delta >= 0 ? y(recordLow) + 18 : y(recordHigh) - 18
          return (
            <g key={`${record.sequence}-${record.environmentId}`}>
              <line x1={candleX} x2={candleX} y1={y(recordHigh)} y2={y(recordLow)} className={rising ? 'live-candle live-candle--up' : 'live-candle live-candle--down'} />
              <rect x={candleX - candleWidth / 2} y={bodyTop} width={candleWidth} height={bodyHeight} rx="1" className={rising ? 'live-candle live-candle--up' : 'live-candle live-candle--down'} />
              {record.eventType === 'position' ? (
                <path
                  d={delta >= 0
                    ? `M ${candleX} ${markerY - 8} L ${candleX - 7} ${markerY + 5} L ${candleX + 7} ${markerY + 5} Z`
                    : `M ${candleX} ${markerY + 8} L ${candleX - 7} ${markerY - 5} L ${candleX + 7} ${markerY - 5} Z`}
                  className={delta >= 0 ? 'live-marker live-marker--buy' : 'live-marker live-marker--sell'}
                />
              ) : null}
              {record.eventType === 'risk' ? <circle cx={candleX} cy={markerY} r="5" className="live-marker live-marker--risk" /> : null}
            </g>
          )
        })}
        <line x1={x(resolvedCursor)} x2={x(resolvedCursor)} y1={PADDING.top} y2={PADDING.top + chartHeight} className="live-chart__cursor" />
        <circle cx={x(resolvedCursor)} cy={PADDING.top + chartHeight} r="5" className="live-chart__cursor-dot" />
        <rect x={Math.min(WIDTH - 142, Math.max(8, x(resolvedCursor) - 54))} y="4" width="108" height="22" rx="7" className="live-chart__cursor-label" />
        <text x={Math.min(WIDTH - 88, Math.max(62, x(resolvedCursor)))} y="19" textAnchor="middle" className="live-chart__cursor-text">
          Step {cursorRecord?.globalStep.toLocaleString('ja-JP')}
        </text>
        <polyline points={navigatorPoints} className="live-chart__navigator" />
      </svg>
      <div className="live-chart__legend" aria-hidden="true">
        <span><i className="legend-dot legend-dot--buy" />BUY</span>
        <span><i className="legend-dot legend-dot--sell" />SELL</span>
        <span><i className="legend-line" />リプレイ位置</span>
      </div>
    </div>
  )
}
