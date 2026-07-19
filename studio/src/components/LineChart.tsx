import type { EquityPoint } from '../data/types'

interface LineChartProps {
  points: EquityPoint[]
}

const WIDTH = 520
const HEIGHT = 150
const PADDING_X = 24
const PADDING_Y = 18

function pathFor(points: EquityPoint[], key: 'rl' | 'baseline') {
  const values = points.map((point) => point[key])
  const min = Math.min(...values, 0.8)
  const max = Math.max(...values, 2)
  return points
    .map((point, index) => {
      const x = PADDING_X + (index / Math.max(1, points.length - 1)) * (WIDTH - PADDING_X * 2)
      const y = HEIGHT - PADDING_Y - ((point[key] - min) / Math.max(0.01, max - min)) * (HEIGHT - PADDING_Y * 2)
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
    })
    .join(' ')
}

export function LineChart({ points }: LineChartProps) {
  return (
    <div className="line-chart">
      <div className="chart-legend" aria-label="チャート凡例">
        <span><i className="legend-swatch legend-swatch--green" />RL (PPO)</span>
        <span><i className="legend-swatch legend-swatch--blue" />Baseline</span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="RLとベースラインの資産曲線">
        {[0, 1, 2, 3].map((line) => {
          const y = PADDING_Y + line * ((HEIGHT - PADDING_Y * 2) / 3)
          return <line key={line} x1={PADDING_X} x2={WIDTH - PADDING_X} y1={y} y2={y} className="chart-grid-line" />
        })}
        <path d={pathFor(points, 'baseline')} className="chart-line chart-line--baseline" />
        <path d={pathFor(points, 'rl')} className="chart-line chart-line--rl" />
      </svg>
      <div className="chart-axis-labels">
        <span>{points.at(0)?.label}</span>
        <span>{points.at(Math.floor(points.length / 2))?.label}</span>
        <span>{points.at(-1)?.label}</span>
      </div>
    </div>
  )
}
