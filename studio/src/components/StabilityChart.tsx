import type { StabilityFold } from '../data/types'

interface StabilityChartProps {
  folds: StabilityFold[]
}

function percent(value: number) {
  return ((value + 1) / 3) * 100
}

export function StabilityChart({ folds }: StabilityChartProps) {
  if (folds.length === 0) return <div className="chart-empty">walk-forward evidenceがありません。</div>
  return (
    <div className="stability-chart" role="img" aria-label="foldごとのSharpe分布">
      <div className="stability-axis"><span>-1.0</span><span>0</span><span>1.0</span><span>2.0</span></div>
      {folds.map((fold) => (
        <div className="stability-row" key={fold.label}>
          <span className="stability-label">{fold.label}</span>
          <div className="stability-track">
            <div
              className="stability-range"
              style={{ left: `${percent(fold.low)}%`, width: `${percent(fold.high) - percent(fold.low)}%` }}
            />
            <i className="stability-zero" />
            <i className="stability-median" style={{ left: `${percent(fold.median)}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}
