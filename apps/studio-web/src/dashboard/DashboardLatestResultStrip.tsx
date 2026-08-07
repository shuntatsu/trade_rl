import type { DashboardLatestResult } from './dashboardCockpitModel'

function metric(value: number | null, digits = 2): string {
  return value === null ? '—' : value.toFixed(digits)
}
function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`
}

export function DashboardLatestResultStrip({ result }: { result: DashboardLatestResult | null }) {
  return (
    <section className="dashboard-latest" aria-labelledby="dashboard-latest-title">
      <div className="dashboard-latest__identity">
        <span>LATEST EVALUATION</span>
        <h2 id="dashboard-latest-title">{result?.runId ?? 'No run'}</h2>
        <small>{result ? `${result.algorithm} · ${result.period}` : '評価結果がありません。'}</small>
      </div>
      <dl>
        <div><dt>Total return</dt><dd>{percent(result?.totalReturn ?? null)}</dd></div>
        <div><dt>Sharpe</dt><dd>{metric(result?.sharpe ?? null)}</dd></div>
        <div><dt>Max DD</dt><dd>{percent(result?.maxDrawdown ?? null)}</dd></div>
        <div><dt>Artifact</dt><dd className={result?.validationStatus === 'INVALID' ? 'text-danger' : 'text-positive'}>{result?.validationStatus ?? '—'}</dd></div>
        <div><dt>Release</dt><dd className="text-danger">{result?.productionStatus ?? 'NO-GO'}</dd></div>
      </dl>
    </section>
  )
}
