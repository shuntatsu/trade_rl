import { AlertTriangle, CheckCircle2, CircleDot, Info, ServerCog } from 'lucide-react'

import { LineChart } from '../components/LineChart'
import { MetricRing } from '../components/MetricRing'
import { Panel } from '../components/Panel'
import { StabilityChart } from '../components/StabilityChart'
import type { StudioOverview } from '../data/types'

interface DashboardPageProps {
  overview: StudioOverview
}

function metric(value: number | null, digits = 2): string {
  return value === null ? '—' : value.toFixed(digits)
}

function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`
}

export function DashboardPage({ overview }: DashboardPageProps) {
  const latestDataset = overview.latestDataset
  return (
    <div className="dashboard-grid">
      <div className="dashboard-row dashboard-row--top">
        <Panel title="システム概要" index={1} accent="green" className="system-panel">
          <div className="metrics-grid">
            {overview.system.metrics.map((item, index) => (
              <MetricRing key={item.label} {...item} tone={index === 3 ? 'blue' : 'green'} />
            ))}
          </div>
        </Panel>

        <Panel title="最新データセット" index={2} accent="cyan" className="dataset-panel">
          {latestDataset ? (
            <div className="dataset-card">
              <div className="dataset-card__title"><ServerCog size={16} aria-hidden="true" />{latestDataset.name}</div>
              <p>{latestDataset.market} / {latestDataset.symbols.join(', ')}</p>
              <p>{latestDataset.timeframes.join('/')} ・ {latestDataset.range}</p>
              <div className="dataset-card__footer">
                <span className={`status-valid${latestDataset.status === 'INVALID' ? ' status-valid--invalid' : ''}`}><CheckCircle2 size={12} aria-hidden="true" />{latestDataset.status}</span>
                <strong>{latestDataset.featureCount} features</strong>
                <small>更新: {latestDataset.updated}</small>
              </div>
            </div>
          ) : <div className="dashboard-empty">検証済みデータセットなし</div>}
        </Panel>

        <Panel title="実行中のジョブ" index={3} accent="blue" className="jobs-panel">
          <div className={`jobs-list${overview.activeJobs.length === 0 ? ' jobs-list--empty' : ''}`}>
            {overview.activeJobs.length === 0 ? <div className="dashboard-empty">実行中ジョブなし</div> : null}
            {overview.activeJobs.slice(0, 2).map((job) => (
              <article className="job-row" key={job.id}>
                <div className="job-row__top"><strong>{job.id}</strong><span>{job.progress}%</span></div>
                <div className="job-row__meta">{job.algorithm} ・ {job.phase} ・ {job.seedProgress}</div>
                <div className="progress-track"><i style={{ width: `${job.progress}%` }} /></div>
              </article>
            ))}
          </div>
        </Panel>
      </div>

      <div className="dashboard-row dashboard-row--middle">
        <Panel title="最新の実験結果サマリー" index={4} accent="green" className="runs-panel">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Run ID</th><th>アルゴリズム</th><th>期間</th><th>Sharpe</th><th>Max DD</th><th>判定</th></tr></thead>
              <tbody>
                {overview.runs.length === 0 ? <tr><td colSpan={6}>run artifactがありません。</td></tr> : null}
                {overview.runs.slice(0, 4).map((run) => (
                  <tr key={run.id}>
                    <td>{run.id}</td><td>{run.algorithm}</td><td>{run.period}</td><td>{metric(run.sharpe)}</td><td>{percent(run.maxDrawdown)}</td>
                    <td><span className="run-status run-status--no-go">{run.productionStatus}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="直近のアラート" index={5} accent="purple" className="alerts-panel">
          <div className={`alerts-list${overview.alerts.length === 0 ? ' alerts-list--empty' : ''}`}>
            {overview.alerts.length === 0 ? <div className="dashboard-empty">新しいアラートなし</div> : null}
            {overview.alerts.slice(0, 4).map((alert) => (
              <article className="alert-row" key={`${alert.message}-${alert.age}`}>
                <span className={`alert-tag alert-tag--${alert.level}`}>
                  {alert.level === 'warning' ? <AlertTriangle size={12} aria-hidden="true" /> : <Info size={12} aria-hidden="true" />}
                  {alert.level.toUpperCase()}
                </span>
                <p>{alert.message}</p>
                <small>{alert.age}</small>
              </article>
            ))}
          </div>
        </Panel>
      </div>

      <div className="dashboard-row dashboard-row--bottom">
        <Panel title="ベースライン比較" index={6} accent="amber" className="equity-panel"><LineChart points={overview.equity} /></Panel>
        <Panel title="ウォークフォワード安定性" index={7} accent="blue" className="stability-panel"><StabilityChart folds={overview.stability} /></Panel>
        <Panel title="Production Status" index={8} accent="red" className="assessment-panel">
          <div className="assessment">
            <div className="assessment__status"><CircleDot size={18} aria-hidden="true" />{overview.assessment.status}</div>
            <span>主因:</span>
            <ul>{overview.assessment.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
            <button type="button">詳細を見る →</button>
          </div>
        </Panel>
      </div>
    </div>
  )
}
