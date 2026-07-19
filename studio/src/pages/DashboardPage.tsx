import { AlertTriangle, CheckCircle2, CircleDot, Info, ServerCog } from 'lucide-react'

import { LineChart } from '../components/LineChart'
import { MetricRing } from '../components/MetricRing'
import { Panel } from '../components/Panel'
import { StabilityChart } from '../components/StabilityChart'
import type { StudioOverview } from '../data/types'

interface DashboardPageProps {
  overview: StudioOverview
}

export function DashboardPage({ overview }: DashboardPageProps) {
  return (
    <div className="dashboard-grid">
      <div className="dashboard-row dashboard-row--top">
        <Panel title="システム概要" index={1} accent="green" className="system-panel">
          <div className="metrics-grid">
            {overview.system.metrics.map((metric, index) => (
              <MetricRing key={metric.label} {...metric} tone={index === 3 ? 'blue' : 'green'} />
            ))}
          </div>
        </Panel>

        <Panel title="最新データセット" index={2} accent="cyan" className="dataset-panel">
          <div className="dataset-card">
            <div className="dataset-card__title"><ServerCog size={16} aria-hidden="true" />{overview.latestDataset.name}</div>
            <p>{overview.latestDataset.market} / {overview.latestDataset.symbols.join(', ')}</p>
            <p>{overview.latestDataset.timeframes.join('/')} ・ {overview.latestDataset.range}</p>
            <div className="dataset-card__footer">
              <span className="status-valid"><CheckCircle2 size={12} aria-hidden="true" />{overview.latestDataset.status}</span>
              <strong>{overview.latestDataset.featureCount} features</strong>
              <small>更新: {overview.latestDataset.updated}</small>
            </div>
          </div>
        </Panel>

        <Panel title="実行中のジョブ" index={3} accent="blue" className="jobs-panel">
          <div className="jobs-list">
            {overview.activeJobs.map((job) => (
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
                {overview.runs.map((run) => (
                  <tr key={run.id}>
                    <td>{run.id}</td><td>{run.algorithm}</td><td>{run.period}</td><td>{run.sharpe.toFixed(2)}</td><td>{run.maxDrawdown.toFixed(1)}%</td>
                    <td><span className={`run-status run-status--${run.status === 'GO' ? 'go' : 'no-go'}`}>{run.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="直近のアラート" index={5} accent="purple" className="alerts-panel">
          <div className="alerts-list">
            {overview.alerts.map((alert) => (
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
        <Panel title="ベースライン比較" index={6} accent="amber" className="equity-panel">
          <LineChart points={overview.equity} />
        </Panel>

        <Panel title="ウォークフォワード安定性" index={7} accent="blue" className="stability-panel">
          <StabilityChart folds={overview.stability} />
        </Panel>

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
