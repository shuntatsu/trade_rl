import { AlertTriangle, Activity, Database } from 'lucide-react'
import { useMemo, useState } from 'react'

import type { StudioApi } from '../api/studioApi'
import type { JobSummary, TrainingMetricGroup, TrainingMetricSeries } from '../data/types'
import { TrainingMetricChart, type TrainingMetricWindow } from './TrainingMetricChart'
import { TRAINING_METRIC_TAGS } from './trainingMetricGuards'
import { useTrainingMetrics } from './useTrainingMetrics'

interface TrainingDiagnosticsPanelProps {
  job: JobSummary | null
  seed: number | null
  api: StudioApi
}

const groupLabels: Record<TrainingMetricGroup, string> = {
  optimization: '最適化', policy: '方策', value: '価値関数', trading: '取引・リスク',
}
const defaults: Record<TrainingMetricGroup, string> = {
  optimization: 'train/learning_rate',
  policy: 'train/approx_kl',
  value: 'train/explained_variance',
  trading: 'trade_rl/drawdown_mean',
}

function latest(series: TrainingMetricSeries | undefined): number | null {
  return series?.points.at(-1)?.value ?? null
}

export function TrainingDiagnosticsPanel({ job, seed, api }: TrainingDiagnosticsPanelProps) {
  const [group, setGroup] = useState<TrainingMetricGroup>('optimization')
  const [selectedByGroup, setSelectedByGroup] = useState(defaults)
  const [selectedStep, setSelectedStep] = useState<number | null>(null)
  const [windowSize, setWindowSize] = useState<TrainingMetricWindow>(512)
  const metrics = useTrainingMetrics(job?.id ?? null, seed, [...TRAINING_METRIC_TAGS], api)
  const available = metrics.series.filter((series) => series.group === group)
  const selectedTag = available.some((series) => series.tag === selectedByGroup[group])
    ? selectedByGroup[group]
    : available[0]?.tag ?? defaults[group]
  const primary = metrics.series.find((series) => series.tag === selectedTag) ?? null
  const secondary = available.find((series) => series.tag !== selectedTag && series.unit === primary?.unit) ?? null
  const warnings = useMemo(() => {
    const messages: string[] = []
    const kl = latest(metrics.series.find((series) => series.tag === 'train/approx_kl'))
    const clip = latest(metrics.series.find((series) => series.tag === 'train/clip_fraction'))
    const explained = latest(metrics.series.find((series) => series.tag === 'train/explained_variance'))
    if (kl !== null && kl > 0.03) messages.push('Approx KL が診断目安 0.03 を超えています。')
    if (clip !== null && clip > 0.3) messages.push('Clip fraction が診断目安 0.3 を超えています。')
    if (explained !== null && explained < 0) messages.push('Explained variance が負です。')
    if (job?.status === 'running' && metrics.status?.available && metrics.status.lastStep === 0) messages.push('実行中ですが新しいscalar stepがありません。')
    return messages
  }, [job?.status, metrics.series, metrics.status])

  if (!job) return <section className="training-diagnostics training-diagnostics--empty">Runを選択してください。</section>
  return (
    <section className="training-diagnostics" aria-label="学習診断">
      <div className="training-diagnostics-warning"><AlertTriangle size={15} aria-hidden="true" />学習曲線は最適化状態の診断です。汎化性能や収益性は、Checkpoint検証およびWalk-forward評価で判断してください。</div>
      <div className="training-diagnostics-tabs" role="tablist" aria-label="学習指標グループ">
        {(Object.keys(groupLabels) as TrainingMetricGroup[]).map((value) => <button key={value} type="button" role="tab" aria-selected={group === value} onClick={() => { setGroup(value); setSelectedStep(null) }}>{groupLabels[value]}</button>)}
      </div>
      <div className="training-diagnostics-toolbar">
        <label>Metric<select aria-label="学習指標" value={selectedTag} onChange={(event) => setSelectedByGroup((current) => ({ ...current, [group]: event.target.value }))}>{available.length === 0 ? <option value={defaults[group]}>未出力</option> : available.map((series) => <option key={series.tag} value={series.tag}>{series.displayName}</option>)}</select></label>
        <label>Window<select aria-label="表示ウィンドウ" value={windowSize} onChange={(event) => setWindowSize(event.target.value === 'all' ? 'all' : Number(event.target.value) as TrainingMetricWindow)}><option value="all">全期間</option><option value={256}>256</option><option value={512}>512</option><option value={1024}>1024</option></select></label>
      </div>
      {metrics.error ? <div className="training-diagnostics-error"><AlertTriangle size={15} />{metrics.error}</div> : null}
      {!metrics.loading && !metrics.status?.available ? <div className="training-diagnostics-empty"><Database size={18} />TensorBoardが無効、またはevent fileがまだ生成されていません。</div> : null}
      <div className="training-diagnostics-grid">
        <div className="training-diagnostics-charts">
          <TrainingMetricChart series={primary} selectedStep={selectedStep} onSelectedStepChange={setSelectedStep} windowSize={windowSize} />
          <TrainingMetricChart series={secondary} selectedStep={selectedStep} onSelectedStepChange={setSelectedStep} windowSize={windowSize} />
        </div>
        <aside className="training-diagnostics-rail">
          <h2><Activity size={16} />Status</h2>
          <dl><div><dt>latest step</dt><dd>{metrics.status?.lastStep.toLocaleString('ja-JP') ?? '—'}</dd></div><div><dt>seed</dt><dd>{metrics.status?.selectedSeed ?? seed ?? '—'}</dd></div><div><dt>source</dt><dd>{metrics.status?.source ?? '—'}</dd></div><div><dt>generation</dt><dd>{metrics.status?.generation?.slice(0, 10) ?? '—'}</dd></div></dl>
          <h3>診断アラート</h3>
          {warnings.length === 0 ? <p>現在の最新値に明確な警告はありません。</p> : warnings.map((warning) => <p key={warning} className="training-diagnostics-alert"><AlertTriangle size={13} />{warning}</p>)}
        </aside>
      </div>
    </section>
  )
}
