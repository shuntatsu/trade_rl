import type { CheckpointEvaluationItem, TrainingTelemetryRecord } from '../data/types'

function signed(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toLocaleString('ja-JP', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`
}

function plain(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toLocaleString('ja-JP', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export interface ResearchChartInspectorProps {
  committed: TrainingTelemetryRecord | null
  preview: TrainingTelemetryRecord | null
  checkpoint: CheckpointEvaluationItem | null
  checkpointIdentity: string | null
  checkpointOptions: CheckpointEvaluationItem[]
  onCheckpointChange: (identity: string | null) => void
  identityFor: (item: CheckpointEvaluationItem) => string
  labelFor: (item: CheckpointEvaluationItem) => string
}

export function ResearchChartInspector({
  committed,
  preview,
  checkpoint,
  checkpointIdentity,
  checkpointOptions,
  onCheckpointChange,
  identityFor,
  labelFor,
}: ResearchChartInspectorProps) {
  const record = preview ?? committed
  const weight = record?.weightsAfter[0] ?? null
  const executed = record?.executedTarget[0] ?? null
  const action = record?.action[0] ?? null
  const position = weight === null || Math.abs(weight) < 1e-9
    ? 'フラット'
    : weight > 0 ? 'ロング' : 'ショート'

  return (
    <aside className="research-chart-inspector" aria-label="選択時点の研究データ">
      <div className="research-inspector-heading">
        <div>
          <span>{preview ? 'クロスヘア' : '選択時点'}</span>
          <strong>Step {record?.globalStep.toLocaleString('ja-JP') ?? '—'}</strong>
        </div>
        <time>{record?.marketTime?.replace('T', ' ').slice(0, 19) ?? '—'}</time>
      </div>

      <label className="research-inspector-checkpoint">
        <span>Checkpoint evidence</span>
        <select
          aria-label="Checkpoint evaluation evidence"
          value={checkpointIdentity ?? ''}
          onChange={(event) => onCheckpointChange(event.target.value || null)}
        >
          {checkpointOptions.length === 0 ? <option value="">未生成</option> : null}
          {checkpointOptions.map((item) => (
            <option key={identityFor(item)} value={identityFor(item)}>{labelFor(item)}</option>
          ))}
        </select>
      </label>

      <dl className="research-inspector-values">
        <div><dt>Position</dt><dd>{position}{weight === null ? '' : ` ${(Math.abs(weight) * 100).toFixed(1)}%`}</dd></div>
        <div><dt>Action</dt><dd>{plain(action)}</dd></div>
        <div><dt>Executed</dt><dd>{plain(executed)}</dd></div>
        <div><dt>Reward</dt><dd>{signed(record?.reward)}</dd></div>
        <div><dt>Cost</dt><dd>{plain(record?.intervalCost)}</dd></div>
        <div><dt>Drawdown</dt><dd>{record?.drawdown == null ? '—' : `-${(record.drawdown * 100).toFixed(2)}%`}</dd></div>
        <div><dt>OHLC</dt><dd>{record ? `${plain(record.open, 2)} / ${plain(record.high, 2)} / ${plain(record.low, 2)} / ${plain(record.close, 2)}` : '—'}</dd></div>
        <div><dt>RL equity</dt><dd>{plain(record?.portfolioValue, 2)}</dd></div>
        <div><dt>Baseline</dt><dd>{plain(record?.baselinePortfolioValue, 2)}</dd></div>
        <div><dt>Environment</dt><dd>{record ? `Seed ${record.seed} · Env ${record.environmentId}` : '—'}</dd></div>
        <div><dt>Checkpoint</dt><dd>{checkpoint ? `${signed(checkpoint.totalReturn * 100, 2)}% · ${checkpoint.fold}` : '未生成'}</dd></div>
      </dl>

      <p className="research-inspector-note">
        Hoverは確認のみ、クリックは再生位置を確定します。探索結果と決定論的Checkpoint評価を混同しません。
      </p>
    </aside>
  )
}
