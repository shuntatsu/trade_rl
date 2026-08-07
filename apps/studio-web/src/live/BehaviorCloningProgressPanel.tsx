import type { BehaviorCloningProgressResponse } from '../data/types'

function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`
}

function duration(seconds: number | null): string {
  if (seconds === null) return '—'
  const minutes = Math.max(0, Math.round(seconds / 60))
  return minutes < 60 ? `${minutes}分` : `${Math.floor(minutes / 60)}時間${minutes % 60}分`
}

const phaseLabel: Record<BehaviorCloningProgressResponse['phase'], string> = {
  not_started: '待機中', preparing: 'BC準備・学習中', training: 'BC学習中', evaluating: '因果ゲート評価中', passed: 'BC通過', failed: 'BC失敗',
}

export function BehaviorCloningProgressPanel({ progress }: { progress: BehaviorCloningProgressResponse | null }) {
  const visible = progress?.available === true
  const value = progress?.percent ?? 0
  return (
    <section className="bc-progress-panel" aria-label="Behavior Cloning progress">
      <div className="bc-progress-heading">
        <div><span>PRETRAINING</span><h2>Behavior Cloning</h2></div>
        <strong data-phase={progress?.phase ?? 'not_started'}>{phaseLabel[progress?.phase ?? 'not_started']}</strong>
      </div>
      <p>{visible ? '教師トレードから方策を事前学習しています。市場リプレイはPPO開始後に自動表示されます。' : 'Teacher生成またはBC開始を待っています。'}</p>
      <div className="bc-progress-track" role="progressbar" aria-label="BC epoch progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value)}>
        <span style={{ width: `${value}%` }} />
      </div>
      <div className="bc-progress-stats">
        <div><span>Epoch</span><strong>{progress?.epoch ?? '—'} / {progress?.totalEpochs ?? '—'}</strong></div>
        <div><span>進捗</span><strong>{progress?.percent === null || progress?.percent === undefined ? '—' : `${progress.percent.toFixed(1)}%`}</strong></div>
        <div><span>Validation loss</span><strong>{progress?.validationLoss?.toFixed(5) ?? '—'}</strong></div>
        <div><span>Precision</span><strong>{percent(progress?.gatePrecision ?? null)}</strong></div>
        <div><span>Recall</span><strong>{percent(progress?.gateRecall ?? null)}</strong></div>
        <div><span>活動比</span><strong>{progress?.activityRatio?.toFixed(2) ?? '—'}</strong></div>
        <div><span>経過</span><strong>{duration(progress?.elapsedSeconds ?? null)}</strong></div>
        <div><span>残り予測</span><strong>{duration(progress?.estimatedRemainingSeconds ?? null)}</strong></div>
      </div>
      <footer>{[progress?.fold, progress?.configuration, progress?.seed === null || progress?.seed === undefined ? null : `seed-${progress.seed}`].filter(Boolean).join(' · ') || '現在の学習メンバーを探索中'}</footer>
    </section>
  )
}
