import { Activity, AlertTriangle, Database, Pause, Play, Radio, RotateCcw, SkipBack, SkipForward } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type {
  CheckpointEvaluationItem,
  CheckpointEvaluationsResponse,
  JobSummary,
  TrainingTelemetryRecord,
} from '../data/types'
import { MarketReplayChart } from '../live/MarketReplayChart'
import { currentEnvironmentEpisode, telemetryEnvironmentIds } from '../live/telemetryStreams'
import { useTrainingTelemetry } from '../live/useTrainingTelemetry'
import '../liveTraining.css'

interface LiveTrainingPageProps { api?: StudioApi }
type ReplayMode = 'live' | 'buffered'
type TimelineMode = 'candles' | 'events'
type Speed = 1 | 4 | 8

function signed(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toLocaleString('ja-JP', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`
}

function shortDigest(value: string | null): string {
  return value ? `${value.slice(0, 8)}…` : '—'
}

function checkpointIdentity(item: CheckpointEvaluationItem): string {
  return `${item.fold}|${item.configuration}|${item.evaluationDigest}`
}

function checkpointLabel(item: CheckpointEvaluationItem): string {
  return `${item.fold} · ${item.configuration}${item.finalist ? ' · finalist' : ''}`
}

function eventLabel(record: TrainingTelemetryRecord): string {
  if (record.eventType === 'risk') return 'RISK'
  if (record.eventType === 'episode_end') return 'END'
  const before = record.weightsBefore[0] ?? 0
  const after = record.weightsAfter[0] ?? 0
  if (after > before + 1e-9) return 'BUY'
  if (after < before - 1e-9) return 'SELL'
  return 'HOLD'
}

function Sparkline({ values, label }: { values: (number | null)[]; label: string }) {
  const finite = values.filter((value): value is number => value !== null && Number.isFinite(value))
  if (finite.length < 2) return <div className="live-sparkline live-sparkline--empty" aria-label={`${label} データ待機中`} />
  const minimum = Math.min(...finite)
  const maximum = Math.max(...finite)
  const spread = Math.max(maximum - minimum, 1e-9)
  const points = values.map((value, index) => {
    const resolved = value ?? minimum
    const x = values.length === 1 ? 50 : index * 100 / (values.length - 1)
    const y = 30 - (resolved - minimum) / spread * 26
    return `${x},${y}`
  }).join(' ')
  return (
    <svg className="live-sparkline" viewBox="0 0 100 34" preserveAspectRatio="none" role="img" aria-label={label}>
      <polyline points={points} />
    </svg>
  )
}

function MetricCard({ label, value, values, tone = 'positive' }: {
  label: string
  value: string
  values: (number | null)[]
  tone?: 'positive' | 'negative' | 'neutral'
}) {
  return (
    <article className="live-metric-card">
      <span>{label}</span>
      <strong className={`live-tone live-tone--${tone}`}>{value}</strong>
      <Sparkline values={values} label={`${label} 推移`} />
    </article>
  )
}

export function LiveTrainingPage({ api = studioApi }: LiveTrainingPageProps) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [jobId, setJobId] = useState<string | null>(null)
  const [seed, setSeed] = useState<number | null>(null)
  const [environmentId, setEnvironmentId] = useState<number | null>(null)
  const [checkpointEvidenceId, setCheckpointEvidenceId] = useState<string | null>(null)
  const [jobsError, setJobsError] = useState<string | null>(null)
  const [checkpointEvaluations, setCheckpointEvaluations] = useState<CheckpointEvaluationsResponse | null>(null)
  const [checkpointError, setCheckpointError] = useState<string | null>(null)
  const [replayMode, setReplayMode] = useState<ReplayMode>('buffered')
  const [timelineMode, setTimelineMode] = useState<TimelineMode>('candles')
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState<Speed>(4)
  const [cursor, setCursor] = useState(0)
  const telemetry = useTrainingTelemetry(jobId, api, seed)

  useEffect(() => {
    let active = true
    void api.loadJobs().then((response) => {
      if (!active) return
      setJobs(response.items)
      const preferred = response.items.find((job) => job.status === 'running') ?? response.items[0] ?? null
      setJobId(preferred?.id ?? null)
      setJobsError(null)
    }).catch((reason: unknown) => {
      if (!active) return
      setJobsError(reason instanceof Error ? reason.message : 'ジョブを取得できませんでした。')
    })
    return () => { active = false }
  }, [api])

  useEffect(() => {
    setSeed(null)
    setEnvironmentId(null)
    setCheckpointEvidenceId(null)
    setCheckpointEvaluations(null)
    setCheckpointError(null)
    if (!jobId || !api.loadCheckpointEvaluations) return undefined
    let active = true
    void api.loadCheckpointEvaluations(jobId).then((response) => {
      if (!active) return
      setCheckpointEvaluations(response)
    }).catch((reason: unknown) => {
      if (!active) return
      setCheckpointError(reason instanceof Error ? reason.message : 'Checkpoint評価を取得できませんでした。')
    })
    return () => { active = false }
  }, [api, jobId])

  const seedKey = telemetry.status?.availableSeeds.join(',') ?? ''
  useEffect(() => {
    const available = telemetry.status?.availableSeeds ?? []
    if (available.length === 0) return
    if (seed === null || !available.includes(seed)) {
      setSeed(telemetry.status?.selectedSeed ?? available[0])
    }
  }, [seed, seedKey, telemetry.status?.selectedSeed])

  const availableEnvironmentIds = useMemo(
    () => telemetryEnvironmentIds(telemetry.records),
    [telemetry.records],
  )
  const environmentKey = availableEnvironmentIds.join(',')
  const latestEnvironmentId = telemetry.records.at(-1)?.environmentId ?? null
  useEffect(() => {
    if (availableEnvironmentIds.length === 0) {
      setEnvironmentId(null)
      return
    }
    if (environmentId === null || !availableEnvironmentIds.includes(environmentId)) {
      setEnvironmentId(latestEnvironmentId ?? availableEnvironmentIds[0])
    }
  }, [availableEnvironmentIds, environmentId, environmentKey, latestEnvironmentId])

  const effectiveEnvironmentId = environmentId ?? latestEnvironmentId
  const replayRecords = useMemo(
    () => currentEnvironmentEpisode(telemetry.records, effectiveEnvironmentId),
    [effectiveEnvironmentId, telemetry.records],
  )

  useEffect(() => {
    if (replayRecords.length === 0) {
      setCursor(0)
      return
    }
    setCursor((current) => replayMode === 'live' || current === 0
      ? replayRecords.length - 1
      : Math.min(current, replayRecords.length - 1))
  }, [effectiveEnvironmentId, replayMode, replayRecords.length])

  useEffect(() => {
    if (!playing || replayMode === 'live' || replayRecords.length < 2) return undefined
    const timer = window.setInterval(() => {
      setCursor((current) => current >= replayRecords.length - 1 ? 0 : current + 1)
    }, Math.max(90, 700 / speed))
    return () => window.clearInterval(timer)
  }, [playing, replayMode, replayRecords.length, speed])

  const selectedJob = jobs.find((job) => job.id === jobId) ?? null
  const activeRecord = replayRecords[Math.min(cursor, Math.max(0, replayRecords.length - 1))] ?? null
  const latestRecord = replayRecords.at(-1) ?? null
  const effectiveSeed = seed ?? telemetry.status?.selectedSeed ?? null
  const checkpointOptions = useMemo(() => {
    if (effectiveSeed === null) return []
    const candidates = checkpointEvaluations?.items.filter((item) => item.seed === effectiveSeed) ?? []
    const finalists = candidates.filter((item) => item.finalist)
    const options = finalists.length > 0 ? finalists : candidates
    return [...options].sort((left, right) =>
      left.fold.localeCompare(right.fold)
      || left.configuration.localeCompare(right.configuration)
      || left.evaluationDigest.localeCompare(right.evaluationDigest))
  }, [checkpointEvaluations, effectiveSeed])
  const checkpointOptionsKey = checkpointOptions.map(checkpointIdentity).join(',')

  useEffect(() => {
    if (checkpointOptions.length === 0) {
      setCheckpointEvidenceId(null)
      return
    }
    if (!checkpointOptions.some((item) => checkpointIdentity(item) === checkpointEvidenceId)) {
      setCheckpointEvidenceId(checkpointIdentity(checkpointOptions[0]))
    }
  }, [checkpointEvidenceId, checkpointOptions, checkpointOptionsKey])

  const selectedCheckpoint = checkpointOptions.find(
    (item) => checkpointIdentity(item) === checkpointEvidenceId,
  ) ?? checkpointOptions[0] ?? null
  const firstPortfolio = replayRecords.find((record) => record.portfolioValue !== null)?.portfolioValue ?? null
  const equity = activeRecord?.portfolioValue ?? null
  const baseline = activeRecord?.baselinePortfolioValue ?? null
  const pnl = equity !== null && firstPortfolio !== null ? equity - firstPortfolio : null
  const baselineDelta = equity !== null && baseline !== null ? equity - baseline : null
  const checkpointReturn = selectedCheckpoint?.totalReturn ?? null
  const currentWeight = activeRecord?.weightsAfter[0] ?? 0
  const positionDirection = Math.abs(currentWeight) < 1e-9 ? 'フラット' : currentWeight > 0 ? 'ロング' : 'ショート'
  const positionTone = currentWeight > 0 ? 'live-positive' : currentWeight < 0 ? 'live-negative' : ''
  const compressed = timelineMode === 'events'
  const recentEvents = useMemo(
    () => replayRecords.filter((record) => record.eventType !== 'rollout').slice(-8).reverse(),
    [replayRecords],
  )
  const equityValues = replayRecords.map((record) => record.portfolioValue)
  const baselineValues = replayRecords.map((record) => record.baselinePortfolioValue)
  const drawdownValues = replayRecords.map((record) => record.drawdown === null ? null : -record.drawdown * 100)
  const connectionLabel = telemetry.connection === 'live' ? 'LIVE' : telemetry.connection === 'delayed' ? 'DELAYED' : telemetry.connection === 'connecting' ? 'CONNECTING' : 'OFFLINE'

  const jump = (amount: number) => {
    setPlaying(false)
    setCursor((current) => Math.max(0, Math.min(replayRecords.length - 1, current + amount)))
  }

  return (
    <section className="live-page" aria-labelledby="live-training-title">
      <header className="live-header">
        <div className="live-title-block">
          <div className="live-title-row">
            <span className="live-nogo">NO-GO</span>
            <span className={`live-connection live-connection--${telemetry.connection}`}><Radio size={12} aria-hidden="true" />{connectionLabel}</span>
          </div>
          <h1 id="live-training-title">Live Training</h1>
          <p>探索ロールアウトと決定論的Checkpoint評価を分離し、同じseed・environment・episode単位で確認します。</p>
        </div>
        <div className="live-header-controls">
          <label className="live-job-select">Run
            <select value={jobId ?? ''} onChange={(event) => setJobId(event.target.value || null)} aria-label="Live Training job">
              {jobs.length === 0 ? <option value="">実行中ジョブなし</option> : null}
              {jobs.map((job) => <option key={job.id} value={job.id}>{job.runId} · {job.status}</option>)}
            </select>
          </label>
          <label className="live-job-select">Seed
            <select value={effectiveSeed ?? ''} onChange={(event) => { setEnvironmentId(null); setSeed(event.target.value === '' ? null : Number(event.target.value)) }} aria-label="Live Training seed">
              {(telemetry.status?.availableSeeds.length ?? 0) === 0 ? <option value="">seed待機中</option> : null}
              {telemetry.status?.availableSeeds.map((value) => <option key={value} value={value}>Seed {value}</option>)}
            </select>
          </label>
          <label className="live-job-select">Environment
            <select value={effectiveEnvironmentId ?? ''} onChange={(event) => setEnvironmentId(event.target.value === '' ? null : Number(event.target.value))} aria-label="Live Training environment">
              {availableEnvironmentIds.length === 0 ? <option value="">env待機中</option> : null}
              {availableEnvironmentIds.map((value) => <option key={value} value={value}>Env {value}</option>)}
            </select>
          </label>
          <div className="live-segment-group" aria-label="リプレイモード">
            <span>リプレイモード</span>
            <div className="live-segment">
              <button type="button" aria-pressed={replayMode === 'live'} onClick={() => setReplayMode('live')}>ほぼライブ</button>
              <button type="button" aria-pressed={replayMode === 'buffered'} onClick={() => setReplayMode('buffered')}>バッファ再生</button>
            </div>
          </div>
          <div className="live-buffer"><Database size={14} aria-hidden="true" /><strong>{replayRecords.length}/{telemetry.records.length}</strong> stream/total steps</div>
          <div className="live-segment-group" aria-label="タイム軸">
            <span>タイム軸（切替可能）</span>
            <div className="live-segment">
              <button type="button" aria-pressed={timelineMode === 'candles'} onClick={() => setTimelineMode('candles')}>ローソク足ごと</button>
              <button type="button" aria-pressed={timelineMode === 'events'} onClick={() => setTimelineMode('events')}>イベント圧縮</button>
            </div>
          </div>
        </div>
      </header>

      {(jobsError || telemetry.error || checkpointError) ? <div className="live-alert"><AlertTriangle size={16} aria-hidden="true" />{jobsError ?? telemetry.error ?? checkpointError}</div> : null}

      <div className="live-primary-grid">
        <article className="live-market-panel">
          <div className="live-panel-title">
            <div><strong>{activeRecord?.symbol ?? latestRecord?.symbol ?? 'Market'} 市場リプレイ</strong><span>{selectedJob?.runId ?? 'ジョブ待機中'} · Seed {effectiveSeed ?? '—'} · Env {effectiveEnvironmentId ?? '—'} · current episode</span></div>
            <span className="live-step-chip">Replay Step {activeRecord?.globalStep.toLocaleString('ja-JP') ?? '—'}</span>
          </div>
          <MarketReplayChart records={replayRecords} cursorSequence={activeRecord?.sequence ?? null} compressed={compressed} />
          <div className="live-transport" aria-label="リプレイ操作">
            <button type="button" className="live-icon-button live-icon-button--primary" aria-label={playing ? '一時停止' : '再生'} onClick={() => setPlaying((current) => !current)}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
            <button type="button" className="live-icon-button" aria-label="先頭へ戻る" onClick={() => { setPlaying(false); setCursor(0) }}><RotateCcw size={16} /></button>
            <div className="live-speed" aria-label="再生速度">{([1, 4, 8] as Speed[]).map((value) => <button type="button" key={value} aria-pressed={speed === value} onClick={() => setSpeed(value)}>{value}x</button>)}</div>
            <span className="live-sampling">一定ステップごとに表示 <strong>Adaptive / 32 steps</strong></span>
            <div className="live-jump"><button type="button" onClick={() => jump(-10)}><SkipBack size={14} />−10</button><button type="button" onClick={() => jump(10)}>+10<SkipForward size={14} /></button></div>
            <button type="button" className="live-latest" onClick={() => { setCursor(Math.max(0, replayRecords.length - 1)); setReplayMode('live') }}>最新へ</button>
          </div>
        </article>

        <aside className="live-agent-panel" aria-label="エージェント状態">
          <div className="live-panel-title"><div><strong>エージェント状態（現在）</strong><span>再生カーソル同期</span></div><Activity size={17} aria-hidden="true" /></div>
          <label className="live-checkpoint-select">Checkpoint evidence
            <select
              aria-label="Checkpoint evaluation evidence"
              value={checkpointEvidenceId ?? ''}
              onChange={(event) => setCheckpointEvidenceId(event.target.value || null)}
            >
              {checkpointOptions.length === 0 ? <option value="">未生成</option> : null}
              {checkpointOptions.map((item) => (
                <option key={checkpointIdentity(item)} value={checkpointIdentity(item)}>{checkpointLabel(item)}</option>
              ))}
            </select>
          </label>
          <dl>
            <div><dt>現在ポジション</dt><dd className={positionTone}>{positionDirection} {(Math.abs(currentWeight) * 100).toFixed(1)}%</dd></div>
            <div><dt>現在価格</dt><dd>{activeRecord?.close?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? '—'} USDT</dd></div>
            <div><dt>再生区間損益</dt><dd className={(pnl ?? 0) >= 0 ? 'live-positive' : 'live-negative'}>{signed(pnl)} USDT</dd></div>
            <div><dt>ベースライン超過</dt><dd className={(baselineDelta ?? 0) >= 0 ? 'live-positive' : 'live-negative'}>{signed(baselineDelta)} USDT</dd></div>
            <div><dt>報酬</dt><dd>{signed(activeRecord?.reward ?? null, 3)}</dd></div>
            <div><dt>ドローダウン</dt><dd className="live-negative">{activeRecord?.drawdown === null || activeRecord?.drawdown === undefined ? '—' : `-${(activeRecord.drawdown * 100).toFixed(2)}%`}</dd></div>
            <div><dt>Checkpoint評価</dt><dd className={(checkpointReturn ?? 0) >= 0 ? 'live-positive' : 'live-negative'}>{checkpointReturn === null ? '未生成' : `${signed(checkpointReturn * 100)}%${selectedCheckpoint?.finalist ? ' finalist' : ''}`}</dd></div>
            <div><dt>評価range / digest</dt><dd>{selectedCheckpoint ? `${selectedCheckpoint.fold} [${selectedCheckpoint.checkpointRange[0]}, ${selectedCheckpoint.checkpointRange[1]}) · ${shortDigest(selectedCheckpoint.evaluationDigest)}` : '—'}</dd></div>
            <div><dt>環境 / Seed</dt><dd>env {activeRecord?.environmentId ?? '—'} / {effectiveSeed ?? '—'}</dd></div>
            <div><dt>最新受信Step</dt><dd>{latestRecord?.globalStep.toLocaleString('ja-JP') ?? '—'}</dd></div>
          </dl>
          <div className="live-exposure">
            <span>Target weight</span>
            <div><i style={{ width: `${Math.min(100, Math.abs(currentWeight) * 100)}%` }} /></div>
            <strong>{currentWeight.toFixed(3)}</strong>
          </div>
          <div className="live-research-note"><AlertTriangle size={15} aria-hidden="true" /><span>探索とCheckpoint評価は異なる過程です。fold・評価range・digestを明示確認し、実運用・発注には使用しません。</span></div>
        </aside>
      </div>

      <div className="live-metric-grid">
        <MetricCard label="探索リプレイ区間損益" value={`${signed(pnl)} USDT`} values={equityValues} tone={(pnl ?? 0) >= 0 ? 'positive' : 'negative'} />
        <MetricCard label={`決定論Checkpoint · Seed ${effectiveSeed ?? '—'} · ${selectedCheckpoint?.fold ?? '未生成'}`} value={checkpointReturn === null ? '未生成' : `${signed(checkpointReturn * 100)}%`} values={[checkpointReturn === null ? null : checkpointReturn * 100]} tone={checkpointReturn === null ? 'neutral' : checkpointReturn >= 0 ? 'positive' : 'negative'} />
        <MetricCard label="探索中ベースライン比較" value={`${signed(baselineDelta)} USDT`} values={baselineValues} tone={(baselineDelta ?? 0) >= 0 ? 'positive' : 'negative'} />
        <MetricCard label="ドローダウン（探索）" value={activeRecord?.drawdown === null || activeRecord?.drawdown === undefined ? '—' : `-${(activeRecord.drawdown * 100).toFixed(2)}%`} values={drawdownValues} tone="negative" />
      </div>

      <article className="live-events-panel">
        <div className="live-panel-title"><div><strong>イベント（最新）</strong><span>売買・リスク・Episode終了 · Seed {effectiveSeed ?? '—'} · Env {effectiveEnvironmentId ?? '—'}</span></div><span>{telemetry.status?.malformedLines ? `破損行 ${telemetry.status.malformedLines}` : 'sequence verified'}</span></div>
        <div className="live-event-list">
          {recentEvents.length === 0 ? <div className="live-empty-event">重要イベントを待っています。</div> : recentEvents.map((record) => {
            const label = eventLabel(record)
            return (
              <button type="button" key={`${record.sequence}-${record.environmentId}`} aria-label={`Step ${record.globalStep} ${label}`} onClick={() => { setPlaying(false); setCursor(Math.max(0, replayRecords.findIndex((item) => item.sequence === record.sequence))) }}>
                <time>{record.marketTime?.slice(11, 19) ?? record.recordedAt.slice(11, 19)}</time>
                <span className={`live-event-tag live-event-tag--${label.toLowerCase()}`}>{label}</span>
                <strong>{record.symbol}</strong>
                <span>weight {(record.weightsBefore[0] ?? 0).toFixed(3)} → {(record.weightsAfter[0] ?? 0).toFixed(3)}</span>
                <span>{record.close?.toLocaleString('ja-JP', { maximumFractionDigits: 2 }) ?? 'price —'}</span>
                <span>{record.riskReasons.join(', ') || `reward ${signed(record.reward, 3)}`}</span>
                <small>Step {record.globalStep.toLocaleString('ja-JP')}</small>
              </button>
            )
          })}
        </div>
      </article>
    </section>
  )
}
