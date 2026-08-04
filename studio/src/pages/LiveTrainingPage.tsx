import { AlertTriangle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type {
  CheckpointEvaluationItem,
  CheckpointEvaluationsResponse,
  JobSummary,
  TrainingTelemetryRecord,
} from '../data/types'
import { ReplayToolbar, type ReplaySpeed, type ReplaySourceSelection } from '../live/ReplayToolbar'
import { ResearchChartInspector } from '../live/ResearchChartInspector'
import {
  ResearchChartWorkspace,
  type ResearchRangePreset,
} from '../live/ResearchChartWorkspace'
import {
  DEFAULT_RESEARCH_CHART_LAYERS,
  nextEventIndex,
  previousEventIndex,
  type ResearchTimeframe,
} from '../live/researchChartModel'
import { TrainingDiagnosticsPanel } from '../live/TrainingDiagnosticsPanel'
import { BehaviorCloningProgressPanel } from '../live/BehaviorCloningProgressPanel'
import { currentEnvironmentEpisode, telemetryEnvironmentIds } from '../live/telemetryStreams'
import { useBehaviorCloningProgress } from '../live/useBehaviorCloningProgress'
import { useTrainingTelemetry } from '../live/useTrainingTelemetry'
import '../liveTraining.css'

interface LiveTrainingPageProps { api?: StudioApi }
type LiveView = 'replay' | 'diagnostics'

function signed(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toLocaleString('ja-JP', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`
}

function checkpointIdentity(item: CheckpointEvaluationItem): string {
  return `${item.fold}|${item.configuration}|${item.evaluationDigest}`
}

function checkpointLabel(item: CheckpointEvaluationItem): string {
  return `${item.fold} · ${item.configuration}${item.finalist ? ' · finalist' : ''}`
}

function MetricCard({ label, value, tone }: {
  label: string
  value: string
  tone: 'positive' | 'negative' | 'neutral'
}) {
  return (
    <article className="research-summary-card">
      <span>{label}</span>
      <strong className={`live-tone live-tone--${tone}`}>{value}</strong>
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
  const [liveView, setLiveView] = useState<LiveView>('replay')
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState<ReplaySpeed>(4)
  const [cursor, setCursor] = useState(0)
  const [followLatest, setFollowLatest] = useState(true)
  const [layers, setLayers] = useState(DEFAULT_RESEARCH_CHART_LAYERS)
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [timeframe, setTimeframe] = useState<ResearchTimeframe>('15m')
  const [rangePreset, setRangePreset] = useState<ResearchRangePreset>('24h')
  const [previewRecord, setPreviewRecord] = useState<TrainingTelemetryRecord | null>(null)
  const [chartResetToken, setChartResetToken] = useState(0)
  const telemetry = useTrainingTelemetry(jobId, api, seed)
  const behaviorCloning = useBehaviorCloningProgress(jobId, api)

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
    setCursor((current) => followLatest
      ? replayRecords.length - 1
      : Math.min(current, replayRecords.length - 1))
  }, [effectiveEnvironmentId, followLatest, replayRecords.length])

  useEffect(() => {
    if (!playing || replayRecords.length < 2) return undefined
    const timer = window.setInterval(() => {
      setCursor((current) => Math.min(replayRecords.length - 1, current + 1))
    }, Math.max(90, 700 / speed))
    return () => window.clearInterval(timer)
  }, [playing, replayRecords.length, speed])

  const selectedJob = jobs.find((job) => job.id === jobId) ?? null
  const activeRecord = replayRecords[Math.min(cursor, Math.max(0, replayRecords.length - 1))] ?? null
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
  const equity = activeRecord?.portfolioValue ?? null
  const baseline = activeRecord?.baselinePortfolioValue ?? null
  const baselineDelta = equity !== null && baseline !== null ? equity - baseline : null
  const drawdown = activeRecord?.drawdown ?? null

  const commitCursor = (next: number) => {
    setPlaying(false)
    setFollowLatest(next === replayRecords.length - 1)
    setCursor(Math.max(0, Math.min(replayRecords.length - 1, next)))
  }

  const commitRecord = (record: TrainingTelemetryRecord) => {
    const index = replayRecords.findIndex((item) => item.sequence === record.sequence)
    if (index >= 0) commitCursor(index)
  }

  const changeJob = (nextJobId: string | null) => {
    setPlaying(false)
    setFollowLatest(true)
    setJobId(nextJobId)
    setCursor(0)
    setPreviewRecord(null)
    setChartResetToken((current) => current + 1)
  }

  const changeSource = ({ seed: nextSeed, environmentId: nextEnvironmentId }: ReplaySourceSelection) => {
    setPlaying(false)
    setFollowLatest(true)
    setSeed(nextSeed)
    setEnvironmentId(nextEnvironmentId)
    setCursor(0)
    setPreviewRecord(null)
    setChartResetToken((current) => current + 1)
  }

  const resetView = () => {
    setFollowLatest(true)
    setRangePreset('24h')
    setLayers(DEFAULT_RESEARCH_CHART_LAYERS)
    setPreviewRecord(null)
    setCursor(Math.max(0, replayRecords.length - 1))
    setChartResetToken((current) => current + 1)
  }

  return (
    <section className="live-page research-live-page" aria-labelledby="live-training-title">
      <header className="research-live-header">
        <div className="live-title-block">
          <div className="live-title-row"><span className="live-nogo">NO-GO</span></div>
          <h1 id="live-training-title">Live Training</h1>
          <p>市場・方策・報酬・リスクを同じ時刻で直接操作し、探索とCheckpoint評価を分離して確認します。</p>
        </div>
        <div className="live-view-selector" aria-label="Live Training view">
          <button type="button" aria-pressed={liveView === 'replay'} onClick={() => setLiveView('replay')}>市場リプレイ</button>
          <button type="button" aria-pressed={liveView === 'diagnostics'} onClick={() => setLiveView('diagnostics')}>学習診断</button>
        </div>
      </header>

      {(jobsError || telemetry.error || checkpointError || behaviorCloning.error) ? (
        <div className="live-alert"><AlertTriangle size={16} aria-hidden="true" />{jobsError ?? telemetry.error ?? checkpointError ?? behaviorCloning.error}</div>
      ) : null}

      <div hidden={liveView !== 'diagnostics'}><TrainingDiagnosticsPanel job={selectedJob} seed={effectiveSeed} api={api} /></div>

      <div className="research-replay-view" hidden={liveView !== 'replay'}>
        <ReplayToolbar
          jobs={jobs}
          jobId={jobId}
          seeds={telemetry.status?.availableSeeds ?? []}
          seed={effectiveSeed}
          environments={availableEnvironmentIds}
          environmentId={effectiveEnvironmentId}
          playing={playing}
          speed={speed}
          followLatest={followLatest}
          layers={layers}
          hasRecords={replayRecords.length > 0}
          onJobChange={changeJob}
          onSourceChange={changeSource}
          onTogglePlaying={() => setPlaying((current) => !current)}
          onFirst={() => commitCursor(0)}
          onPreviousEvent={() => commitCursor(previousEventIndex(replayRecords, cursor))}
          onNextEvent={() => commitCursor(nextEventIndex(replayRecords, cursor))}
          onLast={() => commitCursor(Math.max(0, replayRecords.length - 1))}
          onSpeedChange={setSpeed}
          onFollowLatestChange={(next) => {
            setFollowLatest(next)
            if (next) setCursor(Math.max(0, replayRecords.length - 1))
          }}
          onLayersChange={setLayers}
          onResetView={resetView}
        />

        {replayRecords.length === 0 ? (
          <BehaviorCloningProgressPanel progress={behaviorCloning.progress} />
        ) : (<>
        <div className="research-summary-grid" aria-label="研究サマリー">
          <MetricCard label="RL資産" value={equity === null ? '—' : equity.toLocaleString('ja-JP', { maximumFractionDigits: 2 })} tone="neutral" />
          <MetricCard label="Baseline差" value={`${signed(baselineDelta)} USDT`} tone={(baselineDelta ?? 0) >= 0 ? 'positive' : 'negative'} />
          <MetricCard label="Drawdown" value={drawdown === null ? '—' : `-${(drawdown * 100).toFixed(2)}%`} tone={drawdown === null ? 'neutral' : 'negative'} />
        </div>

        <div className="research-workspace-grid">
          <div className="research-chart-stack">
            <ResearchChartWorkspace
              records={replayRecords}
              symbol={symbol}
              timeframe={timeframe}
              rangePreset={rangePreset}
              layers={layers}
              followLatest={followLatest}
              committedSequence={activeRecord?.sequence ?? null}
              resetToken={chartResetToken}
              onSymbolChange={setSymbol}
              onTimeframeChange={setTimeframe}
              onRangePresetChange={setRangePreset}
              onPreviewRecord={setPreviewRecord}
              onCommitRecord={commitRecord}
              onManualNavigation={() => setFollowLatest(false)}
            />
            <div className="research-replay-scrubber">
              <span>Step {activeRecord?.globalStep.toLocaleString('ja-JP') ?? '—'}</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, replayRecords.length - 1)}
                value={Math.min(cursor, Math.max(0, replayRecords.length - 1))}
                disabled={replayRecords.length === 0}
                aria-label="再生位置"
                onChange={(event) => commitCursor(Number(event.target.value))}
              />
              <time>{activeRecord?.marketTime?.replace('T', ' ').slice(0, 19) ?? '—'}</time>
            </div>
          </div>

          <ResearchChartInspector
            committed={activeRecord}
            preview={previewRecord}
            checkpoint={selectedCheckpoint}
            checkpointIdentity={checkpointEvidenceId}
            checkpointOptions={checkpointOptions}
            onCheckpointChange={setCheckpointEvidenceId}
            identityFor={checkpointIdentity}
            labelFor={checkpointLabel}
          />
        </div>
        </>)}
      </div>
    </section>
  )
}
