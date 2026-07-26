from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"guarded replacement failed for {relative}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _append_once(relative: str, marker: str, content: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


def apply() -> None:
    _append_once(
        "studio/src/data/types.ts",
        "export interface TrainingMetricPoint",
        '''export interface TrainingMetricPoint {
  step: number
  wallTime: number
  value: number
}

export type TrainingMetricGroup = 'optimization' | 'policy' | 'value' | 'trading'
export type TrainingMetricUnit = 'raw' | 'rate' | 'percent' | 'currency'

export interface TrainingMetricSeries {
  tag: string
  displayName: string
  group: TrainingMetricGroup
  unit: TrainingMetricUnit
  points: TrainingMetricPoint[]
}

export interface TrainingMetricsStatusResponse {
  available: boolean
  selectedSeed: number | null
  availableSeeds: number[]
  availableTags: string[]
  lastStep: number
  source: string | null
  generation: string | null
}

export interface TrainingMetricsResponse {
  seed: number | null
  series: TrainingMetricSeries[]
  nextStep: number
  generation: string | null
  resetRequired: boolean
}
''',
    )

    _write(
        "studio/src/live/trainingMetricGuards.ts",
        '''import type {
  TrainingMetricGroup,
  TrainingMetricSeries,
  TrainingMetricUnit,
  TrainingMetricsResponse,
  TrainingMetricsStatusResponse,
} from '../data/types'

export const TRAINING_METRIC_TAGS = [
  'train/learning_rate',
  'train/loss',
  'train/policy_gradient_loss',
  'train/value_loss',
  'train/entropy_loss',
  'train/approx_kl',
  'train/clip_fraction',
  'train/explained_variance',
  'trade_rl/reward_mean',
  'trade_rl/portfolio_value_mean',
  'trade_rl/drawdown_mean',
  'trade_rl/interval_cost_mean',
  'trade_rl/action_abs_mean',
  'trade_rl/action_abs_max',
] as const

const tagSet = new Set<string>(TRAINING_METRIC_TAGS)
const groups = new Set<TrainingMetricGroup>(['optimization', 'policy', 'value', 'trading'])
const units = new Set<TrainingMetricUnit>(['raw', 'rate', 'percent', 'currency'])

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function nullableSeed(value: unknown): value is number | null {
  return value === null || (Number.isInteger(value) && finite(value) && value >= 0)
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

function isSeries(value: unknown): value is TrainingMetricSeries {
  if (!record(value) || !tagSet.has(String(value.tag))) return false
  if (typeof value.displayName !== 'string') return false
  if (!groups.has(value.group as TrainingMetricGroup) || !units.has(value.unit as TrainingMetricUnit)) return false
  if (!Array.isArray(value.points)) return false
  let previous = -1
  for (const point of value.points) {
    if (!record(point) || !Number.isInteger(point.step) || !finite(point.step) || point.step < 0) return false
    if (!finite(point.wallTime) || !finite(point.value) || point.step <= previous) return false
    previous = point.step
  }
  return true
}

export function isTrainingMetricsStatus(value: unknown): value is TrainingMetricsStatusResponse {
  if (!record(value) || typeof value.available !== 'boolean' || !nullableSeed(value.selectedSeed)) return false
  if (!Array.isArray(value.availableSeeds) || !value.availableSeeds.every((seed) => nullableSeed(seed) && seed !== null)) return false
  if (!stringArray(value.availableTags) || !value.availableTags.every((tag) => tagSet.has(tag))) return false
  if (new Set(value.availableTags).size !== value.availableTags.length) return false
  return Number.isInteger(value.lastStep) && finite(value.lastStep) && value.lastStep >= 0
    && (value.source === null || typeof value.source === 'string')
    && (value.generation === null || (typeof value.generation === 'string' && /^[0-9a-f]{64}$/.test(value.generation)))
}

export function isTrainingMetricsResponse(value: unknown): value is TrainingMetricsResponse {
  if (!record(value) || !nullableSeed(value.seed) || !Array.isArray(value.series)) return false
  if (!Number.isInteger(value.nextStep) || !finite(value.nextStep) || value.nextStep < 0) return false
  if (typeof value.resetRequired !== 'boolean') return false
  if (!(value.generation === null || (typeof value.generation === 'string' && /^[0-9a-f]{64}$/.test(value.generation)))) return false
  if (!value.series.every(isSeries)) return false
  return new Set(value.series.map((series) => series.tag)).size === value.series.length
}
''',
    )

    _replace_once(
        "studio/src/api/studioApi.ts",
        "  TrainingJobRequest,\n",
        "  TrainingJobRequest,\n  TrainingMetricsResponse,\n  TrainingMetricsStatusResponse,\n",
    )
    _replace_once(
        "studio/src/api/studioApi.ts",
        "} from '../live/telemetryGuards'\n",
        "} from '../live/telemetryGuards'\n"
        "import {\n"
        "  isTrainingMetricsResponse,\n"
        "  isTrainingMetricsStatus,\n"
        "} from '../live/trainingMetricGuards'\n",
    )
    _replace_once(
        "studio/src/api/studioApi.ts",
        "export function loadCheckpointEvaluations(\n",
        '''export function loadTrainingMetricsStatus(
  jobId: string,
  seed: number | null = null,
  fetcher: typeof fetch = fetch,
): Promise<TrainingMetricsStatusResponse> {
  const query = seed === null ? '' : `?seed=${encodeURIComponent(seed)}`
  return requestJson(
    `/api/studio/jobs/${encodeURIComponent(jobId)}/training-metrics/status${query}`,
    fetcher,
    isTrainingMetricsStatus,
  )
}

export function loadTrainingMetricScalars(
  jobId: string,
  tags: string[],
  afterStep = 0,
  limit = 512,
  seed: number | null = null,
  generation: string | null = null,
  fetcher: typeof fetch = fetch,
): Promise<TrainingMetricsResponse> {
  const parameters = new URLSearchParams({ after_step: String(afterStep), limit: String(limit) })
  for (const tag of tags) parameters.append('tag', tag)
  if (seed !== null) parameters.set('seed', String(seed))
  if (generation !== null) parameters.set('generation', generation)
  return requestJson(
    `/api/studio/jobs/${encodeURIComponent(jobId)}/training-metrics/scalars?${parameters.toString()}`,
    fetcher,
    isTrainingMetricsResponse,
  )
}

export function loadCheckpointEvaluations(
''',
    )
    _replace_once(
        "studio/src/api/studioApi.ts",
        "  loadCheckpointEvaluations?: (jobId: string) => Promise<CheckpointEvaluationsResponse>\n",
        "  loadCheckpointEvaluations?: (jobId: string) => Promise<CheckpointEvaluationsResponse>\n"
        "  loadTrainingMetricsStatus?: (jobId: string, seed?: number | null) => Promise<TrainingMetricsStatusResponse>\n"
        "  loadTrainingMetricScalars?: (jobId: string, tags: string[], afterStep?: number, limit?: number, seed?: number | null, generation?: string | null) => Promise<TrainingMetricsResponse>\n",
    )
    _replace_once(
        "studio/src/api/studioApi.ts",
        "  loadCheckpointEvaluations,\n",
        "  loadCheckpointEvaluations,\n  loadTrainingMetricsStatus,\n  loadTrainingMetricScalars,\n",
    )

    _write(
        "studio/src/live/useTrainingMetrics.ts",
        '''import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { TrainingMetricSeries, TrainingMetricsStatusResponse } from '../data/types'

const MAX_POINTS = 2_048
const POLL_INTERVAL_MS = 2_000

export interface TrainingMetricsState {
  status: TrainingMetricsStatusResponse | null
  series: TrainingMetricSeries[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

function mergeSeries(current: TrainingMetricSeries[], incoming: TrainingMetricSeries[]): TrainingMetricSeries[] {
  const byTag = new Map(current.map((series) => [series.tag, series]))
  for (const series of incoming) {
    const previous = byTag.get(series.tag)
    const byStep = new Map(previous?.points.map((point) => [point.step, point]) ?? [])
    for (const point of series.points) byStep.set(point.step, point)
    byTag.set(series.tag, {
      ...series,
      points: [...byStep.values()].sort((left, right) => left.step - right.step).slice(-MAX_POINTS),
    })
  }
  return [...byTag.values()]
}

export function useTrainingMetrics(
  jobId: string | null,
  seed: number | null,
  tags: string[],
  api: StudioApi = studioApi,
): TrainingMetricsState {
  const [status, setStatus] = useState<TrainingMetricsStatusResponse | null>(null)
  const [series, setSeries] = useState<TrainingMetricSeries[]>([])
  const [loading, setLoading] = useState(Boolean(jobId))
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef(0)
  const generation = useRef<string | null>(null)
  const request = useRef(0)
  const tagKey = useMemo(() => [...new Set(tags)].sort().join('\u0000'), [tags])
  const stableTags = useMemo(() => tagKey ? tagKey.split('\u0000') : [], [tagKey])

  const refresh = useCallback(async () => {
    if (!jobId || !api.loadTrainingMetricsStatus || !api.loadTrainingMetricScalars) {
      setLoading(false)
      setStatus(null)
      setSeries([])
      return
    }
    const requestId = ++request.current
    try {
      const nextStatus = await api.loadTrainingMetricsStatus(jobId, seed)
      if (requestId !== request.current) return
      setStatus(nextStatus)
      if (!nextStatus.available || stableTags.length === 0) {
        setError(null)
        return
      }
      const page = await api.loadTrainingMetricScalars(
        jobId,
        stableTags,
        cursor.current,
        512,
        seed,
        generation.current,
      )
      if (requestId !== request.current) return
      if (page.resetRequired) {
        cursor.current = 0
        generation.current = page.generation
        setSeries([])
        return
      }
      if (seed !== null && page.seed !== seed) throw new Error('学習指標のseed identityが一致しません。')
      if (generation.current !== null && page.generation !== generation.current) {
        throw new Error('学習指標のgenerationがresetなしで変化しました。')
      }
      generation.current = page.generation
      cursor.current = Math.max(cursor.current, page.nextStep)
      setSeries((current) => mergeSeries(current, page.series))
      setError(null)
    } catch (reason) {
      if (requestId !== request.current) return
      setError(reason instanceof Error ? reason.message : '学習指標を取得できませんでした。')
    } finally {
      if (requestId === request.current) setLoading(false)
    }
  }, [api, jobId, seed, stableTags])

  useEffect(() => {
    request.current += 1
    cursor.current = 0
    generation.current = null
    setStatus(null)
    setSeries([])
    setError(null)
    setLoading(Boolean(jobId))
    void refresh()
    if (!jobId) return undefined
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => {
      window.clearInterval(timer)
      request.current += 1
    }
  }, [jobId, refresh, seed, tagKey])

  return { status, series, loading, error, refresh }
}
''',
    )

    _write(
        "studio/src/live/TrainingMetricChart.tsx",
        '''import { useMemo } from 'react'

import type { TrainingMetricPoint, TrainingMetricSeries } from '../data/types'

export type TrainingMetricWindow = 'all' | 256 | 512 | 1024

export interface TrainingMetricChartProps {
  series: TrainingMetricSeries | null
  selectedStep: number | null
  onSelectedStepChange: (step: number | null) => void
  windowSize: TrainingMetricWindow
}

function formatValue(value: number, unit: TrainingMetricSeries['unit']): string {
  if (unit === 'percent') return `${(value * 100).toLocaleString('ja-JP', { maximumFractionDigits: 3 })}%`
  if (unit === 'currency') return `${value.toLocaleString('ja-JP', { maximumFractionDigits: 2 })} USDT`
  if (unit === 'rate') return value.toExponential(3)
  return value.toLocaleString('ja-JP', { maximumFractionDigits: 5 })
}

function downsample(points: TrainingMetricPoint[], maximum = 1024): TrainingMetricPoint[] {
  if (points.length <= maximum) return points
  const bucketSize = Math.ceil(points.length / (maximum / 2))
  const selected: TrainingMetricPoint[] = []
  for (let start = 0; start < points.length; start += bucketSize) {
    const bucket = points.slice(start, start + bucketSize)
    let minimum = bucket[0]
    let maximumPoint = bucket[0]
    for (const point of bucket) {
      if (point.value < minimum.value) minimum = point
      if (point.value > maximumPoint.value) maximumPoint = point
    }
    selected.push(...(minimum.step <= maximumPoint.step ? [minimum, maximumPoint] : [maximumPoint, minimum]))
  }
  return selected.slice(0, maximum)
}

export function TrainingMetricChart({ series, selectedStep, onSelectedStepChange, windowSize }: TrainingMetricChartProps) {
  const source = series?.points ?? []
  const visible = windowSize === 'all' ? source : source.slice(-windowSize)
  const points = useMemo(() => downsample(visible), [visible])
  if (!series || points.length === 0) return <div className="training-chart-empty">未出力</div>

  const width = 720
  const height = 260
  const left = 64
  const right = 18
  const top = 18
  const bottom = 42
  const minStep = points[0].step
  const maxStep = points.at(-1)?.step ?? minStep
  const values = points.map((point) => point.value)
  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const padding = rawMax === rawMin ? Math.max(Math.abs(rawMax) * 0.05, 1e-9) : (rawMax - rawMin) * 0.08
  const minValue = rawMin - padding
  const maxValue = rawMax + padding
  const x = (step: number) => left + (maxStep === minStep ? (width - left - right) / 2 : (step - minStep) / (maxStep - minStep) * (width - left - right))
  const y = (value: number) => top + (maxValue - value) / (maxValue - minValue) * (height - top - bottom)
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(point.step)} ${y(point.value)}`).join(' ')
  const selectedIndex = selectedStep === null ? points.length - 1 : points.reduce((best, point, index) => Math.abs(point.step - selectedStep) < Math.abs(points[best].step - selectedStep) ? index : best, 0)
  const selected = points[selectedIndex]
  const chooseFromClientX = (clientX: number, target: SVGSVGElement) => {
    const rect = target.getBoundingClientRect()
    const local = rect.width > 0 ? (clientX - rect.left) / rect.width * width : 0
    const nearest = points.reduce((best, point, index) => Math.abs(x(point.step) - local) < Math.abs(x(points[best].step) - local) ? index : best, 0)
    onSelectedStepChange(points[nearest].step)
  }

  return (
    <div className="training-chart-shell">
      <div className="training-chart-summary">
        <strong>{series.displayName}</strong>
        <span>最新 {formatValue(source.at(-1)?.value ?? 0, series.unit)}</span>
        <span>min {formatValue(rawMin, series.unit)} / max {formatValue(rawMax, series.unit)}</span>
      </div>
      <svg
        className="training-metric-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${series.displayName} 学習ステップ推移`}
        tabIndex={0}
        onPointerMove={(event) => chooseFromClientX(event.clientX, event.currentTarget)}
        onPointerLeave={() => onSelectedStepChange(null)}
        onKeyDown={(event) => {
          if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
          event.preventDefault()
          const next = Math.max(0, Math.min(points.length - 1, selectedIndex + (event.key === 'ArrowRight' ? 1 : -1)))
          onSelectedStepChange(points[next].step)
        }}
      >
        <title>{series.displayName}。横軸はglobal stepです。</title>
        {[0, 0.5, 1].map((ratio) => {
          const guideY = top + ratio * (height - top - bottom)
          const value = maxValue - ratio * (maxValue - minValue)
          return <g key={ratio}><line className="training-chart-guide" x1={left} x2={width - right} y1={guideY} y2={guideY} /><text x={left - 8} y={guideY + 4} textAnchor="end">{formatValue(value, series.unit)}</text></g>
        })}
        <text x={left} y={height - 12}>Step {minStep.toLocaleString('ja-JP')}</text>
        <text x={width - right} y={height - 12} textAnchor="end">Step {maxStep.toLocaleString('ja-JP')}</text>
        <path className="training-chart-line" d={path} />
        <line className="training-chart-crosshair" x1={x(selected.step)} x2={x(selected.step)} y1={top} y2={height - bottom} />
        <circle className="training-chart-point" cx={x(selected.step)} cy={y(selected.value)} r={4} />
      </svg>
      <div className="training-chart-tooltip">Step {selected.step.toLocaleString('ja-JP')} · {formatValue(selected.value, series.unit)}</div>
    </div>
  )
}
''',
    )

    _write(
        "studio/src/live/TrainingDiagnosticsPanel.tsx",
        '''import { AlertTriangle, Activity, Database } from 'lucide-react'
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
''',
    )

    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        "import { MarketReplayChart } from '../live/MarketReplayChart'\n",
        "import { MarketReplayChart } from '../live/MarketReplayChart'\n"
        "import { TrainingDiagnosticsPanel } from '../live/TrainingDiagnosticsPanel'\n",
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        "type ReplayMode = 'live' | 'buffered'\n",
        "type LiveView = 'replay' | 'diagnostics'\n"
        "type ReplayMode = 'live' | 'buffered'\n",
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        "  const [replayMode, setReplayMode] = useState<ReplayMode>('buffered')\n",
        "  const [liveView, setLiveView] = useState<LiveView>('replay')\n"
        "  const [replayMode, setReplayMode] = useState<ReplayMode>('buffered')\n",
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        "      </header>\n\n      {(jobsError",
        '''      </header>

      <div className="live-view-selector" aria-label="Live Training view">
        <button type="button" aria-pressed={liveView === 'replay'} onClick={() => setLiveView('replay')}>市場リプレイ</button>
        <button type="button" aria-pressed={liveView === 'diagnostics'} onClick={() => setLiveView('diagnostics')}>学習診断</button>
      </div>

      {(jobsError''',
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        "      <div className=\"live-primary-grid\">\n",
        "      <div hidden={liveView !== 'diagnostics'}><TrainingDiagnosticsPanel job={selectedJob} seed={effectiveSeed} api={api} /></div>\n"
        "      <div className=\"live-replay-workspace\" hidden={liveView !== 'replay'}>\n"
        "      <div className=\"live-primary-grid\">\n",
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        "      </article>\n    </section>\n  )\n}\n",
        "      </article>\n      </div>\n    </section>\n  )\n}\n",
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        '<label className="live-job-select">Environment\n',
        '<label className="live-job-select" hidden={liveView !== \'replay\'}>Environment\n',
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        '<div className="live-segment-group" aria-label="リプレイモード">\n',
        '<div className="live-segment-group" aria-label="リプレイモード" hidden={liveView !== \'replay\'}>\n',
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        '<div className="live-buffer"><Database',
        '<div className="live-buffer" hidden={liveView !== \'replay\'}><Database',
    )
    _replace_once(
        "studio/src/pages/LiveTrainingPage.tsx",
        '<div className="live-segment-group" aria-label="タイム軸">\n',
        '<div className="live-segment-group" aria-label="タイム軸" hidden={liveView !== \'replay\'}>\n',
    )

    _append_once(
        "studio/src/liveTraining.css",
        ".training-diagnostics {",
        '''.live-view-selector { display: inline-flex; gap: 4px; padding: 3px; border: 1px solid var(--border); border-radius: 9px; background: rgba(10, 18, 30, .72); }
.live-view-selector button { border: 0; border-radius: 7px; padding: 7px 15px; background: transparent; color: var(--text-muted); font-weight: 700; }
.live-view-selector button[aria-pressed="true"] { background: var(--accent); color: #07111f; }
.live-replay-workspace { display: contents; }
.live-replay-workspace[hidden], [hidden] { display: none !important; }
.training-diagnostics { min-height: 0; display: grid; grid-template-rows: auto auto auto minmax(0, 1fr); gap: 8px; padding: 10px; border: 1px solid var(--border); border-radius: 12px; background: rgba(8, 15, 27, .86); overflow: hidden; }
.training-diagnostics-warning, .training-diagnostics-error, .training-diagnostics-empty { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 8px; background: rgba(245, 158, 11, .12); color: #f6d28b; font-size: 12px; }
.training-diagnostics-tabs { display: flex; gap: 5px; }
.training-diagnostics-tabs button { border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; background: transparent; color: var(--text-muted); }
.training-diagnostics-tabs button[aria-selected="true"] { border-color: var(--accent); color: var(--text); background: rgba(56, 189, 248, .12); }
.training-diagnostics-toolbar { display: flex; gap: 10px; }
.training-diagnostics-toolbar label { display: flex; align-items: center; gap: 6px; color: var(--text-muted); font-size: 12px; }
.training-diagnostics-toolbar select { min-width: 160px; }
.training-diagnostics-grid { min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 220px; gap: 10px; }
.training-diagnostics-charts { min-height: 0; display: grid; grid-template-rows: repeat(2, minmax(0, 1fr)); gap: 8px; }
.training-diagnostics-rail { overflow: auto; padding: 10px; border: 1px solid var(--border); border-radius: 10px; background: rgba(4, 10, 19, .5); }
.training-diagnostics-rail h2, .training-diagnostics-rail h3 { display: flex; align-items: center; gap: 6px; margin: 0 0 8px; }
.training-diagnostics-rail dl { margin: 0 0 12px; }
.training-diagnostics-rail dl div { display: grid; grid-template-columns: 76px minmax(0, 1fr); gap: 6px; padding: 4px 0; border-bottom: 1px solid rgba(148, 163, 184, .12); }
.training-diagnostics-rail dt { color: var(--text-muted); }
.training-diagnostics-rail dd { margin: 0; overflow-wrap: anywhere; }
.training-diagnostics-alert { display: flex; align-items: flex-start; gap: 5px; color: #fbbf24; }
.training-chart-shell { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; padding: 8px; border: 1px solid var(--border); border-radius: 10px; background: rgba(6, 12, 23, .72); }
.training-chart-summary { display: flex; gap: 14px; align-items: baseline; font-size: 11px; color: var(--text-muted); }
.training-chart-summary strong { margin-right: auto; color: var(--text); font-size: 13px; }
.training-metric-chart { width: 100%; height: 100%; min-height: 120px; outline: none; }
.training-metric-chart text { fill: var(--text-muted); font-size: 10px; }
.training-chart-guide { stroke: rgba(148, 163, 184, .18); stroke-width: 1; }
.training-chart-line { fill: none; stroke: var(--accent); stroke-width: 2; vector-effect: non-scaling-stroke; }
.training-chart-crosshair { stroke: rgba(248, 250, 252, .45); stroke-dasharray: 4 4; }
.training-chart-point { fill: var(--accent); stroke: #fff; stroke-width: 1.5; }
.training-chart-tooltip { justify-self: end; color: var(--text-muted); font-size: 11px; }
.training-chart-empty { display: grid; place-items: center; min-height: 120px; border: 1px dashed var(--border); border-radius: 10px; color: var(--text-muted); }
@media (max-width: 1100px) { .training-diagnostics-grid { grid-template-columns: minmax(0, 1fr) 180px; } }
''',
    )

    _write(
        "studio/src/live/TrainingMetricChart.test.tsx",
        '''import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { TrainingMetricChart } from './TrainingMetricChart'

const series = {
  tag: 'train/learning_rate', displayName: 'Learning rate', group: 'optimization' as const, unit: 'rate' as const,
  points: [{ step: 10, wallTime: 1, value: 0.00012 }, { step: 20, wallTime: 2, value: 0.0001 }],
}

describe('TrainingMetricChart', () => {
  it('renders an accessible global-step SVG and supports keyboard selection', () => {
    const change = vi.fn()
    render(<TrainingMetricChart series={series} selectedStep={10} onSelectedStepChange={change} windowSize="all" />)
    const chart = screen.getByRole('img', { name: /Learning rate 学習ステップ推移/ })
    expect(screen.getByText(/Step 10/)).toBeInTheDocument()
    fireEvent.keyDown(chart, { key: 'ArrowRight' })
    expect(change).toHaveBeenCalledWith(20)
  })

  it('renders empty and equal-valued series safely', () => {
    const { rerender } = render(<TrainingMetricChart series={null} selectedStep={null} onSelectedStepChange={() => undefined} windowSize="all" />)
    expect(screen.getByText('未出力')).toBeInTheDocument()
    rerender(<TrainingMetricChart series={{ ...series, points: [{ step: 1, wallTime: 1, value: 2 }] }} selectedStep={null} onSelectedStepChange={() => undefined} windowSize="all" />)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })
})
''',
    )

    _write(
        "studio/src/live/useTrainingMetrics.test.tsx",
        '''import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import { useTrainingMetrics } from './useTrainingMetrics'

function api(): StudioApi {
  return {
    loadTrainingMetricsStatus: vi.fn(async () => ({ available: true, selectedSeed: 3, availableSeeds: [3], availableTags: ['train/learning_rate'], lastStep: 20, source: 'events', generation: 'a'.repeat(64) })),
    loadTrainingMetricScalars: vi.fn(async () => ({ seed: 3, series: [{ tag: 'train/learning_rate', displayName: 'Learning rate', group: 'optimization', unit: 'rate', points: [{ step: 20, wallTime: 1, value: 0.0001 }] }], nextStep: 20, generation: 'a'.repeat(64), resetRequired: false })),
  } as StudioApi
}

describe('useTrainingMetrics', () => {
  it('loads status and finite scalar series for the selected seed', async () => {
    const source = api()
    const { result } = renderHook(() => useTrainingMetrics('job-1', 3, ['train/learning_rate'], source))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.status?.selectedSeed).toBe(3)
    expect(result.current.series[0].points[0].step).toBe(20)
  })
})
''',
    )

    _write(
        "studio/src/live/TrainingDiagnosticsPanel.test.tsx",
        '''import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { JobSummary } from '../data/types'
import { TrainingDiagnosticsPanel } from './TrainingDiagnosticsPanel'

const job = { id: 'job-1', status: 'running', runId: 'run-1' } as JobSummary
const api = {
  loadTrainingMetricsStatus: vi.fn(async () => ({ available: true, selectedSeed: 3, availableSeeds: [3], availableTags: ['train/learning_rate'], lastStep: 10, source: 'events', generation: 'a'.repeat(64) })),
  loadTrainingMetricScalars: vi.fn(async () => ({ seed: 3, series: [{ tag: 'train/learning_rate', displayName: 'Learning rate', group: 'optimization', unit: 'rate', points: [{ step: 10, wallTime: 1, value: 0.00012 }] }], nextStep: 10, generation: 'a'.repeat(64), resetRequired: false })),
} as unknown as StudioApi

describe('TrainingDiagnosticsPanel', () => {
  it('keeps optimization diagnostics separate from generalization evidence', async () => {
    render(<TrainingDiagnosticsPanel job={job} seed={3} api={api} />)
    expect(screen.getByText(/Checkpoint検証およびWalk-forward評価/)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('img', { name: /Learning rate/ })).toBeInTheDocument())
  })
})
''',
    )

    _append_once(
        "studio/src/api/studioApi.test.ts",
        "loads and validates allowlisted training metrics",
        '''describe('training metrics API', () => {
  it('loads and validates allowlisted training metrics', async () => {
    const { loadTrainingMetricScalars, loadTrainingMetricsStatus } = await import('./studioApi')
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.includes('/status')) return new Response(JSON.stringify({ available: true, selectedSeed: 3, availableSeeds: [3], availableTags: ['train/learning_rate'], lastStep: 20, source: 'events', generation: 'a'.repeat(64) }))
      return new Response(JSON.stringify({ seed: 3, series: [{ tag: 'train/learning_rate', displayName: 'Learning rate', group: 'optimization', unit: 'rate', points: [{ step: 20, wallTime: 1, value: 0.0001 }] }], nextStep: 20, generation: 'a'.repeat(64), resetRequired: false }))
    })
    await expect(loadTrainingMetricsStatus('job-1', 3, fetcher)).resolves.toMatchObject({ selectedSeed: 3 })
    await expect(loadTrainingMetricScalars('job-1', ['train/learning_rate'], 0, 512, 3, null, fetcher)).resolves.toMatchObject({ nextStep: 20 })
  })
})
''',
    )


if __name__ == "__main__":
    apply()
