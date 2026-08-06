import type { ActiveJob, RunSummary, StudioAlert, StudioOverview } from '../data/types'

export type DashboardStageKey = 'data' | 'training' | 'evaluation' | 'evidence' | 'release'
export type DashboardStageState = 'BLOCKED' | 'ATTENTION' | 'ACTIVE' | 'READY' | 'IDLE' | 'UNKNOWN'
export type DashboardDecisionStage = DashboardStageKey | 'alert'
export type DashboardDecisionSeverity = 'critical' | 'warning' | 'info'
export type DashboardWorkspace = 'data' | 'runs' | 'live' | 'compare' | 'evidence'

export interface DashboardAction {
  label: string
  workspace: DashboardWorkspace
  params: Record<string, string>
}

export interface DashboardStageView {
  key: DashboardStageKey
  label: string
  state: DashboardStageState
  headline: string
  detail: string
  count: number | null
  primaryEntityId: string | null
}

export interface DashboardDecisionItem {
  id: string
  stage: DashboardDecisionStage
  severity: DashboardDecisionSeverity
  title: string
  explanation: string
  occurredAt: string | null
  age: string | null
  action: DashboardAction | null
}

export interface DashboardLatestResult {
  resourceId: string
  runId: string
  algorithm: string
  period: string
  totalReturn: number | null
  sharpe: number | null
  maxDrawdown: number | null
  validationStatus: 'VALID' | 'INVALID'
  productionStatus: 'NO-GO'
  completedAt: string
}

export interface DashboardEnvironmentView {
  gpuName: string
  cudaReady: boolean
  pythonVersion: string
  metrics: StudioOverview['system']['metrics']
}

export interface DashboardCockpitModel {
  productionStatus: 'NO-GO'
  stages: DashboardStageView[]
  decisions: DashboardDecisionItem[]
  primaryDecisionId: string | null
  latestResult: DashboardLatestResult | null
  environment: DashboardEnvironmentView
}

const SEVERITY_PRIORITY: Record<DashboardDecisionSeverity, number> = {
  critical: 0,
  warning: 1,
  info: 2,
}

function dataStage(overview: StudioOverview): DashboardStageView {
  const dataset = overview.latestDataset
  if (dataset === null) {
    return { key: 'data', label: 'Data', state: 'UNKNOWN', headline: 'Dataset unavailable', detail: '検証対象のDatasetがありません。', count: null, primaryEntityId: null }
  }
  if (dataset.status === 'INVALID') {
    return { key: 'data', label: 'Data', state: 'BLOCKED', headline: dataset.name, detail: dataset.validationError ?? 'Dataset validation failed.', count: dataset.symbolCount, primaryEntityId: dataset.id }
  }
  return { key: 'data', label: 'Data', state: 'READY', headline: dataset.name, detail: `${dataset.market} · ${dataset.timeframes.join(' / ')}`, count: dataset.symbolCount, primaryEntityId: dataset.id }
}

function trainingStage(data: DashboardStageView, jobs: ActiveJob[]): DashboardStageView {
  const active = jobs[0]
  if (active) {
    return { key: 'training', label: 'Training', state: 'ACTIVE', headline: active.phase, detail: `${active.algorithm} · ${active.seedProgress}`, count: jobs.length, primaryEntityId: active.id }
  }
  if (data.state === 'READY') return { key: 'training', label: 'Training', state: 'IDLE', headline: 'Ready to start', detail: '実行中の学習Jobはありません。', count: 0, primaryEntityId: null }
  if (data.state === 'BLOCKED') return { key: 'training', label: 'Training', state: 'BLOCKED', headline: 'Waiting for Data', detail: 'Dataset blockerを先に解消してください。', count: 0, primaryEntityId: null }
  return { key: 'training', label: 'Training', state: 'UNKNOWN', headline: 'Upstream unknown', detail: 'Dataset状態を確認できません。', count: 0, primaryEntityId: null }
}

function evaluationStage(training: DashboardStageView, latest: RunSummary | null): DashboardStageView {
  if (latest?.status === 'INVALID') return { key: 'evaluation', label: 'Evaluation', state: 'BLOCKED', headline: latest.runId, detail: latest.validationError ?? 'Run validation failed.', count: 1, primaryEntityId: latest.id }
  if (latest?.status === 'VALID') return { key: 'evaluation', label: 'Evaluation', state: 'READY', headline: latest.runId, detail: `${latest.algorithm} · ${latest.period}`, count: 1, primaryEntityId: latest.id }
  if (training.state === 'ACTIVE' || training.state === 'IDLE') return { key: 'evaluation', label: 'Evaluation', state: 'IDLE', headline: 'No evaluation yet', detail: '検証済みRunを待っています。', count: 0, primaryEntityId: null }
  return { key: 'evaluation', label: 'Evaluation', state: 'UNKNOWN', headline: 'Upstream unavailable', detail: '評価可能なRunがありません。', count: 0, primaryEntityId: null }
}

function evidenceStage(overview: StudioOverview): DashboardStageView {
  const evidence = overview.evidence
  if (evidence.status === 'VERIFIED' && evidence.blockerCount === 0) {
    return { key: 'evidence', label: 'Evidence', state: 'READY', headline: 'Evidence verified', detail: `${evidence.verifiedCount} / ${evidence.requiredCount} required nodes`, count: 0, primaryEntityId: evidence.runResourceId }
  }
  if (evidence.status === 'INVALID' || evidence.blockerCount > 0) {
    return { key: 'evidence', label: 'Evidence', state: 'BLOCKED', headline: `${evidence.blockerCount} blocker${evidence.blockerCount === 1 ? '' : 's'}`, detail: `${evidence.verifiedCount} / ${evidence.requiredCount} required nodes verified`, count: evidence.blockerCount, primaryEntityId: evidence.runResourceId }
  }
  if (evidence.status === 'INCOMPLETE') {
    return { key: 'evidence', label: 'Evidence', state: 'ATTENTION', headline: 'Evidence incomplete', detail: `${evidence.verifiedCount} / ${evidence.requiredCount} required nodes verified`, count: evidence.requiredCount - evidence.verifiedCount, primaryEntityId: evidence.runResourceId }
  }
  return { key: 'evidence', label: 'Evidence', state: 'UNKNOWN', headline: 'Evidence unavailable', detail: '検査可能なRunがありません。', count: null, primaryEntityId: null }
}

function releaseStage(overview: StudioOverview): DashboardStageView {
  const reasons = overview.assessment.reasons
  return {
    key: 'release',
    label: 'Release',
    state: reasons.length ? 'BLOCKED' : 'ATTENTION',
    headline: 'NO-GO',
    detail: reasons[0] ?? 'Release blockerの理由が記録されていません。',
    count: reasons.length,
    primaryEntityId: null,
  }
}

function addDecision(target: DashboardDecisionItem[], seen: Set<string>, item: DashboardDecisionItem): void {
  if (seen.has(item.id)) return
  seen.add(item.id)
  target.push(item)
}

function alertStage(alert: StudioAlert): DashboardDecisionStage {
  if (alert.id.startsWith('dataset:')) return 'data'
  if (alert.id.startsWith('run:')) return 'evaluation'
  if (alert.id.startsWith('job:')) return 'training'
  return 'alert'
}

function timeValue(value: string | null): number | null {
  if (value === null) return null
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : null
}

function dependencyPriority(item: DashboardDecisionItem): number {
  if (item.stage === 'data') return 0
  if (item.stage === 'evaluation' && item.severity !== 'info') return 1
  if (item.stage === 'evidence') return 2
  if (item.stage === 'release') return 3
  if (item.stage === 'training') return 4
  if (item.stage === 'evaluation') return 5
  return 6
}

function sortDecisions(items: DashboardDecisionItem[]): DashboardDecisionItem[] {
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => {
      const dependency = dependencyPriority(left.item) - dependencyPriority(right.item)
      if (dependency !== 0) return dependency
      const severity = SEVERITY_PRIORITY[left.item.severity] - SEVERITY_PRIORITY[right.item.severity]
      if (severity !== 0) return severity
      const actionable = Number(right.item.action !== null) - Number(left.item.action !== null)
      if (actionable !== 0) return actionable
      const leftTime = timeValue(left.item.occurredAt)
      const rightTime = timeValue(right.item.occurredAt)
      if (leftTime !== null && rightTime !== null && leftTime !== rightTime) return rightTime - leftTime
      return left.index - right.index || left.item.id.localeCompare(right.item.id)
    })
    .map(({ item }) => item)
}

function buildDecisions(overview: StudioOverview, latest: RunSummary | null): DashboardDecisionItem[] {
  const decisions: DashboardDecisionItem[] = []
  const seen = new Set<string>()
  const dataset = overview.latestDataset
  if (dataset === null) {
    addDecision(decisions, seen, { id: 'dataset:no-valid', stage: 'data', severity: 'warning', title: 'Datasetがありません', explanation: '学習前に検証済みDatasetを登録してください。', occurredAt: null, age: null, action: { label: 'Data Labを開く', workspace: 'data', params: {} } })
  } else if (dataset.status === 'INVALID') {
    addDecision(decisions, seen, { id: `dataset:${dataset.id}:invalid`, stage: 'data', severity: 'critical', title: `Dataset ${dataset.name} が無効です`, explanation: dataset.validationError ?? 'Dataset validation failed.', occurredAt: null, age: dataset.updated, action: { label: 'Data Labで確認', workspace: 'data', params: { dataset: dataset.id } } })
  }

  for (const job of overview.activeJobs) {
    addDecision(decisions, seen, { id: `job:${job.id}:active`, stage: 'training', severity: 'info', title: `${job.id} を監視`, explanation: `${job.algorithm} · ${job.phase} · ${job.seedProgress}`, occurredAt: null, age: null, action: { label: 'Live Trainingを開く', workspace: 'live', params: { job: job.id } } })
  }

  if (latest?.status === 'INVALID') {
    addDecision(decisions, seen, { id: `run:${latest.id}:invalid`, stage: 'evaluation', severity: 'critical', title: `Run ${latest.runId} が無効です`, explanation: latest.validationError ?? 'Run artifact validation failed.', occurredAt: latest.completedAt, age: null, action: { label: 'Run Centerを開く', workspace: 'runs', params: {} } })
  } else if (latest?.status === 'VALID') {
    const validRuns = overview.runs.filter((item) => item.status === 'VALID')
    const comparisonAction = validRuns.length >= 2
      ? { label: 'Compareで検証', workspace: 'compare' as const, params: { left: validRuns[1].id, right: validRuns[0].id } }
      : null
    addDecision(decisions, seen, { id: `run:${latest.id}:valid`, stage: 'evaluation', severity: 'info', title: `Run ${latest.runId} を確認`, explanation: '有効な評価結果です。優劣はCompareの適格性契約で判定します。', occurredAt: latest.completedAt, age: null, action: comparisonAction })
  }

  const evidence = overview.evidence
  if (evidence.status === 'INVALID' || evidence.blockerCount > 0) {
    addDecision(decisions, seen, { id: `evidence:${evidence.runResourceId ?? 'unknown'}:invalid`, stage: 'evidence', severity: 'critical', title: `Evidence blocker ${evidence.blockerCount}件`, explanation: `${evidence.verifiedCount} / ${evidence.requiredCount} required nodes verified.`, occurredAt: evidence.updatedAt, age: null, action: evidence.runResourceId ? { label: 'Evidenceを検査', workspace: 'evidence', params: { evidenceRun: evidence.runResourceId } } : null })
  } else if (evidence.status === 'INCOMPLETE') {
    addDecision(decisions, seen, { id: `evidence:${evidence.runResourceId ?? 'unknown'}:incomplete`, stage: 'evidence', severity: 'warning', title: 'Evidenceが未完了です', explanation: `${evidence.verifiedCount} / ${evidence.requiredCount} required nodes verified.`, occurredAt: evidence.updatedAt, age: null, action: evidence.runResourceId ? { label: 'Evidenceを確認', workspace: 'evidence', params: { evidenceRun: evidence.runResourceId } } : null })
  } else if (evidence.status === 'UNAVAILABLE' && latest !== null) {
    addDecision(decisions, seen, { id: `evidence:${latest.id}:unavailable`, stage: 'evidence', severity: 'warning', title: 'Evidenceを取得できません', explanation: 'Runは存在しますがEvidence summaryがありません。', occurredAt: null, age: null, action: { label: 'Evidenceを確認', workspace: 'evidence', params: { evidenceRun: latest.id } } })
  }

  const reasons = overview.assessment.reasons
  addDecision(decisions, seen, { id: 'release:no-go', stage: 'release', severity: reasons.length ? 'critical' : 'warning', title: 'ReleaseはNO-GOです', explanation: reasons.join(' · ') || 'Release blockerの理由が記録されていません。', occurredAt: null, age: null, action: overview.evidence.runResourceId ? { label: 'Evidenceから確認', workspace: 'evidence', params: { evidenceRun: overview.evidence.runResourceId } } : null })

  for (const alert of overview.alerts) {
    addDecision(decisions, seen, { id: alert.id, stage: alertStage(alert), severity: alert.level === 'warning' ? 'warning' : 'info', title: alert.message, explanation: alert.age, occurredAt: alert.occurredAt, age: alert.age, action: null })
  }
  return sortDecisions(decisions)
}

export function buildDashboardCockpitModel(overview: StudioOverview): DashboardCockpitModel {
  const data = dataStage(overview)
  const training = trainingStage(data, overview.activeJobs)
  const latest = overview.runs[0] ?? null
  const evaluation = evaluationStage(training, latest)
  const evidence = evidenceStage(overview)
  const release = releaseStage(overview)
  const decisions = buildDecisions(overview, latest)
  const primary = decisions.find((item) => item.action !== null) ?? decisions[0] ?? null
  return {
    productionStatus: 'NO-GO',
    stages: [data, training, evaluation, evidence, release],
    decisions,
    primaryDecisionId: primary?.id ?? null,
    latestResult: latest === null ? null : {
      resourceId: latest.id,
      runId: latest.runId,
      algorithm: latest.algorithm,
      period: latest.period,
      totalReturn: latest.totalReturn,
      sharpe: latest.sharpe,
      maxDrawdown: latest.maxDrawdown,
      validationStatus: latest.status,
      productionStatus: latest.productionStatus,
      completedAt: latest.completedAt,
    },
    environment: {
      gpuName: overview.system.gpuName,
      cudaReady: overview.system.cudaReady,
      pythonVersion: overview.system.pythonVersion,
      metrics: overview.system.metrics,
    },
  }
}
