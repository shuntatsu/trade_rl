import { demoOverview } from '../data/demoOverview'
import type {
  ConfigListResponse,
  DatasetListResponse,
  JobListResponse,
  JobLogResponse,
  JobSummary,
  RunListResponse,
  RunComparison,
  EvidenceReport,
  ServingMonitorReport,
  StudioOverview,
  StudioOverviewResult,
  TrainingJobRequest,
} from '../data/types'

export class StudioApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'StudioApiError'
    this.status = status
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isStudioOverview(value: unknown): value is StudioOverview {
  if (!isRecord(value)) return false
  return Boolean(
    value.system &&
      'latestDataset' in value &&
      Array.isArray(value.activeJobs) &&
      Array.isArray(value.runs) &&
      Array.isArray(value.alerts) &&
      Array.isArray(value.equity) &&
      Array.isArray(value.stability) &&
      value.assessment,
  )
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload: unknown = await response.json()
    if (isRecord(payload) && typeof payload.detail === 'string') return payload.detail
  } catch {
    // Fall through to status text.
  }
  return response.statusText || `Studio API request failed (${response.status})`
}

async function requestJson<T>(
  path: string,
  fetcher: typeof fetch,
  init?: RequestInit,
  validate?: (value: unknown) => value is T,
): Promise<T> {
  const response = await fetcher(path, init)
  if (!response.ok) throw new StudioApiError(response.status, await errorMessage(response))
  const payload: unknown = await response.json()
  if (validate && !validate(payload)) {
    throw new StudioApiError(502, 'Studio API returned an invalid response')
  }
  return payload as T
}

function isRunComparison(value: unknown): value is RunComparison {
  if (!isRecord(value)) return false
  return (
    typeof value.leftRunId === 'string' &&
    typeof value.rightRunId === 'string' &&
    Array.isArray(value.metrics) &&
    Array.isArray(value.configDifferences) &&
    Array.isArray(value.folds) &&
    Array.isArray(value.wealth) &&
    value.productionStatus === 'NO-GO'
  )
}

function isEvidenceReport(value: unknown): value is EvidenceReport {
  if (!isRecord(value) || !isRecord(value.files)) return false
  return (
    typeof value.runId === 'string' &&
    typeof value.runKind === 'string' &&
    (value.status === 'VALID' || value.status === 'INVALID') &&
    Array.isArray(value.nodes) &&
    value.productionStatus === 'NO-GO'
  )
}

function isServingMonitorReport(value: unknown): value is ServingMonitorReport {
  if (!isRecord(value)) return false
  return (
    (value.state === 'IDLE' || value.state === 'VALID' || value.state === 'INVALID') &&
    Array.isArray(value.checks) &&
    value.productionStatus === 'NO-GO'
  )
}

export async function loadStudioOverview(
  fetcher: typeof fetch = fetch,
): Promise<StudioOverviewResult> {
  try {
    const response = await fetcher('/api/studio/overview')
    if (!response.ok) return { source: 'demo', overview: demoOverview }
    const payload: unknown = await response.json()
    if (!isStudioOverview(payload)) return { source: 'demo', overview: demoOverview }
    return { source: 'api', overview: payload }
  } catch {
    return { source: 'demo', overview: demoOverview }
  }
}

export function loadDatasets(fetcher: typeof fetch = fetch): Promise<DatasetListResponse> {
  return requestJson('/api/studio/datasets', fetcher)
}

export function loadRuns(fetcher: typeof fetch = fetch): Promise<RunListResponse> {
  return requestJson('/api/studio/runs', fetcher)
}

export function loadConfigs(fetcher: typeof fetch = fetch): Promise<ConfigListResponse> {
  return requestJson('/api/studio/configs', fetcher)
}

export function loadJobs(fetcher: typeof fetch = fetch): Promise<JobListResponse> {
  return requestJson('/api/studio/jobs', fetcher)
}

export function submitTrainingJob(
  request: TrainingJobRequest,
  fetcher: typeof fetch = fetch,
): Promise<JobSummary> {
  return requestJson('/api/studio/jobs/training', fetcher, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}

export function cancelJob(jobId: string, fetcher: typeof fetch = fetch): Promise<JobSummary> {
  return requestJson(`/api/studio/jobs/${encodeURIComponent(jobId)}/cancel`, fetcher, {
    method: 'POST',
  })
}

export function loadJobLog(
  jobId: string,
  fetcher: typeof fetch = fetch,
): Promise<JobLogResponse> {
  return requestJson(`/api/studio/jobs/${encodeURIComponent(jobId)}/log?limit=200`, fetcher)
}


export function loadRunComparison(
  leftRunId: string,
  rightRunId: string,
  fetcher: typeof fetch = fetch,
): Promise<RunComparison> {
  const query = `left_run_id=${encodeURIComponent(leftRunId)}&right_run_id=${encodeURIComponent(rightRunId)}`
  return requestJson(`/api/studio/compare?${query}`, fetcher, undefined, isRunComparison)
}

export function loadEvidenceReport(
  runId: string,
  fetcher: typeof fetch = fetch,
): Promise<EvidenceReport> {
  return requestJson(
    `/api/studio/runs/${encodeURIComponent(runId)}/evidence`,
    fetcher,
    undefined,
    isEvidenceReport,
  )
}

export function loadServingMonitor(
  fetcher: typeof fetch = fetch,
): Promise<ServingMonitorReport> {
  return requestJson('/api/studio/serving', fetcher, undefined, isServingMonitorReport)
}

export interface StudioApi {
  loadDatasets: () => Promise<DatasetListResponse>
  loadRuns: () => Promise<RunListResponse>
  loadConfigs: () => Promise<ConfigListResponse>
  loadJobs: () => Promise<JobListResponse>
  submitTrainingJob: (request: TrainingJobRequest) => Promise<JobSummary>
  cancelJob: (jobId: string) => Promise<JobSummary>
  loadJobLog: (jobId: string) => Promise<JobLogResponse>
  loadRunComparison: (leftRunId: string, rightRunId: string) => Promise<RunComparison>
  loadEvidenceReport: (runId: string) => Promise<EvidenceReport>
  loadServingMonitor: () => Promise<ServingMonitorReport>
}

export const studioApi: StudioApi = {
  loadDatasets,
  loadRuns,
  loadConfigs,
  loadJobs,
  submitTrainingJob,
  cancelJob,
  loadJobLog,
  loadRunComparison,
  loadEvidenceReport,
  loadServingMonitor,
}
