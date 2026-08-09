import { demoOverview } from '../data/demoOverview'
import { offlineOverview } from '../data/offlineOverview'
import type {
  CheckpointEvaluationsResponse,
  BehaviorCloningProgressResponse,
  ConfigListResponse,
  DatasetListResponse,
  EvidenceReport,
  JobListResponse,
  JobLogResponse,
  JobSummary,
  RunComparison,
  RunListResponse,
  ServingMonitorReport,
  StudioOverviewResult,
  TelemetryEventsResponse,
  TelemetryStatusResponse,
  TrainingJobRequest,
  TrainingMetricsResponse,
  TrainingMetricsStatusResponse,
} from '../data/types'
import { isBehaviorCloningProgress } from '../live/behaviorCloningGuards'
import {
  isCheckpointEvaluations,
  isTelemetryEvents,
  isTelemetryStatus,
} from '../live/telemetryGuards'
import {
  isTrainingMetricsResponse,
  isTrainingMetricsStatus,
} from '../live/trainingMetricGuards'
import {
  isConfigList,
  isDatasetList,
  isEvidenceReport,
  isJob,
  isJobList,
  isJobLog,
  isRecord,
  isRunComparison,
  isRunList,
  isServingMonitorReport,
  isStudioOverview,
} from './guards'

export class StudioApiError extends Error {
  readonly status: number
  readonly code: string | null

  constructor(status: number, message: string, code: string | null = null) {
    super(message)
    this.name = 'StudioApiError'
    this.status = status
    this.code = code
  }
}

async function errorPayload(response: Response): Promise<{ message: string; code: string | null }> {
  try {
    const payload: unknown = await response.json()
    if (isRecord(payload)) {
      if (typeof payload.detail === 'string') return { message: payload.detail, code: null }
      if (isRecord(payload.detail) && typeof payload.detail.message === 'string') {
        return {
          message: payload.detail.message,
          code: typeof payload.detail.code === 'string' ? payload.detail.code : null,
        }
      }
    }
  } catch {
    // Fall through to status text.
  }
  return { message: response.statusText || `Studio API request failed (${response.status})`, code: null }
}

async function requestJson<T>(
  path: string,
  fetcher: typeof fetch,
  validate: (value: unknown) => value is T,
  init?: RequestInit,
): Promise<T> {
  const response = await fetcher(path, init)
  if (!response.ok) {
    const resolved = await errorPayload(response)
    throw new StudioApiError(response.status, resolved.message, resolved.code)
  }
  const payload: unknown = await response.json()
  if (!validate(payload)) throw new StudioApiError(502, 'Studio API returned an invalid response', 'invalid_response')
  return payload
}

export async function loadStudioOverview(
  fetcher: typeof fetch = fetch,
  options: { demo?: boolean } = {},
): Promise<StudioOverviewResult> {
  if (options.demo) return { source: 'demo', overview: demoOverview, error: null }
  try {
    const response = await fetcher('/api/studio/overview')
    if (!response.ok) {
      const resolved = await errorPayload(response)
      return { source: 'offline', overview: offlineOverview, error: resolved.message }
    }
    const payload: unknown = await response.json()
    if (!isStudioOverview(payload)) {
      return { source: 'offline', overview: offlineOverview, error: 'Studio API returned an invalid overview' }
    }
    return { source: 'live', overview: payload, error: null }
  } catch (reason) {
    return {
      source: 'offline',
      overview: offlineOverview,
      error: reason instanceof Error ? reason.message : 'Studio API is offline',
    }
  }
}

export const loadDatasets = (fetcher: typeof fetch = fetch): Promise<DatasetListResponse> =>
  requestJson('/api/studio/datasets', fetcher, isDatasetList)
export const loadRuns = (fetcher: typeof fetch = fetch): Promise<RunListResponse> =>
  requestJson('/api/studio/runs', fetcher, isRunList)
export const loadConfigs = (fetcher: typeof fetch = fetch): Promise<ConfigListResponse> =>
  requestJson('/api/studio/configs', fetcher, isConfigList)
export const loadJobs = (fetcher: typeof fetch = fetch): Promise<JobListResponse> =>
  requestJson('/api/studio/jobs', fetcher, isJobList)

export function submitTrainingJob(request: TrainingJobRequest, fetcher: typeof fetch = fetch): Promise<JobSummary> {
  return requestJson('/api/studio/jobs/training', fetcher, isJob, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}

export function cancelJob(jobId: string, fetcher: typeof fetch = fetch): Promise<JobSummary> {
  return requestJson(`/api/studio/jobs/${encodeURIComponent(jobId)}/cancel`, fetcher, isJob, { method: 'POST' })
}

export function loadJobLog(jobId: string, fetcher: typeof fetch = fetch): Promise<JobLogResponse> {
  return requestJson(`/api/studio/jobs/${encodeURIComponent(jobId)}/log?limit=200`, fetcher, isJobLog)
}

export function loadTelemetryStatus(
  jobId: string,
  seed: number | null = null,
  fetcher: typeof fetch = fetch,
): Promise<TelemetryStatusResponse> {
  const query = seed === null ? '' : `?seed=${encodeURIComponent(seed)}`
  return requestJson(
    `/api/studio/jobs/${encodeURIComponent(jobId)}/telemetry/status${query}`,
    fetcher,
    isTelemetryStatus,
  )
}

export function loadTelemetryEvents(
  jobId: string,
  afterSequence = 0,
  limit = 512,
  seed: number | null = null,
  streamGeneration: string | null = null,
  fetcher: typeof fetch = fetch,
): Promise<TelemetryEventsResponse> {
  const parameters = new URLSearchParams({
    after_sequence: String(afterSequence),
    limit: String(limit),
  })
  if (seed !== null) parameters.set('seed', String(seed))
  if (streamGeneration !== null) {
    parameters.set('stream_generation', streamGeneration)
  }
  return requestJson(
    `/api/studio/jobs/${encodeURIComponent(jobId)}/telemetry/events?${parameters.toString()}`,
    fetcher,
    isTelemetryEvents,
  )
}

export function loadTrainingMetricsStatus(
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
  jobId: string,
  fetcher: typeof fetch = fetch,
): Promise<CheckpointEvaluationsResponse> {
  return requestJson(
    `/api/studio/jobs/${encodeURIComponent(jobId)}/checkpoint-evaluations`,
    fetcher,
    isCheckpointEvaluations,
  )
}

export function loadBehaviorCloningProgress(
  jobId: string,
  fetcher: typeof fetch = fetch,
): Promise<BehaviorCloningProgressResponse> {
  return requestJson(
    `/api/studio/jobs/${encodeURIComponent(jobId)}/behavior-cloning/progress`,
    fetcher,
    isBehaviorCloningProgress,
  )
}

export function loadRunComparison(
  leftResourceId: string,
  rightResourceId: string,
  fetcher: typeof fetch = fetch,
): Promise<RunComparison> {
  const query = `left_resource_id=${encodeURIComponent(leftResourceId)}&right_resource_id=${encodeURIComponent(rightResourceId)}`
  return requestJson(`/api/studio/compare?${query}`, fetcher, isRunComparison)
}

export function loadEvidenceReport(runResourceId: string, fetcher: typeof fetch = fetch): Promise<EvidenceReport> {
  return requestJson(`/api/studio/runs/${encodeURIComponent(runResourceId)}/evidence`, fetcher, isEvidenceReport)
}

export function loadServingMonitor(fetcher: typeof fetch = fetch): Promise<ServingMonitorReport> {
  return requestJson('/api/studio/serving', fetcher, isServingMonitorReport)
}

export interface StudioApi {
  loadDatasets: () => Promise<DatasetListResponse>
  loadRuns: () => Promise<RunListResponse>
  loadConfigs: () => Promise<ConfigListResponse>
  loadJobs: () => Promise<JobListResponse>
  submitTrainingJob: (request: TrainingJobRequest) => Promise<JobSummary>
  cancelJob: (jobId: string) => Promise<JobSummary>
  loadJobLog: (jobId: string) => Promise<JobLogResponse>
  loadTelemetryStatus: (jobId: string, seed?: number | null) => Promise<TelemetryStatusResponse>
  loadTelemetryEvents: (jobId: string, afterSequence?: number, limit?: number, seed?: number | null, streamGeneration?: string | null) => Promise<TelemetryEventsResponse>
  loadCheckpointEvaluations?: (jobId: string) => Promise<CheckpointEvaluationsResponse>
  loadBehaviorCloningProgress?: (jobId: string) => Promise<BehaviorCloningProgressResponse>
  loadTrainingMetricsStatus?: (jobId: string, seed?: number | null) => Promise<TrainingMetricsStatusResponse>
  loadTrainingMetricScalars?: (jobId: string, tags: string[], afterStep?: number, limit?: number, seed?: number | null, generation?: string | null) => Promise<TrainingMetricsResponse>
  loadRunComparison: (leftResourceId: string, rightResourceId: string) => Promise<RunComparison>
  loadEvidenceReport: (runResourceId: string) => Promise<EvidenceReport>
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
  loadTelemetryStatus,
  loadTelemetryEvents,
  loadCheckpointEvaluations,
  loadBehaviorCloningProgress,
  loadTrainingMetricsStatus,
  loadTrainingMetricScalars,
  loadRunComparison,
  loadEvidenceReport,
  loadServingMonitor,
}
