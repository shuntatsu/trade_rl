import { demoOverview } from '../data/demoOverview'
import { offlineOverview } from '../data/offlineOverview'
import type {
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
  TrainingJobRequest,
} from '../data/types'
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
  loadRunComparison,
  loadEvidenceReport,
  loadServingMonitor,
}
