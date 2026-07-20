import { describe, expect, it, vi } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import {
  StudioApiError,
  cancelJob,
  loadConfigs,
  loadDatasets,
  loadEvidenceReport,
  loadJobLog,
  loadJobs,
  loadRunComparison,
  loadRuns,
  loadServingMonitor,
  loadStudioOverview,
  submitTrainingJob,
} from './studioApi'

const dataset = {
  id: 'dataset-111111111111111111111111',
  datasetId: 'd'.repeat(64),
  name: 'btc-eth',
  relativePath: 'artifacts/datasets/btc-eth',
  market: 'continuous_24_7',
  symbols: ['BTCUSDT', 'ETHUSDT'],
  timeframes: ['1h'],
  range: '2026-01-01 → 2026-02-01',
  status: 'VALID' as const,
  featureCount: 226,
  barCount: 744,
  symbolCount: 2,
  updated: '2026-07-19T00:00:00+00:00',
  validationError: null,
}
const datasetResponse = { items: [dataset], total: 1, invalid: 0 }

const job = {
  id: 'job-1', schemaVersion: 'studio_job_v2' as const, kind: 'training' as const, status: 'running' as const,
  runId: 'run-1', configResourceId: 'config-111111111111111111111111', datasetResourceId: dataset.id,
  configDigest: 'c'.repeat(64), datasetId: dataset.datasetId, configPath: 'configs/training.json',
  datasetPath: dataset.relativePath, artifactRoot: 'artifacts/research', submittedAt: '2026-07-19T00:00:00+00:00',
  ownerInstanceId: 'studio-instance', startedAt: '2026-07-19T00:00:01+00:00', completedAt: null,
  pid: 123, pidStartToken: '99', exitCode: null, cancellable: true, error: null,
}

const comparison = {
  leftResourceId: 'run-111111111111111111111111', rightResourceId: 'run-222222222222222222222222',
  leftRunId: 'run left', rightRunId: 'run/right', eligibility: { status: 'COMPARABLE' as const, reasons: [], datasetId: dataset.datasetId },
  metrics: [], configDifferences: [], folds: [], wealth: [], productionStatus: 'NO-GO' as const,
}
const evidence = {
  runResourceId: comparison.leftResourceId, runId: 'run left', runKind: 'research_exploratory', status: 'VALID' as const,
  productionStatus: 'NO-GO' as const, nodes: [], files: { status: 'VERIFIED' as const, declaredCount: 1, verifiedCount: 1, totalSizeBytes: 10 }, validationError: null,
}
const serving = {
  state: 'IDLE' as const, productionStatus: 'NO-GO' as const, activeBundleDigest: null, datasetId: null,
  runKind: null, policyDigest: null, actionSchema: null, observationSchema: null, releaseAttestationPresent: false,
  checks: [], paperSnapshot: null, validationError: null,
}

describe('loadStudioOverview', () => {
  it('returns live API data for a successful response', async () => {
    const result = await loadStudioOverview(async () => new Response(JSON.stringify(demoOverview), { status: 200 }))
    expect(result.source).toBe('live')
    expect(result.error).toBeNull()
  })

  it('reports offline without substituting fabricated demo artifacts', async () => {
    const result = await loadStudioOverview(async () => { throw new Error('offline') })
    expect(result.source).toBe('offline')
    expect(result.overview.latestDataset).toBeNull()
    expect(result.error).toBe('offline')
  })

  it('uses demo data only when explicitly requested', async () => {
    const result = await loadStudioOverview(fetch, { demo: true })
    expect(result).toEqual({ source: 'demo', overview: demoOverview, error: null })
  })
})

describe('studio runtime API', () => {
  it('runtime-validates all list payloads', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.endsWith('/datasets')) return new Response(JSON.stringify(datasetResponse))
      if (path.endsWith('/runs')) return new Response(JSON.stringify({ items: [], total: 0, invalid: 0 }))
      if (path.endsWith('/configs')) return new Response(JSON.stringify({ items: [], total: 0, invalid: 0 }))
      if (path.endsWith('/jobs')) return new Response(JSON.stringify({ items: [job], total: 1 }))
      return new Response('not found', { status: 404 })
    })
    await expect(loadDatasets(fetcher)).resolves.toEqual(datasetResponse)
    await expect(loadRuns(fetcher)).resolves.toEqual({ items: [], total: 0, invalid: 0 })
    await expect(loadConfigs(fetcher)).resolves.toEqual({ items: [], total: 0, invalid: 0 })
    await expect(loadJobs(fetcher)).resolves.toEqual({ items: [job], total: 1 })

    await expect(loadDatasets(async () => new Response(JSON.stringify({ items: [{}], total: 1, invalid: 0 })))).rejects.toMatchObject({ status: 502 })
  })

  it('submits immutable catalog resource identities', async () => {
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(JSON.parse(String(init?.body))).toEqual({ configResourceId: job.configResourceId, datasetResourceId: job.datasetResourceId, runId: 'run-1' })
      return new Response(JSON.stringify(job), { status: 201 })
    })
    await expect(submitTrainingJob({ configResourceId: job.configResourceId, datasetResourceId: job.datasetResourceId, runId: 'run-1' }, fetcher)).resolves.toEqual(job)
  })

  it('parses typed API errors', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.endsWith('/log?limit=200')) return new Response(JSON.stringify({ jobId: 'job-1', lines: ['training'], truncated: false }))
      if (path.endsWith('/cancel')) return new Response(JSON.stringify({ ...job, status: 'cancelled', cancellable: false }))
      return new Response(JSON.stringify({ detail: { code: 'resource_not_found', message: 'missing' } }), { status: 404 })
    })
    await expect(loadJobLog('job-1', fetcher)).resolves.toMatchObject({ lines: ['training'] })
    await expect(cancelJob('job-1', fetcher)).resolves.toMatchObject({ status: 'cancelled' })
    await expect(loadDatasets(fetcher)).rejects.toEqual(expect.objectContaining<Partial<StudioApiError>>({ status: 404, code: 'resource_not_found', message: 'missing' }))
  })
})

describe('studio audit API', () => {
  it('uses resource IDs in encoded endpoints', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.includes('/compare?')) return new Response(JSON.stringify(comparison))
      if (path.includes('/evidence')) return new Response(JSON.stringify(evidence))
      if (path.endsWith('/serving')) return new Response(JSON.stringify(serving))
      return new Response('not found', { status: 404 })
    })
    await expect(loadRunComparison(comparison.leftResourceId, comparison.rightResourceId, fetcher)).resolves.toEqual(comparison)
    await expect(loadEvidenceReport(comparison.leftResourceId, fetcher)).resolves.toEqual(evidence)
    await expect(loadServingMonitor(fetcher)).resolves.toEqual(serving)
    expect(fetcher).toHaveBeenCalledWith(`/api/studio/compare?left_resource_id=${comparison.leftResourceId}&right_resource_id=${comparison.rightResourceId}`, undefined)
    expect(fetcher).toHaveBeenCalledWith(`/api/studio/runs/${comparison.leftResourceId}/evidence`, undefined)
  })

  it('rejects malformed audit payloads', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ status: 'ok' })))
    await expect(loadRunComparison('left', 'right', fetcher)).rejects.toMatchObject({ status: 502 })
    await expect(loadEvidenceReport('left', fetcher)).rejects.toMatchObject({ status: 502 })
    await expect(loadServingMonitor(fetcher)).rejects.toMatchObject({ status: 502 })
  })
})
