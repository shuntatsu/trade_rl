import { describe, expect, it, vi } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import {
  StudioApiError,
  cancelJob,
  loadConfigs,
  loadDatasets,
  loadJobLog,
  loadJobs,
  loadRuns,
  loadStudioOverview,
  submitTrainingJob,
} from './studioApi'

const datasetResponse = {
  items: [
    {
      id: 'dataset-1',
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
    },
  ],
  total: 1,
  invalid: 0,
}

const job = {
  id: 'job-1',
  kind: 'training' as const,
  status: 'running' as const,
  runId: 'run-1',
  configPath: 'configs/training.json',
  datasetPath: 'artifacts/datasets/btc-eth',
  artifactRoot: 'artifacts/research',
  submittedAt: '2026-07-19T00:00:00+00:00',
  startedAt: '2026-07-19T00:00:01+00:00',
  completedAt: null,
  pid: 123,
  exitCode: null,
  error: null,
}

describe('loadStudioOverview', () => {
  it('returns API data for a successful response', async () => {
    const result = await loadStudioOverview(async () =>
      new Response(JSON.stringify(demoOverview), { status: 200 }),
    )

    expect(result.source).toBe('api')
    expect(result.overview.latestDataset?.name).toBe(demoOverview.latestDataset?.name)
  })

  it('falls back to deterministic demo data when the request fails', async () => {
    const result = await loadStudioOverview(async () => {
      throw new Error('offline')
    })

    expect(result).toEqual({ source: 'demo', overview: demoOverview })
  })

  it('falls back when the response shape is invalid', async () => {
    const result = await loadStudioOverview(async () =>
      new Response(JSON.stringify({ status: 'ok' }), { status: 200 }),
    )

    expect(result.source).toBe('demo')
  })
})

describe('studio runtime API', () => {
  it('loads dataset, run, config, and job lists from their endpoints', async () => {
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
  })

  it('submits a training job as JSON', async () => {
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.method).toBe('POST')
      expect(init?.headers).toEqual({ 'Content-Type': 'application/json' })
      expect(JSON.parse(String(init?.body))).toEqual({
        configPath: 'configs/training.json',
        datasetPath: 'artifacts/datasets/btc-eth',
        runId: 'run-1',
      })
      return new Response(JSON.stringify(job), { status: 201 })
    })

    await expect(
      submitTrainingJob(
        {
          configPath: 'configs/training.json',
          datasetPath: 'artifacts/datasets/btc-eth',
          runId: 'run-1',
        },
        fetcher,
      ),
    ).resolves.toEqual(job)
  })

  it('loads logs, cancels jobs, and exposes API errors', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.endsWith('/log?limit=200')) {
        return new Response(JSON.stringify({ jobId: 'job-1', lines: ['training'], truncated: false }))
      }
      if (path.endsWith('/cancel')) return new Response(JSON.stringify({ ...job, status: 'cancelled' }))
      return new Response(JSON.stringify({ detail: 'missing' }), { status: 404 })
    })

    await expect(loadJobLog('job-1', fetcher)).resolves.toEqual({
      jobId: 'job-1',
      lines: ['training'],
      truncated: false,
    })
    await expect(cancelJob('job-1', fetcher)).resolves.toMatchObject({ status: 'cancelled' })
    await expect(loadDatasets(fetcher)).rejects.toEqual(
      expect.objectContaining<Partial<StudioApiError>>({ status: 404, message: 'missing' }),
    )
  })
})

describe('studio audit API', () => {
  it('loads comparison, evidence, and serving reports from encoded endpoints', async () => {
    const comparison = {
      leftRunId: 'run left',
      rightRunId: 'run/right',
      metrics: [],
      configDifferences: [],
      folds: [],
      wealth: [],
      productionStatus: 'NO-GO' as const,
    }
    const evidence = {
      runId: 'run left',
      runKind: 'research_exploratory',
      status: 'VALID' as const,
      productionStatus: 'NO-GO' as const,
      nodes: [],
      files: { status: 'VERIFIED' as const, declaredCount: 1, verifiedCount: 1, totalSizeBytes: 10 },
      validationError: null,
    }
    const serving = {
      state: 'IDLE' as const,
      productionStatus: 'NO-GO' as const,
      activeBundleDigest: null,
      datasetId: null,
      runKind: null,
      policyDigest: null,
      actionSchema: null,
      observationSchema: null,
      releaseAttestationPresent: false,
      checks: [],
      paperSnapshot: null,
      validationError: null,
    }
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.includes('/compare?')) return new Response(JSON.stringify(comparison))
      if (path.includes('/evidence')) return new Response(JSON.stringify(evidence))
      if (path.endsWith('/serving')) return new Response(JSON.stringify(serving))
      return new Response('not found', { status: 404 })
    })

    const { loadEvidenceReport, loadRunComparison, loadServingMonitor } = await import('./studioApi')

    await expect(loadRunComparison('run left', 'run/right', fetcher)).resolves.toEqual(comparison)
    await expect(loadEvidenceReport('run left', fetcher)).resolves.toEqual(evidence)
    await expect(loadServingMonitor(fetcher)).resolves.toEqual(serving)
    expect(fetcher).toHaveBeenCalledWith(
      '/api/studio/compare?left_run_id=run%20left&right_run_id=run%2Fright',
      undefined,
    )
    expect(fetcher).toHaveBeenCalledWith('/api/studio/runs/run%20left/evidence', undefined)
  })

  it('rejects malformed audit payloads instead of rendering untrusted shapes', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ status: 'ok' })))
    const { loadEvidenceReport, loadRunComparison, loadServingMonitor } = await import('./studioApi')

    await expect(loadRunComparison('left', 'right', fetcher)).rejects.toMatchObject({ status: 502 })
    await expect(loadEvidenceReport('left', fetcher)).rejects.toMatchObject({ status: 502 })
    await expect(loadServingMonitor(fetcher)).rejects.toMatchObject({ status: 502 })
  })
})
