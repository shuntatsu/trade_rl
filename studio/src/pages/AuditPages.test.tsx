import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { EvidenceReport, RunSummary, ServingMonitorReport } from '../data/types'
import { EvidencePage } from './EvidencePage'
import { ServingPage } from './ServingPage'

const run: RunSummary = {
  id: 'run-111111111111111111111111', runId: 'run-001', manifestDigest: '1'.repeat(64), relativePath: 'research/runs/run-001', runKind: 'research_selected_final', algorithm: 'ppo', datasetId: 'dataset-1',
  period: '2026-01-01 — 2026-01-02', createdAt: '2026-01-01', completedAt: '2026-01-02', fileCount: 9,
  sharpe: 1.2, maxDrawdown: 0.08, totalReturn: 0.2, productionStatus: 'NO-GO', status: 'VALID', validationError: null,
}

const evidence: EvidenceReport = {
  runResourceId: run.id,
  runId: 'run-001', runKind: 'research_selected_final', status: 'VALID', productionStatus: 'NO-GO', validationError: null,
  files: { status: 'VERIFIED', declaredCount: 9, verifiedCount: 9, totalSizeBytes: 4096 },
  nodes: [
    { key: 'dataset', label: 'Dataset identity', status: 'PRESENT', required: true, digest: 'a'.repeat(64), path: 'dataset-reference.json', detail: 'identity is bound' },
    { key: 'config', label: 'Training config', status: 'VERIFIED', required: true, digest: 'b'.repeat(64), path: 'training-config.json', detail: 'declared file closure verified' },
  ],
}

const serving: ServingMonitorReport = {
  state: 'VALID', productionStatus: 'NO-GO', activeBundleDigest: 'c'.repeat(64), datasetId: 'dataset-1', runKind: 'research_selected_final',
  policyDigest: 'd'.repeat(64), actionSchema: 'target_weight_v1', observationSchema: 'sequence_v1', releaseAttestationPresent: true,
  checks: [{ key: 'closure', label: 'Bundle closure', status: 'PASS', detail: '9 declared files verified' }],
  paperSnapshot: { recordedAt: '2026-07-20T00:00:00Z', bundleDigest: 'c'.repeat(64), datasetId: 'dataset-1', decisionIndex: 42, targetWeights: { BTCUSDT: 0.4, ETHUSDT: -0.2, CASH: 0.8 }, latencyMs: 8.4, snapshotDigest: 'e'.repeat(64) },
  validationError: null,
}

describe('EvidencePage', () => {
  it('renders the validated artifact chain and file closure', async () => {
    const api: Pick<StudioApi, 'loadRuns' | 'loadEvidenceReport'> = {
      loadRuns: vi.fn().mockResolvedValue({ items: [run], total: 1, invalid: 0 }),
      loadEvidenceReport: vi.fn().mockResolvedValue(evidence),
    }
    render(<EvidencePage api={api} />)

    expect(await screen.findByText('Dataset identity')).toBeInTheDocument()
    expect(screen.getByText('Evidence coverage')).toBeInTheDocument()
    expect(screen.getByText('Training config')).toBeInTheDocument()
    expect(screen.getAllByText('VERIFIED').length).toBeGreaterThan(0)
    expect(screen.getByText('9 / 9')).toBeInTheDocument()
    expect(screen.getAllByText('NO-GO').length).toBeGreaterThan(0)
  })
})

describe('ServingPage', () => {
  it('shows a read-only active bundle and paper inference snapshot', async () => {
    const api: Pick<StudioApi, 'loadServingMonitor'> = {
      loadServingMonitor: vi.fn().mockResolvedValue(serving),
    }
    render(<ServingPage api={api} />)

    expect(await screen.findByText('Bundle closure')).toBeInTheDocument()
    expect(screen.getByText('READ ONLY')).toBeInTheDocument()
    expect(screen.getByText('BTCUSDT')).toBeInTheDocument()
    expect(screen.getByText('40.00%')).toBeInTheDocument()
    expect(screen.getByText('-20.00%')).toBeInTheDocument()
    expect(screen.getByText('Non-cash gross')).toBeInTheDocument()
    expect(screen.getByLabelText('ETHUSDT target weight -20.00%')).toHaveAttribute('data-direction', 'short')
    expect(screen.queryByRole('button', { name: /activate|注文|発注/i })).not.toBeInTheDocument()
  })
})
