import { render, screen, waitFor } from '@testing-library/react'
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
