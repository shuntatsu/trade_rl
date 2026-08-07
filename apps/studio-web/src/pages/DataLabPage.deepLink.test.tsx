import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { DatasetSummary } from '../data/types'
import { DataLabPage } from './DataLabPage'

const firstDataset: DatasetSummary = {
  id: 'dataset-111111111111111111111111',
  datasetId: 'd'.repeat(64),
  name: 'btc-eth',
  relativePath: 'artifacts/datasets/btc-eth',
  market: 'continuous_24_7',
  symbols: ['BTCUSDT', 'ETHUSDT'],
  timeframes: ['15m', '1h'],
  range: '2026-01-01 → 2026-02-01',
  status: 'VALID',
  featureCount: 226,
  barCount: 744,
  symbolCount: 2,
  updated: '2026-07-19T00:00:00+00:00',
  validationError: null,
}

const requestedDataset: DatasetSummary = {
  ...firstDataset,
  id: 'dataset-222222222222222222222222',
  datasetId: 'e'.repeat(64),
  name: 'sol-market',
  relativePath: 'artifacts/datasets/sol-market',
  symbols: ['SOLUSDT'],
  symbolCount: 1,
}

function api(): StudioApi {
  return {
    loadDatasets: vi.fn().mockResolvedValue({
      items: [firstDataset, requestedDataset],
      total: 2,
      invalid: 0,
    }),
  } as unknown as StudioApi
}

beforeEach(() => {
  window.history.replaceState({}, '', '/')
})

describe('DataLabPage Dashboard drill-through', () => {
  it('selects the Dataset resource requested by the URL', async () => {
    window.history.replaceState({}, '', `/?page=data&dataset=${requestedDataset.id}`)

    render(<DataLabPage api={api()} />)

    const requested = await screen.findByRole('button', { name: /sol-market valid/i })
    await waitFor(() => expect(requested).toHaveClass('runtime-row--selected'))
    expect(screen.getAllByText('SOLUSDT').length).toBeGreaterThan(0)
  })
})
