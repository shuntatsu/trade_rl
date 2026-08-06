import { describe, expect, it } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { isRunComparison, isStudioOverview } from './guards'

const comparison = {
  leftResourceId: 'run-left-resource',
  rightResourceId: 'run-right-resource',
  leftRunId: 'run-left',
  rightRunId: 'run-right',
  eligibility: { status: 'COMPARABLE', reasons: [], datasetId: 'd'.repeat(64) },
  metrics: [],
  configDifferences: [],
  folds: [],
  wealth: [
    { label: 'start', foldIndex: null, left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 },
    { label: '10', foldIndex: 0, left: 1.01, right: 1.02, leftBaseline: 1.005, rightBaseline: 1.005 },
  ],
  productionStatus: 'NO-GO',
}

describe('isStudioOverview', () => {
  it('accepts the complete decision snapshot', () => {
    expect(isStudioOverview(demoOverview)).toBe(true)
  })

  it.each([
    ['missing Evidence', { ...demoOverview, evidence: undefined }],
    ['negative Evidence count', { ...demoOverview, evidence: { ...demoOverview.evidence, blockerCount: -1 } }],
    ['verified count above required', { ...demoOverview, evidence: { ...demoOverview.evidence, verifiedCount: 5, requiredCount: 4 } }],
    ['missing alert identity', { ...demoOverview, alerts: [{ ...demoOverview.alerts[0], id: undefined }] }],
    ['invalid alert time', { ...demoOverview, alerts: [{ ...demoOverview.alerts[0], occurredAt: 1 }] }],
  ])('rejects %s', (_label, value) => {
    expect(isStudioOverview(value)).toBe(false)
  })
})

describe('isRunComparison', () => {
  it('accepts authoritative nullable fold identities', () => {
    expect(isRunComparison(comparison)).toBe(true)
  })

  it.each([
    ['missing fold identity', { ...comparison, wealth: [{ ...comparison.wealth[1], foldIndex: undefined }] }],
    ['fractional fold identity', { ...comparison, wealth: [{ ...comparison.wealth[1], foldIndex: 0.5 }] }],
    ['negative fold identity', { ...comparison, wealth: [{ ...comparison.wealth[1], foldIndex: -1 }] }],
  ])('rejects %s', (_label, value) => {
    expect(isRunComparison(value)).toBe(false)
  })
})
