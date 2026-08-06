import { describe, expect, it } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { isStudioOverview } from './guards'

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
