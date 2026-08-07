import { describe, expect, it } from 'vitest'

import type { RunComparison } from '../data/types'
import {
  buildComparisonDirectLabels,
  buildComparisonWorkspaceModel,
  summarizeComparisonRange,
} from './comparisonWorkspaceModel'

const comparison = {
  leftResourceId: 'left-resource',
  rightResourceId: 'right-resource',
  leftRunId: 'left-run',
  rightRunId: 'right-run',
  eligibility: { status: 'COMPARABLE', reasons: ['aligned'], datasetId: 'd'.repeat(64) },
  metrics: [
    { key: 'total_return', label: 'Total return', leftValue: 0.1, rightValue: 0.2, delta: 0.1, preference: 'higher' },
    { key: 'total_cost', label: 'Total cost', leftValue: 0.01, rightValue: 0.02, delta: 0.01, preference: 'lower' },
  ],
  configDifferences: [],
  folds: [
    { label: 'Fold 1', leftSelectedReturn: 0.02, leftBaselineReturn: 0.01, rightSelectedReturn: 0.04, rightBaselineReturn: 0.01 },
    { label: 'Fold 2', leftSelectedReturn: 0.03, leftBaselineReturn: 0.01, rightSelectedReturn: 0.05, rightBaselineReturn: 0.01 },
  ],
  wealth: [
    { label: 'start', foldIndex: null, left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 },
    { label: '10', foldIndex: 0, left: 1.01, right: 1.02, leftBaseline: 1.005, rightBaseline: 1.005 },
    { label: '11', foldIndex: 0, left: 1.02, right: 1.04, leftBaseline: 1.01, rightBaseline: 1.01 },
    { label: '20', foldIndex: 1, left: 1.03, right: 1.06, leftBaseline: 1.015, rightBaseline: 1.015 },
    { label: '21', foldIndex: 1, left: 1.04, right: 1.08, leftBaseline: 1.02, rightBaseline: 1.02 },
  ],
  productionStatus: 'NO-GO',
} satisfies RunComparison & { wealth: Array<RunComparison['wealth'][number] & { foldIndex: number | null }> }

describe('comparisonWorkspaceModel', () => {
  it('uses one wealth domain and a zero-inclusive delta domain', () => {
    const model = buildComparisonWorkspaceModel(comparison)
    expect(model.wealthDomain.minimum).toBeLessThan(1)
    expect(model.wealthDomain.maximum).toBeGreaterThan(1.08)
    expect(model.deltaDomain.minimum).toBeLessThanOrEqual(0)
    expect(model.deltaDomain.maximum).toBeGreaterThanOrEqual(0.04)
    expect(model.points.at(-1)?.delta).toBeCloseTo(0.04)
  })

  it('derives contiguous fold spans from authoritative point identities', () => {
    const model = buildComparisonWorkspaceModel(comparison)
    expect(model.foldSpans).toEqual([
      expect.objectContaining({ foldIndex: 0, startIndex: 1, endIndex: 2, label: 'Fold 1' }),
      expect.objectContaining({ foldIndex: 1, startIndex: 3, endIndex: 4, label: 'Fold 2' }),
    ])
  })

  it('separates colliding direct labels deterministically', () => {
    const model = buildComparisonWorkspaceModel(comparison)
    const positions = model.directLabels.map((item) => item.position)
    expect(new Set(positions.map((item) => item.toFixed(4))).size).toBe(positions.length)
    expect(positions.every((item) => item >= 0 && item <= 1)).toBe(true)
  })

  it('builds direct labels from the visible endpoint rather than the global final point', () => {
    const model = buildComparisonWorkspaceModel(comparison)
    const labels = buildComparisonDirectLabels(model.points[2], model.wealthDomain)
    expect(labels.find((item) => item.key === 'right')?.value).toBe(1.04)
    expect(labels.find((item) => item.key === 'right')?.value).not.toBe(1.08)
  })

  it('summarizes a selected ordinal range without claiming wall-clock time', () => {
    const model = buildComparisonWorkspaceModel(comparison)
    const summary = summarizeComparisonRange(model, 1, 4)
    expect(summary.startLabel).toBe('10')
    expect(summary.endLabel).toBe('21')
    expect(summary.leftReturn).toBeCloseTo(1.04 / 1.01 - 1)
    expect(summary.rightReturn).toBeCloseTo(1.08 / 1.02 - 1)
    expect(summary.relativeReturn).toBeGreaterThan(0)
    expect(summary.winner).toBe('right')
  })

  it('applies metric preferences instead of treating every positive delta as improvement', () => {
    const model = buildComparisonWorkspaceModel(comparison)
    expect(model.metricVerdicts.map((item) => [item.key, item.verdict])).toEqual([
      ['total_return', 'improved'],
      ['total_cost', 'worse'],
    ])
  })
})
