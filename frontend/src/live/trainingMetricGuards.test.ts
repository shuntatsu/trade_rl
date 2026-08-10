import { describe, expect, it } from 'vitest'

import { isTrainingMetricsStatus } from './trainingMetricGuards'

describe('training metric guards', () => {
  it('accepts reward component metrics for intermediate audits', () => {
    expect(isTrainingMetricsStatus({
      available: true,
      selectedSeed: 0,
      availableSeeds: [0],
      availableTags: ['trade_rl/reward_absolute_component_mean'],
      lastStep: 1024,
      source: null,
      generation: null,
    })).toBe(true)
  })
})
