import type { StudioOverview } from './types'

export const offlineOverview: StudioOverview = {
  system: {
    gpuName: 'Unavailable',
    cudaReady: false,
    pythonVersion: 'Unavailable',
    metrics: [],
  },
  latestDataset: null,
  activeJobs: [],
  runs: [],
  alerts: [
    {
      id: 'studio:offline',
      level: 'warning',
      message: 'Studio APIへ接続できません。artifact情報は表示していません。',
      age: 'now',
      occurredAt: null,
    },
  ],
  equity: [],
  stability: [],
  evidence: {
    runResourceId: null,
    status: 'UNAVAILABLE',
    requiredCount: 0,
    verifiedCount: 0,
    blockerCount: 0,
    updatedAt: null,
  },
  assessment: {
    status: 'NO-GO',
    reasons: ['Studio APIがオフラインです', '研究artifactの検証状態を確認できません'],
  },
}
