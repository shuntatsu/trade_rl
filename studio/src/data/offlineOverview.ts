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
      level: 'warning',
      message: 'Studio APIへ接続できません。artifact情報は表示していません。',
      age: 'now',
    },
  ],
  equity: [],
  stability: [],
  assessment: {
    status: 'NO-GO',
    reasons: ['Studio APIがオフラインです', '研究artifactの検証状態を確認できません'],
  },
}
