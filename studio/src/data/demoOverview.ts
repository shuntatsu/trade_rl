import type { StudioOverview } from './types'

export const demoOverview: StudioOverview = {
  system: {
    gpuName: 'NVIDIA RTX 4090',
    cudaReady: true,
    pythonVersion: '3.12.3',
    metrics: [
      { label: 'GPU', value: 65, detail: 'メモリ 14.2 / 24GB' },
      { label: 'CPU', value: 18, detail: '16 cores' },
      { label: 'メモリ', value: 32, detail: '20.5 / 64GB' },
      { label: 'ディスク', value: 42, detail: '420 / 1TB' },
    ],
  },
  latestDataset: {
    name: 'binance_spot_multi_tf_v1',
    market: 'Spot',
    symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    timeframes: ['15m', '1h', '4h', '1d'],
    range: '2023-01-01 → 2024-12-31',
    status: 'VALID',
    featureCount: 226,
    updated: '2日前',
  },
  activeJobs: [
    { id: 'job_20250719_001', algorithm: 'PPO', phase: 'fold 2/6', seedProgress: 'seed 3/5', progress: 61 },
    { id: 'job_20250719_002', algorithm: 'SAC', phase: 'training', seedProgress: 'seed 1/3', progress: 23 },
  ],
  runs: [
    { id: 'run_20250718_001', algorithm: 'PPO', period: '2023-01 → 2024-12', sharpe: 1.32, maxDrawdown: -12.4, status: 'NO-GO' },
    { id: 'run_20250717_003', algorithm: 'SAC', period: '2023-01 → 2024-12', sharpe: 0.85, maxDrawdown: -14.8, status: 'NO-GO' },
    { id: 'run_20250716_002', algorithm: 'PPO', period: '2023-01 → 2024-12', sharpe: 1.78, maxDrawdown: -9.3, status: 'GO' },
    { id: 'run_20250715_001', algorithm: 'TD3', period: '2023-01 → 2024-12', sharpe: 0.92, maxDrawdown: -11.7, status: 'NO-GO' },
  ],
  alerts: [
    { level: 'warning', message: 'seedが1つだけです（推奨: ≥3）', age: '2時間前' },
    { level: 'warning', message: 'feeまたはslippageが0に設定されています', age: '3時間前' },
    { level: 'info', message: '新しいデータセットが生成されました', age: '5時間前' },
    { level: 'info', message: 'CUDAドライバは最新です', age: '1日前' },
  ],
  equity: [
    { label: '2023-01', rl: 1.0, baseline: 1.0 },
    { label: '2023-04', rl: 1.08, baseline: 0.99 },
    { label: '2023-07', rl: 1.16, baseline: 1.03 },
    { label: '2023-10', rl: 1.27, baseline: 1.08 },
    { label: '2024-01', rl: 1.34, baseline: 1.12 },
    { label: '2024-04', rl: 1.42, baseline: 1.15 },
    { label: '2024-07', rl: 1.53, baseline: 1.22 },
    { label: '2024-10', rl: 1.69, baseline: 1.31 },
    { label: '2024-12', rl: 1.84, baseline: 1.43 },
  ],
  stability: [
    { label: 'Fold 1', low: -0.62, median: 0.14, high: 1.72 },
    { label: 'Fold 2', low: -0.48, median: 0.31, high: 1.24 },
    { label: 'Fold 3', low: -0.71, median: 0.08, high: 0.96 },
    { label: 'Fold 4', low: -0.53, median: 0.44, high: 1.18 },
    { label: 'Fold 5', low: -0.69, median: -0.12, high: 0.73 },
    { label: 'Fold 6', low: -0.42, median: 0.27, high: 1.04 },
  ],
  assessment: {
    status: 'NO-GO',
    reasons: ['最低seedがbaselineを下回った', 'コスト2倍で優位性が消失', 'fold間の結果が不安定'],
  },
}
