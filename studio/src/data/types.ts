export type ProductionStatus = 'GO' | 'NO-GO'
export type AlertLevel = 'warning' | 'info'

export interface SystemMetric {
  label: string
  value: number
  detail: string
}

export interface DatasetSummary {
  name: string
  market: string
  symbols: string[]
  timeframes: string[]
  range: string
  status: 'VALID' | 'INVALID'
  featureCount: number
  updated: string
}

export interface ActiveJob {
  id: string
  algorithm: string
  phase: string
  seedProgress: string
  progress: number
}

export interface RunSummary {
  id: string
  algorithm: string
  period: string
  sharpe: number
  maxDrawdown: number
  status: ProductionStatus
}

export interface StudioAlert {
  level: AlertLevel
  message: string
  age: string
}

export interface EquityPoint {
  label: string
  rl: number
  baseline: number
}

export interface StabilityFold {
  label: string
  low: number
  median: number
  high: number
}

export interface ProductionAssessment {
  status: ProductionStatus
  reasons: string[]
}

export interface StudioOverview {
  system: {
    gpuName: string
    cudaReady: boolean
    pythonVersion: string
    metrics: SystemMetric[]
  }
  latestDataset: DatasetSummary
  activeJobs: ActiveJob[]
  runs: RunSummary[]
  alerts: StudioAlert[]
  equity: EquityPoint[]
  stability: StabilityFold[]
  assessment: ProductionAssessment
}

export interface StudioOverviewResult {
  source: 'api' | 'demo'
  overview: StudioOverview
}
