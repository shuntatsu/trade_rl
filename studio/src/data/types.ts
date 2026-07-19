export type ProductionStatus = 'NO-GO'
export type AlertLevel = 'warning' | 'info'
export type ValidationStatus = 'VALID' | 'INVALID'
export type JobStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelling'
  | 'cancelled'

export interface SystemMetric {
  label: string
  value: number
  detail: string
}

export interface DatasetSummary {
  id: string
  name: string
  relativePath: string
  market: string
  symbols: string[]
  timeframes: string[]
  range: string
  status: ValidationStatus
  featureCount: number
  barCount: number
  symbolCount: number
  updated: string
  validationError: string | null
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
  relativePath: string
  runKind: string
  algorithm: string
  datasetId: string
  period: string
  createdAt: string
  completedAt: string
  fileCount: number
  sharpe: number | null
  maxDrawdown: number | null
  totalReturn: number | null
  productionStatus: ProductionStatus
  status: ValidationStatus
  validationError: string | null
}

export interface ConfigSummary {
  name: string
  relativePath: string
  algorithm: string
  status: ValidationStatus
  validationError: string | null
}

export interface JobSummary {
  id: string
  kind: 'training'
  status: JobStatus
  runId: string
  configPath: string
  datasetPath: string
  artifactRoot: string
  submittedAt: string
  startedAt: string | null
  completedAt: string | null
  pid: number | null
  exitCode: number | null
  error: string | null
}

export interface TrainingJobRequest {
  configPath: string
  datasetPath: string
  runId: string
  artifactRoot?: string
}

export interface JobLogResponse {
  jobId: string
  lines: string[]
  truncated: boolean
}

export interface DatasetListResponse {
  items: DatasetSummary[]
  total: number
  invalid: number
}

export interface RunListResponse {
  items: RunSummary[]
  total: number
  invalid: number
}

export interface ConfigListResponse {
  items: ConfigSummary[]
  total: number
  invalid: number
}

export interface JobListResponse {
  items: JobSummary[]
  total: number
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
  latestDataset: DatasetSummary | null
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
