export type ProductionStatus = 'NO-GO'
export type AlertLevel = 'warning' | 'info'
export type ValidationStatus = 'VALID' | 'INVALID'
export type RuntimeSource = 'live' | 'offline' | 'demo'
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
  datasetId: string
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
  runId: string
  manifestDigest: string | null
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
  id: string
  configDigest: string | null
  name: string
  relativePath: string
  algorithm: string
  status: ValidationStatus
  validationError: string | null
}

export interface JobSummary {
  id: string
  schemaVersion: 'studio_job_v2'
  kind: 'training'
  status: JobStatus
  runId: string
  configResourceId: string
  datasetResourceId: string
  configDigest: string
  datasetId: string
  configPath: string
  datasetPath: string
  artifactRoot: string
  submittedAt: string
  ownerInstanceId: string
  startedAt: string | null
  completedAt: string | null
  pid: number | null
  pidStartToken: string | null
  exitCode: number | null
  cancellable: boolean
  error: string | null
}

export interface TrainingJobRequest {
  configResourceId: string
  datasetResourceId: string
  runId: string
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
  source: RuntimeSource
  overview: StudioOverview
  error: string | null
}

export interface ComparisonMetric {
  key: string
  label: string
  leftValue: number | null
  rightValue: number | null
  delta: number | null
  preference: 'higher' | 'lower' | 'neutral'
}

export interface ConfigDifference {
  path: string
  left: string | null
  right: string | null
}

export interface FoldComparison {
  label: string
  leftSelectedReturn: number | null
  leftBaselineReturn: number | null
  rightSelectedReturn: number | null
  rightBaselineReturn: number | null
}

export interface ComparisonSeriesPoint {
  label: string
  left: number | null
  right: number | null
  leftBaseline: number | null
  rightBaseline: number | null
}

export interface ComparisonEligibility {
  status: 'COMPARABLE' | 'PARTIALLY_COMPARABLE' | 'NOT_COMPARABLE'
  reasons: string[]
  datasetId: string | null
}

export interface RunComparison {
  leftResourceId: string
  rightResourceId: string
  leftRunId: string
  rightRunId: string
  eligibility: ComparisonEligibility
  metrics: ComparisonMetric[]
  configDifferences: ConfigDifference[]
  folds: FoldComparison[]
  wealth: ComparisonSeriesPoint[]
  productionStatus: ProductionStatus
}

export type EvidenceNodeStatus = 'VERIFIED' | 'PRESENT' | 'ABSENT' | 'INVALID'

export interface EvidenceNode {
  key: string
  label: string
  status: EvidenceNodeStatus
  required: boolean
  digest: string | null
  path: string | null
  detail: string
}

export interface FileIntegritySummary {
  status: 'VERIFIED' | 'INVALID'
  declaredCount: number
  verifiedCount: number
  totalSizeBytes: number
}

export interface EvidenceReport {
  runResourceId: string
  runId: string
  runKind: string
  status: ValidationStatus
  productionStatus: ProductionStatus
  nodes: EvidenceNode[]
  files: FileIntegritySummary
  validationError: string | null
}

export interface ServingCheck {
  key: string
  label: string
  status: 'PASS' | 'WARN' | 'FAIL'
  detail: string
}

export interface PaperInferenceSnapshot {
  recordedAt: string
  bundleDigest: string
  datasetId: string
  decisionIndex: number
  targetWeights: Record<string, number>
  latencyMs: number
  snapshotDigest: string
}

export interface ServingMonitorReport {
  state: 'IDLE' | 'VALID' | 'INVALID'
  productionStatus: ProductionStatus
  activeBundleDigest: string | null
  datasetId: string | null
  runKind: string | null
  policyDigest: string | null
  actionSchema: string | null
  observationSchema: string | null
  releaseAttestationPresent: boolean
  checks: ServingCheck[]
  paperSnapshot: PaperInferenceSnapshot | null
  validationError: string | null
}
