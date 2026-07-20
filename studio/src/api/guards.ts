import type {
  ConfigListResponse,
  DatasetListResponse,
  EvidenceReport,
  JobListResponse,
  JobLogResponse,
  JobSummary,
  RunComparison,
  RunListResponse,
  ServingMonitorReport,
  StudioOverview,
} from '../data/types'

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const isString = (value: unknown): value is string => typeof value === 'string'
const isBoolean = (value: unknown): value is boolean => typeof value === 'boolean'
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const isNullableString = (value: unknown): value is string | null => value === null || isString(value)
const isNullableNumber = (value: unknown): value is number | null => value === null || isNumber(value)
const isStrings = (value: unknown): value is string[] => Array.isArray(value) && value.every(isString)
const isValidation = (value: unknown) => value === 'VALID' || value === 'INVALID'

function isDataset(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.datasetId)
    && isString(value.name)
    && isString(value.relativePath)
    && isString(value.market)
    && isStrings(value.symbols)
    && isStrings(value.timeframes)
    && isString(value.range)
    && isValidation(value.status)
    && isNumber(value.featureCount)
    && isNumber(value.barCount)
    && isNumber(value.symbolCount)
    && isString(value.updated)
    && isNullableString(value.validationError)
}

function isRun(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.runId)
    && isNullableString(value.manifestDigest)
    && isString(value.relativePath)
    && isString(value.runKind)
    && isString(value.algorithm)
    && isString(value.datasetId)
    && isString(value.period)
    && isString(value.createdAt)
    && isString(value.completedAt)
    && isNumber(value.fileCount)
    && isNullableNumber(value.sharpe)
    && isNullableNumber(value.maxDrawdown)
    && isNullableNumber(value.totalReturn)
    && value.productionStatus === 'NO-GO'
    && isValidation(value.status)
    && isNullableString(value.validationError)
}

function isConfig(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isNullableString(value.configDigest)
    && isString(value.name)
    && isString(value.relativePath)
    && isString(value.algorithm)
    && isValidation(value.status)
    && isNullableString(value.validationError)
}

export function isJob(value: unknown): value is JobSummary {
  const statuses = ['queued', 'running', 'succeeded', 'failed', 'cancelling', 'cancelled']
  return isRecord(value)
    && isString(value.id)
    && value.schemaVersion === 'studio_job_v2'
    && value.kind === 'training'
    && statuses.includes(String(value.status))
    && isString(value.runId)
    && isString(value.configResourceId)
    && isString(value.datasetResourceId)
    && isString(value.configDigest)
    && isString(value.datasetId)
    && isString(value.configPath)
    && isString(value.datasetPath)
    && isString(value.artifactRoot)
    && isString(value.submittedAt)
    && isString(value.ownerInstanceId)
    && isNullableString(value.startedAt)
    && isNullableString(value.completedAt)
    && isNullableNumber(value.pid)
    && isNullableString(value.pidStartToken)
    && isNullableNumber(value.exitCode)
    && isBoolean(value.cancellable)
    && isNullableString(value.error)
}

export function isDatasetList(value: unknown): value is DatasetListResponse {
  return isRecord(value)
    && Array.isArray(value.items)
    && value.items.every(isDataset)
    && isNumber(value.total)
    && isNumber(value.invalid)
}

export function isRunList(value: unknown): value is RunListResponse {
  return isRecord(value)
    && Array.isArray(value.items)
    && value.items.every(isRun)
    && isNumber(value.total)
    && isNumber(value.invalid)
}

export function isConfigList(value: unknown): value is ConfigListResponse {
  return isRecord(value)
    && Array.isArray(value.items)
    && value.items.every(isConfig)
    && isNumber(value.total)
    && isNumber(value.invalid)
}

export function isJobList(value: unknown): value is JobListResponse {
  return isRecord(value)
    && Array.isArray(value.items)
    && value.items.every(isJob)
    && isNumber(value.total)
}

export function isJobLog(value: unknown): value is JobLogResponse {
  return isRecord(value)
    && isString(value.jobId)
    && isStrings(value.lines)
    && isBoolean(value.truncated)
}

function isActiveJob(value: unknown): boolean {
  return isRecord(value) && isString(value.id) && isString(value.algorithm) && isString(value.phase) && isString(value.seedProgress) && isNumber(value.progress)
}

function isAlert(value: unknown): boolean {
  return isRecord(value) && (value.level === 'warning' || value.level === 'info') && isString(value.message) && isString(value.age)
}

function isEquity(value: unknown): boolean {
  return isRecord(value) && isString(value.label) && isNumber(value.rl) && isNumber(value.baseline)
}

function isStability(value: unknown): boolean {
  return isRecord(value) && isString(value.label) && isNumber(value.low) && isNumber(value.median) && isNumber(value.high)
}

function isComparisonMetric(value: unknown): boolean {
  return isRecord(value) && isString(value.key) && isString(value.label) && isNullableNumber(value.leftValue) && isNullableNumber(value.rightValue) && isNullableNumber(value.delta) && ['higher', 'lower', 'neutral'].includes(String(value.preference))
}

function isConfigDifference(value: unknown): boolean {
  return isRecord(value) && isString(value.path) && isNullableString(value.left) && isNullableString(value.right)
}

function isFold(value: unknown): boolean {
  return isRecord(value) && isString(value.label) && isNullableNumber(value.leftSelectedReturn) && isNullableNumber(value.leftBaselineReturn) && isNullableNumber(value.rightSelectedReturn) && isNullableNumber(value.rightBaselineReturn)
}

function isWealth(value: unknown): boolean {
  return isRecord(value) && isString(value.label) && isNullableNumber(value.left) && isNullableNumber(value.right) && isNullableNumber(value.leftBaseline) && isNullableNumber(value.rightBaseline)
}

function isEvidenceNode(value: unknown): boolean {
  return isRecord(value) && isString(value.key) && isString(value.label) && ['VERIFIED', 'PRESENT', 'ABSENT', 'INVALID'].includes(String(value.status)) && isBoolean(value.required) && isNullableString(value.digest) && isNullableString(value.path) && isString(value.detail)
}

function isServingCheck(value: unknown): boolean {
  return isRecord(value) && isString(value.key) && isString(value.label) && ['PASS', 'WARN', 'FAIL'].includes(String(value.status)) && isString(value.detail)
}

function isPaperSnapshot(value: unknown): boolean {
  return isRecord(value) && isString(value.recordedAt) && isString(value.bundleDigest) && isString(value.datasetId) && isNumber(value.decisionIndex) && isRecord(value.targetWeights) && Object.values(value.targetWeights).every(isNumber) && isNumber(value.latencyMs) && isString(value.snapshotDigest)
}

export function isStudioOverview(value: unknown): value is StudioOverview {
  if (!isRecord(value) || !isRecord(value.system) || !isRecord(value.assessment)) return false
  return isString(value.system.gpuName)
    && isBoolean(value.system.cudaReady)
    && isString(value.system.pythonVersion)
    && Array.isArray(value.system.metrics)
    && value.system.metrics.every((item) => isRecord(item) && isString(item.label) && isNumber(item.value) && isString(item.detail))
    && (value.latestDataset === null || isDataset(value.latestDataset))
    && Array.isArray(value.activeJobs)
    && value.activeJobs.every(isActiveJob)
    && Array.isArray(value.runs)
    && value.runs.every(isRun)
    && Array.isArray(value.alerts)
    && value.alerts.every(isAlert)
    && Array.isArray(value.equity)
    && value.equity.every(isEquity)
    && Array.isArray(value.stability)
    && value.stability.every(isStability)
    && value.assessment.status === 'NO-GO'
    && isStrings(value.assessment.reasons)
}

export function isRunComparison(value: unknown): value is RunComparison {
  if (!isRecord(value) || !isRecord(value.eligibility)) return false
  return isString(value.leftResourceId)
    && isString(value.rightResourceId)
    && isString(value.leftRunId)
    && isString(value.rightRunId)
    && ['COMPARABLE', 'PARTIALLY_COMPARABLE', 'NOT_COMPARABLE'].includes(String(value.eligibility.status))
    && isStrings(value.eligibility.reasons)
    && isNullableString(value.eligibility.datasetId)
    && Array.isArray(value.metrics)
    && value.metrics.every(isComparisonMetric)
    && Array.isArray(value.configDifferences)
    && value.configDifferences.every(isConfigDifference)
    && Array.isArray(value.folds)
    && value.folds.every(isFold)
    && Array.isArray(value.wealth)
    && value.wealth.every(isWealth)
    && value.productionStatus === 'NO-GO'
}

export function isEvidenceReport(value: unknown): value is EvidenceReport {
  if (!isRecord(value) || !isRecord(value.files)) return false
  return isString(value.runResourceId)
    && isString(value.runId)
    && isString(value.runKind)
    && isValidation(value.status)
    && Array.isArray(value.nodes)
    && value.nodes.every(isEvidenceNode)
    && ['VERIFIED', 'INVALID'].includes(String(value.files.status))
    && isNumber(value.files.declaredCount)
    && isNumber(value.files.verifiedCount)
    && isNumber(value.files.totalSizeBytes)
    && value.productionStatus === 'NO-GO'
    && isNullableString(value.validationError)
}

export function isServingMonitorReport(value: unknown): value is ServingMonitorReport {
  return isRecord(value)
    && ['IDLE', 'VALID', 'INVALID'].includes(String(value.state))
    && Array.isArray(value.checks)
    && value.checks.every(isServingCheck)
    && value.productionStatus === 'NO-GO'
    && isNullableString(value.activeBundleDigest)
    && isNullableString(value.datasetId)
    && isNullableString(value.runKind)
    && isNullableString(value.policyDigest)
    && isNullableString(value.actionSchema)
    && isNullableString(value.observationSchema)
    && isBoolean(value.releaseAttestationPresent)
    && (value.paperSnapshot === null || isPaperSnapshot(value.paperSnapshot))
    && isNullableString(value.validationError)
}
