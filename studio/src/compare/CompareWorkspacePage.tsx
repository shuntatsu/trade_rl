import { GitCompareArrows, PanelRightOpen, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { RunComparison } from '../data/types'
import { ComparisonInspectorSheet } from './ComparisonInspectorSheet'
import { InteractiveComparisonWorkspace } from './InteractiveComparisonWorkspace'
import {
  buildComparisonWorkspaceModel,
  summarizeComparisonRange,
} from './comparisonWorkspaceModel'
import {
  useRunComparisonWorkspace,
  type CompareApi,
} from './useRunComparisonWorkspace'

interface CompareWorkspacePageProps {
  api?: CompareApi
}

function eligibilityClass(status: RunComparison['eligibility']['status']): string {
  return status.toLowerCase().replaceAll('_', '-')
}

export function CompareWorkspacePage({ api }: CompareWorkspacePageProps) {
  const workspace = useRunComparisonWorkspace(api)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const inspectorTrigger = useRef<HTMLButtonElement | null>(null)
  const model = useMemo(
    () => workspace.comparison
      ? buildComparisonWorkspaceModel(workspace.comparison)
      : null,
    [workspace.comparison],
  )
  const rangeSummary = useMemo(
    () => model && workspace.committedRange
      ? summarizeComparisonRange(
          model,
          workspace.committedRange.start,
          workspace.committedRange.end,
        )
      : null,
    [model, workspace.committedRange],
  )

  const {
    committedPoint,
    committedRange,
    clearSelection,
  } = workspace

  useEffect(() => {
    if (!model) return
    const maximum = Math.max(0, model.points.length - 1)
    const pointValid = committedPoint === null || committedPoint <= maximum
    const rangeValid = committedRange === null
      || (committedRange.start <= maximum && committedRange.end <= maximum)
    if (!pointValid || !rangeValid) clearSelection()
  }, [clearSelection, committedPoint, committedRange, model])

  const closeInspector = useCallback(() => {
    setInspectorOpen(false)
    window.setTimeout(() => inspectorTrigger.current?.focus(), 0)
  }, [])

  const verdictCounts = useMemo(() => {
    const counts = { improved: 0, worse: 0, tie: 0, unavailable: 0 }
    for (const item of model?.metricVerdicts ?? []) counts[item.verdict] += 1
    return counts
  }, [model])

  return (
    <section
      className="runtime-page compare-workspace-page"
      aria-labelledby="compare-title"
    >
      <header className="runtime-toolbar compare-workspace-header">
        <div>
          <span className="runtime-eyebrow">IMMUTABLE RUN REVIEW</span>
          <h1 id="compare-title">比較</h1>
          <p>
            sealed evaluation index上で、右runが左runとの差をどこで広げたかを確認します。日時は推測しません。
          </p>
        </div>
        <div className="runtime-toolbar__actions">
          <span className="runtime-danger">NO-GO</span>
          <button
            ref={inspectorTrigger}
            type="button"
            className="runtime-button runtime-button--quiet"
            onClick={() => setInspectorOpen(true)}
            disabled={!workspace.comparison}
          >
            <PanelRightOpen size={14} aria-hidden="true" />Details
          </button>
          <button
            type="button"
            className="runtime-button runtime-button--quiet"
            onClick={() => void workspace.refresh()}
            disabled={workspace.loading}
          >
            <RefreshCw size={14} aria-hidden="true" />再読込
          </button>
        </div>
      </header>

      <section className="compare-control-bar" aria-label="Run pair controls">
        <label>
          Left run
          <select
            aria-label="Left run"
            value={workspace.leftRunId}
            onChange={(event) => {
              const value = event.target.value
              workspace.setLeftRunId(value)
              void workspace.loadComparison(value, workspace.rightRunId)
            }}
          >
            {workspace.runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.runId} · {run.algorithm}
              </option>
            ))}
          </select>
        </label>
        <GitCompareArrows size={18} aria-hidden="true" />
        <label>
          Right run
          <select
            aria-label="Right run"
            value={workspace.rightRunId}
            onChange={(event) => {
              const value = event.target.value
              workspace.setRightRunId(value)
              void workspace.loadComparison(workspace.leftRunId, value)
            }}
          >
            {workspace.runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.runId} · {run.algorithm}
              </option>
            ))}
          </select>
        </label>
        {workspace.comparison ? (
          <div
            className={`compare-eligibility compare-eligibility--${eligibilityClass(workspace.comparison.eligibility.status)}`}
            aria-label="comparison eligibility"
          >
            <strong>{workspace.comparison.eligibility.status}</strong>
            <span>
              {workspace.comparison.leftRunId} ↔ {workspace.comparison.rightRunId}
            </span>
            <small>
              {workspace.comparison.eligibility.reasons.join(' · ')
                || 'dataset and sealed evaluation identities align'}
            </small>
          </div>
        ) : null}
      </section>

      <section
        className="compare-verdict-ribbon"
        aria-label="preference-aware metric summary"
      >
        <div>
          <span>RIGHT VS LEFT</span>
          <strong>
            {verdictCounts.improved} improved · {verdictCounts.worse} worse · {verdictCounts.tie} tie
          </strong>
        </div>
        <p>Preference-aware metric verdicts · no automatic winner</p>
        {rangeSummary ? (
          <div>
            <span>SELECTED RANGE</span>
            <strong>
              {rangeSummary.startLabel} → {rangeSummary.endLabel} · relative{' '}
              {rangeSummary.relativeReturn === null
                ? '—'
                : `${rangeSummary.relativeReturn >= 0 ? '+' : ''}${(rangeSummary.relativeReturn * 100).toFixed(2)}%`}
            </strong>
          </div>
        ) : null}
      </section>

      <div className="compare-workspace-frame" aria-busy={workspace.loading}>
        {workspace.loading && !workspace.comparison && !workspace.error ? (
          <div className="runtime-empty" role="status">比較を読み込み中です…</div>
        ) : null}
        {workspace.error ? <div className="runtime-error">{workspace.error}</div> : null}
        {!workspace.error && !workspace.comparison && !workspace.loading ? (
          <div className="runtime-empty">比較可能なRun pairがありません。</div>
        ) : null}
        {workspace.comparison?.eligibility.status === 'NOT_COMPARABLE' ? (
          <div className="runtime-error">
            このRun pairは比較できません。{workspace.comparison.eligibility.reasons.join(' · ')}
          </div>
        ) : null}
        {model && workspace.comparison?.eligibility.status !== 'NOT_COMPARABLE' ? (
          <InteractiveComparisonWorkspace
            model={model}
            committedPoint={workspace.committedPoint}
            committedRange={workspace.committedRange}
            onCommitPoint={workspace.setCommittedPoint}
            onCommitRange={workspace.setCommittedRange}
          />
        ) : null}
        {workspace.comparison && model ? (
          <ComparisonInspectorSheet
            open={inspectorOpen}
            comparison={workspace.comparison}
            model={model}
            selectedPoint={workspace.committedPoint}
            rangeSummary={rangeSummary}
            onClose={closeInspector}
          />
        ) : null}
      </div>
    </section>
  )
}
