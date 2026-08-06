import { Settings2, ShieldAlert } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { DashboardActionQueue } from '../dashboard/DashboardActionQueue'
import { DashboardDecisionRibbon } from '../dashboard/DashboardDecisionRibbon'
import { DashboardEnvironmentSheet } from '../dashboard/DashboardEnvironmentSheet'
import { DashboardLatestResultStrip } from '../dashboard/DashboardLatestResultStrip'
import { DashboardReadinessPipeline } from '../dashboard/DashboardReadinessPipeline'
import {
  buildDashboardCockpitModel,
  type DashboardDecisionItem,
  type DashboardStageKey,
  type DashboardWorkspace,
} from '../dashboard/dashboardCockpitModel'
import type { StudioOverview } from '../data/types'
import { readDashboardSelection, replaceDashboardSelection } from '../state/urlState'

export type DashboardFreshness = 'LIVE' | 'STALE' | 'OFFLINE' | 'DEMO'

interface DashboardPageProps {
  overview: StudioOverview
  freshness?: DashboardFreshness
  sourceError?: string | null
  onNavigate?: (workspace: DashboardWorkspace, params: Record<string, string>) => void
}

function editableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName))
}

export function DashboardPage({ overview, freshness = 'LIVE', sourceError = null, onNavigate = () => undefined }: DashboardPageProps) {
  const model = useMemo(() => buildDashboardCockpitModel(overview), [overview])
  const initialSelection = useMemo(() => readDashboardSelection(window.location.search), [])
  const [committedStage, setCommittedStage] = useState<DashboardStageKey | null>(initialSelection.stage)
  const [committedDecisionId, setCommittedDecisionId] = useState<string | null>(initialSelection.decision)
  const [previewStage, setPreviewStage] = useState<DashboardStageKey | null>(null)
  const [previewDecisionId, setPreviewDecisionId] = useState<string | null>(null)
  const [environmentOpen, setEnvironmentOpen] = useState(false)
  const environmentTrigger = useRef<HTMLButtonElement | null>(null)

  const committedDecision = model.decisions.find((item) => item.id === committedDecisionId) ?? null
  const previewDecision = model.decisions.find((item) => item.id === previewDecisionId) ?? null
  const primaryDecision = model.decisions.find((item) => item.id === model.primaryDecisionId) ?? null
  const displayedDecision = previewDecision ?? committedDecision ?? primaryDecision
  const activeStage = previewStage
    ?? (previewDecision?.stage !== 'alert' ? previewDecision?.stage : null)
    ?? committedStage
    ?? (committedDecision?.stage !== 'alert' ? committedDecision?.stage : null)
    ?? null

  const commitSelection = useCallback((stage: DashboardStageKey | null, decision: string | null) => {
    setCommittedStage(stage)
    setCommittedDecisionId(decision)
    replaceDashboardSelection({ stage, decision })
  }, [])

  useEffect(() => {
    const stageExists = committedStage === null || model.stages.some((item) => item.key === committedStage)
    const decisionExists = committedDecisionId === null || model.decisions.some((item) => item.id === committedDecisionId)
    if (!stageExists || !decisionExists) commitSelection(stageExists ? committedStage : null, decisionExists ? committedDecisionId : null)
  }, [commitSelection, committedDecisionId, committedStage, model.decisions, model.stages])

  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (editableTarget(event.target)) return
      if ((event.key === 'e' || event.key === 'E') && !environmentOpen) {
        event.preventDefault()
        setEnvironmentOpen(true)
      } else if (event.key === 'Escape' && !environmentOpen && (committedStage !== null || committedDecisionId !== null)) {
        event.preventDefault()
        commitSelection(null, null)
      }
    }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [commitSelection, committedDecisionId, committedStage, environmentOpen])

  const selectStage = (stage: DashboardStageKey) => {
    const retained = committedDecision?.stage === stage ? committedDecision.id : null
    commitSelection(stage, retained)
  }
  const selectDecision = (decision: DashboardDecisionItem) => {
    commitSelection(decision.stage === 'alert' ? null : decision.stage, decision.id)
  }
  const executeAction = (decision: DashboardDecisionItem) => {
    if (decision.action) onNavigate(decision.action.workspace, decision.action.params)
  }
  const closeEnvironment = useCallback(() => {
    setEnvironmentOpen(false)
    window.setTimeout(() => environmentTrigger.current?.focus(), 0)
  }, [])

  return (
    <section className="dashboard-cockpit" aria-labelledby="dashboard-cockpit-title">
      <header className="dashboard-cockpit__header">
        <div><span className="dashboard-eyebrow">RESEARCH DECISION COCKPIT</span><h1 id="dashboard-cockpit-title">次に直すべき一点を特定する</h1><p>DataからReleaseまで、検証済みの契約だけで安全な次の操作を決めます。</p></div>
        <div className="dashboard-cockpit__status">
          <span className={`dashboard-freshness dashboard-freshness--${freshness.toLowerCase()}`} title={sourceError ?? undefined}>{freshness}</span>
          <span className="dashboard-no-go"><ShieldAlert size={15} aria-hidden="true" />NO-GO</span>
          <button ref={environmentTrigger} type="button" className="dashboard-environment-trigger" onClick={() => setEnvironmentOpen(true)}><Settings2 size={15} aria-hidden="true" />Environment <kbd>E</kbd></button>
        </div>
      </header>

      <DashboardDecisionRibbon decision={displayedDecision} onAction={executeAction} />

      <div className="dashboard-cockpit__main">
        <DashboardReadinessPipeline
          stages={model.stages}
          activeStage={activeStage}
          committedStage={committedStage}
          onPreview={setPreviewStage}
          onSelect={selectStage}
        />
        <DashboardActionQueue
          decisions={model.decisions}
          activeDecisionId={previewDecisionId ?? committedDecisionId}
          committedDecisionId={committedDecisionId}
          activeStage={activeStage}
          onPreview={setPreviewDecisionId}
          onSelect={selectDecision}
          onAction={executeAction}
        />
      </div>

      <DashboardLatestResultStrip result={model.latestResult} />
      <DashboardEnvironmentSheet open={environmentOpen} environment={model.environment} onClose={closeEnvironment} />
    </section>
  )
}
