import { ArrowRight, CircleAlert } from 'lucide-react'

import type { DashboardDecisionItem } from './dashboardCockpitModel'

interface DashboardDecisionRibbonProps {
  decision: DashboardDecisionItem | null
  onAction: (decision: DashboardDecisionItem) => void
}

export function DashboardDecisionRibbon({ decision, onAction }: DashboardDecisionRibbonProps) {
  if (decision === null) {
    return (
      <section className="dashboard-decision dashboard-decision--empty" aria-label="次の安全な操作">
        <CircleAlert size={20} aria-hidden="true" />
        <div><span>NEXT SAFE ACTION</span><strong>判断可能な操作がありません</strong><p>DataとEvidenceの状態を取得してください。</p></div>
      </section>
    )
  }
  return (
    <section className={`dashboard-decision dashboard-decision--${decision.severity}`} aria-label="次の安全な操作" aria-live="polite">
      <div className="dashboard-decision__copy">
        <span>NEXT SAFE ACTION · {decision.stage.toUpperCase()}</span>
        <strong>{decision.title}</strong>
        <p>{decision.explanation}</p>
      </div>
      {decision.action ? (
        <button type="button" className="dashboard-primary-action" onClick={() => onAction(decision)}>
          {decision.action.label}<ArrowRight size={15} aria-hidden="true" />
        </button>
      ) : <span className="dashboard-decision__readonly">READ ONLY</span>}
    </section>
  )
}
