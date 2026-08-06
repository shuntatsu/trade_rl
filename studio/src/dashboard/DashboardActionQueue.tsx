import { ArrowUpRight, ListChecks } from 'lucide-react'
import { useRef } from 'react'

import type { DashboardDecisionItem, DashboardStageKey } from './dashboardCockpitModel'

interface DashboardActionQueueProps {
  decisions: DashboardDecisionItem[]
  activeDecisionId: string | null
  committedDecisionId: string | null
  activeStage: DashboardStageKey | null
  onPreview: (id: string | null) => void
  onSelect: (decision: DashboardDecisionItem) => void
  onAction: (decision: DashboardDecisionItem) => void
}

export function DashboardActionQueue({ decisions, activeDecisionId, committedDecisionId, activeStage, onPreview, onSelect, onAction }: DashboardActionQueueProps) {
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const moveFocus = (index: number, direction: number) => {
    const next = Math.max(0, Math.min(decisions.length - 1, index + direction))
    refs.current[next]?.focus()
  }
  return (
    <section className="dashboard-queue-panel" aria-labelledby="dashboard-queue-title">
      <header className="dashboard-section-header dashboard-section-header--compact">
        <div><span>RANKED</span><h2 id="dashboard-queue-title">Action Queue</h2></div>
        <strong>{decisions.length}</strong>
      </header>
      {decisions.length === 0 ? <div className="dashboard-queue-empty"><ListChecks size={24} aria-hidden="true" />判断項目はありません。</div> : null}
      <ol className="dashboard-queue">
        {decisions.map((decision, index) => {
          const related = activeStage !== null && decision.stage === activeStage
          return (
            <li key={decision.id} className={`dashboard-queue__item${related ? ' dashboard-queue__item--related' : ''}${activeDecisionId === decision.id ? ' dashboard-queue__item--active' : ''}${committedDecisionId === decision.id ? ' dashboard-queue__item--committed' : ''}`}>
              <button
                ref={(node) => { refs.current[index] = node }}
                type="button"
                className="dashboard-queue__select"
                aria-pressed={committedDecisionId === decision.id}
                onMouseEnter={() => onPreview(decision.id)}
                onMouseLeave={() => onPreview(null)}
                onFocus={() => onPreview(decision.id)}
                onBlur={() => onPreview(null)}
                onClick={() => onSelect(decision)}
                onKeyDown={(event) => {
                  if (event.key === 'ArrowDown') { event.preventDefault(); moveFocus(index, 1) }
                  if (event.key === 'ArrowUp') { event.preventDefault(); moveFocus(index, -1) }
                  if (event.key === 'Home') { event.preventDefault(); refs.current[0]?.focus() }
                  if (event.key === 'End') { event.preventDefault(); refs.current.at(-1)?.focus() }
                }}
              >
                <span className={`dashboard-severity dashboard-severity--${decision.severity}`}>{decision.severity}</span>
                <strong>{decision.title}</strong>
                <small>{decision.explanation}</small>
                <i>{decision.stage.toUpperCase()}{decision.age ? ` · ${decision.age}` : ''}</i>
              </button>
              {decision.action ? <button type="button" className="dashboard-queue__action" aria-label={`${decision.title}: ${decision.action.label}`} onClick={() => onAction(decision)}><ArrowUpRight size={15} aria-hidden="true" /></button> : null}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
