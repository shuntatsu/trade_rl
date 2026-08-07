import { useRef } from 'react'

import type { DashboardStageKey, DashboardStageView } from './dashboardCockpitModel'

interface DashboardReadinessPipelineProps {
  stages: DashboardStageView[]
  activeStage: DashboardStageKey | null
  committedStage: DashboardStageKey | null
  onPreview: (stage: DashboardStageKey | null) => void
  onSelect: (stage: DashboardStageKey) => void
}

export function DashboardReadinessPipeline({ stages, activeStage, committedStage, onPreview, onSelect }: DashboardReadinessPipelineProps) {
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const moveFocus = (index: number, direction: number) => {
    const next = Math.max(0, Math.min(stages.length - 1, index + direction))
    refs.current[next]?.focus()
  }
  return (
    <section className="dashboard-pipeline-panel" aria-labelledby="readiness-pipeline-title">
      <div className="dashboard-section-header">
        <div><span>RESEARCH DEPENDENCY</span><h2 id="readiness-pipeline-title">Research Readiness Pipeline</h2></div>
        <small>状態は進捗率ではなく、依存関係と検証結果です。</small>
      </div>
      <ol className="dashboard-pipeline" aria-label="研究準備ステージ">
        {stages.map((stage, index) => (
          <li key={stage.key} className={`dashboard-pipeline__item dashboard-pipeline__item--${stage.state.toLowerCase()}`}>
            <button
              ref={(node) => { refs.current[index] = node }}
              type="button"
              className={`dashboard-stage${activeStage === stage.key ? ' dashboard-stage--active' : ''}${committedStage === stage.key ? ' dashboard-stage--committed' : ''}`}
              aria-pressed={committedStage === stage.key}
              onMouseEnter={() => onPreview(stage.key)}
              onMouseLeave={() => onPreview(null)}
              onFocus={() => onPreview(stage.key)}
              onBlur={() => onPreview(null)}
              onClick={() => onSelect(stage.key)}
              onKeyDown={(event) => {
                if (event.key === 'ArrowRight') { event.preventDefault(); moveFocus(index, 1) }
                if (event.key === 'ArrowLeft') { event.preventDefault(); moveFocus(index, -1) }
                if (event.key === 'Home') { event.preventDefault(); refs.current[0]?.focus() }
                if (event.key === 'End') { event.preventDefault(); refs.current.at(-1)?.focus() }
              }}
            >
              <span className="dashboard-stage__index">0{index + 1}</span>
              <span className="dashboard-stage__state">{stage.state}</span>
              <strong>{stage.label}</strong>
              <b>{stage.headline}</b>
              <small>{stage.detail}</small>
              {stage.count !== null ? <i>{stage.count}</i> : null}
            </button>
          </li>
        ))}
      </ol>
    </section>
  )
}
