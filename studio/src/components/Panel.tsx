import type { ReactNode } from 'react'

interface PanelProps {
  title: string
  index: number
  accent?: 'cyan' | 'green' | 'blue' | 'purple' | 'amber' | 'red'
  className?: string
  actions?: ReactNode
  children: ReactNode
}

export function Panel({ title, index, accent = 'cyan', className = '', actions, children }: PanelProps) {
  return (
    <section className={`panel panel--${accent} ${className}`} aria-label={title}>
      <div className="panel__header">
        <div className="panel__title"><span>{index}</span><h2>{title}</h2></div>
        {actions ? <div className="panel__actions">{actions}</div> : null}
      </div>
      <div className="panel__body">{children}</div>
    </section>
  )
}
