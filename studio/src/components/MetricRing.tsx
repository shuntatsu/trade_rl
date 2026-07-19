interface MetricRingProps {
  label: string
  value: number
  detail: string
  tone?: 'green' | 'blue'
}

export function MetricRing({ label, value, detail, tone = 'green' }: MetricRingProps) {
  const clamped = Math.max(0, Math.min(100, value))
  const angle = clamped * 3.6
  return (
    <article className="metric-card">
      <span className="metric-card__label">{label}</span>
      <div
        className={`metric-ring metric-ring--${tone}`}
        style={{ '--metric-angle': `${angle}deg` } as React.CSSProperties}
        aria-label={`${label} ${clamped}%`}
      >
        <div>{clamped}<small>%</small></div>
      </div>
      <span className="metric-card__detail">{detail}</span>
    </article>
  )
}
