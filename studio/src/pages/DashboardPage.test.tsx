import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { DashboardPage } from './DashboardPage'

function finalPathY(path: SVGPathElement): number {
  const tokens = path.getAttribute('d')?.trim().split(/\s+/) ?? []
  const value = Number(tokens.at(-1))
  if (!Number.isFinite(value)) throw new Error('chart path has no finite final y coordinate')
  return value
}

describe('DashboardPage', () => {
  it('renders the complete research overview', () => {
    render(<DashboardPage overview={demoOverview} />)

    expect(screen.getByText('システム概要')).toBeInTheDocument()
    expect(screen.getByText('最新データセット')).toBeInTheDocument()
    expect(screen.getByText('実行中のジョブ')).toBeInTheDocument()
    expect(screen.getByText('最新の実験結果サマリー')).toBeInTheDocument()
    expect(screen.getByText('直近のアラート')).toBeInTheDocument()
    expect(screen.getByText('ベースライン比較')).toBeInTheDocument()
    expect(screen.getByText('ウォークフォワード安定性')).toBeInTheDocument()
    expect(screen.getByText('Production Status')).toBeInTheDocument()
    expect(screen.getByText('binance_spot_multi_tf_v1')).toBeInTheDocument()
  })

  it('projects RL and baseline wealth through one shared y-axis domain', () => {
    const { container } = render(
      <DashboardPage
        overview={{
          ...demoOverview,
          equity: [
            { label: 'start', rl: 1, baseline: 1 },
            { label: 'end', rl: 3, baseline: 1.2 },
          ],
        }}
      />,
    )

    const rlPath = container.querySelector<SVGPathElement>('.chart-line--rl')
    const baselinePath = container.querySelector<SVGPathElement>('.chart-line--baseline')
    if (rlPath === null || baselinePath === null) throw new Error('equity paths were not rendered')

    expect(finalPathY(rlPath)).toBeCloseTo(18, 1)
    expect(finalPathY(baselinePath)).toBeCloseTo(111.3, 1)
  })
})
