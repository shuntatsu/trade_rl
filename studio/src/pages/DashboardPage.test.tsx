import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { DashboardPage } from './DashboardPage'

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
})
