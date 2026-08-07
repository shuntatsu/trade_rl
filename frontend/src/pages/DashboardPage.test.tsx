import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { DashboardPage } from './DashboardPage'

describe('DashboardPage', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/?workspace=dashboard')
  })

  it('renders the decision cockpit instead of equal-weight overview panels', () => {
    render(<DashboardPage overview={demoOverview} freshness="DEMO" />)
    expect(screen.getByRole('heading', { name: 'Research Readiness Pipeline' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Action Queue' })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { pressed: false }).length).toBeGreaterThanOrEqual(5)
    expect(screen.queryByText('システム概要')).not.toBeInTheDocument()
    expect(within(screen.getByRole('region', { name: '次の安全な操作' })).getByText('ReleaseはNO-GOです')).toBeInTheDocument()
  })

  it('commits a stage with click and persists it in the URL', async () => {
    const user = userEvent.setup()
    render(<DashboardPage overview={demoOverview} />)
    await user.click(screen.getByRole('button', { name: /Data/i }))
    expect(new URL(window.location.href).searchParams.get('stage')).toBe('data')
  })

  it('restores the committed selection when Dashboard URL state changes through popstate', async () => {
    render(<DashboardPage overview={demoOverview} />)
    const pipeline = screen.getByRole('list', { name: '研究準備ステージ' })
    const evidenceStage = within(pipeline).getByRole('button', { name: /Evidence/i })

    act(() => {
      window.history.pushState(null, '', '/?workspace=dashboard&stage=evidence')
      window.dispatchEvent(new PopStateEvent('popstate'))
    })

    await waitFor(() => expect(evidenceStage).toHaveAttribute('aria-pressed', 'true'))
  })

  it('opens Environment with E and restores trigger focus after closing', async () => {
    const user = userEvent.setup()
    render(<DashboardPage overview={demoOverview} />)
    await user.keyboard('E')
    expect(screen.getByRole('dialog', { name: 'Environment' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Environmentを閉じる' }))
    await waitFor(() => expect(screen.getByRole('button', { name: /Environment/ })).toHaveFocus())
  })

  it('does not intercept modified browser or operating-system shortcuts', () => {
    render(<DashboardPage overview={demoOverview} />)
    fireEvent.keyDown(window, { key: 'e', ctrlKey: true })
    expect(screen.queryByRole('dialog', { name: 'Environment' })).not.toBeInTheDocument()
  })

  it('drills through using the action contract', async () => {
    const user = userEvent.setup()
    const onNavigate = vi.fn()
    render(<DashboardPage overview={demoOverview} onNavigate={onNavigate} />)
    await user.click(screen.getAllByRole('button', { name: /Evidenceから確認/ })[0])
    expect(onNavigate).toHaveBeenCalledWith('evidence', { evidenceRun: demoOverview.evidence.runResourceId })
  })
})
