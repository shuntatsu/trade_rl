import { render, screen, waitFor } from '@testing-library/react'
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
    expect(screen.getByText('ReleaseはNO-GOです')).toBeInTheDocument()
  })

  it('commits a stage with click and persists it in the URL', async () => {
    const user = userEvent.setup()
    render(<DashboardPage overview={demoOverview} />)
    await user.click(screen.getByRole('button', { name: /Data/i }))
    expect(new URL(window.location.href).searchParams.get('stage')).toBe('data')
  })

  it('opens Environment with E and restores trigger focus after closing', async () => {
    const user = userEvent.setup()
    render(<DashboardPage overview={demoOverview} />)
    await user.keyboard('E')
    expect(screen.getByRole('dialog', { name: 'Environment' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Environmentを閉じる' }))
    await waitFor(() => expect(screen.getByRole('button', { name: /Environment/ })).toHaveFocus())
  })

  it('drills through using the action contract', async () => {
    const user = userEvent.setup()
    const onNavigate = vi.fn()
    render(<DashboardPage overview={demoOverview} onNavigate={onNavigate} />)
    await user.click(screen.getAllByRole('button', { name: /Evidenceから確認/ })[0])
    expect(onNavigate).toHaveBeenCalledWith('evidence', { evidenceRun: demoOverview.evidence.runResourceId })
  })
})
