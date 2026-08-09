import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'

import { demoOverview } from './data/demoOverview'
import { App } from './App'

const initialOverview = { source: 'demo' as const, overview: demoOverview, error: null }

beforeEach(() => window.history.replaceState(null, '', '/?workspace=dashboard'))

describe('App', () => {
  it('renders the fixed shell and Dashboard decision cockpit', () => {
    render(<App initialOverview={initialOverview} />)
    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByText('DEMO DATA')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Research Readiness Pipeline' })).toBeInTheDocument()
    expect(screen.getAllByText('NO-GO').length).toBeGreaterThan(0)
  })

  it('switches workspaces through history-backed navigation', async () => {
    const user = userEvent.setup()
    render(<App initialOverview={initialOverview} />)
    await user.click(screen.getByRole('button', { name: /Data Lab/i }))
    expect(screen.getByRole('heading', { name: 'Data Lab' })).toBeInTheDocument()
    expect(new URL(window.location.href).searchParams.get('workspace')).toBe('data')
  })
})
