import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { demoOverview } from './data/demoOverview'
import { App } from './App'

const initialOverview = { source: 'demo' as const, overview: demoOverview, error: null }

describe('App', () => {
  it('renders the fixed shell and research status', () => {
    render(<App initialOverview={initialOverview} />)

    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('navigation')).toBeInTheDocument()
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByText('DEMO DATA')).toBeInTheDocument()
    expect(screen.getAllByText('NO-GO').length).toBeGreaterThan(0)
  })

  it('switches workspaces without a document navigation', async () => {
    const user = userEvent.setup()
    render(<App initialOverview={initialOverview} />)

    await user.click(screen.getByRole('button', { name: /Data Lab/i }))

    expect(screen.getByRole('heading', { name: 'Data Lab' })).toBeInTheDocument()
    expect(screen.queryByText('最新の実験結果サマリー')).not.toBeInTheDocument()
  })
})
