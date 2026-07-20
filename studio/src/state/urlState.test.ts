import { describe, expect, it } from 'vitest'

import { readWorkspace, replaceParams } from './urlState'

describe('urlState', () => {
  it('restores a valid workspace and rejects unknown values', () => {
    expect(readWorkspace('?workspace=compare')).toBe('compare')
    expect(readWorkspace('?workspace=unknown')).toBe('dashboard')
  })

  it('updates search parameters without discarding existing state', () => {
    window.history.replaceState(null, '', '/?workspace=compare&left=run-a')
    replaceParams({ right: 'run-b', left: null })
    expect(window.location.search).toContain('workspace=compare')
    expect(window.location.search).toContain('right=run-b')
    expect(window.location.search).not.toContain('left=')
  })
})
