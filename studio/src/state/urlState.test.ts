import { beforeEach, describe, expect, it } from 'vitest'

import { pushWorkspace, readDashboardSelection, readWorkspace, replaceDashboardSelection } from './urlState'

beforeEach(() => window.history.replaceState(null, '', 'http://localhost/?workspace=dashboard'))

describe('URL state', () => {
  it('reads only supported Dashboard stages', () => {
    expect(readDashboardSelection('?stage=evidence&decision=item-1')).toEqual({ stage: 'evidence', decision: 'item-1' })
    expect(readDashboardSelection('?stage=unknown')).toEqual({ stage: null, decision: null })
  })

  it('replaces committed Dashboard selection without adding navigation state', () => {
    replaceDashboardSelection({ stage: 'data', decision: 'dataset:x:invalid' })
    expect(new URL(window.location.href).searchParams.get('stage')).toBe('data')
    expect(new URL(window.location.href).searchParams.get('decision')).toBe('dataset:x:invalid')
  })

  it('pushes workspace drill-through and clears Dashboard-only state', () => {
    window.history.replaceState(null, '', 'http://localhost/?workspace=dashboard&stage=data&decision=x')
    pushWorkspace('evidence', { evidenceRun: 'run-1' })
    expect(readWorkspace(window.location.search)).toBe('evidence')
    const params = new URLSearchParams(window.location.search)
    expect(params.get('stage')).toBeNull()
    expect(params.get('decision')).toBeNull()
    expect(params.get('evidenceRun')).toBe('run-1')
  })
})
