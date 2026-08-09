import type { StudioOverviewResult } from '../data/types'

export interface StudioOverviewPollingState extends StudioOverviewResult {
  lastSuccessfulResponseAt: number | null
}

export function initialPollingState(
  initial: StudioOverviewResult,
  receivedAt: number,
): StudioOverviewPollingState {
  return {
    ...initial,
    lastSuccessfulResponseAt: initial.source === 'live' ? receivedAt : null,
  }
}

export function applyPollingResult(
  previous: StudioOverviewPollingState,
  result: StudioOverviewResult,
  receivedAt: number,
): StudioOverviewPollingState {
  if (result.source === 'live') {
    return { ...result, lastSuccessfulResponseAt: receivedAt }
  }
  if (result.source === 'demo') {
    return { ...result, lastSuccessfulResponseAt: previous.lastSuccessfulResponseAt }
  }
  if (previous.lastSuccessfulResponseAt !== null) {
    return {
      source: 'stale',
      overview: previous.overview,
      error: result.error ?? 'Studio API refresh failed',
      lastSuccessfulResponseAt: previous.lastSuccessfulResponseAt,
    }
  }
  return { ...result, source: 'offline', lastSuccessfulResponseAt: null }
}

export function applyPollingFailure(
  previous: StudioOverviewPollingState,
  message: string,
): StudioOverviewPollingState {
  return previous.lastSuccessfulResponseAt === null
    ? { ...previous, source: 'offline', error: message }
    : { ...previous, source: 'stale', error: message }
}
