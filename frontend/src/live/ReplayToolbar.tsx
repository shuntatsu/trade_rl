import {
  ChevronsLeft,
  ChevronsRight,
  ChevronLeft,
  ChevronRight,
  Layers3,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
} from 'lucide-react'
import { useEffect, useState } from 'react'

import type { JobSummary } from '../data/types'
import type { ResearchChartLayers } from './researchChartModel'

export type ReplaySpeed = 1 | 4 | 8

export interface ReplaySourceSelection {
  seed: number | null
  environmentId: number | null
}

export interface ReplayToolbarProps {
  jobs: JobSummary[]
  jobId: string | null
  seeds: number[]
  seed: number | null
  environments: number[]
  environmentId: number | null
  playing: boolean
  speed: ReplaySpeed
  followLatest: boolean
  layers: ResearchChartLayers
  hasRecords: boolean
  onJobChange: (jobId: string | null) => void
  onSourceChange: (selection: ReplaySourceSelection) => void
  onTogglePlaying: () => void
  onFirst: () => void
  onPreviousEvent: () => void
  onNextEvent: () => void
  onLast: () => void
  onSpeedChange: (speed: ReplaySpeed) => void
  onFollowLatestChange: (followLatest: boolean) => void
  onLayersChange: (layers: ResearchChartLayers) => void
  onResetView: () => void
}

interface LayerOption {
  key: keyof ResearchChartLayers
  label: string
}

const LAYER_OPTIONS: LayerOption[] = [
  { key: 'positionEvents', label: '売買イベント' },
  { key: 'riskEvents', label: 'Riskイベント' },
  { key: 'baseline', label: 'Baseline' },
  { key: 'executedWeight', label: 'Executed Weight' },
  { key: 'rewardCost', label: 'Reward / Cost' },
  { key: 'drawdown', label: 'Drawdown' },
]

function sourceSummary(seed: number | null, environmentId: number | null): string {
  const seedLabel = seed === null ? 'Seed未選択' : `Seed ${seed}`
  const environmentLabel = environmentId === null ? 'Env未選択' : `Env ${environmentId}`
  return `${seedLabel} · ${environmentLabel}`
}

export function ReplayToolbar({
  jobs,
  jobId,
  seeds,
  seed,
  environments,
  environmentId,
  playing,
  speed,
  followLatest,
  layers,
  hasRecords,
  onJobChange,
  onSourceChange,
  onTogglePlaying,
  onFirst,
  onPreviousEvent,
  onNextEvent,
  onLast,
  onSpeedChange,
  onFollowLatestChange,
  onLayersChange,
  onResetView,
}: ReplayToolbarProps) {
  const [sourceOpen, setSourceOpen] = useState(false)
  const [layersOpen, setLayersOpen] = useState(false)
  const [draftSeed, setDraftSeed] = useState<number | null>(seed)
  const [draftEnvironmentId, setDraftEnvironmentId] = useState<number | null>(environmentId)

  useEffect(() => {
    if (!sourceOpen) {
      setDraftSeed(seed)
      setDraftEnvironmentId(environmentId)
    }
  }, [environmentId, seed, sourceOpen])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setSourceOpen(false)
      setLayersOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const toggleSource = () => {
    setDraftSeed(seed)
    setDraftEnvironmentId(environmentId)
    setSourceOpen((open) => !open)
    setLayersOpen(false)
  }

  const toggleLayers = () => {
    setLayersOpen((open) => !open)
    setSourceOpen(false)
  }

  return (
    <div className="research-replay-toolbar" aria-label="研究リプレイ操作">
      <div className="research-replay-toolbar__source">
        <label className="research-toolbar-field">
          <span>Run</span>
          <select
            aria-label="Live Training Run"
            value={jobId ?? ''}
            onChange={(event) => onJobChange(event.target.value || null)}
          >
            {jobs.length === 0 ? <option value="">実行ジョブなし</option> : null}
            {jobs.map((job) => (
              <option key={job.id} value={job.id}>{job.runId} · {job.status}</option>
            ))}
          </select>
        </label>

        <div className="research-toolbar-popover-anchor">
          <button
            type="button"
            className="research-toolbar-button"
            aria-expanded={sourceOpen}
            aria-haspopup="dialog"
            onClick={toggleSource}
          >
            <SlidersHorizontal size={15} aria-hidden="true" />
            対象を変更
            <span>{sourceSummary(seed, environmentId)}</span>
          </button>

          {sourceOpen ? (
            <div className="research-toolbar-popover research-toolbar-popover--source" role="dialog" aria-label="リプレイ対象を変更">
              <label className="research-toolbar-field">
                <span>Seed</span>
                <select
                  aria-label="Live Training Seed"
                  value={draftSeed ?? ''}
                  onChange={(event) => setDraftSeed(event.target.value === '' ? null : Number(event.target.value))}
                >
                  {seeds.length === 0 ? <option value="">Seed待機中</option> : null}
                  {seeds.map((value) => <option key={value} value={value}>Seed {value}</option>)}
                </select>
              </label>
              <label className="research-toolbar-field">
                <span>Environment</span>
                <select
                  aria-label="Live Training Environment"
                  value={draftEnvironmentId ?? ''}
                  onChange={(event) => setDraftEnvironmentId(event.target.value === '' ? null : Number(event.target.value))}
                >
                  {environments.length === 0 ? <option value="">Env待機中</option> : null}
                  {environments.map((value) => <option key={value} value={value}>Env {value}</option>)}
                </select>
              </label>
              <button
                type="button"
                className="research-toolbar-button research-toolbar-button--primary"
                onClick={() => {
                  onSourceChange({ seed: draftSeed, environmentId: draftEnvironmentId })
                  setSourceOpen(false)
                }}
              >
                対象を適用
              </button>
            </div>
          ) : null}
        </div>
      </div>

      <div className="research-replay-toolbar__transport" aria-label="リプレイ再生操作">
        <button type="button" aria-label="先頭へ" disabled={!hasRecords} onClick={onFirst}>
          <ChevronsLeft size={17} aria-hidden="true" />
        </button>
        <button type="button" aria-label="前のイベント" disabled={!hasRecords} onClick={onPreviousEvent}>
          <ChevronLeft size={17} aria-hidden="true" />
        </button>
        <button
          type="button"
          className="research-transport-primary"
          aria-label={playing ? '一時停止' : '再生'}
          disabled={!hasRecords}
          onClick={onTogglePlaying}
        >
          {playing ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
        </button>
        <button type="button" aria-label="次のイベント" disabled={!hasRecords} onClick={onNextEvent}>
          <ChevronRight size={17} aria-hidden="true" />
        </button>
        <button type="button" aria-label="最新へ" disabled={!hasRecords} onClick={onLast}>
          <ChevronsRight size={17} aria-hidden="true" />
        </button>
        <label className="research-toolbar-speed">
          <span>速度</span>
          <select
            aria-label="再生速度"
            value={speed}
            onChange={(event) => onSpeedChange(Number(event.target.value) as ReplaySpeed)}
          >
            <option value={1}>1×</option>
            <option value={4}>4×</option>
            <option value={8}>8×</option>
          </select>
        </label>
      </div>

      <div className="research-replay-toolbar__view">
        <label className="research-follow-toggle">
          <input
            type="checkbox"
            checked={followLatest}
            onChange={(event) => onFollowLatestChange(event.target.checked)}
          />
          <span>最新へ追従</span>
        </label>

        <div className="research-toolbar-popover-anchor">
          <button
            type="button"
            className="research-toolbar-button"
            aria-expanded={layersOpen}
            aria-haspopup="dialog"
            onClick={toggleLayers}
          >
            <Layers3 size={15} aria-hidden="true" />
            表示項目
          </button>
          {layersOpen ? (
            <div className="research-toolbar-popover research-toolbar-popover--layers" role="dialog" aria-label="チャート表示項目">
              {LAYER_OPTIONS.map(({ key, label }) => (
                <label key={key} className="research-layer-option">
                  <input
                    type="checkbox"
                    checked={layers[key]}
                    onChange={(event) => onLayersChange({ ...layers, [key]: event.target.checked })}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          ) : null}
        </div>

        <button type="button" className="research-toolbar-button" onClick={onResetView}>
          <RotateCcw size={15} aria-hidden="true" />
          表示リセット
        </button>
      </div>
    </div>
  )
}
