import { FlaskConical, Play, ShieldAlert } from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { ConfigSummary, DatasetSummary, JobSummary } from '../data/types'

interface ExperimentsPageProps { api?: StudioApi }
const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/

function defaultRunId(): string {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
  return `studio-${stamp}`
}

export function ExperimentsPage({ api = studioApi }: ExperimentsPageProps) {
  const [configs, setConfigs] = useState<ConfigSummary[]>([])
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [configResourceId, setConfigResourceId] = useState('')
  const [datasetResourceId, setDatasetResourceId] = useState('')
  const [runId, setRunId] = useState(defaultRunId)
  const [job, setJob] = useState<JobSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const [configResponse, datasetResponse] = await Promise.all([api.loadConfigs(), api.loadDatasets()])
        if (cancelled) return
        const validConfigs = configResponse.items.filter((item) => item.status === 'VALID' && item.configDigest)
        const validDatasets = datasetResponse.items.filter((item) => item.status === 'VALID')
        setConfigs(validConfigs)
        setDatasets(validDatasets)
        setConfigResourceId(validConfigs[0]?.id ?? '')
        setDatasetResourceId(validDatasets[0]?.id ?? '')
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : '実験設定を取得できませんでした。')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => { cancelled = true }
  }, [api])

  const selectedConfig = useMemo(() => configs.find((item) => item.id === configResourceId) ?? null, [configs, configResourceId])
  const selectedDataset = useMemo(() => datasets.find((item) => item.id === datasetResourceId) ?? null, [datasets, datasetResourceId])
  const valid = Boolean(configResourceId && datasetResourceId && RUN_ID.test(runId))

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!valid) return
    setSubmitting(true)
    setError(null)
    try { setJob(await api.submitTrainingJob({ configResourceId, datasetResourceId, runId })) }
    catch (reason) { setError(reason instanceof Error ? reason.message : '学習ジョブを開始できませんでした。') }
    finally { setSubmitting(false) }
  }

  return (
    <section className="runtime-page" aria-labelledby="experiments-title">
      <header className="runtime-toolbar">
        <div><span className="runtime-eyebrow">EXPLORATORY TRAINING</span><h1 id="experiments-title">実験</h1><p>検証済みidentityを選び、既存workflowへ渡します。</p></div>
        <span className="runtime-danger"><ShieldAlert size={14} aria-hidden="true" />NO-GO</span>
      </header>
      <div className="runtime-split runtime-split--experiment">
        <form className="runtime-pane runtime-form" onSubmit={(event) => void submit(event)}>
          <div className="runtime-pane__header"><strong>学習ジョブ</strong><span>local process</span></div>
          <div className="runtime-form__body">
            <label><span>Training config</span><select value={configResourceId} onChange={(event) => setConfigResourceId(event.target.value)} disabled={loading || configs.length === 0}>{configs.length === 0 ? <option value="">利用可能な設定なし</option> : null}{configs.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.algorithm.toUpperCase()}</option>)}</select></label>
            <label><span>Dataset</span><select value={datasetResourceId} onChange={(event) => setDatasetResourceId(event.target.value)} disabled={loading || datasets.length === 0}>{datasets.length === 0 ? <option value="">利用可能なデータなし</option> : null}{datasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.symbolCount} symbols</option>)}</select></label>
            <label><span>Run ID</span><input aria-label="Run ID" value={runId} onChange={(event) => setRunId(event.target.value)} aria-invalid={!RUN_ID.test(runId)} /><small>英数字で開始し、英数字・`.`・`_`・`-`のみ使用できます。</small></label>
            {error ? <div className="runtime-error">{error}</div> : null}
            <button type="submit" className="runtime-button runtime-button--primary" disabled={!valid || loading || submitting}><Play size={15} aria-hidden="true" />{submitting ? '開始中…' : '学習ジョブを開始'}</button>
          </div>
        </form>
        <div className="runtime-pane runtime-pane--detail">
          <div className="runtime-pane__header"><strong>実行プレビュー</strong><FlaskConical size={15} aria-hidden="true" /></div>
          <div className="runtime-experiment-preview">
            <article><span>Algorithm</span><strong>{selectedConfig?.algorithm.toUpperCase() ?? '—'}</strong></article>
            <article><span>Config</span><code>{selectedConfig?.relativePath ?? '—'}</code><small>{selectedConfig?.configDigest ?? '—'}</small></article>
            <article><span>Dataset</span><strong>{selectedDataset?.name ?? '—'}</strong><small>{selectedDataset?.datasetId ?? '—'}</small></article>
            <article><span>Feature contract</span><strong>{selectedDataset ? `${selectedDataset.featureCount} features` : '—'}</strong></article>
            <article className="runtime-notice"><ShieldAlert size={16} aria-hidden="true" /><p>研究用exploratory runのみを開始します。ライブ注文・本番許可・利益保証は行いません。</p></article>
            {job ? <article className="runtime-job-result" aria-live="polite"><span>受理済み</span><strong>{job.id}</strong><small>{job.runId} · {job.status}</small></article> : null}
          </div>
        </div>
      </div>
    </section>
  )
}
