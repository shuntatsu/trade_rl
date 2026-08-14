import { chromium } from '@playwright/test'
import { statSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { startQaDistServer } from './qa-dist-server.mjs'

const studioRoot = path.resolve(fileURLToPath(new URL('..', import.meta.url)))
const outputDir = process.env.STUDIO_QA_OUTPUT_DIR
  ? path.resolve(studioRoot, '..', process.env.STUDIO_QA_OUTPUT_DIR)
  : '/mnt/data'
await mkdir(outputDir, { recursive: true })

const dataset = {
  id: 'dataset-111111111111111111111111', datasetId: 'd'.repeat(64), name: 'qa-market',
  relativePath: 'datasets/qa-market', market: 'spot', symbols: ['BTCUSDT', 'ETHUSDT'],
  timeframes: ['15m', '1h'], range: '2026-01-01 — 2026-02-01', status: 'VALID',
  featureCount: 226, barCount: 2_976, symbolCount: 2, universeSymbolCount: 15,
  updated: '2026-08-06T00:00:00+00:00', validationError: null,
}
const runs = [
  { id: 'run-111111111111111111111111', runId: 'run-001', manifestDigest: '1'.repeat(64), relativePath: 'research/runs/run-001', runKind: 'research_exploratory', algorithm: 'ppo', datasetId: dataset.datasetId, period: '2026-01-01 — 2026-01-02', createdAt: '2026-01-01T00:00:00Z', completedAt: '2026-01-02T00:00:00Z', fileCount: 8, sharpe: .8, maxDrawdown: .1, totalReturn: .12, productionStatus: 'NO-GO', status: 'VALID', validationError: null },
  { id: 'run-222222222222222222222222', runId: 'run-002', manifestDigest: '2'.repeat(64), relativePath: 'research/runs/run-002', runKind: 'research_selected_final', algorithm: 'sac', datasetId: dataset.datasetId, period: '2026-02-01 — 2026-02-02', createdAt: '2026-02-01T00:00:00Z', completedAt: '2026-02-02T00:00:00Z', fileCount: 10, sharpe: 1.1, maxDrawdown: .08, totalReturn: .18, productionStatus: 'NO-GO', status: 'VALID', validationError: null },
]
const fixtures = {
  overview: {
    system: { gpuName: 'QA GPU', cudaReady: false, pythonVersion: '3.12', metrics: [{ label: 'CPU', value: 30, detail: 'QA' }] },
    latestDataset: dataset,
    activeJobs: [{ id: 'job-qa', algorithm: 'ppo', phase: 'fold 2/6', seedProgress: 'seed 1/3', progress: 40 }],
    runs: runs.toReversed(),
    alerts: [{ id: 'alert:qa', level: 'info', message: 'QA fixture', age: 'now', occurredAt: null }],
    evidence: { runResourceId: runs[1].id, status: 'VERIFIED', requiredCount: 4, verifiedCount: 4, blockerCount: 0, updatedAt: null },
    equity: [], stability: [], assessment: { status: 'NO-GO', reasons: ['QA release blocker'] },
  },
  datasets: { items: [dataset], total: 1, invalid: 0 },
  configs: { items: [], total: 0, invalid: 0 },
  jobs: { items: [], total: 0 },
  runs: { items: runs, total: 2, invalid: 0 },
  comparison: {
    leftResourceId: runs[0].id, rightResourceId: runs[1].id, leftRunId: runs[0].runId, rightRunId: runs[1].runId,
    eligibility: { status: 'COMPARABLE', reasons: [], datasetId: dataset.datasetId }, productionStatus: 'NO-GO',
    metrics: [{ key: 'total_return', label: 'Total return', leftValue: .12, rightValue: .18, delta: .06, preference: 'higher' }],
    configDifferences: [{ path: 'training.algorithm', left: 'ppo', right: 'sac' }],
    folds: [{ label: 'Fold 1', leftSelectedReturn: .05, leftBaselineReturn: .03, rightSelectedReturn: .08, rightBaselineReturn: .03 }],
    wealth: [{ label: '0', left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 }, { label: '1', left: 1.12, right: 1.18, leftBaseline: 1.04, rightBaseline: 1.04 }],
  },
  evidence: {
    runResourceId: runs[0].id, runId: runs[0].runId, runKind: 'research_exploratory', status: 'VALID', productionStatus: 'NO-GO', validationError: null,
    files: { status: 'VERIFIED', declaredCount: 4, verifiedCount: 4, totalSizeBytes: 1_024 },
    nodes: [{ key: 'run_manifest', label: 'Run manifest', status: 'VERIFIED', required: true, digest: 'a'.repeat(64), path: 'run.json', detail: 'manifest and file closure verified' }],
  },
  serving: {
    state: 'IDLE', productionStatus: 'NO-GO', activeBundleDigest: null, datasetId: null, runKind: null,
    policyDigest: null, actionSchema: null, observationSchema: null, releaseAttestationPresent: false,
    checks: [], paperSnapshot: null, validationError: null,
  },
}

const browserCandidates = [
  process.env.CHROMIUM_PATH,
  chromium.executablePath(),
  process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, 'Google', 'Chrome', 'Application', 'chrome.exe'),
  process.env['PROGRAMFILES(X86)'] && path.join(process.env['PROGRAMFILES(X86)'], 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
  '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
].filter(Boolean)
const executablePath = browserCandidates.find((candidate) => {
  try { return statSync(candidate).isFile() } catch { return false }
})
if (!executablePath) throw new Error(`Chromium executable was not found. Checked: ${browserCandidates.join(', ')}`)

const distServer = await startQaDistServer(studioRoot)
const browser = await chromium.launch({ headless: true, executablePath, args: ['--no-sandbox'] })
try {
  const viewports = [
    { width: 1536, height: 1024, name: '1536' },
    { width: 1440, height: 900, name: '1440' },
    { width: 1180, height: 800, name: '1180' },
  ]
  for (const viewport of viewports) {
    const page = await browser.newPage({ viewport })
    await page.route('**/api/studio/**', async (route) => {
      const url = new URL(route.request().url())
      let payload = null
      if (url.pathname.endsWith('/overview')) payload = fixtures.overview
      else if (url.pathname.endsWith('/datasets')) payload = fixtures.datasets
      else if (url.pathname.endsWith('/configs')) payload = fixtures.configs
      else if (url.pathname.endsWith('/jobs')) payload = fixtures.jobs
      else if (url.pathname.endsWith('/runs')) payload = fixtures.runs
      else if (url.pathname.endsWith('/compare')) payload = fixtures.comparison
      else if (url.pathname.endsWith('/evidence')) payload = fixtures.evidence
      else if (url.pathname.endsWith('/serving')) payload = fixtures.serving
      if (payload === null) return route.abort()
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) })
    })
    await page.goto(`${distServer.origin}/?workspace=dashboard`, { waitUntil: 'networkidle' })
    await page.getByRole('heading', { name: 'Research Readiness Pipeline' }).waitFor()
    await page.getByRole('heading', { name: 'Action Queue' }).waitFor()
    if (await page.locator('.dashboard-stage').count() !== 5) throw new Error('Dashboard must render exactly five readiness stages')

    const dimensions = await page.evaluate(() => ({
      viewportHeight: window.innerHeight,
      htmlScrollHeight: document.documentElement.scrollHeight,
      bodyScrollHeight: document.body.scrollHeight,
      rootHeight: document.getElementById('root')?.getBoundingClientRect().height ?? 0,
      cockpitHeight: document.querySelector('.dashboard-cockpit')?.getBoundingClientRect().height ?? 0,
    }))
    if (dimensions.htmlScrollHeight !== dimensions.viewportHeight || dimensions.bodyScrollHeight !== dimensions.viewportHeight || dimensions.rootHeight !== dimensions.viewportHeight || dimensions.cockpitHeight <= 0) {
      throw new Error(`Dashboard overflow at ${viewport.width}x${viewport.height}: ${JSON.stringify(dimensions)}`)
    }
    await page.screenshot({ path: path.join(outputDir, `trade-rl-studio-dashboard-${viewport.name}.png`), fullPage: false })

    for (const workspace of ['Data Lab', '実験', 'Run Center', '比較', 'Evidence Explorer', 'Serving Monitor']) {
      await page.getByRole('button', { name: new RegExp(workspace, 'i') }).click()
      await page.getByRole('heading', { name: workspace }).waitFor()
      const overflow = await page.evaluate(() => ({ html: document.documentElement.scrollHeight, body: document.body.scrollHeight, viewport: window.innerHeight }))
      if (overflow.html !== overflow.viewport || overflow.body !== overflow.viewport) throw new Error(`${workspace} introduced page overflow at ${viewport.width}x${viewport.height}`)
    }
    await page.close()
  }
} finally {
  await browser.close()
  await distServer.close()
}
