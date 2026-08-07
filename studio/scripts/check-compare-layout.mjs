import { chromium } from '@playwright/test'
import { statSync } from 'node:fs'
import { mkdir, readFile, readdir } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const studioRoot = path.resolve(fileURLToPath(new URL('..', import.meta.url)))
const assetsDir = path.join(studioRoot, 'dist', 'assets')
const assets = await readdir(assetsDir)
const cssFile = assets.find((name) => name.endsWith('.css'))
const jsFile = assets.find((name) => name.endsWith('.js'))
if (!cssFile || !jsFile) {
  throw new Error('Build assets are missing; run npm run build first')
}

const [css, rawJs] = await Promise.all([
  readFile(path.join(assetsDir, cssFile), 'utf8'),
  readFile(path.join(assetsDir, jsFile), 'utf8'),
])
const js = rawJs.replaceAll('</script>', '<\\/script>')
const html = `<!doctype html><html lang="ja"><head><base href="http://127.0.0.1:4173/"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}</style></head><body><div id="root"></div><script type="module">${js}</script></body></html>`
const outputDir = process.env.STUDIO_QA_OUTPUT_DIR
  ? path.resolve(studioRoot, '..', process.env.STUDIO_QA_OUTPUT_DIR)
  : '/mnt/data'
await mkdir(outputDir, { recursive: true })

const dataset = {
  id: 'dataset-111111111111111111111111',
  datasetId: 'd'.repeat(64),
  name: 'qa-market',
  relativePath: 'datasets/qa-market',
  market: 'spot',
  symbols: ['BTCUSDT', 'ETHUSDT'],
  timeframes: ['15m', '1h'],
  range: '2026-01-01 — 2026-02-01',
  status: 'VALID',
  featureCount: 226,
  barCount: 2_976,
  symbolCount: 2,
  universeSymbolCount: 15,
  updated: '2026-08-06T00:00:00+00:00',
  validationError: null,
}
const runs = [
  {
    id: 'run-111111111111111111111111',
    runId: 'run-left',
    manifestDigest: '1'.repeat(64),
    relativePath: 'research/runs/run-left',
    runKind: 'research_exploratory',
    algorithm: 'ppo',
    datasetId: dataset.datasetId,
    period: '2026-01-01 — 2026-01-02',
    createdAt: '2026-01-01T00:00:00Z',
    completedAt: '2026-01-02T00:00:00Z',
    fileCount: 8,
    sharpe: 0.8,
    maxDrawdown: 0.1,
    totalReturn: 0.12,
    productionStatus: 'NO-GO',
    status: 'VALID',
    validationError: null,
  },
  {
    id: 'run-222222222222222222222222',
    runId: 'run-right',
    manifestDigest: '2'.repeat(64),
    relativePath: 'research/runs/run-right',
    runKind: 'research_selected_final',
    algorithm: 'sac',
    datasetId: dataset.datasetId,
    period: '2026-02-01 — 2026-02-02',
    createdAt: '2026-02-01T00:00:00Z',
    completedAt: '2026-02-02T00:00:00Z',
    fileCount: 10,
    sharpe: 1.1,
    maxDrawdown: 0.08,
    totalReturn: 0.18,
    productionStatus: 'NO-GO',
    status: 'VALID',
    validationError: null,
  },
]
const comparison = {
  leftResourceId: runs[0].id,
  rightResourceId: runs[1].id,
  leftRunId: runs[0].runId,
  rightRunId: runs[1].runId,
  eligibility: {
    status: 'COMPARABLE',
    reasons: ['dataset and sealed evaluation ranges align'],
    datasetId: dataset.datasetId,
  },
  productionStatus: 'NO-GO',
  metrics: [
    {
      key: 'total_return',
      label: 'Total return',
      leftValue: 0.12,
      rightValue: 0.18,
      delta: 0.06,
      preference: 'higher',
    },
    {
      key: 'total_cost',
      label: 'Total cost',
      leftValue: 0.006,
      rightValue: 0.009,
      delta: 0.003,
      preference: 'lower',
    },
  ],
  configDifferences: [
    { path: 'training.algorithm', left: 'ppo', right: 'sac' },
    { path: 'execution.fee_rate', left: '0.001', right: '0.002' },
  ],
  folds: [
    {
      label: 'Fold 1',
      leftSelectedReturn: 0.05,
      leftBaselineReturn: 0.03,
      rightSelectedReturn: 0.08,
      rightBaselineReturn: 0.03,
    },
    {
      label: 'Fold 2',
      leftSelectedReturn: 0.04,
      leftBaselineReturn: 0.02,
      rightSelectedReturn: 0.07,
      rightBaselineReturn: 0.02,
    },
  ],
  wealth: [
    { label: 'start', foldIndex: null, left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 },
    { label: '10', foldIndex: 0, left: 1.01, right: 1.015, leftBaseline: 1.004, rightBaseline: 1.004 },
    { label: '11', foldIndex: 0, left: 1.025, right: 1.04, leftBaseline: 1.008, rightBaseline: 1.008 },
    { label: '12', foldIndex: 0, left: 1.05, right: 1.08, leftBaseline: 1.015, rightBaseline: 1.015 },
    { label: '20', foldIndex: 1, left: 1.045, right: 1.09, leftBaseline: 1.018, rightBaseline: 1.018 },
    { label: '21', foldIndex: 1, left: 1.07, right: 1.13, leftBaseline: 1.025, rightBaseline: 1.025 },
    { label: '22', foldIndex: 1, left: 1.1, right: 1.17, leftBaseline: 1.033, rightBaseline: 1.033 },
    { label: '23', foldIndex: 1, left: 1.12, right: 1.2, leftBaseline: 1.04, rightBaseline: 1.04 },
  ],
}
const overview = {
  system: {
    gpuName: 'QA GPU',
    cudaReady: false,
    pythonVersion: '3.12',
    metrics: [],
  },
  latestDataset: dataset,
  activeJobs: [],
  runs: runs.toReversed(),
  alerts: [
    {
      id: 'alert:qa',
      level: 'info',
      message: 'QA fixture',
      age: 'now',
      occurredAt: null,
    },
  ],
  evidence: {
    runResourceId: runs[1].id,
    status: 'VERIFIED',
    requiredCount: 4,
    verifiedCount: 4,
    blockerCount: 0,
    updatedAt: null,
  },
  equity: [],
  stability: [],
  assessment: {
    status: 'NO-GO',
    reasons: ['QA release blocker'],
  },
}

const browserCandidates = [
  process.env.CHROMIUM_PATH,
  chromium.executablePath(),
  process.env.PROGRAMFILES
    && path.join(process.env.PROGRAMFILES, 'Google', 'Chrome', 'Application', 'chrome.exe'),
  process.env['PROGRAMFILES(X86)']
    && path.join(
      process.env['PROGRAMFILES(X86)'],
      'Microsoft',
      'Edge',
      'Application',
      'msedge.exe',
    ),
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
].filter(Boolean)
const executablePath = browserCandidates.find((candidate) => {
  try {
    return statSync(candidate).isFile()
  } catch {
    return false
  }
})
if (!executablePath) {
  throw new Error(`Chromium executable was not found. Checked: ${browserCandidates.join(', ')}`)
}

const browser = await chromium.launch({
  headless: true,
  executablePath,
  args: ['--no-sandbox'],
})
try {
  for (const viewport of [
    { width: 1536, height: 1024, name: '1536' },
    { width: 1440, height: 900, name: '1440' },
    { width: 1180, height: 800, name: '1180' },
  ]) {
    const page = await browser.newPage({ viewport })
    await page.route('**/api/studio/**', async (route) => {
      const url = new URL(route.request().url())
      let payload = null
      if (url.pathname.endsWith('/overview')) payload = overview
      else if (url.pathname.endsWith('/runs')) {
        payload = { items: runs, total: runs.length, invalid: 0 }
      } else if (url.pathname.endsWith('/compare')) payload = comparison
      if (payload === null) return route.abort()
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(payload),
      })
    })

    await page.setContent(html, { waitUntil: 'networkidle' })
    await page.getByRole('button', { name: '比較' }).click()
    await page.getByRole('heading', { name: '比較' }).waitFor()
    const surface = page.getByRole('application', { name: 'Run comparison chart' })
    await surface.waitFor()
    if (await page.locator('[data-pane="wealth"]').count() !== 1) {
      throw new Error('Compare must render exactly one wealth pane')
    }
    if (await page.locator('[data-pane="delta"]').count() !== 1) {
      throw new Error('Compare must render exactly one delta pane')
    }

    const initialBox = await surface.boundingBox()
    if (!initialBox || initialBox.height < 340) {
      throw new Error(`Compare workspace is too small at ${viewport.width}x${viewport.height}`)
    }

    await page.mouse.click(
      initialBox.x + initialBox.width * 0.55,
      initialBox.y + initialBox.height * 0.35,
    )
    await page.waitForFunction(() => new URLSearchParams(location.search).has('comparePoint'))

    await page.getByRole('button', { name: 'Range' }).click()
    await page.mouse.move(
      initialBox.x + initialBox.width * 0.3,
      initialBox.y + initialBox.height * 0.3,
    )
    await page.mouse.down()
    await page.mouse.move(
      initialBox.x + initialBox.width * 0.72,
      initialBox.y + initialBox.height * 0.3,
      { steps: 8 },
    )
    await page.mouse.up()
    await page.locator('.comparison-range-readout').waitFor()
    await page.waitForFunction(() => {
      const params = new URLSearchParams(location.search)
      return params.has('compareStart') && params.has('compareEnd')
    })

    const rangeBeforeZoom = await surface.getAttribute('data-visible-range')
    await surface.hover({ position: { x: initialBox.width * 0.5, y: initialBox.height * 0.3 } })
    await page.mouse.wheel(0, -500)
    await page.waitForFunction(
      (before) => document.querySelector('[aria-label="Run comparison chart"]')?.getAttribute('data-visible-range') !== before,
      rangeBeforeZoom,
    )
    await page.getByRole('button', { name: 'Reset view' }).click()
    if (await surface.getAttribute('data-visible-range') !== '0:7') {
      throw new Error('Compare reset did not restore the full ordinal range')
    }

    const heightBeforeInspector = (await surface.boundingBox())?.height ?? 0
    await page.getByRole('button', { name: 'Details' }).click()
    await page.getByRole('dialog', { name: 'Comparison inspector' }).waitFor()
    const heightAfterInspector = (await surface.boundingBox())?.height ?? 0
    if (Math.abs(heightAfterInspector - heightBeforeInspector) > 1) {
      throw new Error(
        `Comparison inspector changed chart height: before=${heightBeforeInspector}, after=${heightAfterInspector}`,
      )
    }

    const overflow = await page.evaluate(() => ({
      viewport: window.innerHeight,
      html: document.documentElement.scrollHeight,
      body: document.body.scrollHeight,
    }))
    if (overflow.html !== overflow.viewport || overflow.body !== overflow.viewport) {
      throw new Error(
        `Compare overflow at ${viewport.width}x${viewport.height}: ${JSON.stringify(overflow)}`,
      )
    }

    await page.screenshot({
      path: path.join(outputDir, `trade-rl-studio-compare-${viewport.name}.png`),
      fullPage: false,
    })
    await page.close()
  }
} finally {
  await browser.close()
}
