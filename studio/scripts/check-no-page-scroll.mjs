import { chromium } from '@playwright/test'
import { statSync } from 'node:fs'
import { readFile, readdir } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'

const studioRoot = path.resolve(new URL('..', import.meta.url).pathname)
const assetsDir = path.join(studioRoot, 'dist', 'assets')
const assets = await readdir(assetsDir)
const cssFile = assets.find((name) => name.endsWith('.css'))
const jsFile = assets.find((name) => name.endsWith('.js'))
if (!cssFile || !jsFile) throw new Error('Build assets are missing; run npm run build first')

const [css, rawJs] = await Promise.all([
  readFile(path.join(assetsDir, cssFile), 'utf8'),
  readFile(path.join(assetsDir, jsFile), 'utf8'),
])
const js = rawJs.replaceAll('</script>', '<\\/script>')
const html = `<!doctype html><html lang="ja"><head><base href="http://127.0.0.1:4173/"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}</style></head><body><div id="root"></div><script type="module">${js}</script></body></html>`

const auditFixtures = {
  runs: {
    items: [
      { id: 'run-001', relativePath: 'research/runs/run-001', runKind: 'research_exploratory', algorithm: 'ppo', datasetId: 'dataset-1', period: '2026-01-01 — 2026-01-02', createdAt: '2026-01-01', completedAt: '2026-01-02', fileCount: 8, sharpe: 0.8, maxDrawdown: 0.1, totalReturn: 0.12, productionStatus: 'NO-GO', status: 'VALID', validationError: null },
      { id: 'run-002', relativePath: 'research/runs/run-002', runKind: 'research_selected_final', algorithm: 'sac', datasetId: 'dataset-1', period: '2026-02-01 — 2026-02-02', createdAt: '2026-02-01', completedAt: '2026-02-02', fileCount: 10, sharpe: 1.1, maxDrawdown: 0.08, totalReturn: 0.18, productionStatus: 'NO-GO', status: 'VALID', validationError: null },
    ], total: 2, invalid: 0,
  },
  comparison: {
    leftRunId: 'run-001', rightRunId: 'run-002', productionStatus: 'NO-GO',
    metrics: [
      { key: 'total_return', label: 'Total return', leftValue: .12, rightValue: .18, delta: .06, preference: 'higher' },
      { key: 'sharpe', label: 'Sharpe', leftValue: .8, rightValue: 1.1, delta: .3, preference: 'higher' },
      { key: 'max_drawdown', label: 'Max drawdown', leftValue: .1, rightValue: .08, delta: -.02, preference: 'lower' },
      { key: 'total_cost', label: 'Total cost', leftValue: .006, rightValue: .009, delta: .003, preference: 'lower' },
    ],
    configDifferences: [
      { path: 'training.algorithm', left: 'ppo', right: 'sac' },
      { path: 'execution.fee_rate', left: '0.001', right: '0.002' },
      { path: 'training.sequence_encoder', left: 'false', right: 'true' },
    ],
    folds: [
      { label: 'Fold 1', leftSelectedReturn: .05, leftBaselineReturn: .03, rightSelectedReturn: .08, rightBaselineReturn: .03 },
      { label: 'Fold 2', leftSelectedReturn: .02, leftBaselineReturn: .025, rightSelectedReturn: .06, rightBaselineReturn: .025 },
    ],
    wealth: [
      { label: '0', left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 },
      { label: '1', left: 1.02, right: 1.03, leftBaseline: 1.01, rightBaseline: 1.01 },
      { label: '2', left: 1.05, right: 1.09, leftBaseline: 1.025, rightBaseline: 1.025 },
      { label: '3', left: 1.12, right: 1.18, leftBaseline: 1.04, rightBaseline: 1.04 },
    ],
  },
  evidence: {
    runId: 'run-001', runKind: 'research_selected_final', status: 'VALID', productionStatus: 'NO-GO', validationError: null,
    files: { status: 'VERIFIED', declaredCount: 10, verifiedCount: 10, totalSizeBytes: 162430 },
    nodes: [
      { key: 'run_manifest', label: 'Run manifest', status: 'VERIFIED', required: true, digest: 'a'.repeat(64), path: 'run.json', detail: 'manifest and file closure verified' },
      { key: 'dataset_reference', label: 'Dataset reference', status: 'VERIFIED', required: true, digest: 'b'.repeat(64), path: 'dataset-reference.json', detail: 'declared file closure verified' },
      { key: 'configuration', label: 'Research configuration', status: 'VERIFIED', required: true, digest: 'c'.repeat(64), path: 'training-config.json', detail: 'declared file closure verified' },
      { key: 'policy_ensemble', label: 'Policy ensemble', status: 'VERIFIED', required: true, digest: 'd'.repeat(64), path: 'ensemble.json', detail: 'declared file closure verified' },
      { key: 'selection_authorization', label: 'Selection authorization', status: 'VERIFIED', required: true, digest: 'e'.repeat(64), path: 'selection-authorization.json', detail: 'declared file closure verified' },
      { key: 'confirmation_evidence', label: 'Confirmation evidence', status: 'ABSENT', required: false, digest: null, path: null, detail: 'optional artifact is absent' },
    ],
  },
  serving: {
    state: 'VALID', productionStatus: 'NO-GO', activeBundleDigest: 'f'.repeat(64), datasetId: 'dataset-1', runKind: 'research_selected_final', policyDigest: 'd'.repeat(64), actionSchema: 'target_weights_v1', observationSchema: 'sequence_observation_v4', releaseAttestationPresent: true, validationError: null,
    checks: [
      { key: 'pointer', label: 'Active pointer', status: 'PASS', detail: 'pointer schema and path are valid' },
      { key: 'closure', label: 'Bundle closure', status: 'PASS', detail: '9 declared files verified' },
      { key: 'identity', label: 'Bundle identity', status: 'PASS', detail: 'dataset, action, observation, and policy identities are bound' },
      { key: 'paper_snapshot', label: 'Paper inference snapshot', status: 'PASS', detail: 'latest snapshot identity is valid' },
    ],
    paperSnapshot: { recordedAt: '2026-07-20T00:00:00Z', bundleDigest: 'f'.repeat(64), datasetId: 'dataset-1', decisionIndex: 42, targetWeights: { BTCUSDT: .4, ETHUSDT: -.2, BNBUSDT: .1, CASH: .7 }, latencyMs: 8.4, snapshotDigest: '9'.repeat(64) },
  },
}

const browserCandidates = [
  process.env.CHROMIUM_PATH,
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
  const viewports = [
    { width: 1536, height: 1024, screenshot: '/mnt/data/trade-rl-studio-dashboard.png' },
    { width: 1440, height: 900, screenshot: '/mnt/data/trade-rl-studio-dashboard-1440.png' },
  ]

  for (const viewport of viewports) {
    const page = await browser.newPage({ viewport })
    await page.route('**/api/studio/**', async (route) => {
      const url = new URL(route.request().url())
      let payload = null
      if (url.pathname.endsWith('/runs')) payload = auditFixtures.runs
      else if (url.pathname.endsWith('/compare')) payload = auditFixtures.comparison
      else if (url.pathname.endsWith('/evidence')) payload = auditFixtures.evidence
      else if (url.pathname.endsWith('/serving')) payload = auditFixtures.serving
      if (payload === null) return route.abort()
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) })
    })
    await page.setContent(html, { waitUntil: 'networkidle' })
    await page.getByText('最新の実験結果サマリー').waitFor()

    const dimensions = await page.evaluate(() => ({
      viewportHeight: window.innerHeight,
      htmlScrollHeight: document.documentElement.scrollHeight,
      bodyScrollHeight: document.body.scrollHeight,
      rootHeight: document.getElementById('root')?.getBoundingClientRect().height ?? 0,
    }))

    if (
      dimensions.htmlScrollHeight !== dimensions.viewportHeight ||
      dimensions.bodyScrollHeight !== dimensions.viewportHeight ||
      dimensions.rootHeight !== dimensions.viewportHeight
    ) {
      throw new Error(`page overflow at ${viewport.width}x${viewport.height}: ${JSON.stringify(dimensions)}`)
    }

    await page.screenshot({ path: viewport.screenshot, fullPage: false })
    for (const workspace of ['Data Lab', '実験', 'Run Center', '比較', 'Evidence Explorer', 'Serving Monitor']) {
      await page.getByRole('button', { name: new RegExp(workspace, 'i') }).click()
      await page.getByRole('heading', { name: workspace }).waitFor()
      if (workspace === '比較') await page.screenshot({ path: '/mnt/data/trade-rl-studio-compare.png', fullPage: false })
      if (workspace === 'Evidence Explorer') await page.screenshot({ path: '/mnt/data/trade-rl-studio-evidence.png', fullPage: false })
      if (workspace === 'Serving Monitor') await page.screenshot({ path: '/mnt/data/trade-rl-studio-serving.png', fullPage: false })
      const afterNavigation = await page.evaluate(() => ({
        htmlScrollHeight: document.documentElement.scrollHeight,
        bodyScrollHeight: document.body.scrollHeight,
        viewportHeight: window.innerHeight,
      }))
      if (
        afterNavigation.htmlScrollHeight !== afterNavigation.viewportHeight ||
        afterNavigation.bodyScrollHeight !== afterNavigation.viewportHeight
      ) {
        throw new Error(`${workspace} introduced overflow at ${viewport.width}x${viewport.height}`)
      }
    }
    await page.close()
  }
} finally {
  await browser.close()
}
