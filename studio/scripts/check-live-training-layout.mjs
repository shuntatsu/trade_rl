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
if (!cssFile || !jsFile) throw new Error('Build assets are missing; run npm run build first')

const [css, rawJs] = await Promise.all([
  readFile(path.join(assetsDir, cssFile), 'utf8'),
  readFile(path.join(assetsDir, jsFile), 'utf8'),
])
const js = rawJs.replaceAll('</script>', '<\/script>')
const html = `<!doctype html><html lang="ja"><head><base href="http://127.0.0.1:4173/"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}</style></head><body><div id="root"></div><script type="module">${js}</script></body></html>`

const browserCandidates = [
  process.env.CHROMIUM_PATH,
  chromium.executablePath(),
  process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, 'Google', 'Chrome', 'Application', 'chrome.exe'),
  process.env['PROGRAMFILES(X86)'] && path.join(process.env['PROGRAMFILES(X86)'], 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
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
if (!executablePath) throw new Error(`Chromium executable was not found. Checked: ${browserCandidates.join(', ')}`)

const outputDir = process.env.STUDIO_QA_OUTPUT_DIR
  ? path.resolve(studioRoot, '..', process.env.STUDIO_QA_OUTPUT_DIR)
  : '/mnt/data'
await mkdir(outputDir, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath, args: ['--no-sandbox'] })

const job = {
  id: 'job-live',
  schemaVersion: 'studio_job_v2',
  kind: 'training',
  status: 'running',
  runId: 'qa-interactive-chart',
  configResourceId: 'config-qa',
  datasetResourceId: 'dataset-qa',
  configDigest: 'c'.repeat(64),
  datasetId: 'd'.repeat(64),
  configPath: 'configs/qa.json',
  datasetPath: 'datasets/qa',
  artifactRoot: 'research',
  submittedAt: '2026-07-31T08:00:00+00:00',
  ownerInstanceId: 'studio-qa',
  startedAt: '2026-07-31T08:00:01+00:00',
  completedAt: null,
  pid: 42,
  pidStartToken: '1',
  exitCode: null,
  cancellable: true,
  error: null,
}

const telemetry = Array.from({ length: 8 }, (_, index) => {
  const sequence = index + 1
  const hour = 8 + Math.floor(index / 4)
  const minuteInHour = String((index % 4) * 15).padStart(2, '0')
  const before = sequence === 2 ? 0.1 : sequence === 4 ? 0.4 : 0.2
  const after = sequence === 2 ? 0.4 : sequence === 4 ? 0.15 : before
  const close = 67_000 + index * 135 + (index % 2 === 0 ? 45 : -35)
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: `2026-07-31T${String(hour).padStart(2, '0')}:${minuteInHour}:00+00:00`,
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType: sequence === 2 || sequence === 4 ? 'position' : sequence === 6 ? 'risk' : 'rollout',
    marketIndex: 100 + sequence,
    marketTime: `2026-07-31T${String(hour).padStart(2, '0')}:${minuteInHour}:00.123456789`,
    symbol: 'BTCUSDT',
    open: close - 35,
    high: close + 80,
    low: close - 95,
    close,
    action: [after],
    executedTarget: [after - 0.02],
    weightsBefore: [before],
    weightsAfter: [after],
    portfolioValue: 100_000 + sequence * 240,
    baselinePortfolioValue: 100_000 + sequence * 110,
    reward: (sequence - 4) * 0.02,
    drawdown: sequence === 6 ? 0.018 : 0.004 + sequence * 0.0004,
    intervalCost: 1.2 + sequence * 0.15,
    intervalReturn: 0.001 * sequence,
    riskReasons: sequence === 6 ? ['drawdown'] : [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
})

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  await page.route('**/api/studio/**', async (route) => {
    const url = new URL(route.request().url())
    let payload = null
    if (url.pathname.endsWith('/overview')) {
      payload = {
        system: { gpuName: 'QA GPU', cudaReady: false, pythonVersion: '3.12', metrics: [] },
        latestDataset: null,
        activeJobs: [],
        runs: [],
        alerts: [{ level: 'info', message: 'QA fixture', age: 'now' }],
        equity: [],
        stability: [],
        assessment: { status: 'NO-GO', reasons: ['QA fixture'] },
      }
    } else if (url.pathname.endsWith('/jobs')) {
      payload = { items: [job], total: 1 }
    } else if (url.pathname.endsWith('/telemetry/status')) {
      payload = {
        available: true,
        selectedSeed: 7,
        availableSeeds: [7],
        recordCount: telemetry.length,
        lastSequence: telemetry.at(-1).sequence,
        malformedLines: 0,
        sizeBytes: 8192,
        source: 'research/.staging/qa-interactive-chart/seed-7/telemetry/training-telemetry.jsonl',
        streamGeneration: '33333333-3333-4333-8333-333333333333',
      }
    } else if (url.pathname.endsWith('/telemetry/events')) {
      const afterSequence = Number(url.searchParams.get('after_sequence') ?? 0)
      const items = telemetry.filter((record) => record.sequence > afterSequence)
      payload = {
        seed: 7,
        items,
        nextSequence: items.at(-1)?.sequence ?? afterSequence,
        truncated: false,
        malformedLines: 0,
        sequenceGaps: [],
        streamGeneration: '33333333-3333-4333-8333-333333333333',
        resetRequired: false,
      }
    } else if (url.pathname.endsWith('/checkpoint-evaluations')) {
      payload = {
        available: true,
        productionStatus: 'NO-GO',
        items: [{
          fold: 'fold-000',
          configuration: 'residual',
          seed: 7,
          policyDigest: 'a'.repeat(64),
          evaluationDigest: 'b'.repeat(64),
          score: Math.log1p(0.02),
          totalReturn: 0.02,
          finalist: true,
          checkpointRange: [100, 120],
          source: 'research/qa/checkpoint-selection.json',
        }],
      }
    } else if (url.pathname.endsWith('/training-metrics/status')) {
      payload = {
        available: false,
        selectedSeed: null,
        availableSeeds: [],
        availableTags: [],
        lastStep: 0,
        source: null,
        generation: null,
      }
    }
    if (payload === null) return route.abort()
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) })
  })

  await page.setContent(html, { waitUntil: 'networkidle' })
  await page.getByText('最新の実験結果サマリー').waitFor()
  await page.getByRole('button', { name: 'Live Training' }).click()
  await page.getByRole('heading', { name: 'Live Training' }).waitFor()
  await page.getByLabel('Live Training Run').waitFor()
  await page.getByRole('button', { name: /対象を変更/ }).waitFor()
  const chartSurface = page.locator('.synchronized-chart-canvas')
  await chartSurface.waitFor()
  await chartSurface.locator('canvas').first().waitFor()
  await page.locator('.research-replay-scrubber').getByText('Step 256', { exact: true }).waitFor()

  if (await page.locator('.live-connection').count() !== 0) {
    throw new Error('Decorative connection chrome returned to the research workspace')
  }
  if (await page.locator('.research-summary-grid').count() !== 0) {
    throw new Error('Legacy summary cards returned above the synchronized chart')
  }
  const details = page.locator('.synchronized-details')
  if (await details.evaluate((node) => node.hasAttribute('open'))) {
    throw new Error('Selection and evidence details must start collapsed')
  }

  const canvasCount = await chartSurface.locator('canvas').count()
  if (canvasCount < 3) {
    throw new Error(`Synchronized chart did not render all panes: canvasCount=${canvasCount}`)
  }
  if (await page.locator('.synchronized-pane-labels > span').count() !== 3) {
    throw new Error('Synchronized chart must expose exactly three pane labels')
  }

  const replayGeometry = await page.evaluate(() => {
    const toolbar = document.querySelector('.research-replay-toolbar')
    const workspace = document.querySelector('.synchronized-workspace-shell')
    const chart = document.querySelector('.synchronized-chart-canvas')
    if (!(toolbar instanceof HTMLElement) || !(workspace instanceof HTMLElement) || !(chart instanceof HTMLElement)) {
      throw new Error('Synchronized replay workspace nodes are missing')
    }
    const toolbarBox = toolbar.getBoundingClientRect()
    const workspaceBox = workspace.getBoundingClientRect()
    const chartBox = chart.getBoundingClientRect()
    return {
      viewportHeight: window.innerHeight,
      toolbarHeight: toolbarBox.height,
      workspaceHeight: workspaceBox.height,
      workspaceBottom: workspaceBox.bottom,
      chartHeight: chartBox.height,
    }
  })
  if (replayGeometry.toolbarHeight > 120) {
    throw new Error(`Research replay toolbar consumed the workspace: ${JSON.stringify(replayGeometry)}`)
  }
  if (replayGeometry.workspaceHeight < 320 || replayGeometry.chartHeight < 300) {
    throw new Error(`Synchronized chart was compressed below the usable workspace: ${JSON.stringify(replayGeometry)}`)
  }
  if (replayGeometry.workspaceBottom > replayGeometry.viewportHeight + 1) {
    throw new Error(`Synchronized replay workspace overflowed the viewport: ${JSON.stringify(replayGeometry)}`)
  }

  const followLatest = page.getByRole('checkbox', { name: '最新へ追従' })
  if (!(await followLatest.isChecked())) throw new Error('Latest-follow did not start enabled')
  const chartBox = await chartSurface.boundingBox()
  if (!chartBox) throw new Error('Synchronized chart has no measurable bounds')
  await page.mouse.move(chartBox.x + chartBox.width * 0.7, chartBox.y + chartBox.height * 0.35)
  await page.mouse.down()
  await page.mouse.move(chartBox.x + chartBox.width * 0.42, chartBox.y + chartBox.height * 0.35, { steps: 8 })
  await page.mouse.up()
  await page.waitForFunction(() => {
    const label = [...document.querySelectorAll('label')].find((candidate) => candidate.textContent?.includes('最新へ追従'))
    const input = label?.querySelector('input[type="checkbox"]')
    return input instanceof HTMLInputElement && !input.checked
  })

  await page.mouse.click(chartBox.x + chartBox.width * 0.45, chartBox.y + chartBox.height * 0.22)
  await page.getByRole('button', { name: '再生' }).waitFor()
  await page.screenshot({ path: path.join(outputDir, 'trade-rl-studio-live-training-replay.png'), fullPage: false })

  const chartHeightBeforeDetails = await chartSurface.evaluate((node) => node.getBoundingClientRect().height)
  await details.locator('summary').click()
  await page.getByRole('complementary', { name: '選択時点の研究データ' }).waitFor()
  const chartHeightAfterDetails = await chartSurface.evaluate((node) => node.getBoundingClientRect().height)
  if (chartHeightAfterDetails < chartHeightBeforeDetails - 1) {
    throw new Error(`Details disclosure compressed the chart: before=${chartHeightBeforeDetails}, after=${chartHeightAfterDetails}`)
  }
  await page.screenshot({ path: path.join(outputDir, 'trade-rl-studio-live-training-details.png'), fullPage: false })

  await page.getByRole('button', { name: '学習診断' }).click()
  await page.locator('.training-diagnostics').waitFor()
  const geometry = await page.evaluate(() => {
    const selector = document.querySelector('.live-view-selector')
    const diagnostics = document.querySelector('.training-diagnostics')
    if (!(selector instanceof HTMLElement) || !(diagnostics instanceof HTMLElement)) {
      throw new Error('Live Training diagnostics layout nodes are missing')
    }
    const selectorBox = selector.getBoundingClientRect()
    const diagnosticsBox = diagnostics.getBoundingClientRect()
    return {
      viewportHeight: window.innerHeight,
      selectorHeight: selectorBox.height,
      diagnosticsHeight: diagnosticsBox.height,
      diagnosticsBottom: diagnosticsBox.bottom,
    }
  })
  if (geometry.selectorHeight > 80) {
    throw new Error(`Live Training view selector consumed the workspace: ${JSON.stringify(geometry)}`)
  }
  if (geometry.diagnosticsHeight < 300) {
    throw new Error(`Live Training diagnostics was compressed below the usable workspace: ${JSON.stringify(geometry)}`)
  }
  if (geometry.diagnosticsBottom > geometry.viewportHeight + 1) {
    throw new Error(`Live Training diagnostics overflowed the viewport: ${JSON.stringify(geometry)}`)
  }

  await page.screenshot({ path: path.join(outputDir, 'trade-rl-studio-live-training-diagnostics.png'), fullPage: false })
  await page.close()
} finally {
  await browser.close()
}