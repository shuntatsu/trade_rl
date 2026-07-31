import { chromium } from '@playwright/test'
import { statSync } from 'node:fs'
import { mkdir, readFile, readdir } from 'node:fs/promises'
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
const js = rawJs.replaceAll('</script>', '<\/script>')
const html = `<!doctype html><html lang="ja"><head><base href="http://127.0.0.1:4173/"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}</style></head><body><div id="root"></div><script type="module">${js}</script></body></html>`

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

const outputDir = process.env.STUDIO_QA_OUTPUT_DIR
  ? path.resolve(studioRoot, '..', process.env.STUDIO_QA_OUTPUT_DIR)
  : '/mnt/data'
await mkdir(outputDir, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath, args: ['--no-sandbox'] })

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
      payload = { items: [], total: 0 }
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
  if (await page.getByText('LIVE', { exact: true }).count() !== 0) {
    throw new Error('Decorative LIVE chrome returned to the research workspace')
  }

  const replayGeometry = await page.evaluate(() => {
    const toolbar = document.querySelector('.research-replay-toolbar')
    const workspace = document.querySelector('.research-workspace-grid')
    if (!(toolbar instanceof HTMLElement) || !(workspace instanceof HTMLElement)) {
      throw new Error('Research replay workspace nodes are missing')
    }
    const toolbarBox = toolbar.getBoundingClientRect()
    const workspaceBox = workspace.getBoundingClientRect()
    return {
      viewportHeight: window.innerHeight,
      toolbarHeight: toolbarBox.height,
      workspaceHeight: workspaceBox.height,
      workspaceBottom: workspaceBox.bottom,
    }
  })
  if (replayGeometry.toolbarHeight > 120) {
    throw new Error(`Research replay toolbar consumed the workspace: ${JSON.stringify(replayGeometry)}`)
  }
  if (replayGeometry.workspaceHeight < 320) {
    throw new Error(`Research replay chart was compressed below the usable workspace: ${JSON.stringify(replayGeometry)}`)
  }
  if (replayGeometry.workspaceBottom > replayGeometry.viewportHeight + 1) {
    throw new Error(`Research replay workspace overflowed the viewport: ${JSON.stringify(replayGeometry)}`)
  }
  await page.screenshot({ path: path.join(outputDir, 'trade-rl-studio-live-training-replay.png'), fullPage: false })

  await page.getByRole('button', { name: '学習診断' }).click()
  await page.getByText('Runを選択してください。').waitFor()

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
