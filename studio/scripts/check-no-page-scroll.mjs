import { chromium } from '@playwright/test'
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
const html = `<!doctype html><html lang="ja"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}</style></head><body><div id="root"></div><script type="module">${js}</script></body></html>`

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  args: ['--no-sandbox'],
})

try {
  const viewports = [
    { width: 1536, height: 1024, screenshot: '/mnt/data/trade-rl-studio-dashboard.png' },
    { width: 1440, height: 900, screenshot: '/mnt/data/trade-rl-studio-dashboard-1440.png' },
  ]

  for (const viewport of viewports) {
    const page = await browser.newPage({ viewport })
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
    for (const workspace of ['Data Lab', '実験', 'Run Center']) {
      await page.getByRole('button', { name: new RegExp(workspace, 'i') }).click()
      await page.getByRole('heading', { name: workspace }).waitFor()
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
