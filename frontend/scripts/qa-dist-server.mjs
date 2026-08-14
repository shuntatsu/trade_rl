import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'

const CONTENT_TYPES = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.webp', 'image/webp'],
])

function containedPath(root, requestPath) {
  const relative = decodeURIComponent(requestPath).replace(/^\/+/, '') || 'index.html'
  const resolved = path.resolve(root, relative)
  const prefix = `${root}${path.sep}`
  if (resolved !== root && !resolved.startsWith(prefix)) {
    throw new Error(`QA asset path escapes dist root: ${requestPath}`)
  }
  return resolved
}

export async function startQaDistServer(studioRoot) {
  const distRoot = path.resolve(studioRoot, 'dist')
  const indexPath = path.join(distRoot, 'index.html')
  await stat(indexPath)

  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url ?? '/', 'http://127.0.0.1')
      const filePath = containedPath(distRoot, requestUrl.pathname)
      const body = await readFile(filePath)
      response.writeHead(200, {
        'content-type': CONTENT_TYPES.get(path.extname(filePath)) ?? 'application/octet-stream',
        'cache-control': 'no-store',
      })
      response.end(body)
    } catch (error) {
      response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' })
      response.end(error instanceof Error ? error.message : 'not found')
    }
  })

  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject)
      resolve()
    })
  })
  const address = server.address()
  if (address === null || typeof address === 'string') {
    await new Promise((resolve) => server.close(resolve))
    throw new Error('QA dist server did not bind an IPv4 port')
  }

  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolve, reject) => {
      server.close((error) => error ? reject(error) : resolve())
    }),
  }
}
