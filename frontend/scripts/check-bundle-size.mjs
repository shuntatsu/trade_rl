import { readdir, stat } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const MAX_CHUNK_BYTES = 500 * 1024
const assetsDirectory = fileURLToPath(new URL('../dist/assets/', import.meta.url))

async function main() {
  let entries
  try {
    entries = await readdir(assetsDirectory)
  } catch (error) {
    throw new Error(`frontend build assets are unavailable at ${assetsDirectory}`, { cause: error })
  }

  const chunks = entries.filter((entry) => entry.endsWith('.js')).sort()
  if (chunks.length === 0) {
    throw new Error(`frontend build emitted no JavaScript chunks in ${assetsDirectory}`)
  }

  const oversized = []
  for (const chunk of chunks) {
    const chunkPath = path.join(assetsDirectory, chunk)
    const { size } = await stat(chunkPath)
    console.log(`${chunk}: ${(size / 1024).toFixed(2)} KiB`)
    if (size > MAX_CHUNK_BYTES) oversized.push({ chunk, size })
  }

  if (oversized.length > 0) {
    const details = oversized
      .map(({ chunk, size }) => `${chunk}=${size} bytes`)
      .join(', ')
    throw new Error(
      `frontend JavaScript chunk limit exceeded (${MAX_CHUNK_BYTES} bytes): ${details}`,
    )
  }
}

await main()
