from __future__ import annotations

from pathlib import Path


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_index(lines: list[str], predicate: object, name: str) -> int:
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    matches = [index for index, line in enumerate(lines) if predicate(line)]
    if len(matches) != 1:
        raise RuntimeError(f"{name}: expected one match, found {len(matches)}")
    return matches[0]


def interface_bounds(lines: list[str], declaration: str) -> tuple[int, int]:
    start = unique_index(lines, lambda line: line == declaration, declaration)
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return start, end
    raise RuntimeError(f"{declaration}: closing brace not found")


def insert_interface_field(
    lines: list[str],
    declaration: str,
    after: str,
    field: str,
) -> None:
    start, end = interface_bounds(lines, declaration)
    if field in lines[start:end]:
        return
    matches = [index for index in range(start, end) if lines[index] == after]
    if len(matches) != 1:
        raise RuntimeError(
            f"{declaration} {after}: expected one match, found {len(matches)}"
        )
    lines.insert(matches[0] + 1, field)


def update_types() -> None:
    path = Path("studio/src/data/types.ts")
    lines = read_lines(path)
    insert_interface_field(
        lines,
        "export interface TelemetryStatusResponse {",
        "  source: string | null",
        "  streamGeneration: string | null",
    )
    insert_interface_field(
        lines,
        "export interface TelemetryEventsResponse {",
        "  sequenceGaps: [number, number][]",
        "  streamGeneration: string | null",
    )
    insert_interface_field(
        lines,
        "export interface TelemetryEventsResponse {",
        "  streamGeneration: string | null",
        "  resetRequired: boolean",
    )
    write_lines(path, lines)


def update_guards() -> None:
    path = Path("studio/src/live/telemetryGuards.ts")
    lines = read_lines(path)
    if not any(line.startswith("const generationPattern =") for line in lines):
        index = unique_index(
            lines,
            lambda line: line == "const eventTypes = new Set([",
            "eventTypes declaration",
        )
        lines[index:index] = [
            "const generationPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/",
            "const isNullableGeneration = (value: unknown): value is string | null =>",
            "  value === null || (typeof value === 'string' && generationPattern.test(value))",
        ]

    status_line = "    && (value.source === null || typeof value.source === 'string')"
    status_index = unique_index(lines, lambda line: line == status_line, "status guard")
    status_generation = "    && isNullableGeneration(value.streamGeneration)"
    if status_generation not in lines[status_index : status_index + 3]:
        lines.insert(status_index + 1, status_generation)

    gaps_line = "    && Array.isArray(value.sequenceGaps)"
    gaps_index = unique_index(lines, lambda line: line == gaps_line, "events gaps guard")
    generation_conditions = [
        "    && isNullableGeneration(value.streamGeneration)",
        "    && typeof value.resetRequired === 'boolean'",
        "    && (!value.resetRequired || (value.items.length === 0 && value.nextSequence === 0))",
    ]
    if generation_conditions[0] not in lines[gaps_index : gaps_index + 6]:
        lines[gaps_index + 1 : gaps_index + 1] = generation_conditions
    write_lines(path, lines)


def block_bounds(
    lines: list[str],
    start_predicate: object,
    end_predicate: object,
    name: str,
) -> tuple[int, int]:
    start = unique_index(lines, start_predicate, f"{name} start")
    for end in range(start + 1, len(lines)):
        if callable(end_predicate) and end_predicate(lines[end]):
            return start, end
    raise RuntimeError(f"{name}: end not found")


def update_api() -> None:
    path = Path("studio/src/api/studioApi.ts")
    lines = read_lines(path)
    function_start = unique_index(
        lines,
        lambda line: line == "export async function loadTelemetryEvents(",
        "loadTelemetryEvents function",
    )
    function_end = next(
        index
        for index in range(function_start + 1, len(lines))
        if lines[index].startswith("export async function ")
    )
    fetcher_line = "  fetcher: typeof fetch = fetch,"
    fetcher_matches = [
        index
        for index in range(function_start, function_end)
        if lines[index] == fetcher_line
    ]
    if len(fetcher_matches) != 1:
        raise RuntimeError(
            f"events fetcher: expected one match, found {len(fetcher_matches)}"
        )
    generation_argument = "  streamGeneration: string | null = null,"
    if generation_argument not in lines[function_start:function_end]:
        lines.insert(fetcher_matches[0], generation_argument)
        function_end += 1

    seed_query = "  if (seed !== null) params.set('seed', String(seed))"
    seed_matches = [
        index
        for index in range(function_start, function_end)
        if lines[index] == seed_query
    ]
    if len(seed_matches) != 1:
        raise RuntimeError(
            f"events seed query: expected one match, found {len(seed_matches)}"
        )
    if not any(
        "stream_generation" in line
        for line in lines[seed_matches[0] + 1 : seed_matches[0] + 5]
    ):
        lines[seed_matches[0] + 1 : seed_matches[0] + 1] = [
            "  if (streamGeneration !== null) {",
            "    params.set('stream_generation', streamGeneration)",
            "  }",
        ]

    interface_start = unique_index(
        lines,
        lambda line: line == "export interface StudioApi {",
        "StudioApi interface",
    )
    method_start = next(
        index
        for index in range(interface_start, len(lines))
        if lines[index] == "  loadTelemetryEvents: ("
    )
    method_end = next(
        index
        for index in range(method_start + 1, len(lines))
        if lines[index] == "  ) => Promise<TelemetryEventsResponse>"
    )
    interface_generation = "    streamGeneration?: string | null,"
    if interface_generation not in lines[method_start:method_end]:
        seed_line = "    seed?: number | null,"
        seed_matches = [
            index
            for index in range(method_start, method_end)
            if lines[index] == seed_line
        ]
        if len(seed_matches) != 1:
            raise RuntimeError(
                f"StudioApi seed argument: expected one match, found {len(seed_matches)}"
            )
        lines.insert(seed_matches[0] + 1, interface_generation)
    write_lines(path, lines)


def update_live_page_test() -> None:
    path = Path("studio/src/pages/LiveTrainingPage.test.tsx")
    lines = read_lines(path)
    status_index = unique_index(
        lines,
        lambda line: "source: items.length > 0 ?" in line,
        "Live status source",
    )
    status_generation = (
        "        streamGeneration: items.length > 0 "
        "? '33333333-3333-4333-8333-333333333333' : null,"
    )
    if status_generation not in lines[status_index + 1 : status_index + 3]:
        lines.insert(status_index + 1, status_generation)

    gaps_index = unique_index(
        lines,
        lambda line: line.strip() == "sequenceGaps: [],",
        "Live events gaps",
    )
    events_generation = (
        "        streamGeneration: '33333333-3333-4333-8333-333333333333',"
    )
    if events_generation not in lines[gaps_index + 1 : gaps_index + 4]:
        lines[gaps_index + 1 : gaps_index + 1] = [
            events_generation,
            "        resetRequired: false,",
        ]
    write_lines(path, lines)


HOOK = """import { useCallback, useEffect, useRef, useState } from 'react'

import type { StudioApi } from '../api/studioApi'
import type {
  TelemetryStatusResponse,
  TrainingTelemetryRecord,
} from '../data/types'

const POLL_INTERVAL_MS = 1000
const MAX_BUFFERED_RECORDS = 50_000

function mergeRecords(
  current: TrainingTelemetryRecord[],
  incoming: TrainingTelemetryRecord[],
): TrainingTelemetryRecord[] {
  if (incoming.length === 0) return current
  const bySequence = new Map(current.map((item) => [item.sequence, item]))
  for (const item of incoming) bySequence.set(item.sequence, item)
  return [...bySequence.values()]
    .sort((left, right) => left.sequence - right.sequence)
    .slice(-MAX_BUFFERED_RECORDS)
}

export interface TrainingTelemetryState {
  status: TelemetryStatusResponse | null
  records: TrainingTelemetryRecord[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

export function useTrainingTelemetry(
  jobId: string | null,
  api: StudioApi,
  seed: number | null = null,
): TrainingTelemetryState {
  const [status, setStatus] = useState<TelemetryStatusResponse | null>(null)
  const [records, setRecords] = useState<TrainingTelemetryRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sequence = useRef(0)
  const generation = useRef<string | null>(null)

  const refresh = useCallback(async () => {
    if (jobId === null) return
    setLoading(true)

    async function loadSnapshot(allowRetry: boolean): Promise<void> {
      const expectedGeneration = generation.current
      const [nextStatus, page] = await Promise.all([
        api.loadTelemetryStatus(jobId, seed),
        api.loadTelemetryEvents(
          jobId,
          sequence.current,
          512,
          seed,
          expectedGeneration,
        ),
      ])

      const retry = async (
        nextGeneration: string | null,
        reason: string,
      ): Promise<void> => {
        setRecords([])
        setStatus(null)
        sequence.current = 0
        generation.current = nextGeneration
        if (!allowRetry) {
          throw new Error(`Telemetry stream changed repeatedly: ${reason}`)
        }
        await loadSnapshot(false)
      }

      if (page.resetRequired) {
        if (page.streamGeneration === null) {
          throw new Error('Telemetry reset response has no stream generation')
        }
        await retry(page.streamGeneration, 'cursor generation is stale')
        return
      }

      if (
        nextStatus.streamGeneration !== null
        && page.streamGeneration !== null
        && nextStatus.streamGeneration !== page.streamGeneration
      ) {
        await retry(null, 'status and events generations differ')
        return
      }

      if (
        nextStatus.available
        && (
          nextStatus.streamGeneration === null
          || page.streamGeneration === null
        )
      ) {
        throw new Error('Available telemetry response has no stream generation')
      }

      const resolvedGeneration = page.streamGeneration
        ?? nextStatus.streamGeneration
      if (generation.current === null) {
        generation.current = resolvedGeneration
      } else if (
        resolvedGeneration !== null
        && generation.current !== resolvedGeneration
      ) {
        throw new Error('Telemetry generation changed without a reset response')
      }

      setStatus(nextStatus)
      if (page.items.length > 0) {
        setRecords((current) => mergeRecords(current, page.items))
      }
      sequence.current = page.nextSequence
    }

    try {
      await loadSnapshot(true)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Telemetry unavailable')
    } finally {
      setLoading(false)
    }
  }, [api, jobId, seed])

  useEffect(() => {
    sequence.current = 0
    generation.current = null
    setRecords([])
    setStatus(null)
    setError(null)
    if (jobId === null) return
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [jobId, refresh, seed])

  return { status, records, loading, error, refresh }
}
"""


def update_hook() -> None:
    Path("studio/src/live/useTrainingTelemetry.ts").write_text(HOOK, encoding="utf-8")


def main() -> None:
    update_types()
    update_guards()
    update_api()
    update_live_page_test()
    update_hook()
    print("frontend telemetry generation patch applied")


if __name__ == "__main__":
    main()
