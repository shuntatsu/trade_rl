from __future__ import annotations

from pathlib import Path

from apply_frontend_telemetry_generation import (
    read_lines,
    unique_index,
    update_guards,
    update_hook,
    update_live_page_test,
    update_types,
    write_lines,
)


def update_api() -> None:
    path = Path("studio/src/api/studioApi.ts")
    lines = read_lines(path)
    function_start = unique_index(
        lines,
        lambda line: line == "export function loadTelemetryEvents(",
        "loadTelemetryEvents function",
    )
    function_end = next(
        index
        for index in range(function_start + 1, len(lines))
        if lines[index].startswith("export function ")
    )

    generation_argument = "  streamGeneration: string | null = null,"
    if generation_argument not in lines[function_start:function_end]:
        fetcher_matches = [
            index
            for index in range(function_start, function_end)
            if lines[index] == "  fetcher: typeof fetch = fetch,"
        ]
        if len(fetcher_matches) != 1:
            raise RuntimeError(
                f"events fetcher: expected one match, found {len(fetcher_matches)}"
            )
        lines.insert(fetcher_matches[0], generation_argument)
        function_end += 1

    seed_query = "  if (seed !== null) parameters.set('seed', String(seed))"
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
            "    parameters.set('stream_generation', streamGeneration)",
            "  }",
        ]

    old_signature = (
        "  loadTelemetryEvents: (jobId: string, afterSequence?: number, "
        "limit?: number, seed?: number | null) => Promise<TelemetryEventsResponse>"
    )
    new_signature = (
        "  loadTelemetryEvents: (jobId: string, afterSequence?: number, "
        "limit?: number, seed?: number | null, streamGeneration?: string | null) "
        "=> Promise<TelemetryEventsResponse>"
    )
    if new_signature not in lines:
        signature_index = unique_index(
            lines,
            lambda line: line == old_signature,
            "StudioApi loadTelemetryEvents signature",
        )
        lines[signature_index] = new_signature

    write_lines(path, lines)


def main() -> None:
    update_types()
    update_guards()
    update_api()
    update_live_page_test()
    update_hook()
    print("frontend telemetry generation patch v2 applied")


if __name__ == "__main__":
    main()
