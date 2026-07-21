# Live Training Replay Design

## Goal

Add a chart-first `Live Training` workspace to Trade RL Studio that replays training-time exploration as an understandable, research-only trading visualization. The screen must make price movement, position changes, trade-like markers, reward, equity, baseline performance and drawdown observable without coupling browser availability to training progress.

## Product decisions

- Use the approved chart-first Concept B visual hierarchy.
- Keep the existing fixed-viewport, no-page-scroll Studio shell.
- Provide both `ほぼライブ` and buffered replay modes; buffered replay is the default.
- Provide both candle-by-candle and compressed-event timeline modes.
- Keep the market replay chart as the dominant surface.
- Display trade markers, current position, unrealized/realized PnL, exploration state and recent events together.
- Separate training rollout from deterministic checkpoint evaluation. This implementation delivers the training-rollout vertical slice and leaves the data contract ready for checkpoint evaluation.
- Keep `NO-GO` and `research only` visible. Do not add order entry or exchange connectivity.

## Architecture

Training telemetry is written to an append-only JSON Lines stream under the seed artifact directory. Stable-Baselines3 emits bounded, sampled records through a callback assembled with the existing checkpoint callback. The callback consumes the environment `info` dictionaries already produced by `ResidualMarketEnv`; it never changes actions, rewards or reset behavior.

Studio resolves telemetry only through a known job and its declared artifact root/run ID. Historical windows are returned through ordinary JSON endpoints. The browser polls for new records using a monotonically increasing sequence cursor in this first slice. The API contract is intentionally compatible with a later Server-Sent Events endpoint, but polling avoids introducing an untested streaming lifecycle into the initial vertical slice.

The React application keeps remote records, playback cursor and live-follow state separate. Incoming records continue to accumulate while playback is paused. The chart is implemented as an accessible SVG component using code-native rendering, consistent with the existing Studio dependency policy.

## Telemetry contract

Schema: `training_telemetry_v1`.

Each record contains:

- `sequence`: monotonically increasing positive integer within one seed stream.
- `recorded_at`: UTC timestamp.
- `global_step`, `environment_step`, `seed`, `environment_id`.
- `event_type`: `rollout`, `position`, `risk`, `episode_end`, `checkpoint` or `gap`.
- `market_index` and optional market timestamp.
- OHLC price data for the selected/primary symbol.
- raw action and executed target weights.
- realized weights before/after the step.
- portfolio value, baseline portfolio value, reward, drawdown, interval cost and interval return.
- risk reasons, emergency-deleverage and terminal flags.

Records contain JSON-native values only. NumPy arrays and domain objects are reduced before serialization.

## Sampling and performance

The callback records one normal rollout sample every configured interval, defaulting to 32 environment decisions. It always records significant position changes, risk constraints, emergency deleveraging and terminal steps. Writes are buffered and flushed in batches. Telemetry failure disables further telemetry for that training process and does not abort training.

The implementation must not modify environment outputs, action selection or reward calculation. With telemetry disabled, no telemetry callback is created.

## Studio API

- `GET /api/studio/jobs/{job_id}/telemetry/status`
- `GET /api/studio/jobs/{job_id}/telemetry/events?after_sequence=&limit=`

The status response includes availability, last sequence, record count and source path relative to the project root. The events response includes ordered records, next cursor and a truncation flag. Paths outside declared artifact roots are rejected.

## Frontend

Add the `live` workspace between Run Center and Compare. The page contains:

- replay-mode and timeline-mode segmented controls;
- large BTCUSDT-style market replay chart with buy/sell/reduce markers and replay cursor;
- pause/play, speed, sampling and jump controls;
- current agent-state rail;
- equity, baseline, reward and drawdown mini-panels;
- synchronized recent-event list;
- loading, offline, empty and telemetry-gap states.

The screen uses the approved dark navy/charcoal palette, cyan/teal selection accent, green gains/buys, red losses/sells, amber risk and blue information/checkpoints. Existing Studio design tokens remain authoritative; new tokens extend them rather than creating a separate theme.

## Failure handling

- Missing telemetry shows an explanatory empty state and leaves Run Center usable.
- Invalid JSON lines are represented as an explicit gap; they are never silently interpolated.
- A run/job change resets playback state and prevents cross-run mixing.
- Polling resumes from the last accepted sequence.
- The browser caps retained records and compacts old display data without changing server artifacts.

## Testing

Python tests cover record reduction, append/read ordering, malformed-line gaps, sampling significance, API job isolation and API cursor semantics. Frontend tests cover navigation, runtime contract validation, mode switching, pause-with-receive behavior, cursor movement and empty/error states. Existing typecheck, production build and no-page-scroll checks remain required.

## Non-goals

Direct exchange orders, manual trading, API-key handling, production authorization, mobile layout, multi-user access, remote internet exposure and checkpoint deterministic evaluation execution are not part of this slice.
