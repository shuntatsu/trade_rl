# Live Training Replay Design

## Goal

Add a chart-first `Live Training` workspace to Trade RL Studio that replays training-time exploration as an understandable, research-only trading visualization and places existing deterministic checkpoint-evaluation evidence beside it. The screen must make price movement, position changes, reward, equity, baseline performance and drawdown observable without coupling browser availability to training progress or presenting exploration as model quality.

## Product decisions

- Use the approved chart-first Concept B visual hierarchy.
- Keep the existing fixed-viewport, no-page-scroll Studio shell.
- Provide both `ほぼライブ` and buffered replay modes; buffered replay is the default.
- Provide both candle-by-candle and compressed-event timeline modes.
- Keep the market replay chart as the dominant surface.
- Display position-change markers, current target weight, replay-period PnL, reward, drawdown and recent events together.
- Keep each seed in an independent stream and require an explicit seed selector; records and sequence cursors are never mixed across seeds.
- Separate exploration telemetry from deterministic checkpoint evaluation. Studio reads the existing seed-aware `checkpoint-selection.json` evidence and never executes evaluation, ranking or reselection.
- Display checkpoint evaluation range, evaluation digest and finalist status so the evidence cannot be mistaken for the live replay interval.
- Keep `NO-GO` and `research only` visible. Do not add order entry or exchange connectivity.

## Architecture

Training telemetry is written to an append-only JSON Lines stream under each seed artifact directory. Stable-Baselines3 emits bounded, sampled records through a callback assembled with the existing checkpoint callback. The callback consumes the environment transitions and obtains the exact primary-symbol OHLC interval from the vector environment only when a record is retained. It never changes actions, rewards, environment state or reset behavior.

Studio resolves telemetry and checkpoint evidence only through a known job and its declared artifact root/run ID. Telemetry windows are returned through ordinary JSON endpoints using a monotonically increasing per-seed sequence cursor. Polling avoids introducing an untested streaming lifecycle while remaining compatible with a later Server-Sent Events transport.

The checkpoint-evaluation reader projects only existing `checkpoint_selection_v2_seed_aware` evidence. It verifies schema, finite scores, range ordering, policy and evaluation digests, candidate/finalist identity, duplicate identities and finalist score equality. Invalid evidence fails closed with an artifact error.

The React application keeps remote records, playback cursor, selected seed and live-follow state separate. Incoming records continue to accumulate while playback is paused. Changing seed clears the local buffer and restarts from sequence zero for the selected stream. The chart is implemented as an accessible SVG component using code-native rendering, consistent with the existing Studio dependency policy.

## Telemetry contract

Schema: `training_telemetry_v1`.

Each record contains:

- `sequence`: monotonically increasing positive integer within one seed stream.
- `recorded_at`: UTC timestamp.
- `global_step`, `environment_step`, `seed`, `environment_id`.
- `event_type`: `rollout`, `position`, `risk`, `episode_end`, `checkpoint` or `gap`.
- `market_index` and market timestamp.
- exact aggregated OHLC for the retained decision interval and primary symbol.
- raw action and executed target weights.
- realized weights before/after the step.
- portfolio value, baseline portfolio value, reward, drawdown, interval cost and interval return.
- risk reasons, emergency-deleverage and terminal flags.

Records contain JSON-native values only. NumPy arrays and domain objects are reduced before serialization.

## Sampling and performance

The callback records one normal rollout sample every configured interval, defaulting to 32 environment decisions. It always records significant position changes, risk constraints, emergency deleveraging and terminal steps. Writes are buffered and flushed in batches. Telemetry failure disables further telemetry for that training process and does not abort training.

The browser retains at most 2,048 records for the selected seed. Telemetry files remain append-only and are not compacted or modified by Studio.

## Studio API

- `GET /api/studio/jobs/{job_id}/telemetry/status?seed=`
- `GET /api/studio/jobs/{job_id}/telemetry/events?seed=&after_sequence=&limit=`
- `GET /api/studio/jobs/{job_id}/checkpoint-evaluations`

The status response includes the selected seed, all available seeds, availability, last sequence, record count and source path relative to the project root. The events response includes the stream seed, ordered records, next cursor and truncation flag. Unknown jobs, project-root escapes, symlinks and record/stream seed mismatches are rejected.

The checkpoint response contains only deterministic evidence already produced by the maintained walk-forward workflow: configuration, seed, policy digest, evaluation digest, score, total return, finalist state, checkpoint range and source. It always reports production status `NO-GO`.

## Frontend

Add the `live` workspace between Run Center and Compare. The page contains:

- run and seed selectors;
- replay-mode and timeline-mode segmented controls;
- large primary-symbol market replay chart with position-change and risk markers plus replay cursor;
- pause/play, speed, sampling and jump controls;
- current agent-state rail;
- exploration-period equity, baseline and drawdown panels;
- deterministic checkpoint-return panel for the selected seed;
- checkpoint evaluation range, digest and finalist indication;
- synchronized recent-event list;
- loading, offline, empty, malformed-evidence and telemetry-gap states.

The screen uses the approved dark navy/charcoal palette, cyan/teal selection accent, green gains/buys, red losses/sells, amber risk and blue information/checkpoints. Existing Studio design tokens remain authoritative; new tokens extend them rather than creating a separate theme.

## Failure handling

- Missing telemetry shows an explanatory empty state and leaves Run Center usable.
- Invalid JSON lines increase the malformed count and are never silently interpolated.
- A run or seed change resets playback and prevents cross-run/cross-seed mixing.
- Polling resumes from the last accepted sequence for the selected seed.
- An absent checkpoint-selection artifact is shown as `未生成`; malformed or identity-inconsistent evidence fails closed.
- Telemetry and checkpoint-display failures never stop training.

## Testing

Python tests cover record reduction, exact OHLC extraction, append/read ordering, malformed-line handling, sampling significance, job isolation, seed discovery, seed-scoped cursor semantics and checkpoint-evidence validation. Frontend tests cover navigation, runtime contract validation, seed switching, checkpoint comparison, mode switching, pause-with-receive behavior, cursor movement and empty/error states. Existing typecheck, production build, exact-head CI, fixed-viewport, PostgreSQL, cross-platform and training-image checks remain required.

## Non-goals

Direct exchange orders, manual trading, API-key handling, production authorization, mobile layout, multi-user access, remote internet exposure, checkpoint evaluation execution, candidate selection and release approval are not part of this workspace.
