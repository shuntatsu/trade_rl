# Research Trading Chart Workspace Design

## Goal

Replace the static SVG market replay in Studio Live Training with a desktop-only, TradingView-style research workspace that lets a researcher inspect market movement, policy decisions, learning signals, risk, and performance on one synchronized time axis.

## Approved Scope

The first release is the balanced desktop layout. It is not a TradingView clone and does not include drawing tools, saved layouts, annotations, multi-chart tiling, mobile layout, image export, or order entry.

The visible workspace contains:

- one always-visible Run selector
- a secondary “対象を変更” popover for Seed and Environment
- centered replay transport: first event, previous event, play/pause, next event, latest event, and 1x/4x/8x speed
- a “最新へ追従” switch instead of a decorative LIVE badge
- a “表示項目” popover and explicit display reset
- Symbol, timeframe, OHLC, and range presets immediately above the chart they affect
- a replay scrubber immediately below the chart
- three summary metrics only: RL equity, baseline delta, and drawdown
- a synchronized right-side inspector for the selected or hovered point

The removed “接続 / LIVE / Seed / Env” metric card must not return. Seed and Environment remain available only as source-selection controls and research evidence in the inspector.

## Rendering Architecture

Use `lightweight-charts` 5.2.0. React owns application state, controls, replay position, filtering, and inspector contents. Lightweight Charts owns financial rendering, scales, zoom, pan, crosshair, hit testing, and panes.

Use one chart instance with four panes so every series shares one time scale and crosshair:

1. Market pane: candlesticks and BUY/SELL/RISK/END markers.
2. Policy pane: target weight and executed weight.
3. Learning pane: reward and interval cost.
4. Performance pane: RL equity, baseline equity, and drawdown.

Keep the TradingView attribution logo enabled as required by the library license terms.

## Component Boundaries

### `researchChartModel.ts`

Pure transformation layer. It parses telemetry timestamps, filters by symbol, aggregates records to 15m/1h/4h/1d buckets, produces all chart series, derives markers from the displayed primary asset, and maps chart time back to a representative telemetry record. It must not access React or the chart runtime.

### `ReplayToolbar.tsx`

Owns only presentation and callbacks for Run/source selection, transport, speed, latest-follow mode, layer visibility, and reset. Seed and Environment changes are applied together from the source popover to avoid partially changing the inspected stream.

### `ResearchChartWorkspace.tsx`

Owns the Lightweight Charts lifecycle. It creates one chart, adds the four panes and series, updates data, subscribes to crosshair and click events, applies range presets, follows the latest point when requested, and cleans up observers and subscriptions on unmount.

### `ResearchChartInspector.tsx`

Renders the committed replay record by default and the crosshair-preview record while the pointer is over a valid time. Hover is preview-only; chart click commits replay position.

### `LiveTrainingPage.tsx`

Continues to own telemetry loading, job/seed/environment selection, checkpoint evidence, replay timing, and the diagnostics/replay view switch. It composes the toolbar and research workspace rather than implementing chart geometry.

## Data Contract

Telemetry is filtered to the selected symbol. The primary displayed asset is index 0 in `weightsBefore`, `weightsAfter`, `action`, and `executedTarget`, matching the existing replay contract.

For each timeframe bucket:

- open: first finite open, falling back to first close
- high: maximum finite high/open/close
- low: minimum finite low/open/close
- close: last finite close, falling back to last open
- metric values: last finite record in the bucket
- representative record: last record by market time and sequence

Timestamp parsing accepts nanosecond telemetry strings by truncating fractional seconds to milliseconds before `Date.parse`. Timestamp-less or non-finite records are excluded rather than assigned invented times.

A position marker is BUY only when primary weight delta is positive and SELL only when it is negative. A zero primary delta emits no directional marker even when another asset caused the position event.

## Interaction State

- Hover: preview the nearest record in the inspector without stopping playback.
- Click: commit that record as the replay cursor and pause playback.
- Pan/zoom: manual navigation disables latest-follow mode.
- Latest-follow: keeps the right edge and replay cursor at the newest record as telemetry arrives.
- Range presets: 1H, 24H, 7D, and all available history.
- Reset: restore 24H, fit price scale, clear hover, enable latest-follow, and show all default layers.
- Scrubber: commit a raw replay-record index; aggregated chart panes move to the corresponding bucket.
- Previous/next event: skip rollout records and navigate only position, risk, or episode-end events.

## Error and Empty States

The existing page-level telemetry error remains authoritative. The chart renders an accessible empty state if no valid records exist for the selected symbol/timeframe. Invalid chart events, missing optional values, and empty series do not throw. A ResizeObserver absence in tests or older runtimes falls back to measured container dimensions.

## Performance

Only one chart instance is visible. The model transformation is memoized by records, symbol, and timeframe. The chart receives aggregated data and does not render one DOM node per candle. Updates use Lightweight Charts series APIs; React does not recreate the chart for cursor changes. The initial target is smooth interaction with at least 50,000 raw telemetry records after aggregation.

## Accessibility

All controls use native buttons, selects, checkboxes, and range input with visible labels or `aria-label`. The Canvas chart has an accessible summary describing the selected symbol, timeframe, visible range, and committed record. Essential values remain in the OHLC header and inspector; they are not available only by hover.

## Testing

- Pure unit tests cover timestamp parsing, aggregation, marker truthfulness, event navigation, and source-symbol filtering.
- Component tests cover toolbar popovers, atomic source application, transport callbacks, layer toggles, and latest-follow.
- Chart tests mock Lightweight Charts at the API boundary and verify pane assignment, series data, marker creation, range changes, click commitment, cleanup, and no chart recreation on cursor-only updates.
- Page tests cover the approved control hierarchy, removal of LIVE UI, environment isolation, replay pause/click behavior, and checkpoint evidence selection.
- Existing Studio unit tests, typecheck, Vite build, fixed-viewport Playwright checks, and full repository CI must pass.

## Delivery

Implement on `agent/research-trading-chart-workspace`, based on PR #316 head. Open a Draft PR against `main` so CI runs while clearly declaring the dependency on PR #316. Do not merge either PR without an explicit user request.
