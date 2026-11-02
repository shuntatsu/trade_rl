# Issue #810 P3: synthetic rollout-boundary result

Status: Completed synthetic diagnostic; `NOT_ESTABLISHED`.

## Result

The frozen six-cell PPO diagnostic completed, but none of the six cells met the preregistered acceptance gates. The 48-transition in-rollout positive control failed, so this run does not establish actor credit for either the in-rollout or 72-transition cross-boundary payoff. No parameter, seed, threshold, or evaluation window was changed after results were observed.

All six fits completed the requested 4,096 timesteps and 4,096 joined transitions, producing 64 rollouts and 1,280 optimizer-step calls per cell. Across the 18 trained held-out windows, no window reached the required five of six cue-aligned snapshots; the best window reached three of six. The trained policy's terminal log return was between `-0.008449` and `0.000000`, so none cleared the required `L > 0.03`. No window exceeded the fixed-long control by the required `0.02` log-wealth uplift. Nine of the 18 windows passed both neutral-block flat/no-pending snapshots. These are synthetic fixture outcomes, not market returns.

| Payoff transition | Seed | Cell acceptance | Trained cue-aligned snapshots across 3 held-out windows | Neutral flat/no-pending snapshots | Trained terminal log-return range |
| ---: | ---: | --- | ---: | ---: | ---: |
| 48 | 0 | Failed | 8 / 18 | 0 / 6 | `-0.008449` to `-0.001038` |
| 48 | 7 | Failed | 9 / 18 | 0 / 6 | `-0.000914` to `-0.000903` |
| 48 | 17 | Failed | 0 / 18 | 6 / 6 | `0.000000` |
| 72 | 0 | Failed | 0 / 18 | 6 / 6 | `0.000000` |
| 72 | 7 | Failed | 9 / 18 | 0 / 6 | `-0.000887` to `-0.000877` |
| 72 | 17 | Failed | 0 / 18 | 6 / 6 | `0.000000` |

Each held-out window contains six non-neutral cue snapshots and two neutral snapshots. Every one of the 18 trained evaluations completed with a valid terminal settlement, exact-flat quantities, and no active orders. The same terminal invariants held for all 72 policy-window evaluations across initialized, trained, fixed-long, and cash policies. Cash ended at zero log return; fixed-long returned approximately `-0.00088` per window. A valid terminal endpoint is a software/accounting result and does not change the failed actor or return gates.

## Provenance

- Frozen protocol: [`ppo-rollout-boundary-protocol.md`](ppo-rollout-boundary-protocol.md), SHA-256 `7834eaac30265c89f3f352a8f4c7590c48126292d7377f708e9dbf83925567e3`.
- Measured source HEAD: `1f1c9f6954b40675b6eaba1d37141b6f4a16bc05`. The run manifest records a clean worktree and hashes for all eight source modules; the independent receipt audit confirmed those module hashes and all six cell receipts match the run records.
- Run schema: `issue810_p3_rollout_boundary_run_v1`; disposition: `NOT_ESTABLISHED`; six planned and six completed cells.
- Run artifact: `work/p3r2/run.json`, SHA-256 `995c2759f94aa62cbb738b68fae7e601d752bd0abafa94c0ee166eabcf698de8`, completed `2026-10-28T22:24:45.366854Z`.

The run JSON, per-cell receipts, and checkpoint artifacts remain in the task's external run directory and are not included in this source commit. This page is a concise result record, not a replacement for those artifacts.

## Interpretation and limits

The run shows that the fixed synthetic fit/evaluation pipeline completed and preserved terminal accounting, while the trained actor failed the predeclared held-out alignment and after-cost acceptance rules. The 48-transition positive control's failure makes the 72-transition comparison non-establishing for cross-rollout actor credit. This is not evidence of a profitable strategy, market edge, real-data generalization, improved stressed wealth, G4 approval, unused-period validation, or live readiness. Historic research classifications remain unchanged.
