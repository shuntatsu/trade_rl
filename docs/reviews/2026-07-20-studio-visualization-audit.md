# Trade RL Studio Visualization Audit

## Scope

Reviewed Compare, Evidence Explorer, and Serving Monitor as read-only research decision surfaces. The audit focused on analytical truth, comparison direction, uncertainty/missingness, accessibility, and the no-page-scroll constraint.

## Findings and corrections

1. **Cumulative wealth used independent per-series scaling.** This made every line occupy nearly the full chart height and visually overstated similarity. Corrected to one shared y-domain across both runs and both baselines, including a visible 1.00 reference.
2. **Metric delta color ignored metric preference.** A positive delta was shown as favorable even for lower-is-better metrics such as cost and maximum drawdown. Corrected to preference-aware `改善 / 悪化 / 同等` labels with signed deltas.
3. **Fold stability was a dense text row.** Replaced with zero-centred in-cell bars so negative and positive fold returns are directly comparable without relying on color alone.
4. **Paper target-weight bars discarded sign.** Replaced absolute-only bars with a signed diverging scale, added explicit numeric values, and exposed non-cash gross and net exposure.
5. **Evidence items were shown as a linear chain.** The UI did not prove a causal dependency between every adjacent item. Renamed the surface to Evidence coverage and removed connector lines.
6. **Chart accessibility depended on visual marks.** Added a semantic chart title/description and a screen-reader-only final-value table. Essential values remain visible without hover.

## Verification

- deterministic component fixtures cover preference-aware outcomes, shared scale metadata, signed weights, and evidence terminology
- all Studio component tests, TypeScript checks, and production build pass
- fixed-viewport browser checks pass at 1536×1024 and 1440×900 for every workspace
- human screenshot review confirms shared scale labels, baseline pattern encoding, zero-centred fold bars, and signed exposure bars
