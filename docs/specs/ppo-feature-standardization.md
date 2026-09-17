# PPO fit-only feature standardization

Status: Active

The objective is a working profit-seeking trading bot under the user's 20%
research drawdown tolerance. The completed baseline loses money across all five
PPO seeds; the separate risk-only comparison is still running from frozen source.
Its code and evidence must not change during this work.

A training-only audit found roughly three orders of magnitude between selected
feature standard deviations. Add an opt-in fitted transform of local feature
values, leaving the default Observation v2 contract byte-for-byte unchanged.
This capability alone is not a claim that normalization improves profitability.

Fit one pooled, symbol-balanced mean and population variance per selected feature
using only finite, available values from permitted fit symbols and decision rows
`[start_index, stop_index)`. Each fit symbol has equal total weight per feature;
reject a symbol/feature with no usable training values.
The pooled variance includes between-symbol differences: the mean of each
symbol's within-symbol variance plus its squared mean distance from the pooled
mean. Singleton interleaved environments use the same pooled fitted transform.

Constant-feature scales
use 1 when the raw standard deviation is at most 1e-12. No clipping, reward
normalization, online updates, symbol identifiers, or additional observations.

The same frozen transform must be applied by the training environment and returned
inference strategy. Unavailable/nonfinite values remain zero after transformation;
mask, staleness, intent and weight retain their existing layout and semantics.
Validate ordered feature indices and training scope. Both training layouts share
one transform fitted before optimization, never fitted separately per episode.

Persist canonical JSON metadata alongside any normalized policy: schema, source
Dataset ID, ordered feature indices/names, fit symbols/window, means, scales and
usable counts. A policy without its fitted transform is not a complete normalized
model artifact. The schema is separate from unchanged legacy Observation v2.
The model bundle binds policy bytes and normalizer metadata under one canonical
manifest digest. Reload requires that expected digest and the feed's feature
names, rejecting a missing/swapped transform or mismatched feature schema before
loading the policy. The existing unnormalized factory remains unchanged; a new
normalized study must explicitly preserve the fitted transform in its factories.

Synthetic tests precede implementation: future/unselected-symbol perturbation
invariance; symbol balancing despite missingness; missing/constant feature handling;
train/inference identity; reset/layout propagation; exact default compatibility;
metadata roundtrip and malformed/scope-mismatched transforms rejected.

No real normalized candidate is fit until the risk comparison has finished and a
new five-seed single-factor protocol has fixed its baseline, transform, budget,
data/evaluation scope and relative/absolute gates. No simultaneous reward, layout,
network or execution change may enter that comparison. Prospective paper evidence
and correct execution remain necessary before operational profit claims.
