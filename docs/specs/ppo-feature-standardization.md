# PPO fit-only feature standardization

Status: Active

The objective is a working profit-seeking trading bot under the user's 20%
research drawdown tolerance. The completed baseline loses money across all five
PPO seeds; the completed risk-only comparison returned KEEP_BASELINE with only
three of five paired improvements and a negative family median. Preserve both
studies as immutable evidence. The next comparison uses the original default-risk
five-seed baseline, not the failed risk treatment or a selected profitable seed.

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

No real normalized candidate is fit until a new single-factor protocol fixes the
original five-seed baseline, this transform, 262144 steps per seed, sequential
layout, the same 12 inputs and Dataset, before-2023 fit, and 2023–2024 development.
Use the completed risk study's relative gate and the original absolute gates:
four of five paired wins, positive median paired delta, complete replay, no new
termination, and candidate drawdowns at most 20%; profitability, terminal flatness
and stress qualification remain separate. No simultaneous reward, layout,
network or execution change may enter that comparison. Prospective paper evidence
and correct execution remain necessary before operational profit claims.
