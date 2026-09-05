# Universal Trade RL U1 Normalization Clip Amendment

Status: **Normative U1 V1 amendment**  
Production status: **NO-GO**

This amendment closes an omission in Section 9 of `2026-08-31-universal-trade-rl-u1-observation-reward-design.md`. It does not change the implemented U1 algorithm; it makes the already implemented and identity-bound clipping rule explicit.

## 1. Fixed transform semantics

For each available continuous market-sequence value, U1 V1 applies the fitted equal-symbol statistics:

```text
z = (x - mean) / scale
policy_value = clip(z, -10.0, +10.0)
```

The clipping threshold is exactly:

```text
normalizer_clip_value = 10.0
```

It is static and precommitted for U1 V1. It is not a Development/Admission tuning parameter and must not be changed after observing evaluation results within the same U1 generation.

For `available=false`, the policy value remains exactly `0.0`; the raw placeholder must not influence the tensor. Availability and staleness remain separate channels and are not replaced by clipping.

## 2. Identity requirements

`normalizer_clip_value` is part of the normalizer statistics identity. A change to the clipping threshold therefore changes `statistics_digest` and cannot be treated as the same fitted normalization contract.

The frozen U1 artifact contract additionally binds `normalizer_clip_value=10.0`. A normalizer using any other clipping threshold is invalid for U1 V1 materialization and U2 handoff.

## 3. Test oracle

U1 verification must include all of the following:

- standardized values above `+10 sigma` become exactly `+10.0`;
- standardized values below `-10 sigma` become exactly `-10.0`;
- unavailable raw placeholders remain policy-invariant and produce value `0.0`;
- `u1_contract.json` rejects a non-10.0 U1 V1 normalizer clip.

## 4. Scope

This amendment only specifies U1 V1 market-sequence normalization. It does not add reward shaping, dynamic clipping, symbol-specific normalization, Development/Admission fitting, or Production authorization.
