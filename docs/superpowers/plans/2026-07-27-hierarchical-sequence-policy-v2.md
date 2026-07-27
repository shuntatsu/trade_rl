# Hierarchical Sequence Policy v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the obsolete concat-only sequence fusion and ambiguous encoder configuration with one causal gated hierarchical sequence policy spanning native time, timeframe, and asset axes.

**Architecture:** Native causal TCNs emit one latent per timeframe. Explicit quality-aware timeframe tokens and one asset-context token pass through a gated pre-norm transformer; its context output then passes through a gated cross-asset transformer. Configuration, checkpoints, diagnostics, export, and serving use one immutable architecture identity and fail closed on v1 or mismatched artifacts.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium, Stable-Baselines3, NumPy, pytest, Ruff, MyPy, TensorBoard, TorchScript/ONNX.

## Global Constraints

- The maintained run schema becomes `training_run_config_v2`.
- Do not retain a runtime `concat_mlp_v1` fallback.
- Historical v1 evidence remains immutable, but v1 checkpoints are not resumed by v2.
- The sequence policy remains causal and uses only completed point-in-time-available bars.
- Oracle BC and PPO must resolve the exact same feature extractor and architecture identity.
- Parameter count must remain below `max_policy_parameters`.
- Structured sequence observations must never be flattened implicitly.
- No Transformer-XL episode memory and no full candle-by-asset attention in this generation.

---

### Task 1: Replace ambiguous encoder configuration

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `tests/rl/test_training_config_active_fields.py`
- Modify: `tests/workflows/test_training_run_config.py`
- Modify: maintained JSON configurations under `examples/binance-multitimeframe/`

**Interfaces:**
- Produces: `ResidualTrainingConfig.observation_encoder: str`
- Produces: v2 sequence fields named in the design spec
- Removes: `sequence_encoder`, `asset_set_encoder`, ambiguous sequence attention fields

- [ ] **Step 1: Write failing enum and v1 rejection tests**

```python
def test_observation_encoder_is_one_closed_choice() -> None:
    with pytest.raises(ValueError, match="observation_encoder"):
        ResidualTrainingConfig(
            timesteps=128,
            gamma=0.99,
            seeds=(0,),
            observation_encoder="unknown",
        )


def test_training_run_v1_fails_with_migration_message() -> None:
    payload = minimal_run_payload()
    payload["schema_version"] = "training_run_config_v1"
    with pytest.raises(ValueError, match="migrate.*training_run_config_v2"):
        TrainingRunConfig.from_mapping(payload)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest tests/rl/test_training_config_active_fields.py tests/workflows/test_training_run_config.py -q
```

Expected: failures because `observation_encoder` and v2 schema handling do not exist.

- [ ] **Step 3: Implement the closed encoder enum and renamed fields**

Use exactly:

```python
observation_encoder: str = "asset_set"
sequence_tcn_capacity: str = "standard"
sequence_timeframe_attention_heads: int = 8
sequence_timeframe_attention_layers: int = 2
sequence_timeframe_ffn_multiplier: int = 3
sequence_timeframe_gate_bias: float = -2.0
sequence_asset_attention_heads: int = 8
sequence_asset_attention_layers: int = 2
sequence_asset_ffn_multiplier: int = 3
sequence_asset_gate_bias: float = -2.0
```

Validate `observation_encoder` against:

```python
{"flat_mlp", "asset_set", "hierarchical_sequence_v2"}
```

Require `MultiInputPolicy` and PPO-like algorithms only for `hierarchical_sequence_v2`. Apply inactive-default validation to every sequence-only field when another encoder is active. Include every field in `digest_payload()`.

- [ ] **Step 4: Bump and validate the run schema**

Set:

```python
schema_version: str = "training_run_config_v2"
```

Reject v1 with an explicit migration message. Convert maintained example JSON files to v2 and remove deleted keys.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 1 test command and expect PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/training.py trade_rl/workflows/training_run.py tests/rl/test_training_config_active_fields.py tests/workflows/test_training_run_config.py examples/binance-multitimeframe
git commit -m "refactor: replace ambiguous encoder configuration"
```

### Task 2: Add generic gated transformer primitives

**Files:**
- Create: `trade_rl/rl/gated_transformer.py`
- Create: `tests/rl/test_gated_transformer.py`

**Interfaces:**
- Produces: `GatedResidual`
- Produces: `GatedTransformerBlock`
- Produces: `GatedTransformerStack`

- [ ] **Step 1: Write failing mask, zeroing, and gate initialization tests**

```python
def test_masked_tokens_cannot_change_unmasked_outputs() -> None:
    stack = GatedTransformerStack(
        d_model=24,
        heads=4,
        layers=2,
        ffn_multiplier=3,
        dropout=0.0,
        gate_bias=-2.0,
    ).eval()
    value = torch.randn(2, 5, 24)
    mask = torch.tensor([[True, True, False, True, True]] * 2)
    changed = value.clone()
    changed[:, 2] += 10000.0
    with torch.no_grad():
        left = stack(value, valid=mask)
        right = stack(changed, valid=mask)
    torch.testing.assert_close(left[:, mask[0]], right[:, mask[0]])
    assert torch.count_nonzero(left[:, 2]) == 0


def test_residual_gates_start_near_closed() -> None:
    block = GatedTransformerBlock(
        d_model=16,
        heads=4,
        ffn_multiplier=3,
        dropout=0.0,
        gate_bias=-2.0,
    )
    assert torch.allclose(block.attention_gate.gate, torch.full((16,), -2.0))
    assert torch.allclose(block.ffn_gate.gate, torch.full((16,), -2.0))
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest tests/rl/test_gated_transformer.py -q
```

- [ ] **Step 3: Implement pre-norm attention and FFN with per-channel gates**

Required residual formula:

```python
return residual + torch.sigmoid(self.gate).view(1, 1, -1) * branch
```

Zero invalid tokens before the first layer and after every layer. Reject all-invalid rows unless a caller-provided valid context token prevents them.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 2 command and expect PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/gated_transformer.py tests/rl/test_gated_transformer.py
git commit -m "feat: add gated transformer primitives"
```

### Task 3: Build quality-aware cross-timeframe fusion

**Files:**
- Create: `trade_rl/rl/timeframe_fusion.py`
- Create: `tests/rl/test_timeframe_fusion.py`

**Interfaces:**
- Consumes: `GatedTransformerStack`
- Produces: `TimeframeQualitySummary`
- Produces: `CrossTimeframeFusion`

- [ ] **Step 1: Write failing quality and masking tests**

```python
def test_fully_missing_timeframe_is_data_invariant() -> None:
    fusion = build_fusion().eval()
    latents, available, staleness, context = inputs()
    available["4h"].zero_()
    mutated = {key: value.clone() for key, value in latents.items()}
    mutated["4h"] += 10000.0
    with torch.no_grad():
        left = fusion(latents, available, staleness, context)
        right = fusion(mutated, available, staleness, context)
    torch.testing.assert_close(left, right)


def test_no_history_falls_back_to_finite_context() -> None:
    fusion = build_fusion().eval()
    latents, available, staleness, context = inputs()
    for value in available.values():
        value.zero_()
    output = fusion(latents, available, staleness, context)
    assert torch.isfinite(output).all()
    assert output.shape == context.shape
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest tests/rl/test_timeframe_fusion.py -q
```

- [ ] **Step 3: Implement explicit quality features**

For each timeframe and asset calculate:

```python
usable = available.any(dim=-1) if available.ndim == 4 else available
has_any = usable.any(dim=-1)
available_fraction = usable.float().mean(dim=-1)
positions = torch.arange(window, device=usable.device).expand_as(usable)
last_index = positions.masked_fill(~usable, -1).max(dim=-1).values
last_fraction = last_index.clamp_min(0).float() / max(window - 1, 1)
```

At the selected timestep calculate per-channel availability fraction and finite log-staleness mean/max. Project the resulting quality vector and add it to projected latent plus timeframe and deterministic duration embeddings.

- [ ] **Step 4: Implement context-token querying and masking**

Input order must be exactly:

```python
("context", "15m", "1h", "4h", "1d")
```

Return only token index zero. A missing timeframe must be key/value masked and exact-zeroed.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Task 3 command and expect PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/timeframe_fusion.py tests/rl/test_timeframe_fusion.py
git commit -m "feat: add quality-aware timeframe fusion"
```

### Task 4: Replace the maintained sequence-policy core

**Files:**
- Modify: `trade_rl/rl/sequence_policy.py`
- Modify: `trade_rl/rl/policies.py`
- Modify: `tests/rl/test_sequence_policy_core.py`

**Interfaces:**
- Consumes: `CrossTimeframeFusion`, `GatedTransformerStack`
- Produces: v2 `SequencePolicyArchitecture`
- Produces: v2 `MultiTimeframeAssetEncoder`

- [ ] **Step 1: Write failing hierarchy and gradient tests**

```python
def test_every_valid_timeframe_receives_gradient() -> None:
    encoder, inputs = build_v2_encoder_and_inputs(requires_grad=True)
    tokens, pooled = encoder(**inputs)
    (tokens.square().mean() + pooled.square().mean()).backward()
    for timeframe in ("15m", "1h", "4h", "1d"):
        assert inputs["sequences"][timeframe].grad is not None
        assert torch.count_nonzero(inputs["sequences"][timeframe].grad) > 0


def test_inactive_assets_are_exact_zero_after_both_attention_axes() -> None:
    encoder, inputs = build_v2_encoder_and_inputs()
    inputs["active"][:, 1] = False
    tokens, _ = encoder(**inputs)
    assert torch.count_nonzero(tokens[:, 1]) == 0
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest tests/rl/test_sequence_policy_core.py -q
```

- [ ] **Step 3: Remove concat fusion and standard cross-asset transformer**

Delete the old `asset_fusion` concatenation path and `nn.TransformerEncoder` cross-asset stack. Build:

```python
context = self.context_encoder(torch.cat((snapshot, asset_state), dim=-1))
asset_tokens = self.timeframe_fusion(
    timeframe_latents,
    available,
    staleness,
    context,
)
asset_tokens = asset_tokens + self.symbol_embedding(symbol_ids)
contextual = self.cross_asset(asset_tokens, valid=active_mask)
```

Update `MultiTimeframeAssetEncoder.forward()` to receive explicit `staleness` mapping. Preserve public imports for causal TCN classes through re-export if files are split.

- [ ] **Step 4: Update the SB3 feature extractor call**

Pass values, availability, and staleness separately. Do not concatenate staleness into normalized market channels before quality computation. The native TCN input may still receive transformed staleness channels through one explicitly named builder helper, but quality features must use the original planes.

- [ ] **Step 5: Run sequence tests and verify GREEN**

Run the Task 4 command and expect PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/sequence_policy.py trade_rl/rl/policies.py tests/rl/test_sequence_policy_core.py
git commit -m "feat: replace sequence policy with hierarchical attention"
```

### Task 5: Bind architecture identity through assembly and checkpoints

**Files:**
- Create: `trade_rl/rl/sequence_architecture.py`
- Modify: `trade_rl/integrations/sb3_model_assembly.py`
- Modify: `trade_rl/rl/checkpointing.py`
- Modify: `trade_rl/integrations/sb3_checkpoint_assembly.py`
- Modify: related tests under `tests/integrations/` and `tests/rl/`

**Interfaces:**
- Produces: `SequenceArchitectureIdentity`
- Produces: `sequence_architecture_digest`

- [ ] **Step 1: Write failing identity mismatch tests**

```python
def test_checkpoint_rejects_different_timeframe_attention_identity(tmp_path: Path) -> None:
    saved = config(sequence_timeframe_attention_layers=2)
    requested = config(sequence_timeframe_attention_layers=3)
    checkpoint = save_checkpoint_with_config(tmp_path, saved)
    with pytest.raises(ValueError, match="sequence architecture identity mismatch"):
        load_checkpoint_with_config(checkpoint, requested)
```

- [ ] **Step 2: Verify RED**

Run focused checkpoint and assembly tests.

- [ ] **Step 3: Implement immutable identity and digest**

Include all fields specified in the design. Build identity once during policy assembly and attach it to `SB3PolicyAssembly`. Persist its digest in checkpoint metadata and compare it before loading weights.

- [ ] **Step 4: Verify GREEN**

Run focused tests and expect PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/sequence_architecture.py trade_rl/integrations/sb3_model_assembly.py trade_rl/rl/checkpointing.py trade_rl/integrations/sb3_checkpoint_assembly.py tests
git commit -m "feat: bind sequence architecture identity"
```

### Task 6: Add bounded diagnostics and runtime acceleration coverage

**Files:**
- Create: `trade_rl/rl/sequence_diagnostics.py`
- Modify: TensorBoard callback integration
- Modify: `tests/integrations/test_sequence_runtime_acceleration.py`
- Add focused diagnostics tests

**Interfaces:**
- Produces: diagnostic-only attention and gate summaries
- Does not alter normal policy forward outputs

- [ ] **Step 1: Write failing no-side-effect diagnostic test**

```python
def test_diagnostic_probe_does_not_change_policy_output_or_gradients() -> None:
    model, observation = build_model_and_observation()
    before = deterministic_output_and_gradients(model, observation)
    collect_sequence_diagnostics(model, observation)
    after = deterministic_output_and_gradients(model, observation)
    assert_tree_close(before, after)
```

- [ ] **Step 2: Verify RED**

Run focused diagnostics tests.

- [ ] **Step 3: Implement bounded diagnostic forward**

Use forward hooks or explicit diagnostic methods only during the probe. Detach collected tensors immediately and record aggregate scalars only.

- [ ] **Step 4: Verify GREEN and acceleration parity**

Run eager/compile and FP32/BF16 tests.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/sequence_diagnostics.py trade_rl tests
git commit -m "feat: add sequence attention diagnostics"
```

### Task 7: Add structured export and canonical serving loader

**Files:**
- Modify: `trade_rl/rl/export.py`
- Modify: serving package and loader modules
- Modify: `trade_rl/serving/runtime.py`
- Add structured export and serving tests

**Interfaces:**
- Produces: structured actor wrapper with canonical ordered tensor tuple
- Produces: architecture-bound serving bundle

- [ ] **Step 1: Write failing structured parity tests**

Construct a corpus containing all cases listed in the design and verify eager policy actions against restored TorchScript/ONNX actions.

- [ ] **Step 2: Verify RED**

Run focused export tests.

- [ ] **Step 3: Implement canonical structured wrapper and loader**

The wrapper signature must use a fixed tensor order derived from `SEQUENCE_OBSERVATION_SCHEMA`; no dict iteration order may define the external contract.

- [ ] **Step 4: Verify GREEN**

Run export, package, and serving runtime tests.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/export.py trade_rl/serving tests
git commit -m "feat: serve structured sequence policies canonically"
```

### Task 8: Full verification and immutable comparison harness

**Files:**
- Add or modify evaluation workflow and evidence docs
- Update `docs/ARCHITECTURE.md`
- Update maintained status document

- [ ] **Step 1: Run static and unit verification**

```bash
ruff format --check .
ruff check .
mypy trade_rl
pytest -q
python -m compileall trade_rl
```

- [ ] **Step 2: Run CPU integration smoke**

Run the maintained structured Oracle BC-to-PPO smoke with v2 configuration.

- [ ] **Step 3: Run CUDA/Docker smoke on the RTX host**

Record peak memory, throughput, parameter count, diagnostics, checkpoint resume, and deterministic serving parity.

- [ ] **Step 4: Run immutable A/B/C walk-forward comparison**

Use the pre-v2 commit for A, current v2 branch for B, and a parameter-matched non-attention control worktree for C. Do not add old-model runtime branches to production code.

- [ ] **Step 5: Publish GO/NO-GO evidence**

Update status as code-complete separately from model-promotion status. Keep production `NO-GO` unless all existing gates pass.

- [ ] **Step 6: Commit**

```bash
git add docs evaluation tests
git commit -m "docs: publish hierarchical sequence v2 evidence"
```
