# Multi-Timeframe Research Lifecycle

この文書は、15m・1h・4h・1dを使う維持対象Research workflowのPhase分離だけを説明します。Model構造は[ARCHITECTURE.md](ARCHITECTURE.md)、設定値は[CONFIGURATION.md](CONFIGURATION.md)を参照してください。

> Production status remains `NO-GO`。Waiting stateは承認待ちであり、収益性やRelease承認を意味しません。

## Phase 1: `develop`

Datasetを再構築してIdentityを確認し、Nested walk-forward、Configuration selection、Execution sensitivity、Research gateを実行します。

成功時は不変なSelection proposalを生成し、`awaiting_selection_authorization`で停止できます。Selection結果を同じProcess内で自己承認しません。

## Phase 2: `train-selected`

Selection proposal and authorizationを外部Ed25519署名で検証してから、Normalizer fittingとModel構築を開始します。Runtimeへ渡すのはpublic keys onlyです。

Selected-final training forbids injected resume checkpoints by contract. 成功時はSelected-final Runを公開し、`awaiting_fresh_confirmation`で停止できます。

## Phase 3: `finalize`

Selected-final完了後のFresh confirmationとPaper reconciliationを検証し、Final gate stateを記録します。以前のArtifactやEvidenceを上書きしません。

## Exit semantics

- Waiting for external approval: 正常終了
- Research rejection: Candidate非昇格
- Integrity、Identity、Infrastructure failure: 非ゼロ失敗

Phaseを跨ぐ値は、PathではなくDigestと署名済みIdentityで結合します。詳しいGPU操作は[Docker GPU Runbook](operations/docker-gpu-full-training.md)を参照してください。
