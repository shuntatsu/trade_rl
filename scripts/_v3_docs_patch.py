from __future__ import annotations

from pathlib import Path


def _insert_before(path: str, anchor: str, block: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(anchor) != 1:
        raise RuntimeError(f"expected one documentation anchor in {path}")
    if block.strip() in text:
        return
    source.write_text(text.replace(anchor, block + anchor, 1), encoding="utf-8")


universal_block = r'''
### Artifact-bound V3 research runner

V3 primitivesを実データで最後まで評価するため、`scripts/run_universal_causal_alpha_v3_research.py`をresearch-only entrypointとして使用します。Runnerは次の順序を固定し、途中結果を次段階の選択条件へ流用しません。

```text
strict authored JSON
  -> artifact/runtime identity closure
  -> chronological V3 fit
  -> signal gate
  -> immutable candidate freeze
  -> resumable production replay
  -> economic selection
  -> untouched teacher admission
  -> research-only teacher package
```

Signal gateはearlier selection contractだけを使用し、fitは各contract開始時点より前に完全実現したlabelへ限定します。72h label intervalが重複するpredictionは独立標本として水増しせず、deterministic non-overlapping cohortへ落としてからscope-level block bootstrapを行います。Signal gateに失敗した場合、candidate freeze、production replay、teacher-admission holdoutへ進みません。

Candidate freeze後のproduction replayはproduction environment、既存execution cost、causal liquidity capを使用します。Replay結果は`(candidate_digest, symbol, episode_index)`ごとのimmutable recordとしてatomicに保存し、resume時にはrun/freeze/contract/code identityを再検証します。Valid recordが存在するscopeだけskipし、unknown/corrupt/identity-drifted recordはfail closedします。`below_minimum_notional`と`zero_quantity_after_rounding`は既存契約どおりexplained no-fillとして扱い、それ以外のexecution rejection、hard-risk violation、`-5%` lower-tail breachは不可逆reject条件です。

Teacher admissionはcandidate selection完了後にselected candidateだけへ開きます。Persist済みper-symbol admission recordは再利用しますが、process crashがevaluation完了とrecord永続化の間に発生した場合まで「物理的にexactly once」とは主張しません。保証するのは、persist済みholdout resultを再評価せず、各symbolについてexactly one durable admission recordを受理することです。Admission failure時にはresearch-only teacher packageを生成しません。

CLIのterminal research outcomeは、admittedがexit code 0、signal rejectionが**exit code 2**、economic selection rejectionが**exit code 3**、teacher admission rejectionが**exit code 4**です。いずれのV3 evidenceも`promotion_eligible=false`で、Software successをProduction GOへ読み替えません。

Downstream learner pathはこのrunnerの非目標です。`DAgger -> BC`、critic warm start、`anchored PPO`/Lagrangian系は**only after teacher admission**で別工程として実行します。Admission前に`anchored_target_residual`やDAggerを使ってcanonical gateを迂回しません。

'''
_insert_before(
    "docs/UNIVERSAL_TRAINING.md",
    "## 6. Teacher admissionとfail-closed順序",
    universal_block,
)

research_block = r'''
### Executable V3 deterministic workflow

V3はprimitive定義だけでなく、`run_universal_causal_alpha_v3_research.py`から実データworkflowを実行できます。Workflowは`signal gate -> candidate freeze -> resumable production replay -> economic selection -> teacher admission -> research-only teacher package`をartifact-boundに接続します。Signal/economic/holdout scopeはchronologicalに分離され、candidate freezeより前にholdoutを開きません。

Resumeのsource of truthはprogress表示ではなくimmutable scope recordです。Run manifest、generator code、candidate freeze、episode contractのIdentityが一致するrecordだけを再利用し、corrupt/unknown recordはfail closedします。CLIではsignal rejectionを**exit code 2**、selection rejectionを**exit code 3**、admission rejectionを**exit code 4**として区別します。

V3 teacher admissionがPASSした場合でも、それはRL upliftやProduction認可ではありません。次のlearner工程は`DAgger -> BC`および`anchored PPO`を含み、**only after teacher admission**で別のquality gateとして実行します。

'''
_insert_before(
    "docs/RESEARCH_STATUS.md",
    "これらはsoftware contractです。Teacher holdoutの経済成績が実データで合格した、",
    research_block,
)
