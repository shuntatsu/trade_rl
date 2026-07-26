from __future__ import annotations

from pathlib import Path


SUPERSEDED = """> [!IMPORTANT]\n> **Superseded for actor composition, episode aggregation, estimator scheduling, and feasibility-probe behavior by:**\n> `docs/superpowers/specs/2026-07-26-pr-c-lagrangian-stability-correction.md`\n> `docs/superpowers/plans/2026-07-26-pr-c-lagrangian-stability-correction.md`\n> Where this document conflicts with those files, the correction specification and plan are normative.\n\n"""


def add_header(path: Path, title: str) -> str:
    content = path.read_text()
    if "Superseded for actor composition" not in content:
        marker = f"{title}\n\n"
        if content.count(marker) != 1:
            raise RuntimeError(f"{path}: title marker mismatch")
        content = content.replace(marker, marker + SUPERSEDED)
    return content


main_path = Path(
    "docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-lagrangian.md"
)
main = add_header(main_path, "# Stabilized Lagrangian PPO Implementation Plan")
main = main.replace(
    "with independently normalized Cost Critic advantages and stabilized per-cost dual variables",
    "with raw Cost Critic advantages, final-only combined normalization, and stabilized per-cost dual variables",
)
main = main.replace(
    "- Reward advantages and each cost advantage are normalized independently; cost columns are never concatenated before normalization.",
    "- The actor composes raw reward and raw cost advantages first; only the final combined vector uses the pinned SB3 normalization. Per-cost standardization is diagnostics-only.",
)
main = main.replace(
    "- `trade_rl/rl/lagrangian_advantages.py`: pure independent normalization and reward-minus-cost advantage composition.",
    "- `trade_rl/rl/lagrangian_advantages.py`: pure raw reward-minus-cost composition plus diagnostics-only normalization helpers.",
)
main_path.write_text(main)


addendum_path = Path(
    "docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-stability-addendum.md"
)
addendum = add_header(
    addendum_path,
    "# Lagrangian PPO Stability Addendum Implementation Plan",
)
addendum = addendum.replace(
    "a pre-training zero-action joint-feasibility witness",
    "a pre-training canonical-action diagnostic probe",
)
addendum = addendum.replace(
    "add a separate witness evaluator that runs a fresh unwrapped environment with the minimal zero action before model construction. Backend integration records evidence and rejects only an explicitly infeasible configured constraint set.",
    "add a separate canonical-action probe that runs a fresh environment before model construction. Backend integration records warning evidence for budget violations and rejects only malformed execution, metadata, or unsupported action semantics.",
)
addendum = addendum.replace(
    "- A witness uses a fresh environment and zero action, never the training environment state.",
    "- A canonical-action probe uses a fresh environment and the maintained zero-action semantics, never the training environment state.",
)
addendum = addendum.replace(
    "- A witness must complete the configured number of episodes; reaching the maximum step count without completion fails closed.",
    "- A probe must complete the configured number of valid episodes; reaching the maximum step count without completion fails closed.",
)
addendum = addendum.replace(
    "- A witness estimate must use the same aggregation semantics and cost ordering as the Lagrangian schema.",
    "- A probe estimate must use the same aggregation semantics and cost ordering as the Lagrangian schema.",
)
addendum = addendum.replace(
    "- Witness settings and witness evidence identity must be included in training/checkpoint identity.",
    "- Probe settings and probe evidence identity must be included in training/checkpoint identity.",
)
addendum = addendum.replace(
    "- Ordinary `ppo` and `cost_critic_ppo` cannot accept active witness settings.",
    "- Ordinary `ppo` and `cost_critic_ppo` cannot accept active probe settings.",
)
addendum = addendum.replace("lagrangian_witness.py", "lagrangian_probe.py")
addendum = addendum.replace("test_lagrangian_witness.py", "test_lagrangian_probe.py")
addendum = addendum.replace("witness", "probe")
addendum = addendum.replace("Witness", "Probe")
addendum = addendum.replace(
    "rejects infeasible budgets",
    "warns on budget violations without rejecting training",
)
addendum = addendum.replace(
    "penalty contributions equal `normalized_cost_advantages * multipliers[None, :]`",
    "penalty contributions equal `raw_cost_advantages * multipliers[None, :]`",
)
addendum = addendum.replace(
    "raw costs, normalized cost advantages, frozen multipliers",
    "raw costs, raw cost advantages, optional standardized advantages for correlation only, frozen multipliers",
)
addendum_path.write_text(addendum)
