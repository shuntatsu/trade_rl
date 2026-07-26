from __future__ import annotations

from pathlib import Path


def read(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines))


def first_index(lines: list[str], needle: str, *, start: int = 0) -> int:
    for index in range(start, len(lines)):
        if lines[index] == needle:
            return index
    raise RuntimeError(f"anchor missing: {needle.rstrip()!r}")


def patch_model() -> None:
    path = Path("trade_rl/integrations/lagrangian_ppo.py")
    lines = read(path)
    if not any("CanonicalActionProbeEvidence" in line for line in lines):
        index = first_index(
            lines,
            "from trade_rl.rl.lagrangian_advantages import combine_lagrangian_advantages\n",
        )
        lines[index + 1 : index + 1] = [
            "from trade_rl.rl.lagrangian_probe import CanonicalActionProbeEvidence\n"
        ]

    if not any("canonical_action_probe_evidence:" in line for line in lines):
        index = first_index(
            lines,
            "        lagrangian_schema: LagrangianSchema | None = None,\n",
        )
        lines[index + 1 : index + 1] = [
            "        canonical_action_probe_evidence: (\n",
            "            CanonicalActionProbeEvidence | None\n",
            "        ) = None,\n",
        ]

    if not any("self.canonical_action_probe_evidence =" in line for line in lines):
        index = first_index(lines, "        self.lagrangian_schema = resolved_schema\n")
        lines[index:index] = [
            "        if canonical_action_probe_evidence is not None and not isinstance(\n",
            "            canonical_action_probe_evidence, CanonicalActionProbeEvidence\n",
            "        ):\n",
            "            raise TypeError(\n",
            "                \"canonical_action_probe_evidence has an invalid type\"\n",
            "            )\n",
            "        self.canonical_action_probe_evidence = (\n",
            "            canonical_action_probe_evidence\n",
            "        )\n",
        ]

    if not any("probe_evidence = getattr(" in line for line in lines):
        index = first_index(
            lines,
            "        reports = getattr(self, \"last_dual_update_reports\", None)\n",
        )
        end = first_index(
            lines,
            "            self.last_dual_update_reports = {}\n",
            start=index,
        )
        lines[end + 1 : end + 1] = [
            "        probe_evidence = getattr(\n",
            "            self, \"canonical_action_probe_evidence\", None\n",
            "        )\n",
            "        if probe_evidence is not None and not isinstance(\n",
            "            probe_evidence, CanonicalActionProbeEvidence\n",
            "        ):\n",
            "            raise TypeError(\n",
            "                \"canonical_action_probe_evidence has an invalid type\"\n",
            "            )\n",
        ]

    if not any(
        line == "        probe_evidence = self.canonical_action_probe_evidence\n"
        for line in lines
    ):
        index = first_index(
            lines,
            "        payload = super().checkpoint_identity_payload()\n",
        )
        lines[index:index] = [
            "        probe_evidence = self.canonical_action_probe_evidence\n"
        ]

    if not any('"canonical_action_probe_digest"' in line for line in lines):
        index = first_index(
            lines,
            "                \"controller_state_version\": self.lagrangian_controller.state_version,\n",
        )
        lines[index + 1 : index + 1] = [
            "                \"canonical_action_probe\": (\n",
            "                    None\n",
            "                    if probe_evidence is None\n",
            "                    else probe_evidence.digest_payload()\n",
            "                ),\n",
            "                \"canonical_action_probe_digest\": (\n",
            "                    None if probe_evidence is None else probe_evidence.digest\n",
            "                ),\n",
        ]
    write(path, lines)


def patch_backend() -> None:
    path = Path("trade_rl/integrations/sb3_training.py")
    lines = read(path)
    if not any("canonical_action_probe_evidence = None" in line for line in lines):
        index = first_index(
            lines,
            "            algorithm_config = build_algorithm_config(config)\n",
        )
        lines[index + 1 : index + 1] = [
            "            canonical_action_probe_evidence = None\n",
            "            if isinstance(algorithm_config, LagrangianPPOConfig):\n",
            "                from trade_rl.rl.lagrangian_probe import (\n",
            "                    run_canonical_action_feasibility_probe,\n",
            "                )\n",
            "\n",
            "                canonical_action_probe_evidence = (\n",
            "                    run_canonical_action_feasibility_probe(\n",
            "                        environment_factory=self.environment_factory,\n",
            "                        schema=algorithm_config.lagrangian_schema,\n",
            "                        episode_count=algorithm_config.probe_episodes,\n",
            "                        max_steps_per_episode=(\n",
            "                            algorithm_config.probe_max_steps_per_episode\n",
            "                        ),\n",
            "                    )\n",
            "                )\n",
        ]

    if not any("canonical_action_probe_evidence=(" in line for line in lines):
        index = first_index(
            lines,
            "                        lagrangian_schema=algorithm_config.lagrangian_schema,\n",
        )
        lines[index + 1 : index + 1] = [
            "                        canonical_action_probe_evidence=(\n",
            "                            canonical_action_probe_evidence\n",
            "                        ),\n",
        ]

    if not any("model.canonical_action_probe_evidence =" in line for line in lines):
        index = first_index(
            lines,
            "                elif isinstance(algorithm_config, CostCriticPPOConfig):\n",
        )
        lines[index:index] = [
            "                    model.canonical_action_probe_evidence = (\n",
            "                        canonical_action_probe_evidence\n",
            "                    )\n",
        ]

    if not any('architecture_details["lagrangian_probe"]' in line for line in lines):
        schema_index = first_index(
            lines,
            "                    \"schema_digest\": algorithm_config.lagrangian_schema.digest,\n",
        )
        close_index = first_index(lines, "                }\n", start=schema_index)
        lines[close_index + 1 : close_index + 1] = [
            "                if canonical_action_probe_evidence is None:\n",
            "                    raise RuntimeError(\n",
            "                        \"canonical action probe evidence is unavailable\"\n",
            "                    )\n",
            "                architecture_details[\"lagrangian_probe\"] = {\n",
            "                    \"digest\": canonical_action_probe_evidence.digest,\n",
            "                    \"payload\": (\n",
            "                        canonical_action_probe_evidence.digest_payload()\n",
            "                    ),\n",
            "                    \"violated_costs\": list(\n",
            "                        canonical_action_probe_evidence.violated_costs\n",
            "                    ),\n",
            "                    \"warning\": canonical_action_probe_evidence.warning,\n",
            "                }\n",
        ]
    write(path, lines)


patch_model()
patch_backend()
