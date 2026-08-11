from pathlib import Path

entrypoint_path = Path("trade_rl/workflows/universal_full_research_entrypoint.py")
entrypoint = entrypoint_path.read_text()
old = '''    algorithm_configs: dict[FullResearchAlgorithm, ResidualTrainingConfig] = {
        algorithm: config.training for algorithm, config in configs.items()
    }
'''
new = '''    algorithm_configs: Mapping[
        FullResearchAlgorithm | str, ResidualTrainingConfig
    ] = {algorithm: config.training for algorithm, config in configs.items()}
'''
if old not in entrypoint:
    raise SystemExit("entrypoint algorithm_configs anchor not found")
entrypoint = entrypoint.replace(old, new, 1)
entrypoint_path.write_text(entrypoint)
compile(entrypoint, str(entrypoint_path), "exec")

script_path = Path("scripts/run_universal_full_research.py")
script = script_path.read_text()
if "from collections.abc import Mapping, Sequence\n" not in script:
    script = script.replace(
        "import json\nfrom pathlib import Path\nfrom typing import Sequence\n",
        "import json\nfrom collections.abc import Mapping, Sequence\nfrom pathlib import Path\n",
        1,
    )
if "from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime\n" not in script:
    script = script.replace(
        "from trade_rl.workflows.universal_research import FullResearchAlgorithm\n",
        "from trade_rl.workflows.universal_research import FullResearchAlgorithm\n"
        "from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime\n",
        1,
    )
old_configs = '''    authored_configs = {
        FullResearchAlgorithm.PPO: TrainingRunConfig.from_json(args.ppo_config),
        FullResearchAlgorithm.LAGRANGIAN: TrainingRunConfig.from_json(
            args.lagrangian_config
        ),
        FullResearchAlgorithm.DISCOUNTED: TrainingRunConfig.from_json(
            args.discounted_config
        ),
    }
'''
new_configs = '''    authored_configs: Mapping[
        FullResearchAlgorithm | str, TrainingRunConfig
    ] = {
        FullResearchAlgorithm.PPO: TrainingRunConfig.from_json(args.ppo_config),
        FullResearchAlgorithm.LAGRANGIAN: TrainingRunConfig.from_json(
            args.lagrangian_config
        ),
        FullResearchAlgorithm.DISCOUNTED: TrainingRunConfig.from_json(
            args.discounted_config
        ),
    }
'''
if old_configs not in script:
    raise SystemExit("CLI authored_configs anchor not found")
script = script.replace(old_configs, new_configs, 1)
old_factory = '''    def runtime_factory(*, algorithm, run_config):
        return raw_runtime_factory(
'''
new_factory = '''    def runtime_factory(
        *,
        algorithm: FullResearchAlgorithm,
        run_config: TrainingRunConfig,
    ) -> UniversalTrainingRuntime:
        return raw_runtime_factory(
'''
if old_factory not in script:
    raise SystemExit("CLI runtime_factory anchor not found")
script = script.replace(old_factory, new_factory, 1)
script_path.write_text(script)
compile(script, str(script_path), "exec")
