from pathlib import Path

path = Path("trade_rl/integrations/sb3_environment.py")
text = path.read_text()
if "class _FilteredEnvironmentFactory:" not in text:
    marker = "\ndef _build_training_environment(\n"
    if marker not in text:
        raise SystemExit("SB3 environment build marker not found")
    block = '''

class _FilteredEnvironmentFactory:
    """Preserve optional worker-index routing through the training info filter."""

    def __init__(self, factory: Callable[[], gym.Env[Any, Any]]) -> None:
        self.factory = factory

    def __call__(self) -> gym.Env[Any, Any]:
        return _filtered_training_environment(self.factory)

    def for_environment_index(
        self, index: int
    ) -> Callable[[], gym.Env[Any, Any]]:
        indexed = getattr(self.factory, "for_environment_index", None)
        selected = indexed(index) if callable(indexed) else self.factory
        return partial(_filtered_training_environment, selected)


def _filtered_environment_factory(
    factory: Callable[[], gym.Env[Any, Any]],
) -> _FilteredEnvironmentFactory:
    return _FilteredEnvironmentFactory(factory)


def _environment_factories(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
) -> list[Callable[[], gym.Env[Any, Any]]]:
    indexed = getattr(factory, "for_environment_index", None)
    if callable(indexed):
        return [indexed(index) for index in range(n_envs)]
    return [factory for _ in range(n_envs)]
'''
    text = text.replace(marker, block + marker, 1)
text = text.replace(
    "    factories = [factory for _ in range(n_envs)]\n",
    "    factories = _environment_factories(factory, n_envs)\n",
    1,
)
path.write_text(text)
compile(text, str(path), "exec")

path = Path("trade_rl/integrations/sb3_training.py")
text = path.read_text()
if "_filtered_environment_factory as _filtered_environment_factory" not in text:
    marker = "from trade_rl.integrations.sb3_environment import (\n    _filtered_training_environment as _filtered_training_environment,\n)\n"
    if marker not in text:
        raise SystemExit("SB3 filtered environment import marker not found")
    replacement = marker + (
        "from trade_rl.integrations.sb3_environment import (\n"
        "    _filtered_environment_factory as _filtered_environment_factory,\n"
        ")\n"
    )
    text = text.replace(marker, replacement, 1)
old = (
    "                return _build_training_environment(\n"
    "                    lambda: _filtered_training_environment(self.environment_factory),\n"
    "                    config.n_envs,\n"
)
new = (
    "                return _build_training_environment(\n"
    "                    _filtered_environment_factory(self.environment_factory),\n"
    "                    config.n_envs,\n"
)
if old not in text:
    raise SystemExit("SB3 build_parallel_environment filter marker not found")
text = text.replace(old, new, 1)
path.write_text(text)
compile(text, str(path), "exec")
