from pathlib import Path

path = Path("trade_rl/integrations/sb3_environment.py")
text = path.read_text()

marker = "\n\ndef _build_parallel_sequence_training_environment(\n"
if "class _CompactFilteredEnvironmentFactory" not in text:
    if marker not in text:
        raise SystemExit("parallel sequence builder marker not found")
    block = r'''


class _CompactFilteredEnvironmentFactory:
    """Preserve worker-index routing while enabling compact sequence observations."""

    def __init__(self, factory: Callable[[], gym.Env[Any, Any]]) -> None:
        self.factory = factory

    def __call__(self) -> gym.Env[Any, Any]:
        return _compact_filtered_training_environment(self.factory)

    def for_environment_index(self, index: int) -> Callable[[], gym.Env[Any, Any]]:
        indexed = getattr(self.factory, "for_environment_index", None)
        selected = indexed(index) if callable(indexed) else self.factory
        return partial(_compact_filtered_training_environment, selected)


def _compact_filtered_environment_factory(
    factory: Callable[[], gym.Env[Any, Any]],
) -> _CompactFilteredEnvironmentFactory:
    return _CompactFilteredEnvironmentFactory(factory)
'''
    text = text.replace(marker, block + marker, 1)

old = '''    workers = _build_training_environment(\n        partial(_compact_filtered_training_environment, factory),\n        n_envs,\n        subprocesses=True,\n    )\n'''
new = '''    workers = _build_training_environment(\n        _compact_filtered_environment_factory(factory),\n        n_envs,\n        subprocesses=True,\n    )\n'''
if old in text:
    text = text.replace(old, new, 1)
elif "_compact_filtered_environment_factory(factory)" not in text:
    raise SystemExit("parallel sequence compact factory call not found")

path.write_text(text)
compile(text, str(path), "exec")
