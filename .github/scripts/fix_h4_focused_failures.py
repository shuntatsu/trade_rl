from __future__ import annotations

from pathlib import Path


def replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{seam} changed: expected one exact match, got {count}")
    return source.replace(old, new)


sb3_path = Path("trade_rl/integrations/sb3_training.py")
sb3 = sb3_path.read_text(encoding="utf-8")
sb3 = replace_once(
    sb3,
    '''def _effective_vector_environment_kind(config: ResidualTrainingConfig) -> str:
    if config.n_envs == 1:
        return "direct"
    if config.vector_environment_mode == "in_process":
        return "in_process"
    if config.sequence_encoder:
        return (
            "subprocess_compact_sequence"
            if config.vector_environment_mode == "subprocess"
            else "in_process"
        )
    return "subprocess"
''',
    '''def _effective_vector_environment_kind(config: ResidualTrainingConfig) -> str:
    if config.n_envs == 1:
        return "direct"
    if config.vector_environment_mode != "subprocess":
        return "in_process"
    if config.sequence_encoder:
        return "subprocess_compact_sequence"
    return "subprocess"
''',
    seam="effective vector environment compatibility",
)
sb3_path.write_text(sb3, encoding="utf-8")


parallel_path = Path("trade_rl/integrations/parallel_sequence_env.py")
parallel = parallel_path.read_text(encoding="utf-8")
parallel = replace_once(
    parallel,
    '''        if not isinstance(reconstructor, SequenceRolloutReconstructor):
            raise TypeError("parallel sequence environment reconstructor is invalid")
        self.reconstructor = reconstructor
''',
    '''        if not callable(getattr(reconstructor, "reconstruct", None)):
            raise TypeError("parallel sequence environment reconstructor is invalid")
        self.reconstructor = reconstructor
''',
    seam="reconstructor protocol validation",
)
parallel_path.write_text(parallel, encoding="utf-8")


test_path = Path("tests/integrations/test_parallel_sequence_environments.py")
test = test_path.read_text(encoding="utf-8")
test = replace_once(
    test,
    '''    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        del attr_name, indices
        return []
''',
    '''    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        del indices
        if attr_name == "render_mode":
            return [None, None]
        return []
''',
    seam="fake VecEnv render mode",
)
test = replace_once(
    test,
    '''        (4, False, "auto", "subprocess"),
''',
    '''        (4, False, "auto", "in_process"),
''',
    seam="auto vector mode expectation",
)
test_path.write_text(test, encoding="utf-8")
