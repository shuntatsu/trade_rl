from pathlib import Path

path = Path("trade_rl/learning/episode_oracle_teacher.py")
text = path.read_text()
if "maximum_stop_index: int | None = None" in text:
    raise SystemExit(0)

old_signature = '''def sample_oracle_episode_contracts(\n    dataset: MarketDataset,\n    *,\n    minimum_start_index: int,\n    config: OracleEpisodeSamplingConfig,\n    initial_weight_provider: InitialWeightProvider | None = None,\n) -> tuple[OracleEpisodeContract, ...]:\n'''
new_signature = '''def sample_oracle_episode_contracts(\n    dataset: MarketDataset,\n    *,\n    minimum_start_index: int,\n    maximum_stop_index: int | None = None,\n    config: OracleEpisodeSamplingConfig,\n    initial_weight_provider: InitialWeightProvider | None = None,\n) -> tuple[OracleEpisodeContract, ...]:\n'''
if old_signature not in text:
    raise SystemExit("sample_oracle_episode_contracts signature not found")
text = text.replace(old_signature, new_signature, 1)

old_range = '''    maximum_start = dataset.n_bars - config.episode_bars - 1\n    if maximum_start < minimum_start_index:\n        raise ValueError("dataset does not contain a complete episode horizon")\n'''
new_range = '''    if maximum_stop_index is None:\n        resolved_stop = dataset.n_bars\n    else:\n        if (\n            isinstance(maximum_stop_index, bool)\n            or not isinstance(maximum_stop_index, int)\n            or maximum_stop_index <= minimum_start_index\n            or maximum_stop_index > dataset.n_bars\n        ):\n            raise ValueError("maximum_stop_index is outside the dataset train range")\n        resolved_stop = maximum_stop_index\n    maximum_start = resolved_stop - config.episode_bars - 1\n    if maximum_start < minimum_start_index:\n        raise ValueError("dataset train range does not contain a complete episode horizon")\n'''
if old_range not in text:
    raise SystemExit("episode range block not found")
text = text.replace(old_range, new_range, 1)

old_batch_signature = '''def build_episode_oracle_batch(\n    dataset: MarketDataset,\n    *,\n    minimum_start_index: int,\n    sampling_config: OracleEpisodeSamplingConfig,\n'''
new_batch_signature = '''def build_episode_oracle_batch(\n    dataset: MarketDataset,\n    *,\n    minimum_start_index: int,\n    maximum_stop_index: int | None = None,\n    sampling_config: OracleEpisodeSamplingConfig,\n'''
if old_batch_signature not in text:
    raise SystemExit("build_episode_oracle_batch signature not found")
text = text.replace(old_batch_signature, new_batch_signature, 1)

old_call = '''    contracts = sample_oracle_episode_contracts(\n        dataset,\n        minimum_start_index=minimum_start_index,\n        config=sampling_config,\n        initial_weight_provider=initial_weight_provider,\n    )\n'''
new_call = '''    contracts = sample_oracle_episode_contracts(\n        dataset,\n        minimum_start_index=minimum_start_index,\n        maximum_stop_index=maximum_stop_index,\n        config=sampling_config,\n        initial_weight_provider=initial_weight_provider,\n    )\n'''
if old_call not in text:
    raise SystemExit("sample_oracle_episode_contracts call not found")
text = text.replace(old_call, new_call, 1)

path.write_text(text)
compile(text, str(path), "exec")
