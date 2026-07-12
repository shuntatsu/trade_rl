from pathlib import Path


path = Path(__file__).resolve().parents[1] / "mars_lite/pipeline/residual_pipeline.py"
text = path.read_text(encoding="utf-8")

old_import = '''from mars_lite.eval.relative_evaluation import (
    _moving_block_mean_test,
    evaluate_relative_agent,
)
'''
new_import = '''from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.relative_evaluation import (
    _moving_block_mean_test,
    evaluate_relative_agent,
)
'''
if text.count(old_import) != 1:
    raise RuntimeError("unexpected residual pipeline import layout")
text = text.replace(old_import, new_import)

old_split = '''    train_fs = fs.slice(0, train_end)
    val_fs = fs.slice(val_start, val_end)
    test_fs = fs.slice(test_start, n)
'''
new_split = '''    trend_config = TrendFamilyConfig(
        base_timeframe=getattr(args, "base_timeframe", "1h")
    )
    trend_family = TrendFamily(trend_config)
    history_bars = max(
        trend_config.fast_lookback,
        trend_config.base_lookback,
        trend_config.slow_lookback,
    ) + trend_config.rebalance_every
    train_fs = fs.slice(0, train_end)
    val_window = with_history_context(
        fs, start=val_start, end=val_end, history_bars=history_bars
    )
    test_window = with_history_context(
        fs, start=test_start, end=n, history_bars=history_bars
    )
    val_fs = val_window.feature_set
    test_fs = test_window.feature_set
'''
if text.count(old_split) != 1:
    raise RuntimeError("unexpected residual pipeline split layout")
text = text.replace(old_split, new_split)

old_trend = '''    trend_family = TrendFamily(
        TrendFamilyConfig(base_timeframe=getattr(args, "base_timeframe", "1h"))
    )
'''
if text.count(old_trend) != 1:
    raise RuntimeError("unexpected duplicate trend family layout")
text = text.replace(old_trend, "")

old_baselines = '''        impact_rate=env_kwargs["impact_rate"],
    )
'''
new_baselines = '''        impact_rate=env_kwargs["impact_rate"],
        start_idx=test_window.start_idx,
    )
'''
if text.count(old_baselines) != 1:
    raise RuntimeError("unexpected baseline evaluation layout")
text = text.replace(old_baselines, new_baselines)

old_report = '''        "split": {
            "train_bars": train_fs.n_bars,
            "validation_bars": val_fs.n_bars,
            "test_bars": test_fs.n_bars,
        },
'''
new_report = '''        "split": {
            "train_bars": train_fs.n_bars,
            "validation_bars": val_window.scored_bars,
            "validation_context_bars": val_window.start_idx,
            "test_bars": test_window.scored_bars,
            "test_context_bars": test_window.start_idx,
        },
'''
if text.count(old_report) != 1:
    raise RuntimeError("unexpected split report layout")
text = text.replace(old_report, new_report)

old_view = '''        env = BaselineResidualTradingEnv(
            fs,
            episode_bars=fs.n_bars - 2,
            **env_kwargs,
        )
        obs, _ = env.reset(options={"start_idx": 0})
'''
new_view = '''        start_idx = int(getattr(fs, "_evaluation_start_idx", 0))
        env = BaselineResidualTradingEnv(
            fs,
            episode_bars=max(1, fs.n_bars - 2 - start_idx),
            **env_kwargs,
        )
        obs, _ = env.reset(options={"start_idx": start_idx})
'''
if text.count(old_view) != 1:
    raise RuntimeError("unexpected residual return view layout")
text = text.replace(old_view, new_view)

path.write_text(text, encoding="utf-8")
