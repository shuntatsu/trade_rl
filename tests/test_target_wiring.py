"""
--target / --beta-neutral / --warmup-days / --base-timeframe のCLI配線テスト

gate1_diagnostic.py が「絶対リターン(raw)には信号が無いが市場中立の相対
アルファ(cs_demean)には有意な信号がある」と判定した場合に、その target を
学習パイプライン全体（ICゲート判定・Ridge教師）へ流し込めることを保証する。
過去にこれらのフラグがCLIから欠落し、run_signal_check/ridge_teacher が
target を受け取れるのに常に既定(raw)で呼ばれていた回帰を防ぐ。

--base-timeframe は同様に診断で「1hでは不合格だが4hなら合格」だった場合に
実際に4h足で学習パイプラインを回すための配線。合わせて④ボラターゲティングの
年率換算(bars_per_year)がbase_timeframeの実バー数に追従することも検証する
（1h想定のまま4h等を使うと推定年率ボラが水増しされる回帰を防ぐ）。
"""

import inspect

from mars_lite.pipeline.cli import build_parser


def test_cli_exposes_target_beta_neutral_warmup():
    parser = build_parser()
    args = parser.parse_args(
        ["--target", "cs_demean", "--beta-neutral", "--warmup-days", "100"]
    )
    assert args.target == "cs_demean"
    assert args.beta_neutral is True
    assert args.warmup_days == 100


def test_target_defaults_to_raw():
    args = build_parser().parse_args([])
    assert args.target == "raw"
    assert args.beta_neutral is False
    assert args.warmup_days == 0


def test_target_choices_restricted():
    parser = build_parser()
    for good in ("raw", "cs_demean", "vol_norm"):
        assert parser.parse_args(["--target", good]).target == good


def test_train_ppo_accepts_signal_target():
    """evaluator が使う training_engine.train_ppo が signal_target を受ける"""
    from mars_lite.pipeline.training_engine import train_ppo

    assert "signal_target" in inspect.signature(train_ppo).parameters


def test_build_post_processor_threads_beta_neutral():
    from types import SimpleNamespace

    from mars_lite.pipeline.training_engine import build_post_processor

    pp = build_post_processor(
        SimpleNamespace(postproc="full", target_vol=0.5, beta_neutral=True)
    )
    assert pp.cfg.beta_neutral is True
    pp_off = build_post_processor(
        SimpleNamespace(postproc="full", target_vol=0.5, beta_neutral=False)
    )
    assert pp_off.cfg.beta_neutral is False


def test_cli_exposes_base_timeframe():
    parser = build_parser()
    assert parser.parse_args([]).base_timeframe == "1h"
    assert parser.parse_args(["--base-timeframe", "4h"]).base_timeframe == "4h"
    for good in ("15m", "1h", "4h", "1d"):
        assert parser.parse_args(["--base-timeframe", good]).base_timeframe == good


def test_build_post_processor_scales_bars_per_year_with_base_timeframe():
    """4h等に切り替えたときtarget_volの年率換算が1h想定のままにならない
    （水増しされた推定ボラでグロスを過剰に絞る回帰を防ぐ）"""
    from types import SimpleNamespace

    from mars_lite.pipeline.training_engine import build_post_processor

    pp_1h = build_post_processor(
        SimpleNamespace(postproc="full", target_vol=0.5, base_timeframe="1h")
    )
    assert pp_1h.cfg.bars_per_year == 24 * 365

    pp_4h = build_post_processor(
        SimpleNamespace(postproc="full", target_vol=0.5, base_timeframe="4h")
    )
    assert pp_4h.cfg.bars_per_year == 6 * 365

    pp_default = build_post_processor(SimpleNamespace(postproc="full", target_vol=0.5))
    assert pp_default.cfg.bars_per_year == 24 * 365


def test_build_feature_set_threads_base_timeframe(monkeypatch):
    """dataset_builder.build_feature_set がFeaturePipelineへbase_timeframeを渡す"""
    from types import SimpleNamespace

    import mars_lite.pipeline.dataset_builder as dataset_builder

    captured = {}
    real_init = dataset_builder.FeaturePipeline.__init__

    def spy_init(self, symbols, base_timeframe="1h", **kwargs):
        captured["base_timeframe"] = base_timeframe
        real_init(self, symbols, base_timeframe=base_timeframe, **kwargs)

    monkeypatch.setattr(dataset_builder.FeaturePipeline, "__init__", spy_init)

    args = SimpleNamespace(
        source="synthetic",
        days=30,
        alpha="none",
        alpha_strength=0.002,
        seed=0,
        base_timeframe="4h",
    )
    dataset_builder.build_feature_set(args)
    assert captured["base_timeframe"] == "4h"
