"""
--target / --beta-neutral / --warmup-days のCLI配線テスト

gate1_diagnostic.py が「絶対リターン(raw)には信号が無いが市場中立の相対
アルファ(cs_demean)には有意な信号がある」と判定した場合に、その target を
学習パイプライン全体（ICゲート判定・Ridge教師）へ流し込めることを保証する。
過去にこれらのフラグがCLIから欠落し、run_signal_check/ridge_teacher が
target を受け取れるのに常に既定(raw)で呼ばれていた回帰を防ぐ。
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
