"""
憲法テスト: docs/ARCHITECTURE.md の実測既定値をピン留めする。

このテストが落ちたら、既定値をうっかり変更したということ。
意図的な変更なら、同じベンチマークで根拠を示した上でARCHITECTURE.mdの
該当ベンチマーク台帳(§5)を更新し、このテストの期待値も合わせて更新すること。
"""

from mars_lite.config import RunConfig, DataConfig, FeatureConfig, EnvConfig, TrainConfig


def test_train_config_measured_defaults():
    """ARCHITECTURE.md §2.1 学習レシピの実測値"""
    cfg = TrainConfig()
    assert cfg.gamma == 0.5, "0.995では-26%→0.5で+46%に反転（実測）"
    assert cfg.gae_lambda == 0.9
    assert cfg.ent_coef == 0.002
    assert cfg.net_size == "small"
    assert cfg.n_steps == 256
    assert cfg.batch_size == 256
    assert cfg.n_epochs == 6
    assert cfg.bc_warmstart is True
    assert cfg.bc_teacher == "auto"
    assert cfg.ensemble == 1  # CLI既定は1。実データでは3推奨（別途明示指定）


def test_env_config_measured_defaults():
    """ARCHITECTURE.md §2.2 環境・報酬の実測値"""
    cfg = EnvConfig()
    assert cfg.lambda_turnover == 0.04
    assert cfg.reward_scale == 100.0
    assert cfg.fee_rate == 0.0005
    assert cfg.spread_rate == 0.0002
    assert cfg.impact_rate == 0.0001
    assert cfg.min_trade_delta == 0.04
    assert cfg.bars_per_year == 24 * 365


def test_postproc_config_measured_defaults():
    """ARCHITECTURE.md §2.3 後処理の実測値"""
    cfg = RunConfig().postproc
    assert cfg.ema_alpha == 1.0  # PostProcessConfig生の既定（make_default_processorが0.5に上書き）
    assert cfg.max_weight == 0.4
    assert cfg.dd_derisk_start == 0.10
    assert cfg.dd_derisk_floor == 0.3
    assert cfg.disagreement_penalty == 1.0


def test_feature_config_defaults():
    cfg = FeatureConfig()
    assert cfg.horizon == 4
    assert cfg.feature_norm == "none"
    assert cfg.feature_mask is False


def test_run_config_json_roundtrip():
    rc = RunConfig()
    rc.train.gamma = 0.7
    rc.data.symbols = ["BTCUSDT", "ETHUSDT"]
    d = rc.to_dict()
    rc2 = RunConfig.from_dict(d)
    assert rc2.train.gamma == 0.7
    assert rc2.data.symbols == ["BTCUSDT", "ETHUSDT"]
    assert rc2.env.lambda_turnover == rc.env.lambda_turnover
    assert rc2.postproc.dd_derisk_start == rc.postproc.dd_derisk_start
    assert rc2.serve.guardrails.max_data_age_hours == rc.serve.guardrails.max_data_age_hours


def test_run_config_from_dict_ignores_unknown_keys():
    d = RunConfig().to_dict()
    d["data"]["unknown_field"] = "should be ignored"
    rc = RunConfig.from_dict(d)
    assert rc.data.source == "synthetic"
