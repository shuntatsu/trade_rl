"""レジーム・ハイブリッドルーター（mars_lite/learning/regime_router.py）のテスト"""

import json

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.regime_taxonomy import FINE_REGIMES, label_fine_regimes
from mars_lite.learning.regime_router import (
    RouterTable,
    _confirm_series,
    derive_router_table,
    make_router_weight_fn,
    regime_contributions,
)


def _fs(days=60, alpha="none"):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=0)
    return FeaturePipeline(src.symbols).build(src)


class TestConfirmSeries:
    def test_stable_series_unchanged(self):
        labels = np.array(["a", "a", "a", "a"], dtype=object)
        out = _confirm_series(labels, confirm_bars=2)
        assert list(out) == ["a", "a", "a", "a"]

    def test_single_bar_blip_is_suppressed(self):
        """1本だけの切り替わりはconfirm_bars=2で確定しない"""
        labels = np.array(["a", "a", "b", "a", "a"], dtype=object)
        out = _confirm_series(labels, confirm_bars=2)
        # "b"が1本しか続かないので"a"のまま
        assert list(out) == ["a", "a", "a", "a", "a"]

    def test_sustained_switch_confirms_after_n_bars(self):
        labels = np.array(["a", "b", "b", "b", "b"], dtype=object)
        out = _confirm_series(labels, confirm_bars=2)
        # index0:a / index1:b(候補1本目、未確定なのでa継続) / index2:b(候補2本目=確定→b)
        assert list(out) == ["a", "a", "b", "b", "b"]

    def test_causal_prefix_matches(self):
        """先頭プレフィックスだけで計算しても全系列計算の同区間と一致（未来リーク無し）"""
        rng = np.random.default_rng(0)
        labels = rng.choice(["a", "b", "c"], size=200).astype(object)
        full = _confirm_series(labels, confirm_bars=3)
        prefix = _confirm_series(labels[:80], confirm_bars=3)
        assert list(full[:80]) == list(prefix)


class TestRegimeContributions:
    def test_bar_accounting_partitions_segment(self):
        """各レジームのn_barsの合計はセグメント長と一致する（重複・欠落なし）"""
        fs = _fs()
        labels = label_fine_regimes(fs)
        from mars_lite.learning.baselines import flat_strategy

        start, end = 100, 400
        contrib = regime_contributions(fs, flat_strategy, labels, start, end)
        total = sum(v["n_bars"] for v in contrib.values())
        assert total == end - start

    def test_flat_strategy_has_zero_growth(self):
        """flat_strategyは常に無取引なので、どのレジームでも寄与growth=0"""
        fs = _fs()
        labels = label_fine_regimes(fs)
        from mars_lite.learning.baselines import flat_strategy

        contrib = regime_contributions(fs, flat_strategy, labels, 100, 400)
        for r in FINE_REGIMES:
            assert contrib[r]["growth"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_segment_returns_zeroed_dict(self):
        fs = _fs()
        labels = label_fine_regimes(fs)
        from mars_lite.learning.baselines import flat_strategy

        contrib = regime_contributions(fs, flat_strategy, labels, 10, 11)
        assert all(v["n_bars"] == 0 for v in contrib.values())


class TestDeriveRouterTable:
    def test_all_regimes_default_tf_when_sample_too_small(self):
        """min_regime_barsを極端に大きくすると全レジームが既定"tf"になる"""
        fs = _fs()
        labels = label_fine_regimes(fs)
        table = derive_router_table(fs, labels, end=fs.n_bars, min_regime_bars=10**9)
        assert set(table.assignments.keys()) == set(FINE_REGIMES)
        assert all(v == "tf" for v in table.assignments.values())
        assert all(
            table.derivation[r]["reason"] == "insufficient_sample" for r in FINE_REGIMES
        )

    def test_table_only_uses_data_before_end(self):
        """endより後のデータを変えても、endより前のデータが同じなら表は変わらない"""
        fs = _fs(days=120)
        labels = label_fine_regimes(fs)
        end = fs.n_bars // 2
        table_full = derive_router_table(fs, labels, end=end, min_regime_bars=50)
        fs_truncated = fs.slice(0, end)
        labels_truncated = label_fine_regimes(fs_truncated)
        table_truncated = derive_router_table(
            fs_truncated, labels_truncated, end=end, min_regime_bars=50
        )
        assert table_full.assignments == table_truncated.assignments

    def test_assignments_are_valid_values(self):
        fs = _fs(days=120)
        labels = label_fine_regimes(fs)
        table = derive_router_table(fs, labels, end=fs.n_bars, min_regime_bars=50)
        assert all(v in ("tf", "flat") for v in table.assignments.values())


class TestRouterTableSerialization:
    def test_roundtrip_via_dict(self):
        table = RouterTable(
            assignments={r: "tf" for r in FINE_REGIMES},
            labeler_params={
                "trend_threshold": 0.5,
                "vol_threshold": 0.0,
                "age_bars": 24,
            },
            derivation={"trend_up_early": {"reason": "default_or_positive"}},
        )
        d = table.to_dict()
        restored = RouterTable.from_dict(d)
        assert restored.assignments == table.assignments
        assert restored.labeler_params == table.labeler_params
        assert restored.confirm_bars == table.confirm_bars

    def test_roundtrip_via_file(self, tmp_path):
        table = RouterTable(
            assignments={r: "flat" for r in FINE_REGIMES},
            labeler_params={
                "trend_threshold": 0.5,
                "vol_threshold": 0.0,
                "age_bars": 24,
            },
        )
        path = tmp_path / "router_config.json"
        table.save(path)
        restored = RouterTable.load(path)
        assert restored.assignments == table.assignments
        # JSONとして妥当であることも確認
        json.loads(path.read_text(encoding="utf-8"))


class TestMakeRouterWeightFn:
    def test_flat_regime_dispatches_to_flat(self):
        fs = _fs()
        table = RouterTable(
            assignments={r: "flat" for r in FINE_REGIMES},
            labeler_params={},
        )
        wf = make_router_weight_fn(fs, table)
        w = wf(fs, 50, np.zeros(fs.n_symbols))
        assert np.allclose(w, 0.0)

    def test_tf_regime_matches_trend_following_when_no_switch(self):
        """切替バーでない限り、'tf'割当はtrend_following_strategyと同じ出力"""
        from mars_lite.learning.baselines import trend_following_strategy

        fs = _fs()
        table = RouterTable(
            assignments={r: "tf" for r in FINE_REGIMES},
            labeler_params={},
        )
        wf = make_router_weight_fn(fs, table)
        prev = np.zeros(fs.n_symbols)
        # 全レジームがtfなので確定系列は変化せず、どのtでも切替は発生しない
        for t in [30, 60, 90]:
            expected = trend_following_strategy(fs, t, prev)
            actual = wf(fs, t, prev)
            np.testing.assert_allclose(actual, expected)

    def test_forced_rebalance_on_regime_switch(self):
        """レジーム切替バーでは下位戦略にw=zerosが渡される（prevではない）"""

        def echo_strategy(fs_, t, w):
            # wをそのまま返す（w=0で呼ばれたかどうかを判定できるようにする）
            return w.copy()

        fs = _fs()
        n = fs.n_symbols
        # 2レジームだけ使い、片方をtf(echo_strategyに差し替え不可のため
        # ここではflat以外の唯一の選択肢がtfなので、確実に切替を起こすため
        # ラベルを手動で構成する代わりにconfirm_bars=1で即時切替させる
        table = RouterTable(
            assignments={
                FINE_REGIMES[0]: "flat",
                **{r: "flat" for r in FINE_REGIMES[1:]},
            },
            labeler_params={},
            confirm_bars=1,
        )
        # 全部flatだと切替検知の意味がないので、1レジームだけ"specialist"にして
        # echo_strategyを割り当てる
        table.assignments[FINE_REGIMES[0]] = "specialist"
        specialists = {FINE_REGIMES[0]: echo_strategy}
        wf = make_router_weight_fn(fs, table, specialists=specialists)

        raw_labels = label_fine_regimes(fs)
        # 実際に切替が起きるバーを探す（specialist regime <-> flat の境界）
        confirmed_target = FINE_REGIMES[0]
        switch_t = None
        for t in range(1, len(raw_labels) - 1):
            if (
                raw_labels[t - 1] != confirmed_target
                and raw_labels[t] == confirmed_target
            ):
                switch_t = t
                break
        if switch_t is None:
            pytest.skip("このseedでは対象レジームへの切替バーが見つからなかった")

        nonzero_prev = np.full(n, 0.3)
        result = wf(fs, switch_t, nonzero_prev)
        # echo_strategyがw=zerosで呼ばれていれば結果はゼロになるはず
        assert np.allclose(result, 0.0), (
            "切替バーでprevウェイトがそのまま素通しされている"
        )

    def test_no_forced_rebalance_when_underlying_strategy_unchanged(self):
        """
        レジームラベルが変わっても、割当先の戦略が同じ(tf->tf等)なら
        強制リバランス(w=zeros呼び出し)は起きない。

        実データ検証で発見したバグ: 生ラベルの変化だけで強制リバランスすると、
        全レジームが同一戦略に割り当てられているfoldでも回転率が
        trend_following単体の約1.9倍になっていた（不要な往復コスト）。
        """

        def echo_strategy(fs_, t, w):
            return w.copy()

        fs = _fs()
        n = fs.n_symbols
        # 全レジームを同じspecialist(echo_strategy)に割り当てる
        table = RouterTable(
            assignments={r: "specialist" for r in FINE_REGIMES},
            labeler_params={},
            confirm_bars=1,
        )
        specialists = {r: echo_strategy for r in FINE_REGIMES}
        wf = make_router_weight_fn(fs, table, specialists=specialists)

        raw_labels = label_fine_regimes(fs)
        # レジームラベルが実際に変化するバーを探す
        switch_t = None
        for t in range(1, len(raw_labels) - 1):
            if raw_labels[t - 1] != raw_labels[t]:
                switch_t = t
                break
        assert switch_t is not None, "テストデータにレジーム変化が無い"

        nonzero_prev = np.full(n, 0.3)
        result = wf(fs, switch_t, nonzero_prev)
        # echo_strategyが実際のprevウェイトで呼ばれていれば、結果はprevと一致する
        np.testing.assert_allclose(result, nonzero_prev)
