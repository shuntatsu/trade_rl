from pathlib import Path

execution_path = Path("trade_rl/simulation/execution.py")
execution = execution_path.read_text(encoding="utf-8")
old_validation = '''        requirements = (
            (
                "tick_size",
                self.cost.tick_size,
                self.rule_stress.tick_size_factor > 1.0
                or self.rule_stress.adverse_tick_rounding,
            ),
            (
                "lot_size",
                self.cost.lot_size,
                self.rule_stress.lot_size_factor > 1.0,
            ),
            (
                "minimum_notional",
                self.cost.minimum_notional,
                self.rule_stress.minimum_notional_factor > 1.0,
            ),
        )
        for field_name, floor, required in requirements:
            if required and np.any(
                self._base_rule_array(field_name, floor=floor) <= 0.0
            ):
'''
new_validation = '''        requirements = (
            (
                "tick_size",
                self.rule_stress.tick_size_factor > 1.0
                or self.rule_stress.adverse_tick_rounding,
            ),
            ("lot_size", self.rule_stress.lot_size_factor > 1.0),
            (
                "minimum_notional",
                self.rule_stress.minimum_notional_factor > 1.0,
            ),
        )
        for field_name, required in requirements:
            source_rules = self.dataset.resolved_array(field_name)
            if required and np.any(source_rules <= 0.0):
'''
if old_validation not in execution:
    raise SystemExit("execution-rule validation anchor missing")
execution_path.write_text(
    execution.replace(old_validation, new_validation, 1), encoding="utf-8"
)

test_path = Path("tests/simulation/test_execution_sensitivity.py")
test = test_path.read_text(encoding="utf-8")
old_test = '''        MarketExecutor(
            dataset,
            ExecutionCostConfig.zero(),
            rule_stress=ExecutionRuleStress(name="lot_2x", lot_size_factor=2.0),
        )
'''
new_test = '''        MarketExecutor(
            dataset,
            ExecutionCostConfig(lot_size=0.05),
            rule_stress=ExecutionRuleStress(name="lot_2x", lot_size_factor=2.0),
        )
'''
if old_test not in test:
    raise SystemExit("zero-source regression-test anchor missing")
test_path.write_text(test.replace(old_test, new_test, 1), encoding="utf-8")
