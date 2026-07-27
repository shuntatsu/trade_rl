from __future__ import annotations

import ast
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    updated = text.replace(old, new, 1)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def _normalize_gpu_smoke() -> None:
    _replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''            "observation_encoder": "invalid_legacy_combination"
            if (True) and (False)
            else "hierarchical_sequence_v2"
            if (True)
            else "asset_set"
            if (False)
            else "flat_mlp",
''',
        '''            "observation_encoder": "hierarchical_sequence_v2",
''',
    )


def _enable_sequence_bc_to_ppo_audit() -> None:
    path = "tools/run_training_capability_audit.py"
    _replace_once(
        path,
        '''        sequence_dropout=0.0,
        max_policy_parameters=2_000_000,
        device="cpu",
''',
        '''        sequence_dropout=0.0,
        max_policy_parameters=2_000_000,
        behavior_cloning_epochs=1,
        behavior_cloning_batch_size=16,
        behavior_cloning_validation_fraction=0.1,
        device="cpu",
''',
    )
    _replace_once(
        path,
        '''    if architecture["architecture"].get("encoder") != "MultiTimeframeTCNEncoder":
        raise RuntimeError("structured sequence encoder was not instantiated")
    return {
''',
        '''    if architecture["architecture"].get("encoder") != "MultiTimeframeTCNEncoder":
        raise RuntimeError("structured sequence encoder was not instantiated")
    behavior_cloning_path = output.parent / "behavior-cloning.json"
    if not behavior_cloning_path.is_file():
        raise RuntimeError("structured sequence behavior cloning evidence is missing")
    behavior_cloning = json.loads(behavior_cloning_path.read_text(encoding="utf-8"))
    for field in ("initial_mse", "final_mse"):
        if not np.isfinite(float(behavior_cloning[field])):
            raise RuntimeError(
                f"structured sequence behavior cloning {field} is invalid"
            )
    return {
''',
    )
    _replace_once(
        path,
        '''        "actual_timesteps": result.actual_timesteps,
        "observation_schema": result.observation_schema,
''',
        '''        "actual_timesteps": result.actual_timesteps,
        "behavior_cloning": {
            "final_mse": behavior_cloning["final_mse"],
            "initial_mse": behavior_cloning["initial_mse"],
            "sample_count": behavior_cloning["sample_count"],
            "status": "pass",
        },
        "observation_schema": result.observation_schema,
''',
    )


def _constant_bool(node: ast.expr) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _constant_bool(node.operand)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_constant_bool(value) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is False for value in values):
                return False
            if all(value is True for value in values):
                return True
            return None
        if isinstance(node.op, ast.Or):
            if any(value is True for value in values):
                return True
            if all(value is False for value in values):
                return False
            return None
    return None


class _ConstantIfSimplifier(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        node = self.generic_visit(node)
        if not isinstance(node, ast.IfExp):
            return node
        condition = _constant_bool(node.test)
        if condition is None:
            return node
        self.changed = True
        chosen = node.body if condition else node.orelse
        return ast.copy_location(chosen, node)


def _resolved_expression(node: ast.IfExp) -> ast.expr | None:
    simplifier = _ConstantIfSimplifier()
    resolved = simplifier.visit(copy.deepcopy(node))
    if not simplifier.changed or not isinstance(resolved, ast.expr):
        return None
    return ast.fix_missing_locations(resolved)


def _char_column(line: str, byte_column: int) -> int:
    return len(line.encode("utf-8")[:byte_column].decode("utf-8"))


def _absolute_offset(lines: list[str], line_number: int, byte_column: int) -> int:
    return sum(len(line) for line in lines[: line_number - 1]) + _char_column(
        lines[line_number - 1], byte_column
    )


def _simplify_constant_ternaries(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines(keepends=True)
    candidates: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        resolved = _resolved_expression(node)
        if resolved is None or node.end_lineno is None or node.end_col_offset is None:
            continue
        start = _absolute_offset(lines, node.lineno, node.col_offset)
        end = _absolute_offset(lines, node.end_lineno, node.end_col_offset)
        candidates.append((start, end, ast.unparse(resolved)))
    candidates.sort(key=lambda item: (item[0], -item[1]))
    selected: list[tuple[int, int, str]] = []
    for candidate in candidates:
        start, end, _replacement = candidate
        if any(
            start >= outer_start and end <= outer_end
            for outer_start, outer_end, _ in selected
        ):
            continue
        selected.append(candidate)
    if not selected:
        return 0
    updated = text
    for start, end, replacement in sorted(selected, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    ast.parse(updated, filename=str(path))
    path.write_text(updated, encoding="utf-8")
    return len(selected)


def _normalize_migrated_test_expressions() -> None:
    changed = 0
    for root_name in ("tests", "examples", "tools"):
        for path in sorted((ROOT / root_name).rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            changed += _simplify_constant_ternaries(path)
    if changed <= 0:
        raise RuntimeError("expected migrated constant ternaries to be normalized")


def main() -> None:
    _normalize_gpu_smoke()
    _enable_sequence_bc_to_ppo_audit()
    _normalize_migrated_test_expressions()


if __name__ == "__main__":
    main()
