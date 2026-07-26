from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text()
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement point, found {count}")
    path.write_text(content.replace(old, new))


diagnostics_path = Path("trade_rl/rl/lagrangian_diagnostics.py")
content = diagnostics_path.read_text()
if "constraint_reports = tuple(entry[name] for entry in history)" not in content:
    replace_once(
        diagnostics_path,
        "        reports = tuple(entry[name] for entry in history)\n"
        "        upper_cap = tuple(report.at_upper_cap for report in reports)\n"
        "        lower_bound = tuple(report.at_lower_bound for report in reports)\n",
        "        constraint_reports = tuple(entry[name] for entry in history)\n"
        "        upper_cap = tuple(\n"
        "            report.at_upper_cap for report in constraint_reports\n"
        "        )\n"
        "        lower_bound = tuple(\n"
        "            report.at_lower_bound for report in constraint_reports\n"
        "        )\n",
    )
    content = diagnostics_path.read_text()
    replacements = (
        ("[report.multiplier_after for report in reports]", "[report.multiplier_after for report in constraint_reports]"),
        ("            for report in reports\n            if report.constraint_residual is not None", "            for report in constraint_reports\n            if report.constraint_residual is not None"),
        ("            for report in reports\n            if report.updated", "            for report in constraint_reports\n            if report.updated"),
        ("                rollout_count=len(reports),", "                rollout_count=len(constraint_reports),"),
        ("                saturation_fraction=sum(upper_cap) / len(reports),", "                saturation_fraction=sum(upper_cap) / len(constraint_reports),"),
        ("                lower_bound_fraction=sum(lower_bound) / len(reports),", "                lower_bound_fraction=sum(lower_bound) / len(constraint_reports),"),
    )
    for old, new in replacements:
        count = content.count(old)
        if count != 1:
            raise RuntimeError(
                f"{diagnostics_path}: expected one typed history replacement, found {count}"
            )
        content = content.replace(old, new)
    diagnostics_path.write_text(content)


evidence_path = Path("trade_rl/rl/lagrangian_evidence.py")
content = evidence_path.read_text()
if "LagrangianConstraintSpec," not in content:
    replace_once(
        evidence_path,
        "from trade_rl.rl.lagrangian import (\n"
        "    DualUpdateReport,\n"
        "    LagrangianSchema,\n",
        "from trade_rl.rl.lagrangian import (\n"
        "    DualUpdateReport,\n"
        "    LagrangianConstraintSpec,\n"
        "    LagrangianSchema,\n",
    )

content = evidence_path.read_text()
if "rollout evidence schema is missing support metadata" not in content:
    replace_once(
        evidence_path,
        "        ):\n"
        "            consumed = report.consumed_denominator\n",
        "        ):\n"
        "            if not isinstance(spec, LagrangianConstraintSpec):\n"
        "                raise TypeError(\n"
        "                    \"rollout evidence schema is missing support metadata\"\n"
        "                )\n"
        "            consumed = report.consumed_denominator\n",
    )
