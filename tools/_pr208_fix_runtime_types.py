from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{label} no longer matches")
    updated = source.replace(old, new, 1)
    compile(updated, str(path), "exec")
    path.write_text(updated, encoding="utf-8")


model = Path("trade_rl/integrations/sb3_model_assembly.py")
replacements = (
    ("    policy_identifier: object\n", "    policy_identifier: Any\n", "policy identifier"),
    (
        "    sequence_metadata: Mapping[str, object] | None\n",
        "    sequence_metadata: Mapping[str, Any] | None\n",
        "sequence metadata",
    ),
    (
        "    sequence_reconstructor: object | None\n",
        "    sequence_reconstructor: Any | None\n",
        "sequence reconstructor",
    ),
    (
        "    rollout_buffer_class: object | None = None\n",
        "    rollout_buffer_class: Any | None = None\n",
        "rollout buffer class",
    ),
    (
        "    rollout_buffer_kwargs: Mapping[str, object] | None = None\n",
        "    rollout_buffer_kwargs: Mapping[str, Any] | None = None\n",
        "rollout buffer kwargs",
    ),
    (
        ") -> object:\n    \"\"\"Construct one SB3 algorithm from validated immutable assembly inputs.\"\"\"\n",
        ") -> Any:\n    \"\"\"Construct one SB3 algorithm from validated immutable assembly inputs.\"\"\"\n",
        "model return type",
    ),
)
for old, new, label in replacements:
    replace_once(model, old, new, label)

checkpoint = Path("trade_rl/integrations/sb3_checkpoint_assembly.py")
replace_once(checkpoint, "    model: object\n", "    model: Any\n", "loaded model")
replace_once(checkpoint, "    fresh_model: object,\n", "    fresh_model: Any,\n", "fresh model")
