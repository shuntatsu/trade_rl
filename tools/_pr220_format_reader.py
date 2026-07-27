from pathlib import Path

path = Path("trade_rl/studio/training_metrics.py")
source = path.read_text(encoding="utf-8")
source = source.replace(
    """                if (
                    step < 0
                    or not math.isfinite(wall_time)
                    or not math.isfinite(value)
                ):
""",
    """                if step < 0 or not math.isfinite(wall_time) or not math.isfinite(value):
""",
    1,
)
source = source.replace(
    """                    previous = (
                        None if cached is None else cached.files.get(event_file)
                    )
""",
    """                    previous = None if cached is None else cached.files.get(event_file)
""",
    1,
)
compile(source, str(path), "exec")
path.write_text(source, encoding="utf-8")
