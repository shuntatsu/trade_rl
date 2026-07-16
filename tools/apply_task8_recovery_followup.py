from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "trade_rl/integrations/sb3_training.py"
    text = path.read_text(encoding="utf-8")
    old = '''                resume_manifest, resume_path = load_replay_buffer_artifact(
                    self.resume_replay_artifact
                )
                if resume_manifest.algorithm != config.algorithm:
                    raise ValueError("replay buffer algorithm mismatch")
                if resume_manifest.environment_digest != identity["environment_digest"]:
                    raise ValueError("replay buffer environment identity mismatch")
                model.load_replay_buffer(str(resume_path))
'''
    new = '''                replay_manifest, resume_path = load_replay_buffer_artifact(
                    self.resume_replay_artifact
                )
                if replay_manifest.algorithm != config.algorithm:
                    raise ValueError("replay buffer algorithm mismatch")
                if replay_manifest.environment_digest != identity["environment_digest"]:
                    raise ValueError("replay buffer environment identity mismatch")
                model.load_replay_buffer(str(resume_path))
'''
    if old not in text:
        raise RuntimeError("missing replay resume variable anchor")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
