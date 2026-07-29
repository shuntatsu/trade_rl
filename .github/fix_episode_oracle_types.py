from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "trade_rl/learning/episode_teacher_artifact.py",
    '''        if int(info.get("start_index", -1)) != contract.start:
            raise ValueError("episode teacher environment reset start mismatch")
''',
    '''        raw_start = info.get("start_index")
        if (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, int)
            or raw_start != contract.start
        ):
            raise ValueError("episode teacher environment reset start mismatch")
''',
)

replace_once(
    "trade_rl/learning/episode_behavior_cloning.py",
    '''class EpisodeDataset(Protocol):
    sample_count: int
    episode_ids: np.ndarray
''',
    '''class EpisodeDataset(Protocol):
    @property
    def sample_count(self) -> int: ...

    @property
    def episode_ids(self) -> np.ndarray: ...
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''                    episode_batch: EpisodeOracleBatch | None = None
                    episode_split: BehaviorCloningSplit | None = None
                    if teacher_kind == "oracle":
''',
    '''                    episode_batch: EpisodeOracleBatch | None = None
                    episode_split: BehaviorCloningSplit | None = None
                    teacher_dataset: SupervisedPolicyDataset
                    if teacher_kind == "oracle":
''',
)
