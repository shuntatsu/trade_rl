#!/usr/bin/env python3
"""Reuse one teacher environment for each deterministic episode chunk."""

from __future__ import annotations

from pathlib import Path


PATH = Path("trade_rl/learning/episode_teacher_artifact.py")

HELPERS = '''_EpisodeItem = tuple[OracleEpisodeContract, np.ndarray]
_EpisodeResult = tuple[EpisodeSupervisedPolicyDataset, int]
_EpisodeChunk = tuple[_EpisodeItem, ...]


def _collect_episode_chunk(
    environment_factory: Any,
    batch: EpisodeOracleBatch,
    teacher_config_digest: str,
    items: _EpisodeChunk,
) -> tuple[_EpisodeResult, ...]:
    if not items:
        raise ValueError("teacher rollout episode chunk must not be empty")
    environment = environment_factory()
    try:
        collected: list[_EpisodeResult] = []
        for contract, targets in items:
            isolated_contract = OracleEpisodeContract(
                dataset_id=contract.dataset_id,
                episode_index=0,
                start=contract.start,
                stop=contract.stop,
                initial_state_mode=contract.initial_state_mode,
                initial_weights=contract.initial_weights,
            )
            episode_batch = EpisodeOracleBatch(
                dataset_id=batch.dataset_id,
                teacher_config_digest=batch.teacher_config_digest,
                sampling_config_digest=batch.sampling_config_digest,
                contracts=(isolated_contract,),
                targets=(targets,),
                solver_provenance=batch.solver_provenance,
            )
            episode = collect_episode_teacher_rollout(
                environment,
                episode_batch,
                teacher_config_digest=teacher_config_digest,
            )
            collected.append((episode, contract.episode_index))
        return tuple(collected)
    finally:
        environment.close()


def _collect_forked_episode_chunk(
    items: _EpisodeChunk,
) -> tuple[_EpisodeResult, ...]:
    if (
        _FORK_EPISODE_ENVIRONMENT_FACTORY is None
        or _FORK_EPISODE_BATCH is None
        or _FORK_EPISODE_TEACHER_DIGEST is None
    ):
        raise RuntimeError("forked episode teacher worker is not initialized")
    return _collect_episode_chunk(
        _FORK_EPISODE_ENVIRONMENT_FACTORY,
        _FORK_EPISODE_BATCH,
        _FORK_EPISODE_TEACHER_DIGEST,
        items,
    )


def _episode_item_chunks(
    items: tuple[_EpisodeItem, ...],
    *,
    maximum_chunks: int,
) -> tuple[_EpisodeChunk, ...]:
    if not items:
        return ()
    chunk_count = min(maximum_chunks, len(items))
    minimum_size, remainder = divmod(len(items), chunk_count)
    chunks: list[_EpisodeChunk] = []
    offset = 0
    for chunk_index in range(chunk_count):
        size = minimum_size + (1 if chunk_index < remainder else 0)
        chunks.append(items[offset : offset + size])
        offset += size
    return tuple(chunks)


'''

PARALLEL = '''    pending_chunks = _episode_item_chunks(
        tuple(pending_items),
        maximum_chunks=worker_count,
    )
    if pending_chunks and "fork" in mp.get_all_start_methods():
        global _FORK_EPISODE_BATCH
        global _FORK_EPISODE_ENVIRONMENT_FACTORY
        global _FORK_EPISODE_TEACHER_DIGEST
        _FORK_EPISODE_BATCH = batch
        _FORK_EPISODE_ENVIRONMENT_FACTORY = environment_factory
        _FORK_EPISODE_TEACHER_DIGEST = teacher_config_digest
        try:
            context = mp.get_context("fork")
            with context.Pool(
                processes=len(pending_chunks),
                maxtasksperchild=1,
            ) as pool:
                for values in pool.imap(
                    _collect_forked_episode_chunk,
                    pending_chunks,
                    chunksize=1,
                ):
                    for value in values:
                        persist(value)
        finally:
            _FORK_EPISODE_BATCH = None
            _FORK_EPISODE_ENVIRONMENT_FACTORY = None
            _FORK_EPISODE_TEACHER_DIGEST = None
    elif pending_chunks:

        def collect_chunk(items: _EpisodeChunk) -> tuple[_EpisodeResult, ...]:
            return _collect_episode_chunk(
                environment_factory,
                batch,
                teacher_config_digest,
                items,
            )

        with ThreadPoolExecutor(
            max_workers=len(pending_chunks),
            thread_name_prefix="teacher-rollout",
        ) as executor:
            for values in executor.map(collect_chunk, pending_chunks):
                for value in values:
                    persist(value)
'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    helper_start = text.index("def _collect_isolated_episode(")
    helper_end = text.index(
        "def collect_episode_teacher_rollout_parallel(",
        helper_start,
    )
    parallel_start = text.index(
        '    if pending_items and "fork" in mp.get_all_start_methods():',
        helper_end,
    )
    parallel_end = text.index("    collected = tuple(", parallel_start)
    updated = (
        text[:helper_start]
        + HELPERS
        + text[helper_end:parallel_start]
        + PARALLEL
        + text[parallel_end:]
    )
    PATH.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
