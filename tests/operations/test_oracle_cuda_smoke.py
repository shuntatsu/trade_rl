from __future__ import annotations

from trade_rl.operations import oracle_cuda_smoke


def test_oracle_cuda_smoke_injects_adapter_before_delegating(monkeypatch) -> None:
    delegated: list[tuple[list[str] | None, object]] = []

    def fake_benchmark_main(
        argv: list[str] | None = None,
        *,
        accelerator_backend: object = None,
    ) -> int:
        delegated.append((argv, accelerator_backend))
        return 17

    monkeypatch.setattr(oracle_cuda_smoke, "_benchmark_main", fake_benchmark_main)

    arguments = ["--backend", "all", "--episode-count", "1"]
    assert oracle_cuda_smoke.main(arguments) == 17
    assert delegated == [(arguments, oracle_cuda_smoke.solve_torch_cuda_oracle_batch)]
