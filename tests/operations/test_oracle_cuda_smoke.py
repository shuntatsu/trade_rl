from __future__ import annotations

from trade_rl.operations import oracle_cuda_smoke


def test_oracle_cuda_smoke_registers_adapter_before_delegating(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    delegated: list[list[str] | None] = []

    def fake_register(name: str, backend: object) -> None:
        calls.append((name, backend))

    def fake_benchmark_main(argv: list[str] | None = None) -> int:
        delegated.append(argv)
        return 17

    monkeypatch.setattr(
        oracle_cuda_smoke,
        "register_oracle_accelerator_backend",
        fake_register,
    )
    monkeypatch.setattr(oracle_cuda_smoke, "_benchmark_main", fake_benchmark_main)

    arguments = ["--backend", "all", "--episode-count", "1"]
    assert oracle_cuda_smoke.main(arguments) == 17
    assert calls == [("cuda", oracle_cuda_smoke.solve_torch_cuda_oracle_batch)]
    assert delegated == [arguments]
