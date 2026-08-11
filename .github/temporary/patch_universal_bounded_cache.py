from pathlib import Path

path = Path("trade_rl/rl/universal_single_instrument_env.py")
text = path.read_text()
if "max_cached_environments: int | None = None" in text:
    raise SystemExit(0)
text = text.replace(
    "        training_contract_digest: str | None = None,\n    ) -> None:\n",
    "        training_contract_digest: str | None = None,\n"
    "        max_cached_environments: int | None = None,\n"
    "    ) -> None:\n",
    1,
)
anchor = "        if training_contract_digest is not None:\n            training_contract_digest = require_sha256(\n                training_contract_digest,\n                field=\"training_contract_digest\",\n            )\n"
replacement = anchor + "        if max_cached_environments is not None and (\n            isinstance(max_cached_environments, bool)\n            or not isinstance(max_cached_environments, int)\n            or max_cached_environments <= 0\n        ):\n            raise ValueError(\"max_cached_environments must be null or a positive integer\")\n"
if anchor not in text:
    raise SystemExit("cache validation anchor not found")
text = text.replace(anchor, replacement, 1)
anchor = "        self._training_identity_enabled = training_contract_digest is not None\n"
if anchor not in text:
    raise SystemExit("cache state anchor not found")
text = text.replace(
    anchor,
    anchor + "        self._max_cached_environments = max_cached_environments\n",
    1,
)
marker = "    def _load_environment(\n        self,\n        route: InstrumentRoute,\n    ) -> ConcreteSingleInstrumentEnv:\n"
if marker not in text:
    raise SystemExit("load environment marker not found")
helper = '''    def _evict_cached_environment_for(self, symbol: str) -> bool:\n        limit = self._max_cached_environments\n        if limit is None or symbol in self._environments or len(self._environments) < limit:\n            return False\n        if not self._episode_complete:\n            raise RuntimeError("cannot evict a child environment during an active episode")\n        victim_symbol = next(iter(self._environments))\n        victim = self._environments.pop(victim_symbol)\n        self._environment_object_ids.discard(id(victim))\n        if self._active_environment is victim:\n            self._active_environment = None\n            self._active_episode_binding = None\n        reference_evicted = (\n            hasattr(self, "_reference_environment")\n            and self._reference_environment is victim\n        )\n        victim.close()\n        return reference_evicted\n\n'''
text = text.replace(marker, helper + marker, 1)
old = "        binding = self._bindings[symbol]\n        environment = self._environment_factory(binding)\n"
new = "        reference_evicted = self._evict_cached_environment_for(symbol)\n        binding = self._bindings[symbol]\n        environment = self._environment_factory(binding)\n"
if old not in text:
    raise SystemExit("load binding anchor not found")
text = text.replace(old, new, 1)
old = "        self._environments[symbol] = environment\n        self._environment_object_ids.add(object_id)\n        return environment\n"
new = "        self._environments[symbol] = environment\n        self._environment_object_ids.add(object_id)\n        if reference_evicted:\n            self._reference_environment = environment\n        return environment\n"
if old not in text:
    raise SystemExit("cache installation anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
compile(text, str(path), "exec")

runner = Path("trade_rl/workflows/universal_training_runner.py")
rtext = runner.read_text()
if "max_cached_environments: int | None = 1" not in rtext:
    rtext = rtext.replace(
        "    training_contract_digest: str\n    run_seed: int\n",
        "    training_contract_digest: str\n    run_seed: int\n    max_cached_environments: int | None = 1\n",
        1,
    )
    rtext = rtext.replace(
        "            training_contract_digest=self.training_contract_digest,\n        )\n",
        "            training_contract_digest=self.training_contract_digest,\n"
        "            max_cached_environments=self.max_cached_environments,\n"
        "        )\n",
        1,
    )
    rtext = rtext.replace(
        '                "run_seed": self.run_seed,\n',
        '                "run_seed": self.run_seed,\n'
        '                "max_cached_environments": self.max_cached_environments,\n',
        1,
    )
    runner.write_text(rtext)
    compile(rtext, str(runner), "exec")
