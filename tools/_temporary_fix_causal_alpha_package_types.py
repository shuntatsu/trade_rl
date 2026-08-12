from pathlib import Path

path = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = path.read_text(encoding="utf-8")
old = '''        mappings = {
            "batches": dict(self.batches),
            "partitions": dict(self.partitions),
            "samples": dict(self.samples),
            "batch_evidence": dict(self.batch_evidence),
        }
        for field, values in mappings.items():
            if set(values) != set(symbols):
                raise ValueError(
                    f"causal alpha package {field} must exactly match train_symbols"
                )
        if self.selection.selected_candidate_digest != self.selected_candidate_digest:
            raise ValueError("causal alpha package selected candidate identity drifted")
        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"causal alpha package {field} is invalid")
        for symbol in symbols:
            batch = mappings["batches"][symbol]
            if getattr(batch, "teacher_config_digest", None) != self.teacher_config_digest:
                raise ValueError("causal alpha package batch teacher identity drifted")
            for field in ("partitions", "samples", "batch_evidence"):
                digest = getattr(mappings[field][symbol], "digest", None)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(
                        f"causal alpha package {field} digest is unavailable"
                    )
        expected = content_digest(
            {
                "batch_digests": {
                    symbol: getattr(mappings["batches"][symbol], "digest", None)
                    for symbol in symbols
                },
                "batch_evidence_digests": {
                    symbol: mappings["batch_evidence"][symbol].digest
                    for symbol in symbols
                },
                "partition_digests": {
                    symbol: mappings["partitions"][symbol].digest for symbol in symbols
                },
                "sample_digests": {
                    symbol: mappings["samples"][symbol].digest for symbol in symbols
                },
                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
                "selection_digest": self.selection.digest,
                "teacher_config_digest": self.teacher_config_digest,
                "train_symbols": symbols,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher package digest mismatch")
        object.__setattr__(self, "train_symbols", symbols)
        for field, values in mappings.items():
            object.__setattr__(self, field, values)
        object.__setattr__(self, "digest", expected)
'''
new = '''        batches = dict(self.batches)
        partitions = dict(self.partitions)
        samples = dict(self.samples)
        batch_evidence = dict(self.batch_evidence)
        for field, values in (
            ("batches", batches),
            ("partitions", partitions),
            ("samples", samples),
            ("batch_evidence", batch_evidence),
        ):
            if set(values) != set(symbols):
                raise ValueError(
                    f"causal alpha package {field} must exactly match train_symbols"
                )
        if self.selection.selected_candidate_digest != self.selected_candidate_digest:
            raise ValueError("causal alpha package selected candidate identity drifted")
        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"causal alpha package {field} is invalid")
        for symbol in symbols:
            batch = batches[symbol]
            if batch.teacher_config_digest != self.teacher_config_digest:
                raise ValueError("causal alpha package batch teacher identity drifted")
            if len(partitions[symbol].digest) != 64:
                raise ValueError("causal alpha package partition digest is unavailable")
            if len(samples[symbol].digest) != 64:
                raise ValueError("causal alpha package sample digest is unavailable")
            if len(batch_evidence[symbol].digest) != 64:
                raise ValueError(
                    "causal alpha package batch evidence digest is unavailable"
                )
        expected = content_digest(
            {
                "batch_digests": {
                    symbol: batches[symbol].digest for symbol in symbols
                },
                "batch_evidence_digests": {
                    symbol: batch_evidence[symbol].digest for symbol in symbols
                },
                "partition_digests": {
                    symbol: partitions[symbol].digest for symbol in symbols
                },
                "sample_digests": {
                    symbol: samples[symbol].digest for symbol in symbols
                },
                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
                "selection_digest": self.selection.digest,
                "teacher_config_digest": self.teacher_config_digest,
                "train_symbols": symbols,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher package digest mismatch")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "batch_evidence", batch_evidence)
        object.__setattr__(self, "digest", expected)
'''
if text.count(old) != 1:
    raise SystemExit("causal alpha package post-init target drifted")
path.write_text(text.replace(old, new), encoding="utf-8")
