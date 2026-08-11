from pathlib import Path

features_path = Path("trade_rl/data/universal_features.py")
text = features_path.read_text()
if "def universal_feature_schema_digest_from_names(" not in text:
    old = '''def universal_feature_schema_digest(features: Iterable[NamedFeature]) -> str:\n    names = tuple(feature.name for feature in features)\n    if not names:\n        raise ValueError("universal feature schema must not be empty")\n    return content_digest(\n        {\n            "version": "universal_target_local_features_v1",\n            "ordered_feature_names": names,\n            "instrument_descriptors": UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,\n        }\n    )\n'''
    new = '''def universal_feature_schema_digest_from_names(feature_names: Iterable[str]) -> str:\n    names = tuple(str(name) for name in feature_names)\n    if not names or any(not name for name in names):\n        raise ValueError("universal feature schema must not be empty")\n    if len(set(names)) != len(names):\n        raise ValueError("universal feature schema names must be unique")\n    return content_digest(\n        {\n            "version": "universal_target_local_features_v1",\n            "ordered_feature_names": names,\n            "instrument_descriptors": UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,\n        }\n    )\n\n\ndef universal_feature_schema_digest(features: Iterable[NamedFeature]) -> str:\n    return universal_feature_schema_digest_from_names(\n        feature.name for feature in features\n    )\n'''
    if old not in text:
        raise SystemExit("universal feature digest block not found")
    text = text.replace(old, new, 1)
    features_path.write_text(text)
    compile(text, str(features_path), "exec")

training_path = Path("trade_rl/workflows/universal_training.py")
training = training_path.read_text()
if "universal_feature_schema_digest_from_names" not in training:
    import_anchor = "from trade_rl.data.contracts import FeatureSpec\n"
    if import_anchor not in training:
        raise SystemExit("universal training import anchor not found")
    training = training.replace(
        import_anchor,
        import_anchor
        + "from trade_rl.data.universal_features import universal_feature_schema_digest_from_names\n",
        1,
    )
old_helper = '''\n\ndef _universal_feature_schema_digest(feature_names: Sequence[str]) -> str:\n    names = tuple(str(value) for value in feature_names)\n    if not names or len(set(names)) != len(names) or any(not value for value in names):\n        raise ValueError("Universal feature names must be non-empty and unique")\n    return content_digest(\n        {\n            "feature_names": names,\n            "profile": "binance_universal_target_local_v1",\n            "schema_version": "universal_feature_schema_v1",\n        }\n    )\n'''
if old_helper in training:
    training = training.replace(old_helper, "", 1)
training = training.replace(
    "    feature_schema_digest = _universal_feature_schema_digest(feature_names)\n",
    "    feature_schema_digest = universal_feature_schema_digest_from_names(feature_names)\n",
    1,
)
training_path.write_text(training)
compile(training, str(training_path), "exec")
