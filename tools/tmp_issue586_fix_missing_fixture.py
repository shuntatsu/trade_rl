from pathlib import Path

path = Path("tests/evaluation/experiments/bootstrap/test_issue586_target_source_validator.py")
text = path.read_text()
old = '''    missing = [dict(item) for item in reports]\n    missing[0]["archive_available"] = False\n    missing[0]["checksum_available"] = False\n    missing[0]["checksum_verified"] = False\n    assert decide_target_source_status(missing) == "PARTIAL_USDM_15M_TARGET_SOURCE"\n'''
new = '''    missing = [dict(item) for item in reports]\n    missing[0] = validate_target_archive_bytes(\n        symbol=TARGET_SYMBOLS[0],\n        date=TARGET_DATES[0],\n        archive_bytes=None,\n        checksum_bytes=None,\n    )\n    assert decide_target_source_status(missing) == "PARTIAL_USDM_15M_TARGET_SOURCE"\n'''
if text.count(old) != 1:
    raise SystemExit("missing-source fixture block not found exactly once")
path.write_text(text.replace(old, new, 1))
