import copy

import pytest

from scripts.download_driving_pilot_model import FILES, WEIGHT_SHA, validate_inventory


def inventory():
    rows = [{"type": "file", "path": p, "size": 10, "oid": "0" * 40} for p in FILES]
    rows[-1].update(size=1015025832, lfs={"oid": WEIGHT_SHA})
    return rows


def test_fixed_allowlist_excludes_other_formats():
    rows = inventory() + [{"type": "directory", "path": "onnx"},
                          {"type": "file", "path": "other.bin", "size": 999999999}]
    assert [r["path"] for r in validate_inventory(rows)] == list(FILES)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "weight_hash", "weight_size", "budget"])
def test_drift_rejected(kind):
    rows = inventory()
    if kind == "missing":
        rows.pop(0)
    elif kind == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "weight_hash":
        rows[-1]["lfs"]["oid"] = "0" * 64
    elif kind == "weight_size":
        rows[-1]["size"] += 1
    else:
        rows[0]["size"] = 100000000
    with pytest.raises((ValueError, KeyError)):
        validate_inventory(rows)
