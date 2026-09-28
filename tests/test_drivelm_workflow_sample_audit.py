import json

import pytest

from scripts.audit_drivelm_workflow_sample import audit_sample


def sample(**fields):
    return json.dumps({"s": {"key_frames": {"f": {"QA": {
        "perception": [{"Q": "synthetic question", "A": "synthetic answer", **fields}]
    }}}}}).encode()


def test_null_and_missing_are_distinct():
    report = audit_sample(sample(con_up=None))
    assert report["relation_fields"]["con_up"]["null"] == 1
    assert report["relation_fields"]["con_down"]["missing"] == 1
    assert report["graph_status"] == "NO_NON_NULL_RELATION_FIELDS"
    assert report["inferred_edge_count"] == 0


def test_nonnull_is_not_verified_dependency():
    report = audit_sample(sample(con_up=["unknown_node"]))
    assert report["graph_status"] == "RELATION_FIELDS_REQUIRE_SEMANTIC_AUDIT"
    assert not report["observed_dependency_graph_verified"]


@pytest.mark.parametrize("raw", [b'{}', b'[]', b'{"s":{}}',
                                 b'{"s":{},"s":{}}', b'{"s":NaN}'])
def test_invalid_samples_fail(raw):
    with pytest.raises(ValueError):
        audit_sample(raw)


def test_identity_is_byte_sensitive():
    raw = sample(con_up=None)
    first, second = audit_sample(raw), audit_sample(raw + b'\n')
    assert first["qa_count"] == second["qa_count"] == 1
    assert first["source_sha256"] != second["source_sha256"]
    assert not first["performance_evidence"]
