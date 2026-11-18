import gzip
import hashlib
import json

import pytest

from scripts.audit_stream_request_exposure import audit_session


def session(tmp_path, *, intervention=False, wrong_ids=False):
    # A spends two frames in the system. B arrives later and is still pending at the horizon.
    ids = [["a"], ["a", "b"], ["b"]]
    if wrong_ids:
        ids[-1] = ["a"]
    path = tmp_path / "receipts.gz"
    with gzip.open(path, "wt") as handle:
        for frame, pending in enumerate(ids):
            row = dict(agent="sagh", seed=7, case_id="toy", frame_index=frame,
                       credit_contract="uniform_released_pending_team_share_not_causal",
                       request_rewards=dict.fromkeys(pending, 0))
            handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metrics = dict(steps=3, frame_receipt_count=3, completed_requests=1, released_requests=2,
                   request_count=2, per_request=[
                       dict(request_id="a", released=True, arrival_frame=0, completed=True, completion_frame_index=1),
                       dict(request_id="b", released=True, arrival_frame=1, completed=False, completion_frame_index=None)])
    if intervention:
        package = dict(metrics=metrics, agent="sagh", receipts_sha256=digest)
    else:
        package = dict(results={"sagh":dict(**metrics, receipts_sha256=digest)})
    result = tmp_path / "result.json"
    result.write_text(json.dumps(dict(**package, seed=7, case_id="toy")))
    return result, path


@pytest.mark.parametrize("intervention", [True, False])
def test_pending_tail_and_inclusive_delivery(tmp_path, intervention):
    row = audit_session(*session(tmp_path, intervention=intervention), "sagh", 3)
    assert row["mean_observed_frames"] == 2
    assert row["by_arrival_frame"]["1"] == {"requests": 1, "request_frames": 2}


def test_equal_count_with_wrong_identity_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="pending request IDs"):
        audit_session(*session(tmp_path, wrong_ids=True), "sagh", 3)


def test_partial_censoring_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="truncated session"):
        audit_session(*session(tmp_path), "sagh", 4)
