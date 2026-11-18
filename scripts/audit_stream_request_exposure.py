"""Measure release-to-delivery frames through the fixed end of a stream session."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def audit_session(result_path, receipts_path, agent, horizon):
    """Verify every frame's active/queued request IDs against the episode summary."""
    package = json.loads(result_path.read_text())
    if "results" in package:
        metrics = package["results"][agent]
        expected_hash = metrics["receipts_sha256"]
    else:
        if package["agent"] != agent:
            raise ValueError("agent mismatch")
        metrics = package["metrics"]
        expected_hash = package["receipts_sha256"]
    digest = hashlib.sha256()
    with receipts_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("receipt hash mismatch")

    steps = metrics["steps"]
    if not 0 < steps <= horizon or metrics["frame_receipt_count"] != steps:
        raise ValueError("invalid session length")
    requests = {}
    for row in metrics["per_request"]:
        rid = row["request_id"]
        arrival = row["arrival_frame"]
        completion = row["completion_frame_index"]
        if rid in requests or not row["released"] or not 0 <= arrival < steps:
            raise ValueError("expected unique released requests")
        if row["completed"] != (completion is not None):
            raise ValueError("completion flag disagrees with timestamp")
        if completion is None and steps < horizon:
            raise ValueError("incomplete request in a truncated session")
        if completion is not None and not arrival <= completion < steps:
            raise ValueError("invalid completion frame")
        # Delivered requests consume their final frame. Censored requests stop at the horizon.
        requests[rid] = (arrival, horizon if completion is None else completion + 1)
    if not requests or len(requests) != metrics["released_requests"] or len(requests) != metrics["request_count"]:
        raise ValueError("incomplete release inventory")
    if sum(r["completed"] for r in metrics["per_request"]) != metrics["completed_requests"]:
        raise ValueError("inconsistent completed request count")

    exposure = Counter()
    frame_count = 0
    open_file = gzip.open if receipts_path.suffix == ".gz" else open
    with open_file(receipts_path, "rt") as handle:
        for frame, line in enumerate(handle):
            receipt = json.loads(line)
            if (receipt["case_id"], receipt["seed"], receipt["agent"], receipt["frame_index"]) != (
                package["case_id"], package["seed"], agent, frame
            ):
                raise ValueError(f"unexpected receipt identity at frame {frame}")
            if receipt["credit_contract"] != "uniform_released_pending_team_share_not_causal":
                raise ValueError("receipt contract does not identify all released pending requests")
            expected = {rid for rid, (start, stop) in requests.items() if start <= frame < stop}
            if expected != set(receipt["request_rewards"]):
                raise ValueError(f"pending request IDs disagree at frame {frame}")
            exposure.update(expected)
            frame_count += 1
    if frame_count != steps:
        raise ValueError("missing or extra frames")

    waves = {}
    for arrival in sorted({start for start, _ in requests.values()}):
        ids = [rid for rid, (start, _) in requests.items() if start == arrival]
        waves[str(arrival)] = {"requests": len(ids), "request_frames": sum(exposure[rid] for rid in ids)}
    total = sum(exposure.values())
    return dict(case_id=package["case_id"], seed=package["seed"], agent=agent,
                horizon=horizon, requests=len(requests), completed=metrics["completed_requests"],
                request_frames=total, mean_observed_frames=total / len(requests), by_arrival_frame=waves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--horizon", type=int, required=True, help="scheduled session duration in frames")
    args = parser.parse_args()
    print(json.dumps(audit_session(args.result, args.receipts, args.agent, args.horizon), indent=2))


if __name__ == "__main__":
    main()
