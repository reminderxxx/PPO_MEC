"""Create independent, short-lived pre-grant quiescence evidence, without recovery writes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.restricted_recovery import (
    observe_restricted_recovery_quiescence,
    validate_restricted_recovery_request,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-request-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    request_path = Path(args.authorization_request_path)
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("frozen request is missing or symlinked")
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    validate_restricted_recovery_request(request, check_live=True)
    evidence = observe_restricted_recovery_quiescence(request)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": evidence["status"], "failure_codes": evidence["failure_codes"],
                      "evidence_path": str(output), "expires_at": evidence["expires_at"]}, indent=2))
    if evidence["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
