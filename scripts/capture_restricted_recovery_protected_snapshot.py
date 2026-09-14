"""Create an explicit, create-only protected-workspace snapshot for I5-A."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_restricted_recovery_acceptance_artifacts import (
    build_protected_snapshot,
    write_create_only,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--capture-kind", required=True)
    parser.add_argument("--capture-source", required=True)
    args = parser.parse_args()
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build_protected_snapshot(
        Path(args.workspace_root),
        snapshot_id=args.snapshot_id,
        capture_kind=args.capture_kind,
        capture_source=args.capture_source,
    )
    write_create_only(output, snapshot)
    print(output)


if __name__ == "__main__":
    main()
