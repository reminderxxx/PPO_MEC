#!/usr/bin/env python3
"""Qualify or execute the exact G14R22 one-time holdout package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.dedicated_holdout_execution import (
    HoldoutExecutionError,
    execute_package,
    read_json,
    validate_command_package,
    validate_unsigned_request,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--request-path", type=Path)
    value.add_argument("--command-package-path", type=Path, required=True)
    value.add_argument("--grant-path", type=Path)
    value.add_argument("--one-time-token-file", type=Path)
    value.add_argument("--check", choices=("qualify", "execute", "acceptance"), required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    package = read_json(args.command_package_path)
    if args.check == "acceptance":
        request = {"request_sha256": package["acceptance_request_sha256"]}
        receipt = execute_package(request, package, None, None, acceptance=True)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    if args.request_path is None:
        raise HoldoutExecutionError("--request-path is required")
    request = read_json(args.request_path)
    audit = validate_unsigned_request(request, package)
    if args.check == "qualify":
        print(json.dumps({"status": "pass", "execution_authorized": False, **audit}, indent=2))
        return 0
    if args.grant_path is None or args.one_time_token_file is None:
        raise HoldoutExecutionError("execute requires --grant-path and --one-time-token-file")
    grant = read_json(args.grant_path)
    if args.one_time_token_file.is_symlink() or not args.one_time_token_file.is_file():
        raise HoldoutExecutionError("one-time token file is missing or a symlink")
    receipt = execute_package(request, package, grant, args.one_time_token_file.read_bytes())
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
