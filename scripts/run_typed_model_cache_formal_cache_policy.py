"""Run one frozen cache-policy cell and emit its policy-neutral oracle replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import load_and_validate_manifest
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    validate_no_holdout_capability,
    validate_protocol_v1_1,
)
from src.oracles.cache_request_replay import build_policy_neutral_replay_from_manifest
from src.runtime.portable_resource_identity import (
    add_portable_resource_arguments,
    load_registry,
    resolve_resource,
)
from src.runtime.generated_checkpoint_resources import (
    add_generated_checkpoint_resource_arguments,
    audit_forwarded_resource_arguments,
    resolve_generated_checkpoint_arguments,
)
from src.runtime.formal_protocol_capabilities import get_protocol_capabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--fairness-manifest-path", required=True)
    parser.add_argument("--evaluation-unit-id", required=True)
    parser.add_argument("--request-replay-path", required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    add_portable_resource_arguments(parser)
    add_generated_checkpoint_resource_arguments(parser)
    parser.add_argument("--seed-checkpoint-manifest-path", default="")
    parser.add_argument("--checkpoint-provenance-manifest-path", default="")
    parser.add_argument("--resolved-execution-context-path", default="")
    parser.add_argument("--formal-training-execution-binding-path", default="")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8-sig"))
    validate_protocol_v1_1(protocol)
    capabilities = get_protocol_capabilities(
        protocol.get("typed_model_cache_formal_protocol_version")
    )
    fairness_audit = None
    generated_audit = None
    if capabilities.generated_checkpoint_resource_required:
        static_registry = load_registry(args.resource_registry_path)
        fairness_audit = resolve_resource(
            static_registry,
            args.fairness_manifest_resource_id,
            expected_role="fairness_manifest",
            explicit_paths=[args.fairness_manifest_path],
            roots={
                "worktree_root": args.repository_root or ROOT,
                "data_root": args.data_root or None,
                "protocol_artifact_root": args.protocol_artifact_root or None,
                "checkpoint_root": args.checkpoint_root or None,
            },
            manifest_path=args.resource_registry_path,
        )
        resolved_context = json.loads(
            Path(args.resolved_execution_context_path).read_text(encoding="utf-8-sig")
        )
        execution_binding = json.loads(
            Path(args.formal_training_execution_binding_path).read_text(encoding="utf-8-sig")
        )
        generated_audit = resolve_generated_checkpoint_arguments(
            args,
            expected_capacity_label=(
                str(args.runtime_config_resource_id).split(".", 1)[1]
                if "." in str(args.runtime_config_resource_id) else None
            ),
            protocol=protocol,
            resolved_execution_context=resolved_context,
            execution_binding=execution_binding,
        )
    manifest, report = load_and_validate_manifest(
        args.fairness_manifest_path, root=ROOT, check_files=True
    )
    if report.get("status") != "pass":
        raise FormalExecutionError("cache-policy fairness manifest validation failed")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise FormalExecutionError("cache-policy wrapper command is empty")
    validate_no_holdout_capability(command)
    forwarding_audit = (
        audit_forwarded_resource_arguments(command, args)
        if capabilities.generated_checkpoint_resource_required else None
    )
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    replay_path = Path(args.request_replay_path)
    if replay_path.exists():
        raise FileExistsError(f"refusing to overwrite request replay: {replay_path}")
    replay = build_policy_neutral_replay_from_manifest(
        root=ROOT,
        manifest=manifest,
        evaluation_unit_id=args.evaluation_unit_id,
    )
    if capabilities.request_subject_lifecycle_required:
        lifecycle = replay.get("formal_request_subject_lifecycle") or {}
        if lifecycle.get("contract_version") != "1.0.0" or not replay.get(
            "formal_request_exposure_fingerprint"
        ):
            raise FormalExecutionError(
                "Protocol 2.2 request replay lacks frozen subject lifecycle identity"
            )
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "benchmark_returncode": result.returncode,
                "request_replay_path": str(replay_path.resolve()),
                "request_replay_fingerprint": replay["request_replay_fingerprint"],
                "static_fairness_resource_audit": fairness_audit,
                "generated_checkpoint_resource_audit": generated_audit,
                "outer_nested_resource_forwarding_audit": forwarding_audit,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
