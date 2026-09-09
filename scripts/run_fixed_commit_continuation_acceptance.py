"""Synthetic-only acceptance entry; cannot qualify or resume a production run."""
import argparse
import json
from pathlib import Path
import sys

from continuation_executor.identity import absolute_path, canonical, read_json, verify_executor

SCIENTIFIC_COMMIT = "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-identity", required=True)
    parser.add_argument("--fixture-root")
    parser.add_argument("--internal-public-startup-host", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--proposal", help=argparse.SUPPRESS)
    parser.add_argument("--contract", help=argparse.SUPPRESS)
    parser.add_argument("--approval", help=argparse.SUPPRESS)
    parser.add_argument("--installation", help=argparse.SUPPRESS)
    parser.add_argument("--phase", help=argparse.SUPPRESS)
    parser.add_argument("--check", help=argparse.SUPPRESS)
    parser.add_argument("--startup-wait-seconds", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal_public_startup_host:
        try:
            from continuation_executor.startup_binding import SyntheticAcceptanceBinding
            from execute_fixed_commit_continuation import main as public_main
            binding = SyntheticAcceptanceBinding(read_json(args.installation))
            public_argv = ["--executor-identity", args.executor_identity,
                "--proposal", args.proposal, "--contract", args.contract,
                "--approval", args.approval, "--phase", args.phase,
                "--check", args.check, "--startup-wait-seconds", args.startup_wait_seconds]
            return public_main(public_argv, _startup_binding=binding)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(json.dumps({"version":"1.0.0", "event":"terminal", "status":"rejected",
                "reason_code":"SYNTHETIC_BINDING_REJECTED", "reason":str(exc),
                "execution_authorized":False}), flush=True)
            return 2
    try:
        if args.fixture_root is None:
            raise ValueError("--fixture-root is required")
        identity = read_json(args.executor_identity)
        identity_report = verify_executor(identity, Path(__file__).resolve().parents[1])
        fixture = absolute_path(args.fixture_root)
        if fixture.exists() or not fixture.name.startswith("synthetic_"):
            raise ValueError("acceptance requires a new synthetic_* fixture root")
        from continuation_executor.scientific import load_native
        native = load_native(str(Path.cwd()), SCIENTIFIC_COMMIT)
        manifest_path=Path.cwd()/"configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/execution_environment_manifest.json"
        manifest=read_json(manifest_path)
        if sys.executable != manifest["runtime_location"]["resolved_python_absolute_path"]:
            raise ValueError("actual acceptance interpreter differs from frozen environment")
        protocol=read_json(manifest_path.with_name("protocol_v2_9_manifest.json"))
        environment=native["environment"].resolve_execution_environment(
            clean_worktree_root=Path.cwd(),execution_commit=SCIENTIFIC_COMMIT,python_executable=sys.executable,
            environment_manifest=manifest,expected_identity=manifest["scientific_identity"],
            protocol_bound_extensions=native["environment"].protocol_bound_extensions_from_protocol(protocol),
            forbidden_source_roots=[str(Path(__file__).resolve().parents[1])],require_clean_git_worktree=True)
        # Scope, identity and scientific source checks precede the first write.
        fixture.mkdir()
        from continuation_executor.public_startup_acceptance import run_public_startup_acceptance
        public_startup = run_public_startup_acceptance(
            fixture/"synthetic_public_startup", identity, Path(__file__).resolve())
        from continuation_executor.synthetic_chain import run_chain
        result = run_chain(fixture, native, identity)
        result["public_startup_handoff"] = public_startup
        result["executor"] = identity_report
        result["pre_instrumentation_environment"] = environment.runtime_audit
        result["real_execution_authorized"] = False
        (fixture/"acceptance_report.json").write_bytes(canonical(result)+b"\n")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status":"rejected", "reason":str(exc), "real_execution_authorized":False}))
        return 2
    print(json.dumps({"status":"pass", "report":str(fixture/"acceptance_report.json"),
                     "real_execution_authorized":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
