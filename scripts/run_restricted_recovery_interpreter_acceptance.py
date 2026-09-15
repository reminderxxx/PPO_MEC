"""Build G14R20-I5-B evidence without issuing a grant or creating a run root."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.build_restricted_recovery_acceptance_artifacts import (
    build_protected_snapshot,
    compare_protected_snapshots,
)
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ALLOWED_CELL_IDS,
    ORIGINAL_PROJECT_GRANT_PATH,
    ORIGINAL_PROJECT_ROOT,
    ORIGINAL_REQUEST_PATH,
    ORIGINAL_RUN_ROOT,
    RestrictedRecoveryError,
    audit_original_recovery_source,
    restricted_cell_layout,
    stable_source_audit_projection,
    validate_recovery_executor_live,
    validate_restricted_recovery_request,
)


PREPARE = ROOT / "scripts/prepare_typed_model_cache_restricted_recovery.py"
SUPPORT = ROOT / "scripts/run_typed_model_cache_formal_support.py"
BENCHMARK = ROOT / "scripts/benchmark_main_results.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def git_value(revision: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", revision], text=True
    ).strip()


def require_clean_checkout() -> None:
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "README.md",
            "docs",
            "scripts",
            "src",
            "tests",
            "configs",
        ],
        text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    )
    if status.strip():
        raise RestrictedRecoveryError("interpreter acceptance requires a clean checkout")


def run(command: list[str], *, environment: Mapping[str, str]) -> dict[str, Any]:
    started_at = now_iso()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(environment),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "started_at": started_at,
        "completed_at": now_iso(),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def rehash_request(request: dict[str, Any]) -> None:
    execution = request["recovery_execution"]
    binding = execution["python_environment_binding"]
    binding["binding_sha256"] = canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    context = execution["resolved_execution_context"]
    context["context_sha256"] = canonical_sha256(
        {key: value for key, value in context.items() if key != "context_sha256"}
    )
    execution["resolved_execution_context_sha256"] = context["context_sha256"]
    execution["command_plan_sha256"] = canonical_sha256(execution["command_plan"])
    execution["recovery_execution_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in execution.items()
            if key != "recovery_execution_identity_sha256"
        }
    )
    request["authorization_request_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "authorization_request_sha256"
        }
    )


def rejected_request_case(
    request: Mapping[str, Any], mutation, *, label: str, recovery_root: Path
) -> dict[str, Any]:
    changed = deepcopy(dict(request))
    mutation(changed["recovery_execution"])
    rehash_request(changed)
    try:
        validate_restricted_recovery_request(changed, check_live=False)
    except RestrictedRecoveryError as exc:
        return {
            "label": label,
            "status": "rejected_before_dispatch_or_recovery_write",
            "error": str(exc),
            "recovery_root_exists": recovery_root.exists(),
        }
    raise RestrictedRecoveryError(f"negative case was accepted: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--recovery-execution-id", required=True)
    args = parser.parse_args()

    require_clean_checkout()
    output_root = Path(args.output_root).absolute()
    recovery_root = Path(args.recovery_root).absolute()
    launch = Path(args.python_executable).absolute()
    if output_root.exists():
        raise RestrictedRecoveryError("acceptance output root must be create-only")
    if recovery_root.exists():
        raise RestrictedRecoveryError("future recovery root must not exist")
    if recovery_root.name != args.recovery_execution_id:
        raise RestrictedRecoveryError("future recovery root/ID mismatch")
    output_root.mkdir(parents=True, exist_ok=False)
    request_path = output_root / "unsigned_recovery_request.json"
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    before = audit_original_recovery_source()
    protection_start = build_protected_snapshot(
        ORIGINAL_PROJECT_ROOT,
        snapshot_id="g14r20_i5_b_start",
        capture_kind="g14r20_i5_b_interpreter_acceptance_start",
        capture_source="run_restricted_recovery_interpreter_acceptance.py",
    )
    write_create_only(output_root / "protection_start.json", protection_start)

    head = git_value("HEAD")
    tree = git_value("HEAD^{tree}")
    prepare_command = [
        str(launch),
        str(PREPARE),
        "--action",
        "prepare",
        "--original-request-path",
        str(ORIGINAL_REQUEST_PATH),
        "--original-project-grant-path",
        str(ORIGINAL_PROJECT_GRANT_PATH),
        "--original-run-root",
        str(ORIGINAL_RUN_ROOT),
        "--recovery-execution-id",
        args.recovery_execution_id,
        "--recovery-root",
        str(recovery_root),
        "--executor-checkout",
        str(ROOT),
        "--executor-commit",
        head,
        "--python-executable",
        str(launch),
        "--output-path",
        str(request_path),
    ]
    prepared = run(prepare_command, environment=environment)
    if prepared["return_code"] != 0 or not request_path.is_file():
        raise RestrictedRecoveryError(
            "production request builder failed: " + str(prepared["stderr"])
        )
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    request_validation = validate_restricted_recovery_request(request, check_live=True)
    live_validation = validate_recovery_executor_live(request)
    execution = request["recovery_execution"]
    commands = execution["command_plan"]["commands"]
    command_python = str(commands[0][0])
    if any(command[0] != command_python for command in commands):
        raise RestrictedRecoveryError("production commands do not share one interpreter")
    if command_python != str(launch):
        raise RestrictedRecoveryError("production builder replaced the venv launch path")
    staged_commands = []
    for cell_id in ALLOWED_CELL_IDS:
        _, _, _, builder, _ = restricted_cell_layout(request, cell_id)
        staged = builder(recovery_root / ".acceptance_placeholder", cell_id)
        if staged[0] != command_python:
            raise RestrictedRecoveryError("cell builder replaced the venv launch path")
        staged_commands.append(staged)

    probe_code = r'''
import hashlib, importlib, importlib.metadata as md, json, os, site, sys
request = json.load(open(os.environ["PPO_MEC_I5B_REQUEST"], encoding="utf-8"))
context = request["recovery_execution"]["resolved_execution_context"]
from src.runtime.resolved_formal_execution_context import resolved_python_for_nested_consumer
packages = sorted(
    (dist.metadata.get("Name", "").strip().lower(), dist.version)
    for dist in md.distributions() if dist.metadata.get("Name")
)
package_text = "\n".join(f"{name}=={version}" for name, version in packages)
modules = {}
for name in (
    "src.runtime.restricted_recovery",
    "scripts.run_typed_model_cache_formal_support",
    "scripts.benchmark_main_results",
):
    module = importlib.import_module(name)
    modules[name] = str(module.__file__)
print(json.dumps({
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "site_packages": site.getsitepackages(),
    "installed_packages": packages,
    "dependency_fingerprint": hashlib.sha256(package_text.encode()).hexdigest(),
    "module_import_origins": modules,
    "nested_resolved_python": resolved_python_for_nested_consumer(
        context, observed_sys_executable=sys.executable
    ),
}, sort_keys=True))
'''
    probe_environment = dict(environment, PPO_MEC_I5B_REQUEST=str(request_path))
    child_probe = run([command_python, "-c", probe_code], environment=probe_environment)
    if child_probe["return_code"] != 0:
        raise RestrictedRecoveryError("real venv child probe failed: " + child_probe["stderr"])
    child_identity = json.loads(child_probe["stdout"])
    binding = execution["python_environment_binding"]
    expected_environment = binding["environment_identity"]
    if (
        child_identity["sys_executable"] != command_python
        or child_identity["sys_prefix"] != binding["observed_sys_prefix"]
        or child_identity["sys_base_prefix"] != binding["observed_sys_base_prefix"]
        or child_identity["nested_resolved_python"] != command_python
        or child_identity["dependency_fingerprint"]
        != expected_environment["dependency_fingerprint"]
        or any(
            not Path(origin).resolve().is_relative_to(ROOT.resolve())
            for origin in child_identity["module_import_origins"].values()
        )
    ):
        raise RestrictedRecoveryError("real child interpreter/environment identity drift")

    support_help = run([command_python, str(SUPPORT), "--help"], environment=environment)
    benchmark_help = run([command_python, str(BENCHMARK), "--help"], environment=environment)
    for label, result in (("support", support_help), ("nested benchmark", benchmark_help)):
        if result["return_code"] != 0 or "usage:" not in result["stdout"].lower():
            raise RestrictedRecoveryError(f"{label} real --help subprocess failed")
    if recovery_root.exists():
        raise RestrictedRecoveryError("interpreter checks created the future recovery root")

    negative_cases: list[dict[str, Any]] = []
    system_python = str(binding["binary_realpath_audit_only"])
    for label, candidate in (("system_python", system_python),):
        negative_output = output_root / f"{label}_request_should_not_exist.json"
        negative_command = list(prepare_command)
        negative_command[negative_command.index("--python-executable") + 1] = candidate
        negative_command[negative_command.index("--output-path") + 1] = str(negative_output)
        result = run(negative_command, environment=environment)
        if result["return_code"] == 0 or negative_output.exists() or recovery_root.exists():
            raise RestrictedRecoveryError(f"{label} was not rejected before write")
        negative_cases.append(
            {
                "label": label,
                "status": "rejected_by_production_builder_before_request_or_recovery_write",
                "candidate": candidate,
                "return_code": result["return_code"],
                "stderr": result["stderr"],
                "request_exists": negative_output.exists(),
                "recovery_root_exists": recovery_root.exists(),
            }
        )
    with tempfile.TemporaryDirectory(prefix="g14r20_i5b_wrong_venv_") as temporary:
        wrong_root = Path(temporary) / "wrong_venv"
        created = run(
            [command_python, "-m", "venv", "--without-pip", str(wrong_root)],
            environment=environment,
        )
        if created["return_code"] != 0:
            raise RestrictedRecoveryError("could not create isolated wrong-venv negative")
        wrong_python = wrong_root / "bin/python"
        wrong_output = output_root / "wrong_venv_request_should_not_exist.json"
        wrong_command = list(prepare_command)
        wrong_command[wrong_command.index("--python-executable") + 1] = str(wrong_python)
        wrong_command[wrong_command.index("--output-path") + 1] = str(wrong_output)
        rejected = run(wrong_command, environment=environment)
        if rejected["return_code"] == 0 or wrong_output.exists() or recovery_root.exists():
            raise RestrictedRecoveryError("wrong virtual environment was not rejected")
        negative_cases.append(
            {
                "label": "wrong_virtual_environment",
                "status": "rejected_by_production_builder_before_request_or_recovery_write",
                "return_code": rejected["return_code"],
                "stderr": rejected["stderr"],
                "request_exists": wrong_output.exists(),
                "recovery_root_exists": recovery_root.exists(),
            }
        )
    negative_cases.extend(
        [
            rejected_request_case(
                request,
                lambda item: item["python_environment_binding"][
                    "environment_identity"
                ].update(dependency_fingerprint="0" * 64),
                label="dependency_environment_drift",
                recovery_root=recovery_root,
            ),
            rejected_request_case(
                request,
                lambda item: item["resolved_execution_context"][
                    "runtime_location"
                ].update(resolved_python_absolute_path=system_python),
                label="context_child_interpreter_mismatch",
                recovery_root=recovery_root,
            ),
        ]
    )
    if any(row["recovery_root_exists"] for row in negative_cases):
        raise RestrictedRecoveryError("a negative case wrote the future recovery root")

    command_package = {
        "version": "1.0.0",
        "producer": "final production restricted-recovery builder",
        "executor_checkout": str(ROOT.resolve()),
        "executor_commit": head,
        "executor_git_tree": tree,
        "authorization_request_sha256": request["authorization_request_sha256"],
        "python_launch_path": command_python,
        "python_binary_realpath_audit_only": system_python,
        "production_commands": commands,
        "production_command_plan_sha256": execution["command_plan_sha256"],
        "cell_builder_commands": staged_commands,
        "support_help_command": support_help["command"],
        "nested_benchmark_help_command": benchmark_help["command"],
        "command_builder_replaced_launch_path": False,
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    write_create_only(output_root / "command_package.json", command_package)
    write_create_only(output_root / "negative_cases.json", negative_cases)
    write_create_only(
        output_root / "interpreter_acceptance.json",
        {
            "version": "1.0.0",
            "status": "pass",
            "request_validation": request_validation,
            "live_executor_validation": live_validation,
            "child_identity": child_identity,
            "environment_identity": expected_environment,
            "runtime_audit": binding["runtime_audit"],
            "support_help": support_help,
            "nested_benchmark_help": benchmark_help,
            "scientific_rollout_count": 0,
            "model_load_count": 0,
            "help_process_count": 2,
            "real_interpreter_probe_process_count": 1,
            "synthetic_child_adapter_used_for_interpreter_acceptance": False,
            "future_recovery_root_exists": recovery_root.exists(),
        },
    )
    after = audit_original_recovery_source()
    protection_end = build_protected_snapshot(
        ORIGINAL_PROJECT_ROOT,
        snapshot_id="g14r20_i5_b_end",
        capture_kind="g14r20_i5_b_interpreter_acceptance_end",
        capture_source="run_restricted_recovery_interpreter_acceptance.py",
    )
    write_create_only(output_root / "protection_end.json", protection_end)
    protected_comparison = compare_protected_snapshots(protection_start, protection_end)
    if (
        stable_source_audit_projection(before) != stable_source_audit_projection(after)
        or protected_comparison["status"] != "pass"
    ):
        raise RestrictedRecoveryError("protected source changed during acceptance")
    summary = {
        "version": "1.0.0",
        "goal": "G14R20-I5-B",
        "status": "READY_FOR_CENTRAL_REVIEW_NOT_AUTHORIZED_FOR_RECOVERY",
        "executor_commit": head,
        "executor_git_tree": tree,
        "authorization_request_sha256": request["authorization_request_sha256"],
        "environment_fingerprint": expected_environment["environment_fingerprint"],
        "dependency_fingerprint": expected_environment["dependency_fingerprint"],
        "negative_case_count": len(negative_cases),
        "protected_source_unchanged": True,
        "protected_seven_files_unchanged": True,
        "historical_start_evidence": "unavailable",
        "historical_protection_verdict": "UNVERIFIED",
        "old_i5_i5_a_packages_modified": False,
        "old_i5_non_executable_reason": "production request resolved the venv launcher symlink to system Python",
        "old_i5_a_non_executable_reason": "acceptance runner preserved its venv, but production recovery request still resolved the launcher symlink",
        "future_recovery_root": str(recovery_root),
        "future_recovery_root_exists": recovery_root.exists(),
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    write_create_only(output_root / "acceptance_summary.json", summary)
    report = (
        "# G14R20-I5-B interpreter acceptance\n\n"
        f"- status: `{summary['status']}`\n"
        f"- executor commit/tree: `{head}` / `{tree}`\n"
        f"- launch path: `{command_python}`\n"
        f"- audit-only binary realpath: `{system_python}`\n"
        f"- environment/dependency fingerprint: `{summary['environment_fingerprint']}` / "
        f"`{summary['dependency_fingerprint']}`\n"
        "- production request/context/cell builder/support/nested interpreter chain: pass\n"
        "- real support and nested benchmark `--help`: pass; rollout/model load: 0/0\n"
        "- system Python, wrong venv, dependency drift, context/child mismatch: rejected before recovery write\n"
        "- historical protection: `UNVERIFIED` because task-start evidence remains unavailable\n"
        "- recovery grant/real recovery/holdout: false/false/false\n"
    )
    report_path = output_root / "acceptance_report.md"
    with report_path.open("xb") as stream:
        stream.write(report.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())

    inventory = []
    for path in sorted(output_root.iterdir()):
        if path.name == "artifact_integrity.json" or not path.is_file():
            continue
        inventory.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    integrity = {
        "version": "1.0.0",
        "artifact_root": str(output_root),
        "executor_commit": head,
        "executor_git_tree": tree,
        "file_count": len(inventory),
        "files": inventory,
        "inventory_sha256": canonical_sha256(inventory),
    }
    write_create_only(output_root / "artifact_integrity.json", integrity)
    require_clean_checkout()
    if recovery_root.exists():
        raise RestrictedRecoveryError("future recovery root exists after acceptance")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
