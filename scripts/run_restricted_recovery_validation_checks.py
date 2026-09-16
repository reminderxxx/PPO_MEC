"""Run commit-bound supplemental checks for restricted-recovery acceptance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


PROTECTED_PATHS = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git_value(checkout: Path, revision: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", revision], text=True
    ).strip()


def run_check(name: str, command: list[str], checkout: Path, env: dict[str, str]) -> dict:
    started_at = now_iso()
    result = subprocess.run(
        command,
        cwd=checkout,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "started_at": started_at,
        "completed_at": now_iso(),
        "command": command,
        "return_code": result.returncode,
        "status": "pass" if result.returncode == 0 else "fail",
        "output": result.stdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-checkout", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--receipt-path", required=True)
    args = parser.parse_args()

    checkout = Path(args.executor_checkout).resolve(strict=True)
    python_path = Path(args.python_executable).absolute()
    if not python_path.is_file():
        raise ValueError("python executable is missing")
    python = str(python_path)
    receipt_path = Path(args.receipt_path).resolve()
    if receipt_path.exists():
        raise ValueError("receipt path must be create-only")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = git_value(checkout, args.baseline_commit)
    before_commit = git_value(checkout, "HEAD")
    before_tree = git_value(checkout, "HEAD^{tree}")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/g14r20_i5_a_validation_pycache")
    scoped_status = [
        "git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all",
        "--", "README.md", "docs", "scripts", "src", "tests", "configs",
    ]
    checks = [
        run_check("smoke", [python, "-B", "scripts/smoke_test.py"], checkout, env),
        run_check(
            "compile_import",
            [python, "-B", "-m", "compileall", "-q", "scripts", "src", "tests"],
            checkout,
            env,
        ),
        run_check(
            "target_imports",
            [
                python,
                "-B",
                "-c",
                (
                    "import scripts.build_restricted_recovery_acceptance_artifacts; "
                    "import scripts.build_restricted_recovery_parent_bootstrap_artifacts; "
                    "import scripts.capture_restricted_recovery_protected_snapshot; "
                    "import scripts.run_restricted_recovery_acceptance_tests; "
                    "import src.runtime.restricted_recovery"
                ),
            ],
            checkout,
            env,
        ),
        run_check(
            "diff_check",
            ["git", "-C", str(checkout), "diff", "--check", f"{baseline}..HEAD"],
            checkout,
            env,
        ),
        run_check("clean_scope", scoped_status, checkout, env),
        run_check(
            "protected_scope_diff_check",
            [
                "git", "-C", str(checkout), "diff", "--exit-code", baseline, "HEAD", "--",
                *PROTECTED_PATHS,
            ],
            checkout,
            env,
        ),
    ]
    # `git status --porcelain` returns zero even when it prints changes.
    clean_scope = next(check for check in checks if check["name"] == "clean_scope")
    if clean_scope["output"].strip():
        clean_scope["status"] = "fail"
    after_commit = git_value(checkout, "HEAD")
    after_tree = git_value(checkout, "HEAD^{tree}")
    identity_unchanged = before_commit == after_commit and before_tree == after_tree
    overall = all(check["status"] == "pass" for check in checks) and identity_unchanged
    receipt = {
        "restricted_recovery_validation_receipt_version": "1.0.0",
        "started_at": checks[0]["started_at"],
        "completed_at": now_iso(),
        "executor_checkout": str(checkout),
        "executor_commit": before_commit,
        "executor_git_tree": before_tree,
        "executor_identity_unchanged": identity_unchanged,
        "baseline_commit": baseline,
        "isolated_python_executable": python,
        "checks": checks,
        "status": "pass" if overall else "fail",
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
