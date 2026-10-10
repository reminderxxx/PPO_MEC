"""Verify the public-rule v2 feasibility filter on a frozen public state."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA_VERSION = "cscwd_public_rule_feasibility_fix_audit_v1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_estimator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_cscwd_rule_fix_estimator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load estimator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(estimator_path: Path, public_info_path: Path, phase_audit_path: Path) -> dict[str, Any]:
    module = _load_estimator(estimator_path)
    snapshot = json.loads(public_info_path.read_text(encoding="utf-8"))
    phase_audit = json.loads(phase_audit_path.read_text(encoding="utf-8"))
    if snapshot["phase_artifact_sha256"] != _sha256(phase_audit_path):
        raise RuntimeError("public snapshot is not bound to the phase audit")
    info = snapshot["info"]
    before = _canonical_sha(info)
    estimate = module.estimate_public_actions(
        info["semantic_state"],
        info["action_mask"],
    )
    classification = module.classify_public_rule_candidates(estimate)
    immediate = int(module.CausalPublicImmediateRule().select_action_from_info(info))
    two_step = int(module.CausalPublicTwoStepRule().select_action_from_info(info))
    after = _canonical_sha(info)

    old_rows = [
        row
        for row in phase_audit["rows"]
        if row["public_state_sha256"] == snapshot["original_full_info_sha256"]
        and row["method"]
        in {"causal_public_immediate_rule", "causal_public_two_step_rule"}
    ]
    if len(old_rows) != 2 or any(
        row["public_action4"]["raw_full_step_contact_fit"] != "no"
        for row in old_rows
    ):
        raise RuntimeError("exact pre-fix public-rule witness drift")
    if classification["4"] != "excluded_explicit_raw_contact_infeasible":
        raise RuntimeError("v2 rule did not exclude exact known-infeasible action 4")
    if immediate != 2 or two_step != 2:
        raise RuntimeError("v2 rule did not use the contact-independent fallback")
    if before != after:
        raise RuntimeError("public rule mutated the frozen public info")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "estimator_sha256": _sha256(estimator_path),
            "public_info_sha256": _sha256(public_info_path),
            "phase_audit_sha256": _sha256(phase_audit_path),
            "original_full_info_sha256": snapshot["original_full_info_sha256"],
            "window_id": snapshot["window_id"],
            "design_id": snapshot["design_id"],
            "step": snapshot["step"],
        },
        "rule_identity": {
            "schema_version": module.PUBLIC_RULE_SCHEMA_VERSION,
            "capability_profile": module.CausalPublicImmediateRule.capability_profile,
            "objective_profile": module.CausalPublicImmediateRule.objective_profile,
        },
        "pre_fix_exact_rows": [
            {
                "method": row["method"],
                "selected_action": 4,
                "raw_full_step_contact_fit": row["public_action4"][
                    "raw_full_step_contact_fit"
                ],
                "public_contact_budget_seconds": row[
                    "public_contact_budget_seconds"
                ],
                "raw_recorded_step_cost_seconds": row[
                    "raw_recorded_step_cost_seconds"
                ],
                "raw_rejection_reason": row["raw_rejection_reason"],
            }
            for row in old_rows
        ],
        "post_fix_exact_state": {
            "action_classification": classification,
            "action_raw_full_step_contact_fit": {
                key: row["raw_full_step_contact_fit"]
                for key, row in estimate["actions"].items()
            },
            "immediate_selected_action": immediate,
            "two_step_selected_action": two_step,
            "fallback_action": 2,
            "public_info_unchanged": before == after,
            "hidden_future_fields_consumed": False,
        },
        "interpretation": {
            "corrected": (
                "both public rules now reject every non-fallback action explicitly known "
                "not to fit raw full-step contact and choose legal vehicle fallback"
            ),
            "not_corrected": (
                "the frozen raw atomic contact contract and migration reachability remain unchanged"
            ),
        },
        "claim_boundary": {
            "raw_rows_read": 0,
            "environment_steps": 0,
            "training_steps": 0,
            "checkpoint_selection": False,
            "performance_claim": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--public-info", type=Path, required=True)
    parser.add_argument("--phase-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.estimator, args.public_info, args.phase_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
