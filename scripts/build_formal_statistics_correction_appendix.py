"""Build an immutable correction appendix from existing formal benchmark rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manage_typed_model_cache_formal_artifacts import _claim_evidence_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-statistics-path", required=True)
    parser.add_argument("--source-formal-gate-path", required=True)
    parser.add_argument("--formal-agent-order-contract-path", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ordered_unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    source_statistics_path = Path(args.source_statistics_path).resolve()
    source_gate_path = Path(args.source_formal_gate_path).resolve()
    order_contract_path = Path(args.formal_agent_order_contract_path).resolve()
    output_root = Path(args.output_root).resolve()
    source_artifact_root = source_statistics_path.parent.parent
    if output_root == source_artifact_root or output_root.is_relative_to(source_artifact_root):
        raise ValueError("correction appendix must be outside the immutable source artifact root")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("correction appendix output root must be absent or empty")

    source_statistics = read_json(source_statistics_path)
    source_gate = read_json(source_gate_path)
    source_rows = [Path(item).resolve() for item in source_statistics.get("source_rows_path", [])]
    if not source_rows or any(not path.is_file() for path in source_rows):
        raise ValueError("source statistics rows are missing")
    if not order_contract_path.is_file():
        raise ValueError("formal agent order contract is missing")

    immutable_inputs = [source_statistics_path, source_gate_path, order_contract_path, *source_rows]
    input_hashes_before = {str(path): sha256_file(path) for path in immutable_inputs}
    rows = list(source_statistics.get("rows", []))
    if not rows:
        raise ValueError("source statistics contains no comparison rows")
    candidate_agents = ordered_unique([row.get("candidate_agent") for row in rows])
    if len(candidate_agents) != 1 or not candidate_agents[0]:
        raise ValueError("source statistics candidate identity is ambiguous")
    baseline_agents = ordered_unique([row.get("baseline_agent") for row in rows])
    metrics = ordered_unique([row.get("metric") for row in rows])
    if any(not value for value in [*baseline_agents, *metrics]):
        raise ValueError("source statistics comparison identity is incomplete")

    command = [sys.executable, str(ROOT / "scripts/analyze_top_journal_statistics.py")]
    for path in source_rows:
        command.extend(["--rows_path", str(path)])
    command.extend(
        [
            "--candidate_agent",
            str(candidate_agents[0]),
            "--baseline_agents",
            *[str(item) for item in baseline_agents],
            "--metrics",
            *[str(item) for item in metrics],
            "--pair_keys",
            *[str(item) for item in source_statistics.get("pair_keys", [])],
            "--outer_cluster_keys",
            *[str(item) for item in source_statistics.get("outer_cluster_keys", [])],
            "--inner_cluster_keys",
            *[str(item) for item in source_statistics.get("inner_cluster_keys", [])],
            "--ci_method",
            str(source_statistics.get("requested_ci_method", "bca")),
            "--bootstrap_samples",
            str(int(source_statistics.get("bootstrap_samples", 10000))),
            "--random_seed",
            "1401",
            "--formal-agent-order-contract-path",
            str(order_contract_path),
            "--output_root",
            str(output_root),
        ]
    )
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    corrected_path = output_root / "paired_statistics.json"
    corrected = read_json(corrected_path)
    corrected["correction_appendix"] = {
        "source_statistics_path": str(source_statistics_path),
        "source_statistics_sha256": input_hashes_before[str(source_statistics_path)],
        "source_formal_gate_path": str(source_gate_path),
        "source_formal_gate_sha256": input_hashes_before[str(source_gate_path)],
        "source_artifact_mutated": False,
    }
    write_json(corrected_path, corrected)

    stable_fields = [
        "candidate_agent",
        "baseline_agent",
        "metric",
        "available_paired_count",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "raw_mean_delta_candidate_minus_baseline",
        "raw_ci95_low",
        "raw_ci95_high",
    ]
    original_by_identity = {
        (row.get("candidate_agent"), row.get("baseline_agent"), row.get("metric")): row
        for row in rows
    }
    corrected_by_identity = {
        (row.get("candidate_agent"), row.get("baseline_agent"), row.get("metric")): row
        for row in corrected.get("rows", [])
    }
    if original_by_identity.keys() != corrected_by_identity.keys():
        raise ValueError("corrected comparison identity differs from source statistics")
    stable_mismatches = []
    for identity, original_row in original_by_identity.items():
        corrected_row = corrected_by_identity[identity]
        for field in stable_fields:
            if original_row.get(field) != corrected_row.get(field):
                stable_mismatches.append(
                    {
                        "identity": identity,
                        "field": field,
                        "source": original_row.get(field),
                        "corrected": corrected_row.get(field),
                    }
                )
    if stable_mismatches:
        raise ValueError(f"non-sign-test statistics drift: {stable_mismatches[:3]}")

    original_claim_rows = list(source_gate.get("claim_evidence_map", []))
    corrected_claim_rows = _claim_evidence_rows(corrected)
    correction = {
        "correction_contract_version": "formal_statistics_correction_appendix_v1.0.0",
        "source_gate_claim_evidence": original_claim_rows,
        "corrected_claim_evidence": corrected_claim_rows,
        "source_status_counts": dict(Counter(row.get("status") for row in original_claim_rows)),
        "corrected_status_counts": dict(Counter(row.get("status") for row in corrected_claim_rows)),
        "corrected_holm_rejection_count_alpha_0_05": sum(
            1
            for row in corrected.get("rows", [])
            if isinstance(row.get("holm_sign_test_pvalue"), (int, float))
            and float(row["holm_sign_test_pvalue"]) < 0.05
        ),
        "row_count": len(corrected_claim_rows),
    }
    write_json(output_root / "claim_evidence_correction.json", correction)

    input_hashes_after = {str(path): sha256_file(path) for path in immutable_inputs}
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("immutable source artifact changed during correction generation")
    provenance = {
        "correction_contract_version": "formal_statistics_correction_appendix_v1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_git_commit": git_head(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "immutable_source_hashes": input_hashes_before,
        "source_artifact_mutated": False,
        "recompute_command": command,
        "stable_non_sign_test_fields_match_source": True,
        "corrected_statistics_protocol_version": corrected.get("statistics_protocol_version"),
        "sign_test_unit_rule": corrected.get("sign_test_unit_rule"),
    }
    write_json(output_root / "correction_provenance.json", provenance)

    artifact_paths = sorted(
        path for path in output_root.iterdir() if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    manifest = {
        "artifact_manifest_version": "1.0.0",
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(output_root / "artifact_integrity_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "output_root": str(output_root),
                "source_artifact_mutated": False,
                "corrected_status_counts": correction["corrected_status_counts"],
                "holm_rejection_count": correction["corrected_holm_rejection_count_alpha_0_05"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
