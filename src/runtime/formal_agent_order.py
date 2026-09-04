"""Versioned identity and fail-fast validation for formal benchmark agent order."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.runtime.formal_invalid_run_registry import (
    PermanentlyInvalidFormalReferenceError,
    reject_permanently_invalid_formal_references,
)


FORMAL_AGENT_ORDER_CONTRACT_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_AGENT_ORDER_CONTRACT_PATH = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
    / "formal_agent_order_contract.json"
)


class FormalAgentOrderError(ValueError):
    """Raised when an agent set, role, or order differs from the frozen contract."""


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalAgentOrderError(f"non-finite order-contract value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FormalAgentOrderError(f"non-string order-contract key at {path}")
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def canonical_sha256(value: Any) -> str:
    _reject_non_finite(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in contract.items()
        if key != "semantic_sha256"
    }


def load_formal_agent_order_contract(
    path: str | Path = DEFAULT_FORMAL_AGENT_ORDER_CONTRACT_PATH,
) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalAgentOrderError(f"duplicate order-contract JSON key: {key}")
            result[key] = value
        return result

    target = Path(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8-sig"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FormalAgentOrderError(f"non-finite order-contract constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalAgentOrderError(f"unable to load formal agent order contract: {target}") from exc
    if not isinstance(payload, dict):
        raise FormalAgentOrderError("formal agent order contract must be a JSON object")
    validate_formal_agent_order_contract(payload)
    return payload


def _validate_unique_order(value: Any, field: str, expected_count: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise FormalAgentOrderError(f"{field} must be a non-empty string list")
    if len(value) != expected_count or len(set(value)) != expected_count:
        raise FormalAgentOrderError(
            f"{field} must contain exactly {expected_count} unique agents"
        )
    return list(value)


def validate_formal_agent_order_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    _reject_non_finite(contract)
    expected_fields = {
        "formal_agent_order_contract_version",
        "identity_rules",
        "reactive_agent_order",
        "learned_agent_order",
        "main_benchmark_agent_order",
        "checkpoint_free_report_only_agent_roles",
        "row_display_order",
        "pairwise_statistics_identity",
        "permanently_rejected_run_ids",
        "semantic_sha256",
    }
    if set(contract) != expected_fields:
        raise FormalAgentOrderError("formal agent order contract has missing or unknown fields")
    if contract.get("formal_agent_order_contract_version") != FORMAL_AGENT_ORDER_CONTRACT_VERSION:
        raise FormalAgentOrderError("unsupported formal agent order contract version")
    reactive = _validate_unique_order(contract.get("reactive_agent_order"), "reactive_agent_order", 5)
    learned = _validate_unique_order(contract.get("learned_agent_order"), "learned_agent_order", 10)
    main = _validate_unique_order(contract.get("main_benchmark_agent_order"), "main_benchmark_agent_order", 15)
    if main != [*reactive, *learned]:
        raise FormalAgentOrderError("main benchmark order must be reactive order followed by learned order")
    if set(reactive) & set(learned):
        raise FormalAgentOrderError("reactive and learned roles overlap")
    if "popularity_cache_heuristic" in main:
        raise FormalAgentOrderError("report-only popularity heuristic entered the main benchmark order")
    report_only = contract.get("checkpoint_free_report_only_agent_roles")
    if not isinstance(report_only, list) or not report_only:
        raise FormalAgentOrderError("checkpoint-free/report-only roles are missing")
    report_names: list[str] = []
    for row in report_only:
        if not isinstance(row, Mapping) or set(row) != {"agent", "role"}:
            raise FormalAgentOrderError("checkpoint-free/report-only role schema drift")
        name = row.get("agent")
        role = row.get("role")
        if not isinstance(name, str) or not name or not isinstance(role, str) or not role:
            raise FormalAgentOrderError("checkpoint-free/report-only role is invalid")
        report_names.append(name)
    if len(report_names) != len(set(report_names)) or set(report_names) & set(main):
        raise FormalAgentOrderError("checkpoint-free/report-only role membership drift")
    if contract.get("row_display_order") != main:
        raise FormalAgentOrderError("row display order differs from main benchmark identity")
    pairwise = contract.get("pairwise_statistics_identity")
    expected_pairwise = {
        "candidate_agent": "sa_ghmappo",
        "baseline_agent_order": [name for name in main if name != "sa_ghmappo"],
        "comparison_key_order": ["candidate_agent", "baseline_agent", "metric"],
    }
    if pairwise != expected_pairwise:
        raise FormalAgentOrderError("pairwise/statistics comparison identity drift")
    rules = contract.get("identity_rules")
    if not isinstance(rules, Mapping) or any(rules.get(name) is not True for name in (
        "json_object_insertion_order_is_identity_forbidden",
        "alphabetical_order_inference_forbidden",
        "same_set_different_order_rejected",
        "single_resolver_required",
    )):
        raise FormalAgentOrderError("formal agent order identity rules are incomplete")
    rejected = contract.get("permanently_rejected_run_ids")
    if not isinstance(rejected, list) or len(rejected) != len(set(rejected)):
        raise FormalAgentOrderError("permanently rejected run IDs must be a unique list")
    expected_hash = canonical_sha256(contract_projection(contract))
    if contract.get("semantic_sha256") != expected_hash:
        raise FormalAgentOrderError("formal agent order contract semantic SHA-256 mismatch")
    return {
        "status": "pass",
        "formal_agent_order_contract_version": FORMAL_AGENT_ORDER_CONTRACT_VERSION,
        "semantic_sha256": expected_hash,
        "reactive_agent_order": reactive,
        "learned_agent_order": learned,
        "main_benchmark_agent_order": main,
        "row_display_order": list(main),
        "statistics_candidate_agent": "sa_ghmappo",
        "statistics_baseline_agent_order": expected_pairwise["baseline_agent_order"],
    }


def _exact_order(observed: Sequence[Any], expected: Sequence[str], field: str) -> None:
    values = [str(value) for value in observed]
    if len(values) != len(set(values)):
        raise FormalAgentOrderError(f"{field} contains duplicate agents")
    if values != list(expected):
        if set(values) == set(expected):
            raise FormalAgentOrderError(f"{field} has the correct set but wrong order")
        missing = [name for name in expected if name not in values]
        extra = [name for name in values if name not in expected]
        raise FormalAgentOrderError(
            f"{field} membership drift: missing={missing}, extra={extra}"
        )


def _argv_agents(argv: Sequence[Any], field: str) -> list[str]:
    tokens = [str(token) for token in argv]
    if "--agents" not in tokens:
        raise FormalAgentOrderError(f"{field} lacks --agents")
    start = tokens.index("--agents") + 1
    end = next((index for index in range(start, len(tokens)) if tokens[index].startswith("--")), len(tokens))
    return tokens[start:end]


def resolve_formal_agent_order(
    *,
    contract: Mapping[str, Any] | None = None,
    contract_path: str | Path | None = None,
    protocol: Mapping[str, Any] | None = None,
    scientific_config: Mapping[str, Any] | None = None,
    fairness_manifests: Sequence[Mapping[str, Any]] = (),
    reactive_baseline_order: Sequence[str] | None = None,
    command_templates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is not None and contract_path is not None:
        raise FormalAgentOrderError("provide contract or contract_path, not both")
    payload = dict(contract) if contract is not None else load_formal_agent_order_contract(
        contract_path or DEFAULT_FORMAL_AGENT_ORDER_CONTRACT_PATH
    )
    audit = validate_formal_agent_order_contract(payload)
    reactive = audit["reactive_agent_order"]
    learned = audit["learned_agent_order"]
    main = audit["main_benchmark_agent_order"]
    checks: list[str] = ["order_contract"]
    if scientific_config is not None:
        _exact_order(scientific_config.get("learned_agent_order", []), learned, "scientific config learned order")
        checks.append("scientific_config")
    if protocol is not None:
        binding = protocol.get("formal_agent_order_contract", {})
        if binding.get("version") != FORMAL_AGENT_ORDER_CONTRACT_VERSION or binding.get(
            "semantic_sha256"
        ) != audit["semantic_sha256"]:
            raise FormalAgentOrderError("Protocol formal agent order contract hash drift")
        controller_rows = protocol.get("agent_matrix", {}).get("controller_table", [])
        learned_rows = [
            str(row.get("agent"))
            for row in controller_rows
            if isinstance(row, Mapping)
            and row.get("training_requirement") == "clean_typed_checkpoint_per_seed_and_capacity"
        ]
        _exact_order(learned_rows, learned, "Protocol learned-agent matrix")
        report_names = [
            str(row.get("agent"))
            for row in controller_rows
            if isinstance(row, Mapping)
            and row.get("training_requirement") != "clean_typed_checkpoint_per_seed_and_capacity"
        ]
        expected_report_names = [
            str(row["agent"])
            for row in payload["checkpoint_free_report_only_agent_roles"]
            if row["role"] != "exact_oracle_cell"
        ]
        _exact_order(report_names, expected_report_names, "Protocol checkpoint-free/report-only roles")
        oracle_names = [
            str(row.get("agent"))
            for row in protocol.get("agent_matrix", {}).get("exact_oracle_cells", [])
            if isinstance(row, Mapping)
        ]
        expected_oracle_names = [
            str(row["agent"])
            for row in payload["checkpoint_free_report_only_agent_roles"]
            if row["role"] == "exact_oracle_cell"
        ]
        _exact_order(oracle_names, expected_oracle_names, "Protocol exact-oracle report order")
        config_names = protocol.get("training_budget", {}).get("agent_configs", {})
        if not isinstance(config_names, Mapping) or set(config_names) != set(learned):
            raise FormalAgentOrderError("Protocol training agent config membership drift")
        _exact_order(
            protocol.get("training_budget", {}).get("learned_agent_order", []),
            learned,
            "Protocol training-budget learned order",
        )
        train_contexts = (
            protocol.get("execution_contract", {})
            .get("command_templates", {})
            .get("train", {})
            .get("matrix_contexts", [])
        )
        if not isinstance(train_contexts, list) or len(train_contexts) != 150:
            raise FormalAgentOrderError("Protocol training command matrix must contain 150 cells")
        training_agent_sequence = list(
            dict.fromkeys(str(row.get("agent")) for row in train_contexts if isinstance(row, Mapping))
        )
        _exact_order(training_agent_sequence, learned, "Protocol training command agent order")
        if any(
            sum(1 for row in train_contexts if row.get("agent") == agent) != 15
            for agent in learned
        ):
            raise FormalAgentOrderError("Protocol training command agent cell count drift")
        statistics = protocol.get("statistics", {})
        if statistics.get("candidate_agent") != "sa_ghmappo":
            raise FormalAgentOrderError("Protocol statistics candidate-agent identity drift")
        _exact_order(
            statistics.get("baseline_agent_order", []),
            audit["statistics_baseline_agent_order"],
            "Protocol statistics baseline order",
        )
        if statistics.get("formal_agent_order_contract_semantic_sha256") != audit["semantic_sha256"]:
            raise FormalAgentOrderError("Protocol statistics order-contract hash drift")
        claims = protocol.get("claim_evidence_map", {})
        _exact_order(
            claims.get("paper_display_agent_order", []),
            main,
            "Protocol paper display order",
        )
        if claims.get("formal_agent_order_contract_semantic_sha256") != audit["semantic_sha256"]:
            raise FormalAgentOrderError("Protocol claim/display order-contract hash drift")
        checks.append("protocol")
        command_templates = command_templates or protocol.get("execution_contract", {}).get(
            "command_templates", {}
        )
    if reactive_baseline_order is not None:
        _exact_order(reactive_baseline_order, reactive, "reactive baseline order")
        checks.append("reactive_baselines")
    for index, manifest in enumerate(fairness_manifests):
        typed = manifest.get("cache_contract", {}).get("typed_model_cache", {})
        _exact_order(typed.get("controller_agents", []), learned, f"fairness controller order[{index}]")
        baseline_names = [
            row.get("agent_identity", {}).get("name")
            for row in manifest.get("baseline_matrix", [])
            if isinstance(row, Mapping)
        ]
        _exact_order(baseline_names, reactive, f"fairness reactive order[{index}]")
        checks.append(f"fairness_manifest[{index}]")
    if command_templates is not None:
        checked_templates = 0
        for phase, spec in command_templates.items():
            if not isinstance(spec, Mapping):
                continue
            argv = spec.get("argv", [])
            if isinstance(argv, list) and "--agents" in argv:
                _exact_order(_argv_agents(argv, f"command template {phase}"), main, f"command template {phase}")
                checked_templates += 1
        if checked_templates == 0:
            raise FormalAgentOrderError("Protocol command templates contain no --agents consumer")
        checks.append(f"command_templates:{checked_templates}")
    return {
        **audit,
        "validated_components": checks,
        "order_audit_semantic_sha256": canonical_sha256(
            {
                "contract_semantic_sha256": audit["semantic_sha256"],
                "reactive_agent_order": reactive,
                "learned_agent_order": learned,
                "main_benchmark_agent_order": main,
                "validated_components": checks,
            }
        ),
    }


def reject_permanently_invalid_run_references(
    paths: Sequence[str | Path], *, contract: Mapping[str, Any] | None = None
) -> None:
    payload = dict(contract) if contract is not None else load_formal_agent_order_contract()
    validate_formal_agent_order_contract(payload)
    rejected = set(payload["permanently_rejected_run_ids"])
    # G14C v8/v9 failed after training but before any valid dev selection.  This
    # implementation-level denylist extends rejection without changing the
    # frozen Agent Order Contract 1.0.0 scientific semantic identity.
    rejected.add("typed_model_cache_formal_20260828_101804_g14c_v8")
    rejected.add("typed_model_cache_formal_20260830_113339_g14c_v9")
    try:
        reject_permanently_invalid_formal_references(paths)
    except PermanentlyInvalidFormalReferenceError as exc:
        raise FormalAgentOrderError(str(exc)) from exc
    for path in paths:
        parts = set(Path(path).resolve().parts)
        hit = sorted(rejected & parts)
        if hit:
            raise FormalAgentOrderError(
                f"permanently invalid formal run reference rejected: {hit[0]}"
            )


__all__ = [
    "DEFAULT_FORMAL_AGENT_ORDER_CONTRACT_PATH",
    "FORMAL_AGENT_ORDER_CONTRACT_VERSION",
    "FormalAgentOrderError",
    "canonical_sha256",
    "contract_projection",
    "load_formal_agent_order_contract",
    "reject_permanently_invalid_run_references",
    "resolve_formal_agent_order",
    "validate_formal_agent_order_contract",
]
