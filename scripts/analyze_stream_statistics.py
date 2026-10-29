"""Strict paired stream statistics; v2 supports synthetic software QA only.

No environment, agent, reward, legacy statistics or scientific artifact is imported.
Native rows and physical eligibility must be exported under a frozen manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

PLAN_SCHEMA = "strict_stream_statistics_plan_v2"
MANIFEST_SCHEMA = "strict_stream_statistics_manifest_v2"
NATIVE_SCHEMA = "strict_stream_native_row_v2"
ROW_SCHEMA = "strict_stream_canonical_row_v2"
BUNDLE_SCHEMA = "strict_stream_statistics_bundle_v2"
RESULT_SCHEMA = "strict_stream_statistics_result_v2"
INTERVAL_SEMANTICS = "closed_native_sample_ticks"
RESAMPLING_DESIGN = "global_crossed_seed__block_window_nested_regime_v2"
ESTIMAND = "equal_physical_blocks_equal_windows_weighted_global_seed_window_regime_v2"
KIND = "synthetic_software_fixture"
BOOTSTRAP_DRAWS = 10000
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_WALL_SECONDS = 7200
METRICS = {
    "native_return": ("reward", "higher", "native_reward_units", False),
    "completed_requests": ("completed_requests", "higher", "delivered_requests", True),
    "unused_optional_transfer_bytes": ("optional_unused_transfer_bytes", "lower", "bytes", True),
    "state_transfer_bytes": ("state_transfer_bytes", "lower", "bytes", True),
    "state_transfer_link_frames": ("state_transfer_link_frames", "lower", "link_frames", True),
    "handoff_failed_count": ("handoff_failed_count", "lower", "events", True),
}
PHYSICAL_FIELDS = (
    "raw_source_file_sha256", "acquisition_origin_id", "source_segment_run_id",
    "clock_kind", "tick_duration_numerator", "tick_duration_denominator",
    "raw_start_tick", "raw_end_tick", "verified_outer_time_unit_id",
)
EXOGENOUS_FIELDS = (
    "split_id", "workload_regime_id", "request_schedule_fingerprint",
    "catalogue_fingerprint", "initial_cache_state_fingerprint",
    "rsu_geometry_fingerprint", "cost_model_fingerprint", "evaluation_policy_mode",
)
PROVENANCE_FIELDS = (
    "source_output_file_sha256", "source_row_id", "input_manifest_sha256",
    "checkpoint_manifest_sha256", "eligibility_audit_sha256", "native_metric_ledger_sha256",
)


class ContractError(ValueError):
    """A missing, conflicting or unsupported scientific identity is fatal."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), "invalid SHA256: " + field)
    return value


def text(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), "missing text: " + field)
    return value


def integer(value: Any, field: str) -> int:
    require(type(value) is int, "integer required (no float truncation): " + field)
    return value


def rational(value: Any, field: str, positive: bool = False) -> Fraction:
    require(isinstance(value, list) and len(value) == 2, "rational [numerator,denominator] required: " + field)
    n, d = integer(value[0], field), integer(value[1], field)
    require(d > 0 and math.gcd(n, d) == 1, "noncanonical rational: " + field)
    result = Fraction(n, d)
    require(not positive or result > 0, "positive rational required: " + field)
    return result


def fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def exact_keys(obj: Any, keys: Any, label: str) -> None:
    require(isinstance(obj, dict) and set(obj) == set(keys), "exact fields required: " + label)


def unique(values: list[Any], label: str) -> None:
    require(len(values) == len(set(values)), "duplicate " + label)


def descriptor(value: dict, label: str) -> None:
    exact_keys(value, ("path", "sha256"), label)
    require(Path(text(value["path"], label + "/path")).is_absolute(), "absolute input path required")
    sha(value["sha256"], label + "/sha256")


def validate_contract(plan: dict, manifest: dict) -> dict:
    require(plan["schema"] == PLAN_SCHEMA and manifest["schema"] == MANIFEST_SCHEMA, "unsupported plan/manifest schema")
    require(plan["evidence_kind"] == manifest["evidence_kind"] == KIND,
            "v2 is software-only; scientific input requires a separately qualified contract")
    require(plan["analysis_mode"] == "software_only" and plan["paper_goal_complete"] is False,
            "software qualification cannot promote a scientific claim")
    require(integer(plan["bootstrap_draws"], "bootstrap_draws") == BOOTSTRAP_DRAWS, "exactly 10000 bootstrap draws required")
    require(0 <= integer(plan["random_seed"], "random_seed") < 2 ** 64, "invalid public bootstrap RNG seed")
    require(integer(plan["fixed_final_episode"], "fixed_final_episode") >= 0, "invalid fixed final episode")
    descriptor(plan["manifest"], "manifest")
    require(isinstance(plan["native_inputs"], list) and 1 <= len(plan["native_inputs"]) <= 16, "invalid native file matrix")
    for item in plan["native_inputs"]:
        descriptor(item, "native_inputs")
    unique([item["path"] for item in plan["native_inputs"]], "native input path")
    unique([item["sha256"] for item in plan["native_inputs"]], "native input SHA")
    require(plan["confidence_level"] == 0.95, "frozen 95% confidence level required")
    require(plan["unavailable_policy"] == "entire_hypothesis_unavailable_keep_full_family", "unsupported missing-endpoint handling")
    require(plan["sign_tie_tolerance"] == 0, "v2 uses exact zero sign ties")
    require(manifest["interval_semantics"] == INTERVAL_SEMANTICS, "unsupported/missing native interval semantics")
    require(plan["resampling_design"] == RESAMPLING_DESIGN, "unsupported global seed resampling design")
    seeds = plan["training_seeds"]
    require(isinstance(seeds, list) and 1 <= len(seeds) <= 16, "invalid seed matrix")
    for seed in seeds:
        integer(seed, "training_seed")
    unique(seeds, "seed")
    regimes = plan["workload_regimes"]
    require(isinstance(regimes, list) and 1 <= len(regimes) <= 16, "invalid workload matrix")
    for regime in regimes:
        text(regime, "workload_regime_id")
    unique(regimes, "workload regime")
    weights = {}
    for name, names in (("seed_weights", [str(s) for s in seeds]), ("regime_weights", regimes)):
        exact_keys(plan[name], names, name)
        values = [rational(plan[name][key], name + "/" + key, positive=True) for key in names]
        require(sum(values) == 1, "weights must sum exactly to one: " + name)
        weights[name] = np.asarray([float(v) for v in values])
    require(plan["estimand"] == ESTIMAND,
            "unknown estimand or weighting")
    metrics = plan["metrics"]
    require(isinstance(metrics, list) and 1 <= len(metrics) <= len(METRICS), "invalid metric family")
    metric_ids = []
    for metric in metrics:
        exact_keys(metric, ("id", "native_field", "direction", "unit"), "metric")
        key = metric["id"]
        require(key in METRICS, "unknown metric")
        require(tuple(metric[f] for f in ("native_field", "direction", "unit")) == METRICS[key][:3],
                "native metric direction/unit mismatch: " + key)
        metric_ids.append(key)
    unique(metric_ids, "metric")
    for field in ("eligibility_audit_sha256", "checkpoint_manifest_sha256", "native_metric_ledger_sha256", "reward_contract_sha256"):
        sha(manifest[field], field)
    require(manifest["physical_eligibility_status"] == "synthetic_qualification_only",
            "this task cannot certify real physical eligibility")
    source_map, by_raw, by_capture, by_origin_run = {}, {}, {}, {}
    for source in manifest["physical_sources"]:
        for field in ("raw_source_file_sha256", "physical_identity_qualification_sha256"):
            sha(source[field], field)
        for field in ("acquisition_origin_id", "source_segment_run_id", "canonical_capture_id", "clock_domain_id"):
            text(source[field], field)
        require(source["clock_kind"] in ("absolute_integer_ticks", "capture_relative_integer_ticks"), "unsupported clock kind")
        dt = rational([source["tick_duration_numerator"], source["tick_duration_denominator"]], "tick_duration", True)
        origin = rational(source["clock_origin_seconds"], "clock_origin_seconds")
        identity = (source["raw_source_file_sha256"], source["acquisition_origin_id"], source["source_segment_run_id"])
        require(identity not in source_map, "duplicate physical source")
        physical = (source["canonical_capture_id"], source["clock_domain_id"], source["clock_kind"], dt, origin)
        for mapping, key in ((by_raw, identity[0]), (by_origin_run, identity[1:]), (by_capture, source["canonical_capture_id"])):
            require(key not in mapping or mapping[key] == physical, "source alias creates inconsistent capture/clock/run")
            mapping[key] = physical
        source_map[identity] = (source, dt, origin)
    require(source_map, "missing physical sources")
    units, intervals, block_units = {}, {}, {}

    def interval(record: dict) -> tuple:
        identity = tuple(record[f] for f in ("raw_source_file_sha256", "acquisition_origin_id", "source_segment_run_id"))
        require(identity in source_map, "unqualified source identity/alias")
        source, dt, origin = source_map[identity]
        for field in ("clock_kind", "tick_duration_numerator", "tick_duration_denominator"):
            require(record[field] == source[field] and type(record[field]) is type(source[field]), "physical clock identity mismatch")
        start, end = integer(record["raw_start_tick"], "raw_start_tick"), integer(record["raw_end_tick"], "raw_end_tick")
        require(end >= start, "reversed closed physical interval")
        return (source["canonical_capture_id"], source["clock_domain_id"]), origin + start * dt, origin + end * dt

    for unit in manifest["physical_units"]:
        outer = text(unit["verified_outer_time_unit_id"], "verified_outer_time_unit_id")
        require(outer not in units, "duplicate outer time identity")
        require(unit["eligibility_status"] == "eligible_synthetic_fixture", "ineligible outer unit")
        block = text(unit["independent_block_id"], "independent_block_id")
        units[outer] = unit
        intervals[outer] = interval(unit)
        block_units.setdefault(block, []).append(outer)
    require(1 <= len(units) <= 32, "software fixture requires 1..32 physical outer units")
    gap = rational(manifest["minimum_gap_seconds"], "minimum_gap_seconds")
    require(gap >= 0, "negative physical gap")
    ordered_units = list(units)
    for i, left in enumerate(ordered_units):
        ld, ls, le = intervals[left]
        for right in ordered_units[i + 1:]:
            rd, rs, re = intervals[right]
            if ld == rd:
                require((le < rs and le + gap <= rs) or (re < ls and re + gap <= ls),
                        "closed native overlap/touching/alias or insufficient raw-time gap")
    for quarantine in manifest["quarantine_intervals"]:
        qd, qs, qe = interval(quarantine)
        for domain, start, end in intervals.values():
            require(domain != qd or (end < qs and end + gap <= qs) or (qe < start and qe + gap <= start),
                    "historical quarantine closed overlap/touching or insufficient gap")
    cases, case_cells = {}, set()
    for case in manifest["cases"]:
        case_id = text(case["case_id"], "case_id")
        require(case_id not in cases, "duplicate case")
        outer = case["verified_outer_time_unit_id"]
        require(outer in units, "unknown case outer identity")
        exact_keys(case["exogenous_pair_identity"], EXOGENOUS_FIELDS, "exogenous pair identity")
        exo = case["exogenous_pair_identity"]
        for field in EXOGENOUS_FIELDS:
            (sha if field.endswith("fingerprint") else text)(exo[field], field)
        require(exo["workload_regime_id"] in regimes, "unknown workload regime")
        require(exo["split_id"] == manifest["split_id"], "cross-split pair")
        require(exo["evaluation_policy_mode"] == manifest["evaluation_policy_mode"], "mixed policy evaluation mode")
        key = (outer, exo["workload_regime_id"])
        require(key not in case_cells, "duplicate outer/regime condition")
        case_cells.add(key)
        cases[case_id] = case
    require(case_cells == {(outer, regime) for outer in units for regime in regimes}, "incomplete outer/regime case matrix")
    methods = {}
    for method in manifest["methods"]:
        name = text(method["method_configuration_id"], "method_configuration_id")
        require(name not in methods, "duplicate method configuration")
        require(method["kind"] in ("learned", "deterministic_rule"), "unsupported method kind")
        if method["kind"] == "deterministic_rule":
            sha(method["rule_source_fingerprint"], "rule_source_fingerprint")
            require(method["deterministic_for_current_inputs"] is True, "nondeterministic rule cannot share physical reference")
        methods[name] = method
    candidate = plan["candidate_method"]
    baselines = plan["baseline_methods"]
    require(isinstance(baselines, list) and 1 <= len(baselines) <= 15, "invalid baseline family")
    unique([candidate] + baselines, "candidate/baseline")
    require(set(methods) == set([candidate] + baselines), "method matrix differs from predeclared comparison family")
    require(methods[candidate]["kind"] == "learned", "candidate must be a learned method")
    family = [{"baseline": base, "metric": metric} for base in baselines for metric in metric_ids]
    require(plan["comparison_family"] == family, "full ordered metric x baseline comparison family required")
    models = {}
    for model in manifest["models"]:
        method, seed = model["method_configuration_id"], model["training_seed"]
        require(method in methods and methods[method]["kind"] == "learned" and seed in seeds and type(seed) is int, "unknown learned model/seed")
        key = (method, seed)
        require(key not in models, "duplicate learned checkpoint identity")
        for field in ("policy_checkpoint_sha256", "training_provenance_fingerprint", "checkpoint_manifest_sha256"):
            sha(model[field], field)
        require(model["checkpoint_manifest_sha256"] == manifest["checkpoint_manifest_sha256"], "checkpoint manifest mismatch")
        require(integer(model["fixed_final_episode"], "fixed_final_episode") == plan["fixed_final_episode"], "fixed-final replacement")
        models[key] = model
    require(set(models) == {(m, s) for m in methods if methods[m]["kind"] == "learned" for s in seeds}, "incomplete learned model/seed matrix")
    return {"seeds": seeds, "regimes": regimes, "weights": weights, "metric_ids": metric_ids,
            "sources": source_map, "units": units, "intervals": intervals, "blocks": block_units,
            "cases": cases, "methods": methods, "models": models, "family": family}


def endpoint(value: dict, metric_id: str) -> dict:
    exact_keys(value, ("status", "value", "reason", "unit", "direction"), "endpoint")
    _, direction, unit, count = METRICS[metric_id]
    require(value["unit"] == unit and value["direction"] == direction, "endpoint native direction/unit mismatch")
    if value["status"] == "available":
        v = value["value"]
        require(type(v) in (int, float) and math.isfinite(v), "invalid/nonfinite native endpoint")
        require(value["reason"] is None, "available endpoint has unavailable reason")
        require(metric_id == "native_return" or v >= 0, "negative native count/cost")
        require(not count or type(v) is int, "integer native count/bytes required")
    else:
        require(value["status"] == "unavailable" and value["value"] is None, "unknown/missing endpoint status")
        text(value["reason"], "unavailable endpoint reason")
    return value


def canonicalize_native(plan: dict, manifest: dict, manifest_sha: str, records: list[tuple[str, dict]]) -> list[dict]:
    ctx = validate_contract(plan, manifest)
    result = []
    for source_sha, native in records:
        sha(source_sha, "source_output_file_sha256")
        require(native["schema"] == NATIVE_SCHEMA and native["evidence_kind"] == KIND, "native row schema/evidence mismatch")
        case_id, method = native["case_id"], native["method_configuration_id"]
        require(case_id in ctx["cases"] and method in ctx["methods"], "unknown native case/method")
        case = ctx["cases"][case_id]
        require(native["case_binding_sha256"] == digest(case), "native row exogenous case binding mismatch")
        model_id = native["model_identity"]
        rule_id = native["rule_reference_identity"]
        learned = ctx["methods"][method]["kind"] == "learned"
        if learned:
            require(rule_id is None and isinstance(model_id, dict), "learned/rule identity mismatch")
            seed = model_id["training_seed"]
            require((method, seed) in ctx["models"], "unknown native checkpoint seed")
            require(canonical_bytes(model_id) == canonical_bytes(ctx["models"][(method, seed)]), "native fixed-final checkpoint/provenance mismatch")
        else:
            require(model_id is None, "rule cannot have fabricated seed/checkpoint")
            exact_keys(rule_id, ("method_configuration_id", "physical_evaluation_record_id", "rule_source_fingerprint", "deterministic_for_current_inputs"), "rule reference")
            require(rule_id["method_configuration_id"] == method and
                    rule_id["rule_source_fingerprint"] == ctx["methods"][method]["rule_source_fingerprint"] and
                    rule_id["deterministic_for_current_inputs"] is True and
                    rule_id["physical_evaluation_record_id"] == native["physical_evaluation_record_id"], "invalid deterministic rule reference")
        exact_keys(native["native_endpoints"], [METRICS[m][0] for m in ctx["metric_ids"]], "native endpoint family")
        endpoints = {m: endpoint(native["native_endpoints"][METRICS[m][0]], m) for m in ctx["metric_ids"]}
        require(native["native_metric_ledger_sha256"] == manifest["native_metric_ledger_sha256"], "native metric ledger mismatch")
        unit = ctx["units"][case["verified_outer_time_unit_id"]]
        result.append({"schema": ROW_SCHEMA, "evidence_kind": KIND,
                       "case_id": case_id, "method_configuration_id": method,
                       "physical_evaluation_record_id": text(native["physical_evaluation_record_id"], "physical evaluation id"),
                       "physical_outer_identity": {f: unit[f] for f in PHYSICAL_FIELDS},
                       "exogenous_pair_identity": dict(case["exogenous_pair_identity"]),
                       "model_identity": model_id, "rule_reference_identity": rule_id,
                       "provenance": {"source_output_file_sha256": source_sha,
                                      "source_row_id": text(native["source_row_id"], "source_row_id"),
                                      "input_manifest_sha256": manifest_sha,
                                      "checkpoint_manifest_sha256": manifest["checkpoint_manifest_sha256"],
                                      "eligibility_audit_sha256": manifest["eligibility_audit_sha256"],
                                      "native_metric_ledger_sha256": native["native_metric_ledger_sha256"]},
                       "endpoints": endpoints})
    validate_rows(plan, manifest, manifest_sha, result, ctx)
    return result


def validate_rows(plan: dict, manifest: dict, manifest_sha: str, rows: list[dict], ctx: dict | None = None) -> tuple[dict, dict]:
    ctx = ctx or validate_contract(plan, manifest)
    sha(manifest_sha, "input_manifest_sha256")
    expected = {(case, method, seed if spec["kind"] == "learned" else None)
                for case in ctx["cases"] for method, spec in ctx["methods"].items()
                for seed in (ctx["seeds"] if spec["kind"] == "learned" else [None])}
    matrix, physical, source_rows = {}, set(), set()
    allowed_files = {item["sha256"] for item in plan["native_inputs"]}
    require(allowed_files, "missing native input binding")
    for row in rows:
        require(row["schema"] == ROW_SCHEMA and row["evidence_kind"] == KIND, "canonical row schema/evidence mismatch")
        case_id, method = row["case_id"], row["method_configuration_id"]
        require(case_id in ctx["cases"] and method in ctx["methods"], "unexpected row case/method")
        case = ctx["cases"][case_id]
        unit = ctx["units"][case["verified_outer_time_unit_id"]]
        exact_keys(row["physical_outer_identity"], PHYSICAL_FIELDS, "physical_outer_identity; no fallback")
        for field in ("raw_start_tick", "raw_end_tick", "tick_duration_numerator", "tick_duration_denominator"):
            integer(row["physical_outer_identity"][field], field)
        require(canonical_bytes(row["physical_outer_identity"]) == canonical_bytes({f: unit[f] for f in PHYSICAL_FIELDS}), "row physical outer identity mismatch")
        exact_keys(row["exogenous_pair_identity"], EXOGENOUS_FIELDS, "exogenous_pair_identity")
        require(row["exogenous_pair_identity"] == case["exogenous_pair_identity"], "unpaired exogenous input/cost/geometry")
        learned = ctx["methods"][method]["kind"] == "learned"
        if learned:
            identity = row["model_identity"]
            require(isinstance(identity, dict), "missing learned identity")
            seed = identity["training_seed"]
            require(type(seed) is int and (method, seed) in ctx["models"] and canonical_bytes(identity) == canonical_bytes(ctx["models"][(method, seed)]), "fixed-final checkpoint or learned provenance mismatch")
            require(row["rule_reference_identity"] is None, "learned row with rule reference")
        else:
            seed = None
            rule = row["rule_reference_identity"]
            exact_keys(rule, ("method_configuration_id", "physical_evaluation_record_id", "rule_source_fingerprint", "deterministic_for_current_inputs"), "rule reference")
            require(row["model_identity"] is None and rule["deterministic_for_current_inputs"] is True and
                    rule["method_configuration_id"] == method and
                    rule["rule_source_fingerprint"] == ctx["methods"][method]["rule_source_fingerprint"] and
                    rule["physical_evaluation_record_id"] == row["physical_evaluation_record_id"], "invalid shared physical rule record")
        key = (case_id, method, seed)
        require(key not in matrix, "duplicate method/seed/case matrix cell")
        matrix[key] = row
        record_id = text(row["physical_evaluation_record_id"], "physical evaluation id")
        require(record_id not in physical, "duplicate physical evaluation record")
        physical.add(record_id)
        provenance = row["provenance"]
        exact_keys(provenance, PROVENANCE_FIELDS, "row provenance")
        for field in PROVENANCE_FIELDS:
            (text if field == "source_row_id" else sha)(provenance[field], field)
        require(provenance["source_output_file_sha256"] in allowed_files and
                provenance["input_manifest_sha256"] == manifest_sha and
                provenance["checkpoint_manifest_sha256"] == manifest["checkpoint_manifest_sha256"] and
                provenance["eligibility_audit_sha256"] == manifest["eligibility_audit_sha256"] and
                provenance["native_metric_ledger_sha256"] == manifest["native_metric_ledger_sha256"], "row provenance/source SHA mismatch")
        source_key = (provenance["source_output_file_sha256"], provenance["source_row_id"])
        require(source_key not in source_rows, "duplicate original source row identity")
        source_rows.add(source_key)
        exact_keys(row["endpoints"], ctx["metric_ids"], "full endpoint family")
        for metric in ctx["metric_ids"]:
            endpoint(row["endpoints"][metric], metric)
    require(set(matrix) == expected, "missing/extra method/seed/case matrix cells; complete-case filtering forbidden")
    return ctx, matrix


def sign_test(values: np.ndarray) -> dict:
    wins, losses, ties = int(np.sum(values > 0)), int(np.sum(values < 0)), int(np.sum(values == 0))
    n = wins + losses
    p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)) if n else 1.0
    return {"wins": wins, "losses": losses, "ties": ties, "independent_block_count": len(values),
            "non_tied_block_count": n, "p_value": p, "alternative": "two_sided", "status": "all_ties" if not n else "available"}


def bootstrap(values: list[np.ndarray], seed_weights: np.ndarray, regime_weights: np.ndarray, rng_seed: int) -> np.ndarray:
    """One global paired trained-seed draw per replicate; regimes remain window-nested.

    This specified product bootstrap retains shared-checkpoint seed effects across
    windows. It is not a proof of universal covariance estimation or CI coverage.
    """
    constant = float(values[0].flat[0])
    if all(np.all(data == constant) for data in values):
        return np.full(BOOTSTRAP_DRAWS, constant)
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    nseed = values[0].shape[1]
    global_seeds = rng.choice(nseed, size=(BOOTSTRAP_DRAWS, nseed), p=seed_weights)
    chosen = rng.integers(n, size=(BOOTSTRAP_DRAWS, n))
    result = np.zeros(BOOTSTRAP_DRAWS)
    for position in range(n):
        for block, data in enumerate(values):
            rows = np.flatnonzero(chosen[:, position] == block)
            if not len(rows):
                continue
            nwindow, nseed, nregime = data.shape
            # Bound temporary arrays even for a full 32-window crossed fixture.
            for start in range(0, len(rows), 128):
                selected = rows[start:start + 128]
                count = len(selected)
                windows = rng.integers(nwindow, size=(count, nwindow))
                seeds = global_seeds[selected]
                regimes = rng.choice(nregime, size=(count, nwindow, nregime), p=regime_weights)
                sample = data[windows[:, :, None, None], seeds[:, None, :, None], regimes[:, :, None, :]]
                result[selected] += sample.mean(axis=(1, 2, 3)) / n
    return result


def interval_summary(draws: np.ndarray, blocks: np.ndarray) -> dict:
    low, high = [float(v) for v in np.quantile(draws, [0.025, 0.975], method="linear")]
    percentile = {"status": "degenerate" if np.all(draws == draws[0]) else "available", "lower": low, "upper": high}
    bca = {"status": "undefined", "reason": None, "lower": None, "upper": None}
    if len(blocks) < 2:
        percentile = {"status": "unavailable", "reason": "fewer_than_two_independent_blocks_no_time_population_ci",
                      "lower": None, "upper": None}
        bca["reason"] = "fewer_than_two_independent_blocks"
        bca["status"] = "unavailable"
    else:
        estimate = float(blocks.mean())
        jack = (blocks.sum() - blocks) / (len(blocks) - 1)
        influence = jack.mean() - jack
        denom = float(np.sum(influence ** 2))
        if np.all(blocks == blocks[0]) or denom == 0:
            bca["reason"] = "zero_jackknife_variance"
        else:
            prob = float((np.sum(draws < estimate) + 0.5 * np.sum(draws == estimate)) / len(draws))
            if prob <= 0 or prob >= 1:
                bca["reason"] = "extreme_bootstrap_bias_probability"
            else:
                normal = NormalDist()
                z0 = normal.inv_cdf(prob)
                acceleration = float(np.sum(influence ** 3) / (6 * denom ** 1.5))
                qs = []
                for q in (0.025, 0.975):
                    z = normal.inv_cdf(q)
                    divisor = 1 - acceleration * (z0 + z)
                    if divisor <= 0:
                        break
                    qs.append(normal.cdf(z0 + (z0 + z) / divisor))
                if len(qs) != 2 or not 0 <= qs[0] <= qs[1] <= 1:
                    bca["reason"] = "invalid_acceleration_projection"
                else:
                    bounds = np.quantile(draws, qs, method="linear")
                    bca = {"status": "available", "reason": None, "lower": float(bounds[0]), "upper": float(bounds[1]),
                           "bias_probability_midrank": prob, "acceleration": acceleration}
    if len(blocks) < 2:
        effect = {"status": "undefined", "reason": "fewer_than_two_independent_blocks", "value": None}
    else:
        sd = float(np.std(blocks, ddof=1))
        effect = ({"status": "undefined", "reason": "zero_between_block_variance", "value": None} if np.all(blocks == blocks[0]) or sd == 0 else
                  {"status": "available", "reason": None, "value": float(blocks.mean() / sd)})
    return {"percentile_95": percentile, "bca_95": bca, "paired_block_cohen_dz": effect}


def analyze_rows(plan: dict, manifest: dict, manifest_sha: str, rows: list[dict]) -> dict:
    ctx, matrix = validate_rows(plan, manifest, manifest_sha, rows)
    outer_order = sorted(ctx["units"])
    block_order = sorted(ctx["blocks"])
    cases = {(c["verified_outer_time_unit_id"], c["exogenous_pair_identity"]["workload_regime_id"]): name for name, c in ctx["cases"].items()}
    sw, rw = ctx["weights"]["seed_weights"], ctx["weights"]["regime_weights"]
    results = []
    logical_count = len(outer_order) * len(ctx["seeds"]) * len(ctx["regimes"])
    for family in ctx["family"]:
        baseline, metric = family["baseline"], family["metric"]
        sign = 1 if METRICS[metric][1] == "higher" else -1
        cube = np.zeros((len(outer_order), len(ctx["seeds"]), len(ctx["regimes"])))
        unavailable, baseline_ids = [], set()
        for oi, outer in enumerate(outer_order):
            for si, seed in enumerate(ctx["seeds"]):
                for ri, regime in enumerate(ctx["regimes"]):
                    case_id = cases[(outer, regime)]
                    candidate = matrix[(case_id, plan["candidate_method"], seed)]
                    base = matrix[(case_id, baseline, seed if ctx["methods"][baseline]["kind"] == "learned" else None)]
                    baseline_ids.add(base["physical_evaluation_record_id"])
                    left, right = candidate["endpoints"][metric], base["endpoints"][metric]
                    if left["status"] != "available" or right["status"] != "available":
                        unavailable.append({"case_id": case_id, "seed": seed, "candidate_status": left["status"], "baseline_status": right["status"]})
                    else:
                        cube[oi, si, ri] = left["value"] - right["value"]
        item = {"baseline": baseline, "metric": metric, "direction": METRICS[metric][1], "unit": METRICS[metric][2],
                "logical_paired_row_count": logical_count, "distinct_physical_outer_count": len(outer_order),
                "effective_independent_block_count": len(block_order), "baseline_physical_evaluation_count": len(baseline_ids),
                "logical_rule_reference_count": logical_count if ctx["methods"][baseline]["kind"] == "deterministic_rule" else 0,
                "raw_candidate_minus_baseline": None, "candidate_favoring_benefit": None,
                "percentile_95": None, "bca_95": None, "paired_block_cohen_dz": None, "sign_test": None,
                "holm_p_value": None, "unavailable_pairs": unavailable}
        if unavailable:
            item["status"] = "unavailable"
        else:
            blocks = [cube[[outer_order.index(outer) for outer in sorted(ctx["blocks"][block])]] for block in block_order]
            raw_means = np.asarray([float(np.mean(np.einsum("wsr,s,r->w", data, sw, rw))) for data in blocks])
            benefits = raw_means * sign
            draws = bootstrap([data * sign for data in blocks], sw, rw, plan["random_seed"])
            item.update({"status": "available", "raw_candidate_minus_baseline": float(raw_means.mean()),
                         "candidate_favoring_benefit": float(benefits.mean()),
                         "block_effects": [{"independent_block_id": block, "raw_delta": float(raw), "benefit": float(benefit)}
                                           for block, raw, benefit in zip(block_order, raw_means, benefits)],
                         "sign_test": sign_test(benefits), **interval_summary(draws, benefits)})
        results.append(item)
    # Unavailable hypotheses retain their position in the FULL family as p=1.
    pvals = [(item["sign_test"]["p_value"] if item["status"] == "available" else 1.0, i) for i, item in enumerate(results)]
    running = 0.0
    correction = []
    for rank, (p, idx) in enumerate(sorted(pvals), 1):
        adjusted = min(1.0, max(running, (len(results) - rank + 1) * p))
        running = adjusted
        if results[idx]["status"] == "available":
            results[idx]["holm_p_value"] = adjusted
        correction.append({"family_index": idx, "rank": rank, "p_for_correction": p,
                           "unavailable_placeholder": results[idx]["status"] != "available", "adjusted_p_for_correction": adjusted})
    physical = []
    for outer in outer_order:
        domain, start, end = ctx["intervals"][outer]
        physical.append({"verified_outer_time_unit_id": outer, "canonical_capture_id": domain[0], "clock_domain_id": domain[1],
                         "start_seconds_exact": fraction_json(start), "end_seconds_exact": fraction_json(end),
                         "independent_block_id": ctx["units"][outer]["independent_block_id"]})
    return {"schema": RESULT_SCHEMA, "evidence_kind": KIND, "status": "software_only_statistics_computed",
            "paper_goal_complete": False, "scientific_claim_ready": False,
            "physical_eligibility": "external trusted-manifest boundary; synthetic fixture certifies no real capture or holdout",
            "low_power": len(block_order) < 12, "minimum_internal_outer_guideline": 12,
            "estimand": plan["estimand"], "seed_weights": plan["seed_weights"], "regime_weights": plan["regime_weights"],
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "random_seed": plan["random_seed"], "confidence_level": 0.95,
                          "resampling": RESAMPLING_DESIGN, "quantile_method": "linear",
                          "seed_scope": "one paired training-seed index vector per replicate shared across all selected windows/blocks",
                          "regime_scope": "window-nested weighted resampling",
                          "coverage_status": "synthetic algorithm witness only; no population coverage calibration"},
            "interval_semantics": INTERVAL_SEMANTICS,
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "physical_units": physical, "physical_evaluation_count": len(rows), "family": results,
            "holm": {"declared_family_size": len(results), "includes_unavailable": True, "correction": correction},
            "limits": ["synthetic software qualification only", "no independent real physical-time or hidden eligibility certified",
                       "three consumed historical periods cannot become confirmatory generalization through this software",
                       "global-seed product bootstrap is not a universal covariance or coverage guarantee",
                       "optional unused bytes remain observational, not a causal avoided-transfer estimand"]}


class IOBudget:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.read_bytes = 0

    def check(self) -> None:
        require(time.monotonic() - self.started <= MAX_WALL_SECONDS, "software wall budget exceeded")

    def read(self, path: Path, expected: str) -> bytes:
        self.check()
        sha(expected, "file SHA")
        require(path.is_file() and not path.is_symlink(), "input absent or symlink")
        size = path.stat().st_size
        require(self.read_bytes + size <= MAX_INPUT_BYTES, "software input byte budget exceeded")
        data = path.read_bytes()
        self.read_bytes += len(data)
        require(self.read_bytes <= MAX_INPUT_BYTES and len(data) == size, "input changed while reading/budget")
        require(hashlib.sha256(data).hexdigest() == expected, "input SHA mismatch: " + str(path))
        self.check()
        return data

    def receipt(self) -> dict:
        return {"read_bytes": self.read_bytes, "wall_seconds": time.monotonic() - self.started,
                "training_frames": 0, "evaluation_frames": 0, "fit_calls": 0, "downloads": 0}


def load_json(data: bytes) -> Any:
    def object_pairs(pairs: list[tuple]) -> dict:
        names = [key for key, _ in pairs]
        unique(names, "JSON object key")
        return dict(pairs)

    def invalid_constant(value: str) -> Any:
        raise ContractError("nonfinite JSON constant: " + value)

    return json.loads(data, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def create_json(path: Path, value: Any) -> None:
    payload = canonical_bytes(value) + b"\n"
    require(len(payload) <= MAX_OUTPUT_BYTES, "software output byte budget exceeded")
    with path.open("xb") as handle:
        handle.write(payload)


def load_bindings(plan_path: Path, expected_sha: str, budget: IOBudget) -> tuple[dict, dict, str, list[tuple[str, dict]]]:
    plan = load_json(budget.read(plan_path, expected_sha))
    manifest_desc = plan["manifest"]
    manifest_sha = manifest_desc["sha256"]
    manifest = load_json(budget.read(Path(manifest_desc["path"]), manifest_sha))
    validate_contract(plan, manifest)
    descriptors = plan["native_inputs"]
    require(isinstance(descriptors, list) and 1 <= len(descriptors) <= 16, "invalid native file descriptors")
    unique([str(Path(d["path"]).resolve()) for d in descriptors], "native input path")
    unique([d["sha256"] for d in descriptors], "native input file SHA")
    records = []
    for descriptor in descriptors:
        data = budget.read(Path(descriptor["path"]), descriptor["sha256"])
        require(data.endswith(b"\n"), "native JSONL must terminate its last record")
        for line in data.splitlines():
            require(line.strip(), "blank native JSONL record")
            records.append((descriptor["sha256"], load_json(line)))
            require(len(records) <= 100000, "software row budget exceeded")
    return plan, manifest, manifest_sha, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("produce", "analyze"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected_plan_sha", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--expected_bundle_sha")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    claim = output.with_name(output.name + ".claim.json")
    failure = output.with_name(output.name + ".failure.json")
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(not output.exists() and not failure.exists(), "create-only output/failure already exists")
    create_json(claim, {"started_at": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, str(Path(__file__).resolve())] + (sys.argv[1:] if argv is None else argv),
                        "operation": args.operation, "expected_plan_sha256": args.expected_plan_sha, "code_sha256": code_sha})
    budget = IOBudget()
    try:
        plan, manifest, manifest_sha, records = load_bindings(args.plan, args.expected_plan_sha, budget)
        rows = canonicalize_native(plan, manifest, manifest_sha, records)
        if args.operation == "produce":
            require(args.bundle is None and args.expected_bundle_sha is None, "produce cannot consume a bundle")
            value = {"schema": BUNDLE_SCHEMA, "evidence_kind": KIND, "plan_sha256": args.expected_plan_sha,
                     "manifest_sha256": manifest_sha, "native_inputs": plan["native_inputs"], "producer_code_sha256": code_sha,
                     "rows": rows, "io_receipt": budget.receipt()}
        else:
            require(args.bundle is not None and args.expected_bundle_sha is not None, "analyze requires exact bundle binding")
            bundle = load_json(budget.read(args.bundle, args.expected_bundle_sha))
            require(bundle["schema"] == BUNDLE_SCHEMA and bundle["evidence_kind"] == KIND and
                    bundle["plan_sha256"] == args.expected_plan_sha and bundle["manifest_sha256"] == manifest_sha and
                    bundle["native_inputs"] == plan["native_inputs"] and bundle["producer_code_sha256"] == code_sha,
                    "bundle plan/native/source provenance mismatch")
            require(canonical_bytes(bundle["rows"]) == canonical_bytes(rows), "canonical producer/consumer source-row disagreement")
            value = analyze_rows(plan, manifest, manifest_sha, bundle["rows"])
            value.update({"plan_sha256": args.expected_plan_sha, "manifest_sha256": manifest_sha,
                          "bundle_sha256": args.expected_bundle_sha, "code_sha256": code_sha,
                          "native_inputs": plan["native_inputs"], "command_claim_path": str(claim), "io_receipt": budget.receipt()})
        budget.check()
        create_json(output, value)
        print(json.dumps({"status": value.get("status", "canonical_bundle_created"), "output": str(output), "io_receipt": budget.receipt()}))
        return 0
    except Exception as exc:
        create_json(failure, {"status": "failed_preserved_no_retry", "error_type": type(exc).__name__, "error": str(exc),
                              "expected_plan_sha256": args.expected_plan_sha, "code_sha256": code_sha, "io_receipt": budget.receipt()})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
