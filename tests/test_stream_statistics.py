"""At most 32 named synthetic fixtures; no scientific files or env calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_stream_statistics.py"
SPEC = importlib.util.spec_from_file_location("strict_stream_statistics_qa", SCRIPT)
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def write(path: Path, obj) -> str:
    payload = stats.canonical_bytes(obj) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


class SyntheticFixture:
    def __init__(self, root: Path, name: str, split_files: bool = False):
        self.path = root / name
        self.path.mkdir()
        self.name, self.split_files = name, split_files
        self.manifest = {
            "schema": stats.MANIFEST_SCHEMA, "evidence_kind": stats.KIND,
            "interval_semantics": stats.INTERVAL_SEMANTICS,
            "physical_eligibility_status": "synthetic_qualification_only",
            "eligibility_audit_sha256": h("synthetic eligibility"),
            "checkpoint_manifest_sha256": h("synthetic checkpoint manifest"),
            "native_metric_ledger_sha256": h("synthetic native ledger"),
            "reward_contract_sha256": h("synthetic reward contract"),
            "split_id": "synthetic_dev", "evaluation_policy_mode": "fixed_final_deterministic",
            "minimum_gap_seconds": [0, 1], "quarantine_intervals": [],
            "physical_sources": [{
                "raw_source_file_sha256": h("synthetic raw physical source"),
                "acquisition_origin_id": "synthetic_capture_A", "source_segment_run_id": "native_run_1",
                "canonical_capture_id": "capture_A", "clock_domain_id": "relative_run_1",
                "clock_kind": "capture_relative_integer_ticks", "tick_duration_numerator": 1,
                "tick_duration_denominator": 10, "clock_origin_seconds": [0, 1],
                "physical_identity_qualification_sha256": h("synthetic physical qualification")
            }], "physical_units": [], "cases": [], "methods": [
                {"method_configuration_id": "main", "kind": "learned"},
                {"method_configuration_id": "control", "kind": "learned"},
                {"method_configuration_id": "rule", "kind": "deterministic_rule",
                 "rule_source_fingerprint": h("synthetic deterministic rule"), "deterministic_for_current_inputs": True}
            ], "models": []}
        seeds, regimes = [7, 17, 27], ["mixed", "full"]
        for method in ("main", "control"):
            for seed in seeds:
                self.manifest["models"].append({"method_configuration_id": method, "training_seed": seed,
                    "policy_checkpoint_sha256": h("synthetic cp %s %s" % (method, seed)),
                    "training_provenance_fingerprint": h("synthetic training %s %s" % (method, seed)),
                    "checkpoint_manifest_sha256": self.manifest["checkpoint_manifest_sha256"], "fixed_final_episode": 96})
        source = self.manifest["physical_sources"][0]
        for outer in range(4):
            unit = {field: source[field] for field in stats.PHYSICAL_FIELDS[:6]}
            unit.update({"raw_start_tick": 1 + outer * 10, "raw_end_tick": 4 + outer * 10,
                         "verified_outer_time_unit_id": "outer_%d" % outer,
                         "independent_block_id": "block_%d" % outer, "eligibility_status": "eligible_synthetic_fixture"})
            self.manifest["physical_units"].append(unit)
            for regime in regimes:
                case_id = "case_%d_%s" % (outer, regime)
                exo = {"split_id": "synthetic_dev", "workload_regime_id": regime,
                       "evaluation_policy_mode": "fixed_final_deterministic"}
                for field in stats.EXOGENOUS_FIELDS:
                    if field.endswith("fingerprint"):
                        exo[field] = h("synthetic " + case_id + " " + field)
                self.manifest["cases"].append({"case_id": case_id, "verified_outer_time_unit_id": unit["verified_outer_time_unit_id"],
                                               "exogenous_pair_identity": exo})
        metric_ids = ["native_return", "completed_requests", "unused_optional_transfer_bytes"]
        self.plan = {"schema": stats.PLAN_SCHEMA, "evidence_kind": stats.KIND, "analysis_mode": "software_only",
                     "paper_goal_complete": False, "bootstrap_draws": 10000, "random_seed": 47,
                     "confidence_level": 0.95, "unavailable_policy": "entire_hypothesis_unavailable_keep_full_family",
                     "sign_tie_tolerance": 0, "training_seeds": seeds, "workload_regimes": regimes,
                     "seed_weights": {str(s): [1, 3] for s in seeds}, "regime_weights": {r: [1, 2] for r in regimes},
                     "estimand": stats.ESTIMAND, "resampling_design": stats.RESAMPLING_DESIGN,
                     "candidate_method": "main", "baseline_methods": ["control", "rule"], "fixed_final_episode": 96,
                     "metrics": [{"id": m, "native_field": stats.METRICS[m][0], "direction": stats.METRICS[m][1], "unit": stats.METRICS[m][2]} for m in metric_ids],
                     "comparison_family": [{"baseline": base, "metric": m} for base in ("control", "rule") for m in metric_ids]}
        models = {(m["method_configuration_id"], m["training_seed"]): m for m in self.manifest["models"]}
        self.records = []
        for ci, case in enumerate(self.manifest["cases"]):
            for method in ("main", "control", "rule"):
                for si, seed in enumerate(seeds if method != "rule" else [None]):
                    ident = "%s/%s/%s" % (case["case_id"], method, seed)
                    base = 100 + ci + (si if seed is not None else 0)
                    values = {"native_return": base + (ci // 2 + 1 + si * 0.2 + (ci % 2) * 0.1 if method == "main" else 0),
                              "completed_requests": 9 if method == "main" else 8,
                              "unused_optional_transfer_bytes": 10 if method == "main" else 100}
                    endpoints = {stats.METRICS[m][0]: {"status": "available", "value": values[m], "reason": None,
                                  "unit": stats.METRICS[m][2], "direction": stats.METRICS[m][1]} for m in metric_ids}
                    self.records.append({"schema": stats.NATIVE_SCHEMA, "evidence_kind": stats.KIND,
                        "case_id": case["case_id"], "case_binding_sha256": stats.digest(case),
                        "method_configuration_id": method, "physical_evaluation_record_id": ident, "source_row_id": ident,
                        "model_identity": copy.deepcopy(models[(method, seed)]) if seed is not None else None,
                        "rule_reference_identity": None if seed is not None else {"method_configuration_id": "rule",
                            "physical_evaluation_record_id": ident, "rule_source_fingerprint": h("synthetic deterministic rule"), "deterministic_for_current_inputs": True},
                        "native_metric_ledger_sha256": self.manifest["native_metric_ledger_sha256"], "native_endpoints": endpoints})
        self.frozen = False

    def freeze(self, rebind_cases: bool = False):
        assert not self.frozen
        if rebind_cases:
            cases = {c["case_id"]: c for c in self.manifest["cases"]}
            for row in self.records:
                row["case_binding_sha256"] = stats.digest(cases[row["case_id"]])
        self.manifest_sha = write(self.path / "manifest.json", self.manifest)
        self.plan["manifest"] = {"path": str(self.path / "manifest.json"), "sha256": self.manifest_sha}
        self.plan["native_inputs"] = []
        self.native_records = []
        groups = [[row for row in self.records if row["method_configuration_id"] == name] for name in ("main", "control", "rule")] if self.split_files else [self.records]
        for index, records in enumerate(groups):
            path = self.path / ("native_%d.jsonl" % index)
            data = b"".join(stats.canonical_bytes(row) + b"\n" for row in records)
            with path.open("xb") as handle:
                handle.write(data)
            source_sha = hashlib.sha256(data).hexdigest()
            self.plan["native_inputs"].append({"path": str(path), "sha256": source_sha})
            self.native_records.extend((source_sha, row) for row in records)
        self.plan_sha = write(self.path / "plan.json", self.plan)
        self.frozen = True
        return self

    def rows(self):
        if not self.frozen:
            self.freeze()
        return stats.canonicalize_native(self.plan, self.manifest, self.manifest_sha, self.native_records)

    def result(self):
        result = stats.analyze_rows(self.plan, self.manifest, self.manifest_sha if self.frozen else self.freeze().manifest_sha, self.rows())
        write(self.path / "result.json", result)
        return result

    def rejected(self, callback, match: str):
        try:
            callback()
        except (stats.ContractError, KeyError) as error:
            assert match in str(error)
            write(self.path / "expected_rejection.json", {"status": "rejection_witness_passed", "type": type(error).__name__, "error": str(error)})
        else:
            pytest.fail("bad fixture was accepted: " + self.name)

    def cli(self, operation: str, filename: str, *extra: str):
        if not self.frozen:
            self.freeze()
        return stats.main([operation, "--plan", str(self.path / "plan.json"), "--expected_plan_sha", self.plan_sha,
                           "--output", str(self.path / filename), *extra])


@pytest.fixture(scope="session")
def factory(tmp_path_factory):
    custom = os.environ.get("STREAM_STATISTICS_QA_ROOT")
    root = Path(custom) if custom else tmp_path_factory.mktemp("stream_statistics_software")
    if custom:
        root.mkdir(parents=True, exist_ok=False)
    started, names = time.monotonic(), []

    def create(name: str, split_files: bool = False):
        assert len(names) < 32 and name not in names
        assert time.monotonic() - started <= 7200
        names.append(name)
        return SyntheticFixture(root, name, split_files)

    yield create
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert size <= 128 * 1024 * 1024
    write(root / "fixture_registry.json", {"schema": "strict_statistics_fixture_registry_v1", "fixture_count": len(names),
        "names": names, "software_only": True, "artifact_bytes_before_registry": size, "wall_seconds": time.monotonic() - started,
        "training": 0, "evaluation": 0, "downloads": 0, "real_data_statistics": 0})


def item(result, baseline="control", metric="native_return"):
    return next(r for r in result["family"] if r["baseline"] == baseline and r["metric"] == metric)


def test_s0_complete_pairing_stays_four_outer(factory):
    fixture = factory("S0_complete_positive")
    result = fixture.result()
    r = item(result)
    assert r["logical_paired_row_count"] == 24
    assert r["distinct_physical_outer_count"] == r["effective_independent_block_count"] == 4
    assert r["sign_test"]["independent_block_count"] == 4
    assert result["low_power"] and not result["scientific_claim_ready"]
    assert r["candidate_favoring_benefit"] == pytest.approx(2.75)
    assert result["bootstrap"]["draws"] == 10000
    assert r["bca_95"]["status"] == "available"


def test_s1_split_files_and_one_file_same_join(factory):
    joined = factory("S1_joined_file").result()
    split = factory("S1_separate_method_files", split_files=True).result()
    assert joined["family"] == split["family"]
    assert joined["holm"] == split["holm"]


@pytest.mark.parametrize("field", ["verified_outer_time_unit_id", "raw_source_file_sha256", "raw_start_tick"])
def test_s2_missing_physical_key_rejected(factory, field):
    fixture = factory("S2_missing_" + field)
    rows = fixture.rows()
    del rows[0]["physical_outer_identity"][field]
    fixture.rejected(lambda: stats.analyze_rows(fixture.plan, fixture.manifest, fixture.manifest_sha, rows), "no fallback")


def test_s3_rule_one_physical_reference_not_seed_samples(factory):
    fixture = factory("S3_single_rule_reference")
    result = fixture.result()
    r = item(result, "rule")
    assert r["logical_paired_row_count"] == r["logical_rule_reference_count"] == 24
    assert r["baseline_physical_evaluation_count"] == 8
    assert r["sign_test"]["independent_block_count"] == 4
    assert result["physical_evaluation_count"] == 56


@pytest.mark.parametrize("field,value", [("direction", "higher"), ("unit", "megabytes")])
def test_s4_metric_contract_rejects_wrong_direction_or_unit(factory, field, value):
    fixture = factory("S4_wrong_" + field)
    fixture.plan["metrics"][2][field] = value
    fixture.freeze()
    fixture.rejected(fixture.rows, "direction/unit")


def test_s4_s5_bytes_sign_and_degenerate_status(factory):
    fixture = factory("S5_constant_ninety_byte_benefit")
    result = fixture.result()
    r = item(result, metric="unused_optional_transfer_bytes")
    assert r["raw_candidate_minus_baseline"] == -90
    assert r["candidate_favoring_benefit"] == 90
    assert r["percentile_95"] == {"status": "degenerate", "lower": 90, "upper": 90}
    assert r["bca_95"]["status"] == "undefined"
    assert r["bca_95"]["reason"] == "zero_jackknife_variance"
    assert r["bca_95"]["lower"] is None
    assert r["paired_block_cohen_dz"]["value"] is None
    assert r["paired_block_cohen_dz"]["reason"] == "zero_between_block_variance"
    ties = stats.sign_test(np.zeros(4))
    assert ties["status"] == "all_ties" and ties["ties"] == 4
    assert ties["non_tied_block_count"] == 0 and ties["p_value"] == 1
    zero = stats.interval_summary(np.zeros(10000), np.zeros(4))
    assert zero["paired_block_cohen_dz"]["value"] is None
    write(fixture.path / "zero_effect_and_ties.json", {"sign_test": ties, "intervals": zero})


def test_s6_same_sha_cannot_be_different_segment_clock(factory):
    fixture = factory("S6_same_source_relabelled")
    alias = copy.deepcopy(fixture.manifest["physical_sources"][0])
    alias.update({"source_segment_run_id": "invented_run", "canonical_capture_id": "invented_capture", "clock_domain_id": "invented_clock"})
    fixture.manifest["physical_sources"].append(alias)
    fixture.freeze()
    fixture.rejected(fixture.rows, "source alias")


def test_s6_different_file_alias_still_overlaps(factory):
    fixture = factory("S6_different_file_same_capture_overlap")
    alias = copy.deepcopy(fixture.manifest["physical_sources"][0])
    alias["raw_source_file_sha256"] = h("synthetic copy with different bytes")
    fixture.manifest["physical_sources"].append(alias)
    first, second = fixture.manifest["physical_units"][:2]
    second.update({"raw_source_file_sha256": alias["raw_source_file_sha256"], "raw_start_tick": first["raw_start_tick"], "raw_end_tick": first["raw_end_tick"]})
    fixture.freeze()
    fixture.rejected(fixture.rows, "overlap/alias")


def test_s7_exact_fraction_ticks_not_zero_second_truncation(factory):
    fixture = factory("S7_exact_fractional_clock")
    result = fixture.result()
    first = result["physical_units"][0]
    assert first["start_seconds_exact"] == [1, 10]
    assert first["end_seconds_exact"] == [2, 5]
    assert result["family"][0]["distinct_physical_outer_count"] == 4
    rows = fixture.rows()
    rows[0]["physical_outer_identity"]["raw_start_tick"] = 1.0
    fixture.rejected(lambda: stats.analyze_rows(fixture.plan, fixture.manifest, fixture.manifest_sha, rows), "no float truncation")


def test_missing_matrix_cell_never_complete_case_filters(factory):
    fixture = factory("reject_missing_matrix")
    fixture.records.pop()
    fixture.freeze()
    fixture.rejected(fixture.rows, "complete-case filtering forbidden")


def test_duplicate_matrix_cell_rejected(factory):
    fixture = factory("reject_duplicate_matrix")
    fixture.records.append(copy.deepcopy(fixture.records[0]))
    fixture.freeze()
    fixture.rejected(fixture.rows, "duplicate method/seed/case")


def test_pair_cost_geometry_mismatch_rejected(factory):
    fixture = factory("reject_exogenous_geometry")
    rows = fixture.rows()
    rows[0]["exogenous_pair_identity"]["rsu_geometry_fingerprint"] = h("different RSU")
    fixture.rejected(lambda: stats.analyze_rows(fixture.plan, fixture.manifest, fixture.manifest_sha, rows), "unpaired exogenous")


def test_native_checkpoint_replacement_rejected(factory):
    fixture = factory("reject_checkpoint_replacement")
    fixture.records[0]["model_identity"]["policy_checkpoint_sha256"] = h("post-selection replacement")
    fixture.freeze()
    fixture.rejected(fixture.rows, "checkpoint/provenance")


def test_unavailable_endpoint_preserves_full_holm_family(factory):
    fixture = factory("unavailable_full_family")
    fixture.records[0]["native_endpoints"]["reward"].update({"status": "unavailable", "value": None, "reason": "synthetic missing endpoint"})
    result = fixture.result()
    assert len(result["family"]) == result["holm"]["declared_family_size"] == 6
    assert item(result)["status"] == "unavailable"
    assert item(result)["raw_candidate_minus_baseline"] is None
    assert item(result)["sign_test"] is None and item(result)["holm_p_value"] is None
    unavailable = [r for r in result["holm"]["correction"] if r["unavailable_placeholder"]]
    assert len(unavailable) == 2 and all(r["p_for_correction"] == 1 for r in unavailable)
    assert item(result, metric="completed_requests")["holm_p_value"] == 0.75


def test_missing_metric_field_rejected(factory):
    fixture = factory("reject_missing_endpoint")
    del fixture.records[0]["native_endpoints"]["reward"]
    fixture.freeze()
    fixture.rejected(fixture.rows, "native endpoint family")


def test_nonfinite_metric_rejected(factory):
    fixture = factory("reject_nonfinite_endpoint")
    rows = fixture.rows()
    rows[0]["endpoints"]["native_return"]["value"] = float("nan")
    fixture.rejected(lambda: stats.analyze_rows(fixture.plan, fixture.manifest, fixture.manifest_sha, rows), "nonfinite")


def test_unfrozen_weights_rejected(factory):
    fixture = factory("reject_weight_matrix")
    fixture.plan["seed_weights"].pop("27")
    fixture.freeze()
    fixture.rejected(fixture.rows, "seed_weights")


def test_shrunk_comparison_family_rejected(factory):
    fixture = factory("reject_shrunk_family")
    fixture.plan["comparison_family"].pop()
    fixture.freeze()
    fixture.rejected(fixture.rows, "full ordered")


def test_cli_full_production_consumer_and_code_binding(factory):
    fixture = factory("cli_producer_consumer_source_binding")
    assert fixture.cli("produce", "bundle.json") == 0
    bundle_path = fixture.path / "bundle.json"
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert fixture.cli("analyze", "analysis.json", "--bundle", str(bundle_path), "--expected_bundle_sha", bundle_sha) == 0
    result = json.loads((fixture.path / "analysis.json").read_text())
    assert result["bundle_sha256"] == bundle_sha
    assert result["code_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert result["io_receipt"]["training_frames"] == result["io_receipt"]["evaluation_frames"] == 0
    bad = json.loads(bundle_path.read_text())
    bad["producer_code_sha256"] = h("unqualified source")
    bad_sha = write(fixture.path / "bad_bundle.json", bad)
    assert fixture.cli("analyze", "rejected_source.json", "--bundle", str(fixture.path / "bad_bundle.json"), "--expected_bundle_sha", bad_sha) == 1
    assert not (fixture.path / "rejected_source.json").exists()
    assert "provenance mismatch" in (fixture.path / "rejected_source.json.failure.json").read_text()


def test_cli_row_tamper_rejected_independently(factory):
    fixture = factory("cli_canonical_row_tamper")
    assert fixture.cli("produce", "bundle.json") == 0
    bad = json.loads((fixture.path / "bundle.json").read_text())
    bad["rows"][0]["endpoints"]["native_return"]["value"] += 900
    bad_sha = write(fixture.path / "bad_bundle.json", bad)
    assert fixture.cli("analyze", "rejected_row.json", "--bundle", str(fixture.path / "bad_bundle.json"), "--expected_bundle_sha", bad_sha) == 1
    assert "source-row disagreement" in (fixture.path / "rejected_row.json.failure.json").read_text()


def test_cli_native_sha_mismatch_preserves_failure(factory):
    fixture = factory("cli_native_sha_mismatch").freeze()
    with (fixture.path / "native_0.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    assert fixture.cli("produce", "rejected_input.json") == 1
    assert "input SHA mismatch" in (fixture.path / "rejected_input.json.failure.json").read_text()


def test_create_only_single_claim_and_budgets(factory, monkeypatch):
    fixture = factory("cli_create_only_and_read_budget").freeze()
    monkeypatch.setattr(stats, "MAX_INPUT_BYTES", 1)
    assert fixture.cli("produce", "budget_failure.json") == 1
    failure = (fixture.path / "budget_failure.json.failure.json").read_bytes()
    with pytest.raises(stats.ContractError, match="create-only"):
        fixture.cli("produce", "budget_failure.json")
    assert (fixture.path / "budget_failure.json.failure.json").read_bytes() == failure
    monkeypatch.setattr(stats, "MAX_INPUT_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(stats, "MAX_OUTPUT_BYTES", 2)
    with pytest.raises(stats.ContractError, match="output byte budget"):
        stats.create_json(fixture.path / "too_large.json", {"payload": "no"})
    budget = stats.IOBudget()
    budget.started -= 7201
    with pytest.raises(stats.ContractError, match="wall budget"):
        budget.check()


def test_quarantine_exact_time_rejected(factory):
    fixture = factory("reject_historical_quarantine")
    fixture.manifest["quarantine_intervals"] = [copy.deepcopy(fixture.manifest["physical_units"][0])]
    fixture.freeze()
    fixture.rejected(fixture.rows, "quarantine overlap")


def test_no_float_tick_or_fractional_clock_truncation(factory):
    fixture = factory("reject_float_tick")
    fixture.manifest["physical_units"][0]["raw_start_tick"] = 0.1
    fixture.freeze()
    fixture.rejected(fixture.rows, "no float truncation")


def test_shared_block_does_not_create_independent_samples(factory):
    fixture = factory("grouped_windows_shared_independent_block")
    for unit in fixture.manifest["physical_units"]:
        unit["independent_block_id"] = "single_capture_dependency_block"
    result = fixture.result()
    r = item(result)
    assert r["distinct_physical_outer_count"] == 4
    assert r["effective_independent_block_count"] == r["sign_test"]["independent_block_count"] == 1
    assert r["bca_95"]["reason"] == "fewer_than_two_independent_blocks"
    assert r["paired_block_cohen_dz"]["value"] is None


def test_fixed_final_episode_mismatch_rejected(factory):
    fixture = factory("reject_wrong_fixed_final")
    fixture.manifest["models"][0]["fixed_final_episode"] = 88
    fixture.freeze()
    fixture.rejected(fixture.rows, "fixed-final replacement")


def test_rule_seed_replication_is_not_allowed(factory):
    fixture = factory("reject_fabricated_rule_seed")
    rule = next(r for r in fixture.records if r["method_configuration_id"] == "rule")
    rule["model_identity"] = {"training_seed": 7}
    fixture.freeze()
    fixture.rejected(fixture.rows, "fabricated seed")


def test_duplicate_json_key_and_scientific_payload_rejected(factory):
    fixture = factory("reject_ambiguous_json_and_science")
    fixture.plan["evidence_kind"] = "audited_native_stream"
    fixture.freeze()
    fixture.rejected(fixture.rows, "software-only")
    with pytest.raises(stats.ContractError, match="duplicate JSON object key"):
        stats.load_json(b'{"seed":7,"seed":17}')
    with pytest.raises(stats.ContractError, match="nonfinite JSON"):
        stats.load_json(b'{"reward":NaN}')
