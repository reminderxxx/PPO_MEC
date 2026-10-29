"""V2: eight once-materialized fixture families, immutable inputs across rounds.

Only this file is run for the separate v2 contract. The legacy31-test suite is
not invoked. prepare_fixtures() is an explicit one-time preparation entry point;
the pytest fixture never calls it or creates a directory.
"""
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

BUILDER_PATH = Path(__file__).with_name("test_stream_statistics.py")
SPEC = importlib.util.spec_from_file_location("stream_v2_source_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
stats = builder.stats
FAMILIES = (
    "V20_complete_matrix_and_retained_guards", "V21_closed_outer_boundary",
    "V22_closed_quarantine_boundary", "V23_integer_state_bytes",
    "V24_integer_link_frames", "V25_global_shared_checkpoint_seed",
    "V26_weighted_unequal_windows", "V27_explicit_v2_semantics",
)
VIEWS = {
    FAMILIES[0]: ["full_producer_consumer", "S0_24_pairs_4_blocks", "single_rule_8_physical",
                  "missing_outer", "duplicate_matrix", "missing_matrix", "wrong_geometry", "wrong_checkpoint",
                  "wrong_direction_unit", "full_family_unavailable", "degenerate_bytes_and_ties",
                  "alias_relabel", "fraction_tick_and_source_tamper", "create_only"],
    FAMILIES[1]: ["touching_closed_outer_reject", "exact_positive_gap_equality_accept"],
    FAMILIES[2]: ["touching_closed_quarantine_reject"],
    FAMILIES[3]: ["fractional_state_byte_reject"],
    FAMILIES[4]: ["fractional_link_frame_reject"],
    FAMILIES[5]: ["global_seed_draw_sequence_and_support", "analyzed_seed_scope"],
    FAMILIES[6]: ["hand_weighted_unequal_window_estimand", "one_block_no_time_population_CI"],
    FAMILIES[7]: ["missing_interval_semantics", "unsupported_half_open_semantics", "v1_plan", "v1_manifest",
                  "v1_native_row", "v1_canonical_row", "v1_bundle", "scientific_payload_reject"],
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_fixtures(root: Path):
    """Exactly8directories once. Re-entry refuses before creating anything."""
    assert not root.exists(), "fixture root exists: reuse read-only; never rematerialize"
    root.mkdir()
    for index, name in enumerate(FAMILIES):
        fixture = builder.SyntheticFixture(root, name, split_files=True)
        metric_ids = list(stats.METRICS)
        fixture.plan["metrics"] = [{"id": m, "native_field": stats.METRICS[m][0],
                                    "direction": stats.METRICS[m][1], "unit": stats.METRICS[m][2]} for m in metric_ids]
        fixture.plan["comparison_family"] = [{"baseline": b, "metric": m} for b in ("control", "rule") for m in metric_ids]
        for row in fixture.records:
            for metric in ("state_transfer_bytes", "state_transfer_link_frames", "handoff_failed_count"):
                native, direction, unit, _ = stats.METRICS[metric]
                row["native_endpoints"][native] = {"status": "available", "value": 10 if row["method_configuration_id"] == "main" else 20,
                                                   "reason": None, "unit": unit, "direction": direction}
        if index == 1:
            fixture.manifest["physical_units"][1]["raw_start_tick"] = fixture.manifest["physical_units"][0]["raw_end_tick"]
        elif index == 2:
            quarantine = copy.deepcopy(fixture.manifest["physical_units"][0])
            quarantine["raw_start_tick"] = quarantine["raw_end_tick"]
            quarantine["raw_end_tick"] += 1
            fixture.manifest["quarantine_intervals"] = [quarantine]
        elif index == 3:
            fixture.records[0]["native_endpoints"]["state_transfer_bytes"]["value"] = 1.5
        elif index == 4:
            fixture.records[0]["native_endpoints"]["state_transfer_link_frames"]["value"] = 2.25
        elif index in (5, 6):
            for row in fixture.records:
                if row["method_configuration_id"] != "main":
                    continue
                seed_index = fixture.plan["training_seeds"].index(row["model_identity"]["training_seed"])
                outer = int(row["case_id"].split("_")[1])
                regime = 1 if row["case_id"].endswith("full") else 0
                case_index = outer * 2 + regime
                control = 100 + case_index + seed_index
                delta = seed_index * 6 if index == 5 else outer * 100 + seed_index * 10 + regime * 4
                row["native_endpoints"]["reward"]["value"] = control + delta
            if index == 6:
                fixture.plan["seed_weights"] = {"7": [1, 2], "17": [1, 3], "27": [1, 6]}
                fixture.plan["regime_weights"] = {"mixed": [3, 4], "full": [1, 4]}
                for outer, unit in enumerate(fixture.manifest["physical_units"]):
                    unit["independent_block_id"] = "block_A_three_windows" if outer < 3 else "block_B_one_window"
        fixture.freeze()
        if index in (1, 6):
            view = copy.deepcopy(fixture.manifest)
            if index == 1:
                view["physical_units"][1]["raw_start_tick"] += 1
                view["minimum_gap_seconds"] = [1, 10]
                stem = "positive_exact_gap"
            else:
                for unit in view["physical_units"]:
                    unit["independent_block_id"] = "one_physical_dependency_block"
                stem = "one_block"
            manifest_path = fixture.path / (stem + "_manifest.json")
            view_sha = builder.write(manifest_path, view)
            plan_view = copy.deepcopy(fixture.plan)
            plan_view["manifest"] = {"path": str(manifest_path), "sha256": view_sha}
            builder.write(fixture.path / (stem + "_plan.json"), plan_view)
        builder.write(fixture.path / "prospective_views.json", {
            "family": name, "views": VIEWS[name], "in_memory_views": "Explicit adversarial/kernel views of this family; no real producer authentication claim for mutated in-memory rows.",
            "source_file_mutation": False, "new_fixture_directory_on_test_or_rerun": False})
    sources = [{"path": str(path.relative_to(root)), "sha256": file_sha(path), "bytes": path.stat().st_size}
               for path in sorted(root.rglob("*")) if path.is_file()]
    registry = {"contract": "stream_statistics_v2_eight_immutable_families_v1", "fixture_count": 8,
                "families": list(FAMILIES), "source_files": sources, "source_bytes": sum(s["bytes"] for s in sources)}
    builder.write(root.parent / "fixture_source_registry.json", registry)
    return registry


class Family:
    def __init__(self, root, name, round_id, calls):
        self.path, self.round_id, self.calls = root / name, round_id, calls

    def load(self, stem="plan"):
        path = self.path / (stem + ".json")
        plan, manifest, manifest_sha, records = stats.load_bindings(path, file_sha(path), stats.IOBudget())
        return plan, manifest, manifest_sha, records

    def invoke(self, label, operation, callback, reject=None):
        record = {"family": self.path.name, "view": label, "operation": operation,
                  "source_fixture_mutation": False, "round": self.round_id}
        self.calls.append(record)
        try:
            value = callback()
            assert reject is None, "invalid view accepted: " + label
            record["status"] = "pass"
            record["value"] = value
        except (stats.ContractError, KeyError) as error:
            assert reject is not None and reject in str(error), str(error)
            record.update({"status": "expected_rejection_pass", "error_type": type(error).__name__, "error": str(error)})
            value = None
        except Exception as error:
            record.update({"status": "unexpected_failure", "error_type": type(error).__name__, "error": str(error)})
            builder.write(self.path / (self.round_id + "_" + label + ".json"), record)
            raise
        builder.write(self.path / (self.round_id + "_" + label + ".json"), record)
        return value

    def cli(self, label, operation, expected=0, extra=None):
        output = self.path / (self.round_id + "_" + label + ".json")
        args = [operation, "--plan", str(self.path / "plan.json"), "--expected_plan_sha", file_sha(self.path / "plan.json"), "--output", str(output)] + (extra or [])
        self.calls.append({"family": self.path.name, "view": label, "operation": "cli_" + operation, "round": self.round_id})
        assert stats.main(args) == expected
        return output


@pytest.fixture(scope="session")
def families():
    location = os.environ.get("STREAM_STATISTICS_V2_FIXTURES")
    if not location:
        pytest.skip("v2requires explicitly pre-materialized eight immutable fixture directories")
    root = Path(location)
    round_id = os.environ["STREAM_STATISTICS_V2_ROUND"]
    assert round_id in ("round_01", "round_02")
    registry_path = root.parent / "fixture_source_registry.json"
    registry_sha = file_sha(registry_path)
    registry = json.loads(registry_path.read_text())
    assert registry["fixture_count"] == 8
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == sorted(FAMILIES)
    for item in registry["source_files"]:
        assert file_sha(root / item["path"]) == item["sha256"]
    calls, started = [], time.monotonic()
    yield lambda index: Family(root, FAMILIES[index], round_id, calls)
    # No mkdir or fixture builder is called by this test session.
    assert file_sha(registry_path) == registry_sha
    for item in registry["source_files"]:
        assert file_sha(root / item["path"]) == item["sha256"]
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == sorted(FAMILIES)
    builder.write(root.parent / (round_id + "_call_receipt.json"), {
        "fixture_source_registry_sha256": registry_sha, "fixture_directories": 8,
        "new_fixture_directories_this_test_run": 0, "calls": calls, "call_count": len(calls),
        "wall_seconds": time.monotonic() - started, "science_calls": 0, "source_hashes_unchanged": True})


def canonical(family, stem="plan"):
    plan, manifest, ms, records = family.load(stem)
    return plan, manifest, ms, stats.canonicalize_native(plan, manifest, ms, records)


def pick(result, metric="native_return", baseline="control"):
    return next(item for item in result["family"] if item["metric"] == metric and item["baseline"] == baseline)


def test_v20_complete_matrix_rule_and_retained_safeguards(families):
    f = families(0)
    bundle = f.cli("bundle", "produce")
    output = f.cli("analysis", "analyze", extra=["--bundle", str(bundle), "--expected_bundle_sha", file_sha(bundle)])
    result = json.loads(output.read_text())
    assert result["schema"] == stats.RESULT_SCHEMA and result["interval_semantics"] == "closed_native_sample_ticks"
    assert len(result["family"]) == result["holm"]["declared_family_size"] == 12
    positive, rule = pick(result), pick(result, baseline="rule")
    assert positive["logical_paired_row_count"] == 24 and positive["effective_independent_block_count"] == 4
    assert rule["baseline_physical_evaluation_count"] == 8 and rule["sign_test"]["independent_block_count"] == 4
    assert positive["candidate_favoring_benefit"] == pytest.approx(2.75)
    assert result["scientific_claim_ready"] is False and result["low_power"]
    plan, manifest, ms, rows = canonical(f)
    def reject_rows(label, modify, error):
        changed = copy.deepcopy(rows)
        modify(changed)
        f.invoke(label, "canonical_consumer_adversarial_view", lambda: stats.analyze_rows(plan, manifest, ms, changed), error)
    reject_rows("missing_outer", lambda r: r[0]["physical_outer_identity"].pop("verified_outer_time_unit_id"), "no fallback")
    reject_rows("missing_matrix", lambda r: r.pop(), "complete-case")
    reject_rows("duplicate_matrix", lambda r: r.append(copy.deepcopy(r[0])), "duplicate method/seed/case")
    reject_rows("wrong_geometry", lambda r: r[0]["exogenous_pair_identity"].update(rsu_geometry_fingerprint=builder.h("wrong geometry")), "unpaired exogenous")
    reject_rows("wrong_checkpoint", lambda r: r[0]["model_identity"].update(policy_checkpoint_sha256=builder.h("wrong CP")), "checkpoint")
    reject_rows("wrong_unit", lambda r: r[0]["endpoints"]["state_transfer_bytes"].update(unit="seconds"), "direction/unit")
    reject_rows("float_tick", lambda r: r[0]["physical_outer_identity"].update(raw_start_tick=1.0), "no float truncation")
    missing = copy.deepcopy(rows)
    missing[0]["endpoints"]["native_return"].update(status="unavailable", value=None, reason="frozen synthetic unavailable view")
    unavailable = f.invoke("unavailable_family", "kernel_in_memory_view", lambda: stats.analyze_rows(plan, manifest, ms, missing))
    assert len(unavailable["family"]) == 12 and pick(unavailable)["status"] == "unavailable"
    assert sum(r["unavailable_placeholder"] for r in unavailable["holm"]["correction"]) == 2
    degenerate = pick(result, "unused_optional_transfer_bytes")
    assert degenerate["raw_candidate_minus_baseline"] == -90 and degenerate["candidate_favoring_benefit"] == 90
    assert degenerate["percentile_95"]["status"] == "degenerate" and degenerate["bca_95"]["lower"] is None
    assert stats.sign_test(np.zeros(4))["ties"] == 4
    alias = copy.deepcopy(manifest)
    source = copy.deepcopy(alias["physical_sources"][0])
    source.update(source_segment_run_id="fake_run", canonical_capture_id="fake_capture", clock_domain_id="fake_clock")
    alias["physical_sources"].append(source)
    f.invoke("alias_relabel", "manifest_adversarial_view", lambda: stats.validate_contract(plan, alias), "source alias")
    corrupt = json.loads(bundle.read_text())
    corrupt["rows"][0]["endpoints"]["native_return"]["value"] += 999
    bad_path = f.path / (f.round_id + "_derived_tampered_bundle.json")
    builder.write(bad_path, corrupt)
    f.cli("tampered_consumer", "analyze", 1, ["--bundle", str(bad_path), "--expected_bundle_sha", file_sha(bad_path)])
    with pytest.raises(stats.ContractError, match="create-only"):
        f.cli("bundle", "produce")


def test_v21_touching_outer_reject_and_positive_gap_equality(families):
    f = families(1)
    f.cli("touching_outer", "produce", 1)
    failure = json.loads((f.path / (f.round_id + "_touching_outer.json.failure.json")).read_text())
    assert "touching" in failure["error"]
    plan, manifest, ms, rows = canonical(f, "positive_exact_gap_plan")
    result = f.invoke("positive_gap_equality", "frozen_positive_gap_view", lambda: stats.analyze_rows(plan, manifest, ms, rows))
    assert result["physical_units"][0]["end_seconds_exact"] == [2, 5]
    assert result["physical_units"][1]["start_seconds_exact"] == [1, 2]


def test_v22_touching_quarantine_reject(families):
    f = families(2)
    f.cli("touching_quarantine", "produce", 1)
    assert "quarantine" in (f.path / (f.round_id + "_touching_quarantine.json.failure.json")).read_text()


def test_v23_fractional_state_bytes_reject(families):
    f = families(3)
    f.cli("fractional_state_bytes", "produce", 1)
    assert "integer native" in (f.path / (f.round_id + "_fractional_state_bytes.json.failure.json")).read_text()


def test_v24_fractional_link_frames_reject(families):
    f = families(4)
    f.cli("fractional_link_frames", "produce", 1)
    assert "integer native" in (f.path / (f.round_id + "_fractional_link_frames.json.failure.json")).read_text()


def test_v25_shared_checkpoint_global_seed_support(families):
    f = families(5)
    plan, manifest, ms, rows = canonical(f)
    _, matrix = stats.validate_rows(plan, manifest, ms, rows)
    values = []
    for outer in range(4):
        data = np.zeros((1, 3, 2))
        for si, seed in enumerate(plan["training_seeds"]):
            for ri, regime in enumerate(plan["workload_regimes"]):
                case = "case_%d_%s" % (outer, regime)
                data[0, si, ri] = matrix[(case, "main", seed)]["endpoints"]["native_return"]["value"] - matrix[(case, "control", seed)]["endpoints"]["native_return"]["value"]
        values.append(data)
    draws = stats.bootstrap(values, np.array([1/3, 1/3, 1/3]), np.array([0.5, 0.5]), plan["random_seed"])
    indices = np.random.default_rng(plan["random_seed"]).choice(3, size=(10000, 3), p=[1/3, 1/3, 1/3])
    expected = np.array([0, 6, 12])[indices].mean(axis=1)
    assert np.array_equal(draws, expected)
    f.invoke("exact_global_seed_support", "bootstrap_calls_1", lambda: {"support": np.unique(draws).tolist(),
             "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(), "matches_global_seed_sequence_exactly": True})
    result = f.invoke("global_seed_analysis", "analyze_rows", lambda: stats.analyze_rows(plan, manifest, ms, rows))
    assert pick(result)["candidate_favoring_benefit"] == 6
    assert "shared across" in result["bootstrap"]["seed_scope"]


def test_v26_hand_weighted_estimand_and_one_block_outer_guard(families):
    f = families(6)
    plan, manifest, ms, rows = canonical(f)
    result = f.invoke("weighted_unequal_windows", "analyze_rows", lambda: stats.analyze_rows(plan, manifest, ms, rows))
    assert pick(result)["raw_candidate_minus_baseline"] == pytest.approx(623 / 3)
    effects = pick(result)["block_effects"]
    assert [r["raw_delta"] for r in effects] == pytest.approx([323/3, 923/3])
    plan1, manifest1, ms1, rows1 = canonical(f, "one_block_plan")
    one = f.invoke("one_block_no_outer_population_CI", "frozen_one_block_view", lambda: stats.analyze_rows(plan1, manifest1, ms1, rows1))
    r = pick(one)
    assert r["effective_independent_block_count"] == 1
    assert r["percentile_95"]["status"] == r["bca_95"]["status"] == "unavailable"
    assert r["percentile_95"]["lower"] is r["bca_95"]["upper"] is None
    assert r["paired_block_cohen_dz"]["value"] is None


def test_v27_explicit_semantics_and_v1_rejection(families):
    f = families(7)
    plan, manifest, ms, records = f.load()
    no_semantics = copy.deepcopy(manifest)
    del no_semantics["interval_semantics"]
    f.invoke("missing_semantics", "manifest_adversarial_view", lambda: stats.validate_contract(plan, no_semantics), "interval_semantics")
    half_open = copy.deepcopy(manifest)
    half_open["interval_semantics"] = "half_open_ticks"
    f.invoke("unsupported_semantics", "manifest_adversarial_view", lambda: stats.validate_contract(plan, half_open), "native interval semantics")
    for label, old_plan, old_manifest in (("v1_plan", dict(plan, schema="strict_stream_statistics_plan_v1"), manifest),
                                           ("v1_manifest", plan, dict(manifest, schema="strict_stream_statistics_manifest_v1"))):
        f.invoke(label, "schema_adversarial_view", lambda p=old_plan, m=old_manifest: stats.validate_contract(p, m), "unsupported plan/manifest")
    old_native = copy.deepcopy(records)
    old_native[0][1]["schema"] = "strict_stream_native_row_v1"
    f.invoke("v1_native", "schema_adversarial_view", lambda: stats.canonicalize_native(plan, manifest, ms, old_native), "native row schema")
    rows = stats.canonicalize_native(plan, manifest, ms, records)
    rows[0]["schema"] = "strict_stream_canonical_row_v1"
    f.invoke("v1_canonical", "schema_adversarial_view", lambda: stats.analyze_rows(plan, manifest, ms, rows), "canonical row schema")
    scientific = dict(plan, evidence_kind="audited_native_stream")
    f.invoke("scientific_payload", "schema_adversarial_view", lambda: stats.validate_contract(scientific, manifest), "software-only")
    bundle = f.cli("bundle", "produce")
    old_bundle = json.loads(bundle.read_text())
    old_bundle["schema"] = "strict_stream_statistics_bundle_v1"
    old_path = f.path / (f.round_id + "_derived_v1_bundle.json")
    builder.write(old_path, old_bundle)
    f.cli("v1_bundle", "analyze", 1, ["--bundle", str(old_path), "--expected_bundle_sha", file_sha(old_path)])
