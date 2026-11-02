"""One offline, single-use audit of the prospectively frozen Helmet24 matrix.

This consumer uses only pure identity, image, tensor and arithmetic helpers.
It never calls another audit entry point or loads a pretrained model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import signal
import sys

if __package__:
    from . import audit_helmet_orientation_choice as orient
else:
    import audit_helmet_orientation_choice as orient

old = orient.old
require, regular, load = old.require, old.regular, old.load
digest, create, encoded, clock = old.digest, old.create, old.encoded, old.clock
CAPS = {"model_processes": 2, "processor_load": 2, "base_load": 2, "adapter_load": 1, "activation": 1,
        "generate": 0, "score_forward": 96, "native_forward": 96, "worker_seconds": 2400,
        "producer_seconds": 4800, "consumer_seconds": 300, "wall_seconds": 5100,
        "artifact_bytes": 268435456, "log_bytes": 33554432, "HTTP": 0, "training": 0}
SCOPE_SHA = "a6a1755b10d35de9d6d6b60708b25504f1c283e7bbd1e91e3f8d612657680be2"
SELECTION_SHA = "388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743"
ARMS = ("base", "helmet")
CASES = tuple("p%02d" % n for n in range(24))
INPUT_IDS = tuple(case + "_display" for case in CASES)
LABELS = ("HELMET", "NO_HELMET")
PROMPTS, CANDIDATES = orient.PROMPTS, orient.CANDIDATES
WORKER_INPUT_KEYS = ("input_id", "case_id", "view", "path", "sha256", "rgb_sha256", "size")
MAIN_INPUT_KEYS = set(WORKER_INPUT_KEYS) | {"source_sample_id", "source_path", "source_sha256", "bbox_xywh", "orientation"}
CALL_OPS = ("processor_load", "base_load", "adapter_load", "activation", "generate", "score_forward", "native_forward")


def display_pixels(source, bbox):
    from PIL import Image, ImageOps
    require(isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox), "native integral bbox without rounding")
    with Image.open(regular(source)) as image:
        image.verify()
    with Image.open(source) as image:
        require(image.width * image.height <= 60000000, "prospective source pixel cap")
        orientation, raw_size = image.getexif().get(274), list(image.size)
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        x, y, width, height = old.native_bbox(bbox, rgb.size)
        require(64 <= width <= 256 and 64 <= height <= 256, "frozen prospective ROI dimensions")
        return {"pixels": rgb.crop((x, y, x + width, y + height)).tobytes(), "size": [width, height],
                "orientation": orientation, "raw_size": raw_size, "display_size": list(rgb.size)}


def checked_reference(ref):
    require(isinstance(ref, dict) and digest(ref["path"]) == ref["sha256"], "immutable named reference SHA")
    return load(ref["path"])


def qualified_inputs(plan, claim):
    scope = checked_reference(plan["scope_reference"])
    require(plan["scope_reference"]["sha256"] == SCOPE_SHA and scope["caps"] == CAPS and
            scope["before_current6_result_read"] is True and scope["resources"] == plan["resources"] and
            scope["runtime"] == plan["runtime"] and scope["prompts"] == PROMPTS and scope["candidate_ids"] == CANDIDATES,
            "prospective protocol fixed before current-six result access")
    protocol = checked_reference(scope["protocol_reference"])
    require(protocol["resources"] == plan["resources"] and protocol["runtime"] == plan["runtime"] and
            protocol["resource_reference"] == plan["resource_reference"] and protocol["prompts"] == PROMPTS and
            protocol["candidate_ids"] == CANDIDATES, "unchanged prior protocol plan only; no prior outcomes read")
    selection = checked_reference(plan["selection_reference"])
    require(plan["selection_reference"]["sha256"] == SELECTION_SHA == scope["selection_sha256"] and
            selection["selected_image_count"] == 24 and selection["selected_counts"] == {"Helmet": 12, "NoHelmet": 12} and
            [row["sample_id"] for row in selection["samples"]] == ["prospective_%02d" % n for n in range(24)],
            "same prospective frozen24; no replacement")
    ref = plan["acquisition_reference"]
    for name in ("plan", "audit", "terminal", "input_manifest"):
        require(digest(ref[name + "_path"]) == ref[name + "_sha256"], "accepted acquisition reference: " + name)
    acquisition, audit, terminal, manifest = (load(ref[name + "_path"]) for name in ("plan", "audit", "terminal", "input_manifest"))
    require(audit["audit_success"] is True and audit["input_qualified"] is True and audit["selection_sha256"] == SELECTION_SHA and
            audit["plan_sha256"] == ref["plan_sha256"] and terminal["audit_success"] is True and terminal["input_qualified"] is True and
            terminal["sha256"] == ref["audit_sha256"] and regular(terminal["audit"]) == regular(ref["audit_path"]) and
            audit["output_files"][ref["input_manifest_path"]] == ref["input_manifest_sha256"] and
            manifest["schema"] == "helmet24_display_inputs_v1" and manifest["selection_sha256"] == SELECTION_SHA,
            "matching successful acquisition audit/terminal/manifest; no audit rerun")
    # The acquisition plan itself must bind the very same frozen selection.
    require(acquisition["selection"] == plan["selection_reference"], "acquisition selection reference")
    qc = checked_reference(plan["visual_qualification_reference"])
    require(qc["input_manifest_sha256"] == ref["input_manifest_sha256"] and qc["selection_sha256"] == SELECTION_SHA and
            qc["all24_accepted"] is True and qc["model_outputs_read"] is False and qc["reviewer"] and qc["reviewed_at"] and
            qc["cases"] == [{"sample_id": "prospective_%02d" % n, "accepted": True} for n in range(24)] and
            qc["boot_uuid"] == claim["boot_uuid"] and type(qc["completed_clock"]) in (int, float) and
            terminal["finished_clock"] <= qc["completed_clock"] < claim["started_clock"] and
            digest(qc["contact_sheet_path"]) == qc["contact_sheet_sha256"] == audit["output_files"][qc["contact_sheet_path"]],
            "all24 independent semantic-ROI review frozen before any model; invalid case blocks")
    require(len(manifest["samples"]) == len(plan["inputs"]) == 24 and [row["input_id"] for row in plan["inputs"]] == list(INPUT_IDS),
            "complete exact24 input order")
    labels, rows, publisher_hashes, image_ids, file_ids = {}, {}, set(), set(), set()
    for number, (selected, accepted, row) in enumerate(zip(selection["samples"], manifest["samples"], plan["inputs"])):
        case_id, sample_id = CASES[number], "prospective_%02d" % number
        publisher = selected["publisher_file"]
        details, annotation = publisher["content_details"], selected["native_annotation"]
        bbox = annotation["bbox"]
        require(all(type(v) in (int, float) and math.isfinite(v) and int(v) == v for v in bbox), "unchanged exact native ROI")
        bbox = list(map(int, bbox))
        label = {"Helmet": "HELMET", "NoHelmet": "NO_HELMET"}[selected["category_name"]]
        require(accepted["sample_id"] == row["source_sample_id"] == sample_id and set(row) == MAIN_INPUT_KEYS and
                row["input_id"] == INPUT_IDS[number] and row["case_id"] == case_id and row["view"] == "display" and
                accepted["source_path"] == row["source_path"] and accepted["path"] == row["path"] and
                accepted["source_sha256"] == row["source_sha256"] == details["sha256_hash"] and
                regular(row["source_path"]).stat().st_size == accepted["source_bytes"] == details["size"] and
                ("size" not in publisher or publisher["size"] == details["size"]) and
                digest(row["source_path"]) == details["sha256_hash"] and
                accepted["bbox_xywh"] == row["bbox_xywh"] == bbox and accepted["category_name"] == selected["category_name"] and
                accepted["image_id"] == selected["image_id"] == annotation["image_id"] == selected["native_image"]["id"] and
                accepted["annotation_id"] == selected["annotation_id"] == annotation["id"] and
                plan["consumer_groundtruth"][case_id] == label, "exact publisher image/nativeGT/full-case binding")
        require(details["sha256_hash"] not in publisher_hashes and selected["image_id"] not in image_ids and publisher["id"] not in file_ids and
                details["sha256_hash"] not in selection["exclusions"]["old_publisher_sha256"] and
                selected["image_id"] not in selection["exclusions"]["old_image_ids"], "24 unique unconsumed source images/hashes")
        publisher_hashes.add(details["sha256_hash"])
        image_ids.add(selected["image_id"])
        file_ids.add(publisher["id"])
        pixels = display_pixels(row["source_path"], bbox)
        require(accepted["raw_size"] == pixels["raw_size"] and accepted["display_size"] == pixels["display_size"] and
                accepted["orientation"] == row["orientation"] == pixels["orientation"] and
                accepted["size"] == row["size"] == pixels["size"] and accepted["sha256"] == row["sha256"] and
                accepted["rgb_sha256"] == row["rgb_sha256"], "canonical EXIF display/native ROI identity")
        orient.verify_png(accepted["path"], pixels["pixels"], pixels["size"], accepted["sha256"], accepted["rgb_sha256"])
        orient.verify_png(row["path"], pixels["pixels"], pixels["size"], row["sha256"], row["rgb_sha256"])
        labels[case_id], rows[INPUT_IDS[number]] = label, row
    require(set(plan["consumer_groundtruth"]) == set(CASES) and all(list(labels.values()).count(label) == 12 for label in LABELS), "frozen balanced24 private GT")
    return labels, rows, {"audit_sha256": ref["audit_sha256"], "plan_sha256": ref["plan_sha256"],
                          "input_manifest_sha256": ref["input_manifest_sha256"], "visual_QC_sha256": plan["visual_qualification_reference"]["sha256"]}


def call_matrix(arm):
    rows = [("processor_load", "processor"), ("base_load", "base")]
    if arm == "helmet":
        rows += [("adapter_load", "adapter"), ("activation", "activate")]
    return rows + [("score_forward", name + "_m%d" % mapping) for name in INPUT_IDS for mapping in (0, 1)]


def events(path, arm, pid, started, finished):
    expected, order, stack, windows = call_matrix(arm), [], [], {}
    counts, last = {op: 0 for op in CALL_OPS}, started
    for row in old.operation_lines(path):
        op, edge, call_id, t = row["op"], row["edge"], row["call_id"], row["clock"]
        require(op in CALL_OPS and edge in ("begin", "end") and row["pid"] == pid and row["arm"] == arm and
                type(t) in (int, float) and math.isfinite(t) and last <= t <= finished, "actual event identity/time")
        last = t
        if edge == "begin":
            counts[op] += 1
            if op == "native_forward":
                require(len(stack) == 1 and stack[0] == ("score_forward", call_id) and row["parent_op"] == "score_forward" and
                        row["inference_mode"] is True and row["grad_enabled"] is False, "actual scoring forward / inference-mode parent")
                windows[stack[0]]["native_forwards"] += 1
            else:
                require(not stack and len(order) < len(expected) and expected[len(order)] == (op, call_id), "fixed48-score sequence; no generation/retry")
                order.append((op, call_id))
                windows[(op, call_id)] = {"started": t, "native_forwards": 0}
            stack.append((op, call_id))
        else:
            require(stack and stack[-1] == (op, call_id) and row["status"] == "PASS", "matched successful event end")
            if op == "native_forward":
                require(len(stack) == 2 and row["parent_op"] == "score_forward", "native end parent")
            stack.pop()
            if op != "native_forward":
                windows[(op, call_id)]["finished"] = t
    require(not stack and order == expected, "complete fixed48-score ledger; partial fails")
    require(counts["generate"] == 0 and counts["score_forward"] == counts["native_forward"] == 48 and
            all(window["native_forwards"] == int(op == "score_forward") for (op, _), window in windows.items()), "exact one native forward per score / no hidden forward")
    return counts, windows


def worker_spec(plan, arm, inputs):
    ref = plan["worker_plans"][arm]
    spec = checked_reference(ref)
    require(set(spec) == {"arm", "runtime", "resources", "source_commit", "source_sha", "caps", "inputs", "prompts", "candidate_ids"} and
            spec["arm"] == arm and all(spec[key] == plan[key] for key in ("runtime", "resources", "source_commit", "source_sha", "caps", "prompts", "candidate_ids")) and
            spec["inputs"] == [{key: inputs[name][key] for key in WORKER_INPUT_KEYS} for name in INPUT_IDS], "exact label-free fixed worker plan")
    return ref["sha256"]


def audit_worker(root, arm, plan, inputs, original_parameters, vocabulary):
    out = regular(root / arm, directory=True)
    start, complete = load(out / "start.json"), load(out / "complete.json")
    process, intent = load(root / (arm + "_process.json")), load(root / (arm + "_process_intent.json"))
    ref_sha = worker_spec(plan, arm, inputs)
    command = [plan["runtime"]["python"], "-u", str(old.SOURCE_ROOT / "scripts/run_helmet_prospective_choice.py"),
               "--worker_plan", plan["worker_plans"][arm]["path"], "--worker_sha", ref_sha, "--out", str(out)]
    require(intent["command"] == command and intent["started"] == process["started"] and
            type(intent["seconds"]) in (int, float) and 0 < intent["seconds"] <= CAPS["worker_seconds"] and
            process["returncode"] == 0 and process["exception"] is None and process["terminated_group"] is False,
            "one successful supervised worker / no retry")
    pid = start["pid"]
    require(type(pid) is int and pid > 0 and pid == process["pid"] == complete["pid"] and start["arm"] == arm and
            start["worker_plan_sha256"] == ref_sha and process["started"] <= start["started_clock"] <= complete["finished_clock"] <= process["finished"] and
            process["finished"] - process["started"] <= CAPS["worker_seconds"] and complete["versions"] == plan["runtime"]["versions"] and
            complete["module_origins"] == plan["runtime"]["module_origins"] and complete["threads"] == [1, 1] and
            not (out / "failure.json").exists(), "actual fresh worker runtime/interval")
    counts, windows = events(out / "operations.jsonl", arm, pid, start["started_clock"], complete["finished_clock"])
    require(set(complete["counts"]) <= set(CALL_OPS) and all(complete["counts"].get(op, 0) == counts[op] for op in CALL_OPS), "independent actual event totals")
    identity = load(out / "identity.json")
    require(identity["base_weight_sha256"] == plan["resources"]["base"]["weight_sha256"] and identity["model_training"] is False and
            identity["scoring_kwargs"] == {"use_cache": False}, "same real eval base / explicit no-cache scoring")
    config = identity["generation_config"]
    require(config["bos_token_id"] == 0 and config["eos_token_id"] == 49279 and config["pad_token_id"] == 2, "native special-token config retained; no generate calls")
    if arm == "base":
        require(identity["is_peft"] is False and not (out / "parameters.json").exists(), "adapter-free base")
    else:
        require(identity["is_peft"] is True and identity["adapter_weight_sha256"] == plan["resources"]["adapter"]["weight_sha256"] and
                identity["active"] == identity["available"] == ["helmet"] and identity["enabled"] is True, "one actual enabled frozen adapter")
        old.loaded_parameters(load(out / "parameters.json"), original_parameters)
    verified = load(out / "resources.json")
    require(set(verified) == set(plan["resources"]), "worker complete resource groups")
    for name, resource in plan["resources"].items():
        require(set(verified[name]) == set(resource["files"]), "worker exact resource file inventory")
        for relative, record in resource["files"].items():
            require(verified[name][relative]["bytes"] == record["bytes"] and verified[name][relative]["sha256"] == record["sha256"], "worker native resource SHA/size")
    names = [name + "_m%d" % mapping for name in INPUT_IDS for mapping in (0, 1)]
    require(sorted(path.name for path in regular(out / "scores", directory=True).iterdir()) == sorted(names) and not (out / "generated").exists(), "48 exact scores and no generated outputs")
    scores, fingerprints = [], {}
    for input_id in INPUT_IDS:
        for mapping, prompt in enumerate(PROMPTS):
            name = input_id + "_m%d" % mapping
            directory = out / "scores" / name
            claim, receipt = load(directory / "claim.json"), load(directory / "receipt.json")
            require(not any(name in claim["rendered_prompt"] for name in INPUT_IDS), "opaque input IDs never enter prompt")
            fingerprints[name] = orient.actual_inputs(claim, inputs[input_id], prompt)
            window = windows[("score_forward", name)]
            require(claim["mapping"] == mapping and receipt["native_forwards"] == window["native_forwards"] == 1 and
                    receipt["use_cache"] is False and receipt["selected_position"] == claim["selected_position"] and
                    start["started_clock"] <= claim["started_clock"] <= window["started"] <= window["finished"] <= receipt["finished_clock"] <= complete["finished_clock"],
                    "one first-next-token score and complete raw-call interval")
            values = orient.logit_file(directory / "logits.npy", receipt, vocabulary)
            scores.append({"arm": arm, "input_id": input_id, "case_id": inputs[input_id]["case_id"], "mapping": mapping,
                           **values, "p_Helmet_mapping": values["p_A_conditional"] if mapping == 0 else 1.0 - values["p_A_conditional"],
                           "logits_sha256": receipt["logits_sha256"], "claim_sha256": digest(directory / "claim.json"),
                           "receipt_sha256": digest(directory / "receipt.json")})
    return {"pid": pid, "started": process["started"], "finished": process["finished"], "counts": counts,
            "generation_config": config, "fingerprints": fingerprints, "scores": scores}


def failure_bill(root):
    arms, aggregate, upper = [], {op: 0 for op in CALL_OPS}, {op: 0 for op in CALL_OPS}
    for arm in ARMS:
        intent_state, intent = old.optional_metadata(root / (arm + "_process_intent.json"))
        process_state, process = old.optional_metadata(root / (arm + "_process.json"))
        start_state, start = old.optional_metadata(root / arm / "start.json")
        pids = sorted({obj["pid"] for obj in (process, start) if isinstance(obj, dict) and type(obj.get("pid")) is int and obj["pid"] > 0})
        observed, errors = {op: 0 for op in CALL_OPS}, []
        path = root / arm / "operations.jsonl"
        if path.exists() or path.is_symlink():
            try:
                for number, line in enumerate(regular(path).read_text(encoding="utf-8").splitlines()):
                    try:
                        row = old.json_object(line)
                        require(isinstance(row, dict), "event object required")
                        if row.get("edge") == "begin" and row.get("op") in observed:
                            observed[row["op"]] += 1
                    except (ValueError, TypeError) as error:
                        errors.append({"line": number + 1, "error": str(error)})
            except (OSError, ValueError, TypeError) as error:
                errors.append({"error": str(error)})
        else:
            errors.append({"error": "missing ledger; calls unknown, never certified zero"})
        limits = {"processor_load": 1, "base_load": 1, "adapter_load": int(arm == "helmet"), "activation": int(arm == "helmet"),
                  "generate": 0, "score_forward": 48, "native_forward": 48}
        row = {"arm": arm, "launch_intent": intent_state, "process_receipt": process_state, "worker_start": start_state,
               "launch_attempt_observed": int(any(obj is not None for obj in (intent, process, start))), "launch_attempt_upper": 1,
               "PID_created": True if pids else False if process is not None and process.get("pid", "missing") is None else None,
               "observed_PIDs": pids, "worker_start_observed": start is not None, "observed_begin_counts": observed,
               "unknown_inclusive_upper": {op: max(observed[op], limits[op]) for op in CALL_OPS}, "ledger_errors": errors,
               "counts_are_lower_bounds": True}
        for op in CALL_OPS:
            aggregate[op] += observed[op]
            upper[op] += row["unknown_inclusive_upper"][op]
        arms.append(row)
    consumer_state, _ = old.optional_metadata(root / "consumer_process_intent.json")
    return {"arms": arms, "observed_begin_counts": aggregate, "unknown_inclusive_upper": upper,
            "consumer_launch_intent": consumer_state, "consumer_is_not_a_model_process": True,
            "missing_or_partial_ledger_does_not_mean_zero": True, "old_and_new_acquisition_costs_not_reset": True}


def summarise(workers, labels):
    mapping = [row for arm in ARMS for row in workers[arm]["scores"]]
    rows, summary, by_arm = [], {}, {}
    for arm in ARMS:
        index = {(row["input_id"], row["mapping"]): row for row in workers[arm]["scores"]}
        require(len(index) == 48, "all48 distinct arm mapping scores")
        confusion = {truth: {prediction: 0 for prediction in (*LABELS, "ABSTAIN")} for truth in LABELS}
        class_correct = {label: 0 for label in LABELS}
        by_arm[arm] = {}
        for case_id, input_id in zip(CASES, INPUT_IDS):
            zero, one = index[(input_id, 0)], index[(input_id, 1)]
            probability, prediction = orient.mapped_mean(zero["p_A_conditional"], one["p_A_conditional"])
            truth, correct = labels[case_id], prediction == labels[case_id]
            confusion[truth][prediction] += 1
            class_correct[truth] += int(correct)
            per_map = ["HELMET" if entry["p_Helmet_mapping"] > .5 else "NO_HELMET" if entry["p_Helmet_mapping"] < .5 else "ABSTAIN" for entry in (zero, one)]
            row = {"arm": arm, "case_id": case_id, "input_id": input_id, "groundtruth": truth,
                   "p_Helmet": probability, "prediction": prediction, "correct": correct, "abstain": prediction == "ABSTAIN",
                   "mapping_disagreement": per_map[0] != per_map[1], "map0_Helmet_probability": zero["p_Helmet_mapping"],
                   "map1_Helmet_probability": one["p_Helmet_mapping"]}
            rows.append(row)
            by_arm[arm][case_id] = row
        correct = sum(class_correct.values())
        summary[arm] = {"correct": correct, "total_cases": 24, "descriptive_accuracy": correct / 24,
                        "class_correct": class_correct, "class_total": {label: 12 for label in LABELS}, "confusion": confusion,
                        "abstentions_counted_incorrect": sum(confusion[label]["ABSTAIN"] for label in LABELS),
                        "all24_conditionally_correct": correct == 24}
    paired = {"both_correct": 0, "base_only_correct": 0, "adapter_only_correct": 0, "both_wrong": 0}
    for case in CASES:
        base, helmet = by_arm["base"][case]["correct"], by_arm["helmet"][case]["correct"]
        paired["both_correct" if base and helmet else "base_only_correct" if base else "adapter_only_correct" if helmet else "both_wrong"] += 1
    return {"mapping_scores": mapping, "case_scores": rows, "summary": summary, "paired_correctness_counts": paired,
            "mean_formula": "pHelmet=0.5+0.5*(pA_map0-pA_map1)", "independent_labelled_cases": "UNVERIFIED",
            "interpretation": "24 fixed public cases,12 per native class; descriptive only, no CI/confirmatoryp or natural-prevalence claim",
            "probabilities": "conditional A/B scores, not calibrated probabilities; full-vocabulary A+B mass retained",
            "training_overlap": "UNVERIFIED", "integrity_pass_is_independent_of_accuracy": True}


def audit(plan_path, expected_plan_sha, root, output):
    root, plan_path, output = regular(root, directory=True), regular(plan_path), Path(output).absolute()
    terminal, attempt = root / "terminal.json", root / "consumer_claim.json"
    require(not any(path.exists() or path.is_symlink() for path in (output, terminal, attempt, root / "consumer_failure.json")), "single-use prospective consumer; no rerun")
    regular(output.parent, directory=True)
    started, claim = clock(), None
    create(attempt, {"started_clock": started, "consumer_sha256": digest(__file__), "expected_plan_sha256": expected_plan_sha,
                     "single_use": True, "old_audit_reruns": 0, "model_calls": 0})
    result = {"reviewed_at": datetime.now(timezone.utc).isoformat(), "literature_cutoff": "No new literature search",
              "target_venue": "CSCWD2027; TMC internal evidence benchmark only", "artifact_run_id": root.name,
              "policy_version": "tmc_review_policy_v3_20260621", "git_commit": None, "evidence_level": "E0_UNAVAILABLE",
              "audit_success": False, "matrix_complete": False, "prospective_fixed_cases": 24, "class_counts": {"Helmet": 12, "NoHelmet": 12},
              "hidden": 0, "HTTP": 0, "training": 0, "generate": 0, "training_independence": "UNVERIFIED", "generalisation": "UNVERIFIED",
              "natural_prevalence": False, "canonical_promotion": False, "paper_goal_complete": False,
              "integrity_pass_is_independent_of_accuracy": True, "requires_matching_terminal": True, "previous_costs_and_outputs_retained": True}
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("prospective consumer deadline")))
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, CAPS["consumer_seconds"] - 2)
        require(digest(plan_path) == digest(root / "plan.json") == expected_plan_sha, "external/copied frozen plan SHA")
        plan, claim = load(plan_path), load(root / "claim.json")
        result["git_commit"] = plan["source_commit"]
        require(plan["schema"] == "helmet24_choice_plan_v1" and plan["caps"] == CAPS and plan["prompts"] == PROMPTS and
                plan["candidate_ids"] == CANDIDATES and Path(plan["raw_root"]).absolute() == root and Path(plan["audit_output"]).absolute() == output,
                "fixed prospective matrix and caps")
        require(claim["plan_sha256"] == expected_plan_sha and claim["boot_uuid"] == old.boot(), "new scope plan/boot binding")
        grant = load(root / "grant.json")
        require(digest(root / "grant.json") == claim["grant_sha256"] and grant["plan_sha256"] == expected_plan_sha and grant["allow_execution"] is True, "fixed new-scope authorization")
        elapsed = clock() - claim["started_clock"]
        require(0 <= elapsed < CAPS["wall_seconds"] - 2, "new5100-second absolute deadline")
        signal.setitimer(signal.ITIMER_REAL, min(CAPS["consumer_seconds"] - 2, CAPS["wall_seconds"] - elapsed - 2))
        dependencies = {"scripts/audit_helmet_prospective_choice.py", "scripts/run_helmet_prospective_choice.py", "scripts/audit_helmet_orientation_choice.py", "scripts/run_helmet_orientation_choice.py",
                        "scripts/audit_helmet_quality_pilot.py", "scripts/run_helmet_quality_pilot.py"}
        require(dependencies.issubset(plan["source_sha"]) and plan["source_sha"]["scripts/audit_helmet_prospective_choice.py"] == digest(__file__), "all new/shared source SHA bindings")
        old.source_identity(plan)
        old.runtime_identity(plan["runtime"])
        stored, logs, outer = old.tree_bytes(plan["storage_roots"])
        require(plan["storage_roots"] == [str(root.parent)] and output.parent == root.parent and any(plan_path.is_relative_to(base) for base in outer) and
                stored + 1048576 <= CAPS["artifact_bytes"] and logs + 1048576 <= CAPS["log_bytes"], "all new-scope tree bytes/logs/final reserve")
        originals = old.resource_identity(plan)
        labels, inputs, references = qualified_inputs(plan, claim)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(regular(Path(plan["resources"]["base"]["path"]) / "tokenizer.json")))
        require([tokenizer.encode(value, add_special_tokens=False).ids for value in (" A", " B")] == [[330], [389]], "fixed exact single-token candidates")
        config = load(Path(plan["resources"]["base"]["path"]) / "config.json")
        vocabulary = config.get("text_config", config)["vocab_size"]
        require(type(vocabulary) is int and vocabulary > max(CANDIDATES), "complete native vocabulary")
        workers = {arm: audit_worker(root, arm, plan, inputs, originals, vocabulary) for arm in ARMS}
        require(workers["base"]["pid"] != workers["helmet"]["pid"] and
                claim["started_clock"] <= workers["base"]["started"] <= workers["base"]["finished"] <= workers["helmet"]["started"] <= workers["helmet"]["finished"] <= claim["started_clock"] + CAPS["producer_seconds"],
                "two sequential fresh processes within frozen producer interval")
        require(workers["base"]["fingerprints"] == workers["helmet"]["fingerprints"] and workers["base"]["generation_config"] == workers["helmet"]["generation_config"], "exact actual prompts/IDs/all image tensors/options across arms")
        for arm in ARMS:
            for input_id in INPUT_IDS:
                zero, one = (workers[arm]["fingerprints"][input_id + "_m%d" % mapping]["tensors"] for mapping in (0, 1))
                require(zero["pixel_values"] == one["pixel_values"], "same actual image tensor in both label mappings")
        counts = {op: sum(workers[arm]["counts"][op] for arm in ARMS) for op in CALL_OPS}
        require(all(counts[op] == CAPS[op] for op in CALL_OPS), "complete exact96 explicit/native forwards; zero generation")
        producer_complete = load(root / "producer_complete.json")
        require(not (root / "producer_failure.json").exists() and workers["helmet"]["finished"] <= producer_complete["finished_clock"] <= started and
                producer_complete["finished_clock"] <= claim["started_clock"] + CAPS["producer_seconds"], "successful bounded producer before single consumer")
        findings = summarise(workers, labels)
        exports = {output.parent / "choice_mapping_scores.csv": orient.csv_bytes(findings["mapping_scores"]),
                   output.parent / "choice_case_scores.csv": orient.csv_bytes(findings["case_scores"]),
                   output.parent / "prospective_choice_results.json": encoded(findings)}
        for path, data in exports.items():
            create(path, data)
        final_bytes, final_logs, _ = old.tree_bytes(plan["storage_roots"])
        require(final_bytes + 1048576 <= CAPS["artifact_bytes"] and final_logs + 1048576 <= CAPS["log_bytes"] and
                old.boot() == claim["boot_uuid"] and clock() - claim["started_clock"] < CAPS["wall_seconds"] and clock() - started < CAPS["consumer_seconds"] - 1,
                "final full tree/deadline/boot validation")
        result.update(status="AUDITED_PROSPECTIVE_HELMET24_CHOICE", audit_success=True, matrix_complete=True, evidence_level="E2_ARTIFACT_AUDITED",
                      operation_counts=counts, model_processes=2, summary=findings["summary"], paired_correctness_counts=findings["paired_correctness_counts"],
                      plan_sha256=expected_plan_sha, scope_sha256=SCOPE_SHA, selection_sha256=SELECTION_SHA, qualification_references=references,
                      mean_formula=findings["mean_formula"], output_files={str(path): hashlib.sha256(data).hexdigest() for path, data in exports.items()},
                      artifact_bytes_before_audit=final_bytes, log_bytes_before_audit=final_logs)
    except BaseException as error:
        result.update(status="PROSPECTIVE_CHOICE_UNVERIFIED_FAILURE_PRESERVED", audit_success=False, matrix_complete=False,
                      error_type=type(error).__name__, error=str(error), failure_bill=failure_bill(root))
        create(root / "consumer_failure.json", {"audit_success": False, "error_type": type(error).__name__, "error": str(error), "failure_bill": result["failure_bill"], "consumer_must_not_rerun": True})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result["consumer_elapsed_seconds"] = clock() - started
    result["scope_elapsed_seconds"] = clock() - claim["started_clock"] if isinstance(claim, dict) and type(claim.get("started_clock")) in (int, float) else None
    try:
        remaining = CAPS["consumer_seconds"] - (clock() - started)
        if result["audit_success"]:
            remaining = min(remaining, CAPS["wall_seconds"] - result["scope_elapsed_seconds"])
        require(remaining > 0, "terminal publication reserve")
        signal.setitimer(signal.ITIMER_REAL, min(2, remaining))
        create(output, result)
        create(terminal, {"audit": str(output), "sha256": digest(output), "audit_success": result["audit_success"], "matrix_complete": result["matrix_complete"],
                          "finished_clock": clock(), "consumer_must_not_rerun": True})
    except BaseException as error:
        if not (root / "consumer_failure.json").exists():
            create(root / "consumer_failure.json", {"audit_success": False, "error": str(error), "failure_bill": failure_bill(root), "consumer_must_not_rerun": True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan_sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.plan_sha256, args.root, args.output)
    print(encoded({"status": result["status"], "audit_success": result["audit_success"], "summary": result.get("summary")}).decode(), end="")
    raise SystemExit(0 if result["audit_success"] else 1)


if __name__ == "__main__":
    main()
