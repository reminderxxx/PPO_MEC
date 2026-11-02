"""Single-use offline audit of the consumed-case Helmet orientation diagnostic.

Only immutable helper functions are imported from the old consumer. Its audit()
is never invoked. No model, processor, inference, training or network is used.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import struct
import time
import signal

if __package__:
    from . import audit_helmet_quality_pilot as old
else:
    import audit_helmet_quality_pilot as old

require, regular, load = old.require, old.regular, old.load
digest, create, encoded, clock = old.digest, old.create, old.encoded, old.clock
CAPS = {"model_processes": 2, "processor_load": 2, "base_load": 2, "adapter_load": 1, "activation": 1,
        "generate": 4, "score_forward": 56, "native_forward": 184, "max_new_tokens": 32,
        "producer_seconds": 2400, "worker_seconds": 1200, "consumer_seconds": 300, "wall_seconds": 2700,
        "artifact_bytes": 268435456, "log_bytes": 33554432, "HTTP": 0, "training": 0}
ARMS, CASES = ("base", "helmet"), tuple("q%02d" % n for n in range(6))
PROMPTS = [
    "Look at this cropped head region. Does it contain a motorcycle helmet? A: helmet present. B: no helmet present. Reply with A or B only.",
    "Look at this cropped head region. Does it contain a motorcycle helmet? A: no helmet present. B: helmet present. Reply with A or B only."]
CANDIDATES = [330, 389]
INPUT_IDS = [case + "_display" for case in CASES] + [case + "_gray" for case in CASES] + ["q02_raw", "q03_raw"]
GEN_IDS = ("q02_display", "q03_display")
GENERATION = {"do_sample": False, "num_beams": 1, "use_cache": True, "max_new_tokens": 32}
TENSOR_OPS = ("generate", "score_forward")
CALL_OPS = ("processor_load", "base_load", "adapter_load", "activation", "generate", "score_forward", "native_forward")


def canonical_pixels(source, bbox):
    """One prescribed EXIF-display transform and the separately retained raw control."""
    from PIL import Image, ImageOps
    require(isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox), "unchanged native integer xywh")
    with Image.open(regular(source)) as image:
        image.verify()
    with Image.open(source) as image:
        orientation = image.getexif().get(274)
        display = ImageOps.exif_transpose(image).convert("RGB")
        x, y, w, h = old.native_bbox(bbox, display.size)
        displayed = display.crop((x, y, x + w, y + h)).tobytes()
        raw_image = image.convert("RGB")
        old.native_bbox(bbox, raw_image.size)
        raw = raw_image.crop((x, y, x + w, y + h)).tobytes()
        return {"orientation": orientation, "size": [w, h], "display_size": list(display.size),
                "raw_size": list(raw_image.size), "display": displayed, "raw": raw,
                "gray": bytes([128]) * (w * h * 3)}


def verify_png(path, expected_bytes, size, expected_sha, expected_rgb):
    from PIL import Image
    require(digest(path) == expected_sha and hashlib.sha256(expected_bytes).hexdigest() == expected_rgb, "frozen PNG/pixel SHA")
    with Image.open(regular(path)) as image:
        require(image.format == "PNG" and image.mode == "RGB" and list(image.size) == size and not image.info,
                "clean opaque RGB PNG geometry/metadata")
        image.load()
        require(image.tobytes() == expected_bytes, "PNG pixels differ from prescribed orientation/control")


def probability_values(logits):
    """The A/B conditional probability is not a calibrated full-vocabulary probability."""
    import numpy as np
    require(isinstance(logits, np.ndarray) and logits.ndim == 1 and logits.dtype == np.dtype("<f4") and
            len(logits) > max(CANDIDATES) and np.isfinite(logits).all(), "complete finite float32 vocabulary logits")
    values = logits.astype(np.float64)
    a, b = float(values[CANDIDATES[0]]), float(values[CANDIDATES[1]])
    shift = max(a, b)
    wa, wb = math.exp(a - shift), math.exp(b - shift)
    p_a = wa / (wa + wb)
    all_max = float(np.max(values))
    all_weight = float(np.exp(values - all_max).sum(dtype=np.float64))
    pair_weight = math.exp(a - all_max) + math.exp(b - all_max)
    return {"logit_A": a, "logit_B": b, "p_A_conditional": p_a,
            "p_B_conditional": 1.0 - p_a, "candidate_full_vocab_mass": pair_weight / all_weight}


def mapped_mean(map0_p_a, map1_p_a):
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in (map0_p_a, map1_p_a)), "finite conditional probability")
    # Algebraic two-mapping mean; identical A priors give exactly .5 in float64.
    probability = 0.5 + 0.5 * (map0_p_a - map1_p_a)
    prediction = "HELMET" if probability > .5 else "NO_HELMET" if probability < .5 else "ABSTAIN"
    return probability, prediction


def logit_file(path, receipt, vocabulary):
    import numpy as np
    require(digest(path) == receipt["logits_sha256"] and regular(path).stat().st_size <= 4 * vocabulary + 65536,
            "full logits file SHA/size")
    values = np.load(path, allow_pickle=False, max_header_size=65536)
    require(receipt["shape"] == list(values.shape) == [vocabulary] and receipt["dtype"] == "float32", "native full vocabulary shape/dtype")
    return probability_values(values)


def parent_inputs(plan):
    ref = plan["parent_reference"]
    for name in ("plan", "audit", "terminal", "GT", "raw_results"):
        require(digest(ref[name + "_path"]) == ref[name + "_sha"], "immutable old Q reference: " + name)
    parent, audit, terminal, gt = load(ref["plan_path"]), load(ref["audit_path"]), load(ref["terminal_path"]), load(ref["GT_path"])
    require(audit["audit_success"] is True and audit["paired_matrix_complete"] is True and
            terminal["audit_success"] is True and terminal["sha256"] == ref["audit_sha"] and
            Path(terminal["audit"]).resolve() == Path(ref["audit_path"]).resolve() and
            parent["input_reference"]["groundtruth_path"] == ref["GT_path"] and
            parent["input_reference"]["groundtruth_sha"] == ref["GT_sha"], "old Q artifact terminal/GT binding; no old audit rerun")
    require(audit["output_files"][ref["raw_results_path"]] == ref["raw_results_sha"] and
            plan["cases"] == parent["cases"] and plan["runtime"] == parent["runtime"] and
            plan["resources"] == parent["resources"] and plan["resource_reference"] == parent["resource_reference"] and
            plan["generation_prompt"] == parent["prompt"] == old.PROMPT and plan["generation"] == parent["generation"] == GENERATION,
            "same consumed six / original prompt, resources and runtime")
    require(tuple(item["case_id"] for item in plan["cases"]) == CASES and len(gt["samples"]) == 6, "exact original six")
    labels, pixels = {}, {}
    for case, saved in zip(plan["cases"], gt["samples"]):
        require(case["source_sample_id"] == saved["sample_id"] and case["source_path"] == saved["input_path"] and
                case["source_sha"] == saved["image_sha256"] == digest(case["source_path"]) and
                case["bbox_xywh"] == saved["native_annotation"]["bbox"], "native image / ROI identity")
        expected_label = {"Helmet": "HELMET", "NoHelmet": "NO_HELMET"}[saved["category_name"]]
        require(parent["consumer_groundtruth"][case["case_id"]] == expected_label, "original private groundtruth")
        data = canonical_pixels(case["source_path"], case["bbox_xywh"])
        require(data["size"] == [case["width"], case["height"]] and
                hashlib.sha256(data["raw"]).hexdigest() == case["crop_rgb_sha"], "legacy raw ROI remains exact")
        verify_png(case["crop_path"], data["raw"], data["size"], case["crop_sha"], case["crop_rgb_sha"])
        changed = data["raw"] != data["display"]
        require(changed == (case["case_id"] in ("q02", "q03")), "unique changed orientation cases fixed before model work")
        if changed:
            require(data["orientation"] == 6, "only diagnosed EXIF6 repair")
        labels[case["case_id"]], pixels[case["case_id"]] = expected_label, data
    require([item["input_id"] for item in plan["inputs"]] == INPUT_IDS, "exact display/gray/raw matrix")
    by_case = {case["case_id"]: case for case in plan["cases"]}
    inputs = {}
    for row in plan["inputs"]:
        case = by_case[row["case_id"]]
        view = row["view"]
        require(set(row) == {"input_id", "case_id", "view", "path", "sha256", "rgb_sha256", "size", "source_path", "source_sha256", "bbox_xywh", "orientation"} and
                row["input_id"] == row["case_id"] + "_" + view and view in ("display", "gray", "raw") and
                (view != "raw" or row["case_id"] in ("q02", "q03")) and
                row["source_path"] == case["source_path"] and row["source_sha256"] == case["source_sha"] and
                row["bbox_xywh"] == case["bbox_xywh"], "precise frozen view binding")
        data = pixels[row["case_id"]]
        require(row["orientation"] == data["orientation"] and row["size"] == data["size"], "source EXIF and crop shape")
        path = regular(row["path"])
        require(path.name == row["input_id"] + ".png" and path.is_relative_to(Path(plan["raw_root"]).absolute()), "fresh opaque input path")
        verify_png(path, data[view], row["size"], row["sha256"], row["rgb_sha256"])
        inputs[row["input_id"]] = row
    raw_results = load(ref["raw_results_path"])["results"]
    require(len(raw_results) == 12, "complete historical twelve-output index")
    history = {}
    tokenizer = Path(plan["resources"]["base"]["path"]) / "tokenizer.json"
    for arm in ARMS:
        for case_id in ("q02", "q03"):
            indexed = [row for row in raw_results if row["arm"] == arm and row["case_id"] == case_id]
            require(len(indexed) == 1, "one immutable historical raw generation")
            entry = indexed[0]
            directory = Path(parent["raw_root"]) / arm / "cases" / case_id
            require(digest(directory / "claim.json") == entry["claim_sha256"] and
                    digest(directory / "receipt.json") == entry["receipt_sha256"], "old raw claim/receipt SHA from audited index")
            claim, receipt = load(directory / "claim.json"), load(directory / "receipt.json")
            require(claim["prompt"] == old.PROMPT and claim["generation"] == GENERATION and
                    claim["crop_sha"] == by_case[case_id]["crop_sha"] and receipt["status"] == "PASS" and
                    receipt["raw_text"] == entry["raw_text"] and receipt["output_ids"] == entry["output_ids"], "unchanged old raw output")
            parsed = old.decode_output(tokenizer, receipt["returned_ids"], claim["input_ids"], receipt["output_ids"], receipt["raw_text"])
            history[(arm, case_id)] = {"raw_text": receipt["raw_text"], "output_ids": receipt["output_ids"],
                                      "strict_label": parsed, "receipt_sha256": entry["receipt_sha256"], "inputs": old.tensor_inputs(claim)}
    return labels, inputs, history


def call_matrix(arm):
    result = [("processor_load", "processor"), ("base_load", "base")]
    if arm == "helmet":
        result += [("adapter_load", "adapter"), ("activation", "activate")]
    result += [("generate", name) for name in GEN_IDS]
    result += [("score_forward", name + "_m%d" % mapping) for name in INPUT_IDS for mapping in (0, 1)]
    return result


def events(path, arm, pid, started, finished):
    rows = old.operation_lines(path)
    expected, order, stack, windows = call_matrix(arm), [], [], {}
    counts = {op: 0 for op in CALL_OPS}
    last = started
    for row in rows:
        op, edge, call_id = row["op"], row["edge"], row["call_id"]
        t = row["clock"]
        require(op in CALL_OPS and edge in ("begin", "end") and row["pid"] == pid and row["arm"] == arm and
                type(t) in (int, float) and math.isfinite(t) and last <= t <= finished, "native event identity/time")
        last = t
        if edge == "begin":
            counts[op] += 1
            if op == "native_forward":
                require(len(stack) == 1 and stack[-1][0] in TENSOR_OPS and call_id == stack[-1][1] and
                        row["parent_op"] == stack[-1][0] and row["inference_mode"] is True and row["grad_enabled"] is False,
                        "actual native forward parent / inference-mode evidence")
                windows[(stack[-1][0], call_id)]["native_forwards"] += 1
            else:
                require(not stack and len(order) < len(expected) and expected[len(order)] == (op, call_id), "complete fixed top-call sequence")
                order.append((op, call_id))
                windows[(op, call_id)] = {"started": t, "native_forwards": 0}
            stack.append((op, call_id))
        else:
            require(stack and stack[-1] == (op, call_id) and row["status"] == "PASS", "matched successful native call end")
            if op == "native_forward":
                require(len(stack) == 2 and row["parent_op"] == stack[0][0], "native forward end parent binding")
            stack.pop()
            if op != "native_forward":
                windows[(op, call_id)]["finished"] = t
    require(not stack and order == expected, "no missing / incomplete / extra calls")
    for (op, _), window in windows.items():
        if op == "score_forward":
            require(window["native_forwards"] == 1, "exactly one native explicit score forward")
        elif op not in TENSOR_OPS:
            require(window["native_forwards"] == 0, "no model forward during load/activation")
    require(counts["generate"] == 2 and counts["score_forward"] == 28 and counts["native_forward"] <= 92,
            "per-arm generation/scoring/native-forward bounds")
    return counts, windows


def actual_inputs(claim, row, prompt):
    require(claim["input_id"] == row["input_id"] and claim["input_sha256"] == row["sha256"] and
            claim["prompt"] == prompt and prompt in claim["rendered_prompt"] and
            not any(name in claim["rendered_prompt"] for name in INPUT_IDS), "actual opaque image/frozen prompt")
    identity = old.tensor_inputs(claim)
    ids, mask = claim["input_ids"], claim["attention_mask"]
    require(isinstance(mask, list) and len(mask) == len(ids) and mask and
            all(type(v) is int and v == 1 for v in mask) and claim["selected_position"] == len(ids) - 1,
            "first next-token score must select the last valid unpadded token")
    fingerprint = claim["tensors"]["attention_mask"]
    require(fingerprint["dtype"] == "torch.int64" and fingerprint["shape"] == [1, len(mask)] and
            fingerprint["sha256"] == hashlib.sha256(struct.pack("<%dq" % len(mask), *mask)).hexdigest(),
            "actual all-one attention-mask tensor binding")
    return identity


def worker_plan(plan, arm, inputs):
    ref = plan["worker_plans"][arm]
    require(digest(ref["path"]) == ref["sha256"], "worker plan SHA")
    spec = load(ref["path"])
    require(set(spec) == {"arm", "runtime", "resources", "source_commit", "source_sha", "caps", "inputs",
                          "generation_prompt", "generation", "prompts", "candidate_ids"}, "exact label-free worker plan")
    require(spec["arm"] == arm and all(spec[key] == plan[key] for key in
            ("runtime", "resources", "source_commit", "source_sha", "caps", "generation_prompt", "generation", "prompts", "candidate_ids")),
            "worker/main frozen protocol")
    rows = [{key: inputs[name][key] for key in ("input_id", "case_id", "view", "path", "sha256", "rgb_sha256", "size")} for name in INPUT_IDS]
    require(spec["inputs"] == rows, "no GT, original image name/path or native ROI in worker inputs")
    return ref["sha256"]


def audit_worker(root, arm, plan, inputs, originals, vocabulary):
    out = regular(root / arm, directory=True)
    start, complete = load(out / "start.json"), load(out / "complete.json")
    process, intent = load(root / (arm + "_process.json")), load(root / (arm + "_process_intent.json"))
    ref_sha = worker_plan(plan, arm, inputs)
    command = [plan["runtime"]["python"], "-u", str(old.SOURCE_ROOT / "scripts/run_helmet_orientation_choice.py"),
               "--worker_plan", plan["worker_plans"][arm]["path"], "--worker_sha", ref_sha, "--out", str(out)]
    require(intent["command"] == command and intent["started"] == process["started"] and
            type(intent["seconds"]) in (int, float) and 0 < intent["seconds"] <= CAPS["worker_seconds"] and
            process["returncode"] == 0 and process["exception"] is None and process["terminated_group"] is False,
            "one successful supervised worker launch / no retry")
    pid = start["pid"]
    require(type(pid) is int and pid > 0 and pid == process["pid"] == complete["pid"] and
            start["arm"] == arm and start["worker_plan_sha256"] == ref_sha and
            process["started"] <= start["started_clock"] <= complete["finished_clock"] <= process["finished"] and
            process["finished"] - process["started"] <= CAPS["worker_seconds"] and
            complete["versions"] == plan["runtime"]["versions"] and complete["module_origins"] == plan["runtime"]["module_origins"] and
            complete["threads"] == [1, 1] and not (out / "failure.json").exists(), "actual fresh worker runtime / interval")
    counts, windows = events(out / "operations.jsonl", arm, pid, start["started_clock"], complete["finished_clock"])
    require(set(complete["counts"]) <= set(CALL_OPS) and all(complete["counts"].get(op, 0) == counts[op] for op in CALL_OPS),
            "native event counts independently match final receipt")
    identity = load(out / "identity.json")
    require(identity["base_weight_sha256"] == plan["resources"]["base"]["weight_sha256"] and
            identity["model_training"] is False and identity["generation_kwargs"] == GENERATION,
            "actual eval base / unchanged generation kwargs")
    config = dict(identity["generation_config"])
    require(config["bos_token_id"] == 0 and config["eos_token_id"] == 49279 and config["pad_token_id"] == 2,
            "native generation BOS/EOS/PAD unchanged")
    config.update(identity["generation_kwargs"])
    if arm == "base":
        require(identity["is_peft"] is False and not (out / "parameters.json").exists(), "adapter-free base arm")
    else:
        require(identity["is_peft"] is True and identity["adapter_weight_sha256"] == plan["resources"]["adapter"]["weight_sha256"] and
                identity["active"] == ["helmet"] and identity["available"] == ["helmet"] and identity["enabled"] is True,
                "actual one enabled Helmet adapter")
        old.loaded_parameters(load(out / "parameters.json"), originals)
    verification = load(out / "resources.json")
    require(set(verification) == set(plan["resources"]), "worker checked all frozen resource groups")
    for name, spec in plan["resources"].items():
        require(set(verification[name]) == set(spec["files"]), "worker resource-file inventory")
        for relative, record in spec["files"].items():
            require(verification[name][relative]["bytes"] == record["bytes"] and
                    verification[name][relative]["sha256"] == record["sha256"], "worker resource verification identity")
    require(sorted(path.name for path in regular(out / "generated", directory=True).iterdir()) == sorted(GEN_IDS), "only two new display generations per arm")
    score_names = [name + "_m%d" % mapping for name in INPUT_IDS for mapping in (0, 1)]
    require(sorted(path.name for path in regular(out / "scores", directory=True).iterdir()) == sorted(score_names), "complete fixed 28-score matrix per arm")
    tokenizer = Path(plan["resources"]["base"]["path"]) / "tokenizer.json"
    generated, scores, fingerprints = [], [], {}
    for input_id in GEN_IDS:
        directory = out / "generated" / input_id
        claim, receipt = load(directory / "claim.json"), load(directory / "receipt.json")
        fingerprints[("generate", input_id)] = actual_inputs(claim, inputs[input_id], old.PROMPT)
        window = windows[("generate", input_id)]
        require(start["started_clock"] <= claim["started_clock"] <= window["started"] <= window["finished"] <= receipt["finished_clock"] <= complete["finished_clock"], "full raw generation interval containment")
        parsed = old.decode_output(tokenizer, receipt["returned_ids"], claim["input_ids"], receipt["output_ids"], receipt["raw_text"])
        require(receipt["native_forwards"] == window["native_forwards"] == len(receipt["output_ids"]), "one real autoregressive forward per new output token")
        generated.append({"arm": arm, "case_id": inputs[input_id]["case_id"], "input_id": input_id,
                          "raw_text": receipt["raw_text"], "output_ids": receipt["output_ids"], "strict_label": parsed,
                          "native_forwards": receipt["native_forwards"], "claim_sha256": digest(directory / "claim.json"),
                          "receipt_sha256": digest(directory / "receipt.json")})
    for input_id in INPUT_IDS:
        for mapping, prompt in enumerate(PROMPTS):
            name = input_id + "_m%d" % mapping
            directory = out / "scores" / name
            claim, receipt = load(directory / "claim.json"), load(directory / "receipt.json")
            fingerprints[("score_forward", name)] = actual_inputs(claim, inputs[input_id], prompt)
            window = windows[("score_forward", name)]
            require(claim["mapping"] == mapping and receipt["native_forwards"] == window["native_forwards"] == 1 and
                    receipt["use_cache"] is False and receipt["selected_position"] == claim["selected_position"] and
                    start["started_clock"] <= claim["started_clock"] <= window["started"] <= window["finished"] <= receipt["finished_clock"] <= complete["finished_clock"],
                    "one declared first-next-token scoring forward at valid position")
            probabilities = logit_file(directory / "logits.npy", receipt, vocabulary)
            p_a = probabilities["p_A_conditional"]
            scores.append({"arm": arm, "input_id": input_id, "case_id": inputs[input_id]["case_id"], "view": inputs[input_id]["view"],
                           "mapping": mapping, **probabilities, "p_Helmet_mapping": p_a if mapping == 0 else 1.0 - p_a,
                           "logits_sha256": receipt["logits_sha256"], "claim_sha256": digest(directory / "claim.json"),
                           "receipt_sha256": digest(directory / "receipt.json")})
    require(counts["native_forward"] == 28 + sum(row["native_forwards"] for row in generated), "all native forwards assigned exactly once")
    return {"pid": pid, "started": process["started"], "finished": process["finished"], "counts": counts,
            "fingerprints": fingerprints, "generation_config": config, "generated": generated, "scores": scores}


def failure_bill(root):
    arms, aggregate, upper = [], {op: 0 for op in CALL_OPS}, {op: 0 for op in CALL_OPS}
    for arm in ARMS:
        intent_state, intent = old.optional_metadata(root / (arm + "_process_intent.json"))
        process_state, process = old.optional_metadata(root / (arm + "_process.json"))
        start_state, start = old.optional_metadata(root / arm / "start.json")
        row = {"arm": arm, "launch_intent": intent_state, "process_receipt": process_state, "worker_start": start_state}
        pids = sorted({obj["pid"] for obj in (process, start) if isinstance(obj, dict) and type(obj.get("pid")) is int and obj["pid"] > 0})
        row.update(launch_attempt_observed=int(intent is not None or process is not None or start is not None),
                   launch_attempt_upper=1, PID_created=True if pids else False if process is not None and process.get("pid", "missing") is None else None,
                   observed_PIDs=pids, worker_start_observed=start is not None)
        observed = {op: 0 for op in CALL_OPS}
        errors, begins = [], []
        path = root / arm / "operations.jsonl"
        if path.exists() or path.is_symlink():
            try:
                row["operations_sha256"] = digest(path)
                for number, line in enumerate(regular(path).read_text(encoding="utf-8").splitlines()):
                    try:
                        event = old.json_object(line)
                        require(isinstance(event, dict), "event object required")
                        if event.get("edge") == "begin":
                            if event.get("op") in observed:
                                observed[event["op"]] += 1
                            if len(begins) < 64:
                                begins.append({key: event.get(key) for key in ("op", "call_id", "parent_op", "clock", "pid", "arm")})
                    except (ValueError, TypeError) as error:
                        errors.append({"line": number + 1, "error": str(error)})
            except (OSError, ValueError, TypeError) as error:
                errors.append({"error": str(error)})
        else:
            errors.append({"error": "missing ledger; no zero-call certification"})
        limits = {"processor_load": 1, "base_load": 1, "adapter_load": int(arm == "helmet"), "activation": int(arm == "helmet"),
                  "generate": 2, "score_forward": 28, "native_forward": 92}
        row.update(observed_begin_counts=observed, unknown_inclusive_upper={op: max(observed[op], limits[op]) for op in CALL_OPS},
                   begins_preview=begins, ledger_errors=errors, counts_are_lower_bounds=True)
        for op in CALL_OPS:
            aggregate[op] += observed[op]
            upper[op] += row["unknown_inclusive_upper"][op]
        arms.append(row)
    consumer_state, _ = old.optional_metadata(root / "consumer_process_intent.json")
    return {"arms": arms, "observed_begin_counts": aggregate, "unknown_inclusive_upper": upper,
            "consumer_launch_intent": consumer_state, "consumer_is_not_a_model_process": True,
            "missing_or_partial_ledger_does_not_mean_zero": True, "old_costs_and_all_raw_outputs_retained": True}


def summarise(workers, labels, history):
    mapping_rows = [row for arm in ARMS for row in workers[arm]["scores"]]
    means, generations, interventions, accuracy = [], [], [], {}
    for arm in ARMS:
        mapped = {(row["input_id"], row["mapping"]): row for row in workers[arm]["scores"]}
        condition = {}
        for input_id in INPUT_IDS:
            first, second = mapped[(input_id, 0)], mapped[(input_id, 1)]
            probability, prediction = mapped_mean(first["p_A_conditional"], second["p_A_conditional"])
            per_map = ["HELMET" if row["p_Helmet_mapping"] > .5 else "NO_HELMET" if row["p_Helmet_mapping"] < .5 else "ABSTAIN"
                       for row in (first, second)]
            row = {"arm": arm, "input_id": input_id, "case_id": first["case_id"], "view": first["view"],
                   "p_Helmet": probability, "prediction": prediction, "abstain": prediction == "ABSTAIN",
                   "mapping_disagreement": per_map[0] != per_map[1],
                   "map0_Helmet_probability": first["p_Helmet_mapping"], "map1_Helmet_probability": second["p_Helmet_mapping"]}
            if first["view"] == "display":
                row.update(groundtruth=labels[first["case_id"]], correct=prediction == labels[first["case_id"]])
            else:
                row.update(groundtruth=None, correct=None)
            means.append(row)
            condition[input_id] = row
        display = [row for row in means if row["arm"] == arm and row["view"] == "display"]
        correct = sum(row["correct"] for row in display)
        accuracy[arm] = {"correct": correct, "total_consumed_cases": 6, "descriptive_accuracy": correct / 6,
                         "abstentions_counted_incorrect": sum(row["abstain"] for row in display),
                         "all_six_conditionally_correct": correct == 6,
                         "scope": "forced two-token diagnostic on consumed cases; no independent validation"}
        for case_id in CASES:
            row = {"arm": arm, "case_id": case_id,
                   "display_minus_gray_Helmet_probability": condition[case_id + "_display"]["p_Helmet"] - condition[case_id + "_gray"]["p_Helmet"]}
            if case_id in ("q02", "q03"):
                row["display_minus_raw_Helmet_probability"] = condition[case_id + "_display"]["p_Helmet"] - condition[case_id + "_raw"]["p_Helmet"]
            interventions.append(row)
        for row in workers[arm]["generated"]:
            legacy = history[(arm, row["case_id"])]
            generations.append({**row, "legacy_raw_text": legacy["raw_text"], "legacy_raw_output_ids": legacy["output_ids"],
                                "legacy_raw_strict_label": legacy["strict_label"], "legacy_receipt_sha256": legacy["receipt_sha256"],
                                "new_display_strict_correct": row["strict_label"] == labels[row["case_id"]],
                                "suffix_changed": row["output_ids"] != legacy["output_ids"],
                                "note": "legacy raw q02/q03 crop is not a head-ROI quality reference"})
    return {"mapping_scores": mapping_rows, "condition_scores": means, "display_summary": accuracy,
            "paired_input_interventions": interventions, "orientation_generation_pairs": generations,
            "mean_formula": "pHelmet=0.5+0.5*(pA_map0-pA_map1)",
            "candidate_probability_scope": "A/B conditional score, not calibrated; full-vocabulary candidate mass separately retained",
            "gray_and_raw_are_controls_not_independent_labelled_examples": True}


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def audit(plan_path, expected_plan_sha, root, output):
    root, plan_path, output = regular(root, directory=True), regular(plan_path), Path(output).absolute()
    terminal, attempt = root / "terminal.json", root / "consumer_claim.json"
    require(not any(path.exists() or path.is_symlink() for path in (output, terminal, attempt, root / "consumer_failure.json")),
            "single-use orientation consumer; no rerun")
    regular(output.parent, directory=True)
    started = clock()
    create(attempt, {"started_clock": started, "consumer_sha256": digest(__file__), "expected_plan_sha256": expected_plan_sha,
                     "single_use": True, "old_audit_reruns": 0, "model_calls": 0})
    result = {"reviewed_at": datetime.now(timezone.utc).isoformat(), "literature_cutoff": "No new literature search",
              "target_venue": "CSCWD2027; TMC policy internal only", "artifact_run_id": root.name,
              "policy_version": "tmc_review_policy_v3_20260621", "git_commit": None, "evidence_level": "E0_UNAVAILABLE",
              "audit_success": False, "matrix_complete": False, "consumed_cases": 6, "new_independent_cases": 0,
              "old_Q_costs_and_negative_results_unchanged": True, "new_protocol_is_not_pure_format_causal_effect": True,
              "conditional_scores_are_not_calibrated": True, "training_independence": "UNVERIFIED",
              "generalisation": "UNVERIFIED", "hidden": 0, "HTTP": 0, "training": 0,
              "canonical_promotion": False, "paper_goal_complete": False, "requires_matching_terminal": True}
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("orientation consumer deadline")))
    previous_timer, claim = signal.getitimer(signal.ITIMER_REAL), None
    try:
        signal.setitimer(signal.ITIMER_REAL, CAPS["consumer_seconds"] - 2)
        require(digest(plan_path) == digest(root / "plan.json") == expected_plan_sha, "external/copied plan SHA")
        plan, claim = load(plan_path), load(root / "claim.json")
        result["git_commit"] = plan["source_commit"]
        require(plan["caps"] == CAPS and plan["prompts"] == PROMPTS and plan["candidate_ids"] == CANDIDATES and
                Path(plan["raw_root"]).absolute() == root and Path(plan["audit_output"]).absolute() == output and
                plan["generation"] == GENERATION and plan["generation_prompt"] == old.PROMPT,
                "fixed prospective orientation/choice task")
        require(claim["plan_sha256"] == expected_plan_sha and claim["boot_uuid"] == old.boot(), "new scope plan/boot binding")
        grant = load(root / "grant.json")
        require(digest(root / "grant.json") == claim["grant_sha256"] and grant["plan_sha256"] == expected_plan_sha and
                grant["allow_execution"] is True, "new fixed scope authorization")
        elapsed = clock() - claim["started_clock"]
        require(0 <= elapsed < CAPS["wall_seconds"] - 2, "new 2700s absolute deadline")
        signal.setitimer(signal.ITIMER_REAL, min(CAPS["consumer_seconds"] - 2, CAPS["wall_seconds"] - elapsed - 2))
        dependencies = {"scripts/audit_helmet_orientation_choice.py", "scripts/run_helmet_orientation_choice.py",
                        "scripts/audit_helmet_quality_pilot.py", "scripts/run_helmet_quality_pilot.py"}
        require(dependencies.issubset(plan["source_sha"]) and plan["source_sha"]["scripts/audit_helmet_orientation_choice.py"] == digest(__file__), "all new/shared source SHA bindings")
        old.source_identity(plan)
        old.runtime_identity(plan["runtime"])
        numeric = plan["numeric_addendum"]
        require(digest(numeric["path"]) == numeric["sha256"], "prospective numeric addendum SHA")
        addendum = load(numeric["path"])
        contract = load(addendum["contract_path"])
        require(plan["implementation_contract"] == {"path": addendum["contract_path"], "sha256": addendum["contract_sha256"]} and
                digest(addendum["contract_path"]) == addendum["contract_sha256"] and contract["caps"] == CAPS and
                contract["phase_B"]["prompts"] == PROMPTS and contract["phase_B"]["candidate_ids"] == CANDIDATES and
                addendum["formula"] == "pHelmet=0.5+0.5*(pA_map0-pA_map1)" and addendum["precision"] == "float64" and
                addendum["new_model_results_seen"] == 0 and addendum["rule_may_change_after_results"] is False,
                "numeric rule frozen before new model outcomes")
        stored, logs, outer = old.tree_bytes(plan["storage_roots"])
        require(plan["storage_roots"] == [str(root.parent)] and output.parent == root.parent and
                any(plan_path.is_relative_to(base) for base in outer) and stored + 1048576 <= CAPS["artifact_bytes"] and
                logs + 1048576 <= CAPS["log_bytes"], "charged full new tree / final metadata reserve")
        originals = old.resource_identity(plan)
        labels, inputs, history = parent_inputs(plan)
        from tokenizers import Tokenizer
        tokenizer_path = Path(plan["resources"]["base"]["path"]) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(regular(tokenizer_path)))
        require([tokenizer.encode(value, add_special_tokens=False).ids for value in (" A", " B")] == [[330], [389]], "frozen exact single-token candidates")
        config = load(Path(plan["resources"]["base"]["path"]) / "config.json")
        vocabulary = config.get("text_config", config)["vocab_size"]
        require(type(vocabulary) is int and vocabulary > max(CANDIDATES), "frozen model vocabulary size")
        workers = {arm: audit_worker(root, arm, plan, inputs, originals, vocabulary) for arm in ARMS}
        require(workers["base"]["pid"] != workers["helmet"]["pid"] and
                claim["started_clock"] <= workers["base"]["started"] <= workers["base"]["finished"] <= workers["helmet"]["started"] <=
                workers["helmet"]["finished"] <= claim["started_clock"] + CAPS["producer_seconds"], "two sequential fresh model processes within producer scope")
        require(workers["base"]["fingerprints"] == workers["helmet"]["fingerprints"] and
                workers["base"]["generation_config"] == workers["helmet"]["generation_config"], "identical actual image tensors/prompt/options across arms")
        counts = {op: sum(workers[arm]["counts"][op] for arm in ARMS) for op in CALL_OPS}
        require(all(counts[op] == CAPS[op] for op in CALL_OPS if op != "native_forward") and counts["native_forward"] <= CAPS["native_forward"], "exact new call matrix and native forward upper bound")
        # Image controls preserve all token/layout/mask inputs; only pixel content changes.
        for arm in ARMS:
            fingerprints = workers[arm]["fingerprints"]
            for case_id in CASES:
                for mapping in (0, 1):
                    displayed = fingerprints[("score_forward", case_id + "_display_m%d" % mapping)]
                    alternatives = [case_id + "_gray"] + ([case_id + "_raw"] if case_id in ("q02", "q03") else [])
                    for alternative in alternatives:
                        other = fingerprints[("score_forward", alternative + "_m%d" % mapping)]
                        require(displayed["input_ids"] == other["input_ids"] and displayed["rendered_prompt"] == other["rendered_prompt"] and
                                set(displayed["tensors"]) == set(other["tensors"]) and
                                all(displayed["tensors"][name] == other["tensors"][name] for name in displayed["tensors"] if name != "pixel_values") and
                                displayed["tensors"]["pixel_values"]["shape"] == other["tensors"]["pixel_values"]["shape"], "only image content may change across counterfactual views")
            for input_id in GEN_IDS:
                legacy = history[(arm, inputs[input_id]["case_id"])]["inputs"]
                fresh = fingerprints[("generate", input_id)]
                require(fresh["input_ids"] == legacy["input_ids"] and fresh["rendered_prompt"] == legacy["rendered_prompt"] and
                        set(fresh["tensors"]) == set(legacy["tensors"]) and
                        all(fresh["tensors"][name] == legacy["tensors"][name] for name in fresh["tensors"] if name != "pixel_values") and
                        fresh["tensors"]["pixel_values"]["shape"] == legacy["tensors"]["pixel_values"]["shape"], "old raw/new display generation differs only by pixels")
        findings = summarise(workers, labels, history)
        exports = {output.parent / "choice_mapping_scores.csv": csv_bytes(findings["mapping_scores"]),
                   output.parent / "choice_condition_scores.csv": csv_bytes(findings["condition_scores"]),
                   output.parent / "orientation_choice_results.json": encoded(findings)}
        for path, data in exports.items():
            create(path, data)
        final_bytes, final_logs, _ = old.tree_bytes(plan["storage_roots"])
        require(final_bytes + 1048576 <= CAPS["artifact_bytes"] and final_logs + 1048576 <= CAPS["log_bytes"] and
                old.boot() == claim["boot_uuid"] and clock() - claim["started_clock"] < CAPS["wall_seconds"] and
                clock() - started < CAPS["consumer_seconds"] - 1, "final bytes/logs/deadline/boot")
        result.update(status="AUDITED_CONSUMED_HELMET_ORIENTATION_CHOICE", audit_success=True, matrix_complete=True,
                      evidence_level="E2_ARTIFACT_AUDITED", operation_counts=counts, model_processes=2,
                      legacy_generations_reused=4, summary=findings["display_summary"], plan_sha256=expected_plan_sha,
                      parent_audit_sha256=plan["parent_reference"]["audit_sha"], numeric_addendum_sha256=numeric["sha256"],
                      mean_formula=findings["mean_formula"], output_files={str(path): hashlib.sha256(data).hexdigest() for path, data in exports.items()},
                      artifact_bytes_before_audit=final_bytes, log_bytes_before_audit=final_logs)
    except BaseException as error:
        result.update(status="ORIENTATION_CHOICE_UNVERIFIED_FAILURE_PRESERVED", audit_success=False, matrix_complete=False,
                      error_type=type(error).__name__, error=str(error), failure_bill=failure_bill(root))
        create(root / "consumer_failure.json", {"audit_success": False, "error_type": type(error).__name__, "error": str(error),
                                                "failure_bill": result["failure_bill"], "consumer_must_not_rerun": True})
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
        create(terminal, {"audit": str(output), "sha256": digest(output), "audit_success": result["audit_success"],
                          "matrix_complete": result["matrix_complete"], "finished_clock": clock(), "consumer_must_not_rerun": True})
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
