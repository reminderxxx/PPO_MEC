"""Independent offline audit of one fixed six-ROI paired Helmet quality pilot.

No model, processor, generation, network or producer scoring function is used.
All correctness claims are restricted to this finite descriptive fixture.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import signal
import struct
import subprocess
import sys
import time


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base", "helmet")
CASES = tuple("q%02d" % n for n in range(6))
PROMPT = "Look at this cropped head region. Is a motorcycle helmet worn? Reply with exactly one label: HELMET or NO_HELMET. No explanation."
PARSER = "strip surrounding whitespace only; exact HELMET or NO_HELMET; every other output format is wrong"
CAPS = {"model_processes": 2, "processor_load": 2, "base_load": 2, "adapter_load": 1,
        "activation": 1, "generate": 12, "new_tokens_per_generate": 32,
        "wall_seconds": 1800, "producer_seconds": 1200, "child_seconds": 600,
        "consumer_seconds": 600, "artifact_bytes": 134217728, "log_bytes": 16777216,
        "http": 0, "training": 0, "fitting": 0, "environment_episodes": 0}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def boot():
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"], text=True).strip()


def regular(path, directory=False):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in [path, *path.parents]), "input/reference symlink")
    require(path.is_dir() if directory else path.is_file(), "missing regular input/reference")
    return path


def digest(path):
    h = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def json_object(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(AuditError("nonfinite JSON")))


def load(path):
    return json_object(regular(path).read_text(encoding="utf-8"))


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def create(path, obj):
    with Path(path).open("xb") as stream:
        stream.write(obj if isinstance(obj, bytes) else encoded(obj))


def parse_label(raw_text):
    require(isinstance(raw_text, str), "raw decoded text must be a string")
    stripped = raw_text.strip()
    return stripped if stripped in ("HELMET", "NO_HELMET") else None


def native_bbox(values, image_size):
    require(isinstance(values, list) and len(values) == 4 and
            all(type(v) in (int, float) and math.isfinite(v) and int(v) == v for v in values),
            "native bbox must be integral, without rounding or clipping")
    x, y, width, height = map(int, values)
    require(x >= 0 and y >= 0 and width >= 32 and height >= 32 and
            x + width <= image_size[0] and y + height <= image_size[1], "native bbox outside original")
    return x, y, width, height


def crop_identity(source, source_sha, bbox, opaque_png):
    from PIL import Image
    require(digest(source) == source_sha, "source image SHA changed")
    with Image.open(regular(source)) as image:
        image.verify()
    with Image.open(source) as image:
        source_size = image.size
        x, y, width, height = native_bbox(bbox, source_size)
        # Pillow does not apply EXIF orientation here. No resize or crop repair.
        expected = image.convert("RGB").crop((x, y, x + width, y + height))
    with Image.open(regular(opaque_png)) as actual:
        require(actual.format == "PNG" and actual.mode == "RGB" and actual.size == (width, height),
                "opaque crop must be exact RGB PNG")
        actual.load()
        raw = actual.tobytes()
    require(raw == expected.tobytes(), "opaque crop pixels differ from native RGB ROI")
    return {"width": width, "height": height, "rgb_sha256": hashlib.sha256(raw).hexdigest(),
            "png_sha256": digest(opaque_png), "source_size": list(source_size)}


def original_adapter_tensors(path):
    """Read frozen safetensors bytes directly; never instantiate a model."""
    import numpy as np
    raw = regular(path).read_bytes()
    require(len(raw) >= 8, "safetensors header missing")
    length = struct.unpack("<Q", raw[:8])[0]
    require(2 <= length <= len(raw) - 8, "safetensors header length")
    header = json_object(raw[8:8 + length].decode("utf-8"))
    payload = memoryview(raw)[8 + length:]
    result, spans = {}, []
    for key, item in header.items():
        if key == "__metadata__":
            continue
        shape = item["shape"]
        require(isinstance(shape, list) and shape and all(type(v) is int and v > 0 for v in shape), "adapter native tensor shape")
        begin, end = item["data_offsets"]
        require(type(begin) is int and type(end) is int and 0 <= begin < end <= len(payload), "adapter tensor byte offsets")
        dtype = item["dtype"]
        require(dtype in ("F32", "F16", "BF16"), "adapter floating dtype")
        count = math.prod(shape)
        require(end - begin == count * (4 if dtype == "F32" else 2), "adapter tensor native byte count")
        block = payload[begin:end]
        if dtype == "BF16":
            data = (np.frombuffer(block, dtype="<u2").astype("<u4") << 16).view("<f4")
        else:
            data = np.frombuffer(block, dtype="<f4" if dtype == "F32" else "<f2").astype("<f4", copy=False)
        require(np.isfinite(data).all(), "nonfinite original adapter weight")
        result[key] = {"shape": shape, "dtype": "torch.float32",
                       "sha256": hashlib.sha256(data.tobytes(order="C")).hexdigest()}
        spans.append((begin, end))
    ordered = sorted(spans)
    require(ordered and ordered[0][0] == 0 and ordered[-1][1] == len(payload) and
            all(a[1] == b[0] for a, b in zip(ordered, ordered[1:])), "complete, nonoverlapping adapter native tensors")
    return result


def expected_loaded_name(key, adapter_name="helmet"):
    matches = [marker for marker in (".lora_A.", ".lora_B.") if marker in key]
    require(len(matches) == 1 and key.count(matches[0]) == 1, "fixed non-DoRA LoRA parameter mapping")
    return key.replace(matches[0], matches[0] + adapter_name + ".")


def decode_output(tokenizer_path, returned_ids, input_ids, output_ids, raw_text):
    require(all(isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids)
                for ids in (returned_ids, input_ids, output_ids)) and input_ids and
            0 < len(output_ids) <= 32 and returned_ids[:len(input_ids)] == input_ids and
            returned_ids[len(input_ids):] == output_ids, "complete returned sequence / generated suffix IDs")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(regular(tokenizer_path)))
    require(all(v < tokenizer.get_vocab_size(with_added_tokens=True) for v in output_ids), "output token outside frozen vocabulary")
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    require(decoded == raw_text, "raw suffix text differs from independent local tokenizer decode")
    return parse_label(decoded)


def loaded_parameters(record, original):
    require(record["active_adapter"] == "helmet" and record["adapters_enabled"] is True,
            "actual adapter activation identity")
    rows = record["parameters"]
    require(isinstance(rows, list) and len(rows) == len(original), "complete loaded adapter parameter set")
    seen = set()
    for row in rows:
        key = row["safetensors_key"]
        require(key in original and key not in seen and row["loaded_name"] == expected_loaded_name(key),
                "native-to-loaded adapter parameter mapping")
        expected = original[key]
        require(row["shape"] == expected["shape"] and row["dtype"] in ("torch.float32", "float32") and
                row["sha256"] == expected["sha256"], "loaded adapter values differ from frozen native weights")
        seen.add(key)
    require(seen == set(original), "missing original adapter tensor")


def tree_bytes(roots):
    paths = [regular(path, directory=True) for path in roots]
    require(paths and len(paths) == len(set(paths)), "unique artifact storage roots")
    outer = [p for p in paths if not any(p != q and p.is_relative_to(q) for q in paths)]
    total = logs = 0
    for path in outer:
        for item in path.rglob("*"):
            if item.is_symlink():
                amount = item.lstat().st_size
            elif item.is_file():
                amount = item.stat().st_size
            else:
                continue
            total += amount
            if item.suffix.lower() in (".json", ".jsonl", ".log", ".csv", ".txt"):
                logs += amount
    return total, logs, outer


def source_identity(plan):
    require(re.fullmatch(r"[0-9a-f]{40}", plan["source_commit"]), "source commit format")
    current = subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(current == plan["source_commit"] and "scripts/audit_helmet_quality_pilot.py" in plan["source_sha"], "frozen source commit / consumer SHA")
    for name, expected in plan["source_sha"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and
                digest(SOURCE_ROOT / relative) == expected, "frozen source file changed")


def resource_identity(plan):
    resources = plan["resources"]
    require(set(resources) == {"base", "adapter"}, "only frozen base and Helmet adapter resources")
    reference = plan["resource_reference"]
    require(digest(reference["path"]) == reference["sha256"] ==
            "d517a8b2d305974badd6632d36a7dae4b8ab206eee6f23d1916b2e5871b72332", "frozen original T resource plan SHA")
    original_plan = load(reference["path"])
    original_resources = original_plan["resources"]
    old_runtime = original_plan["old_runtime_identity"]
    old_versions = original_plan["runtime"]["versions"]
    require(plan["runtime"]["python"] == old_runtime["python_executable"] and
            plan["runtime"]["module_origins"] == old_runtime["module_origins"], "original T Python/module-origin mapping")
    require(set(plan["runtime"]["versions"]) == set(old_versions) | {"numpy", "Pillow", "tokenizers"} and
            all(plan["runtime"]["versions"][key] == value for key, value in old_versions.items()),
            "original T library versions; only three newly frozen image/tokenizer packages")
    for current_name, old_name in (("base", "base"), ("adapter", "B")):
        old = original_resources[old_name]
        require(isinstance(old["files"], list), "original T resource file list")
        files = {}
        for item in old["files"]:
            require(set(item) == {"name", "bytes", "sha256"} and item["name"] not in files,
                    "original T unique resource file records")
            files[item["name"]] = {"bytes": item["bytes"], "sha256": item["sha256"]}
        projected = {key: value for key, value in old.items() if key != "files"}
        projected["files"] = files
        if current_name == "adapter":
            projected["name"] = "helmet"
        require(resources[current_name] == projected, "complete resource differs from original T base/B")
    fixed = {"base": (1015025832, "d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2"),
             "adapter": (9641944, "e899c9c13b53bf80bcfc86048f5678a727e8c1838dc824156b521d5cdaafdd2d")}
    for name, resource in resources.items():
        basepath = regular(resource["path"], directory=True)
        files = resource["files"]
        require(isinstance(files, dict) and files and resource["weight_file"] in files and
                (resource["weight_bytes"], resource["weight_sha256"]) == fixed[name], "frozen original resource weights")
        for relative, record in files.items():
            rel = Path(relative)
            require(not rel.is_absolute() and ".." not in rel.parts, "resource file escape")
            path = regular(basepath / rel)
            require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], "resource file size/SHA")
        require(files[resource["weight_file"]] == {"bytes": resource["weight_bytes"], "sha256": resource["weight_sha256"]}, "resource native weight mapping")
    require(resources["adapter"]["name"] == "helmet", "one fixed adapter name")
    adapter_config = load(Path(resources["adapter"]["path"]) / "adapter_config.json")
    require(adapter_config["r"] == 4 and adapter_config.get("use_dora") is False and
            adapter_config.get("task_type") is None, "original non-DoRA Helmet config unchanged")
    return original_adapter_tensors(Path(resources["adapter"]["path"]) / resources["adapter"]["weight_file"])


def delivery_inputs(plan):
    ref = plan["input_reference"]
    for field in ("audit", "terminal", "groundtruth", "csv"):
        require(digest(ref[field + "_path"]) == ref[field + "_sha"], "accepted input reference SHA: " + field)
    prior, terminal, gt = load(ref["audit_path"]), load(ref["terminal_path"]), load(ref["groundtruth_path"])
    require(prior["input_qualified"] is True and prior["audit_success"] is True and prior["images"] == 6 and
            prior["qualified_scope"] == "Helmet6 only" and terminal["input_qualified"] is True and
            terminal["sha256"] == ref["audit_sha"] and Path(terminal["audit"]).resolve() == Path(ref["audit_path"]).resolve(),
            "previous delivered input audit/terminal pair")
    require(prior["output_files"][str(Path(ref["groundtruth_path"]))] == ref["groundtruth_sha"] and
            prior["output_files"][str(Path(ref["csv_path"]))] == ref["csv_sha"], "previous delivered CSV/GT output identity")
    samples = gt["samples"]
    require(len(samples) == len(plan["cases"]) == 6 and
            tuple(item["case_id"] for item in plan["cases"]) == CASES and
            set(plan["consumer_groundtruth"]) == set(CASES), "exact six private GT/cases")
    with regular(ref["csv_path"]).open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 6 and [x["sample_id"] for x in csv_rows] == [x["sample_id"] for x in samples], "accepted CSV/GT sample order")
    labels, crop_records = {}, {}
    for case, saved, csv_row in zip(plan["cases"], samples, csv_rows):
        case_id = case["case_id"]
        require(case["source_sample_id"] == saved["sample_id"] and
                str(Path(case["source_path"])) == saved["input_path"] and case["source_sha"] == saved["image_sha256"], "fixed source image binding")
        native = saved["native_annotation"]["bbox"]
        require(case["bbox_xywh"] == native and all(float(csv_row[k]) == saved[k] for k in ("x", "y", "width", "height")), "unmodified native ROI/CSV")
        label = {"Helmet": "HELMET", "NoHelmet": "NO_HELMET"}[saved["category_name"]]
        require(plan["consumer_groundtruth"][case_id] == label, "private GT must derive only from frozen native category")
        crop_path = regular(case["crop_path"])
        require(crop_path.name == case_id + ".png", "opaque crop filename")
        identity = crop_identity(case["source_path"], case["source_sha"], native, crop_path)
        require(identity["png_sha256"] == case["crop_sha"] and identity["rgb_sha256"] == case["crop_rgb_sha"] and
                identity["width"] == case["width"] and identity["height"] == case["height"] and
                identity["source_size"] == [saved["image_width"], saved["image_height"]], "frozen crop byte/pixel/shape identity")
        labels[case_id], crop_records[case_id] = label, case
    return labels, crop_records


def worker_plan_identity(plan, arm, crops):
    ref = plan["worker_plans"][arm]
    require(digest(ref["path"]) == ref["sha256"], "label-free worker plan SHA")
    worker = load(ref["path"])
    require(set(worker) == {"arm", "runtime", "generation", "prompt", "resources", "inputs", "case_order", "source_commit", "source_sha", "caps"}, "exact label-free worker plan fields")
    require(worker["arm"] == arm and worker["runtime"] == plan["runtime"] and worker["generation"] == plan["generation"] and
            worker["prompt"] == plan["prompt"] and worker["resources"] == plan["resources"] and
            worker["case_order"] == list(CASES) and worker["source_commit"] == plan["source_commit"] and
            worker["source_sha"] == plan["source_sha"] and worker["caps"] == CAPS, "worker/main plan binding")
    expected_inputs = [{key: crops[case_id][key] for key in
                        ("case_id", "crop_path", "crop_sha", "crop_rgb_sha", "width", "height")} for case_id in CASES]
    require(worker["inputs"] == expected_inputs, "worker receives only opaque crop inputs, no GT/source names")
    return ref["sha256"]


def operation_lines(path):
    rows = []
    for line in regular(path).read_text(encoding="utf-8").splitlines():
        require(line.strip(), "empty operation line")
        rows.append(json_object(line))
    return rows


def operation_charge(rows):
    counts = {name: 0 for name in ("processor_load", "base_load", "adapter_load", "activation", "generate")}
    for row in rows:
        name = "activation" if row.get("op") == "activate" else row.get("op")
        if row.get("edge") == "begin" and name in counts:
            counts[name] += 1
    return counts


def operations(rows, arm, pid, started, finished):
    expected = [("processor_load", None), ("base_load", None)]
    if arm == "helmet":
        expected += [("adapter_load", None), ("activation", None)]
    expected += [("generate", case_id) for case_id in CASES]
    require(len(rows) == len(expected) * 2, "complete native operation matrix / no unclosed call")
    generate_windows = {}
    last = started
    for index, (op, case_id) in enumerate(expected):
        begin, end = rows[index * 2:index * 2 + 2]
        for event, edge in ((begin, "begin"), (end, "end")):
            name = "activation" if event.get("op") == "activate" else event.get("op")
            require(name == op and event["edge"] == edge and event["pid"] == pid and event["arm"] == arm and
                    event.get("case_id") == case_id and type(event["clock"]) in (int, float) and
                    math.isfinite(event["clock"]) and last <= event["clock"] <= finished, "native operation identity/order/clock")
            last = event["clock"]
        require(end["status"] == "PASS", "failed native model operation")
        if op == "generate":
            generate_windows[case_id] = (begin["clock"], end["clock"])
    return generate_windows


def tensor_inputs(case):
    ids, tensors = case["input_ids"], case["tensors"]
    require(isinstance(ids, list) and ids and all(type(v) is int and v >= 0 for v in ids), "actual input IDs")
    require(isinstance(tensors, dict) and "input_ids" in tensors and "pixel_values" in tensors and
            len(tensors) >= 2, "real visual model input tensors required")
    for name, tensor in tensors.items():
        require(set(tensor) == {"dtype", "shape", "sha256"} and isinstance(name, str) and
                isinstance(tensor["shape"], list) and tensor["shape"] and
                all(type(v) is int and v > 0 for v in tensor["shape"]) and
                re.fullmatch(r"[0-9a-f]{64}", tensor["sha256"]) and isinstance(tensor["dtype"], str), "actual tensor fingerprint")
    require(tensors["pixel_values"]["dtype"] == "torch.float32" and math.prod(tensors["pixel_values"]["shape"]) > 0 and
            tensors["input_ids"]["dtype"] == "torch.int64" and tensors["input_ids"]["shape"] == [1, len(ids)] and
            tensors["input_ids"]["sha256"] == hashlib.sha256(b"".join(struct.pack("<q", v) for v in ids)).hexdigest(), "actual IDs/visual tensor byte binding")
    return {"input_ids": ids, "tensors": tensors, "rendered_prompt": case["rendered_prompt"]}


def runtime_identity(runtime):
    fixed = {"device": "cpu", "dtype": "float32", "intra_threads": 1, "inter_threads": 1,
             "batch_size": 1, "seed": 0, "local_files_only": True, "trust_remote_code": False,
             "do_sample": False, "num_beams": 1, "use_cache": True, "max_new_tokens": 32}
    require(all(runtime[k] == value for k, value in fixed.items()), "fixed CPU/greedy/runtime contract")
    require(Path(sys.executable).resolve() == Path(runtime["python"]).resolve(), "fixed existing Python interpreter")
    for name, expected in runtime["versions"].items():
        actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
        require(actual == expected, "local library version drift: " + name)
    for origin in runtime["module_origins"].values():
        regular(origin)


def audit_arm(root, arm, plan, crops, originals):
    directory = regular(root / arm, directory=True)
    start, terminal = load(directory / "worker_start.json"), load(directory / "worker_terminal.json")
    process = load(root / ("worker_%s_process.json" % arm))
    worker_sha = worker_plan_identity(plan, arm, crops)
    intent = load(root / ("worker_%s_control_intent.json" % arm))
    control = load(root / ("worker_%s_control.json" % arm))
    expected_command = [plan["runtime"]["python"], "-u", str(SOURCE_ROOT / "scripts/run_helmet_quality_pilot.py"),
                        "--worker_plan", plan["worker_plans"][arm]["path"], "--worker_plan_sha", worker_sha,
                        "--worker_output", str(directory)]
    require(intent["command"] == expected_command and intent["started"] == control["started"] == process["started"] and
            type(intent["seconds"]) in (int, float) and 0 < intent["seconds"] <= CAPS["child_seconds"] and
            all(control[key] == process[key] for key in ("pid", "returncode", "started", "finished", "terminated_group", "exception")) and
            control["exception"] is None and process["exception"] is None,
            "native worker launch intent/control/process cross-binding")
    pid = start["pid"]
    require(type(pid) is int and pid > 0 and start["arm"] == terminal["arm"] == arm and
            terminal["pid"] == process["pid"] == pid and
            start["worker_plan_sha"] == terminal["worker_plan_sha"] == process["worker_plan_sha"] == worker_sha and
            start["runtime"] == plan["runtime"], "fresh worker native identity")
    require(process["returncode"] == 0 and process["terminated_group"] is False and terminal["status"] == "PASS" and
            process["started"] <= start["start_clock"] <= terminal["finished_clock"] <= process["finished"] and
            process["finished"] - process["started"] <= CAPS["child_seconds"] and
            terminal["versions"] == plan["runtime"]["versions"] and
            terminal["module_origins"] == plan["runtime"]["module_origins"] and
            terminal["threads"] == [1, 1] and terminal["device"] == "cpu" and terminal["dtype"] == "float32", "worker completion/runtime/time contract")
    rows = operation_lines(directory / "operations.jsonl")
    windows = operations(rows, arm, pid, start["start_clock"], terminal["finished_clock"])
    identity = load(directory / "model_identity.json")
    require(identity["base_weight_sha256"] == plan["resources"]["base"]["weight_sha256"] and
            identity["model_training"] is False and isinstance(identity["effective_generation_config"], dict), "actual eval base/generation identity")
    effective = identity["effective_generation_config"]
    require(all(effective[k] == value for k, value in plan["generation"].items()) and
            effective["bos_token_id"] == 0 and effective["eos_token_id"] == 49279 and effective["pad_token_id"] == 2,
            "effective native BOS/EOS/PAD and requested greedy options")
    if arm == "base":
        require(identity["is_peft"] is False and identity["adapter_modules"] == 0 and
                not (directory / "loaded_adapter_parameters.json").exists(), "base arm must have no adapter")
    else:
        require(identity["is_peft"] is True and type(identity["adapter_modules"]) is int and identity["adapter_modules"] > 0 and
                identity["adapter_weight_sha256"] == plan["resources"]["adapter"]["weight_sha256"] and
                identity["active_adapters"] == ["helmet"] and identity["available_adapters"] == ["helmet"] and
                identity["enabled"] is True and identity.get("merged_adapters", []) == [], "actual one enabled, unmerged Helmet adapter")
        loaded_parameters(load(directory / "loaded_adapter_parameters.json"), originals)
    found = sorted(p.name for p in regular(directory / "cases", directory=True).iterdir())
    require(found == list(CASES), "no missing/additional/replacement generated case")
    records, inputs = [], {}
    tokenizer_path = Path(plan["resources"]["base"]["path"]) / "tokenizer.json"
    rendered = None
    for case_id in CASES:
        case_dir = regular(directory / "cases" / case_id, directory=True)
        case, output = load(case_dir / "claim.json"), load(case_dir / "receipt.json")
        crop = crops[case_id]
        require(case["case_id"] == output["case_id"] == case_id and case["crop_sha"] == crop["crop_sha"] and
                case["crop_rgb_sha"] == crop["crop_rgb_sha"] and case["prompt"] == PROMPT and case["generation"] == plan["generation"] and
                case["inference_mode"] is True and case["grad_enabled"] is False and output["status"] == "PASS", "actual paired call input/inference/generation identity")
        require(isinstance(case["rendered_prompt"], str) and PROMPT in case["rendered_prompt"] and
                not any(name in case["rendered_prompt"] for name in CASES), "fixed answer-free rendered prompt")
        if rendered is None:
            rendered = case["rendered_prompt"]
        require(case["rendered_prompt"] == rendered, "per-case prompt changed")
        inputs[case_id] = tensor_inputs(case)
        require(start["start_clock"] <= case["started"] == output["started"] <= output["finished"] <= terminal["finished_clock"] and
                case["started"] <= windows[case_id][0] <= windows[case_id][1] <= output["finished"], "raw call clock containment")
        predicted = decode_output(tokenizer_path, output["returned_ids"], case["input_ids"], output["output_ids"], output["raw_text"])
        records.append({"case_id": case_id, "arm": arm, "raw_text": output["raw_text"],
                        "parsed_label": predicted, "format_error": predicted is None,
                        "output_token_count": len(output["output_ids"]), "output_ids": output["output_ids"],
                        "claim_sha256": digest(case_dir / "claim.json"), "receipt_sha256": digest(case_dir / "receipt.json")})
    return {"pid": pid, "started": process["started"], "finished": process["finished"],
            "effective_generation_config": effective, "inputs": inputs,
            "counts": operation_charge(rows), "records": records}


def optional_metadata(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return {"state": "missing", "path": str(path)}, None
    try:
        obj = load(path)
        require(isinstance(obj, dict), "metadata object required")
        return {"state": "valid", "path": str(path), "sha256": digest(path)}, obj
    except (OSError, ValueError, TypeError) as error:
        return {"state": "invalid", "path": str(path), "error": type(error).__name__ + ": " + str(error)}, None


def retained_bill(root):
    counts = {name: 0 for name in ("processor_load", "base_load", "adapter_load", "activation", "generate")}
    uppers = dict(counts)
    arms, launch_observed, launch_upper, pid_observed, pid_unknown = [], 0, 0, 0, 0
    for arm in ARMS:
        path = root / arm / "operations.jsonl"
        intent_state, intent = optional_metadata(root / ("worker_%s_control_intent.json" % arm))
        control_state, control = optional_metadata(root / ("worker_%s_control.json" % arm))
        process_state, process = optional_metadata(root / ("worker_%s_process.json" % arm))
        start_state, start = optional_metadata(root / arm / "worker_start.json")
        row = {"arm": arm, "launch_intent": intent_state, "launch_control": control_state,
               "compatibility_process_receipt": process_state, "worker_start_receipt": start_state,
               "worker_started": start is not None, "process_receipt_exists": process_state["state"] != "missing", "parse_errors": []}
        # PID creation is distinct from entering Python's instrumented worker.
        evidence = [obj for obj in (control, process, start) if isinstance(obj, dict)]
        pids = sorted({obj["pid"] for obj in evidence if type(obj.get("pid")) is int and obj["pid"] > 0})
        row["PID_created"] = True if pids else False if control is not None and control.get("pid", "missing") is None else None
        row["observed_PIDs"] = pids
        row["launch_attempts_observed"] = int(intent is not None or bool(evidence))
        row["launch_attempts_upper_bound"] = 1
        launch_observed += row["launch_attempts_observed"]
        launch_upper += row["launch_attempts_upper_bound"]
        pid_observed += int(row["PID_created"] is True)
        pid_unknown += int(row["PID_created"] is None)
        events = []
        row["ledger_state"] = "missing"
        if path.exists() or path.is_symlink():
            try:
                row["operation_file_sha256"] = digest(path)
                for number, line in enumerate(regular(path).read_text(encoding="utf-8").splitlines()):
                    try:
                        event = json_object(line)
                        require(isinstance(event, dict), "operation object required")
                        events.append(event)
                    except (ValueError, TypeError) as error:
                        row["parse_errors"].append({"line": number + 1, "error": str(error)})
                require(not row["parse_errors"] and start is not None, "invalid/missing worker ledger identity")
                operations(events, arm, start["pid"], start["start_clock"], clock())
                row["ledger_state"] = "complete_verified"
            except (OSError, ValueError, TypeError, KeyError) as error:
                row["ledger_state"] = "partial_or_invalid"
                row["ledger_error"] = type(error).__name__ + ": " + str(error)
        row["counts"] = operation_charge(events)
        limits = {"processor_load": 1, "base_load": 1, "adapter_load": int(arm == "helmet"),
                  "activation": int(arm == "helmet"), "generate": 6}
        row["operation_counts_exact"] = row["ledger_state"] == "complete_verified"
        row["operation_attempt_upper_bounds"] = {name: number if row["operation_counts_exact"] else max(number, limits[name])
                                                 for name, number in row["counts"].items()}
        row["generate_unknown_upper"] = 0 if row["operation_counts_exact"] else max(0, 6 - row["counts"]["generate"])
        begins = [event for event in events if event.get("edge") == "begin"]
        row["native_begin_event_count"] = len(begins)
        row["native_begin_events"] = [{key: event.get(key) for key in ("op", "edge", "clock", "pid", "arm", "case_id")}
                                      for event in begins[:64]]
        row["raw_begin_export_truncated"] = len(begins) > 64
        for name, number in row["counts"].items():
            counts[name] += number
            uppers[name] += row["operation_attempt_upper_bounds"][name]
        arms.append(row)
    consumer_state, _ = optional_metadata(root / "consumer_process_intent.json")
    return {"operation_attempt_counts": counts, "operation_attempt_counts_are_observed_lower_bounds": True,
            "operation_attempt_upper_bounds": uppers,
            "launch_attempts_observed": launch_observed, "launch_attempts_upper_bound": launch_upper,
            "PID_created_observed": pid_observed, "PID_created_unknown_upper": pid_unknown,
            "worker_start_count": sum(row["worker_started"] for row in arms),
            "consumer_launch_intent": consumer_state, "consumer_is_not_a_model_worker": True,
            "missing_or_bad_ledger_never_certifies_zero_model_calls": True,
            "arms": arms, "unclosed_calls_are_charged": True, "all_existing_raw_logs_retained": True}


def audit(plan_path, expected_plan_sha, root, output):
    root, output, plan_path = regular(root, directory=True), Path(output).absolute(), regular(plan_path)
    require(not any(path.exists() or path.is_symlink() for path in
                    (output, root / "terminal.json", root / "consumer_claim.json", root / "consumer_failure.json")), "single-use quality consumer; no rerun")
    regular(output.parent, directory=True)
    started = clock()
    create(root / "consumer_claim.json", {"started_clock": started, "expected_plan_sha": expected_plan_sha,
                                         "consumer_sha256": digest(__file__), "single_use": True})
    result = {"reviewed_at": datetime.now(timezone.utc).isoformat(), "literature_cutoff": "No new literature search",
              "target_venue": "CSCWD2027; TMC policy is internal only", "artifact_run_id": root.name,
              "policy_version": "tmc_review_policy_v3_20260621", "git_commit": None,
              "evidence_level": "E0_UNAVAILABLE", "audit_success": False, "paired_matrix_complete": False,
              "scope": "Six frozen Helmet ROIs; finite descriptive paired quality fixture only",
              "ALPR": 0, "hidden": 0, "training": 0, "fitting": 0, "new_downloads": 0,
              "population_accuracy": "UNVERIFIED", "training_independence": "UNVERIFIED",
              "detection_or_localization_quality": "NOT_MEASURED", "paper_goal_complete": False,
              "requires_matching_terminal": True}
    prior_signal = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("quality consumer deadline")))
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    claim = plan = None
    produced = [root / "consumer_claim.json"]
    try:
        signal.setitimer(signal.ITIMER_REAL, CAPS["consumer_seconds"] - 2)
        require(digest(plan_path) == expected_plan_sha and digest(root / "plan.json") == expected_plan_sha, "external/frozen main plan SHA")
        plan, claim = load(plan_path), load(root / "claim.json")
        result["git_commit"] = plan["source_commit"]
        require(plan["new_caps"] == CAPS and plan["prompt"] == PROMPT and plan["parser"] == PARSER and
                Path(plan["raw_root"]).absolute() == root and Path(plan["audit_output"]).absolute() == output and
                claim["plan_sha"] == expected_plan_sha and claim["boot_uuid"] == boot(), "frozen scope/path/clock identity")
        grant = load(root / "grant.json")
        require(digest(root / "grant.json") == claim["grant_sha256"] and grant["plan_sha256"] == expected_plan_sha and
                grant["allow_quality_pilot"] is True and grant["generate_calls"] == 12 and grant["arms"] == 2 and
                grant["training"] == grant["downloads"] == 0, "single paired-quality authorization binding")
        elapsed = clock() - claim["start_clock"]
        require(0 <= elapsed < CAPS["wall_seconds"] - 2, "global quality deadline")
        signal.setitimer(signal.ITIMER_REAL, min(CAPS["consumer_seconds"] - 2, CAPS["wall_seconds"] - elapsed - 2))
        require(plan["generation"] == {"do_sample": False, "num_beams": 1, "use_cache": True, "max_new_tokens": 32}, "frozen greedy generation kwargs")
        source_identity(plan)
        runtime_identity(plan["runtime"])
        total, logs, outer = tree_bytes(plan["storage_roots"])
        require(all(any(path.is_relative_to(basepath) for basepath in outer) for path in (root, output, plan_path)) and
                total + 1048576 <= CAPS["artifact_bytes"] and logs + 1048576 <= CAPS["log_bytes"], "all new artifacts/logs including external controls")
        originals = resource_identity(plan)
        labels, crops = delivery_inputs(plan)
        arms = {arm: audit_arm(root, arm, plan, crops, originals) for arm in ARMS}
        require(arms["base"]["pid"] != arms["helmet"]["pid"] and
                claim["start_clock"] <= arms["base"]["started"] <= arms["base"]["finished"] <= arms["helmet"]["started"] <=
                arms["helmet"]["finished"] <= claim["start_clock"] + CAPS["producer_seconds"], "two sequential fresh processes / producer time")
        require(arms["base"]["inputs"] == arms["helmet"]["inputs"] and
                arms["base"]["effective_generation_config"] == arms["helmet"]["effective_generation_config"], "all actual input tensors/options matched across arms")
        counts = {name: sum(arms[arm]["counts"][name] for arm in ARMS) for name in arms["base"]["counts"]}
        require(all(counts[name] == CAPS[name] for name in counts), "exact twelve-call/load/activation budget")
        rows, summary, paired = [], {}, []
        for arm in ARMS:
            for row in arms[arm]["records"]:
                row["groundtruth"] = labels[row["case_id"]]
                row["correct"] = row["parsed_label"] == row["groundtruth"]
                rows.append(row)
            correct = sum(row["correct"] for row in arms[arm]["records"])
            summary[arm] = {"correct": correct, "total": 6, "accuracy_descriptive": correct / 6,
                            "format_errors": sum(row["format_error"] for row in arms[arm]["records"]),
                            "finite_fixture_qualified": correct == 6}
        for base_row, helmet_row in zip(arms["base"]["records"], arms["helmet"]["records"]):
            paired.append({"case_id": base_row["case_id"], "base_correct": base_row["correct"],
                           "helmet_correct": helmet_row["correct"],
                           "prediction_disagreement": base_row["parsed_label"] != helmet_row["parsed_label"]})
        table = io.StringIO(newline="")
        fields = [key for key in rows[0] if key != "output_ids"]
        writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
        exports = {output.parent / "helmet_quality_cases.csv": table.getvalue().encode(),
                   output.parent / "helmet_quality_raw_results.json": encoded({"results": rows, "paired": paired, "summary": summary})}
        for path, data in exports.items():
            create(path, data)
            produced.append(path)
        final_total, final_logs, _ = tree_bytes(plan["storage_roots"])
        require(final_total + 1048576 <= CAPS["artifact_bytes"] and final_logs + 1048576 <= CAPS["log_bytes"],
                "new artifacts/logs with final audit/control reserve")
        require(boot() == claim["boot_uuid"] and clock() - claim["start_clock"] < CAPS["wall_seconds"] and
                clock() - started < CAPS["consumer_seconds"] - 1, "quality completion clock/boot")
        result.update(status="AUDITED_HELMET6_PAIRED_QUALITY", evidence_level="E2_ARTIFACT_AUDITED", audit_success=True,
                      paired_matrix_complete=True, generate_calls=12, operation_counts=counts, summary=summary, paired=paired,
                      plan_sha256=expected_plan_sha, groundtruth_sha256=plan["input_reference"]["groundtruth_sha"],
                      output_files={str(path): hashlib.sha256(data).hexdigest() for path, data in exports.items()},
                      storage_before_exports=total, log_bytes_before_exports=logs)
    except BaseException as error:
        result.update(status="QUALITY_PILOT_UNVERIFIED_FAILURE_PRESERVED", error_type=type(error).__name__, error=str(error),
                      audit_success=False, paired_matrix_complete=False, failure_bill=retained_bill(root))
        path = root / "consumer_failure.json"
        create(path, {"audit_success": False, "error_type": type(error).__name__, "error": str(error),
                      "failure_bill": result["failure_bill"], "consumer_must_not_rerun": True})
        produced.append(path)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result["consumer_elapsed_seconds"] = clock() - started
    result["elapsed_scope_seconds"] = clock() - claim["start_clock"] if isinstance(claim, dict) and type(claim.get("start_clock")) in (int, float) else None
    try:
        remaining = CAPS["consumer_seconds"] - (clock() - started)
        if result["audit_success"]:
            remaining = min(remaining, CAPS["wall_seconds"] - result["elapsed_scope_seconds"])
        require(remaining > 0, "quality terminal publication deadline")
        signal.setitimer(signal.ITIMER_REAL, min(2, remaining))
        create(output, result)
        create(root / "terminal.json", {"audit": str(output), "sha256": digest(output), "audit_success": result["audit_success"],
                                        "paired_matrix_complete": result["paired_matrix_complete"], "finished_clock": clock(), "consumer_must_not_rerun": True})
    except BaseException as error:
        path = root / "consumer_failure.json"
        if not path.exists():
            create(path, {"audit_success": False, "error": str(error), "failure_bill": retained_bill(root), "consumer_must_not_rerun": True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_signal)
        if prior_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *prior_timer)
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
