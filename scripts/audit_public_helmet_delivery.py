"""Offline, single-use consumer of the fixed six-image Helmet delivery.

References sealed acquisition records without rerunning their audits. No network,
models, training, replacement samples or task-quality evaluation are performed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import re
import signal
from urllib.parse import parse_qs, urlsplit
import warnings

if __package__:
    from . import audit_public_helmet_pilot as base
else:
    import audit_public_helmet_pilot as base

require, load, digest, create = base.require, base.load, base.digest, base.create
clock, integer, finite = base.clock, base.integer, base.finite
CAPS = {"http": 13, "body": 60826959, "wall": 300, "storage": 268435456}
PRIOR = {"http": 19, "body": 5526965, "read_upper": 65536, "declared_unread_upper": 689404}
PRODUCER_SECONDS = 240
CONSUMER_SECONDS = 60
OUTPUT_CAP = 8388608
INVENTORY_COMMIT = "23e3c55ef17f143a9c48ed33cbb0bf4a1dc74a8a"
CRITICAL_HEADERS = {"content-length", "content-type", "content-encoding", "content-range", "location"}
TARGET_FIELDS = {"sample_id", "image_file_id", "image_filename", "image_metadata", "annotation", "category_name"}


def regular(path, *, directory=False):
    """Never follow symlinks in a provenance or image-input path."""
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in [path, *path.parents]), "reference/input symlink forbidden")
    require(path.is_dir() if directory else path.is_file(), "missing regular reference/input")
    return path


def header_fields(receipt):
    """Preserve raw pairs; reject only conflicting singleton protocol fields."""
    pairs = receipt["header_pairs"]
    require(isinstance(pairs, list), "raw header_pairs required")
    normalized = {}
    for pair in pairs:
        require(isinstance(pair, list) and len(pair) == 2 and
                all(isinstance(value, str) for value in pair), "raw header pair strings")
        key, value = pair
        name = key.lower()
        if name in normalized:
            if name in CRITICAL_HEADERS:
                require(normalized[name] == value, "conflicting critical response header: " + name)
            else:
                normalized[name] += ("\n" if name == "set-cookie" else ", ") + value
        else:
            normalized[name] = value
    require(dict(pairs) == receipt["headers"], "raw header pair/dict provenance")
    return normalized


def public_headers(receipt):
    fields = base.lower_headers(receipt["request_headers"])
    require(set(fields) <= {"accept-encoding", "user-agent", "accept"} and
            fields.get("accept-encoding", "").strip().lower() == "identity",
            "fixed unauthenticated identity-encoding request headers")
    headers = header_fields(receipt)
    require(headers.get("content-encoding", "identity").strip().lower() in ("", "identity"),
            "response identity encoding")
    require(receipt["final_url"] == receipt["url"], "automatic redirect forbidden")
    return headers


def retained_body(folder, receipt):
    for field in ("bytes", "retained_bytes", "upper"):
        integer(receipt[field], "receipt " + field)
    require(receipt["retained_bytes"] <= receipt["bytes"] and receipt["upper"] <= 65536,
            "retained/returned/read-upper accounting")
    path = Path(folder) / "body.bin"
    if path.exists() or path.is_symlink():
        regular(path)
        require(path.stat().st_size == receipt["retained_bytes"] and
                digest(path) == receipt["body_sha256"], "original retained body SHA/size")
    else:
        require(receipt["retained_bytes"] == 0 and receipt["body_sha256"] is None,
                "missing body accounting")
    return path


def complete_transport(receipt, headers, *, metadata=False):
    require(receipt["status"] in ("complete", "redirect") and
            receipt["bytes"] == receipt["retained_bytes"] and receipt["upper"] == 0,
            "complete response accounting")
    length = headers.get("content-length")
    require(length is None or (length.isdecimal() and int(length) == receipt["bytes"]),
            "complete Content-Length")
    if receipt["status"] == "complete":
        require(receipt["status_code"] in ((200, 206) if metadata else (200,)), "complete HTTP status")
        mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        require(mime == "application/json" if metadata else
                mime.startswith("image/") or mime == "application/octet-stream", "complete MIME")


def references(plan):
    """Validate only the necessary immutable source chain, not an old audit run."""
    item = plan["inventory_reference"]
    root = regular(item["root"], directory=True)
    require(root == base.SOURCE_ROOT / "data/raw/task_quality/public_helmet_pilot_20261030_v3",
            "fixed v3 inventory root")
    for filename, field in (("plan.json", "plan_sha"), ("claim.json", "claim_sha"),
                            ("selection.json", "selection_sha"), ("selection_claim.json", "selection_claim_sha")):
        require(digest(regular(root / filename)) == item[field], "v3 reference SHA: " + filename)
    oldplan, oldclaim = load(root / "plan.json"), load(root / "claim.json")
    require(oldplan["source_commit"] == INVENTORY_COMMIT and oldplan["raw_root"] == str(root) and
            oldplan["dataset_id"] == base.DATASET_ID and oldplan["version"] == 1 and
            oldplan["seed_schema"] == base.SELECTION_RULE, "fixed v3 source/dataset identity")
    require(oldclaim["plan_sha"] == item["plan_sha"] and
            digest(regular(root / "grant.json")) == oldclaim["grant_sha256"], "v3 claim/grant identity")
    archived = regular(item["audit_path"])
    require(digest(archived) == item["audit_sha"], "archived v3 audit SHA")
    oldaudit, terminal = load(archived), load(regular(root / "terminal.json"))
    require(oldaudit["input_qualified"] is False and oldaudit["audit_success"] is False and
            oldaudit["error"] == "conflicting critical raw header" and
            terminal["sha256"] == item["audit_sha"] and
            Path(terminal["audit"]).resolve() == archived.resolve(), "old failed terminal unchanged")
    # This pure provenance check does not write, call audit(), or reset old time.
    base.pagination_provenance(oldplan)
    parent = oldplan["parent_audit"]
    require(digest(regular(parent["path"])) == parent["sha256"], "original M12 parent SHA")
    original = load(parent["path"])
    require((original["cumulative_requests"], original["cumulative_body_bytes"],
             original["cumulative_interrupted_upper"]) == (12, 1588374, 0), "original M12 costs once")
    annotation = plan["annotation_reference"]
    require(annotation == oldplan["annotation_reference"], "same complete v2 annotation reference")
    coco_path = regular(annotation["path"])
    require(coco_path == base.SOURCE_ROOT / "data/raw/task_quality/public_helmet_pilot_20261030_v2/request_001/body.bin" and
            digest(coco_path) == annotation["sha256"] == base.ANNOTATION["sha256"] and
            coco_path.stat().st_size == annotation["size"] == base.ANNOTATION["size"], "publisher COCO SHA/size")
    coco = load(coco_path)
    require(isinstance(coco, dict) and all(isinstance(coco.get(k), list) for k in
            ("images", "categories", "annotations")) and len(coco["images"]) == 2392, "native COCO 2392 images")
    paths = sorted(root.glob("request_*"))
    require(len(paths) == len(item["receipt_sha"]) == 4, "exact fixed v3 four-attempt prefix")
    records, entries, seen_urls = [], [], set()
    for number, folder in enumerate(paths):
        regular(folder, directory=True)
        require(folder.name == "request_%03d" % number and
                digest(regular(folder / "receipt.json")) == item["receipt_sha"][number], "v3 receipt SHA/sequence")
        receipt = load(folder / "receipt.json")
        headers = public_headers(receipt)
        body = retained_body(folder, receipt)
        require(receipt["url"] not in seen_urls and receipt["redirect_from"] is None and
                oldclaim["start_clock"] <= receipt["started"] <= receipt["finished"] <= clock() and
                (not records or records[-1]["finished"] <= receipt["started"]), "v3 request identity/clock")
        seen_urls.add(receipt["url"])
        if number < 3:
            require(receipt["role"] == "image_metadata_page_%d" % number and
                    receipt["status"] == "complete" and receipt["status_code"] == (206 if number < 2 else 200),
                    "v3 fixed native page response")
            complete_transport(receipt, headers, metadata=True)
            u = base.safe_url(receipt["url"], initial=True)
            require(u.path == f"/public-api/datasets/{base.DATASET_ID}/files" and
                    parse_qs(u.query, keep_blank_values=True) == {
                        "folder_id": [base.IMAGES_FOLDER_ID], "version": ["1"],
                        "$start": [str(number * 1000)], "$limit": ["1000"]}, "v3 native page URL")
            obj = load(body)
            page = obj.get("files") if isinstance(obj, dict) else obj
            require(isinstance(page, list) and len(page) == (1000 if number < 2 else 392), "v3 complete page lengths")
            require(headers.get("content-range") == "items %d-%d/2392" %
                    (number * 1000, number * 1000 + len(page) - 1), "v3 native item range")
            entries.extend(page)
        records.append(receipt)
    require(sum(r["bytes"] for r in records) == 1807008 and
            sum(r["upper"] for r in records) == 65536 and
            PRIOR["http"] == 12 + 3 + len(records) and
            PRIOR["body"] == 1588374 + 2131583 + sum(r["bytes"] for r in records), "prefix costs counted once")
    metadata, file_ids = {}, set()
    for entry in entries:
        name, file_id, *_ = base.file_metadata(entry)
        require(name not in metadata and file_id not in file_ids, "unique official image IDs/names")
        metadata[name] = entry
        file_ids.add(file_id)
    expected, rejected = base.expected_selection(coco, metadata)
    selection, freeze = load(root / "selection.json"), load(root / "selection_claim.json")
    require(selection["selection_rule"] == base.SELECTION_RULE and
            freeze["selection_sha256"] == item["selection_sha"] and
            freeze["boot_uuid"] == oldclaim["boot_uuid"] and
            records[2]["finished"] <= freeze["frozen_clock"] < records[3]["started"], "original pre-image selection freeze")
    targets = selection["selected"]
    require(isinstance(targets, list) and len(targets) == 6, "frozen six targets")
    ids = set()
    for target, expected_target in zip(targets, expected):
        require(set(target) == TARGET_FIELDS and
                isinstance(target["sample_id"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", target["sample_id"]) and
                target["sample_id"] not in ids and all(target[k] == expected_target[k] for k in TARGET_FIELDS - {"sample_id"}),
                "original targets differ from independently reconstructed native GT selection")
        ids.add(target["sample_id"])
    first_ref = plan["first_redirect_reference"]
    require(regular(first_ref["receipt_path"]) == root / "request_003/receipt.json" and
            digest(first_ref["receipt_path"]) == first_ref["sha256"] == item["receipt_sha"][3], "first historical redirect SHA")
    first = records[3]
    first_headers = header_fields(first)
    require(first["status"] == "failed" and first["status_code"] == 302 and
            first["role"] == "image_sample_0" and first["url"] == targets[0]["image_metadata"]["content_details"]["download_url"] and
            first["bytes"] == first["retained_bytes"] == 0 and first["upper"] == 65536 and
            first["body_sha256"] is None and not (root / "request_003/body.bin").exists() and
            first["error"] == "conflicting response header", "historical failed302 is not a completed image")
    location = first_headers.get("location")
    base.publisher_location(location)
    return {"root": root, "claim": oldclaim, "records": records, "targets": targets,
            "expected": expected, "rejected": rejected, "location": location,
            "selection_sha": item["selection_sha"], "coco_sha": annotation["sha256"]}


def request_inventory(root, claim, ref):
    paths = sorted(root.glob("request_*"))
    require(len(paths) <= CAPS["http"], "delivery HTTP cap")
    records, logical, seen = [], {}, set()
    for number, folder in enumerate(paths):
        regular(folder, directory=True)
        require(folder.name == "request_%03d" % number, "delivery contiguous request sequence")
        receipt = load(regular(folder / "receipt.json"))
        role = receipt["role"]
        require(isinstance(role, str) and re.fullmatch(r"image_sample_[0-5]", role) and
                receipt["status"] in ("complete", "redirect", "failed"), "image-only terminal request role")
        require(not records or records[-1]["status"] != "failed", "request after failed prefix")
        headers = public_headers(receipt)
        body = retained_body(folder, receipt)
        require(receipt["bytes"] + receipt["upper"] <= base.IMAGE_BYTES and
                receipt["url"] not in seen, "image cap / URL retry forbidden")
        seen.add(receipt["url"])
        started, finished = finite(receipt["started"], "request start"), finite(receipt["finished"], "request finish")
        require(claim["start_clock"] <= started <= finished <= clock() and
                finished <= claim["start_clock"] + PRODUCER_SECONDS and
                (not records or records[-1]["finished"] <= started), "delivery producer clocks / consumer reserve")
        native_url = ref["targets"][int(role[-1])]["image_metadata"]["content_details"]["download_url"]
        parent = receipt["redirect_from"]
        if parent == -1:
            require(type(parent) is int and number == 0 and role == "image_sample_0" and
                    receipt["url"] == ref["location"] and role not in logical,
                    "only first request follows exact historical literal Location")
            base.publisher_location(receipt["url"])
            logical[role] = [number]
        elif parent is None:
            require(role != "image_sample_0" and role not in logical and receipt["url"] == native_url,
                    "fixed original URL; frozen sample0 source must not be retried")
            base.safe_url(receipt["url"], initial=True)
            logical[role] = [number]
        else:
            integer(parent, "redirect parent")
            require(parent == number - 1 and records[parent]["role"] == role and
                    records[parent]["status"] == "redirect" and records[parent]["redirect_from"] is None and
                    receipt["url"] == header_fields(records[parent])["location"], "one literal publisher Location hop")
            base.publisher_location(receipt["url"])
            logical[role].append(number)
        if receipt["status"] in ("complete", "redirect"):
            complete_transport(receipt, headers)
            regular(body)
        if receipt["status"] == "redirect":
            require(parent is None and receipt["status_code"] in (302, 303, 307, 308), "one new publisher redirect only")
            base.publisher_location(headers.get("location"))
        records.append(dict(receipt, index=number, body_path=str(body), receipt_sha256=digest(folder / "receipt.json")))
    body = sum(r["bytes"] for r in records)
    upper = sum(r["upper"] for r in records)
    require(body + upper <= CAPS["body"] and PRIOR["http"] + len(records) <= 32 and
            PRIOR["body"] + PRIOR["read_upper"] + PRIOR["declared_unread_upper"] + body + upper <= 64 * 1024 * 1024,
            "new and cumulative HTTP/body/uncertainty caps")
    return records, logical


def image_rows(root, index, records, logical, ref):
    require(set(logical) == {"image_sample_%d" % n for n in range(6)} and
            isinstance(index["selected"], list) and len(index["selected"]) == 6, "exact six-image delivery inventory")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = base.MAX_PIXELS
    rows, hashes = [], set()
    for number, (target, actual, expected) in enumerate(zip(ref["targets"], index["selected"], ref["expected"])):
        require(set(actual) in (TARGET_FIELDS | {"request_index"}, TARGET_FIELDS | {"request_index", "opaque_input_path"}) and
                all(actual[k] == target[k] for k in TARGET_FIELDS), "delivery cannot replace frozen targets")
        role = "image_sample_%d" % number
        request_index = integer(actual["request_index"], "image request index")
        require(logical[role][-1] == request_index and records[request_index]["status"] == "complete", "complete final image request")
        receipt = records[request_index]
        filename, file_id, size, sha, _ = base.file_metadata(target["image_metadata"])
        require(receipt["bytes"] == size and receipt["body_sha256"] == sha and sha not in hashes,
                "six distinct original publisher image SHA/size")
        hashes.add(sha)
        body = regular(receipt["body_path"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(body) as image:
                width, height = image.size
                require(0 < width <= base.MAX_EDGE and 0 < height <= base.MAX_EDGE and
                        width * height <= base.MAX_PIXELS, "actual image geometry cap")
                image.verify()
            with Image.open(body) as image:
                require(image.size == (width, height), "image reopen geometry")
                image.load()
        require((width, height) == (expected["coco_image"]["width"], expected["coco_image"]["height"]), "actual/native GT dimensions")
        if "opaque_input_path" in actual:
            supplied = Path(actual["opaque_input_path"])
            supplied = supplied if supplied.is_absolute() else root / supplied
            require(regular(supplied) == body, "delivery input is original opaque request body only")
        x, y, w, h = target["annotation"]["bbox"]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in (x, y, w, h)) and
                x >= 0 and y >= 0 and w >= 32 and h >= 32 and x + w <= width and y + h <= height,
                "native unmodified xywh ROI within actual image")
        rows.append({"sample_id": target["sample_id"], "category_name": target["category_name"],
                     "image_file_id": file_id, "image_filename": filename,
                     "annotation_id": target["annotation"]["id"], "image_id": target["annotation"]["image_id"],
                     "native_category_id": target["annotation"]["category_id"], "x": x, "y": y,
                     "width": w, "height": h, "image_width": width, "image_height": height,
                     "image_bytes": size, "image_sha256": sha, "request_index": request_index,
                     "input_path": str(body), "native_annotation": target["annotation"], "native_image": expected["coco_image"]})
    return rows


def failure_bill(root):
    bill = base.conservative_bill(root)
    bill.update(fixed_parent_costs_not_readded_recursively=PRIOR,
                cumulative_attempt_count_conservative=PRIOR["http"] + bill["attempt_directory_count"],
                unverified_new_body_upper_conservative=CAPS["body"],
                prior_declared_unread_entity_upper=PRIOR["declared_unread_upper"])
    return bill


def audit(root, output):
    root, output = regular(root, directory=True), Path(output).absolute()
    terminal, attempt = root / "terminal.json", root / "consumer_claim.json"
    require(not any(p.exists() or p.is_symlink() for p in
                    (output, terminal, attempt, root / "consumer_failure.json")), "one delivery consumer only; no rerun")
    regular(output.parent, directory=True)
    started = clock()
    create(attempt, {"started_clock": started, "output": str(output), "consumer_sha256": digest(__file__), "single_use": True})
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("delivery consumer deadline")))
    original_alarm = signal.getitimer(signal.ITIMER_REAL)
    produced = [attempt]
    result = {"reviewed_at": datetime.now(timezone.utc).isoformat(), "literature_cutoff": "not applicable; no search",
              "target_venue": "not a venue-readiness review", "artifact_run_id": root.name,
              "policy_version": "helmet_delivery_input_audit_v1_20261030", "git_commit": None,
              "evidence_level": "E0_UNAVAILABLE", "input_qualified": False, "audit_success": False,
              "qualified_scope": "Helmet6 only", "ALPR_qualified": False, "twelve_input_plan_qualified": False,
              "quality_improvement": "UNVERIFIED", "training_independence": "UNVERIFIED", "hidden_independence": "UNVERIFIED",
              "model_calls": 0, "training": 0, "inference": 0, "scientific_Q": 0, "paper_goal_complete": False,
              "license_boundary": "Publisher page licence declaration only; underlying rights chain unverified.",
              "old_failures_unchanged": True, "old_scope_wall_not_reset": True,
              "wire_total": "UNAVAILABLE; returned bodies and separate uncertainty bounds only", "requires_matching_terminal": True}
    claim = None
    try:
        signal.setitimer(signal.ITIMER_REAL, CONSUMER_SECONDS - 2)
        plan, claim = load(regular(root / "plan.json")), load(regular(root / "claim.json"))
        require(plan["new_caps"] == CAPS and plan["prior"] == PRIOR and plan["producer_seconds"] == PRODUCER_SECONDS and
                plan["dataset_id"] == base.DATASET_ID and plan["version"] == 1 and plan["seed_schema"] == base.SELECTION_RULE and
                Path(plan["raw_root"]).absolute() == root and Path(plan["audit_output"]).absolute() == output, "fixed delivery plan")
        result["git_commit"] = plan["source_commit"]
        require(claim["plan_sha"] == digest(root / "plan.json") and claim["boot_uuid"] == base.boot_uuid(), "delivery plan/boot binding")
        grant = load(regular(root / "grant.json"))
        require(digest(root / "grant.json") == claim["grant_sha256"] and grant["plan_sha256"] == claim["plan_sha"] and
                grant["allow_six_image_downloads"] is True and grant["allow_one_publisher_location_hop"] is True and
                type(grant["model_calls"]) is int and grant["model_calls"] == 0,
                "delivery grant SHA/scope")
        elapsed = clock() - finite(claim["start_clock"], "delivery start clock")
        require(0 <= elapsed < CAPS["wall"] - 2, "fresh delivery scope deadline")
        signal.setitimer(signal.ITIMER_REAL, min(CONSUMER_SECONDS - 2, CAPS["wall"] - elapsed - 2))
        require({"scripts/audit_public_helmet_delivery.py", "scripts/audit_public_helmet_pilot.py",
                 "scripts/finish_public_helmet_pilot.py", "scripts/prepare_public_helmet_pilot.py"}.issubset(plan["source_sha"]) and
                plan["source_sha"]["scripts/audit_public_helmet_delivery.py"] == digest(__file__), "delivery consumer source SHA")
        base.source_binding(plan)
        stored, outer = base.stored_bytes(plan["storage_roots"])
        require(any(root.is_relative_to(p) for p in outer) and any(output.is_relative_to(p) for p in outer) and
                stored + OUTPUT_CAP <= CAPS["storage"], "cumulative storage / consumer reserve")
        ref = references(plan)
        require(claim["start_clock"] >= ref["records"][-1]["finished"] and
                digest(regular(root / "selection.json")) == ref["selection_sha"], "fresh scope follows preserved failure; identical frozen selection copy")
        records, logical = request_inventory(root, claim, ref)
        result.update(new_requests=len(records), new_body_bytes=sum(r["bytes"] for r in records),
                      new_interrupted_upper=sum(r["upper"] for r in records), cumulative_requests=PRIOR["http"] + len(records),
                      cumulative_body_bytes=PRIOR["body"] + sum(r["bytes"] for r in records),
                      cumulative_interrupted_upper=PRIOR["read_upper"] + sum(r["upper"] for r in records),
                      prior_declared_unread_entity_upper=PRIOR["declared_unread_upper"],
                      receipt_sha256=[r["receipt_sha256"] for r in records], storage_before_exports=stored,
                      original_selection_sha256=ref["selection_sha"], grant_sha256=claim["grant_sha256"])
        require(records and all(r["status"] != "failed" for r in records), "failed/empty image prefix cannot qualify")
        rows = image_rows(root, load(regular(root / "index.json")), records, logical, ref)
        table = io.StringIO(newline="")
        fields = [k for k in rows[0] if k not in ("native_annotation", "native_image")]
        writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
        exports = {output.parent / "helmet_samples.csv": table.getvalue().encode(),
                   output.parent / "helmet_groundtruth.json": base.encoded({"dataset_id": base.DATASET_ID, "version": 1,
                       "source_annotation_sha256": ref["coco_sha"], "selection_sha256": ref["selection_sha"],
                       "scope": "Helmet6 native-GT input qualification only; no model labels or task quality",
                       "samples": rows, "candidate_rejections": ref["rejected"]})}
        require(sum(map(len, exports.values())) + 65536 <= OUTPUT_CAP, "delivery consumer export cap")
        for path, data in exports.items():
            create(path, data)
            produced.append(path)
        require(claim["boot_uuid"] == base.boot_uuid() and clock() - claim["start_clock"] < CAPS["wall"] and
                clock() - started < CONSUMER_SECONDS - 1, "delivery completion clock/boot")
        result.update(status="HELMET6_INPUT_QUALIFIED", input_qualified=True, audit_success=True, images=6,
                      class_counts={"Helmet": 3, "NoHelmet": 3}, evidence_level="E2_ARTIFACT_AUDITED",
                      candidate_rejections=ref["rejected"], plan_sha256=digest(root / "plan.json"),
                      output_files={str(p): hashlib.sha256(data).hexdigest() for p, data in exports.items()})
    except BaseException as error:
        result.update(status="HELMET6_INPUT_UNQUALIFIED", input_qualified=False, audit_success=False,
                      evidence_level="E0_UNAVAILABLE", error_type=type(error).__name__, error=str(error), failure_bill=failure_bill(root))
        path = root / "consumer_failure.json"
        create(path, {"error_type": type(error).__name__, "error": str(error), "input_qualified": False,
                      "audit_success": False, "failure_bill": result["failure_bill"], "consumer_must_not_rerun": True})
        produced.append(path)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result["consumer_elapsed_seconds"] = clock() - started
    result["elapsed_new_scope"] = clock() - claim["start_clock"] if isinstance(claim, dict) and type(claim.get("start_clock")) in (int, float) else None
    require(sum(p.stat().st_size for p in produced) + len(base.encoded(result)) + 4096 <= OUTPUT_CAP, "all delivery consumer outputs cap")
    try:
        remaining = CONSUMER_SECONDS - (clock() - started)
        if result["input_qualified"]:
            remaining = min(remaining, CAPS["wall"] - result["elapsed_new_scope"])
        require(remaining > 0, "delivery terminal publication reserve")
        signal.setitimer(signal.ITIMER_REAL, min(2, remaining))
        create(output, result)
        create(terminal, {"audit": str(output), "sha256": digest(output), "input_qualified": result["input_qualified"],
                          "audit_success": result["audit_success"], "finished_clock": clock(), "consumer_must_not_rerun": True})
    except BaseException as error:
        path = root / "consumer_failure.json"
        if not path.exists():
            create(path, {"error_type": type(error).__name__, "error": str(error), "input_qualified": False,
                          "audit_success": False, "failure_bill": failure_bill(root), "consumer_must_not_rerun": True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if original_alarm[0]:
            signal.setitimer(signal.ITIMER_REAL, *original_alarm)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(base.encoded({"status": result["status"], "input_qualified": result["input_qualified"],
                        "audit_success": result["audit_success"]}).decode(), end="")
    raise SystemExit(0 if result["audit_success"] else 1)


if __name__ == "__main__":
    main()
