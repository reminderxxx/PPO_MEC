"""Single-use, offline audit of six publisher-labelled Helmet pilot inputs.

This consumer downloads nothing, invokes no model and does not certify task
quality, training independence, ALPR inputs, or the wider twelve-image plan.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
from urllib.parse import parse_qs, urljoin, urlsplit
import warnings


SOURCE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "j82bnw7gsr"
VERSION = 1
IMAGES_FOLDER_ID = "39245ed4-43d6-497c-9fd9-45adca1a6ca0"
ANNOTATION = {
    "filename": "annotations.json",
    "id": "425f058b-dd8f-4ce5-a761-d9522649e018",
    "size": 2065913,
    "sha256": "5e77fea3d8f1cd93cac42d45065456eeb97a54c31b7534ef453051c9d8853c91",
    "url": "https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/"
           "425f058b-dd8f-4ce5-a761-d9522649e018/file_downloaded",
}
PRIOR = (12, 1588374, 0)
NEW_CAPS = {"http": 20, "body": 65520490, "wall": 900, "storage": 268435456}
CLASS_ORDER = ("Helmet", "NoHelmet")
SELECTION_RULE = (
    "sha256(j82bnw7gsr/1|fileID|annotationID|categoryName);"
    "classes=Helmet,NoHelmet;first3_per_class;one_image;"
    "reject_same_xywh_native_class_conflict"
)
IMAGE_BYTES = 8 * 1024 * 1024
OUTPUT_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20000000
MAX_EDGE = 8192
READ_UPPER = 65536
AUDIT_SECONDS = 30
OFFICIAL_HOSTS = {
    "data.mendeley.com",
    "prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com",
    "prod-dcd-datasets-public-files-eu-west-1.s3.amazonaws.com",
}


class AuditError(ValueError):
    """An input contract failed; the original ledger must remain unchanged."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def boot_uuid():
    return subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"], text=True
    ).strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    # Duplicate JSON keys and non-finite numbers must not acquire new meanings.
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "duplicate JSON key: " + key)
            obj[key] = value
        return obj

    return json.loads(safe_file(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          AuditError("non-finite JSON number: " + value)))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def create(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value))


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be an integer")
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite")
    return value


def identifier(value, label):
    require(type(value) in (str, int) and str(value) != "", label + " identity")
    return (type(value).__name__, value)


def safe_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular original file required: " + str(path))
    return path


def safe_url(url, *, initial=False):
    require(isinstance(url, str), "URL string required")
    u = urlsplit(url)
    official = u.hostname in OFFICIAL_HOSTS or (u.hostname is not None and re.fullmatch(
        r"prod-dcd-datasets-public-files-[a-z0-9-]+\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com",
        u.hostname))
    require(u.scheme == "https" and official and
            not u.username and not u.password and not u.fragment and
            u.port in (None, 443), "official HTTPS URL scope")
    if initial:
        require(u.hostname == "data.mendeley.com", "publisher initial URL host")
    return u


def publisher_location(url):
    u = safe_url(url)
    require(re.fullmatch(
        r"prod-dcd-datasets-public-files-[a-z0-9-]+\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com",
        u.hostname or ""), "publisher public S3 Location required")
    return u


def source_binding(plan):
    require(re.fullmatch(r"[0-9a-f]{40}", plan["source_commit"] or ""), "source commit format")
    current = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    require(current == plan["source_commit"], "source Git commit changed")
    hashes = plan["source_sha"]
    require(isinstance(hashes, dict) and
            "scripts/audit_public_helmet_pilot.py" in hashes, "consumer source SHA required")
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "source names must stay in checkout")
        path = safe_file(SOURCE_ROOT / relative)
        require(path.resolve().is_relative_to(SOURCE_ROOT.resolve()) and
                digest(path) == expected, "source SHA changed: " + name)


def stored_bytes(roots):
    require(isinstance(roots, list) and roots, "storage_roots required")
    paths = [Path(p).resolve() for p in roots]
    require(len(paths) == len(set(paths)), "duplicate storage roots")
    # Overlapping roots are counted once, not once per listed ancestor.
    outer = [p for p in paths if not any(p != q and p.is_relative_to(q) for q in paths)]
    total = 0
    for base in outer:
        require(base.exists() and not base.is_symlink(), "storage root absent or symlink")
        for path in ([base] if base.is_file() else base.rglob("*")):
            if path.is_symlink():
                total += path.lstat().st_size
            elif path.is_file():
                total += path.stat().st_size
    return total, outer


def parent_binding(plan):
    parent = plan["parent_audit"]
    path = safe_file(parent["path"])
    require(digest(path) == parent["sha256"], "parent audit SHA")
    prior = load(path)
    actual = (prior["cumulative_requests"], prior["cumulative_body_bytes"],
              prior["cumulative_interrupted_upper"])
    claimed = (parent["cumulative_requests"], parent["cumulative_body_bytes"],
               parent["cumulative_upper"])
    require(actual == claimed == PRIOR, "fixed parent costs inherited exactly once")
    require(prior["new_requests"] == 2 and prior["new_body_bytes"] == 2403 and
            prior["new_interrupted_upper"] == 0 and prior["failures"] == 0 and
            prior["new300s_pass"] is True and prior["input_qualified"] is False and
            prior["images"] == prior["model_calls"] == 0, "sealed folder parent ledger")
    return actual


def zero_http_provenance(plan, current_claim):
    item = plan.get("zero_http_predecessor")
    if item is None:
        return
    oldroot = Path(item["root"])
    files = {"plan.json": "plan_sha256", "claim.json": "claim_sha256", "terminal.json": "terminal_sha256",
             "producer_failure.json": "producer_failure_sha256", "consumer_process.json": "consumer_process_sha256"}
    for filename, field in files.items():
        require(digest(safe_file(oldroot / filename)) == item[field], "predecessor exact file SHA")
    require(digest(safe_file(item["audit"])) == item["audit_sha256"], "predecessor audit SHA")
    prior_plan, prior_claim = load(oldroot / "plan.json"), load(oldroot / "claim.json")
    prior_audit, prior_terminal = load(item["audit"]), load(oldroot / "terminal.json")
    stopped, control = load(oldroot / "producer_failure.json"), load(oldroot / "consumer_process.json")
    require("zero_http_predecessor" not in prior_plan and
            prior_plan["source_commit"] == "b619c885281eb0493d290cbacac8589b84e85c36" and
            prior_plan["new_caps"] == NEW_CAPS and prior_plan["raw_root"] == str(oldroot) and prior_plan["parent_audit"] == plan["parent_audit"],
            "only initial direct zero-HTTP predecessor allowed")
    require(not list(oldroot.glob("request_*")) and stopped["requests"] == 0 and stopped["error"] == "storage symlink" and
            prior_audit["failure_bill"]["attempt_directory_count"] == 0 and prior_audit["input_qualified"] is False and
            control["returncode"] == 1 and prior_terminal["input_qualified"] is False and
            prior_terminal["sha256"] == item["audit_sha256"] and
            Path(prior_terminal["audit"]).resolve() == Path(item["audit"]).resolve() and
            prior_claim["plan_sha"] == item["plan_sha256"], "verified zero-network failed prefix")
    require(current_claim["start_clock"] == prior_claim["start_clock"] and
            current_claim["boot_uuid"] == prior_claim["boot_uuid"] == boot_uuid(),
            "original absolute deadline/boot must be inherited without reset")


def lower_headers(headers):
    require(isinstance(headers, dict), "headers object required")
    result = {}
    for key, value in headers.items():
        require(isinstance(key, str) and isinstance(value, str), "header strings")
        name = key.lower()
        require(name not in result or result[name] == value, "conflicting case-insensitive header")
        result[name] = value
    return result


def receipt_inventory(root, plan, claim):
    records, logical, seen_urls = [], {}, set()
    directories = sorted(root.glob("request_*"))
    require(len(directories) <= NEW_CAPS["http"], "new request count cap")
    for index, folder in enumerate(directories):
        require(folder.is_dir() and not folder.is_symlink() and
                folder.name == "request_%03d" % index, "contiguous request sequence")
        receipt_path = safe_file(folder / "receipt.json")
        r = load(receipt_path)
        require(r["status"] in ("complete", "redirect", "failed"), "terminal request status")
        require(r["role"] == "annotation" or
                re.fullmatch(r"image_metadata_page_[0-2]|image_sample_[0-5]", r["role"]),
                "fixed logical request role")
        if records:
            require(records[-1]["status"] != "failed", "new request after failed prefix")
        u = safe_url(r["url"], initial=r["redirect_from"] is None)
        require(r["url"] not in seen_urls, "same URL requested more than once")
        seen_urls.add(r["url"])
        require(r["final_url"] == r["url"], "automatic redirect forbidden")
        sent = lower_headers(r["request_headers"])
        require(sent.get("accept-encoding", "").strip().lower() == "identity",
                "explicit request identity encoding")
        require(set(sent) <= {"accept-encoding", "user-agent", "accept"},
                "only fixed public request headers allowed")
        require(not any(name in sent for name in ("authorization", "proxy-authorization",
                                                  "cookie", "range", "content-encoding")),
                "authentication/Range/request encoding forbidden")
        headers = lower_headers(r["headers"])
        require(headers.get("content-encoding", "identity").strip().lower() in ("", "identity"),
                "response identity encoding")
        for key in ("bytes", "retained_bytes", "upper"):
            integer(r[key], "receipt " + key)
        require(r["upper"] <= READ_UPPER and r["retained_bytes"] <= r["bytes"],
                "retained/returned/read-upper bill")
        body = folder / "body.bin"
        if body.exists():
            safe_file(body)
            require(body.stat().st_size == r["retained_bytes"] and
                    digest(body) == r["body_sha256"], "original body SHA/size")
        else:
            require(r["retained_bytes"] == 0 and r["body_sha256"] is None,
                    "missing body accounting")
        limit = (ANNOTATION["size"] + READ_UPPER if r["role"] == "annotation" else
                 2 * 1024 * 1024 if r["role"].startswith("image_metadata_page_") else IMAGE_BYTES)
        require(r["bytes"] + r["upper"] <= limit, "per-request body cap")
        started, finished = finite(r["started"], "started"), finite(r["finished"], "finished")
        require(claim["start_clock"] <= started <= finished <= clock(), "request clock order")
        require(not records or started >= records[-1]["finished"], "overlapping request clocks")
        require(finished - claim["start_clock"] <= NEW_CAPS["wall"] - AUDIT_SECONDS,
                "producer must reserve final 30s for consumer")
        integer(r["status_code"], "status_code")
        parent = r["redirect_from"]
        if parent is None:
            require(r["role"] not in logical, "logical asset retried")
            logical[r["role"]] = [index]
        else:
            integer(parent, "redirect_from")
            require(parent == index - 1 and records[parent]["status"] == "redirect" and
                    records[parent]["redirect_from"] is None and
                    records[parent]["role"] == r["role"], "at most one explicit redirect")
            location = lower_headers(records[parent]["headers"]).get("location")
            require(location and urljoin(records[parent]["url"], location) == r["url"],
                    "exact publisher Location followed once")
            logical[r["role"]].append(index)
        if r["status"] in ("complete", "redirect"):
            require(r["upper"] == 0 and r["bytes"] == r["retained_bytes"],
                    "complete body accounting")
            length = headers.get("content-length")
            require(length is None or integer(int(length), "Content-Length") == r["bytes"],
                    "complete Content-Length")
        if r["status"] == "redirect":
            require(r["status_code"] in (302, 303, 307, 308) and parent is None,
                    "only one official explicit redirect")
            require(r["role"] == "annotation" or r["role"].startswith("image_sample_"),
                    "metadata redirects forbidden; only publisher file assets may redirect")
            location = headers.get("location")
            require(location, "redirect Location missing")
            publisher_location(urljoin(r["url"], location))
        elif r["status"] == "complete":
            require(r["status_code"] == 200, "complete HTTP 200 required")
            mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if r["role"].startswith("image_sample_"):
                require(mime.startswith("image/") or mime == "application/octet-stream",
                        "image transport MIME")
            elif r["role"] == "annotation":
                require(mime in ("application/json", "application/octet-stream"),
                        "annotation JSON/octet-stream MIME with exact publisher SHA")
                body.read_bytes().decode("utf-8")
            else:
                require(mime == "application/json", "JSON metadata MIME")
                body.read_bytes().decode("utf-8")
        r = dict(r, index=index, body_path=str(body), receipt_sha256=digest(receipt_path))
        records.append(r)
    require(len(records) == len(directories), "every attempt billed")
    new_body, upper = sum(r["bytes"] for r in records), sum(r["upper"] for r in records)
    require(new_body + upper <= NEW_CAPS["body"] and
            PRIOR[0] + len(records) <= 32 and PRIOR[1] + new_body + upper <= 64 * 1024 * 1024,
            "new and cumulative request/body caps")
    return records, logical


def complete_role(records, logical, role, request_index):
    integer(request_index, role + " request_index")
    require(role in logical and logical[role][-1] == request_index and
            records[request_index]["status"] == "complete", "complete final role: " + role)
    return records[request_index]


def file_metadata(entry):
    require(isinstance(entry, dict), "official file entry object")
    filename, file_id = entry["filename"], entry["id"]
    require(isinstance(filename, str) and filename not in ("", ".", "..") and
            "/" not in filename and "\\" not in filename, "literal image filename")
    require(isinstance(file_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", file_id),
            "official opaque file ID")
    size = integer(entry["size"], "publisher size", 1)
    detail = entry["content_details"]
    require(isinstance(detail, dict), "publisher content_details")
    if "size" in detail:
        require(type(detail["size"]) is int and detail["size"] == size, "publisher size fields disagree")
    require(re.fullmatch(r"[0-9a-f]{64}", detail["sha256_hash"] or ""), "publisher SHA-256")
    url = detail["download_url"]
    u = safe_url(url, initial=True)
    require(u.path == f"/public-files/datasets/{DATASET_ID}/files/{file_id}/file_downloaded"
            and not u.query, "publisher dataset/file download URL")
    require(entry.get("status") == "COMPLETED" and entry.get("folder_id") == IMAGES_FOLDER_ID,
            "completed image file in exact official folder")
    return filename, file_id, size, detail["sha256_hash"], url


def metadata_and_coco(root, plan, index, records, logical):
    a = complete_role(records, logical, "annotation", index["annotation_request"])
    require(records[logical["annotation"][0]]["url"] == ANNOTATION["url"],
            "fixed annotation source")
    require(a["bytes"] == ANNOTATION["size"] and a["body_sha256"] == ANNOTATION["sha256"],
            "annotation publisher size/SHA")
    coco = load(a["body_path"])
    require(isinstance(coco, dict) and all(isinstance(coco.get(k), list) for k in
            ("images", "annotations", "categories")), "COCO native lists")
    pages = index["image_metadata_requests"]
    require(isinstance(pages, list) and 1 <= len(pages) <= 3, "one to three metadata pages")
    entries = []
    for page, request_index in enumerate(pages):
        r = complete_role(records, logical, "image_metadata_page_%d" % page, request_index)
        initial = records[logical[r["role"]][0]]["url"]
        u = safe_url(initial, initial=True)
        require(u.path == f"/public-api/datasets/{DATASET_ID}/files" and
                parse_qs(u.query, keep_blank_values=True) == {
                    "folder_id": [IMAGES_FOLDER_ID], "version": [str(VERSION)],
                    "$limit": ["1000"], "$start": [str(page * 1000)]},
                "published image list page parameters")
        obj = load(r["body_path"])
        data = obj.get("files") if isinstance(obj, dict) else obj
        require(isinstance(data, list) and len(data) <= 1000 and
                not (isinstance(obj, dict) and "error" in obj), "official list page schema")
        require(len(data) == 1000 if page < len(pages) - 1 else len(data) < 1000,
                "full prefix pages and terminal page below 1000")
        entries.extend(data)
    require(len(entries) == len(coco["images"]), "complete images list count equals COCO images")
    by_filename, file_ids = {}, set()
    for entry in entries:
        name, file_id, *_ = file_metadata(entry)
        require(name not in by_filename and file_id not in file_ids, "unique official filename/fileID")
        by_filename[name] = entry
        file_ids.add(file_id)
    return coco, by_filename


def expected_selection(coco, metadata):
    """Reconstruct selection from native GT; no inference or relabelling."""
    categories, canonical = {}, {}
    for category in coco["categories"]:
        key = identifier(category["id"], "category")
        require(key not in categories and isinstance(category["name"], str), "unique native categories")
        name = re.sub(r"[\s_-]", "", category["name"].casefold())
        mapped = {"helmet": "Helmet", "nohelmet": "NoHelmet"}.get(name)
        if mapped:
            require(mapped not in canonical, "multiple native categories normalize to same class")
            canonical[mapped] = key
        categories[key] = mapped
    require(set(canonical) == set(CLASS_ORDER), "Helmet/NoHelmet native category names required")
    images, names = {}, set()
    for image in coco["images"]:
        key = identifier(image["id"], "image")
        native_name = image["file_name"]
        require(isinstance(native_name, str) and "\\" not in native_name, "native POSIX file_name")
        relative_name = PurePosixPath(native_name)
        require(not relative_name.is_absolute() and ".." not in relative_name.parts and
                native_name not in ("", ".") and relative_name.name,
                "safe relative COCO file_name")
        name = relative_name.name
        require(key not in images and isinstance(name, str) and name not in names and name in metadata,
                "COCO image ID/filename one-to-one with official images")
        integer(image["width"], "COCO width", 1)
        integer(image["height"], "COCO height", 1)
        images[key] = image
        names.add(name)
    require(names == set(metadata), "complete COCO/official filename bijection")
    annotations, groups, eligible = set(), {}, []
    rejected = {"bbox_or_image_bounds": 0, "same_xywh_native_class_conflict": 0}
    for annotation in coco["annotations"]:
        key = identifier(annotation["id"], "annotation")
        image_key = identifier(annotation["image_id"], "annotation image")
        category_key = identifier(annotation["category_id"], "annotation category")
        require(key not in annotations and image_key in images and category_key in categories,
                "native annotation identities/references")
        annotations.add(key)
        bbox = annotation.get("bbox")
        usable = (isinstance(bbox, list) and len(bbox) == 4 and
                  all(type(v) in (int, float) and math.isfinite(v) for v in bbox))
        if usable:
            groups.setdefault((image_key, tuple(bbox)), set()).add(category_key)
        mapped = categories[category_key]
        if not mapped:
            continue
        image = images[image_key]
        meta = metadata[PurePosixPath(image["file_name"]).name]
        size = file_metadata(meta)[2]
        if usable:
            x, y, w, h = bbox
            usable = (x >= 0 and y >= 0 and w >= 32 and h >= 32 and
                      x + w <= image["width"] and y + h <= image["height"] and
                      image["width"] <= MAX_EDGE and image["height"] <= MAX_EDGE and
                      image["width"] * image["height"] <= MAX_PIXELS and size <= IMAGE_BYTES)
        if not usable:
            rejected["bbox_or_image_bounds"] += 1
            continue
        identity = f"{DATASET_ID}/{VERSION}|{meta['id']}|{annotation['id']}|{mapped}"
        eligible.append((hashlib.sha256(identity.encode()).hexdigest(), image_key,
                         mapped, meta, annotation, image))
    candidates = []
    for item in eligible:
        if len(groups[(item[1], tuple(item[4]["bbox"]))]) != 1:
            rejected["same_xywh_native_class_conflict"] += 1
        else:
            candidates.append(item)
    require(len({v[0] for v in candidates}) == len(candidates), "ambiguous deterministic candidate key")
    chosen, used_ids, used_names = [], set(), set()
    for category in CLASS_ORDER:
        count = 0
        for item in sorted((v for v in candidates if v[2] == category), key=lambda v: v[0]):
            _, image_id, _, meta, annotation, image = item
            if image_id in used_ids or meta["filename"] in used_names:
                continue
            chosen.append({"image_file_id": meta["id"], "image_filename": meta["filename"],
                           "image_metadata": meta, "annotation": annotation,
                           "category_name": category, "coco_image": image})
            used_ids.add(image_id)
            used_names.add(meta["filename"])
            count += 1
            if count == 3:
                break
        require(count == 3, "fewer than three independently imaged candidates for " + category)
    return chosen, rejected


def inspect_images(root, plan, index, records, logical, coco, metadata):
    selection_path = safe_file(root / "selection.json")
    frozen = load(root / "selection_claim.json")
    selection = load(selection_path)
    require(frozen["selection_sha256"] == digest(selection_path) and
            frozen["boot_uuid"] == boot_uuid(), "selection SHA/boot freeze")
    require(selection["selection_rule"] == SELECTION_RULE and plan["seed_schema"] == SELECTION_RULE,
            "predefined deterministic selection rule")
    freeze_clock = finite(frozen["frozen_clock"], "selection freeze clock")
    prefix = [r for r in records if not r["role"].startswith("image_sample_")]
    require(prefix and max(r["finished"] for r in prefix) <= freeze_clock,
            "selection frozen after complete annotation/metadata")
    expected, rejected = expected_selection(coco, metadata)
    selected = selection["selected"]
    require(isinstance(selected, list) and len(selected) == 6 and
            isinstance(index["selected"], list) and len(index["selected"]) == 6,
            "exactly six frozen samples")
    require(set(logical) == {"annotation"} | {
        "image_metadata_page_%d" % i for i in range(len(index["image_metadata_requests"]))
    } | {"image_sample_%d" % i for i in range(6)}, "exact logical asset inventory")
    from PIL import Image
    rows, sample_ids, actual_sha = [], set(), set()
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    for number, (target, actual, exp) in enumerate(zip(selected, index["selected"], expected)):
        required = ("sample_id", "image_file_id", "image_filename", "image_metadata",
                    "annotation", "category_name")
        require(set(target) == set(required), "frozen target has exact fields")
        require(isinstance(target["sample_id"], str) and
                re.fullmatch(r"[A-Za-z0-9_-]{1,80}", target["sample_id"]) and
                target["sample_id"] not in sample_ids, "unique sample_id")
        sample_ids.add(target["sample_id"])
        require(all(target[k] == exp[k] for k in required if k != "sample_id"),
                "actual target differs from deterministic native-GT selection")
        require(set(actual) in (set(required) | {"request_index"},
                                set(required) | {"request_index", "opaque_input_path"}) and
                all(actual[k] == target[k] for k in required), "index changed frozen target")
        role = "image_sample_%d" % number
        r = complete_role(records, logical, role, actual["request_index"])
        first = records[logical[role][0]]
        require(first["started"] > freeze_clock, "image GET preceded selection freeze")
        name, file_id, size, sha, url = file_metadata(target["image_metadata"])
        require(first["url"] == url and r["bytes"] == size and r["body_sha256"] == sha,
                "selected original publisher ID/URL/size/SHA")
        require(sha not in actual_sha, "duplicate actual image bytes across selected images")
        actual_sha.add(sha)
        body = safe_file(r["body_path"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(body) as image:
                width, height = image.size
                require(0 < width <= MAX_EDGE and 0 < height <= MAX_EDGE and
                        width * height <= MAX_PIXELS, "actual image geometry cap")
                image.verify()
            with Image.open(body) as image:
                require(image.size == (width, height), "image reopen dimensions changed")
                image.load()
        require((width, height) == (exp["coco_image"]["width"], exp["coco_image"]["height"]),
                "native GT and actual image dimensions differ")
        if "opaque_input_path" in actual:
            opaque = Path(actual["opaque_input_path"])
            if not opaque.is_absolute():
                opaque = root / opaque
            safe_file(opaque)
            require(opaque.resolve().is_relative_to(root.resolve()) and
                    (opaque.resolve() == body.resolve() or
                     re.fullmatch(r"[0-9a-f]{16,64}(?:\.[A-Za-z0-9]{1,8})?", opaque.name)),
                    "input copy must use an opaque, answer-free name")
            require(opaque.stat().st_size == size and digest(opaque) == sha,
                    "opaque original copy SHA/size")
        x, y, w, h = target["annotation"]["bbox"]
        rows.append({"sample_id": target["sample_id"], "category_name": target["category_name"],
                     "image_file_id": file_id, "image_filename": name,
                     "annotation_id": target["annotation"]["id"],
                     "image_id": target["annotation"]["image_id"],
                     "native_category_id": target["annotation"]["category_id"],
                     "x": x, "y": y, "width": w, "height": h,
                     "image_width": width, "image_height": height,
                     "image_bytes": size, "image_sha256": sha,
                     "request_index": actual["request_index"], "input_path": str(body),
                     "native_annotation": target["annotation"], "native_image": exp["coco_image"]})
    return rows, rejected


def conservative_bill(root):
    """Preserve attempts even when a malformed receipt prevents verification."""
    rows = []
    for folder in sorted(root.glob("request_*")):
        row = {"directory": folder.name, "verified": False}
        if folder.is_symlink() or not folder.is_dir():
            row.update(receipt_error="invalid attempt directory; never followed",
                       actual_retained_bytes=0)
            rows.append(row)
            continue
        try:
            r = load(folder / "receipt.json")
            for field in ("status", "status_code", "role", "bytes", "upper", "retained_bytes"):
                row[field] = r.get(field)
            row["receipt_sha256"] = digest(folder / "receipt.json")
        except (OSError, ValueError, TypeError, KeyError) as error:
            row["receipt_error"] = type(error).__name__ + ": " + str(error)
        body = folder / "body.bin"
        row["actual_retained_bytes"] = body.stat().st_size if body.is_file() and not body.is_symlink() else 0
        rows.append(row)
    reported_bytes = sum(r.get("bytes", 0) for r in rows if type(r.get("bytes")) is int and r["bytes"] >= 0)
    reported_upper = sum(r.get("upper", 0) for r in rows if type(r.get("upper")) is int and r["upper"] >= 0)
    return {"attempt_directory_count": len(rows), "attempts": rows,
            "actual_retained_body_bytes": sum(r["actual_retained_bytes"] for r in rows),
            "reported_new_body_bytes_unverified": reported_bytes,
            "reported_new_interrupted_upper_unverified": reported_upper,
            "cumulative_attempt_count_conservative": PRIOR[0] + len(rows),
            "unverified_new_body_upper_conservative": NEW_CAPS["body"],
            "fixed_parent_costs_not_readded_recursively": list(PRIOR)}


def audit(root, output):
    """Audit once; write a terminal audit even when qualification fails."""
    root, output = Path(root), Path(output)
    require(root.is_dir() and not root.is_symlink(), "existing raw root required")
    terminal, attempt = root / "terminal.json", root / "consumer_claim.json"
    require(not output.exists() and not terminal.exists() and not attempt.exists() and
            not (root / "consumer_failure.json").exists(), "one terminal consumer only; no rerun")
    require(output.parent.is_dir() and output.resolve() != attempt.resolve() and
            output.resolve() != terminal.resolve() and not output.is_symlink(), "new audit output path")
    started = clock()
    create(attempt, {"started_clock": started, "output": str(output.resolve()),
                     "consumer_sha256": digest(__file__), "single_use": True})
    original_alarm = signal.getitimer(signal.ITIMER_REAL)
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("consumer shared deadline / 30s cap")))
    result = {"reviewed_at": datetime.now(timezone.utc).isoformat(),
              "literature_cutoff": "not applicable; no literature search",
              "target_venue": "not a venue-readiness review",
              "artifact_run_id": root.name, "policy_version": "helmet_input_audit_v1_20261030",
              "evidence_level": "E0_UNAVAILABLE", "input_qualified": False,
              "qualified_scope": "Helmet6 only", "ALPR_qualified": False,
              "twelve_input_plan_qualified": False, "quality_improvement": "UNVERIFIED",
              "training_independence": "UNVERIFIED", "hidden_independence": "UNVERIFIED",
              "model_calls": 0, "training": 0, "inference": 0, "paper_goal_complete": False,
              "license_boundary": "Publisher page licence declaration only; underlying rights chain unverified.",
              "wire_total": "UNAVAILABLE; returned-body and interrupted-read upper bounds only",
              "prior_deadline_failure_unchanged": True, "audit_success": False,
              "requires_matching_terminal": True, "git_commit": None}
    produced = [attempt]
    try:
        signal.setitimer(signal.ITIMER_REAL, AUDIT_SECONDS - 2)
        plan, claim = load(root / "plan.json"), load(root / "claim.json")
        result["git_commit"] = plan.get("source_commit")
        require(plan["new_caps"] == NEW_CAPS and plan["annotation_expected"] == ANNOTATION and
                plan["images_folder_id"] == IMAGES_FOLDER_ID and
                plan["dataset_id"] == DATASET_ID and plan["version"] == VERSION,
                "fixed dataset/annotation/budget contract")
        require(claim["plan_sha"] == digest(root / "plan.json") and
                claim["boot_uuid"] == boot_uuid(), "plan SHA and current boot identity")
        grant_path = safe_file(root / "grant.json")
        grant = load(grant_path)
        require(digest(grant_path) == claim["grant_sha256"] and
                grant["plan_sha256"] == claim["plan_sha"] and
                grant["allow_six_image_downloads"] is True and
                grant["allow_one_publisher_location_hop"] is True and
                type(grant["model_calls"]) is int and grant["model_calls"] == 0,
                "frozen six-image/one-publisher-hop/zero-model grant")
        result["grant_sha256"] = digest(grant_path)
        elapsed = clock() - finite(claim["start_clock"], "claim start clock")
        require(0 <= elapsed < NEW_CAPS["wall"], "shared acquisition deadline expired")
        require(NEW_CAPS["wall"] - elapsed > 2, "two-second consumer terminal reserve")
        signal.setitimer(signal.ITIMER_REAL, min(AUDIT_SECONDS - 2, NEW_CAPS["wall"] - elapsed - 2))
        source_binding(plan)
        zero_http_provenance(plan, claim)
        prior = parent_binding(plan)
        storage, roots = stored_bytes(plan["storage_roots"])
        require(any(root.resolve().is_relative_to(p) for p in roots) and
                any(output.resolve().is_relative_to(p) for p in roots), "raw/output covered by storage roots")
        require(storage + OUTPUT_BYTES <= NEW_CAPS["storage"], "cumulative storage with consumer reserve")
        records, logical = receipt_inventory(root, plan, claim)
        result.update(new_requests=len(records), new_body_bytes=sum(r["bytes"] for r in records),
                      new_interrupted_upper=sum(r["upper"] for r in records),
                      cumulative_requests=prior[0] + len(records),
                      cumulative_body_bytes=prior[1] + sum(r["bytes"] for r in records),
                      cumulative_interrupted_upper=prior[2] + sum(r["upper"] for r in records),
                      receipt_sha256=[r["receipt_sha256"] for r in records],
                      request_failures=sum(r["status"] == "failed" for r in records),
                      storage_before_exports=storage, parent_audit_sha256=plan["parent_audit"]["sha256"])
        require(records and all(r["status"] != "failed" for r in records),
                "failed or empty acquisition prefix cannot qualify")
        index = load(root / "index.json")
        coco, metadata = metadata_and_coco(root, plan, index, records, logical)
        rows, rejected = inspect_images(root, plan, index, records, logical, coco, metadata)
        table = io.StringIO(newline="")
        fields = [key for key in rows[0] if key not in ("native_annotation", "native_image")]
        writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
        groundtruth = {"dataset_id": DATASET_ID, "version": VERSION,
                       "source_annotation_sha256": ANNOTATION["sha256"],
                       "selection_sha256": digest(root / "selection.json"),
                       "scope": "Helmet6 input qualification only; native GT, no model labels",
                       "samples": rows, "candidate_rejections": rejected}
        exports = {output.parent / "helmet_samples.csv": table.getvalue().encode(),
                   output.parent / "helmet_groundtruth.json": encoded(groundtruth)}
        require(sum(len(value) for value in exports.values()) + 65536 <= OUTPUT_BYTES,
                "consumer new output cap")
        for path, value in exports.items():
            create(path, value)
            produced.append(path)
        require(boot_uuid() == claim["boot_uuid"] and
                0 <= clock() - claim["start_clock"] <= NEW_CAPS["wall"] and
                clock() - started < AUDIT_SECONDS, "consumer completion deadline/boot")
        result.update(status="HELMET6_INPUT_QUALIFIED", input_qualified=True,
                      evidence_level="E2_ARTIFACT_AUDITED", images=6,
                      class_counts={name: 3 for name in CLASS_ORDER},
                      candidate_rejections=rejected, audit_success=True,
                      plan_sha256=digest(root / "plan.json"),
                      selection_sha256=digest(root / "selection.json"),
                      output_files={str(p): hashlib.sha256(v).hexdigest() for p, v in exports.items()})
    except BaseException as error:
        result.update(status="HELMET6_INPUT_UNQUALIFIED", input_qualified=False,
                      audit_success=False, error_type=type(error).__name__, error=str(error),
                      evidence_level="E0_UNAVAILABLE", failure_bill=conservative_bill(root))
        failure = root / "consumer_failure.json"
        create(failure, {"error_type": type(error).__name__, "error": str(error),
                         "input_qualified": False, "audit_success": False,
                         "failure_bill": result["failure_bill"], "consumer_must_not_rerun": True})
        produced.append(failure)
    finally:
        # Keep bounded space/time for terminal accounting after a failed alarm.
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if original_alarm[0]:
            signal.setitimer(signal.ITIMER_REAL, *original_alarm)
    result["consumer_elapsed_seconds"] = clock() - started
    try:
        claim_for_bill = load(root / "claim.json")
        result["elapsed_new_scope"] = clock() - claim_for_bill["start_clock"]
    except (OSError, ValueError, KeyError, TypeError):
        result["elapsed_new_scope"] = None
    require(sum(p.stat().st_size for p in produced if p.exists()) + len(encoded(result)) + 4096 <= OUTPUT_BYTES,
            "consumer output cap including failure bill")
    try:
        final_remaining = AUDIT_SECONDS - (clock() - started)
        if result["elapsed_new_scope"] is not None:
            final_remaining = min(final_remaining, NEW_CAPS["wall"] - result["elapsed_new_scope"])
        require(final_remaining > 0 or not result["input_qualified"], "terminal deadline expired")
        # An already expired acquisition may only publish its failure bill.
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError("consumer terminal publication deadline")))
        signal.setitimer(signal.ITIMER_REAL, max(0.001, min(2, final_remaining)))
        create(output, result)
        create(terminal, {"audit": str(output.resolve()), "sha256": digest(output),
                          "input_qualified": result["input_qualified"],
                          "audit_success": result["audit_success"], "consumer_must_not_rerun": True,
                          "finished_clock": clock()})
        require(clock() - started <= AUDIT_SECONDS, "consumer terminal exceeded 30s")
    except BaseException as error:
        failure = root / "consumer_failure.json"
        if not failure.exists():
            create(failure, {"error_type": type(error).__name__, "error": str(error),
                             "input_qualified": False, "audit_success": False,
                             "audit_terminal_pair_complete": False,
                             "failure_bill": conservative_bill(root), "consumer_must_not_rerun": True})
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
    print(json.dumps({"status": result["status"], "input_qualified": result["input_qualified"],
                      "audit_success": result["audit_success"]}, ensure_ascii=False))
    raise SystemExit(0 if result["audit_success"] else 1)


if __name__ == "__main__":
    main()
