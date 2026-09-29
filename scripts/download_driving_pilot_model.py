"""Download the explicitly approved, pinned pilot model; no inference or images."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = "HuggingFaceTB/SmolVLM-500M-Instruct"
REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"
WEIGHT_SHA = "d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2"
FILES = ("README.md", "added_tokens.json", "chat_template.json", "config.json",
         "generation_config.json", "merges.txt", "preprocessor_config.json",
         "processor_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "model.safetensors")


def validate_inventory(rows):
    by_name = {r["path"]: r for r in rows if r["type"] == "file"}
    if len(by_name) != len([r for r in rows if r["type"] == "file"]):
        raise ValueError("duplicate inventory")
    chosen = [by_name[n] for n in FILES]
    if sum(r["size"] for r in chosen) > 1_030_000_000:
        raise ValueError("approved size budget exceeded")
    weight = by_name["model.safetensors"]
    if weight["size"] != 1015025832 or weight["lfs"]["oid"] != WEIGHT_SHA:
        raise ValueError("weight identity drift")
    return chosen


def download(root):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}", timeout=30) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("metadata too large")
    selected = validate_inventory(json.loads(raw))
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"repository": REPO, "revision": REVISION, "files": [], "status": "running",
               "model_calls": 0, "image_downloads": 0, "started_at_unix": time.time()}
    started = time.monotonic()
    try:
        for item in selected:
            name, size = item["path"], item["size"]
            part = root / (name + ".partial")
            sha = hashlib.sha256()
            git = hashlib.sha1(f"blob {size}\0".encode())
            count = 0
            with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=60) as response, part.open("xb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > size:
                        raise ValueError(f"size overflow: {name}")
                    output.write(chunk)
                    sha.update(chunk)
                    git.update(chunk)
            expected, actual = (item["lfs"]["oid"], sha.hexdigest()) if item.get("lfs") else (item["oid"], git.hexdigest())
            if count != size or actual != expected:
                raise ValueError(f"size/hash mismatch: {name}")
            part.rename(root / name)
            receipt["files"].append({"path": name, "bytes": count, "sha256": sha.hexdigest()})
            print(f"verified {name}: {count} bytes", flush=True)
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["total_verified_bytes"] = sum(r["bytes"] for r in receipt["files"])
        with (root / "download_receipt.json").open("x") as output:
            json.dump(receipt, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    download(args.output)
