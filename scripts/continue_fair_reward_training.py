"""Operational continuation of an immutable scientific plan; no agent changes.

Completed parent cells are referenced read-only. Every remaining cell starts
fresh in a new root, including any interrupted cell. The original pinned runner,
auditor and plotter remain byte-identical. This launcher replaces only resource
scanning and cell dispatch, with a separately frozen operational contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def foreign_training(raw, own_pid):
    # Preserve undecodable bytes as visible escapes, rather than aborting or
    # discarding an entire process line. Only operational metadata sees this.
    text = raw.decode("utf-8", errors="backslashreplace") if isinstance(raw, bytes) else raw
    others = []
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        pid, comm, args = fields
        if int(pid) == own_pid or "python" not in Path(comm).name.lower():
            continue
        try:
            tokens = shlex.split(args)
        except ValueError:
            # ps arguments are not guaranteed to retain shell quoting. A bad
            # quote in a later argument must not hide an earlier train script.
            tokens = args.split()
        scripts = [Path(t.strip("\"'")).name.lower() for t in tokens[1:] if t.strip("\"'").endswith(".py")]
        if any(any(word in script for word in ("train", "benchmark", "closed_loop")) for script in scripts):
            others.append({"pid": int(pid), "command": args})
    return others


def yield_resources(root):
    root = Path(root)
    while True:
        try:
            raw = subprocess.check_output(["ps", "-axo", "pid=,comm=,args="])
            others = foreign_training(raw, os.getpid())
        except (OSError, subprocess.CalledProcessError) as error:
            # Fail closed: a transient ps failure means wait and retry, never
            # assume there is no competing training or restart a scientific cell.
            write(root / "resource_wait.json", {"status": "RESOURCE_SCAN_UNAVAILABLE", "observed_at": time.time(),
                                              "error_type": type(error).__name__})
            time.sleep(5)
            continue
        if not others:
            if (root / "resource_wait.json").exists():
                write(root / "resource_wait.json", {"status": "RESOURCE_AVAILABLE", "observed_at": time.time()})
            return
        write(root / "resource_wait.json", {"status": "YIELDING_TO_OTHER_TRAINING", "observed_at": time.time(), "processes": others})
        time.sleep(5)


def load_frozen(source):
    sys.path.insert(0, str(source / "scripts"))
    sys.path.insert(0, str(source))
    long = importlib.import_module("run_long_fair_reward_training")
    if Path(long.__file__).resolve() != source / "scripts/run_long_fair_reward_training.py":
        raise ValueError("Wrong scientific runner imported")
    return long


def verify_plan(p):
    copy = dict(p)
    expected = copy.pop("plan_sha256")
    if semantic(copy) != expected:
        raise ValueError("Scientific plan changed")
    source = Path(p["source_root"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != p["source_commit"]:
        raise ValueError("Original frozen source HEAD changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], cwd=source, check=True)
    for name, key in [("run_long_fair_reward_training.py", "runner_sha256"),
                      ("run_fair_reward_training.py", "base_runner_sha256"),
                      ("diagnose_sagh_flat_reward.py", "diagnostic_observer_sha256")]:
        if digest(source / "scripts" / name) != p[key]:
            raise ValueError("Original scientific runner changed")
    protected = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *p["protected_files"]], cwd=p["project_root"])
    if hashlib.sha256(protected).hexdigest() != p["protected_diff_sha256"]:
        raise ValueError("Protected user changes differ; never reset them")


def inventory(path):
    return [{"path": str(f.relative_to(path)), "bytes": f.stat().st_size, "sha256": digest(f)}
            for f in sorted(path.rglob("*")) if f.is_file()]


def verify_inventory(path, items):
    actual = {str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()}
    if actual != {item["path"] for item in items}:
        raise ValueError(f"Parent file inventory changed: {path}")
    for item in items:
        f = path / item["path"]
        if f.stat().st_size != item["bytes"] or digest(f) != item["sha256"]:
            raise ValueError(f"Preserved file changed: {f}")


def check_checkpoints(fair, cell, p, name):
    updates = [0, *range(p["evaluate_every_updates"], p["training_steps"]//p["steps_per_update"]+1, p["evaluate_every_updates"])]
    for update in updates:
        f = cell / ("initial.pt" if update == 0 else f"update_{update:04d}.pt")
        cp = fair.torch.load(f, map_location="cpu", weights_only=False)
        if cp["agent_name"] != name or cp["update_count"] != update:
            raise ValueError("Inherited checkpoint identity/update differs")
        if update == 0 and cp["optimizer_state_dict"]["state"]:
            raise ValueError("Inherited initialization optimizer was not empty")
        if any(not fair.torch.isfinite(v).all().item() for v in cp["network_state_dict"].values()):
            raise ValueError("Inherited checkpoint is non-finite")


def prepare(parent, root, failure_report=None):
    if root.exists():
        raise FileExistsError("Continuation is create-only")
    p = read(parent / "frozen_plan.json")
    verify_plan(p)
    failure_evidence = None
    if failure_report is not None:
        report = read(failure_report)
        if report["artifact_run_id"] != parent.name or report["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Failure evidence belongs to a different run")
        for name, value in report["evidence_hashes"].items():
            if digest(parent / name) != value:
                raise ValueError("Original failure evidence changed")
        for entry in report["completed_cells"]:
            if digest(parent / entry["cell"] / "complete.json") != entry["complete_json_sha256"]:
                raise ValueError("Completed cell changed since failure diagnosis")
        partial_cell = parent / "cells" / report["partial_cell"]["agent"] / f"seed{report['partial_cell']['seed']}"
        for name, value in report["partial_cell"]["partial_payload_hashes"].items():
            if digest(partial_cell / name) != value:
                raise ValueError("Interrupted payload changed since failure diagnosis")
        for entry in report["partial_cell"]["checkpoint_inventory"]:
            if digest(partial_cell / entry["name"]) != entry["sha256"]:
                raise ValueError("Interrupted checkpoint changed since failure diagnosis")
        failure_evidence = {"path": str(failure_report), "sha256": digest(failure_report)}
    launcher_root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "scripts", "src"], cwd=launcher_root, check=True)
    launcher_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=launcher_root, text=True).strip()
    long = load_frozen(Path(p["source_root"]))
    long.fair.load_api(Path(p["source_root"]))
    from plot_long_fair_reward_training import validate_cell
    inherited, pending, partial = [], [], []
    for seed in p["seeds"]:
        for name in p["agents"]:
            cell = parent / "cells" / name / f"seed{seed}"
            if (cell / "complete.json").exists():
                validate_cell(cell, p, name, seed)
                identity = read(cell / "identity.json")
                if identity["agent"] != name or identity["seed"] != seed or not identity["fresh_initialization"] or identity["replay_initial_size"] != 0 or identity["plan_sha256"] != p["plan_sha256"]:
                    raise ValueError("Inherited fresh initialization differs")
                check_checkpoints(long.fair, cell, p, name)
                inherited.append({"agent": name, "seed": seed, "path": str(cell), "inventory": inventory(cell)})
                print(f"INHERITED_VALIDATED {name} seed={seed}", flush=True)
            else:
                pending.append({"agent": name, "seed": seed, "initialization": "fresh_no_checkpoint_load"})
                if cell.exists():
                    partial.append({"path": str(cell), "inventory": inventory(cell), "excluded_from_accepted_curves": True})
    if not pending:
        raise ValueError("No pending scientific cells")
    contract = {"version": "fair_reward_operational_continuation_v1", "created_at": datetime.now().astimezone().isoformat(),
                "parent_root": str(parent), "output_root": str(root), "scientific_plan_sha256": p["plan_sha256"],
                "frozen_plan_file_sha256": digest(parent / "frozen_plan.json"), "launcher_root": str(launcher_root),
                "launcher_commit": launcher_commit, "launcher_sha256": digest(__file__), "inherited": inherited,
                "pending": pending, "preserved_partial": partial, "failure_evidence": failure_evidence,
                "parent_evidence": {name: digest(parent / name) for name in ("command.log", "execution_receipt.json") if (parent / name).exists()},
                "scientific_change": "NONE; same original source, config, reward, seeds, windows, interaction budget and plot rules",
                "retry_policy": "Interrupted cells restart fresh in new root; old partial data are excluded and retained; no result-based retry"}
    contract["contract_sha256"] = semantic(contract)
    root.mkdir(parents=True)
    (root / "frozen_plan.json").write_bytes((parent / "frozen_plan.json").read_bytes())
    for entry in inherited:
        target = root / "cells" / entry["agent"] / f"seed{entry['seed']}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(entry["path"], target_is_directory=True)
    write(root / "continuation_contract.json", contract)
    write(root / "status.json", {"status": "CONTINUATION_PREPARED", "inherited_cells": len(inherited), "pending_cells": len(pending),
                                 "plan_sha256": p["plan_sha256"], "contract_sha256": contract["contract_sha256"]})
    print(json.dumps({"status": "CONTINUATION_PREPARED", "inherited": len(inherited), "fresh_pending": len(pending), "contract_sha256": contract["contract_sha256"]}), flush=True)


def verify(root):
    c = read(root / "continuation_contract.json")
    copy = dict(c)
    expected = copy.pop("contract_sha256")
    if semantic(copy) != expected or c["output_root"] != str(root) or digest(__file__) != c["launcher_sha256"]:
        raise ValueError("Operational contract/launcher changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c["launcher_root"], text=True).strip() != c["launcher_commit"]:
        raise ValueError("Launcher HEAD changed")
    p = read(root / "frozen_plan.json")
    verify_plan(p)
    if c.get("failure_evidence") and digest(c["failure_evidence"]["path"]) != c["failure_evidence"]["sha256"]:
        raise ValueError("Independent failure report changed")
    parent = Path(c["parent_root"])
    if digest(root / "frozen_plan.json") != c["frozen_plan_file_sha256"] or digest(parent / "frozen_plan.json") != c["frozen_plan_file_sha256"]:
        raise ValueError("Scientific plan file changed")
    for name, value in c["parent_evidence"].items():
        if digest(parent / name) != value:
            raise ValueError("Original failure/launch evidence changed")
    for entry in c["inherited"]:
        target = root / "cells" / entry["agent"] / f"seed{entry['seed']}"
        if not target.is_symlink() or target.resolve() != Path(entry["path"]):
            raise ValueError("Inherited cell must reference the exact original directory")
        verify_inventory(Path(entry["path"]), entry["inventory"])
    for entry in c["preserved_partial"]:
        verify_inventory(Path(entry["path"]), entry["inventory"])
    return c, p


def dispatch(original, contract, source, root, p, name, seed, train, validation, workflows):
    inherited = {(e["agent"], e["seed"]): e for e in contract["inherited"]}
    if (name, seed) in inherited:
        print(f"REUSE_COMPLETED_READ_ONLY {name} seed={seed}", flush=True)
        return
    if (name, seed) not in {(e["agent"], e["seed"]) for e in contract["pending"]}:
        raise ValueError("Unfrozen pending cell")
    target = root / "cells" / name / f"seed{seed}"
    if target.exists():
        raise FileExistsError("No restart or overwrite within the continuation root")
    attempt = root / "attempt_receipts" / f"{name}_seed{seed}.json"
    attempt.parent.mkdir(exist_ok=True)
    with attempt.open("x", encoding="utf-8") as stream:
        json.dump({"agent": name, "seed": seed, "contract_sha256": contract["contract_sha256"],
                   "plan_sha256": p["plan_sha256"], "fresh_start": True, "checkpoint_load": False}, stream)
    return original(source, root, p, name, seed, train, validation, workflows)


def execute(root):
    contract, p = verify(root)
    # An exited run retains this lock: a further scientific retry needs a new root.
    with (root / "execution_lock.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "contract_sha256": contract["contract_sha256"]}, stream)
    long = load_frozen(Path(p["source_root"]))
    long.fair.load_api(Path(p["source_root"]))
    long.fair.yield_resources = yield_resources
    original = long.run_cell
    long.run_cell = lambda *args: dispatch(original, contract, *args)
    # execute is the byte-identical original scientific scheduler and trainer.
    long.execute(Path(p["source_root"]), root)
    verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "verify"])
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--failure-report", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    os.nice(max(0, 19-os.nice(0)))
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    root = args.output_root.resolve()
    if args.mode == "prepare":
        if args.parent_root is None:
            parser.error("--parent-root required for prepare")
        prepare(args.parent_root.resolve(), root, args.failure_report.resolve() if args.failure_report else None)
    elif args.mode == "run":
        execute(root)
    else:
        c, p = verify(root)
        print(json.dumps({"status": "OPERATIONAL_CONTRACT_VERIFIED", "inherited": len(c["inherited"]),
                          "pending": len(c["pending"]), "plan_sha256": p["plan_sha256"]}))


if __name__ == "__main__":
    main()
