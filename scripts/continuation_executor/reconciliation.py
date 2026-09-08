"""Read-only approved-prefix, native-ledger and committed-payload reconciliation."""
import hashlib
import json

from . import PHASES
from .identity import ContinuationError, absolute_path, verify_file, file_hash, within, digest, strict_json
from .planning import cell_input_hash
from .cells import cell_layout
import importlib


def verify_prefix(anchor):
    data = absolute_path(anchor["path"]).read_bytes()
    n = anchor["byte_count"]
    if len(data) < n or hashlib.sha256(data[:n]).hexdigest() != anchor["prefix_sha256"]:
        raise ContinuationError("approved ledger prefix changed/truncated")
    prefix = data[:n].decode().splitlines()
    if len(prefix) != anchor["record_count"] or (data[:n] and not data[:n].endswith(b"\n")):
        raise ContinuationError("ledger prefix record boundary")
    rows = [strict_json(line) for line in data.decode().splitlines()]
    key = "current_record_hash" if anchor["kind"] == "phase" else "current_ledger_hash"
    if not prefix or rows[len(prefix)-1][key] != anchor["terminal_hash"]:
        raise ContinuationError("prefix terminal identity")
    if any(row["run_identity_fingerprint"] != anchor["run_identity_fingerprint"] for row in rows):
        raise ContinuationError("cross-run ledger successor")
    return rows, rows[len(prefix):]


def reconcile(contract, phase_native, cell_ledger, plans, *, protocol, context, registry_sha256):
    ledgers = {}
    successors = {}
    for anchor in contract["prefixes"]:
        kind = anchor["kind"]
        if kind in ledgers:
            raise ContinuationError("duplicate ledger anchor")
        ledgers[kind], successors[kind] = verify_prefix(anchor)
    if set(ledgers) != {"phase", "cell"}:
        raise ContinuationError("both ledger anchors required")
    phase_native.validate_phase_ledger_v3(ledgers["phase"])
    cell_rows = cell_ledger.records()  # actual native validator and identity
    if cell_rows != ledgers["cell"]:
        raise ContinuationError("cross-ledger read disagreement")
    if any(row["status"] == "failed" for row in ledgers["phase"]):
        raise ContinuationError("failed phase terminal cannot resume")
    if any(row["status"] == "failed_terminal" for row in cell_rows):
        raise ContinuationError("non-retryable cell cannot resume")
    completed = {row["phase"] for row in ledgers["phase"] if row["status"] == "completed"}
    if not {"train", "dev_select", "checkpoint_freeze"} <= completed:
        raise ContinuationError("original train/dev/freeze must already be complete")
    seen_completed = set()
    for row in successors["phase"]:
        phase = row["phase"]
        if phase not in PHASES or not set(PHASES[:PHASES.index(phase)]) <= seen_completed:
            raise ContinuationError("out-of-order or unauthorized phase successor")
        plan = plans[phase]
        if row["input_hash"] != plan["input_hash"] or row["commands"] != plan["commands"]:
            raise ContinuationError("phase input/command identity drift")
        if row["status"] == "completed":
            seen_completed.add(phase)
    cell_native = importlib.import_module(cell_ledger.__class__.__module__)
    for phase in seen_completed:
        if phase in PHASES[:5]:
            cell_ledger.assert_complete_matrix(phase=phase, expected_cell_ids=[
                cell_native.stable_cell_id(phase, coord) for coord in plans[phase]["matrix_contexts"]])
    for row in ledgers["phase"]:
        if row["status"] in {"completed", "completion_candidate"}:
            for relative, expected in row["output_files"].items():
                path = within(str(absolute_path(contract["run_root"]) / relative), contract["run_root"])
                if file_hash(path) != expected:
                    raise ContinuationError("immutable phase output drift")
    for row in successors["phase"]:
        if row["status"] in {"completed", "completion_candidate"}:
            expected=set()
            root=absolute_path(contract["run_root"])
            for pattern in plans[row["phase"]]["expected_outputs"]:
                if pattern.startswith("/") or ".." in pattern.split("/"):
                    raise ContinuationError("phase output pattern escape")
                matches={path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file()}
                if not matches:raise ContinuationError("phase expected output missing")
                expected.update(matches)
            if set(row["output_files"])!=expected:
                raise ContinuationError("phase terminal output coverage drift")
    phase_started = {row["phase"] for row in successors["phase"]}
    cell_phase_index=-1
    for row in successors["cell"]:
        if row["phase"] not in PHASES[:5] or row["phase"] not in phase_started:
            raise ContinuationError("cell successor lacks authorized running phase")
        index=PHASES.index(row["phase"])
        if index<cell_phase_index:raise ContinuationError("out-of-order cell successor")
        cell_phase_index=index
        plan = plans[row["phase"]]
        matches = [index for index, coord in enumerate(plan["matrix_contexts"]) if coord == row["coordinates"]]
        if len(matches) != 1:
            raise ContinuationError("unauthorized cell coordinates")
        argv = plan["commands"][matches[0]]
        expected_input = cell_input_hash(row["phase"], row["coordinates"], argv, protocol, context, registry_sha256)
        final, _, _ = cell_layout(row["phase"], row["coordinates"], argv, cell_native)
        if (row["command_hash"] != digest(argv) or row["input_hash"] != expected_input
                or row["committed_path"] != str(final)
                or row["cell_id"] != cell_native.stable_cell_id(row["phase"], row["coordinates"])):
            raise ContinuationError("cell command/input/publication identity drift")
    committed = [row for row in cell_rows if row["status"] == "committed"]
    if len({row["cell_id"] for row in committed}) != len(committed):
        raise ContinuationError("duplicate committed cell")
    for row in committed:
        cell_ledger.verify_committed(row["cell_id"])
    for row in contract["immutable_files"]:
        verify_file(row)
    return {"phase_records": len(ledgers["phase"]), "cell_records": len(cell_rows),
            "committed_payloads_verified": len(committed), "immutable_files_verified": len(contract["immutable_files"])}
