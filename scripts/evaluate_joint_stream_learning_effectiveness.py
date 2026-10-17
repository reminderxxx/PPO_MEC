"""Frozen development-only 0/32/64 learning audit; never train or resume training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_joint_stream_control import NAMES, actor_hash, make_agent, make_env, stream_episode
from scripts.train_joint_tail_control import tensor_hash

SEEDS = [7, 17, 27]
POINTS = [0, 32, 64]
FROZEN_SOURCE = "35975cb3d73c84ddbabf01f6749bbc2d60c826de"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def key(row):
    return row["agent"], row["seed"], row["episode"], row["case_id"]


def verify_source(run):
    manifest = json.loads((run / "protocol/manifest.json").read_text())
    assert manifest["protocol"] == "joint_stream_learning_v2"
    assert manifest["action_contract"] == "joint_stream_control_v2"
    assert not manifest["smoke"] and manifest["episodes"] == 64
    assert manifest["seeds"] == SEEDS and manifest["checkpoint_episodes"] == [32, 64]
    for path, expected in manifest["source_fingerprints"].items():
        assert digest(ROOT / path) == expected, ("source mismatch", path)
    for path, expected in manifest["assets"].items():
        assert digest(run / "protocol" / path) == expected, ("asset mismatch", path)
    audited = json.loads((run / "audit.json").read_text())
    assert audited["status"] == "pass"
    assert json.loads((run / "launcher_result.json").read_text())["status"] == "complete"
    data = torch.load(run / "protocol/protocol.pt", weights_only=False)
    assert len(data["cases"]["dev"]) == 48
    return manifest, data


def verify_group(run, name):
    home = run / name
    manifest = json.loads((home / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["source_commit"] == FROZEN_SOURCE
    assert manifest["protocol_sha256"] == digest(run / "protocol/manifest.json")
    assert manifest["output_sha256"]["dev.jsonl"] == digest(home / "dev.jsonl")
    assert len(manifest["checkpoints"]) == 6
    for item in manifest["checkpoints"]:
        assert digest(home / item["path"]) == item["sha256"]
    return manifest


def verify_contract(output):
    contract = json.loads((output / "contract.json").read_text())
    assert contract["methods"] == NAMES and contract["seeds"] == SEEDS
    assert contract["checkpoints"] == POINTS and contract["new_training_sessions"] == 0
    assert contract["new_cells"] == 2016 and contract["reused_cells"] == 1440
    assert contract["workers"] in (1, 2)
    assert contract["max_new_evaluation_frames"] == 967680
    assert contract["evaluator_sha256"] == digest(__file__)
    run = Path(contract["source_run"])
    assert digest(run / "protocol/manifest.json") == contract["source_protocol_sha256"]
    original = {}
    for name in NAMES:
        frozen = contract["provenance"][name]
        assert digest(run / name / "manifest.json") == frozen["manifest_sha256"]
        assert digest(run / name / "dev.jsonl") == frozen["dev_sha256"]
        manifest = verify_group(run, name)
        assert manifest["checkpoints"] == frozen["checkpoints"]
        for row in read_rows(run / name / "dev.jsonl"):
            assert key(row) not in original
            original[key(row)] = row
    assert digest(output / "reused_rows.jsonl") == contract["reused_rows_sha256"]
    reused = read_rows(output / "reused_rows.jsonl")
    assert len(reused) == len(original) == 1440
    seen = set()
    for row in reused:
        assert row["evaluation_origin"] == "immutable_original"
        raw = {k: v for k, v in row.items() if k != "evaluation_origin"}
        assert key(raw) not in seen and raw == original[key(raw)]
        seen.add(key(raw))
    return contract


def assert_initial(agent, manifest, seed):
    matches = [c for c in manifest["checkpoints"] if c["seed"] == seed]
    assert len(matches) == 2
    assert {c["initial_tensor_sha256"] for c in matches} == {tensor_hash(agent)}
    assert {c["initial_actor_sha256"] for c in matches} == {actor_hash(agent)}


def forbidden_update(*args, **kwargs):
    raise RuntimeError("This evaluation contract forbids any learning/update call")


class DecisionRecorder:
    """Record selected controls without changing any input, RNG or controller memory."""

    def __init__(self, agent, output, identity):
        self.agent, self.output, self.identity = agent, output, identity
        self.hash = hashlib.sha256()
        self.frames = 0

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def act_controls(self, decisions, frame, *, deterministic=False):
        assert deterministic
        controls, rollout = self.agent.act_controls(decisions, frame, deterministic=True)
        record = dict(self.identity, frame_index=frame, controls={
            vehicle: dict(control=asdict(control),
                          current_node_id=decisions[vehicle]["semantic_state"]["current_workflow_node"]["node_id"])
            for vehicle, control in sorted(controls.items())})
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        self.output.write(line)
        self.hash.update(line.encode())
        self.frames += 1
        return controls, rollout


def qualify(run, output):
    torch.set_num_threads(1)
    protocol, data = verify_source(run)
    reconstructed = []
    for name in NAMES:
        manifest = verify_group(run, name)
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            agent = make_agent(name, seed, 64)
            assert_initial(agent, manifest, seed)
            reconstructed.append(dict(agent=name, seed=seed, tensor_sha256=tensor_hash(agent),
                                      actor_sha256=actor_hash(agent)))
    name, seed = "sa_ghmappo", 7
    agent = type(make_agent(name, seed, 64)).from_checkpoint(run / name / f"{name}_seed7_ep32.pt")
    agent.update = forbidden_update
    before = tensor_hash(agent), actor_hash(agent)
    started = time.monotonic()
    # This fixed consumer-parity check is not used to choose budget or cases.
    import io
    trace = io.StringIO()
    identity = dict(agent=name, seed=seed, episode=32, case_id="dev_000")
    recorded = DecisionRecorder(agent, trace, identity)
    row, _ = stream_episode(make_env(data["cases"]["dev"][0], data, protocol), recorded,
                            deterministic=True)
    old = next(r for r in read_rows(run / name / "dev.jsonl") if key(r) == key(identity))
    assert all(old[k] == v for k, v in row.items()), "Original consumer parity failed"
    assert before == (tensor_hash(agent), actor_hash(agent)), "Evaluation changed weights"
    elapsed = time.monotonic() - started
    value = dict(status="pass", checked_at=now(), initial_agents=reconstructed,
                 source_fingerprints_verified=len(protocol["source_fingerprints"]),
                 parity=dict(identity, exact_fields=len(row), steps=row["steps"],
                             decision_trace_frames=recorded.frames, wall_seconds=elapsed),
                 new_training_sessions=0, optimizer_steps=0,
                 evaluator_sha256=digest(__file__), source_run=str(run))
    write_json(output, value)
    return value


def prepare(run, output, qualification, workers):
    assert workers in (1, 2)
    q = json.loads(qualification.read_text())
    assert q["status"] == "pass" and q["evaluator_sha256"] == digest(__file__)
    assert q["source_run"] == str(run)
    protocol, data = verify_source(run)
    cases = data["cases"]["dev"]
    panel_ids = [c["case_id"] for c in cases]
    probes = protocol["curve_probe_ids"]
    assert len(set(panel_ids)) == 48 and len(set(probes)) == 12
    expected_reuse = {(n, s, e, c) for n in NAMES for s in SEEDS for e in (32, 64)
                      for c in (probes if e == 32 else panel_ids)}
    old_rows, provenance = [], {}
    for name in NAMES:
        m = verify_group(run, name)
        rows = read_rows(run / name / "dev.jsonl")
        old_rows.extend(rows)
        provenance[name] = dict(manifest_sha256=digest(run / name / "manifest.json"),
                                dev_sha256=digest(run / name / "dev.jsonl"), checkpoints=m["checkpoints"])
    assert len(old_rows) == len(expected_reuse) == 1440
    assert {key(r) for r in old_rows} == expected_reuse
    max_frames = sum(len(data["sessions"][c["session"]]["frames"])
                     for c in cases if c["case_id"] not in probes)
    max_frames += sum(len(data["sessions"][c["session"]]["frames"]) for c in cases)
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(protocol="joint_stream_learning_effectiveness_v1", created_at=now(),
                    source_run=str(run), frozen_algorithm_source=FROZEN_SOURCE,
                    evaluation_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    evaluator_sha256=digest(__file__), qualification_sha256=digest(qualification),
                    source_protocol_sha256=digest(run / "protocol/manifest.json"),
                    methods=NAMES, seeds=SEEDS, checkpoints=POINTS, panel_ids=panel_ids,
                    reuse_32_ids=probes, reused_cells=1440, new_cells=2016,
                    max_new_evaluation_frames=max_frames * len(NAMES) * len(SEEDS),
                    qualification_sessions=1, max_qualification_frames=480, workers=workers,
                    total_comparison_cells=3456, new_training_sessions=0, optimizer_steps=0,
                    prediction_shared_state_reward_physical_costs="unchanged",
                    checkpoint_selection="all0/32/64; final64 remains official; never select a peak",
                    case_metadata={c["case_id"]: {k: c[k] for k in
                        ("session", "size", "arrival_mode", "families", "capacity")} for c in cases},
                    provenance=provenance, formal_holdout="sealed", canonical="unchanged",
                    analysis="paired0to32/32to64/0to64; allseeds and full48 plus strata; preserve all negatives",
                    claim_limits=["already consumed development panel", "only two time sessions",
                                  "three snapshots do not establish convergence",
                                  "initialization includes common priors/masks/frozenGRU, not random actions",
                                  "no isolated graph/surrogate/prediction causal attribution"],
                    failure_policy="preserve partial output; no automatic relaunch or extra training")
    assert contract["max_new_evaluation_frames"] == 967680
    write_json(output / "qualification.json", q)
    with (output / "reused_rows.jsonl").open("x") as f:
        for r in old_rows:
            f.write(json.dumps(dict(r, evaluation_origin="immutable_original")) + "\n")
    contract["reused_rows_sha256"] = digest(output / "reused_rows.jsonl")
    write_json(output / "contract.json", contract)
    write_json(output / "progress.json", dict(status="prepared", created_at=now(),
                                               new_cells_completed=0, new_cells=2016, training_sessions=0))
    return contract


def worker(run_string, output_string, name, seed):
    run, output = Path(run_string), Path(output_string)
    torch.set_num_threads(1)
    protocol, data = verify_source(run)
    manifest = verify_group(run, name)
    group = output / f"{name}_seed{seed}"
    if (output / "failure.json").exists():
        raise RuntimeError("Sibling evaluation failed; do not start another group")
    group.mkdir(exist_ok=False)
    count, frames, elapsed = 0, 0, time.monotonic()
    try:
        with (group / "rows.jsonl").open("x") as rows, gzip.open(group / "controls.jsonl.gz", "xt", compresslevel=1) as traces:
            for episode in (0, 32):
                random.seed(seed)
                np.random.seed(seed)
                agent = make_agent(name, seed, 64)
                assert_initial(agent, manifest, seed)
                if episode == 32:
                    checkpoint = next(c for c in manifest["checkpoints"] if c["seed"] == seed and c["episode"] == 32)
                    agent = type(agent).from_checkpoint(run / name / checkpoint["path"])
                    assert tensor_hash(agent) == checkpoint["trained_tensor_sha256"]
                    assert actor_hash(agent) == checkpoint["trained_actor_sha256"]
                before = tensor_hash(agent), actor_hash(agent)
                agent.update = forbidden_update
                for case in data["cases"]["dev"]:
                    if (output / "failure.json").exists():
                        raise RuntimeError("Sibling evaluation failed; stop before next case")
                    if episode == 32 and case["case_id"] in protocol["curve_probe_ids"]:
                        continue
                    identity = dict(agent=name, seed=seed, episode=episode, case_id=case["case_id"])
                    recorder = DecisionRecorder(agent, traces, identity)
                    started = time.monotonic()
                    row, _ = stream_episode(make_env(case, data, protocol), recorder, deterministic=True)
                    assert before == (tensor_hash(agent), actor_hash(agent)), "Evaluation changed parameters"
                    assert row["steps"] <= 480
                    assert math.isfinite(row["reward"])
                    result = dict(identity, **row, evaluation_origin="fresh_deterministic",
                                  tensor_sha256=before[0], actor_sha256=before[1],
                                  control_trace_sha256=recorder.hash.hexdigest(),
                                  wall_seconds=time.monotonic() - started)
                    rows.write(json.dumps(result, allow_nan=False) + "\n")
                    rows.flush()
                    traces.flush()
                    count += 1
                    frames += row["steps"]
                    write_json(group / "progress.json", dict(status="running", updated_at=now(),
                        agent=name, seed=seed, episode=episode, case_id=case["case_id"],
                        new_cells_completed=count, new_cells=84, evaluation_frames=frames,
                        wall_seconds=time.monotonic() - elapsed, training_sessions=0))
        assert count == 84 and frames <= 84 * 480
        result = dict(status="complete", completed_at=now(), agent=name, seed=seed,
                      new_cells_completed=count, evaluation_frames=frames,
                      wall_seconds=time.monotonic() - elapsed,
                      output_sha256={p.name: digest(p) for p in (group / "rows.jsonl", group / "controls.jsonl.gz")})
        write_json(group / "result.json", result)
        return result
    except BaseException:
        try:
            with (output / "failure.json").open("x") as failure:
                json.dump(dict(status="failed", failed_at=now(), agent=name, seed=seed, traceback=traceback.format_exc()), failure)
        except FileExistsError:
            pass
        write_json(group / "failure.json", dict(status="failed", failed_at=now(), agent=name,
            seed=seed, new_cells_completed=count, traceback=traceback.format_exc()))
        raise


def summarize(output):
    assert not (output / "failure.json").exists(), "Failed evaluation must remain failed"
    contract = verify_contract(output)
    rows = read_rows(output / "reused_rows.jsonl")
    results = []
    for name in NAMES:
        for seed in SEEDS:
            group = output / f"{name}_seed{seed}"
            result = json.loads((group / "result.json").read_text())
            assert result["status"] == "complete"
            for file, sha in result["output_sha256"].items():
                assert digest(group / file) == sha
            fresh_rows = read_rows(group / "rows.jsonl")
            assert len(fresh_rows) == result["new_cells_completed"] == 84
            assert sum(r["steps"] for r in fresh_rows) == result["evaluation_frames"]
            frozen = contract["provenance"][name]["checkpoints"]
            initial = next(c for c in frozen if c["seed"] == seed and c["episode"] == 32)
            for row in fresh_rows:
                assert row["evaluation_origin"] == "fresh_deterministic"
                assert row["seed"] == seed and row["agent"] == name
                assert row["episode"] in (0, 32) and 0 < row["steps"] <= 480
                prefix = "initial" if row["episode"] == 0 else "trained"
                assert row["tensor_sha256"] == initial[prefix + "_tensor_sha256"]
                assert row["actor_sha256"] == initial[prefix + "_actor_sha256"]
            results.append(result)
            rows.extend(fresh_rows)
    expected = {(n, s, e, c) for n in NAMES for s in SEEDS for e in POINTS for c in contract["panel_ids"]}
    assert len(rows) == len(expected) == 3456 and {key(r) for r in rows} == expected
    assert sum(r["new_cells_completed"] for r in results) == 2016
    assert sum(r["evaluation_frames"] for r in results) <= contract["max_new_evaluation_frames"]
    cells = {key(r): r for r in rows}

    def metrics(group):
        requests = sum(r["request_count"] for r in group)
        complete = sum(r["completed_requests"] for r in group)
        model = sum(sum(c["transferred_bytes"] for c in r["model_cache"].values()) for r in group) / 1024
        evicted = sum(sum(c["evicted_bytes"] for c in r["model_cache"].values()) for r in group) / 1024
        return dict(cells=len(group), reward=sum(r["reward"] for r in group) / len(group),
                    completed_requests=complete / len(group), completion_percent=100 * complete / requests,
                    completed_nodes=sum(r["completed_nodes"] for r in group) / len(group),
                    model_kib=model / len(group), evicted_kib=evicted / len(group),
                    model_kib_per_completed_request=model / complete if complete else None,
                    state_kib=sum(r["state_transfer_bytes"] for r in group) / 1024 / len(group),
                    target_changes=sum(r["service_target_change_count"] for r in group) / len(group))

    aggregation, paired, strata = {}, {}, {}
    for name in NAMES:
        aggregation[name] = {str(e): metrics([r for r in rows if r["agent"] == name and r["episode"] == e]) for e in POINTS}
        aggregation[name]["seeds"] = {str(s): {str(e): metrics([cells[(name, s, e, c)] for c in contract["panel_ids"]]) for e in POINTS} for s in SEEDS}
        paired[name] = {}
        for a, b in ((0, 32), (32, 64), (0, 64)):
            delta = []
            for seed in SEEDS:
                for case in contract["panel_ids"]:
                    before, after = cells[(name, seed, a, case)], cells[(name, seed, b, case)]
                    delta.append(dict(seed=seed, case_id=case, reward=after["reward"] - before["reward"],
                        completed_requests=after["completed_requests"] - before["completed_requests"],
                        model_kib=(sum(c["transferred_bytes"] for c in after["model_cache"].values()) - sum(c["transferred_bytes"] for c in before["model_cache"].values())) / 1024))
            paired[name][f"{a}_to_{b}"] = dict(mean_reward_delta=sum(d["reward"] for d in delta) / len(delta),
                positive=sum(d["reward"] > 1e-8 for d in delta), negative=sum(d["reward"] < -1e-8 for d in delta),
                tied=sum(abs(d["reward"]) <= 1e-8 for d in delta), cells=delta)
        strata[name] = {}
        for field in ("session", "size", "arrival_mode", "families", "capacity"):
            values = sorted({meta[field] for meta in contract["case_metadata"].values()})
            strata[name][field] = {str(v): {str(e): metrics([r for r in rows if r["agent"] == name and r["episode"] == e and contract["case_metadata"][r["case_id"]][field] == v]) for e in POINTS} for v in values}
    value = dict(status="complete", summarized_at=now(), contract_sha256=digest(output / "contract.json"),
                 comparison_cells=len(rows), new_cells=2016, reused_cells=1440,
                 new_evaluation_frames=sum(r["evaluation_frames"] for r in results),
                 new_training_sessions=0, optimizer_steps=0, methods=aggregation,
                 paired=paired, strata=strata, claim_limits=contract["claim_limits"])
    write_json(output / "summary.json", value)
    write_json(output / "result.json", dict(status="complete", completed_at=now(),
               summary_sha256=digest(output / "summary.json"), new_training_sessions=0,
               new_cells=2016, comparison_cells=3456, new_evaluation_frames=value["new_evaluation_frames"]))
    return value


def run(output):
    assert not (output / "failure.json").exists(), "Failed evaluation cannot restart"
    contract = verify_contract(output)
    assert contract["evaluator_sha256"] == digest(__file__)
    source = Path(contract["source_run"])
    assert contract["source_protocol_sha256"] == digest(source / "protocol/manifest.json")
    verify_source(source)
    # Exclusive marker prevents duplicate starts or silent resumptions.
    with (output / "run_started.json").open("x") as f:
        json.dump(dict(started_at=now(), command=sys.argv, contract_sha256=digest(output / "contract.json")), f)
    started = time.monotonic()
    done = []
    try:
        with ProcessPoolExecutor(max_workers=contract["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = [pool.submit(worker, str(source), str(output), n, s) for n in NAMES for s in SEEDS]
            for future in as_completed(pending):
                try:
                    done.append(future.result())
                except BaseException:
                    if not (output / "failure.json").exists():
                        write_json(output / "failure.json", dict(status="failed", failed_at=now(), traceback=traceback.format_exc()))
                    for p in pending:
                        p.cancel()
                    raise
                write_json(output / "progress.json", dict(status="running", updated_at=now(),
                    completed_groups=[dict(agent=r["agent"], seed=r["seed"]) for r in done],
                    completed_group_cells=sum(r["new_cells_completed"] for r in done), new_cells=2016,
                    wall_seconds=time.monotonic() - started, training_sessions=0,
                    detail="In-flight group progress files contain case IDs/counts; parent updates at group boundaries"))
        summarize(output)
        write_json(output / "progress.json", dict(status="complete", updated_at=now(),
            new_cells_completed=2016, new_cells=2016, training_sessions=0,
            wall_seconds=time.monotonic() - started))
    except BaseException:
        if not (output / "failure.json").exists():
            write_json(output / "failure.json", dict(status="failed", failed_at=now(), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "prepare", "run", "summarize"))
    parser.add_argument("--source_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    if args.mode in ("qualify", "prepare"):
        assert args.source_run and args.source_run.is_dir()
    if args.mode == "qualify":
        qualify(args.source_run.resolve(), args.output)
    elif args.mode == "prepare":
        assert args.qualification
        prepare(args.source_run.resolve(), args.output.resolve(), args.qualification.resolve(), args.workers)
    elif args.mode == "run":
        run(args.output.resolve())
    else:
        summarize(args.output.resolve())


if __name__ == "__main__":
    main()
