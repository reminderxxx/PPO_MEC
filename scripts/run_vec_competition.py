"""Execute the frozen four-RSU/eight-request VEC engineering scenario."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.core.vec_competition_runtime import VECCompetitionRuntime  # noqa: E402
from src.envs.core.vec_competition_policies import ConstantVelocityPrefetchController, NoPrefetchController  # noqa: E402
from src.evaluators.vec_frozen_scenario import load_frozen_scenario  # noqa: E402
from src.evaluators.vec_competition_audit import audit_competition_run  # noqa: E402
from src.evaluators.vec_competition_replay import render_replay_html  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/experiment/vec_competition_peachtree_4rsu_8vehicle_v2.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller", choices=("no_prefetch_v1", "constant_velocity_prefetch_v1"), default="no_prefetch_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("frozen engineering run requires a clean Git worktree; commit the implementation first")
    scenario, metadata = load_frozen_scenario(args.profile, ROOT)
    env = VECCompetitionRuntime(scenario)
    controller = NoPrefetchController() if args.controller == "no_prefetch_v1" else ConstantVelocityPrefetchController(scenario.rsus)
    summary = env.run(controller)
    summary["controller_version"] = controller.version
    summary["ledger_sha256"] = env.ledger_digest()
    audit = audit_competition_run(scenario, env.ledger, summary)
    if not audit["passed"]:
        raise AssertionError(f"ledger audit failed: {audit['failures']}")
    metadata.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command_argv": sys.argv,
        "controller": {**summary["plugin_versions"], "decision": controller.version},
        "measurement_boundary": "observed mobility and DAG topology; inferred RSUs; controlled radio/backhaul/compute/model bytes; metadata state codec",
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("scenario_manifest.json", metadata), ("summary.json", summary), ("ledger_audit.json", audit)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "event_ledger.jsonl").open("w", encoding="utf-8") as file:
        for event in env.ledger:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "association_timeline.json").write_text(json.dumps(env.association_timeline, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "replay.html").write_text(render_replay_html(scenario, env.association_timeline, env.ledger, summary), encoding="utf-8")
    (args.output_dir / "command_log.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
