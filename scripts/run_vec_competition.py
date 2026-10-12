"""Run a trace-driven engineering replay with four RSUs and eight vehicles."""

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

from src.envs.core.vec_competition_env import VECCompetitionEnv  # noqa: E402
from src.evaluators.vec_competition_scenario import build_scenario, is_lfs_pointer  # noqa: E402


def _materialized_path(relative: Path) -> Path:
    candidate = ROOT / relative
    if candidate.is_file() and not is_lfs_pointer(candidate):
        return candidate
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip()
    primary = Path(common).resolve().parent / relative
    return primary if primary.is_file() and not is_lfs_pointer(primary) else candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngsim-csv", type=Path, default=None)
    parser.add_argument("--alibaba-csv", type=Path, default=None)
    parser.add_argument("--catalog-json", type=Path, default=ROOT / "src/data/model_catalog/typed_model_cache_controlled.json")
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--segment", default="")
    parser.add_argument("--cache-capacity-mb", type=float, default=512.0)
    parser.add_argument("--compute-slots", type=int, default=1)
    parser.add_argument("--compute-mb-per-s", type=float, default=40.0)
    parser.add_argument("--radio-mb-per-s", type=float, default=20.0)
    parser.add_argument("--backhaul-mb-per-s", type=float, default=40.0)
    parser.add_argument("--alibaba-size-unit-mb", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/analysis/vec_competition_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}; choose a new run directory")
    ngsim = args.ngsim_csv or next(iter(sorted((ROOT / "data/raw/mobility/ngsim").glob("*.csv"))), None)
    if ngsim is None:
        raise FileNotFoundError("No NGSIM CSV found; supply --ngsim-csv")
    if args.ngsim_csv is None:
        ngsim = _materialized_path(ngsim.relative_to(ROOT))
    alibaba = args.alibaba_csv or _materialized_path(Path("data/raw/workflow/alibaba2018/batch_task.csv"))
    scenario, metadata = build_scenario(
        ngsim, alibaba, args.catalog_json,
        max_rows=args.max_rows, frame_count=args.frames, segment=args.segment,
        cache_capacity_mb=args.cache_capacity_mb, compute_slots=args.compute_slots,
        compute_mb_per_s=args.compute_mb_per_s, radio_mb_per_s=args.radio_mb_per_s,
        backhaul_mb_per_s=args.backhaul_mb_per_s,
        alibaba_size_unit_mb=args.alibaba_size_unit_mb,
    )
    env = VECCompetitionEnv(scenario)
    result = env.run()
    result["ledger_sha256"] = env.ledger_digest()
    metadata.update({
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "argv": sys.argv,
        "contract_version": result["contract_version"],
        "claim_level": "engineering_replay_only; no formal/holdout or model-inference claim",
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "scenario_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "event_ledger.jsonl").open("w", encoding="utf-8") as file:
        for event in env.ledger:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
