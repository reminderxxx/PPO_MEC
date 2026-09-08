"""Test-only child payload producer; no environment, agent training or rollout.

Called using a fixture-local launcher after authority and recursive scope checks.
The original native descriptor function supplies inventory and publication input.
"""
from pathlib import Path
import csv
import json
import sys

from .identity import read_json, within


def main():
    # Deliberately a separate synthetic argv. Real command equivalence is checked
    # independently and never claims this payload is a scientific evaluation.
    fixture, phase, setting = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    argv = sys.argv[4:]
    def arg(name): return argv[argv.index(name)+1]
    output = within(arg("--output-root"), fixture)
    artifact = output / "benchmark_synthetic"
    artifact.mkdir(parents=True)
    p = read_json(fixture/"inputs/protocol.json")
    order = read_json(fixture/"inputs/formal_agent_order_contract.json")
    metrics = p["endpoints"]["primary"]
    with (artifact/"benchmark_rows.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["agent_name","seed","window_id","workflow_id","source_segment_run_id",*metrics])
        writer.writeheader()
        for agent in order["main_benchmark_agent_order"]:
            seeds=dict.fromkeys(row["seed"] for row in p["execution_contract"]["command_templates"]["train"]["matrix_contexts"])
            for seed in seeds:
                for window in range(12):
                    writer.writerow(dict(agent_name=agent,seed=seed,window_id=str(window),workflow_id="synthetic",
                        source_segment_run_id="synthetic",**{metric:1.0 for metric in metrics}))
    (artifact/"aggregate_summary.json").write_text(json.dumps({"test_only":True,"formal_performance_evidence":False}))
    if phase == "formal_cache_policy":
        within(arg("--request-replay-path"),fixture).write_text('{"test_only":true}')
    elif phase != "formal_controller":
        from src.evaluators.formal_cell_transaction import write_child_output_descriptor
        (artifact/"support_provenance.json").write_text(json.dumps({"setting_id":setting,"output":str(artifact),"test_only":True}))
        write_child_output_descriptor(within(arg("--cell-output-descriptor-path"),fixture),
            cell_id=arg("--cell-id"),phase=phase,logical_setting_id=setting,output_root=output,artifact_root=artifact,
            producer_kind="synthetic_test_only",required_payload=["aggregate_summary.json","benchmark_rows.csv","support_provenance.json"])
