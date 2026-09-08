"""Read-only native companion and benchmark checkpoint qualification."""
from .identity import ContinuationError, read_json


def check_frozen_checkpoints(root, run, protocol, context, binding, *, runtime_paths=None):
    from scripts.benchmark_main_results import load_checkpoint_provenance_manifest, validate_benchmark_checkpoint_gate
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime
    runtime_paths=runtime_paths or {row["capacity_label"]:root/row["runtime_config_path"] for row in
        protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]}
    runtimes={capacity:resolve_model_cache_runtime(path,root=root) for capacity,path in runtime_paths.items()}
    companions={capacity:load_checkpoint_provenance_manifest(str(run/"checkpoint_manifests"/capacity/"checkpoint_provenance_manifest.json"))
        for capacity in runtimes}
    rows=read_json(run/"checkpoint_freeze.json")["frozen_checkpoints"]
    coordinates={(r["agent"],int(r["seed"]),r["capacity_label"]) for r in
        protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]}
    if len(rows)!=len(coordinates) or {(r["agent_name"],int(r["seed"]),r["capacity_label"]) for r in rows}!=coordinates:
        raise ContinuationError("frozen checkpoint coordinate coverage drift")
    checked=[]
    for row in rows:
        capacity,agent,seed=row["capacity_label"],row["agent_name"],int(row["seed"])
        result=validate_benchmark_checkpoint_gate(row["checkpoint_path"],expected_agent_name=agent,expected_seed=seed,
            expected_runtime_contract=runtimes[capacity],expected_reward_positive_offset=0.0,
            provenance_envelope=companions[capacity][agent][str(seed)],protocol=protocol,
            resolved_execution_context=context,execution_binding=binding,expected_capacity_label=capacity)
        if result["status"]!="compatible":raise ContinuationError("native checkpoint gate rejected")
        checked.append(dict(agent=agent,seed=seed,capacity=capacity,path=row["checkpoint_path"],sha256=row["checkpoint_sha256"],status="compatible"))
    return {"attempt_count":len(rows),"pass_count":len(checked),"checkpoints":checked}
