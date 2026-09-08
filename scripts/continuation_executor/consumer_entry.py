"""Actual old benchmark main with native loaders/gates and a pre-rollout stop."""
from copy import deepcopy
import inspect
import sys
from types import SimpleNamespace

from .identity import canonical, file_hash, read_json


class BeforeRollout(Exception):
    pass


def check_benchmark_main(core, monitor, case="valid"):
    import scripts.benchmark_main_results as benchmark
    from src.runtime.generated_checkpoint_resources import canonical_sha256
    root, inputs=core["run"],core["inputs"]
    capacity="medium_576mb"
    provenance=root/"checkpoint_manifests"/capacity/"checkpoint_provenance_manifest.json"
    registry_path=root/"generated_checkpoint_resource_registry.json"
    before={path:path.read_bytes() for path in (provenance,registry_path)}
    payload=read_json(provenance)
    seed=str(read_json(inputs/("fairness_"+capacity+".json"))["seed_plan"]["benchmark_run_seeds"][0])
    envelope=payload["sa_ghmappo"][seed]
    fields={"missing_binding":"formal_training_execution_binding_sha256",
        "missing_nullable":"formal_nullable_metric_aggregation_contract_semantic_sha256",
        "protocol":"formal_protocol_semantic_sha256","bundle":"active_formal_bundle_sha256",
        "binding":"formal_training_execution_binding_sha256","context":"resolved_execution_context_sha256",
        "git":"execution_git_commit","runtime":"runtime_contract_sha256"}
    if case.startswith("missing_"):envelope.pop(fields[case])
    elif case in fields:envelope[fields[case]]="0"*(40 if case=="git" else 64)
    elif case=="sha":
        envelope["checkpoint_sha256"]="0"*64
        envelope["checkpoint_identity"]["checkpoint_sha256"]="0"*64
    elif case=="window":
        envelope["train_window_plan_identity"]={"test_only":True,"split":"wrong"}
        envelope["checkpoint_identity"]["window_identity"]=envelope["train_window_plan_identity"]
    elif case in {"agent","seed","capacity"}:
        envelope["checkpoint_identity"][case]="wrong" if case!="seed" else int(seed)+1
    elif case!="valid":raise ValueError("unknown consumer case")
    argv=[str(core["source"]/"scripts/benchmark_main_results.py"),"--agents","sa_ghmappo","--seeds",
        *[str(s) for s in read_json(inputs/("fairness_"+capacity+".json"))["seed_plan"]["benchmark_run_seeds"]],
        "--non-formal-rehearsal","--protocol-path",str(inputs/"protocol.json"),
        "--resolved-execution-context-path",str(root/"resolved_execution_context.json"),
        "--formal-training-execution-binding-path",str(root/"formal_training_execution_binding.json"),
        "--generated-checkpoint-registry-path",str(registry_path),"--resource-registry-path",str(inputs/"static_registry.json"),
        "--repository-root",str(core["source"]),"--data-root",str(inputs),"--protocol-artifact-root",str(inputs),
        "--checkpoint-root",str(root),"--mobility-resource-id","dataset.mobility.ngsim.vehicle_trajectories",
        "--workflow-resource-id","dataset.workflow.alibaba2018.batch_task","--window-plan-resource-id","window_plan.typed_model_cache.formal",
        "--runtime-config-resource-id","runtime_config."+capacity,"--fairness-manifest-resource-id","fairness_manifest.formal."+capacity,
        "--checkpoint-manifest-id","checkpoint_manifest."+capacity,"--checkpoint-provenance-id","checkpoint_provenance."+capacity,
        "--seed_checkpoint_manifest_path",str(root/"checkpoint_manifests"/capacity/"seed_checkpoint_manifest.json"),
        "--checkpoint_provenance_manifest_path",str(provenance),
        "--mobility_csv_path",str(inputs/"mobility.csv"),"--workflow_csv_path",str(inputs/"workflow.csv"),
        "--window_plan_path",str(inputs/"window_plan.json"),
        "--model_cache_runtime_config",str(inputs/("runtime_"+capacity+".yaml")),
        "--cache_baseline_fairness_manifest_path",str(inputs/("fairness_"+capacity+".json")),
        "--max_workflows","1","--min_tasks","5","--max_tasks","20","--max_mobility_rows","24",
        "--primary_vehicle_selection","handoff_pressure","--output_root",str(root.parent/"consumer_output")]
    source,line=inspect.getsourcelines(benchmark.main)
    stop=line+next(i for i,text in enumerate(source) if "summary = run_real_episode(" in text)
    def trace(frame,event,arg):
        if frame.f_code is benchmark.main.__code__:
            if event=="line" and frame.f_lineno==stop:raise BeforeRollout
            return trace
        return None
    previous_trace,previous_argv=sys.gettrace(),sys.argv
    original_bundle=benchmark.load_window_bundle
    counts_before=monitor.calls.copy()
    error=None
    try:
        if case!="valid":
            provenance.write_bytes(canonical(payload)+b"\n")
            registry=read_json(registry_path)
            for resource in registry["resources"]:
                path=root/resource["durable_run_root_relative_path"]
                resource["size_bytes"]=path.stat().st_size
                resource["content_sha256"]=file_hash(path)
            registry["registry_canonical_sha256"]=canonical_sha256({k:v for k,v in registry.items() if k!="registry_canonical_sha256"})
            registry_path.write_bytes(canonical(registry)+b"\n")
        # Only mobility bundle preparation is a test boundary. Actual argument
        # parsing, file/resource/fairness validation, workload parser, companion
        # loaders, registry, training identity and checkpoint gate remain native.
        benchmark.load_window_bundle=lambda **kwargs:SimpleNamespace(rsu_metadata={})
        sys.argv=argv
        sys.settrace(trace)
        try:benchmark.main()
        except BeforeRollout:
            if case!="valid":raise AssertionError("invalid checkpoint reached rollout boundary")
            error="controlled_pre_rollout_stop"
        except ValueError as exc:
            if case=="valid":raise
            error=str(exc)
        if error is None:raise AssertionError("main did not reach the expected boundary")
    finally:
        sys.settrace(previous_trace);sys.argv=previous_argv
        benchmark.load_window_bundle=original_bundle
        for path,data in before.items():
            if path.read_bytes()!=data:path.write_bytes(data)
    observed={name:count-counts_before[name] for name,count in monitor.calls.items() if count>counts_before[name]}
    required=("scripts.benchmark_main_results.main","scripts.benchmark_main_results.load_checkpoint_provenance_manifest",
              "scripts.benchmark_main_results.validate_benchmark_checkpoint_gate")
    if any(observed.get(name,0)!=1 for name in required):raise AssertionError(str(observed))
    return dict(case=case,status="pass",boundary=error,actual_calls=observed,argv=argv,
        setup_boundary="synthetic mobility bundle only; no rollout",main_source=str(core["source"]/"scripts/benchmark_main_results.py"))
