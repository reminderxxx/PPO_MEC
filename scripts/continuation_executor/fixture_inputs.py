"""Create isolated synthetic inputs; never reuse weights or execute training."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from .identity import ContinuationError, canonical, read_json, within


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def build_core(fixture_root, native):
    from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
    from src.runtime.portable_resource_identity import build_registry, build_resource_identity
    old, fixture = Path.cwd(), Path(fixture_root)
    run = within(str(fixture / "synthetic_continuation"), fixture)
    if run.exists():
        raise ContinuationError("fixture builder is create-only")
    inputs = fixture / "inputs"
    inputs.mkdir()
    source = old / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906"
    protocol = read_json(source / "protocol_v2_9_manifest.json")
    resources = benchmark_inputs(fixture, protocol)
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime
    for capacity in ("constrained_288mb", "medium_576mb", "relaxed_864mb"):
        runtime = resolve_model_cache_runtime(inputs / ("runtime_"+capacity+".yaml"), root=old)
        protocol["identity"]["typed_runtime_contract_hashes_by_capacity"][capacity.split("_",1)[0]] = runtime["runtime_contract_sha256"]
    static = build_registry(resources, registry_id="synthetic_continuation_static")
    write(inputs / "static_registry.json", static)
    protocol["portable_resource_identity_contract"]["resource_registry_semantic_sha256"] = static["hashes"]["semantic_sha256"]
    protocol = attach_hashes(protocol)
    native["execution"].validate_protocol_v1_1(protocol)
    write(inputs / "protocol.json", protocol)
    for name in ("agent_training_scientific_config.json", "execution_environment_manifest.json", "formal_agent_order_contract.json"):
        shutil.copyfile(source / name, inputs / name)
    environment = deepcopy(protocol["formal_execution_environment_contract"]["scientific_identity"])
    runtime_audit = {"observed_execution_commit": "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d", "resolution_source": "explicit_python_executable"}
    expansion = native["public_checks"].resolved_expansion_context(protocol, protocol_path=str(inputs / "protocol.json"),
        output_root=str(run), python_executable=sys.executable, active_formal_bundle_sha256="b"*64,
        active_protocol_index_path=str(inputs / "test_only_index.json"), active_bundle_resource_resolution_audit_sha256="c"*64)
    for key, value in list(expansion.items()):
        if isinstance(value, str) and value.startswith(str(old / "data") + "/"):
            expansion[key] = str(inputs / "data" / Path(value).relative_to(old / "data"))
        elif isinstance(value, str) and value.startswith(str(old / "configs") + "/"):
            target = inputs / "configs" / Path(value).relative_to(old / "configs")
            expansion[key] = str(target)
        elif (isinstance(value, str) and value.startswith("/")
              and key not in {"python_executable", "clean_worktree_root", "repository_root"}
              and not value.startswith(str(fixture) + "/")):
            expansion[key] = str(inputs / "unused" / key)
    expansion.update(resource_registry_path=str(inputs / "static_registry.json"), data_root=str(inputs / "data"),
        protocol_artifact_root=str(inputs), agent_scientific_config_path=str(inputs / "agent_training_scientific_config.json"),
        formal_agent_order_contract_path=str(inputs / "formal_agent_order_contract.json"),
        execution_environment_manifest_path=str(inputs / "execution_environment_manifest.json"))
    matrix = native["execution"].validate_command_templates(protocol["execution_contract"]["command_templates"], expansion)
    scientific = read_json(inputs / "agent_training_scientific_config.json")
    binding = native["binding"].build_execution_binding(protocol=protocol, scientific_config=scientific,
        execution_commit=runtime_audit["observed_execution_commit"], environment_identity=environment,
        command_matrix_sha256=matrix["command_matrix_sha256"], active_formal_bundle_sha256="b"*64)
    context = native["context"].build_resolved_formal_execution_context(protocol=protocol, expansion_context=expansion,
        environment_identity=environment, runtime_audit=runtime_audit, environment_manifest_path=inputs / "execution_environment_manifest.json",
        outer_expansion_sha256=matrix["command_matrix_sha256"], phase_count=matrix["phase_count"], command_count=matrix["command_count"],
        execution_binding=binding, active_formal_bundle_sha256="b"*64)
    from .planning import run_identity, cell_identity_fields
    from .identity import digest
    import hashlib
    context_file_hash = hashlib.sha256(canonical(context) + b"\n").hexdigest()
    phase_runner = native["phase"].TransactionalPhaseRunner(output_root=run,
        run_identity_fingerprint=run_identity(protocol, context, binding, context_file_hash, run,
            runtime_audit["observed_execution_commit"], environment["environment_fingerprint"]),
        phase_order=native["execution"].PHASE_ORDER,
        resolved_execution_context_sha256=context["context_sha256"], resolved_execution_context_file_sha256=context_file_hash)
    cell_fields = cell_identity_fields(protocol, context, binding, run.name,
        runtime_audit["observed_execution_commit"], environment["environment_fingerprint"])
    cells = native["cell"].FormalCellLedger(run_root=run, identity=native["cell"].CellExecutionIdentity(**cell_fields))
    write(run / "resolved_execution_context.json", context)
    write(run / "formal_training_execution_binding.json", binding)
    native["context"].load_resolved_formal_execution_context(run / "resolved_execution_context.json", protocol=protocol,
        clean_worktree_root=old, durable_run_root=run, check_git=True)
    native["binding"].validate_execution_binding(binding, protocol=protocol, scientific_config=scientific,
        execution_commit=runtime_audit["observed_execution_commit"], environment_identity=environment,
        command_matrix_sha256=matrix["command_matrix_sha256"], active_formal_bundle_sha256="b"*64)
    return dict(protocol=protocol, binding=binding, context=context, scientific=scientific,
                static=static, matrix=matrix, run=run, inputs=inputs, source=old, cells=cells, phase_runner=phase_runner)


def build_checkpoints(core, native):
    """Actual producer/annotator/read-back, selection, freeze and registry."""
    import torch
    from src.runtime.formal_training_contract import resolve_training_contract
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime, build_checkpoint_provenance
    from scripts.train_algo_pool_real_sample import build_training_identity_metadata, annotate_checkpoint, validate_serialized_formal_checkpoint
    import scripts.manage_typed_model_cache_formal_artifacts as artifact_main
    from contextlib import redirect_stdout
    from .identity import file_hash
    p, context, binding = core["protocol"], core["context"], core["binding"]
    cells, runner, root = core["cells"], core["phase_runner"], core["run"]
    resolved = {agent: resolve_training_contract(agent_name=agent,
        profile_defaults={"episodes":1,"update_every":1,"batch_size":1,"max_steps":1}, cli_values={},
        formal_protocol=p, scientific_config=core["scientific"], execution_binding=binding,
        resolved_execution_context=context) for agent in p["training_budget"]["learned_agent_order"]}
    candidates=[]
    for coord in p["execution_contract"]["command_templates"]["train"]["matrix_contexts"]:
        agent, seed, capacity = coord["agent"], coord["seed"], coord["capacity_label"]
        runtime = resolve_model_cache_runtime(core["inputs"] / ("runtime_"+capacity+".yaml"), root=core["source"])
        final = root / "training" / capacity / agent / ("seed_"+str(seed))
        begun = cells.begin_cell(phase="train",coordinates=coord,command=[],input_hash="synthetic_preexisting",committed_path=final)
        staging=Path(begun["record"]["staging_path"])
        (staging/"checkpoints").mkdir()
        rows=[]
        for update in range(4,33,4):
            metadata = dict(run_id="synthetic_"+agent+"_"+str(seed), agent_name=agent,episodes=resolved[agent].episodes,
                update_count=update,checkpoint_schedule={"checkpoint_every_updates":resolved[agent].checkpoint_every_updates,
                    "expected_update_count":resolved[agent].expected_update_count},
                **build_training_identity_metadata(resolved[agent]),
                typed_runtime_provenance=build_checkpoint_provenance(root=core["source"],agent_name=agent,
                    training_seed=seed,runtime_contract=runtime,reward_positive_offset=0.0,
                    train_window_plan_identity={"test_only":True,"split":"train"}))
            name=f"update_{update:04d}.pt"; path=staging/"checkpoints"/name
            torch.save({"synthetic_test_only":True},path)
            annotate_checkpoint(path,metadata)
            validate_serialized_formal_checkpoint(path,resolved_training=resolved[agent],agent_name=agent,seed=seed,
                runtime_contract_sha256=runtime["runtime_contract_sha256"])
            rows.append(dict(agent_name=agent,seed=seed,capacity_label=capacity,update_index=update,
                checkpoint_path=str(final/"checkpoints"/name),checkpoint_sha256=file_hash(path),
                full_service_ready_byte_hit_rate=0.5,workflow_continuity_rate=0.5,transfer_mb_per_request=1.0,
                end_to_end_workflow_delay=2.0,selection_metric_availability={metric:dict(available_count=1,unavailable_count=0,total_count=1)
                    for metric in ("full_service_ready_byte_hit_rate","workflow_continuity_rate","transfer_mb_per_request","end_to_end_workflow_delay")},
                runtime_contract_sha256=runtime["runtime_contract_sha256"],resolved_agent_config=resolved[agent].agent_config,
                checkpoint_schedule=metadata["checkpoint_schedule"],**build_training_identity_metadata(resolved[agent]),
                non_formal_rehearsal=False,typed_runtime_provenance=metadata["typed_runtime_provenance"]))
        shutil.copyfile(path,staging/"checkpoints/latest.pt")
        write(staging/"train_summary.json",{"test_only":True})
        cells.commit_cell(begun["cell_id"],required_paths=["train_summary.json","checkpoints/latest.pt"])
        candidates.extend(rows)
    # Pre-existing synthetic dev cells carry no rollout or child dispatch.
    for index in range(24):
        begun=cells.begin_cell(phase="dev_select",coordinates={"synthetic_dev_cell":index},command=[],
            input_hash="synthetic_preexisting",committed_path=root/"dev"/str(index))
        staging=Path(begun["record"]["staging_path"])
        write(staging/"dev_summary.json",{"test_only":True})
        cells.commit_cell(begun["cell_id"],required_paths=["dev_summary.json"])
    write(root/"checkpoint_candidates.json",candidates)
    producer_calls=[]
    for action, filename in (("dev_select","dev_selection.json"),("checkpoint_freeze","checkpoint_freeze.json")):
        argv=[str(core["source"]/"scripts/manage_typed_model_cache_formal_artifacts.py"),"--action",action,
            "--protocol-path",str(core["inputs"]/"protocol.json"),"--input-root",str(root),"--output-path",str(root/filename)]
        previous=sys.argv
        try:
            sys.argv=argv
            with (root.parent/(action+"_producer_stdout.json")).open("w") as stream, redirect_stdout(stream):
                artifact_main.main()
        finally:sys.argv=previous
        producer_calls.append(argv)
    freeze=read_json(root/"checkpoint_freeze.json")
    for previous in native["execution"].PHASE_ORDER[:5]:
        runner.run_phase(previous,commands=[],input_hash="synthetic_preexisting",expected_outputs=[])
    registry=native["registry"].build_generated_checkpoint_registry(run_root=root,protocol=p,static_registry=core["static"],
        resolved_execution_context=context,execution_binding=binding)
    native["registry"].atomic_create_registry(root/"generated_checkpoint_resource_registry.json",registry)
    _, audit=native["registry"].load_generated_checkpoint_registry(root/"generated_checkpoint_resource_registry.json",
        run_root=root,expected_run_id=root.name,static_registry_semantic_sha256=core["static"]["hashes"]["semantic_sha256"],
        protocol_semantic_sha256=p["hashes"]["semantic_sha256"],protocol_full_sha256=p["hashes"]["full_sha256"],
        active_formal_bundle_sha256=context["scientific_identity"]["active_formal_bundle_sha256"],
        execution_commit=context["scientific_identity"]["execution_commit"],resolved_execution_context_sha256=context["context_sha256"],
        formal_training_execution_binding_sha256=binding["binding_full_sha256"])
    core.update(registry=registry,registry_audit=audit,freeze=freeze)
    return {"producer_main_argv":producer_calls,"candidate_count":len(candidates),"frozen_count":freeze["frozen_checkpoint_count"],"registry":audit}


def benchmark_inputs(fixture, protocol):
    """Actual fairness builder and validator over small synthetic source tables."""
    import csv
    import yaml
    from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest, BASELINE_NAMES
    from src.runtime.portable_resource_identity import build_resource_identity
    old=Path.cwd(); inputs=fixture/"inputs"
    import subprocess
    if not (fixture/".git").exists():
        subprocess.run(["git","init","-q",str(fixture)],check=True)
        (fixture/"fixture_identity.txt").write_text("Synthetic fixture; not scientific source.\n")
        subprocess.run(["git","-C",str(fixture),"add","fixture_identity.txt"],check=True)
        subprocess.run(["git","-C",str(fixture),"-c","user.name=Synthetic fixture",
            "-c","user.email=fixture@example.invalid","commit","-qm","Synthetic fixture identity"],check=True)
    catalog=fixture/"src/data/model_catalog/typed_model_cache_controlled.json"
    catalog.parent.mkdir(parents=True)
    shutil.copyfile(old/"src/data/model_catalog/typed_model_cache_controlled.json",catalog)
    config_root=fixture/"configs/algo";config_root.mkdir(parents=True)
    for agent in BASELINE_NAMES:shutil.copyfile(old/"configs/algo"/(agent+".yaml"),config_root/(agent+".yaml"))
    mobility=inputs/"mobility.csv";mobility.write_text("Vehicle_ID,Frame_ID,Global_Time,Local_X,Local_Y\n1,0,0,0,0\n")
    workflow=inputs/"workflow.csv"
    with workflow.open("w",newline="") as stream:
        writer=csv.writer(stream)
        for i in range(1,6):writer.writerow(["M"+str(i)+(('_'+str(i-1)) if i>1 else ''),1,"synthetic_job",1,"Terminated",0,10,100,1])
    plan=inputs/"window_plan.json"
    write(plan,{"mobility_source_path":str(mobility),"selected_window_plan":[dict(window_id=str(i),frame_offset=i*24,
        window_length=24,time_index_start=i*24,time_index_end=i*24+23,window_class="test_only",
        recommended_rsu_layout="auto_dominant_tight",source_segment_run_id="synthetic") for i in range(12)]})
    resources=[]
    def resource(path,logical,role):
        resources.append(build_resource_identity(path,logical_resource_id=logical,resource_role=role,
            schema_version="1.0.0",revision="synthetic",expected_logical_relative_path=path.relative_to(inputs).as_posix()))
    resource(mobility,"dataset.mobility.ngsim.vehicle_trajectories","mobility_dataset")
    resource(workflow,"dataset.workflow.alibaba2018.batch_task","workflow_dataset")
    resource(plan,"window_plan.typed_model_cache.formal","window_plan")
    coords=protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]
    seeds=list(dict.fromkeys(row["seed"] for row in coords))
    for capacity,coord in {row["capacity_label"]:row for row in coords}.items():
        runtime_path=inputs/("runtime_"+capacity+".yaml")
        runtime=yaml.safe_load((old/coord["runtime_config_path"]).read_text())
        runtime["catalog_path"]=runtime["typed_catalog_path"]=str(catalog)
        runtime_path.write_text(yaml.safe_dump(runtime))
        manifest=build_manifest(root=fixture,mobility_path=mobility,workflow_path=workflow,window_plan_path=plan,
            catalog_path=catalog,seeds=seeds,max_workflows=1,workflow_selector="ordered",min_tasks=5,max_tasks=20,
            max_steps=12,max_mobility_rows=24,primary_vehicle_selection="handoff_pressure",capacity_unit="mb",
            capacity_value=float(runtime["cache_capacity_profile"]["capacity_mb"]),output_root=str(fixture/"consumer_output"),
            controller_agents=protocol["training_budget"]["learned_agent_order"])
        validation=validate_manifest(manifest,root=fixture,check_files=True)
        if validation["status"] != "pass":raise ContinuationError(str(validation))
        fairness=inputs/("fairness_"+capacity+".json");write(fairness,manifest)
        resource(runtime_path,"runtime_config."+capacity,"runtime_config")
        resource(fairness,"fairness_manifest.formal."+capacity,"fairness_manifest")
    return resources


def load_prepared_core(fixture, native):
    """Development reuse of an unchanged pre-continuation synthetic fixture."""
    from .identity import file_hash
    from .planning import run_identity
    root=fixture/"synthetic_continuation";inputs=fixture/"inputs"
    p=read_json(inputs/"protocol.json");context=read_json(root/"resolved_execution_context.json")
    binding=read_json(root/"formal_training_execution_binding.json");static=read_json(inputs/"static_registry.json")
    native["context"].load_resolved_formal_execution_context(root/"resolved_execution_context.json",protocol=p,
        clean_worktree_root=Path.cwd(),durable_run_root=root,check_git=True)
    identity=native["cell"].CellExecutionIdentity(**read_json(root/"cell_ledger_identity.json")["identity"])
    cells=native["cell"].FormalCellLedger(run_root=root,identity=identity,resume=True)
    runner=native["phase"].TransactionalPhaseRunner(output_root=root,resume=True,
        run_identity_fingerprint=run_identity(p,context,binding,file_hash(root/"resolved_execution_context.json"),root,
            context["scientific_identity"]["execution_commit"],context["scientific_identity"]["environment_fingerprint"]),
        phase_order=native["execution"].PHASE_ORDER,resolved_execution_context_sha256=context["context_sha256"],
        resolved_execution_context_file_sha256=file_hash(root/"resolved_execution_context.json"))
    if len(runner.records())!=15 or len(cells.records())!=348:
        raise ContinuationError("prepared reuse only accepts the original synthetic freeze point")
    registry,audit=native["registry"].load_generated_checkpoint_registry(root/"generated_checkpoint_resource_registry.json",
        run_root=root,expected_run_id=root.name,static_registry_semantic_sha256=static["hashes"]["semantic_sha256"],
        protocol_semantic_sha256=p["hashes"]["semantic_sha256"],protocol_full_sha256=p["hashes"]["full_sha256"],
        active_formal_bundle_sha256=context["scientific_identity"]["active_formal_bundle_sha256"],
        execution_commit=context["scientific_identity"]["execution_commit"],resolved_execution_context_sha256=context["context_sha256"],
        formal_training_execution_binding_sha256=binding["binding_full_sha256"])
    return dict(protocol=p,context=context,binding=binding,static=static,registry=registry,registry_audit=audit,
        run=root,inputs=inputs,source=Path.cwd(),cells=cells,phase_runner=runner)
