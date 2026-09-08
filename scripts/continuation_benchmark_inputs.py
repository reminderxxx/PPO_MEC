"""Test-only benchmark input producer; original fairness validators remain intact."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path


def prepare(inputs, science, runtime_paths):
    from src.evaluators import cache_baseline_fairness as fairness
    from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime
    from src.data.model_catalog.adapter_catalog import AdapterCatalog
    import yaml
    inputs, science = Path(inputs), Path(science)
    catalog = inputs / 'src/data/model_catalog/typed_model_cache_controlled.json'
    mobility = inputs / 'synthetic_mobility.csv'
    mobility.write_text('Vehicle_ID,Frame_ID,Global_Time,Local_X,Local_Y\n1,1,1000,0,0\n')
    workflow = inputs / 'synthetic_workflow.csv'
    workflow.write_text(''.join(f'A{i}' + (f'_{i-1}' if i > 1 else '') +
        f',1,synthetic_job,1,Terminated,{i},{i+1},100,0.1\n' for i in range(1,6)))
    window = {'window_id':'synthetic_window', 'frame_offset':0, 'window_length':1,
        'time_index_start':1000, 'time_index_end':1000, 'window_class':'mechanism_activating',
        'recommended_rsu_layout':'auto_dominant_tight', 'source_segment_id':'synthetic_segment',
        'source_segment_run_id':'synthetic_segment_0'}
    plan = inputs / 'synthetic_window_plan.json'
    plan.write_text(json.dumps({'selected_window_plan':[window]}))
    outputs = {}
    for capacity, path in runtime_paths.items():
        runtime_yaml = yaml.safe_load(path.read_text())
        runtime_yaml['typed_catalog_path'] = str(catalog)
        path.write_text(yaml.safe_dump(runtime_yaml, sort_keys=False))
        runtime = resolve_model_cache_runtime(path, root=science)
        manifest = json.loads((science / 'configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906' /
            f'nonformal_rehearsal_fairness_{capacity}.json').read_text())
        manifest['identity']['purpose'] = 'test-only continuation input; no performance evidence'
        manifest['identity']['protocol_status'] = 'synthetic_only'
        replacements = {'ngsim_vehicle_trajectories':mobility,
            'alibaba_cluster_trace_2018_batch_task':workflow,
            'ppo_mec_sample_adapter_catalog':catalog, 'g07_non_hidden_window_plan':plan}
        manifest['dataset_provenance']['inputs'] = [fairness._path_identity(replacements[row['logical_dataset_id']],
            science, row['logical_dataset_id'], row['provider_parser_identity'])
            for row in manifest['dataset_provenance']['inputs']]
        selection = manifest['dataset_provenance']['selection_filter_parameters']
        selection.update(max_workflows=1, workflow_selector='ordered', min_tasks=5, max_tasks=20,
                         max_mobility_rows=1)
        workflows = fairness.build_selected_workflow_states(workflow_csv_path=workflow,
            max_workflows=1,workflow_selector='ordered',min_tasks=5,max_tasks=20,random_seed=7)
        assert len(workflows) == 1
        state = workflows[0]
        unit = deepcopy(manifest['window_workload_plan']['evaluation_units'][0])
        unit.update(window_id=window['window_id'],workflow_id=state.workflow_id,
            evaluation_unit_id='seed_7/' + window['window_id'] + '/' + state.workflow_id,
            raw_frame_interval={'start':0,'end':0},raw_time_interval={'start':1000,'end':1000},
            source_segment_id=window['source_segment_id'],source_segment_run_id=window['source_segment_run_id'],
            expected_workload_fingerprint=fairness.workload_fingerprint(state),
            workflow_dag_sha256=fairness.sha256_value(fairness._workflow_payload(state)))
        manifest['window_workload_plan'].update(evaluation_units=[unit],window_plan_path=str(plan),
            window_plan_sha256=fairness.sha256_file(plan))
        typed = manifest['cache_contract']['typed_model_cache']
        typed.update(fairness.build_typed_cache_fairness_binding(AdapterCatalog.from_json(catalog),
            runtime_contract=runtime))
        # Baseline configs are copied inputs; validators consume these real files.
        for entry in manifest['baseline_matrix']:
            source = science / entry['config']['path']
            target = inputs / 'baseline_configs' / source.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(source.read_bytes())
            entry['config']['path'] = str(target)
            entry['config']['normalized_absolute_path'] = str(target)
        manifest['artifact_plan']['output_root'] = str(inputs.parent / 'synthetic_run' / 'benchmark_boundary')
        manifest.pop('validation',None)
        semantic = fairness.semantic_protocol_sha256(manifest)
        manifest['identity']['manifest_id'] = 'cbfm-' + semantic[:16]
        manifest['hashes']['semantic_protocol_sha256'] = semantic
        manifest['hashes']['full_manifest_sha256'] = fairness.full_manifest_sha256(manifest)
        output = inputs / ('synthetic_fairness_' + capacity + '.json')
        output.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        loaded, validation = fairness.load_and_validate_manifest(output,root=science,check_files=True)
        outputs[capacity] = {'path':str(output),'runtime':runtime,'validation':validation,
            'workflow_id':state.workflow_id,'window':window,'selection':selection}
    return {'fairness':outputs,'mobility':str(mobility),'workflow':str(workflow),'window_plan':str(plan)}
