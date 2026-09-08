"""Actual benchmark entry acceptance in a synthetic kernel boundary."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')
COMMIT = 'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d'

def external(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    from types import SimpleNamespace
    parser=argparse.ArgumentParser();parser.add_argument('--fixture-root',type=Path,required=True)
    args=parser.parse_args();fixture=args.fixture_root
    security=external('continuation_executor_security');security.validate_fixture_root(fixture)
    fixture.mkdir();kernel=external('continuation_fixture_sandbox').verify_kernel_boundary(fixture)
    adapter=external('continuation_legacy_adapter')
    runner,source=adapter.load_scientific_modules(SCIENCE,
        security.file_hash(SCIENCE/'scripts/run_typed_model_cache_formal_protocol.py'),expected_commit=COMMIT)
    implementation_commit=security.git(ROOT,'rev-parse','HEAD')
    monitor=external('continuation_fixture_monitor').FixtureMonitor(fixture_root=fixture,
        scientific_root=SCIENCE,python_executable=sys.executable,implementation_root=ROOT,
        forbidden_roots=['/Users/howen/Projects/PPO_MEC/data','/Users/howen/Projects/PPO_MEC/artifacts/experiments',
                         SCIENCE/'data',SCIENCE/'artifacts']).install()
    bundle=external('continuation_fixture_builder').build_fixture(fixture,SCIENCE,ROOT,runner,source,adapter,
        implementation_commit=implementation_commit,scientific_commit=COMMIT,benchmark_inputs=True)
    from scripts import benchmark_main_results as benchmark
    data=bundle['benchmark_inputs'];run=bundle['run_root'];inputs=bundle['synthetic_inputs_root']
    def evaluate_case(capacity,agent):
        values=dict(data['fairness'][capacity]['selection'])
        values.update(agents=[agent],seeds=[7],max_steps=1,mobility_source='ngsim',reward_positive_offset=0.0,
            mobility_csv_path=data['mobility'],workflow_csv_path=data['workflow'],window_plan_path=data['window_plan'],
            model_cache_runtime_config=str(bundle['runtime_paths'][capacity]),
            cache_baseline_fairness_manifest_path=data['fairness'][capacity]['path'],
            output_root=str(run/'benchmark_boundary'),resource_registry_path=str(inputs/'static_registry.json'),
            mobility_resource_id='mobility.synthetic',workflow_resource_id='workflow.synthetic',
            window_plan_resource_id='window.synthetic',runtime_config_resource_id='runtime.'+capacity,
            fairness_manifest_resource_id='fairness.'+capacity,
            generated_checkpoint_registry_path=str(run/'generated_checkpoint_resource_registry.json'),
            checkpoint_manifest_id='checkpoint_manifest.'+capacity,checkpoint_provenance_id='checkpoint_provenance.'+capacity,
            seed_checkpoint_manifest_path=str(run/'checkpoint_manifests'/capacity/'seed_checkpoint_manifest.json'),
            checkpoint_provenance_manifest_path=str(run/'checkpoint_manifests'/capacity/'checkpoint_provenance_manifest.json'),
            protocol_path=str(inputs/'synthetic_protocol.json'),
            resolved_execution_context_path=str(run/'resolved_execution_context.json'),
            formal_training_execution_binding_path=str(run/'formal_training_execution_binding.json'),
            non_formal_rehearsal=True)
        argv=[]
        actions={action.dest:action for action in benchmark.build_parser()._actions}
        for key,value in values.items():
            if key not in actions: continue
            option=actions[key].option_strings[0]
            if isinstance(value,bool):
                if value:argv.append(option)
            elif isinstance(value,list):argv.extend([option,*map(str,value)])
            else:argv.extend([option,str(value)])
        class RolloutBoundary(BaseException):pass
        def stop_before_rollout(**kwargs):
            raise RolloutBoundary('explicit synthetic boundary before real episode')
        original_episode,original_bundle=benchmark.run_real_episode,benchmark.load_window_bundle
        # Only synthetic mobility preparation and the explicit rollout boundary change.
        # Checkpoint, companion, registry, fairness and identity validators remain native.
        benchmark.run_real_episode=stop_before_rollout
        benchmark.load_window_bundle=lambda **kwargs:SimpleNamespace(rsu_metadata={})
        original_argv=sys.argv;sys.argv=[str(benchmark.__file__),*argv]
        error=None;reached=False
        calls_before=dict(monitor.scientific_calls)
        try:benchmark.main()
        except RolloutBoundary:reached=True
        except Exception as exc:error=repr(exc)
        finally:
            sys.argv=original_argv
            benchmark.run_real_episode,benchmark.load_window_bundle=original_episode,original_bundle
        report={'test_only':True,'scope':'actual benchmark main with synthetic dataset preparation and pre-rollout stop',
            'benchmark_call_counts':{key:value-calls_before.get(key,0) for key,value in monitor.scientific_calls.items() if value>calls_before.get(key,0)},
            'agent':agent,'capacity':capacity,'argv':argv,'boundary_reached':reached,'error':error,'kernel_boundary':kernel,
            'monitor':monitor.report(),'scientific_modules':adapter.loaded_origins(SCIENCE)}
        return report
    cases=[]
    for capacity in bundle['runtime_paths']:
        for agent in bundle['protocol']['training_budget']['learned_agent_order']:
            case=evaluate_case(capacity,agent);cases.append(case)
            print(json.dumps({'agent':agent,'capacity':capacity,'boundary_reached':case['boundary_reached'],'error':case['error']}),flush=True)
            if not case['boundary_reached']:break
        if not cases[-1]['boundary_reached']:break
    reached=len(cases)==30 and all(case['boundary_reached'] for case in cases)
    error=next((case['error'] for case in cases if case['error']),None)
    report={'test_only':True,'boundary_reached':reached,'error':error,'cases':cases,
        'benchmark_call_counts':{key:sum(case['benchmark_call_counts'].get(key,0) for case in cases)
            for key in set().union(*(case['benchmark_call_counts'] for case in cases))},
        'monitor':monitor.report(),'scientific_modules':adapter.loaded_origins(SCIENCE)}
    (fixture/'benchmark_report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({'boundary_reached':reached,'error':error,'report':str(fixture/'benchmark_report.json')}))
    if not reached:raise SystemExit(2)

if __name__=='__main__':main()
