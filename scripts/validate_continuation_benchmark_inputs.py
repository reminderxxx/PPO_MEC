"""Isolated actual fairness input validation; not full benchmark acceptance."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--fixture-root',type=Path,required=True)
    args=parser.parse_args();fixture=args.fixture_root
    security=external('continuation_executor_security');security.validate_fixture_root(fixture)
    fixture.mkdir()
    kernel=external('continuation_fixture_sandbox').verify_kernel_boundary(fixture)
    adapter=external('continuation_legacy_adapter')
    runner,source=adapter.load_scientific_modules(SCIENCE,
        security.file_hash(SCIENCE/'scripts/run_typed_model_cache_formal_protocol.py'),expected_commit=COMMIT)
    monitor=external('continuation_fixture_monitor').FixtureMonitor(fixture_root=fixture,
        scientific_root=SCIENCE,python_executable=sys.executable,implementation_root=ROOT,
        forbidden_roots=['/Users/howen/Projects/PPO_MEC/data','/Users/howen/Projects/PPO_MEC/artifacts/experiments',
                         SCIENCE/'data',SCIENCE/'artifacts']).install()
    inputs=fixture/'inputs';inputs.mkdir()
    catalog=inputs/'src/data/model_catalog/typed_model_cache_controlled.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes((SCIENCE/'src/data/model_catalog/typed_model_cache_controlled.json').read_bytes())
    protocol=json.loads((SCIENCE/'configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_v2_9_manifest.json').read_text())
    runtimes={}
    for row in protocol['execution_contract']['command_templates']['train']['matrix_contexts']:
        cap=row['capacity_label']
        if cap in runtimes:continue
        path=inputs/Path(row['runtime_config_path']).name
        path.write_bytes((SCIENCE/row['runtime_config_path']).read_bytes());runtimes[cap]=path
    results=external('continuation_benchmark_inputs').prepare(inputs,SCIENCE,runtimes)
    report={'test_only':True,'scope':'synthetic fairness inputs with actual validators; benchmark main not yet tested',
        'kernel_boundary':kernel,'results':results,'monitor':monitor.report(),
        'scientific_modules':adapter.loaded_origins(SCIENCE)}
    (fixture/'input_report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({'status':'pass','capacities':list(results['fairness'])}))

if __name__=='__main__':main()
