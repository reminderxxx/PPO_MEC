"""Invoke original statistics or artifact main under inherited fixture isolation."""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def external(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--scientific-root', type=Path, required=True)
    parser.add_argument('--module', choices=['scripts.run_typed_model_cache_formal_statistics',
                                           'scripts.manage_typed_model_cache_formal_artifacts'], required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.arguments or args.arguments[0] != '--':
        raise ValueError('explicit consumer argument boundary required')
    security = external('continuation_executor_security')
    phase='formal_statistics' if args.module=='scripts.run_typed_model_cache_formal_statistics' else 'formal_gate'
    outputs=[args.arguments[index+1] for index,value in enumerate(args.arguments[:-1])
        if value in {'--output-root','--output-path'}]
    dispatch_authority=external('continuation_fixture_authorization').verify(args.fixture_root,ROOT,phase=phase,output_paths=outputs)
    external('continuation_fixture_sandbox').verify_kernel_boundary(args.fixture_root)
    # This key is only a local reference validator; it does not authorize production.
    validator = security.FixtureTrust(args.fixture_root, b'input-reference-check-only-not-approval', '0'*64)
    validator.validate_references(args.arguments[1:])
    adapter = external('continuation_legacy_adapter')
    adapter.load_scientific_modules(args.scientific_root, security.file_hash(
        args.scientific_root / 'scripts/run_typed_model_cache_formal_protocol.py'),
        expected_commit='a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')
    monitor = external('continuation_fixture_monitor').FixtureMonitor(
        fixture_root=args.fixture_root, scientific_root=args.scientific_root, python_executable=sys.executable,
        implementation_root=ROOT, forbidden_roots=[
            '/Users/howen/Projects/PPO_MEC/artifacts/experiments', '/Users/howen/Projects/PPO_MEC/data',
            args.scientific_root / 'data', args.scientific_root / 'artifacts']).install()
    module = importlib.import_module(args.module)
    sys.argv = [str(Path(module.__file__)), *args.arguments[1:]]
    module.main()
    print(json.dumps({'continuation_fixture_monitor': monitor.report(), 'dispatch_authority':dispatch_authority,
                      'scientific_modules': adapter.loaded_origins(args.scientific_root)}))


if __name__ == '__main__':
    main()
