"""Explicit test-only child payload producer, never a scientific rollout."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def external(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', required=True)
    parser.add_argument('--scientific-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--cell-id', default='')
    parser.add_argument('--cell-phase', required=True)
    parser.add_argument('--cell-output-descriptor-path', default='')
    parser.add_argument('--request-replay-path', default='')
    parser.add_argument('--row-input', default='')
    parser.add_argument('--agents', nargs='+')
    parser.add_argument('--setting', required=True)
    parser.add_argument('--fault', choices=['none', 'missing', 'provenance', 'descriptor', 'exit75', 'exit7'], default='none')
    args = parser.parse_args()
    fixture = Path(args.fixture_root)
    science = Path(args.scientific_root)
    if Path.cwd() != science or os.environ.get('PYTHONPATH') != str(science):
        raise ValueError('fixture child source environment drift')
    security = external('continuation_executor_security')
    for value in (args.output_root, args.cell_output_descriptor_path, args.request_replay_path, args.row_input):
        if value:
            security.contained(value, fixture)
    if args.cell_phase not in security.PHASES[:5]:
        raise ValueError('fixture child phase denied')
    dispatch_authority=external('continuation_fixture_authorization').verify(fixture,ROOT,phase=args.cell_phase,
        output_paths=[args.output_root,args.cell_output_descriptor_path,args.request_replay_path],cell_output_root=args.output_root)
    external('continuation_fixture_sandbox').verify_kernel_boundary(fixture)
    adapter = external('continuation_legacy_adapter')
    adapter.load_scientific_modules(science, security.file_hash(
        science / 'scripts/run_typed_model_cache_formal_protocol.py'),
        expected_commit='a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')
    monitor = external('continuation_fixture_monitor').FixtureMonitor(
        fixture_root=fixture, scientific_root=science, python_executable=sys.executable,
        implementation_root=ROOT, forbidden_roots=[
            '/Users/howen/Projects/PPO_MEC/artifacts/experiments',
            '/Users/howen/Projects/PPO_MEC/data', science / 'data', science / 'artifacts']).install()
    from src.evaluators.formal_cell_transaction import write_child_output_descriptor
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    if args.fault == 'exit7':
        return 7
    if args.fault == 'exit75' and dispatch_authority['cell_attempt']==1:
        return 75
    payload = output / 'synthetic_payload'
    payload.mkdir()
    (payload / 'aggregate_summary.json').write_text('{"test_only":true}\n')
    if args.row_input:
        import csv
        with Path(args.row_input).open() as stream:
            rows = list(csv.DictReader(stream))
        if args.agents and list(dict.fromkeys(row['agent_name'] for row in rows)) != args.agents:
            raise ValueError('synthetic input agent order drift')
        for row in rows:
            row['scalability_setting_id'] = args.setting
        with (payload / 'benchmark_rows.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    else:
        (payload / 'benchmark_rows.csv').write_text('test_only\n1\n')
    (payload / 'support_provenance.json').write_text(json.dumps({
        'setting_id': 'wrong' if args.fault == 'provenance' else args.setting,
        'test_only': True, 'output_root': str(payload)}) + '\n')
    if args.cell_phase == 'formal_cache_policy':
        if not args.request_replay_path:
            raise ValueError('synthetic cache replay path missing')
        Path(args.request_replay_path).write_text('{"test_only":true}\n')
    if args.cell_phase not in {'formal_cache_policy', 'formal_controller'}:
        write_child_output_descriptor(
            args.cell_output_descriptor_path,
            cell_id='wrong' if args.fault == 'descriptor' else args.cell_id,
            phase=args.cell_phase, logical_setting_id=args.setting,
            output_root=output, artifact_root=payload, producer_kind='benchmark_support',
            required_payload=['aggregate_summary.json', 'benchmark_rows.csv', 'support_provenance.json'])
    if args.fault == 'missing':
        (payload / 'benchmark_rows.csv').unlink()
    print(json.dumps({**monitor.report(),'dispatch_authority':dispatch_authority}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
