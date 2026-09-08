"""Read-only proposal, evidence and execution-authorization checks; stdout only."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Load this implementation by exact file, independent of an ambient src package.
spec = importlib.util.spec_from_file_location('continuation_validator', ROOT / 'src/runtime/fixed_commit_continuation.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True)
    parser.add_argument('--check', choices=['structure', 'evidence', 'authorization'], default='evidence')
    args = parser.parse_args(argv)
    try:
        proposal = contract.load(args.proposal)
        report = contract.preflight(proposal, args.check)
        if args.check != 'structure' and not any(x['status'] == 'fail' for x in report['checks']):
            protected = [row for row in proposal['evidence'] if row['role'] == 'committed_output']
            if len(protected) != 1 or not protected[0]['path'].endswith('.json.gz'):
                raise ValueError('CLI requires one complete protected JSON gzip inventory')
            for row in protected:
                if row['role'] == 'committed_output' and row['path'].endswith('.json.gz'):
                    report['protected_inventory'] = contract.validate_protected_inventory(
                        row['path'], immutable_roots=[proposal['run_root'], proposal['worktree_root']])
                    report['committed_payloads'] = contract.validate_committed_payloads(
                        proposal['run_root'], row['path'])
            ctx = contract.load(Path(proposal['run_root']) / 'resolved_execution_context.json')
            if os.path.abspath(ctx['runtime_location']['resolved_python_absolute_path']) != os.path.abspath(sys.executable):
                raise ValueError('CLI must use the original context Python; refusing arbitrary interpreter')
            env = dict(os.environ, PYTHONPATH=proposal['worktree_root'], PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',
                       GIT_OPTIONAL_LOCKS='0')
            env.pop('PYTHONHOME', None)
            env.pop('PYTHONSTARTUP', None)
            completed = subprocess.run([
                ctx['runtime_location']['resolved_python_absolute_path'], '-I', '-B',
                str(ROOT / 'scripts/probe_fixed_commit_continuation.py')],
                input=json.dumps(proposal), text=True, capture_output=True,
                cwd=proposal['worktree_root'], env=env, check=False)
            if completed.returncode:
                raise ValueError('read-only probe failed: ' + completed.stderr)
            report['scientific_probe'] = contract.strict_json(completed.stdout)
            report['current_main_observation'] = {
                'head': contract.git(ROOT, 'rev-parse', 'HEAD'),
                'origin_main': contract.git(ROOT, 'rev-parse', 'origin/main'),
                'scope': 'observation only; cannot rebind original context/checkpoints'}
        failures = any(x['status'] == 'fail' for x in report['checks'])
        failures |= any(x['status'] == 'fail' and x['name'] != 'original_publication_gate_current'
                        for x in report.get('scientific_probe', {}).get('checks', []))
        report['qualification'] = 'fail' if failures else ('structure_only' if args.check == 'structure' else 'unavailable')
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if failures or args.check == 'authorization' else (0 if args.check == 'structure' else 3)
    except Exception as exc:
        print(json.dumps({'error': str(exc), 'execution_authorized': False,
                          'dispatch_count': dict(train=0, dev=0, formal=0, holdout=0)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
