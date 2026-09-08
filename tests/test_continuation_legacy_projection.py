from pathlib import Path
import ast
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('external_adapter', ROOT / 'scripts/continuation_legacy_adapter.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def test_projection_selects_exact_original_branch_without_calling_main():
    path = ROOT / 'scripts/run_typed_model_cache_formal_protocol.py'
    source = path.read_bytes()
    code, receipt = adapter.transaction_projection(source, path)
    assert receipt['public_main_called'] is False
    assert receipt['allowed_phases'] == list(adapter.PHASES)
    assert code.co_filename == str(path)
    assert receipt['first_line'] < receipt['last_line']
    tree = ast.parse(source)
    original = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    calls = []
    class Subprocess:
        def run(self, argv, **kwargs):
            calls.append((argv, kwargs))
            return type('Result', (), {'returncode': 0, 'stdout': 'ok', 'stderr': ''})()
    namespace = {'capabilities': type('Caps', (), {'cell_artifact_publication_required': True})(),
                 'phase': 'formal_statistics', 'cell_transaction_phases': set(adapter.PHASES[:5]),
                 'subprocess': Subprocess(), 'ROOT': ROOT,
                 'environment_resolution': type('Env', (), {'child_environment': {'frozen': 'yes'}})(),
                 'PhaseCommandResult': lambda *args: args}
    exec(code, namespace)
    assert namespace['execute'](['python', 'statistics']) == (0, 'ok', '')
    assert calls == [(['python', 'statistics'], {'cwd': ROOT, 'env': {'frozen': 'yes'},
                                               'text': True, 'capture_output': True, 'check': False})]
    assert original.body[0].lineno == receipt['first_line']


def test_projection_fails_closed_on_unknown_source_shape():
    with pytest.raises(ValueError):
        adapter.transaction_projection(b'def main():\n    pass\n', Path('unknown.py'))
