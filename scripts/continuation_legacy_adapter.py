"""Pinned-source adapter for the original runner's downstream transaction closure.

The public main is never called or monkeypatched. The cell dispatch closure is an
AST projection of its existing publication branch and ordinary subprocess tail;
legacy/train paths cannot be selected by this adapter's phase allowlist. The
projection receipt records exact source and selected AST identities for review.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import os
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

PHASES = ('formal_cache_policy', 'formal_controller', 'formal_ablation',
          'formal_support', 'formal_scalability', 'formal_statistics',
          'formal_gate', 'complete_without_holdout')


def load_scientific_modules(root, expected_runner_sha256, *, expected_commit):
    root = Path(root)
    if Path.cwd() != root:
        raise ValueError('scientific adapter cwd drift')
    if os.environ.get('PYTHONPATH') != str(root):
        raise ValueError('scientific adapter PYTHONPATH drift')
    if any(name == 'src' or name.startswith(('src.', 'scripts.')) for name in sys.modules):
        raise ValueError('scientific adapter requires a fresh interpreter')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != expected_commit:
        raise ValueError('scientific checkout commit drift')
    if git('diff', '--name-only', expected_commit, '--', 'src', 'scripts', 'configs'):
        raise ValueError('scientific tracked source/configuration drift')
    if git('ls-files', '--others', '--exclude-standard', 'src', 'scripts'):
        raise ValueError('untracked scientific shadow source')
    tracked = git('ls-files', 'src', 'scripts').splitlines()
    for relative in tracked:
        for part in (root / relative, *(root / relative).parents):
            if part.is_symlink():
                raise ValueError('scientific source symlink: ' + str(part))
    original = subprocess.check_output(['git', '-C', str(root), 'show',
        expected_commit + ':scripts/run_typed_model_cache_formal_protocol.py'])
    if hashlib.sha256(original).hexdigest() != expected_runner_sha256:
        raise ValueError('scientific runner hash is not from the approved commit')
    path = root / 'scripts/run_typed_model_cache_formal_protocol.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_runner_sha256:
        raise ValueError('pinned runner source drift')
    sys.path.insert(0, str(root))
    runner = importlib.import_module('scripts.run_typed_model_cache_formal_protocol')
    return runner, data


def loaded_origins(root):
    root = Path(root)
    rows = []
    for name, module in list(sys.modules.items()):
        if name == 'src' or name.startswith(('src.', 'scripts.')):
            origin = getattr(module, '__file__', None)
            if origin is None:
                raise ValueError('project module without file origin: ' + name)
            path = Path(origin)
            if root not in path.resolve().parents:
                raise ValueError('scientific module shadow: ' + name)
            rows.append({'module': name, 'path': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def transaction_projection(source, source_path):
    tree = ast.parse(source, filename=str(source_path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    closures = [node for node in ast.walk(main) if isinstance(node, ast.FunctionDef) and node.name == 'execute']
    if len(closures) != 1:
        raise ValueError('original transaction closure shape changed')
    original = closures[0]
    first, tail, result = original.body[0], original.body[-2], original.body[-1]
    if not isinstance(first, ast.If) or 'cell_artifact_publication_required' not in ast.unparse(first.test):
        raise ValueError('original publication branch shape changed')
    if not isinstance(tail, ast.Assign) or 'subprocess.run' not in ast.unparse(tail):
        raise ValueError('original subprocess tail shape changed')
    if not isinstance(result, ast.Return):
        raise ValueError('original transaction return shape changed')
    projected = ast.FunctionDef(name='execute', args=original.args,
                               body=[first, tail, result], decorator_list=[],
                               returns=original.returns, type_comment=None)
    ast.copy_location(projected, original)
    module = ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[]))
    receipt = {'source_sha256': hashlib.sha256(source).hexdigest(),
               'source_path': str(source_path), 'public_main_called': False,
               'selected_original_ast_sha256': hashlib.sha256(
                   ast.dump(ast.Module(body=[first, tail, result], type_ignores=[]),
                            include_attributes=False).encode()).hexdigest(),
               'projection': 'publication branch + subprocess tail; other legacy branches omitted',
               'allowed_phases': list(PHASES),
               'first_line': first.lineno, 'last_line': result.end_lineno}
    return compile(module, str(source_path), 'exec'), receipt


def make_cell_executor(runner, source, *, phase, protocol, context_payload,
                       execution_binding, bundle, environment, cell_ledger,
                       commands, matrix_contexts, generated_registry_audit, observe_result=None):
    if phase not in PHASES:
        raise ValueError('continuation phase not permitted')
    capabilities = runner.get_protocol_capabilities(protocol['typed_model_cache_formal_protocol_version'])
    if not capabilities.cell_artifact_publication_required:
        raise ValueError('continuation requires original publication contract')
    coordinates = {runner.canonical_sha256(list(argv)): dict(matrix_contexts[i])
                   for i, argv in enumerate(commands)}
    if len(coordinates) != len(commands):
        raise ValueError('duplicate command identity')
    namespace = dict(vars(runner))
    namespace.update(phase=phase, protocol=protocol, capabilities=capabilities,
                     resolved_context_payload=context_payload,
                     execution_binding=execution_binding, active_bundle=bundle,
                     environment_resolution=environment, cell_ledger=cell_ledger,
                     cell_transaction_phases=set(PHASES[:5]),
                     coordinate_by_command_hash=coordinates,
                     generated_registry_audit=generated_registry_audit)
    code, receipt = transaction_projection(source, Path(runner.__file__))
    exec(code, namespace)
    original_executor = namespace['execute']
    def execute(argv):
        if runner.canonical_sha256(list(argv)) not in coordinates:
            raise ValueError('dispatch outside frozen command matrix')
        result = original_executor(argv)
        if observe_result is not None:
            observe_result(phase, result)
        return result
    return execute, receipt


def original_identity_expression(source, source_path, name, namespace):
    """Evaluate a selected original identity expression with explicit frozen inputs."""
    tree = ast.parse(source, filename=str(source_path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    candidates = [node for node in ast.walk(main) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if name == 'input_hash':
        # First definition is the base identity; registry wrapping is applied below.
        candidates.sort(key=lambda node: node.lineno)
        candidates = candidates[:1]
    if len(candidates) != 1 or name not in {'input_hash', 'run_identity', 'cell_identity'}:
        raise ValueError('original identity expression is ambiguous: ' + name)
    expression = ast.Expression(body=candidates[0].value)
    result = eval(compile(expression, str(source_path), 'eval'), namespace)
    receipt = {'name': name, 'line': candidates[0].lineno,
               'expression_ast_sha256': hashlib.sha256(
                   ast.dump(expression, include_attributes=False).encode()).hexdigest()}
    return result, receipt


def run_continuation_phase(runner, source, *, phase, protocol, context_payload,
                           execution_binding, bundle, environment, generated_registry_audit,
                           admission, finalize_only=False, observe_result=None):
    """One approved phase transaction using the frozen source's actual writers.

    Caller holds the single-writer lock and has verified anchors/payloads/registry.
    An admitted phase is the transaction lease: expiry blocks the next phase, while
    this original phase transaction may finish publication and its terminal record.
    """
    if phase not in PHASES:
        raise ValueError('continuation phase not permitted')
    admission()
    context = context_payload['resolved_expansion_context']
    run_root = Path(context['output_root'])
    if phase == 'complete_without_holdout':
        runner.validate_complete_without_holdout_gate(run_root, protocol)
        command, expected_outputs, matrix_contexts, retries = [], [], [], 0
    else:
        spec = protocol['execution_contract']['command_templates'][phase]
        plan = runner.expand_command_plan(spec, context)
        command, expected_outputs, matrix_contexts = (
            plan['commands'], plan['expected_outputs'], plan['matrix_contexts'])
        retries = int(spec.get('infrastructure_retries', 1))
    context_file_hash = hashlib.sha256(
        (run_root / 'resolved_execution_context.json').read_bytes()).hexdigest()
    namespace = dict(vars(runner))
    namespace.update(phase=phase, protocol=protocol, command=command,
                     requested_output_root=str(run_root), args=SimpleNamespace(output_root=str(run_root)),
                     resolved_context_payload=context_payload,
                     resolved_context_file_sha256=context_file_hash,
                     execution_binding=execution_binding, active_bundle=bundle,
                     environment_resolution=environment)
    source_path = Path(runner.__file__)
    base_hash, input_receipt = original_identity_expression(source, source_path, 'input_hash', namespace)
    input_hash = runner.canonical_sha256({
        'base_phase_input_hash': base_hash,
        'generated_checkpoint_registry_canonical_sha256': generated_registry_audit['registry_canonical_sha256']})
    run_identity, run_receipt = original_identity_expression(source, source_path, 'run_identity', namespace)
    phase_runner = runner.TransactionalPhaseRunner(
        output_root=run_root, run_identity_fingerprint=run_identity,
        phase_order=runner.PHASE_ORDER, resume=True,
        resolved_execution_context_sha256=context_payload['context_sha256'],
        resolved_execution_context_file_sha256=context_file_hash)
    cell_ledger, cell_receipt = None, None
    if phase in PHASES[:5]:
        identity, cell_receipt = original_identity_expression(source, source_path, 'cell_identity', namespace)
        if not (run_root / 'cell_ledger_identity.json').is_file():
            raise ValueError('continuation requires original cell ledger identity')
        cell_ledger = runner.FormalCellLedger(run_root=run_root, identity=identity, resume=True)
    commands = command if command and isinstance(command[0], list) else ([command] if command else [])
    execute, dispatch_receipt = make_cell_executor(
        runner, source, phase=phase, protocol=protocol, context_payload=context_payload,
        execution_binding=execution_binding, bundle=bundle, environment=environment,
        cell_ledger=cell_ledger, commands=commands, matrix_contexts=matrix_contexts,
        generated_registry_audit=generated_registry_audit, observe_result=observe_result)
    if finalize_only:
        result = phase_runner.finalize_phase_only(
            phase, commands=commands, input_hash=input_hash, expected_outputs=expected_outputs)
    else:
        result = phase_runner.run_phase(
            phase, commands=commands, input_hash=input_hash, expected_outputs=expected_outputs,
            executor=execute, infrastructure_retries=retries)
    if cell_ledger is not None:
        cell_ledger.assert_complete_matrix(
            phase=phase, expected_cell_ids=[runner.stable_cell_id(phase, row) for row in matrix_contexts])
    return {'result': result, 'identity_receipts': [input_receipt, run_receipt, cell_receipt],
            'dispatch_projection_receipt': dispatch_receipt}
