"""Read-only source models through real evaluation parsers and resource loaders.

Only context/contract/reference JSON is written below pytest's isolated directory.
No authorization, rollout, weights, run ledger, or production output is created.
"""
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import benchmark_main_results as benchmark
from scripts.run_typed_model_cache_formal_cache_policy import build_parser
from src.runtime.evaluation_only_execution import (
    build_evaluation_execution_contract, validate_command_matrix_parsers,
    EvaluationOnlyError,
)
from src.runtime.generated_checkpoint_resources import (
    GeneratedCheckpointResourceError, evaluation_model_source_scope,
    resolve_generated_checkpoint_arguments,
)
from tests.test_evaluation_only_source_boundaries import source_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def consumer(tmp_path, source_reference):
    root = tmp_path / 'evaluation_only_fixture'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    contract = build_evaluation_execution_contract(
        source_reference=source_reference, evaluation_run_id=root.name,
        evaluation_run_root=root, executor_checkout=ROOT, executor_commit=commit,
        python_executable=sys.executable,
    )
    from scripts.continuation_executor.locking import writer_lock_path
    assert contract["lock_root"] == str(writer_lock_path(root).parent)
    root.mkdir()
    for name, payload in (
        ('evaluation_model_source_reference.json', source_reference),
        ('evaluation_execution_contract.json', contract),
        ('resolved_execution_context.json', contract['evaluation_execution_context']),
    ):
        (root/name).write_text(json.dumps(payload))
    return root, source_reference, contract


def parse(command):
    original = sys.argv
    try:
        sys.argv = list(command[1:])
        return benchmark.parse_args()
    finally:
        sys.argv = original


@pytest.mark.parametrize('phase', ['formal_cache_policy', 'formal_controller'])
def test_real_outer_nested_loaders_three_capacities(consumer, phase):
    root, source, contract = consumer
    audit = validate_command_matrix_parsers(contract, model_source_reference=source)
    assert audit['parsed_command_count'] == 24 and audit['nested_command_count'] == 3
    observed = set()
    protocol = json.loads(Path(source['protocol_path']).read_text())
    for command in contract['command_plans'][phase]['commands']:
        if phase == 'formal_cache_policy':
            outer = build_parser().parse_args(command[2:])
            command = outer.command
            if command[0] == '--':
                command = command[1:]
        args = parse(command)
        result = resolve_generated_checkpoint_arguments(args, protocol=protocol)
        assert result['evaluation_model_source']['evaluation_only']
        assert json.loads(json.dumps(result)) == result
        assert result['manifest']['checkpoint_count'] == 50
        assert result['provenance']['checkpoint_count'] == 50
        observed.add(result['capacity_label'])
    assert observed == {'constrained_288mb', 'medium_576mb', 'relaxed_864mb'}
    assert not (root/'phase_state.jsonl').exists()


@pytest.mark.parametrize('mutation', ['missing_context', 'missing_contract', 'wrong_run',
    'wrong_executor', 'stale_context_hash', 'wrong_registry', 'wrong_source_binding', 'wrong_companion'])
def test_real_evaluation_loader_rejects_identity_and_path_drift(consumer, mutation):
    root, source, contract = consumer
    args = parse(contract['command_plans']['formal_controller']['commands'][0])
    context_path = root/'resolved_execution_context.json'
    context = deepcopy(contract['evaluation_execution_context'])
    if mutation == 'missing_context': args.resolved_execution_context_path = ''
    elif mutation == 'missing_contract': (root/'evaluation_execution_contract.json').unlink()
    elif mutation == 'wrong_run': context['evaluation_execution_identity']['evaluation_run_id'] = 'other'
    elif mutation == 'wrong_executor': context['evaluation_execution_identity']['executor_commit'] = '0'*40
    elif mutation == 'stale_context_hash': context['context_sha256'] = '0'*64
    elif mutation == 'wrong_registry': args.generated_checkpoint_registry_path = str(root/'absent_registry.json')
    elif mutation == 'wrong_source_binding': args.formal_training_execution_binding_path = str(root/'wrong_binding.json')
    elif mutation == 'wrong_companion': args.seed_checkpoint_manifest_path = str(root/'wrong_manifest.json')
    context_path.write_text(json.dumps(context))
    protocol = json.loads(Path(source['protocol_path']).read_text())
    with pytest.raises((GeneratedCheckpointResourceError, EvaluationOnlyError, FileNotFoundError)):
        resolve_generated_checkpoint_arguments(args, protocol=protocol)
    assert not (root/'phase_state.jsonl').exists()


@pytest.mark.parametrize('phase', ['formal_statistics', 'formal_gate'])
def test_real_statistics_and_gate_reject_missing_evaluation_identity(consumer, phase):
    root, source, contract = consumer
    (root/'evaluation_execution_contract.json').unlink()
    result = subprocess.run(contract['command_plans'][phase]['commands'][0],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'evaluation_execution_contract.json' in result.stderr, result.stderr
    assert not (root/'formal_gate.json').exists()
    assert not (root/'phase_state.jsonl').exists()


def test_default_source_scope_is_json_serializable(tmp_path):
    from argparse import Namespace
    scope = evaluation_model_source_scope(Namespace(), default_run_root=tmp_path)
    assert json.loads(json.dumps(scope)) == scope


def test_live_contract_rejects_wrong_shared_lock_path(consumer):
    from src.runtime.evaluation_only_execution import canonical_sha256, validate_execution_contract
    root, source, contract = consumer
    changed = deepcopy(contract)
    changed['lock_root'] = str(root.parent / '.wrong_lock_directory')
    changed['execution_contract_sha256'] = canonical_sha256(
        {key: value for key, value in changed.items() if key != 'execution_contract_sha256'})
    with pytest.raises(EvaluationOnlyError, match='lock root'):
        validate_execution_contract(changed, model_source_reference=source)
