"""Operational fixes must not change scientific training dispatch or budgets."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("continuation", ROOT / "scripts/continue_fair_reward_training.py")
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def test_invalid_utf8_does_not_hide_competing_training():
    raw = (b"1 /bin/zsh /bin/zsh -c python train_big.py\n"
           b"2 /path/Python /path/Python scripts/train_other.py --name bad\xff\xfe\n"
           b"3 /path/Python /path/Python scripts/continue_fair_reward_training.py run\n"
           b"4 /path/Python /path/Python scripts/check_data_ready.py --name \xe7\n"
           b"5 /path/Python /path/Python scripts/benchmark_main_results.py\n")
    result = continuation.foreign_training(raw, 3)
    assert [x['pid'] for x in result] == [2, 5]
    assert "\\xff" in result[0]['command']


def test_unbalanced_argument_quotes_do_not_hide_training_script():
    raw = b"9 /path/Python /path/Python scripts/train_other.py --label can't\n"
    assert continuation.foreign_training(raw, 3)[0]['pid'] == 9


def test_malformed_nonprocess_lines_and_own_process_are_ignored():
    assert continuation.foreign_training(b"nonsense\nnot_a_pid a b\n3 /path/Python /path/Python train_own.py\n", 3) == []


def test_resource_scan_failure_waits_closed_then_recovers(tmp_path, monkeypatch):
    calls, waits = [], []
    def scan(command):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)
        return b"1 /bin/zsh /bin/zsh\n"
    def sleep(seconds):
        assert json.loads((tmp_path/'resource_wait.json').read_text())['status'] == 'RESOURCE_SCAN_UNAVAILABLE'
        waits.append(seconds)
    monkeypatch.setattr(continuation.subprocess, 'check_output', scan)
    monkeypatch.setattr(continuation.time, 'sleep', sleep)
    continuation.yield_resources(tmp_path)
    assert len(calls) == 2 and waits == [5]
    assert json.loads((tmp_path/'resource_wait.json').read_text())['status'] == 'RESOURCE_AVAILABLE'


def test_foreign_training_yields_without_touching_that_process(tmp_path, monkeypatch):
    replies = iter([b"99999 /path/Python /path/Python scripts/train_foreign.py --x \xff\n", b""])
    monkeypatch.setattr(continuation.subprocess, 'check_output', lambda command: next(replies))
    waits = []
    monkeypatch.setattr(continuation.time, 'sleep', lambda seconds: waits.append(seconds))
    continuation.yield_resources(tmp_path)
    assert waits == [5]


def test_inventory_detects_changed_or_added_payload(tmp_path):
    (tmp_path/'a').write_bytes(b'original')
    items = continuation.inventory(tmp_path)
    continuation.verify_inventory(tmp_path, items)
    (tmp_path/'a').write_bytes(b'modified')
    with pytest.raises(ValueError): continuation.verify_inventory(tmp_path, items)
    (tmp_path/'a').write_bytes(b'original')
    (tmp_path/'b').write_bytes(b'extra')
    with pytest.raises(ValueError): continuation.verify_inventory(tmp_path, items)


def contract():
    return {'inherited':[{'agent':'ppo','seed':7}],
            'pending':[{'agent':'ppo','seed':13,'initialization':'fresh_no_checkpoint_load'}],
            'contract_sha256':'operational'}


def test_completed_parent_cell_never_enters_scientific_trainer(tmp_path):
    def fail(*args):raise AssertionError('inherited cell must not retrain or load weights')
    continuation.dispatch(fail, contract(), Path('/source'), tmp_path, {'plan_sha256':'same'}, 'ppo',7,[],[],[])
    assert not (tmp_path/'attempt_receipts').exists()


def test_pending_cell_delegates_identical_plan_and_original_arguments(tmp_path):
    p={'plan_sha256':'same','training_steps':196608,'training_cycles':32,'reward_positive_offset':0}
    train, valid, workflows = [object()], [object()], [object()]
    seen=[]
    def original(*args): seen.append(args)
    continuation.dispatch(original,contract(),Path('/source'),tmp_path,p,'ppo',13,train,valid,workflows)
    assert seen[0] == (Path('/source'),tmp_path,p,'ppo',13,train,valid,workflows)
    assert seen[0][2] is p
    receipt=json.loads((tmp_path/'attempt_receipts/ppo_seed13.json').read_text())
    assert receipt['fresh_start'] and not receipt['checkpoint_load']
    with pytest.raises(FileExistsError):
        continuation.dispatch(original,contract(),Path('/source'),tmp_path,p,'ppo',13,train,valid,workflows)
    assert len(seen)==1


def test_existing_partial_cell_in_new_root_cannot_be_overwritten(tmp_path):
    (tmp_path/'cells/ppo/seed13').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        continuation.dispatch(lambda *a:None,contract(),Path('/source'),tmp_path,{},'ppo',13,[],[],[])


def test_unknown_cell_is_not_silently_appended_to_budget(tmp_path):
    with pytest.raises(ValueError):
        continuation.dispatch(lambda *a:None,contract(),Path('/source'),tmp_path,{},'ppo',29,[],[],[])
