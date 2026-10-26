"""The diagnosis samples inputs without consuming dev outcomes or changing costs."""
import itertools
from copy import deepcopy

from scripts.probe_stream_main_opportunity import select_train_cases, run_episode
from src.envs.core.joint_stream_rules import StreamCacheRule
from tests.test_stream_control_learning import env


def test_case_selection_is_balanced_and_ignores_dev_and_scores():
    data = dict(sessions=[{'split': 'train'} for _ in range(3)], cases={'train': [], 'dev': None})
    for session, size, mode, families, capacity in itertools.product(range(3),
            ['short', 'medium', 'long'], ['queued_batch', 'stream'], [1, 2], [4096, 8192]):
        data['cases']['train'].append(dict(case_id=str(len(data['cases']['train'])),
            session=session, size=size, arrival_mode=mode, families=families, capacity=capacity))
    chosen = select_train_cases(data)
    assert len(chosen) == 12
    assert all(sum(c['session'] == s for c in chosen) == 4 for s in range(3))
    changed = deepcopy(data)
    for c in changed['cases']['train']:
        c['reward'] = -1000000 if c in chosen else 1000000
    assert [c['case_id'] for c in select_train_cases(changed)] == [c['case_id'] for c in chosen]


def test_probe_retains_all_native_costs_and_frame_receipts(tmp_path):
    import json
    row = run_episode(env(), StreamCacheRule('reactive'), tmp_path/'trace.jsonl', learned=False)
    frames = [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
    assert len(frames) == row['steps']
    assert abs(sum(f['reward'] for f in frames) - row['reward']) < 1e-8
    assert abs(sum(row['components'].values()) - row['reward']) < 1e-8
    assert sum(len(f['events']) for f in frames) == sum(row['statuses'].values())


def test_comparison_uses_one_training_rule_not_per_cell_oracle():
    from scripts.audit_stream_main_opportunity import compare
    rows=[]
    for method,values in {'reactive':[10,0], 'naive_prefetch':[0,8],
            'gain_prefetch':[4,4], 'sa_seed7':[6,6]}.items():
        for i,value in enumerate(values):
            rows.append(dict(method=method,case_id=str(i),reward=value,
                completed_requests=1,components={'completed_nodes':value},statuses={}))
    result=compare(rows)
    assert result['strongest_compatible_rule']=='reactive'
    assert result['mean_main_minus_rule']==1
    import pytest
    with pytest.raises(ValueError,match='unmatched'):
        compare(rows[:-1])
