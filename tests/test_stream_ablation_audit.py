"""Accounting and pairing failures must not turn into favorable evidence."""
import json
import pytest

from scripts.audit_stream_main_ablation import summarize,verify_evaluation
from scripts.probe_stream_main_opportunity import run_episode,file_hash
from src.envs.core.joint_stream_rules import StreamCacheRule
from tests.test_stream_control_learning import env


def test_native_accounting_is_checked_even_with_updated_file_hash(tmp_path):
    path=tmp_path/'final_case.jsonl'
    row=run_episode(env(),StreamCacheRule('reactive'),path,learned=False)
    row.update(phase='final',case_id='case')
    assert verify_evaluation(tmp_path,row)[0]==row['steps']
    lines=path.read_text().splitlines();first=json.loads(lines[0])
    first['receipt']['reward_components']['unfinished_wait']+=1
    lines[0]=json.dumps(first);path.write_text('\n'.join(lines)+'\n')
    row['trace_sha256']=file_hash(path)
    with pytest.raises(AssertionError):verify_evaluation(tmp_path,row)


def test_optimization_and_learning_increment_are_separated_and_negative_kept():
    rows=[]
    for variant,initial,final in [('original',1,2),('relocation',5,4),('no_dag_edges',5,6)]:
        for seed in [7,17]:
            for cid in ['a','b']:
                for phase,value in [('initial',initial),('final',final)]:
                    rows.append(dict(variant=variant,seed=seed,phase=phase,case_id=cid,reward=value,
                        completed_requests=1,counts={'service_target_changed':0},statuses={},components={'completed_nodes':value}))
    rules={cid:{'reward':3} for cid in ['a','b']}
    report=summarize(rows,rules)
    assert report['optimization_effect']==2
    assert report['optimized_minus_rule']==1
    assert report['methods']['relocation']['learned_increment']==-1
    assert report['module_full_minus_deleted']['no_dag_edges']==-2
    with pytest.raises(ValueError,match='duplicate'):
        summarize(rows+[rows[0]],rules)
