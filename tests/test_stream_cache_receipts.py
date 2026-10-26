"""Reject capacity and loading violations independently of reward summaries."""
from copy import deepcopy
import json
import pytest

from scripts.audit_stream_cache_receipts import CacheLedger
from scripts.probe_stream_main_opportunity import run_episode
from src.envs.core.joint_stream_rules import StreamCacheRule
from tests.test_stream_control_learning import env


def sample(tmp_path):
    e=env();case=dict(catalog=e.catalog_objects,arrivals=e.arrivals,
        options={'model_cache_config':{'capacity_bytes':8}})
    run_episode(e,StreamCacheRule('reactive'),tmp_path/'trace',learned=False)
    frames=[json.loads(s) for s in (tmp_path/'trace').read_text().splitlines()]
    return case,frames


def test_real_receipts_reconstruct_pending_and_resident_byte_reservations(tmp_path):
    case,frames=sample(tmp_path);ledger=CacheLedger(case)
    for frame in frames:ledger.consume(frame)
    result=ledger.summary()
    assert result['counts']['verified_compute_events']>0
    assert result['counts']['completed_loads']>0
    assert set(result['objects'])=={'base_model','adapter'}


def test_reward_unchanged_cannot_hide_early_service_or_capacity_violation(tmp_path):
    case,frames=sample(tmp_path)
    early=deepcopy(frames[0]);early['events'][0]['status']='rsu_compute'
    with pytest.raises(AssertionError,match='served before'):
        CacheLedger(case).consume(early)
    undersized=deepcopy(case);undersized['options']['model_cache_config']['capacity_bytes']=1
    with pytest.raises(AssertionError,match='capacity'):
        CacheLedger(undersized).consume(frames[0])
