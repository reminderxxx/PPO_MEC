"""No subprocess science: one campaign claim and failure propagation are isolated."""
import json
from pathlib import Path

import pytest

import scripts.launch_stream_reward_curve as launcher


@pytest.mark.parametrize('field,value', [('maximum_total_physical_frames',1393921),
    ('maximum_scientific_launches',2),('parallel_workers',4),('replay_episodes',[]),
    ('probe_case_ids',['selected_after_rewards'])])
def test_campaign_envelope_refuses_expansion_before_workers(tmp_path,field,value):
    plan=dict(maximum_total_physical_frames=1393920,maximum_artifact_bytes=10737418240,
        maximum_rule_frames=7200,maximum_replay_frames=4320,maximum_scientific_launches=1,
        parallel_workers=3,replay_episodes=launcher.REPLAY_EPISODES,
        probe_case_ids=['train_000','train_038','train_067'])
    plan[field]=value;path=tmp_path/'campaign.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='envelope'):
        launcher.check_campaign(plan,path,launcher.file_hash(path))


def test_one_failed_cell_stops_peers_preserves_failure_and_never_retries(tmp_path,monkeypatch):
    root=tmp_path/'campaign';path=tmp_path/'plan.json'
    plan=dict(root=str(root),parallel_workers=3,maximum_artifact_bytes=10737418240,
        global_deadline_unix=10**20,source_commit='fixture',outputs={m:{str(s):str(root/m/str(s))
        for s in launcher.SEEDS} for m in launcher.METHODS})
    path.write_text(json.dumps(plan));started=[];terminated=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.pid=100000+len(started);self.returncode=1 if not started else None;started.append(self)
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=-15;return -15
    monkeypatch.setattr(launcher,'check_campaign',lambda *args:None)
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    monkeypatch.setattr(launcher.os,'killpg',lambda pid,sig:terminated.append(pid))
    with pytest.raises(RuntimeError,match='preserved cell failure'):
        launcher.run(path,launcher.file_hash(path))
    assert len(started)==3 and set(terminated)=={p.pid for p in started[1:]}
    failure=json.loads((root/'failure.json').read_text())
    assert failure['status']=='failed_preserved' and failure['completed_cells']==[]
    assert (tmp_path/'campaign_execution_claim.json').exists()
    original=(root/'failure.json').read_bytes()
    with pytest.raises(FileExistsError):launcher.run(path,launcher.file_hash(path))
    assert len(started)==3 and (root/'failure.json').read_bytes()==original
