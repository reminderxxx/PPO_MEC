"""Stage4 pure software tests: no environment; <=128 actual software Adam calls."""
from copy import deepcopy
from datetime import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

TRAIN_PATH=ROOT/'scripts/train_sa_group_clip_pair.py'
AUDIT_PATH=ROOT/'scripts/audit_sa_group_clip_pair.py'
LAUNCH_PATH=ROOT/'scripts/launch_sa_group_clip_pair.py'
train=module('stage4_train_under_test',TRAIN_PATH)
audit=module('stage4_audit_under_test',AUDIT_PATH)
launch=module('stage4_launch_under_test',LAUNCH_PATH)
helper=module('stage3_synthetic_helpers',ROOT/'tests/test_sa_separate_gradient_clip.py')

@pytest.fixture(scope='module',autouse=True)
def guard(request):
 from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
 from src.envs.core.multi_request_replay_env import MultiRequestReplayEnv
 counts=dict(adam_steps=0,limit=128,environment_calls=0,scientific_training_sessions=0)
 patch=pytest.MonkeyPatch();original=torch.optim.Adam.step
 def counted(this,*args,**kwargs):
  assert counts['adam_steps']<128
  counts['adam_steps']+=1
  return original(this,*args,**kwargs)
 def forbidden(*args,**kwargs):
  counts['environment_calls']+=1
  raise AssertionError('Stage4 software tests forbid all environment calls')
 patch.setattr(torch.optim.Adam,'step',counted)
 for cls in (MultiRequestStreamEnv,MultiRequestReplayEnv):
  for name in ('__init__','reset','step','step_controls'):
   if hasattr(cls,name):patch.setattr(cls,name,forbidden)
 old_threads=torch.get_num_threads();state=train.rng_state();torch.set_num_threads(1)
 yield counts
 patch.undo();torch.set_num_threads(old_threads);train.restore_rng(state)
 assert counts['adam_steps']<=128 and counts['environment_calls']==0
 if os.environ.get('SA_STAGE4_TEST_RECEIPT'):
  path=Path(os.environ['SA_STAGE4_TEST_RECEIPT']).resolve();assert path.is_relative_to(Path('/tmp').resolve())
  with path.open('x') as f:json.dump(dict(counts,pytest_failures=request.session.testsfailed,synthetic_only=True),f,indent=2)

@pytest.mark.parametrize('enabled',[False,True])
def test_real_default_update_observer_is_exact_over_two_updates(enabled):
 plain=helper.agent(enabled);observed=helper.agent(enabled)
 rows=helper.synthetic_rollouts(plain)
 for update in (1,2):
  random.seed(900+update);np.random.seed(900+update);torch.manual_seed(900+update)
  state=train.rng_state();expected=plain.update(deepcopy(rows));after=train.rng_state()
  train.restore_rng(state);mb=io.StringIO();fw=io.StringIO()
  actual,report=train.observe_update(observed,deepcopy(rows),identity=dict(seed=17,episode=update*4),minibatch_file=mb,forward_file=fw)
  helper.assert_exact(expected,actual);helper.assert_exact(plain.state_dict(),observed.state_dict())
  helper.assert_exact(plain.optimizer.state_dict(),observed.optimizer.state_dict())
  helper.assert_exact(plain.surrogate_optimizer.state_dict(),observed.surrogate_optimizer.state_dict())
  helper.assert_exact(helper.grad_snapshot(plain),helper.grad_snapshot(observed))
  assert train.object_hash(after)==train.object_hash(train.rng_state())
  assert report['actual_backwards']==actual['gradient_steps']
  assert report['additional_backward_calls']==report['additional_optimizer_steps']==0
  assert report['actual_adam_steps']['main']==report['actual_adam_steps']['surrogate']==report['actual_backwards']
  assert report['all_backward_rng_unchanged']
  lines=[json.loads(s) for s in mb.getvalue().splitlines()];forwards=[json.loads(s) for s in fw.getvalue().splitlines()]
  assert len(lines)==report['actual_backwards'] and len(forwards)==report['actual_forward_calls']
  for line in lines:
   assert line['finite'] and line['rng_unchanged']
   assert len(line['clip_calls'])==(3 if enabled else 2)
   assert [x['optimizer'] for x in line['adam_steps']]==['main','surrogate']
   for group in ('policy','value','surrogate'):assert line['postclip'][group]['norm']<=.500001
  assert any(x['choice_coverage'][0]['forced_rows']>0 for x in forwards)
  assert any(x['choice_coverage'][0]['multichoice_rows']>0 for x in forwards)

@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('kind',['unused','zero','three_groups'])
def test_none_and_zero_observation_does_not_insert_optimizer_states(enabled,kind):
 plain=helper.agent(enabled);observed=helper.agent(enabled)
 def loss(model):
  p=dict(model.named_parameters())
  if kind=='unused':return torch.tensor(0.,requires_grad=True)
  if kind=='zero':return next(model.parameters()).sum()*0
  return .3*p['heads.0.0.weight'].flatten()[0]+2*p['critic.0.weight'].flatten()[0]+.2*p['surrogates.0.0.weight'].flatten()[0]
 plain._apply_control_gradients(loss(plain));state=train.rng_state()
 original_clip=torch.nn.utils.clip_grad_norm_;original_step=observed.optimizer.step
 with train.GradientObserver(observed) as observer:observed._apply_control_gradients(loss(observed))
 assert torch.nn.utils.clip_grad_norm_ is original_clip and observed.optimizer.step==original_step
 helper.assert_exact(plain.state_dict(),observed.state_dict())
 helper.assert_exact(plain.optimizer.state_dict(),observed.optimizer.state_dict())
 helper.assert_exact(plain.surrogate_optimizer.state_dict(),observed.surrogate_optimizer.state_dict())
 assert train.object_hash(state)==train.object_hash(train.rng_state())
 assert observer.summary()['actual_backwards']==1
 if kind=='unused':assert observed.optimizer.state_dict()['state']=={}


def test_observer_failure_restores_methods_and_global_clip():
 model=helper.agent(True);before=deepcopy(model.state_dict());step=model.optimizer.step;clip=torch.nn.utils.clip_grad_norm_
 with pytest.raises(FloatingPointError):
  with train.GradientObserver(model):model._apply_control_gradients(next(model.parameters()).sum()*float('nan'))
 assert model.optimizer.step==step and torch.nn.utils.clip_grad_norm_ is clip
 helper.assert_exact(before,model.state_dict());assert model.optimizer.state_dict()['state']=={}


def test_rng_scope_restores_python_numpy_torch_on_exception():
 random.seed(23);np.random.seed(23);torch.manual_seed(23);before=train.rng_hash()
 with pytest.raises(RuntimeError):
  with train.preserved_rng():random.random();np.random.rand();torch.rand(3);raise RuntimeError('software sentinel')
 assert train.rng_hash()==before


def metric(reward=210.,completion=90.,cost=6.):
 return dict(reward=reward,request_completion_percent=completion,model_transfer_kib=96.,model_kib_per_completed_request=cost)

def test_predeclared_tail_stability_each_seed_and_undefined_cost():
 curves={arm:{str(ep):{scope:metric() for scope in ('overall','seed7','seed17','seed27')} for ep in (48,56,64)} for arm in audit.ARMS}
 thresholds=dict(episodes=[48,56,64],reward_relative_range=.01,completion_percentage_point_range=1.,model_cost_relative_range=.02)
 assert all(x['passed'] for x in audit.tail_stability(curves,thresholds)['sa_group_clip'].values())
 curves['sa_group_clip']['56']['seed17']['reward']=214.
 curves['sa_group_clip']['64']['seed27']['model_kib_per_completed_request']=None
 result=audit.tail_stability(curves,thresholds)['sa_group_clip']
 assert result['overall']['passed'] and not result['seed17']['passed'] and not result['seed27']['passed']
 with pytest.raises(ValueError):audit.tail_stability(curves,dict(thresholds,reward_relative_range=.02))


def test_nonregression_keeps_cost_and_completion_even_with_reward_gain():
 reference=metric();candidate=metric(reward=220.)
 assert audit.nonregression(candidate,reference)['passed']
 candidate['model_kib_per_completed_request']=6.0000001
 assert not audit.nonregression(candidate,reference)['passed']
 candidate=metric(reward=220.,completion=89.9)
 assert not audit.nonregression(candidate,reference)['passed']


def test_contract_mutation_rejected_before_model_or_environment(tmp_path):
 p=tmp_path/'contract.json';p.write_text(json.dumps(dict(schema='sa_group_clip_pair_v1',status='frozen',repo_root=str(ROOT),output_root=str(tmp_path),primary_checkpoint=64,arms=train.ARMS,seeds=[7,17,27],episodes=65,checkpoint_episodes=train.POINTS)))
 with pytest.raises(RuntimeError,match='Unexpected seed'):train.verify_contract(p,root=ROOT)


def test_exclusive_start_marker_prevents_second_launch(tmp_path):
 out=tmp_path/'paired';out.mkdir();marker=out/'run_started.json';marker.write_text('{"preserved":true}')
 with pytest.raises(FileExistsError):launch.run(SimpleNamespace(output_root=out))
 assert json.loads(marker.read_text())=={'preserved':True}
 assert not (out/'arms').exists()


def test_coordinator_stops_peer_on_failed_arm_without_real_processes(tmp_path,monkeypatch):
 out=tmp_path/'paired';out.mkdir()
 monkeypatch.setattr(launch,'validate_run',lambda p:dict(source_commit='frozen',arm_output_roots={a:str(out/'arms'/a) for a in launch.ARMS}))
 (out/'contract.json').write_text('{}')
 class FakeProcess:
  nextpid=30001
  def __init__(self,*args,**kwargs):
   self.pid=FakeProcess.nextpid;FakeProcess.nextpid+=1;self.returncode=7 if self.pid==30001 else None
  def poll(self):return self.returncode
  def wait(self,timeout=None):self.returncode=-15 if self.returncode is None else self.returncode;return self.returncode
 processes=[]
 def fake_popen(*args,**kwargs):
  p=FakeProcess();processes.append(p);return p
 killed=[]
 monkeypatch.setattr(launch.subprocess,'Popen',fake_popen)
 def killpg(pid,sig):
  killed.append(pid)
  next(p for p in processes if p.pid==pid).returncode=-15
 monkeypatch.setattr(launch.os,'killpg',killpg)
 with pytest.raises(RuntimeError,match='Paired arm failed'):launch.run(SimpleNamespace(output_root=out))
 assert killed==[30002] and len(processes)==2
 failure=json.loads((out/'failure.json').read_text());assert failure['status']=='failed' and failure['phase']=='training_and_evaluation'
 assert not (out/'audit').exists()


def test_full_panel_consumes_public_hashes_without_internal_adam_metadata():
 arm='sa_legacy_clip';cases={f'dev_{n:03d}':{} for n in range(48)}
 cp={(seed,ep):dict(tensor_sha256='tensor',actor_sha256='actor',_optimizer_steps={'main':{}}) for seed in audit.SEEDS for ep in audit.POINTS}
 data=[dict(seed=seed,episode=ep,case_id=case,arm=arm,agent='sa_ghmappo',reward=1.,steps=1,request_count=1,completed_requests=1,completed_nodes=1,state_transfer_bytes=0,model_transfer_link_frames=0,decision_rows=1,tensor_sha256='tensor',actor_sha256='actor') for seed,ep in cp for case in cases]
 result=audit.audit_panel(data,arm,cp,cases,{})
 assert result['sessions']==1296 and result['frames']==1296
 with pytest.raises(ValueError,match='incomplete/extra'):audit.audit_panel(data[:-1],arm,cp,cases,{})
 with pytest.raises(ValueError,match='duplicate'):audit.audit_panel(data+[data[0]],arm,cp,cases,{})
 data[0]['tensor_sha256']='wrong'
 with pytest.raises(ValueError,match='hash mismatch'):audit.audit_panel(data,arm,cp,cases,{})
