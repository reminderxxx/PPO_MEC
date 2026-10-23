"""Value route contract witnesses; constructed fixtures, no real-data rollouts."""
import copy,io
import pytest
import torch
from tests.test_sagh_auxiliary_execution import aligned
from tests.test_sagh_execution_repair import state

D='block_value_to_encoder_v1'
def gradients(a,loss):
 named=list(a._network.named_parameters());g=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True)
 return {n:torch.zeros_like(p) if v is None else v for (n,p),v in zip(named,g)}

@pytest.mark.parametrize('fusion,memory',[(False,False),(True,False),(False,True),(True,True)])
def test_value_route_preserves_every_nonencoder_gradient_and_scalar(fusion,memory):
 a=aligned(value_encoder_gradient_contract=D,digital_twin_handoff_fusion_enabled=fusion,outcome_memory_fusion_enabled=memory)
 info=state();o=a._forward_policy(a._extract_semantic_state(info),run_metadata=info['run_metadata']);raw=(o['value']-10.)**2
 before=copy.deepcopy(a._network.state_dict());routed=a._route_value_encoder_gradients(raw)
 assert torch.equal(raw.detach(),routed.detach())
 gr=gradients(a,raw);gd=gradients(a,routed);other=0
 for name in gr:
  if name.startswith('encoder.'):
   assert torch.count_nonzero(gd[name])==0
  else:
   assert torch.equal(gr[name],gd[name]),name
   if not any(t in name for t in ['critic','value']) and gr[name].abs().sum()>0:other+=1
 assert sum(float(v.abs().sum()) for k,v in gr.items() if k.startswith('encoder.'))>0
 if fusion or memory:assert other>0,'Must cover shared_other, not just value head'
 assert all(p.grad is None for p in a._network.parameters())
 assert all(torch.equal(v,a._network.state_dict()[k]) for k,v in before.items())
 # Backward is reusable after diagnostics; value head still trains.
 a._optimizer.zero_grad();routed.backward();a._optimizer.step()
 assert all(torch.equal(v,a._network.state_dict()[k]) for k,v in before.items() if k.startswith('encoder.'))
 assert any(not torch.equal(v,a._network.state_dict()[k]) for k,v in before.items() if 'critic' in k)

def test_total_gradient_difference_is_only_raw_value_encoder_contribution():
 a=aligned(value_encoder_gradient_contract=D,digital_twin_handoff_fusion_enabled=True,outcome_memory_fusion_enabled=True)
 info=state();o=a._forward_policy(a._extract_semantic_state(info),run_metadata=info['run_metadata'])
 raw=.5*(o['value']-12.)**2;other=o['slow_logits'].square().sum()+o['fast_logits'].square().sum()+.2*o['event_logits'].square().sum()
 gt=gradients(a,other+raw);gv=gradients(a,raw);gd=gradients(a,other+a._route_value_encoder_gradients(raw))
 for k in gt:
  expected=gt[k]-gv[k] if k.startswith('encoder.') else gt[k]
  assert torch.allclose(gd[k],expected,atol=2e-5,rtol=2e-5),k

def test_disabled_route_is_identity_and_inference_roundtrip(tmp_path):
 from src.evaluators.real_eval_support import build_inference_agent
 a=aligned();x=torch.tensor(2.,requires_grad=True);assert a._route_value_encoder_gradients(x) is x
 b=aligned(value_encoder_gradient_contract=D);info=state();torch.random.manual_seed(44);one,_=a.act(None,copy.deepcopy(info));torch.random.manual_seed(44);two,_=b.act(None,copy.deepcopy(info));assert one==two
 path=tmp_path/'routing.pt';b.save(str(path));c=build_inference_agent('sa_ghmappo',random_seed=7,checkpoint_path=path)
 assert c._value_encoder_gradient_contract==D
 assert torch.equal(x,b._route_value_encoder_gradients(x))
 with pytest.raises(ValueError,match='Unknown value encoder'):aligned(value_encoder_gradient_contract='typo')

def test_td_budget_is_exact_and_create_only(tmp_path):
 from preflight_sagh_value_encoder_routing import InteractionBudget,execute,CONTRACTS
 assert CONTRACTS=={'T':'shared_v1','D':D}
 b=InteractionBudget(io.StringIO())
 for cell in ['T/seed7','T/seed13','D/seed7','D/seed13']:
  b.cell=cell
  for phase,n in [('train',768),('validation',528)]:
   b.phase=phase
   for _ in range(n):b.charge()
 assert b.total==5184
 with pytest.raises(RuntimeError,match='hard cap'):b.charge()
 assert b.total==5184
 with pytest.raises(FileExistsError):execute(tmp_path,tmp_path)
