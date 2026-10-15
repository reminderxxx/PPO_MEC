from copy import deepcopy
import pytest
from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv, WorkflowArrival, StreamingRequestControl
from src.envs.core.multi_request_target_control_env import JointRequestControl
from src.envs.core.joint_model_cache import ModelCacheObject
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.specs import VehicleState, RSUState, WorkflowGraphState, WorkflowNode


def wf(name, nodes=1, adapter='x'):
    ns=[WorkflowNode(f'{name}_n{i}', f'n{i}', 'base', adapter, 1, 1,
                     predecessors=[f'{name}_n{i-1}'] if i else []) for i in range(nodes)]
    return WorkflowGraphState(name, ns, [(ns[i].node_id,ns[i+1].node_id) for i in range(nodes-1)],
                              [n.node_id for n in ns], current_node_id=ns[0].node_id)


def make(arrivals=None, horizon=50, capacity=8):
    arrivals=arrivals or [WorkflowArrival(0,'v',wf('one')), WorkflowArrival(15,'v',wf('two'))]
    return MultiRequestStreamEnv(frames=[{'time_index':f*100,'source_segment_id':'s',
        'vehicles':[VehicleState('v',0,0,0,'base')]} for f in range(horizon)], arrivals=arrivals,
        catalog=[ModelCacheObject('base:base','base_model',4,4,base_model_id='base'),
                 ModelCacheObject('adapter:x','adapter',2,2,('base:base',),adapter_id='x')],
        rsu_states=[RSUState('a',0,0,10)],costs=ReplayCosts(),workload_bytes_per_unit=1,
        shared_state_enabled=False,model_cache_config={'capacity_bytes':capacity,'transfer_bytes_per_frame':4})


def controls(env):
    return {v:StreamingRequestControl('a','a',d['semantic_state']['current_workflow_node']['node_id'],
            request_id=d['semantic_state']['request_id']) for v,d in env.decision_states().items()}


def test_fifo_lifetimes_cache_reuse_and_no_state_or_input_reuse():
    e=make([WorkflowArrival(0,'v',wf('one')),WorkflowArrival(0,'v',wf('two'))]);seen_second=False
    while not e._control_episode_done():
        _,_,_,info=e.step_controls(controls(e))
        if 'v/two' in e._started and not seen_second:
            seen_second=True
            assert e.state_version['v']==0 and e.work_progress['v']==0 and not e.state_copies and not e.input_received
            assert e.model_cache.is_ready('a','x','base')
    s=e.summary();assert s['completed_requests']==2 and s['completed_nodes']==2
    assert {x['request_id'] for x in s['per_request']}=={'v/one','v/two'}
    assert s['per_request'][1]['start_frame']>s['per_request'][0]['completion_frame_index']
    assert e.model_cache.transferred_bytes['a']==6
    assert [x['status'] for x in e.events if x['request_id']=='v/two'][0]=='input_ready'
    assert seen_second


def test_idle_has_receipts_and_does_not_end_before_later_release_or_leak_future():
    e=make([WorkflowArrival(8,'v',wf('secret'))])
    assert e.decision_states()=={} and 'secret' not in str(e.observe())
    for f in range(8):
        _,r,done,info=e.step_controls({});assert not done and info['events']==[]
        assert info['frame_cost_receipt']['frame_index']==f
        assert abs(r-sum(info['frame_cost_receipt']['reward_components'].values()))<1e-10
    assert e.decision_states()['v']['semantic_state']['request_id']=='v/secret'
    assert len(e.frame_receipts)==8


def test_unreleased_graph_hidden_and_released_queue_visible_wait_penalty():
    e=make([WorkflowArrival(0,'v',wf('one',5)),WorkflowArrival(2,'v',wf('two'))])
    assert 'two' not in str(e.decision_states())
    for f in range(3):
        _,_,_,info=e.step_controls({'v':StreamingRequestControl(request_id='v/one')})
        assert info['reward_components']['unfinished_wait']==(-.01 if f<2 else -.02)
    assert e.decision_states()['v']['semantic_state']['queued_workflows'][0]['request_id']=='v/two'


def test_old_and_stale_actions_rejected_without_mutation():
    e=make();before=deepcopy(e.model_cache.snapshot('a'))
    with pytest.raises(ValueError,match='legacy'):e.step_controls({'v':JointRequestControl('a')})
    with pytest.raises(ValueError,match='stale'):e.step_controls({'v':StreamingRequestControl('a',request_id='v/wrong')})
    assert e.frame_index==0 and e.model_cache.snapshot('a')==before


def test_reset_cold_and_terminal_unfinished_and_credit_conserved():
    e=make([WorkflowArrival(0,'v',wf('one',10)),WorkflowArrival(1,'v',wf('two'))],horizon=5)
    while not e._control_episode_done():
        _,r,_,info=e.step_controls(controls(e));q=info['frame_cost_receipt']
        assert abs(sum(q['request_rewards'].values())+q['unassigned_team_reward']-r)<1e-9
    assert e.summary()['request_count']==2 and e.summary()['completed_requests']==0
    assert e.summary()['queued_requests']==1 and e.frame_index==5
    e.reset();assert e.frame_index==0 and not e.model_cache.resident['a'] and not e._finished


def test_background_transfer_during_idle_is_charged():
    e=make([WorkflowArrival(10,'v',wf('one'))]);e.model_cache.begin_frame()
    e.model_cache.request_adapter('a','x','base',available_link_slots=1);e.model_cache.advance()
    _,r,done,info=e.step_controls({});q=info['frame_cost_receipt']
    assert q['model_transfer_link_frames']==1 and r<0 and not done
    assert q['unassigned_team_reward']==r


def test_invalid_arrival_and_catalog_closure_fail():
    with pytest.raises(ValueError,match='inside'):make([WorkflowArrival(51,'v',wf('x'))])
    with pytest.raises(ValueError,match='unique'):make([WorkflowArrival(0,'v',wf('x')),WorkflowArrival(2,'v',wf('x'))])
    with pytest.raises(ValueError,match='cannot fit'):make(capacity=5)


def test_queued_prefetch_bound_to_released_request_only():
    e=make([WorkflowArrival(0,'v',wf('one',5)),WorkflowArrival(2,'v',wf('two'))])
    command=StreamingRequestControl(cache_target='a',cache_node_id='two_n0',request_id='v/one',cache_request_id='v/two')
    with pytest.raises(ValueError,match='released'):e.step_controls({'v':command})
    for _ in range(2):e.step_controls({'v':StreamingRequestControl(request_id='v/one')})
    _,_,_,info=e.step_controls({'v':command})
    assert 'adapter:x' in info['events'][0]['cache_request']['admitted_object_ids']
    assert e.workflows['v'].current_node_id=='one_n0'


def test_prefetch_can_save_real_delivery_wait_without_free_transfer():
    from src.envs.core.joint_stream_rules import StreamCacheRule
    from src.envs.core.joint_causal_observation import CausalMobilityPredictor
    class ObservedStationaryForecast(CausalMobilityPredictor):
        def predict(self, vehicles, associations, rsus, mapper, step_index):
            out=super().predict(vehicles,associations,rsus,mapper,step_index)
            out['predicted_positions_by_vehicle']={v.vehicle_id:[[v.position_x,v.position_y]]*8 for v in vehicles}
            return out
    def run(mode):
        w=wf('one',2);w.nodes[0].input_size=8;w.nodes[1].required_adapter='y'
        e=MultiRequestStreamEnv(frames=[{'time_index':f*100,'source_segment_id':'s','vehicles':[VehicleState('v',0,0,0,'base')]} for f in range(60)],
            arrivals=[WorkflowArrival(0,'v',w)],catalog=[ModelCacheObject('base:base','base_model',4,4,base_model_id='base'),
                ModelCacheObject('adapter:x','adapter',2,2,('base:base',),adapter_id='x'),ModelCacheObject('adapter:y','adapter',2,2,('base:base',),adapter_id='y')],
            rsu_states=[RSUState('a',0,0,10)],costs=ReplayCosts(),workload_bytes_per_unit=1,
            prediction_provider=ObservedStationaryForecast(),shared_state_enabled=False,link_reservation_enabled=True,model_cache_config={'capacity_bytes':8,'transfer_bytes_per_frame':4})
        rule=StreamCacheRule(mode);reward=0
        while not e._control_episode_done():
            _,r,_,_=e.step_controls(rule.act(e.decision_states(),e.frame_index));reward+=r
        return e,reward
    reactive,ra=run('reactive');prefetch,rb=run('gain_prefetch')
    assert reactive.summary()['completed_nodes']==prefetch.summary()['completed_nodes']==2
    assert prefetch.frame_index<reactive.frame_index and rb>ra
    assert prefetch.model_cache.transferred_bytes==reactive.model_cache.transferred_bytes


def test_profile_mapping_preserves_original_dag_and_request_ids_are_distinct():
    from src.envs.core.joint_stream_workloads import request_plan,catalog_from_pool,capacity_lower_bounds
    pool=[wf('source',5)];before=deepcopy(pool[0].to_dict())
    arrivals=request_plan(pool,['v0','v1'],'short','stream',families=2)
    assert pool[0].to_dict()==before and len(arrivals)==4
    assert sorted(a.frame for a in arrivals)==[0,0,240,240]
    assert len({n.node_id for a in arrivals for n in a.workflow.nodes})==20
    assert all(len(a.workflow.edges)==4 and len(a.workflow.nodes)==5 for a in arrivals)
    catalog=catalog_from_pool(pool,2);assert {o.resident_size_bytes for o in catalog if o.object_type=='base_model'}=={2048,3072}
    rates=dict(workflow_profile_unit_bytes=1,workload_bytes_per_unit=1,rsu_work_units_per_frame=1)
    bound=capacity_lower_bounds(arrivals,20,1,rates)
    assert not bound['all_delivery_necessary_bound_pass'] and bound['fifo_serial_bound_violations']


def test_same_frame_fifo_preserves_published_order_not_request_name():
    arrivals=[WorkflowArrival(0,'v',wf('z_first',10)),WorkflowArrival(0,'v',wf('a_second'))]
    e=make(arrivals)
    assert e.decision_states()['v']['semantic_state']['request_id']=='v/z_first'
    assert e.decision_states()['v']['semantic_state']['queued_workflows'][0]['request_id']=='v/a_second'
    from src.envs.core.joint_stream_workloads import capacity_lower_bounds
    rates=dict(workflow_profile_unit_bytes=1,workload_bytes_per_unit=1,rsu_work_units_per_frame=1)
    b=capacity_lower_bounds(arrivals,5,1,rates)
    assert b['fifo_serial_bound_violations']==['v/z_first','v/a_second']


def test_batch_and_stream_share_request_order_when_names_sort_backwards():
    for later in [0,10]:
        e=make([WorkflowArrival(0,'v',wf('z_first')),WorkflowArrival(later,'v',wf('a_second'))])
        order=[]
        while not e._control_episode_done():
            _,_,_,info=e.step_controls(controls(e))
            for x in info['events']:
                if not order or order[-1]!=x['request_id']:order.append(x['request_id'])
        assert order==['v/z_first','v/a_second']


def test_live_admission_rejects_optional_eviction_of_self_or_peer_current_model():
    def run(peer):
        ids=['v0','v1'] if peer else ['v0']
        arrivals=[WorkflowArrival(0,v,wf(v,2)) for v in ids]
        for a in arrivals:a.workflow.nodes[1].required_adapter='y'
        e=MultiRequestStreamEnv(frames=[{'time_index':f*100,'source_segment_id':'s','vehicles':[VehicleState(v,0,0,0,'base') for v in ids]} for f in range(50)],
            arrivals=arrivals,catalog=[ModelCacheObject('base:base','base_model',4,4,base_model_id='base'),ModelCacheObject('adapter:x','adapter',2,2,('base:base',),adapter_id='x'),ModelCacheObject('adapter:y','adapter',2,2,('base:base',),adapter_id='y')],
            rsu_states=[RSUState('a',0,0,10)],costs=ReplayCosts(),workload_bytes_per_unit=1,shared_state_enabled=False,
            model_cache_config={'capacity_bytes':6,'transfer_bytes_per_frame':8})
        e.model_cache.begin_frame();e.model_cache.request_adapter('a','x','base');e.model_cache.advance()
        e.model_cache.begin_frame();e.model_cache.advance();e._sync_cache_view()
        actor='v1' if peer else 'v0'
        cs={v:StreamingRequestControl('a' if v=='v0' else None,request_id=f'{v}/{v}') for v in ids}
        cs[actor]=StreamingRequestControl('a' if not peer else None,'a',f'{actor}_n1',request_id=f'{actor}/{actor}',cache_request_id=f'{actor}/{actor}')
        before=e.model_cache.transferred_bytes['a'];_,_,_,info=e.step_controls(cs);event=next(x for x in info['events'] if x['vehicle_id']==actor)
        assert event['cache_request']['status']=='blocked_current_model_dependency'
        assert event['cache_request']['link_slots_consumed']==0 and e.model_cache.transferred_bytes['a']==before
        assert e.model_cache.is_ready('a','x','base') and not e.model_cache.is_ready('a','y','base')
        with pytest.raises(ValueError,match='current declared'):
            e.step_controls({v:StreamingRequestControl('a',cache_target='a' if v==actor else None,evict_object_ids=('adapter:x',) if v==actor else (),request_id=f'{v}/{v}') for v in ids})
    run(False);run(True)
