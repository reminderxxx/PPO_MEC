"""Causal public-intent link calendars and declared-cost preparation values.

Future slots are conservative plans, not wireless guarantees. Current grants are
validated by the environment; ongoing model transfers cannot be preempted.
"""
from math import ceil, log1p
from src.envs.core.joint_control_planning import migration_opportunities, workflow_service_frames, remaining_model_frames


def service_scale(value, horizon=120):
    return log1p(max(0.,float(value)))/log1p(max(1.,horizon))


def execution_link_need(state, target):
    rows={r['rsu_id']:r for r in state['rsus']};pending=state['output_pending'];rates=state['control_cost_parameters']
    if pending:
        destination=target if target in rows else pending['rsu_id']
    else:destination=target
    if destination not in rows:return None
    copy=state['state_copies_by_rsu'].get(destination,{})
    if state['execution_state_version']>0 and destination!=state['execution_location'] and copy.get('version')!=state['execution_state_version']:
        transfer=state['state_transfers_by_rsu'].get(destination,{})
        amount=transfer.get('remaining_bytes',max(1,state['execution_state_payload_bytes']-copy.get('payload_bytes',0)))
        return destination,'state',max(rates['minimum_state_frames'],ceil(amount/rates['state_bytes_per_frame']))
    if pending:return destination,'output',ceil(pending['remaining_bytes']/rates['output_bytes_per_frame'])
    row=rows[destination];model=remaining_model_frames(state,row)
    if model:return destination,'model',model
    if destination not in state['input_received_targets']:
        amount=state.get('input_remaining_bytes_by_rsu',{}).get(destination,state['current_workflow_node']['input_size']*rates['profile_unit_bytes'])
        return destination,'input',max(1,ceil(amount/rates['input_bytes_per_frame']))
    return None


def public_link_calendar(decisions, targets, horizon=3):
    """Schedule only declared current I/O/restores, model backlog and imminent output."""
    sites=sorted(next(iter(decisions.values()))['semantic_state']['rsus'],key=lambda r:r['rsu_id'])
    calendar={r['rsu_id']:[None]*horizon for r in sites}
    for rid in calendar:
        # Same delayed snapshots for all controllers; maximum reconciles local fresh reports.
        frames=max((sum(r.get('loading_remaining_frames',{}).values())
                    for d in decisions.values() for r in d['semantic_state']['rsus'] if r['rsu_id']==rid),default=0)
        for t in range(min(horizon,frames)):calendar[rid][t]=('background','model')
    ordered=sorted(decisions,key=lambda vid:(decisions[vid]['semantic_state']['episode_frames_remaining']-workflow_service_frames(decisions[vid]['semantic_state']),vid))
    for vid in ordered:
        s=decisions[vid]['semantic_state'];target=targets.get(vid,s['execution_location'] or s['associations'][vid])
        need=execution_link_need(s,target)
        if need:
            rid,kind,count=need
            if kind=='model':continue  # Existing model backlog already booked; fresh admission is conditional.
            remaining=count
            for t in range(horizon):
                if calendar[rid][t] is None and remaining>0:calendar[rid][t]=(vid,kind);remaining-=1
        elif target in calendar and s['current_node_service_steps_remaining']<horizon:
            release=max(1,ceil(s['current_node_service_steps_remaining']))
            count=max(1,ceil(s['current_workflow_node']['output_size']*s['control_cost_parameters']['profile_unit_bytes']/s['control_cost_parameters']['output_bytes_per_frame']))
            for t in range(release,horizon):
                if calendar[target][t] is None and count>0:calendar[target][t]=(vid,'output');count-=1
    return calendar


def reserved_migration_opportunities(state, calendar, *, prediction_enabled=True):
    candidates=migration_opportunities(state,prediction_enabled=prediction_enabled,allow_partial=True,model_progress_aware=True)
    result={};rates=state['control_cost_parameters'];remaining=state['episode_frames_remaining'];sites={r['rsu_id']:r for r in state['rsus']}
    for rid,old in candidates.items():
        row=dict(old);eta=row['eta_frames'];slots=[t for t in range(min(eta,len(calendar[rid]))) if calendar[rid][t] is None]
        model=remaining_model_frames(state,sites[rid]);model_budget=min(model,len(slots));prefix=min(row['prepare_frames'],max(0,len(slots)-model_budget))
        residual=row['residual_recovery_frames'];before=row['reactive_frames']
        after=max(0,row['prepare_frames']-prefix)+residual if prefix else max(0,before-model)
        after+=max(0,model-model_budget)
        saved=max(0.,before-after);extra=max(0.,model_budget+prefix+after-before)
        # Only waiting and explicitly finishable DAG-node deadlines have reward value.
        # Do not give arbitrary fractional completion credit for a copy that changes no finish time.
        work=workflow_service_frames(state)+sites[rid].get('service_work_units',0)/rates['rsu_work_units_per_frame'];finish_before=eta+before+work;finish_after=eta+after+work
        crossing=float(finish_before>remaining>=finish_after)
        net=row['confidence']*((.01*saved if finish_after<=remaining else 0.)+crossing)-.01*extra
        cache_saved=float(model_budget);cache_net=row['confidence']*.01*cache_saved
        state_saved=max(0.,saved-cache_saved)
        state_net=row['confidence']*((.01*state_saved if finish_after<=remaining else 0.)+crossing)-.01*extra
        row.update(prepare_budget_frames=prefix,model_prefetch_frames=model_budget,saved_delay_frames=saved,
                   arrival_delay_frames=after,net_reward_proxy=max(net,cache_net),cache_net_reward_proxy=cache_net,
                   state_net_reward_proxy=state_net,prepare_feasible=prefix>0 and state_net>0 and 0 in slots,
                   reserved_free_slots=slots,combined_net_reward_proxy=net)
        if max(net,cache_net)>0:result[rid]=row
    return result


def control_link_claims(state, control, calendar):
    """Actual selected action claims; never replaces a sampled control."""
    vid=state['primary_vehicle_id'];claims=[];need=execution_link_need(state,control.execution_target)
    if need and need[1]!='model':claims.append((need[0],need[1]))
    if control.prepare_target is not None:claims.append((control.prepare_target,'state'))
    if control.cache_target is not None:claims.append((control.cache_target,'model'))
    unique={}
    for rid,kind in claims:
        if calendar[rid][0] is None or calendar[rid][0][0]==vid:unique.setdefault(rid,kind)
    return tuple(unique.items())


def book_control(calendar, vid, claims, state=None, control=None):
    for rid,slots in calendar.items():
        for t,item in enumerate(slots):
            if item is not None and item[0]==vid:slots[t]=None
    for rid,kind in claims:
        count=1
        if state is not None:
            need=execution_link_need(state,control.execution_target)
            if need and need[:2]==(rid,kind):count=need[2]
            if kind=='state' and control.prepare_target==rid:
                copied=state['state_copies_by_rsu'].get(rid,{}).get('payload_bytes',0)
                count=max(1,ceil((state['execution_state_payload_bytes']-copied)/state['control_cost_parameters']['state_bytes_per_frame']))
            if kind=='model':
                node=next(n for n in state['workflow']['nodes'] if n['node_id']==control.cache_node_id)
                alternate=dict(state,current_workflow_node=node,output_pending={})
                row=next(r for r in state['rsus'] if r['rsu_id']==rid)
                count=max(1,remaining_model_frames(alternate,row))
        for t,item in enumerate(calendar[rid]):
            if item is None and count>0:calendar[rid][t]=(vid,kind);count-=1
