"""Causal, declared-cost planning features, not future-trace oracles."""
from math import ceil, hypot


def workflow_service_frames(state):
    rates = state['control_cost_parameters']; done = set(state['workflow']['completed_node_ids'])
    total = 0.
    for node in state['workflow']['nodes']:
        if node['node_id'] in done:
            continue
        work = max(1.,node['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit'])
        if node['node_id'] == state['current_workflow_node']['node_id']:
            work = max(0.,work-state['current_node_work_units_progress'])
        total += work/rates['rsu_work_units_per_frame']
        total += ceil(max(1,node['input_size']*rates['profile_unit_bytes'])/rates['input_bytes_per_frame'])
        total += ceil(max(1,node['output_size']*rates['profile_unit_bytes'])/rates['output_bytes_per_frame'])
    return total


def migration_opportunities(state, *, prediction_enabled=True, allow_partial=False):
    """Estimate saved critical-path delay minus extra transfer/contention cost.

    Only a predicted loss of *execution-site coverage* is an opportunity.
    Nearest-RSU changes inside overlapping coverage do not qualify. Estimates
    use declared synthetic rates and observed prefix versions, never true futures.
    """
    if allow_partial:
        return partial_migration_opportunities(state,prediction_enabled=prediction_enabled)
    if not prediction_enabled or not state.get('execution_location'):
        return {}
    vid = state['primary_vehicle_id']; p = state['predictions']
    points = p.get('predicted_positions_by_vehicle',{}).get(vid,[])
    confidence = p.get('prediction_confidence_by_vehicle',{}).get(vid,0.)
    if not points or confidence < .5 or state['execution_state_version'] == 0:
        return {}
    rsus = {r['rsu_id']:r for r in state['rsus']}; source = rsus.get(state['execution_location'])
    if source is None:
        return {}
    margin = p.get('position_uncertainty_margin',0.)
    def covered(point, row, extra=0.):
        return hypot(point[0]-row['position_x'],point[1]-row['position_y']) <= max(0.,row['coverage_radius']+extra)
    current_vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==vid)
    if not covered([current_vehicle['position_x'],current_vehicle['position_y']],source):
        return {}
    exits = [i+1 for i,point in enumerate(points) if not covered(point,source,margin)]
    if not exits:
        return {}
    eta = exits[0]; rates = state['control_cost_parameters']; result = {}
    if eta >= state.get('episode_frames_remaining',float('inf')):
        return {}
    if workflow_service_frames(state)+source.get('service_work_units',0) <= eta:
        return {}
    payload = state['execution_state_payload_bytes']; version = state['execution_state_version']
    for rid,r in rsus.items():
        if rid == source['rsu_id'] or not covered(points[eta-1],r,-margin):
            continue
        copy = state['state_copies_by_rsu'].get(rid,{})
        if copy.get('version') == version:
            continue
        missing = max(1,payload-copy.get('payload_bytes',0))
        reactive = max(rates['minimum_state_frames'],ceil(missing/rates['state_bytes_per_frame']))
        transfer = state['state_transfers_by_rsu'].get(rid)
        prepare = max(reactive,0) if transfer is None else max(transfer['remaining_frames'],ceil(transfer['remaining_bytes']/rates['state_bytes_per_frame']))
        # Source may update while copying; a final versioned delta still costs at least one transfer.
        residual = max(rates['minimum_state_frames'],ceil(eta/rates['state_bytes_per_frame']))
        if transfer is not None:
            residual=max(rates['minimum_state_frames'],ceil(max(1,payload-transfer['payload_bytes']+eta)/rates['state_bytes_per_frame']))
            reactive=prepare+residual
        saved = max(0.,reactive-residual)
        remaining_service = max(1.,state['current_node_service_steps_remaining'])
        contention = len(r.get('active_transfer_object_ids',[])) + r.get('service_request_count',0)
        node=state['current_workflow_node'];model_frames=0
        if node['required_base_model'] not in r.get('ready_base_model_ids',[]):
            model_frames+=ceil(rates.get('base_size_bytes',2048)/rates['model_bytes_per_frame'])
        if node['required_adapter'] not in r.get('cached_adapter_ids',[]):
            model_frames+=ceil(rates.get('adapter_size_bytes',512)/rates['model_bytes_per_frame'])
        extra_frames = max(0.,prepare+residual-reactive)
        net = confidence*(.01*saved + saved/(remaining_service+reactive+1)) - .01*(extra_frames+contention*prepare+model_frames)
        if prepare+model_frames <= eta and net > 0:
            result[rid] = dict(eta_frames=eta,prepare_frames=prepare,reactive_frames=reactive,
                               saved_delay_frames=saved,net_reward_proxy=net,confidence=confidence,
                               prepare_feasible=(model_frames==0),model_prefetch_frames=model_frames)
    return result


def request_reward_ledger(events, components):
    """Budget-balanced local service ledger plus equal shared model/control fees.

    Allocation is accounting credit, not a causal marginal-contribution claim.
    Shared model objects can benefit several requests and retain team credit.
    """
    common = (components['model_transfer']+components['shared_state'])/len(events)
    ledger = {}
    for e in events:
        complete = e['status'] in ('node_completed','vehicle_fallback_completed')
        value = float(complete)-.01*float(not e['workflow_completed'])
        value -= .02*float(e['status'].startswith('vehicle_'))
        value -= .01*(e['state_transfer_link_frames']+e['input_transfer_link_frames']+e['output_transfer_link_frames'])
        ledger[e['vehicle_id']] = value+common
    if abs(sum(ledger.values())-sum(components.values())) > 1e-9:
        raise RuntimeError('request credit ledger does not conserve team reward')
    return ledger


def partial_migration_opportunities(state, *, prediction_enabled=True):
    """Value useful prefixes and model-only prefetch, with residual recovery.

    A positive proxy is a deployment candidate, not a guarantee of realized gain.
    Byte growth includes a conservative next-output allowance. No true future is read.
    """
    if not prediction_enabled or not state.get('execution_location') or state['execution_state_version']==0:
        return {}
    vid=state['primary_vehicle_id'];p=state['predictions'];points=p.get('predicted_positions_by_vehicle',{}).get(vid,[])
    confidence=p.get('prediction_confidence_by_vehicle',{}).get(vid,0.)
    if not points or confidence<.5:return {}
    sites={r['rsu_id']:r for r in state['rsus']};source=sites.get(state['execution_location'])
    if source is None:return {}
    def covers(point,row,margin=0.):return hypot(point[0]-row['position_x'],point[1]-row['position_y'])<=max(0.,row['coverage_radius']+margin)
    vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==vid)
    if not covers([vehicle['position_x'],vehicle['position_y']],source):return {}
    margin=p.get('position_uncertainty_margin',0.)
    exits=[i+1 for i,point in enumerate(points) if not covers(point,source,margin)]
    if not exits:return {}
    eta=exits[0];remaining=state.get('episode_frames_remaining',120)
    if eta>=remaining or workflow_service_frames(state)+source.get('service_work_units',0)<=eta:return {}
    rates=state['control_cost_parameters'];rate=rates['state_bytes_per_frame'];node=state['current_workflow_node']
    pending=state['output_pending'];required=state['current_node_service_steps_remaining']
    growth=eta
    # At most the current output (or next output after a pending delivery) is added in the short horizon.
    nodes={n['node_id']:n for n in state['workflow']['nodes']};order=state['workflow']['execution_order'];pos=order.index(node['node_id'])
    if pending and pos+1<len(order):growth+=nodes[order[pos+1]]['output_size']*rates['profile_unit_bytes']
    elif required<=eta:growth+=node['output_size']*rates['profile_unit_bytes']
    payload=state['execution_state_payload_bytes'];result={}
    for rid,row in sites.items():
        if rid==source['rsu_id'] or not covers(points[eta-1],row,-margin):continue
        copy=state['state_copies_by_rsu'].get(rid,{})
        missing=max(1,payload-copy.get('payload_bytes',0))
        transfer=state['state_transfers_by_rsu'].get(rid)
        full=max(rates['minimum_state_frames'],ceil(missing/rate)) if transfer is None else max(transfer['remaining_frames'],ceil(transfer['remaining_bytes']/rate))
        residual=max(rates['minimum_state_frames'],ceil(growth/rate))
        if transfer is not None:residual=max(residual,ceil(max(1,payload-transfer['payload_bytes']+growth)/rate))
        reactive_state=max(rates['minimum_state_frames'],ceil((missing+growth)/rate)) if transfer is None else full+residual
        model=0
        if not pending:
            if node['required_base_model'] not in row.get('ready_base_model_ids',[]):model+=ceil(rates['base_size_bytes']/rates['model_bytes_per_frame'])
            if node['required_adapter'] not in row.get('cached_adapter_ids',[]):model+=ceil(rates['adapter_size_bytes']/rates['model_bytes_per_frame'])
        if copy.get('version')==state['execution_state_version']:full=0
        model_budget=min(model,eta);prefix_budget=min(full,max(0,eta-model_budget))
        after_state=max(0,full-prefix_budget)+residual if prefix_budget else reactive_state
        after=model-model_budget+after_state;before=model+reactive_state
        saved=max(0.,before-after)
        output_frames=ceil(max(1,pending.get('remaining_bytes',node['output_size']*rates['profile_unit_bytes']))/rates['output_bytes_per_frame'])
        service=output_frames if pending else required+output_frames+ceil(max(1,node['input_size']*rates['profile_unit_bytes'])/rates['input_bytes_per_frame'])
        queue=row.get('service_work_units',0)/rates['rsu_work_units_per_frame']
        finish=eta+after+service+queue
        def benefit(delay_saved,arrival_delay):
            if eta+arrival_delay+service+queue>remaining:return 0.
            completion=delay_saved/(service+arrival_delay+queue+1)
            wait=.01*delay_saved if eta+arrival_delay+workflow_service_frames(state)+queue<=remaining else 0.
            return confidence*(completion+wait)
        contention=row.get('service_request_count',0)+len(row.get('active_transfer_object_ids',[]))
        extra=max(0.,model_budget+prefix_budget+after-before)
        combined=benefit(saved,after)-.01*(extra+contention*(model_budget+prefix_budget))
        cache_after=max(0,model-model_budget)+reactive_state
        cache_net=benefit(float(model_budget),cache_after)-.01*contention*model_budget if model_budget else 0.
        state_saved=max(0.,cache_after-after)
        state_extra=max(0.,prefix_budget+after_state-reactive_state)
        state_net=benefit(state_saved,after)-.01*(state_extra+contention*prefix_budget) if prefix_budget>0 else 0.
        net=max(cache_net,state_net)
        if net>0:
            result[rid]=dict(eta_frames=eta,prepare_frames=full,prepare_budget_frames=prefix_budget,
                reactive_frames=before,saved_delay_frames=saved,net_reward_proxy=net,
                state_net_reward_proxy=state_net,cache_net_reward_proxy=cache_net,confidence=confidence,
                prepare_feasible=prefix_budget>0 and state_net>0,model_prefetch_frames=model_budget,
                partial_copy=full>prefix_budget,residual_recovery_frames=residual,arrival_delay_frames=after,combined_net_reward_proxy=combined)
    return result
