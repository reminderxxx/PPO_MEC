"""Observed team I/O protection and forecast node-delivery values (no trace oracle)."""
from math import ceil, hypot
from src.envs.core.joint_control_planning import remaining_model_frames


def node_delivery_times(state, target):
    """Serial remaining DAG delivery estimates; current paid input/output is respected."""
    rates=state['control_cost_parameters'];done=set(state['workflow']['completed_node_ids'])
    nodes={n['node_id']:n for n in state['workflow']['nodes']};current=state['current_workflow_node']['node_id'];times=[];elapsed=0.
    for nid in state['workflow']['execution_order']:
        if nid in done:continue
        n=nodes[nid];pending=state['output_pending'] if nid==current else {}
        if pending:elapsed+=ceil(pending['remaining_bytes']/rates['output_bytes_per_frame'])
        else:
            paid=nid==current and target in state['input_received_targets']
            if not paid:
                amount=state.get('input_remaining_bytes_by_rsu',{}).get(target,n['input_size']*rates['profile_unit_bytes']) if nid==current else n['input_size']*rates['profile_unit_bytes']
                elapsed+=ceil(max(1,amount)/rates['input_bytes_per_frame'])
            work=max(1.,n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit'])
            if nid==current:work=max(0.,state['current_node_work_units_required']-state['current_node_work_units_progress'])
            elapsed+=ceil(work/rates['rsu_work_units_per_frame'])+ceil(max(1,n['output_size']*rates['profile_unit_bytes'])/rates['output_bytes_per_frame'])
        times.append(elapsed)
    return times


def node_delivery_utility(times, delay, horizon, preserved_prefix=0):
    # Node rewards and unfinished-request waiting match the declared physical objective.
    adjusted=[t if i<preserved_prefix else t+delay for i,t in enumerate(times)]
    deliveries=sum(t<=horizon for t in adjusted)
    finish=adjusted[-1] if adjusted else 0.
    return deliveries-.01*min(horizon,max(0.,finish))


def fresh_model_span(state, row, node=None, kind='adapter'):
    """Admission starts a nonpreemptive dependency closure, not one free slot."""
    n=node or state['current_workflow_node'];rates=state['control_cost_parameters'];loading=row.get('loading_remaining_frames',{})
    span=0
    for k,key,ready,size in [('base',n['required_base_model'],row.get('ready_base_model_ids',[]),rates['base_size_bytes']),
                            ('adapter',n['required_adapter'],row.get('cached_adapter_ids',[]),rates['adapter_size_bytes'])]:
        if kind=='base_model' and k=='adapter':continue
        if key not in ready and f'{k}:{key}' not in loading:span+=ceil(size/rates['model_bytes_per_frame'])
    return span


def reactive_link_calendar(decisions, targets, horizon=8):
    """Book actual selected stations, including NEW cold admission and following I/O."""
    first=next(iter(decisions.values()))['semantic_state'];rates=first['control_cost_parameters']
    horizon=max(horizon,ceil(rates['base_size_bytes']/rates['model_bytes_per_frame'])+ceil(rates['adapter_size_bytes']/rates['model_bytes_per_frame']))
    sites=first['rsus'];calendar={r['rsu_id']:[None]*horizon for r in sites};backlogs={}
    for rid in calendar:
        backlog=max((sum(r.get('loading_remaining_frames',{}).values()) for d in decisions.values()
                     for r in d['semantic_state']['rsus'] if r['rsu_id']==rid),default=0)
        backlogs[rid]=backlog
        for t in range(min(horizon,backlog)):calendar[rid][t]=('background','model')
    # Node service rather than full-workflow size: keep short deliverable work moving.
    ordered=sorted(decisions,key=lambda v:(node_delivery_times(decisions[v]['semantic_state'],targets.get(v))[0],v))
    for vid in ordered:
        s=decisions[vid]['semantic_state'];rid=targets.get(vid)
        if rid not in calendar:continue
        rates=s['control_cost_parameters'];row=next(r for r in s['rsus'] if r['rsu_id']==rid);pending=s['output_pending'];phases=[];cursor=0
        copy=s['state_copies_by_rsu'].get(rid,{})
        if s['execution_state_version'] and rid!=s['execution_location'] and copy.get('version')!=s['execution_state_version']:
            transfer=s['state_transfers_by_rsu'].get(rid,{})
            amount=transfer.get('remaining_bytes',max(1,s['execution_state_payload_bytes']-copy.get('payload_bytes',0)))
            phases.append(('state',max(rates['minimum_state_frames'],ceil(amount/rates['state_bytes_per_frame']))))
        if not pending:
            loading=row.get('loading_remaining_frames',{});n=s['current_workflow_node']
            own_loading=sum(loading.get(f'{kind}:{key}',0) for kind,key in [('base',n['required_base_model']),('adapter',n['required_adapter'])])
            fresh=fresh_model_span(s,row)
            if fresh:phases.append(('model',fresh))
            cursor=max(cursor,min(horizon,own_loading))
            if rid not in s['input_received_targets']:
                amount=s.get('input_remaining_bytes_by_rsu',{}).get(rid,n['input_size']*rates['profile_unit_bytes'])
                phases.append(('input',ceil(max(1,amount)/rates['input_bytes_per_frame'])))
            phases.append(('compute',ceil(max(0,s['current_node_work_units_required']-s['current_node_work_units_progress'])/rates['rsu_work_units_per_frame'])))
            phases.append(('output',ceil(max(1,n['output_size']*rates['profile_unit_bytes'])/rates['output_bytes_per_frame'])))
        else:phases.append(('output',ceil(pending['remaining_bytes']/rates['output_bytes_per_frame'])))
        for kind,count in phases:
            if kind=='compute':cursor+=count;continue
            if kind=='model':
                while cursor<horizon and not all(x is None for x in calendar[rid][cursor:min(horizon,cursor+count)]):cursor+=1
                if cursor+count>horizon:cursor=horizon;continue
            for _ in range(count):
                while cursor<horizon and calendar[rid][cursor] is not None:cursor+=1
                if cursor>=horizon:break
                calendar[rid][cursor]=(vid,kind);cursor+=1
    return calendar


def delivery_migration_opportunities(state, calendar, *, decisions=None, targets=None, prediction_enabled=True):
    """Value cache-only and state-only plans separately against team node delivery.

    Only unclaimed slots can prepare. Existing/new reactive transactions cannot
    be displaced. Future arrivals, compute ordering and forecast errors remain estimates.
    """
    if not prediction_enabled or not state.get('execution_location') or not state['execution_state_version']:return {}
    vid=state['primary_vehicle_id'];p=state['predictions'];points=p.get('predicted_positions_by_vehicle',{}).get(vid,[])
    confidence=min(1.,p.get('prediction_confidence_by_vehicle',{}).get(vid,0.))
    if not points or confidence<.5:return {}
    rows={r['rsu_id']:r for r in state['rsus']};source=rows.get(state['execution_location'])
    if source is None:return {}
    def covers(x,r,margin=0):return hypot(x[0]-r['position_x'],x[1]-r['position_y'])<=max(0.,r['coverage_radius']+margin)
    vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==vid)
    if not covers([vehicle['position_x'],vehicle['position_y']],source):return {}
    margin=p.get('position_uncertainty_margin',0.);exits=[i+1 for i,x in enumerate(points) if not covers(x,source,margin)]
    if not exits:return {}
    eta=exits[0];horizon=state['episode_frames_remaining'];rates=state['control_cost_parameters']
    if eta>=horizon:return {}
    def current_queue(rid):
        return sum(ceil(max(0,d['semantic_state']['current_node_work_units_required']-d['semantic_state']['current_node_work_units_progress'])/rates['rsu_work_units_per_frame'])
                   for owner,d in (decisions or {}).items() if owner!=vid and (targets or {}).get(owner)==rid)
    source_initial=remaining_model_frames(state,source)+current_queue(source['rsu_id'])
    source_times=[source_initial+t for t in node_delivery_times(state,source['rsu_id'])]
    if not source_times or source_times[-1]<=eta:return {}
    prefix_count=sum(t<=eta for t in source_times)
    done=set(state['workflow']['completed_node_ids']);lookup={n['node_id']:n for n in state['workflow']['nodes']}
    remaining_nodes=[lookup[nid] for nid in state['workflow']['execution_order'] if nid not in done]
    planning_node=remaining_nodes[prefix_count]
    growth=eta
    for i,n in enumerate(remaining_nodes):
        output=ceil(max(1,n['output_size']*rates['profile_unit_bytes'])/rates['output_bytes_per_frame'])
        if source_times[i]-output<=eta and not (i==0 and state['output_pending']):growth+=n['output_size']*rates['profile_unit_bytes']
    result={}
    for rid,row in rows.items():
        if rid==source['rsu_id'] or not covers(points[eta-1],row,-margin):continue
        slots=[t for t in range(min(eta,len(calendar[rid]))) if calendar[rid][t] is None]
        if 0 not in slots:continue
        copy=state['state_copies_by_rsu'].get(rid,{});transfer=state['state_transfers_by_rsu'].get(rid)
        missing=max(1,state['execution_state_payload_bytes']-copy.get('payload_bytes',0));rate=rates['state_bytes_per_frame']
        full=max(rates['minimum_state_frames'],ceil(missing/rate)) if transfer is None else max(transfer['remaining_frames'],ceil(transfer['remaining_bytes']/rate))
        if copy.get('version')==state['execution_state_version']:full=0
        residual=max(rates['minimum_state_frames'],ceil(growth/rate))
        if transfer is not None:residual=max(residual,ceil(max(1,state['execution_state_payload_bytes']-transfer['payload_bytes']+growth)/rate))
        reactive=max(rates['minimum_state_frames'],ceil((missing+growth)/rate)) if transfer is None else full+residual
        alternate=dict(state,current_workflow_node=planning_node,
                       output_pending=state['output_pending'] if prefix_count==0 else {})
        model=remaining_model_frames(alternate,row)
        # Do not promise prefetch benefit for an adapter already used before departure.
        # Future DAG admissions remain available to the learned cache head; their
        # forecast current-node proxy stays zero until that node is actually current.
        mb=min(model,len(slots)) if prefix_count==0 else 0
        prefix=min(full,len(slots))
        before=model+reactive;cache_after=before-mb
        state_after=model+max(0,full-prefix)+residual if prefix else before
        queue=current_queue(rid)
        dest_times=node_delivery_times(state,rid)
        source_prefix=source_times[prefix_count-1] if prefix_count else source_initial
        dest_prefix=dest_times[prefix_count-1] if prefix_count else 0.
        progress=max(0.,eta-source_prefix)
        times=[source_times[i] if i<prefix_count else eta+max(0.,t-dest_prefix-progress)+queue for i,t in enumerate(dest_times)]
        base=node_delivery_utility(times,before,horizon,prefix_count)
        def benefit(after,budget):
            extra=max(0.,budget+after-before)
            return confidence*(node_delivery_utility(times,after,horizon,prefix_count)-base)-.01*extra-(1-confidence)*.01*budget
        cache_net=benefit(cache_after,mb) if mb else 0.
        state_net=benefit(state_after,prefix) if prefix else 0.
        if max(cache_net,state_net)<=0:continue
        result[rid]=dict(eta_frames=eta,prepare_frames=full,prepare_budget_frames=prefix,reactive_frames=before,
            residual_recovery_frames=residual,arrival_delay_frames=state_after,saved_delay_frames=max(0,before-state_after),
            net_reward_proxy=max(cache_net,state_net),state_net_reward_proxy=state_net,cache_net_reward_proxy=cache_net,
            combined_net_reward_proxy=max(cache_net,state_net),confidence=confidence,prepare_feasible=prefix>0 and state_net>0,
            model_prefetch_frames=mb,reserved_free_slots=slots,blocked_reactive_slots=min(eta,len(calendar[rid]))-len(slots),
            team_opportunity_cost=0.,team_node_gain_state=sum((t if i<prefix_count else t+state_after)<=horizon for i,t in enumerate(times))-sum((t if i<prefix_count else t+before)<=horizon for i,t in enumerate(times)),
            team_node_gain_cache=sum((t if i<prefix_count else t+cache_after)<=horizon for i,t in enumerate(times))-sum((t if i<prefix_count else t+before)<=horizon for i,t in enumerate(times)),
            source_delivery_times=source_times,predeparture_delivered_nodes=prefix_count,planning_node_id=planning_node['node_id'],
            destination_compute_queue_frames=queue,proxy_scope='observed protected reactive demands; unknown future demand not an oracle')
    return result


def book_delivery_control(calendar, vid, claims, state, control):
    """Keep other and own mandatory future phases when adding an idle preparation."""
    from src.envs.core.joint_link_reservation import execution_link_need
    for rid,kind in claims:
        count=1;need=execution_link_need(state,control.execution_target)
        if need and need[:2]==(rid,kind):count=need[2]
        if kind=='state' and control.prepare_target==rid:
            copied=state['state_copies_by_rsu'].get(rid,{}).get('payload_bytes',0)
            count=max(1,ceil((state['execution_state_payload_bytes']-copied)/state['control_cost_parameters']['state_bytes_per_frame']))
        if kind=='model':
            node=next(n for n in state['workflow']['nodes'] if n['node_id']==control.cache_node_id)
            row=next(r for r in state['rsus'] if r['rsu_id']==rid)
            count=max(1,remaining_model_frames(dict(state,current_workflow_node=node,output_pending={}),row))
        count=max(0,count-sum(item==(vid,kind) for item in calendar[rid]))
        for t,item in enumerate(calendar[rid]):
            if item is None and count>0:calendar[rid][t]=(vid,kind);count-=1


def preserves_current_model_demands(state, row, node, kind, decisions, targets, catalog):
    """Simulate declared dependency-safe LRU; optional prefetch may not evict current users."""
    vid=state['primary_vehicle_id'];rid=row['rsu_id']
    if targets.get(vid)==rid and node['node_id']==state['current_workflow_node']['node_id'] and not state['output_pending']:return True
    if 'occupied_bytes' not in row or 'capacity_bytes' not in row:return False
    closure={f"base:{node['required_base_model']}"}
    if kind!='base_model':closure.add(f"adapter:{node['required_adapter']}")
    resident=list(row.get('resident_object_ids',[]));pending=set(row.get('loading_object_ids',[]));rates=state['control_cost_parameters']
    def size(key):return rates['base_size_bytes'] if key.startswith('base:') else rates['adapter_size_bytes']
    missing=closure-set(resident)-pending;need=sum(size(k) for k in missing);used=row['occupied_bytes'];protected=set()
    if used+need<=row['capacity_bytes']:return True
    for other,d in decisions.items():
        s=d['semantic_state']
        if targets.get(other)==rid and not s['output_pending']:
            n=s['current_workflow_node'];protected.update((f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"))
    while used+need>row['capacity_bytes']:
        # Unknown dependency topology is conservatively protected.
        dependencies={catalog[k] for k in [*resident,*pending] if k in catalog}
        if any(k.startswith('adapter:') and k not in catalog for k in [*resident,*pending]):dependencies.update(k for k in resident if k.startswith('base:'))
        victim=next((k for k in resident if k not in closure and k not in dependencies),None)
        if victim is None or victim in protected:return False
        resident.remove(victim);used-=size(victim)
    return True
