"""Diagnostic cache rules, not learned SA/MAPPO or a trace-future oracle."""
from dataclasses import replace
from copy import deepcopy
from math import ceil, hypot
from src.envs.core.multi_request_stream_env import StreamingRequestControl
from src.envs.core.joint_delivery_planning import (reactive_link_calendar, fresh_model_span,
    book_delivery_control, node_delivery_times, node_delivery_utility)
from src.envs.core.joint_link_reservation import control_link_claims


class StreamCacheRule:
    def __init__(self, mode):
        if mode not in ('reactive','naive_prefetch','gain_prefetch'):raise ValueError('unknown rule')
        self.mode=mode;self.memory={};self.last_diagnostics={}

    def act(self, decisions, frame):
        if not decisions:self.last_diagnostics={};return {}
        targets={};load={}
        for v,d in sorted(decisions.items()):
            s=d['semantic_state'];rid=s['request_id'];old=self.memory.get(rid);sites=s['reachable_execution_rsu_ids']
            target=old if old in sites else min(sites,key=lambda r:(load.get(r,0),r)) if sites else None
            targets[v]=target;self.memory[rid]=target;load[target]=load.get(target,0)+1
        calendar=reactive_link_calendar(decisions,targets,8);controls={};diag={}
        catalog=next(iter(decisions.values()))['semantic_state']['control_cost_parameters']['model_catalog']
        deps={key:value['dependency_ids'][0] for key,value in catalog.items() if value['object_type']=='adapter'}
        ids=sorted(decisions);ids=ids[frame%len(ids):]+ids[:frame%len(ids)]
        for v in ids:
            s=decisions[v]['semantic_state'];target=targets[v];control=StreamingRequestControl(target,request_id=s['request_id'])
            details=[];admission_span=0;before=deepcopy(calendar)
            if target:
                row=next(r for r in s['rsus'] if r['rsu_id']==target);node=s['current_workflow_node'];cache=None
                missing=node['required_adapter'] not in row.get('cached_adapter_ids',[]) and node['required_adapter'] not in row.get('loading_adapter_ids',[])
                if missing and not s['output_pending'] and calendar[target][0] in (None,(v,'model')):
                    cache=(s['request_id'],node);control=replace(control,priority=3)
                elif self.mode!='reactive':
                    nodes={n['node_id']:n for n in s['workflow']['nodes']};remaining=[nodes[n] for n in s['workflow']['execution_order'] if n not in s['workflow']['completed_node_ids']]
                    times=node_delivery_times(s,target);candidates=[(s['request_id'],n,times[i-1],i,times) for i,n in enumerate(remaining) if i>0]
                    for q in s['queued_workflows']:
                        n=next(n for n in q['workflow']['nodes'] if n['node_id']==q['workflow']['execution_order'][0])
                        candidates.append((q['request_id'],n,times[-1],len(times),times+[times[-1]+1]))
                    for rid,n,use_at,index,timeline in candidates:
                        if n['required_adapter'] in row.get('cached_adapter_ids',[]) or n['required_adapter'] in row.get('loading_adapter_ids',[]):continue
                        # Exact family sizes from the static public catalog, not future route truth.
                        params=dict(s['control_cost_parameters']);params.update(base_size_bytes=catalog[f"base:{n['required_base_model']}"]['transfer_size_bytes'],adapter_size_bytes=catalog[f"adapter:{n['required_adapter']}"]['transfer_size_bytes'])
                        cs=dict(s,control_cost_parameters=params);span=fresh_model_span(cs,row,n)
                        if not span or span>len(calendar[target]) or any(x is not None for x in calendar[target][:span]):continue
                        if not protects_current_models(cs,row,n,decisions,targets,catalog):continue
                        p=s['predictions'];points=p.get('predicted_positions_by_vehicle',{}).get(v,[])
                        weight=float(p.get('prediction_confidence_by_vehicle',{}).get(v,.5))
                        if not points:weight=.5
                        for point in points[:max(1,ceil(use_at))]:
                            if hypot(point[0]-row['position_x'],point[1]-row['position_y'])>row['coverage_radius']:weight=0.;break
                        if use_at>len(points):weight*=.5**ceil((use_at-len(points))/max(1,len(points)))
                        working={f"adapter:{d['semantic_state']['current_workflow_node']['required_adapter']}" for d in decisions.values() if targets[d['semantic_state']['primary_vehicle_id']]==target}
                        working.add(f"adapter:{n['required_adapter']}");closure=working|{deps[k] for k in working};bytes_needed=sum(catalog[k]['resident_size_bytes'] for k in closure)
                        weight/=max(1.,bytes_needed/max(1,row.get('capacity_bytes',1)))
                        utility=node_delivery_utility(timeline,0,s['episode_frames_remaining'])-node_delivery_utility(timeline,span,s['episode_frames_remaining'],index)
                        net=weight*utility-.01*span*(1-weight)
                        details.append(dict(cache_request_id=rid,node_id=n['node_id'],use_after_frames=use_at,span=span,survival_weight=weight,net_delivery_proxy=net))
                        if self.mode=='naive_prefetch' or (net>1e-9 and use_at<=len(calendar[target])):
                            cache=(rid,n);break
                if cache:
                    rid,n=cache;params=dict(s['control_cost_parameters']);params.update(base_size_bytes=catalog[f"base:{n['required_base_model']}"]['transfer_size_bytes'],adapter_size_bytes=catalog[f"adapter:{n['required_adapter']}"]['transfer_size_bytes']);cs=dict(s,control_cost_parameters=params)
                    span=fresh_model_span(cs,row,n)
                    if span<=len(calendar[target]) and all(x is None or x==(v,'model') for x in calendar[target][:span]):
                        control=replace(control,cache_target=target,cache_node_id=n['node_id'],cache_request_id=rid);admission_span=span
                plan_state=s
                if control.cache_node_id is not None:
                    extra=[n for q in s['queued_workflows'] for n in q['workflow']['nodes']]
                    plan_state=dict(cs,workflow={**s['workflow'],'nodes':s['workflow']['nodes']+extra})
                claims=control_link_claims(plan_state,control,calendar);control=replace(control,link_reservations=claims)
                book_delivery_control(calendar,v,claims,plan_state,control)
            controls[v]=control;diag[v]=dict(candidates=details,link_calendar=before,cache_admission_span=admission_span)
        self.last_diagnostics=diag;return controls


def protects_current_models(state,row,node,decisions,targets,catalog):
    if 'occupied_bytes' not in row:return False
    closure={f"base:{node['required_base_model']}",f"adapter:{node['required_adapter']}"}
    resident=list(row.get('resident_object_ids',[]));pending=set(row.get('loading_object_ids',[]))
    if any(k not in catalog for k in [*resident,*pending]):return False
    needed=sum(catalog[k]['resident_size_bytes'] for k in closure-set(resident)-pending);used=row['occupied_bytes'];protected=set()
    for v,d in decisions.items():
        s=d['semantic_state'];n=s['current_workflow_node']
        if targets[v]==row['rsu_id'] and not s['output_pending']:protected.update((f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"))
    while used+needed>row['capacity_bytes']:
        dependencies={dep for k in [*resident,*pending] for dep in catalog[k]['dependency_ids']}
        victim=next((k for k in resident if k not in closure and k not in dependencies),None)
        if victim is None or victim in protected:return False
        resident.remove(victim);used-=catalog[victim]['resident_size_bytes']
    return True
