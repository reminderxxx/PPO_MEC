"""Independent stream ledger, object lifetime, provenance and deterministic replay audit."""
import argparse,json,sys,hashlib,statistics
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import _hash_file
from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.core.joint_stream_rules import StreamCacheRule
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor


def canonical(event):return (json.dumps(event,sort_keys=True,separators=(',',':'))+'\n').encode()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();home=a.root.resolve();torch.set_num_threads(1)
    m=json.loads((home/'manifest.json').read_text());assert m['status']=='complete'
    if not m['smoke']:assert not m['git_dirty']
    for file,sha in m['output_sha256'].items():assert _hash_file(home/file)==sha
    for file,sha in m['source_fingerprints'].items():assert _hash_file(ROOT/file)==sha, file
    assert _hash_file(home/'protocol.pt')==m['protocol_sha256']
    rows=[json.loads(x) for x in (home/'results.jsonl').open()];assert len(rows)==m['planned_cells'];results={(x['case_id'],x['rule']):x for x in rows};assert len(results)==len(rows)
    cases={c['case_id']:c for c in m['cases']};cells={};event_costs=defaultdict(Counter);event_nodes=Counter();event_finishes={};event_hashes={};count_events=0
    last=None;frame_events=[];counts={};objects={};lives={};completed=set()
    def flush(es):
        if not es:return
        key=(es[0]['case_id'],es[0]['rule']);frame=es[0]['frame_index'];case=cases[key[0]];catalog={o['object_id']:o for o in case['catalog']};requests={r['request_id']:r for r in case['arrivals']}
        required=defaultdict(set)
        for e in es:
            if e['output_source_rsu_id'] is None:
                n=next(n for n in requests[e['request_id']]['workflow']['nodes'] if n['node_id']==e['node_id']);required[e['control']['execution_target']].update((f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"))
        for e in es:
            c=e['control'];rid=c['cache_target'];q=e['cache_request'];request=requests[e['request_id']];assert c['request_id']==e['request_id'] and request['frame']<=frame and e['request_id'] not in completed
            optional=c['cache_node_id'] is not None and (c['cache_request_id']!=e['request_id'] or c['cache_node_id']!=e['node_id'] or e['output_source_rsu_id'] is not None)
            assert not e['explicit_evicted_object_ids']
            if c['cache_node_id'] is not None:
                cache_req=requests[c['cache_request_id']];assert cache_req['vehicle_id']==e['vehicle_id'] and cache_req['frame']<=frame and cache_req['request_id'] not in completed
                span=e['control_planning']['cache_admission_span'];slots=e['control_planning']['link_calendar'][rid];assert span<=len(slots) and all(x is None or x==[e['vehicle_id'],'model'] for x in slots[:span])
            for obj in q.get('evicted_object_ids',[]):
                if optional:assert obj not in required[rid], 'optional cache evicted current team demand'
                assert obj in objects[rid];objects[rid].remove(obj)
                old=lives.pop((rid,obj),None)
                if old is not None and not old['used']:counts['unused_evicted']+=1
            for obj in q.get('admitted_object_ids',[]):
                assert obj not in objects[rid];objects[rid].add(obj)
                if optional:counts['prefetched_objects']+=1;lives[rid,obj]={'used':False}
            for site,keys in objects.items():
                assert sum(catalog[k]['resident_size_bytes'] for k in keys)<=case['capacity']
                assert all(set(catalog[k]['dependency_ids'])<=keys for k in keys)
            if e['status'] in ('rsu_compute','output_ready'):
                n=next(n for n in request['workflow']['nodes'] if n['node_id']==e['node_id'])
                for obj in [f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"]:
                    life=lives.get((e['target_rsu_id'],obj))
                    if life is not None and not life['used']:life['used']=True;counts['used_prefetched_objects']+=1
            if e['status']=='node_completed':event_nodes[key,frame]+=1
            if e['workflow_completed']:event_finishes[key,e['request_id']]=frame;completed.add(e['request_id'])
            for k in ['input_transfer_link_frames','output_transfer_link_frames','state_transfer_link_frames']:event_costs[key,frame][k]+=e[k]
    for line in (home/'events.jsonl').open():
        e=json.loads(line);key=(e['case_id'],e['rule']);identity=(*key,e['frame_index'])
        if identity!=last:
            flush(frame_events);frame_events=[]
            if last is None or key!=last[:2]:
                if last is not None:
                    counts['unused_at_end']+=sum(not x['used'] for x in lives.values());cells[last[:2]]=dict(counts)
                counts=Counter();objects=defaultdict(set);lives={};completed=set();event_hashes[key]=hashlib.sha256()
            last=identity
        frame_events.append(e);event_hashes[key].update(canonical(e));count_events+=1
    flush(frame_events)
    if last:
        counts['unused_at_end']+=sum(not x['used'] for x in lives.values());cells[last[:2]]=dict(counts)
    assert cells.keys()==results.keys()
    sums=defaultdict(float);receipts=Counter();model_frames=Counter()
    for line in (home/'receipts.jsonl').open():
        q=json.loads(line);key=(q['case_id'],q['rule']);frame=q['frame_index'];row=results[key];reward=sum(q['reward_components'].values());assert frame==receipts[key];receipts[key]+=1;sums[key]+=reward;model_frames[key]+=q['model_transfer_link_frames']
        assert all(0<=v<=1 for v in q['link_usage_by_rsu'].values())
        assert abs(sum(q['request_rewards'].values())+q['unassigned_team_reward']-reward)<1e-9
        unfinished=sum(r['arrival_frame']<=frame and (not r['completed'] or r['completion_frame_index']>frame) for r in row['per_request'])
        assert abs(q['reward_components']['unfinished_wait']+.01*unfinished)<1e-9
        assert q['reward_components']['completed_nodes']==event_nodes[key,frame]
        for field in ['input_transfer_link_frames','output_transfer_link_frames','state_transfer_link_frames']:assert q[field]==event_costs[key,frame][field]
    for key,row in results.items():
        assert abs(sums[key]-row['reward'])<1e-8 and receipts[key]==row['steps']==row['frame_receipt_count']
        assert row['completed_nodes']==sum(event_nodes[key,f] for f in range(row['steps']))
        assert row['model_transfer_link_frames']==model_frames[key]
        assert row['completed_requests']==sum(r['completed'] for r in row['per_request'])
        for r in row['per_request']:assert r['completion_frame_index']==event_finishes.get((key,r['request_id']))
    # Read-only policy reproduction of first/last scenario, all three rules.
    data=torch.load(home/'protocol.pt',weights_only=False);pm=json.loads(Path(m['source_protocol']).read_text());bundle=torch.load(Path(m['source_protocol']).parent/'predictor.pt',weights_only=False);replays=[]
    for c in [data['cases'][i] for i in sorted({0,len(data['cases'])-1})]:
        s=data['sessions'][c['session']]
        for rule_name in m['rules']:
            env=MultiRequestStreamEnv(frames=s['frames'],arrivals=c['arrivals'],catalog=c['catalog'],rsu_states=s['rsus'],costs=ReplayCosts(**pm['costs']['replay_costs']),prediction_provider=LearnedMobilityPredictor(bundle),**c['options']);rule=StreamCacheRule(rule_name);h=hashlib.sha256();reward=0.
            while not env._control_episode_done():
                controls=rule.act(env.decision_states(),env.frame_index);_,v,_,info=env.step_controls(controls);reward+=v
                for e in info['events']:h.update(canonical({**dict(case_id=c['case_id'],rule=rule_name),**e,'control_planning':rule.last_diagnostics[e['vehicle_id']]}))
            key=(c['case_id'],rule_name);assert h.hexdigest()==event_hashes[key].hexdigest() and abs(reward-results[key]['reward'])<1e-8;replays.append(dict(case_id=c['case_id'],rule=rule_name,events_equal=True))
    paired={}
    for rule in ['naive_prefetch','gain_prefetch']:
        ds=[results[k,rule]['reward']-results[k,'reactive']['reward'] for k in cases]
        paired[rule]=dict(reward_mean_delta=statistics.mean(ds),wins=sum(x>1e-9 for x in ds),losses=sum(x< -1e-9 for x in ds),ties=sum(abs(x)<=1e-9 for x in ds),completed_delta=statistics.mean(results[k,rule]['completed_requests']-results[k,'reactive']['completed_requests'] for k in cases))
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),client_review_date='2026-10-16',literature_cutoff='2026-10-15',target_venue='IEEE TMC',policy_version='tmc_review_policy_v3_20260621',artifact_run_id=str(home.relative_to(ROOT)),source_commit=m['source_commit'],evidence_level='train-only engineering rules raw/replayed; paper E0_UNAVAILABLE',counts=dict(cells=len(rows),events=count_events,receipts=sum(receipts.values()),reproductions=len(replays)),paired=paired,lifetimes=[dict(case_id=k[0],rule=k[1],**v) for k,v in cells.items()],reproductions=replays,necessary_bound_pass_cases=sum(c['bounds']['all_delivery_necessary_bound_pass'] for c in cases.values()),total_scenarios=len(cases),training_added=0,paper_claim_ready=False,scope='matched scenario descriptive deltas; no learned algorithm ranking; all positive/negative cases retained',input_sha256={n:_hash_file(home/n) for n in ['manifest.json','results.jsonl','events.jsonl','receipts.jsonl','protocol.pt']})
    (home/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('lifetimes','reproductions','input_sha256')},indent=2))


if __name__=='__main__':main()
