"""Independent raw ledgers, checkpoint replays, matched deltas and fixed-final curves."""
import argparse, hashlib, json, statistics, sys
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import _hash_file
from scripts.audit_joint_stream_workloads import canonical
from scripts.train_joint_stream_control import make_agent,make_env,stream_episode,actor_hash
from scripts.train_joint_tail_control import tensor_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();home=a.root.resolve();torch.set_num_threads(1)
    plan=json.loads((home/'protocol/manifest.json').read_text());assert plan['status']=='frozen'
    for path,sha in plan['assets'].items():assert _hash_file(home/'protocol'/path)==sha
    for path,sha in plan['source_fingerprints'].items():assert _hash_file(ROOT/path)==sha,path
    parent_path=Path(plan['source_protocol']);assert _hash_file(parent_path)==plan['source_protocol_sha256']
    parent=json.loads(parent_path.read_text());original_path=Path(parent['source_manifest']);assert _hash_file(original_path)==parent['source_manifest_sha256']
    from scripts.launch_eight_vehicle_joint_experiment import arguments
    original=json.loads(original_path.read_text());args=arguments(original['command'][1:])
    for key,path in [('ngsim',args.mobility_csv_path),('alibaba',args.workflow_csv_path)]:assert _hash_file(path)==plan['raw_data_sha256'][key]
    data=torch.load(home/'protocol/protocol.pt',weights_only=False);cases=data['cases']['dev'];runs=[];train=[];dev=[];checks=[]
    for name in plan['methods']:
        output=home/name;meta=json.loads((output/'manifest.json').read_text());assert meta['status']=='complete'
        assert meta['protocol_sha256']==_hash_file(home/'protocol/manifest.json')
        assert meta['source_commit']==plan['source_commit'] and (plan['smoke'] or not meta['git_dirty'])
        for f,sha in meta['output_sha256'].items():assert _hash_file(output/f)==sha
        training=[json.loads(x) for x in (output/'training.jsonl').open()];evaluations=[json.loads(x) for x in (output/'dev.jsonl').open()]
        assert len(training)==len(plan['seeds'])*plan['episodes']
        assert {(x['seed'],x['episode'],x['case_id']) for x in training}=={(seed,i+1,c) for seed in plan['seeds'] for i,c in enumerate(plan['training_schedule'][str(seed)])}
        for r in training:
            r['agent']=name;assert r['steps']<=len(data['sessions'][0]['frames'])
            if r['losses']:assert all(torch.isfinite(torch.tensor(v)) for v in r['losses'].values())
        assert len(evaluations)==len(plan['seeds'])*(len(cases)+(len(plan['checkpoint_episodes'])-1)*len(plan['curve_probe_ids']))
        expected={(seed,ep,c['case_id']) for seed in plan['seeds'] for ep in plan['checkpoint_episodes'] for c in cases if ep==plan['episodes'] or c['case_id'] in plan['curve_probe_ids']}
        assert {(r['seed'],r['episode'],r['case_id']) for r in evaluations}==expected
        assert len(meta['checkpoints'])==len(plan['seeds'])*len(plan['checkpoint_episodes'])
        for c in meta['checkpoints']:
            path=output/c['path'];assert _hash_file(path)==c['sha256'];assert c['initial_tensor_sha256']!=c['trained_tensor_sha256'];assert c['initial_actor_sha256']!=c['trained_actor_sha256']
            q=type(make_agent(name,c['seed'],plan['episodes'])).from_checkpoint(path);assert tensor_hash(q)==c['trained_tensor_sha256'];assert actor_hash(q)==c['trained_actor_sha256']
            checks.append(dict(agent=name,seed=c['seed'],episode=c['episode'],sha256=c['sha256']))
        train+=training;dev+=evaluations;runs.append(output)
    session_keys=[]
    old=torch.load(parent_path.parent/'protocol.pt',weights_only=False)
    allowed={(f['source_segment_id'],f['time_index']) for w in old['scenarios']['train'] for f in w['frames']}
    for session in data['sessions']:
        keys={(f['source_segment_id'],f['time_index']) for f in session['frames']}
        assert keys<=allowed and not any(keys&other for other in session_keys)
        session_keys.append(keys)
    bundle=data['predictor'];train_offsets={s['frame_offset'] for s in data['sessions'] if s['split']=='train'}
    assert set(bundle['fit_offsets'])|set(bundle['validation_offsets'])==train_offsets
    assert not set(bundle['fit_offsets'])&set(bundle['validation_offsets'])
    assert bundle['seed']==101 and bundle['epochs']==30 and bundle['horizon']==8
    final=[x for x in dev if x['episode']==plan['episodes']]
    (home/'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in final))
    m=dict(planned_cells=len(final),cases=[dict(case_id=c['case_id'],capacity=c['capacity'],catalog=[o.__dict__ for o in c['catalog']],arrivals=[dict(frame=r.frame,vehicle_id=r.vehicle_id,request_id=r.request_id,workflow=r.workflow.to_dict()) for r in c['arrivals']]) for c in cases])
    def lines(name):
        for output in runs:
            with (output/name).open() as f:yield from f
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
    for line in lines('events.jsonl'):
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
    for line in lines('receipts.jsonl'):
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
        for vid in {r['vehicle_id'] for r in cases[key[0]]['arrivals']}:
            expected=[r['request_id'] for r in sorted(cases[key[0]]['arrivals'],key=lambda r:r['frame']) if r['vehicle_id']==vid]
            started=sorted([r for r in row['per_request'] if r['vehicle_id']==vid and r['start_frame'] is not None],key=lambda r:r['start_frame'])
            assert [r['request_id'] for r in started]==expected[:len(started)]
            for i,r in enumerate(started):
                assert r['start_frame']>=r['arrival_frame']
                if i:assert started[i-1]['completion_frame_index'] is not None and started[i-1]['completion_frame_index']<r['start_frame']
    cases=data['cases']['dev']
    replays=[]
    for name in plan['methods']:
        meta=json.loads((home/name/'manifest.json').read_text())
        for seed in plan['seeds']:
            checkpoint=next(c for c in meta['checkpoints'] if c['seed']==seed and c['episode']==plan['episodes'])
            learner=type(make_agent(name,seed,plan['episodes'])).from_checkpoint(home/name/checkpoint['path'])
            c=cases[0];identity=dict(agent=name,rule=f'{name}_seed{seed}',seed=seed,episode=plan['episodes'],case_id=c['case_id'])
            class Sink:
                def __init__(self):self.hash=hashlib.sha256()
                def write(self,line):self.hash.update(canonical(json.loads(line)))
            sink=Sink();row,_=stream_episode(make_env(c,data,plan),learner,deterministic=True,event_file=sink,identity=identity)
            key=(c['case_id'],identity['rule'])
            assert sink.hash.hexdigest()==event_hashes[key].hexdigest() and abs(row['reward']-results[key]['reward'])<1e-8
            replays.append(dict(agent=name,seed=seed,case_id=c['case_id'],events_equal=True))
    by={(r['agent'],r['seed'],r['case_id']):r for r in final};paired={}
    for other in [n for n in plan['methods'] if n!='sa_ghmappo']:
        deltas=[by['sa_ghmappo',s,c['case_id']]['reward']-by[other,s,c['case_id']]['reward'] for s in plan['seeds'] for c in cases]
        completed=[by['sa_ghmappo',s,c['case_id']]['completed_requests']-by[other,s,c['case_id']]['completed_requests'] for s in plan['seeds'] for c in cases]
        paired[other]=dict(reward_mean_delta=statistics.mean(deltas),requests_mean_delta=statistics.mean(completed),wins=sum(d>1e-9 for d in deltas),losses=sum(d< -1e-9 for d in deltas),ties=sum(abs(d)<=1e-9 for d in deltas))
    means={name:{k:statistics.mean(r[k] for r in final if r['agent']==name) for k in ['reward','completed_requests','completed_nodes','model_transfer_link_frames','decision_rows']} for name in plan['methods']}
    curves=[]
    for name in plan['methods']:
        for seed in plan['seeds']:
            for ep in plan['checkpoint_episodes']:
                selected=[r for r in dev if r['agent']==name and r['seed']==seed and r['episode']==ep and r['case_id'] in plan['curve_probe_ids']]
                curves.append(dict(agent=name,seed=seed,episode=ep,cells=len(selected),reward=statistics.mean(r['reward'] for r in selected),completed=statistics.mean(r['completed_requests'] for r in selected)))
    report=dict(status='pass',reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-15',target_venue='IEEE TMC',policy_version='tmc_review_policy_v3_20260621',artifact_run_id=str(home.relative_to(ROOT)),source_commit=plan['source_commit'],
        evidence_level='internal stream development raw/checkpoint replay; paper E0_UNAVAILABLE / Unverifiable',
        counts=dict(final_cells=len(final),events=count_events,receipts=sum(receipts.values()),checkpoints=len(checks),reproductions=len(replays),training_sessions=len(train),training_frames=sum(r['steps'] for r in train),training_decisions=sum(r['decision_rows'] for r in train)),
        means=means,paired_full_minus_baseline=paired,curve_probe=curves,checkpoints=checks,replays=replays,
        lifetimes=[dict(case_id=k[0],rule=k[1],**v) for k,v in cells.items()],
        source_sha256=_hash_file(home/'protocol/manifest.json'),output_sha256={'results.jsonl':_hash_file(home/'results.jsonl')},
        limits=['same source workflow pool, long one source DAG','2 originaltrain sessions used as newdev, not independent paper clusters','synthetic costs; not real model execution','final64 fixed, probe curve only; no convergence guarantee'],paper_claim_ready=False)
    assert report['counts']['training_sessions']==plan['training_session_count']
    assert report['counts']['training_frames']<=plan['maximum_training_frames']
    (home/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines_out=['# 持续请求学习自动回收（开发诊断）','',f"源提交：`{plan['source_commit']}`；审计UTC：{report['reviewed_at']}。",'',
        '固定最终checkpoint，全部seed/条件保留；formal/holdout封存，论文E0_UNAVAILABLE / Unverifiable。','',
        '| 方法 | 平均奖励 | 平均完整请求 / 16 | 平均节点 | 模型链路帧 |','| --- | --- | --- | --- | --- |']
    for name,value in means.items():lines_out.append(f"| {name} | {value['reward']:.6f} | {value['completed_requests']:.6f} | {value['completed_nodes']:.6f} | {value['model_transfer_link_frames']:.6f} |")
    lines_out+=['','## 完整版减基线（配对）','',json.dumps(paired,ensure_ascii=False,indent=2),'','## 后期曲线（仅固定共同probe）','',json.dumps(curves,ensure_ascii=False,indent=2),'',
        '本文件为原始审计派生报告。下一实现前需按项目政策独立复核负例、机会、成本和信用，不因排名自动延长训练或改reward。']
    (home/'results_review.md').write_text('\n'.join(lines_out)+'\n')
    print(json.dumps(dict(status='pass',counts=report['counts'],means=means,paired=paired),indent=2))


if __name__=='__main__':main()
