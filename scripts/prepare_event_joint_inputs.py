"""Freeze outcome-blind, disjoint temporal blocks and unseen DAG confirmation."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluators.real_sample_support import load_real_source_frames,build_sample_rsus
from src.evaluators.main_results_support import build_selected_workflow_states
from scripts.run_persistent_workflow_learning import dump,sha


def associated(vehicle,rsus):
    candidates=[(math.hypot(vehicle.position_x-r.position_x,vehicle.position_y-r.position_y),r.rsu_id,r.coverage_radius) for r in rsus]
    valid=[(d,i) for d,i,r in candidates if d<=r]
    return min(valid)[1] if valid else None


def assert_disjoint(units,previous):
    all_rows=[(u.get('split','old'),u) for u in previous+units]
    for i,(s,a) in enumerate(all_rows):
        for t,b in all_rows[i+1:]:
            if a.get('index')==b.get('index') and a is b:continue
            a0,a1=a['frames'][0]['time_index'],a['frames'][-1]['time_index']
            b0,b1=b['frames'][0]['time_index'],b['frames'][-1]['time_index']
            if max(a0,b0)<=min(a1,b1):raise ValueError('Raw time intervals overlap')


def prepare(args):
    out=Path(args.output);out.mkdir();old=json.loads(Path(args.previous_inputs).read_text());units=[];eligibility=[]
    protocol=dict(rule='Sequential 96-frame blocks, stride192; exclude actual raw-time overlap with prior/selected windows;  first 36 eligible blocks, 16train/8dev/12confirmation',
        subject='Horizon-persistent vehicle with highest observed-input coverage, >=80%; lexicographic ID tie break; no performance ranking',
        workflow='First 16 eligible Alibaba DAGs with 5..20 nodes in source order; train/dev first8, confirmation next8; fixed rotating pairs',
        modeling='Unit service steps and SHA256 model assignment retained; modeled cache/load/RSU infrastructure; not measured AI inference',
        row_limit_fallback=[300000,600000,900000],confirmation='Outcome sealed until final checkpoint and statistical protocol freeze; independence only relative to this experiment, prior project exposure not exhaustively excluded',
        sealed_formal_holdout='Not used or opened',prior_inputs_sha256=sha(args.previous_inputs),argv=sys.argv,formal=False)
    dump(out/'preparation_protocol.json',protocol)
    # Frame ranks change when extending a CSV prefix; raw intervals are authoritative.
    start=0
    wfs=build_selected_workflow_states(args.workflows,16,'ordered',5,20,7)
    if len(wfs)!=16:raise ValueError('Need sixteen eligible DAGs')
    for limit in protocol['row_limit_fallback']:
        raw,source=load_real_source_frames(ROOT,'ngsim',args.mobility,'',limit)
        # If extending the prefix, reconstruct all selections before any performance use.
        units=[];eligibility=[]
        for offset in range(start,min(start+240*192,len(raw)-95),192):
            frames=raw[offset:offset+96]
            if any(frames[i+1]['time_index']-frames[i]['time_index']!=100 for i in range(95)):continue
            lo,hi=frames[0]['time_index'],frames[-1]['time_index']
            exclusions=old['units']+units
            if any(max(lo,u['frames'][0]['time_index'])<=min(hi,u['frames'][-1]['time_index']) for u in exclusions):
                eligibility.append(dict(offset=offset,eligible=False,reason='raw_time_overlap_with_previous_or_selected'))
                continue
            common=set.intersection(*[{v.vehicle_id for v in f['vehicles']} for f in frames])
            if not common:continue
            rsus,metadata=build_sample_rsus(frames,'auto_dominant_tight');subjects=[]
            for subject in sorted(common):
                sequence=[associated(next(v for v in f['vehicles'] if v.vehicle_id==subject),rsus) for f in frames]
                coverage=sum(x is not None for x in sequence)/96.
                direct=sum(a is not None and b is not None and a!=b for a,b in zip(sequence,sequence[1:]))
                subjects.append(dict(subject=subject,coverage=coverage,direct_handoffs=direct))
            best=min(subjects,key=lambda x:(-x['coverage'],x['subject']))
            eligible=best['coverage']>=.8
            eligibility.append(dict(offset=offset,persistent=len(common),best=best,eligible=eligible))
            if not eligible:continue
            index=len(units);split='train' if index<16 else 'dev' if index<24 else 'confirmation';pool=wfs[:8] if index<24 else wfs[8:]
            pair=[pool[(2*index)%8],pool[(2*index+1)%8]]
            serialized=[]
            for f in frames:
                x={k:v for k,v in f.items() if k!='vehicles'};x['vehicles']=[v.to_dict() for v in f['vehicles']];serialized.append(x)
            units.append(dict(index=index,split=split,offset=offset,subject_id=best['subject'],coverage=best['coverage'],direct_handoffs=best['direct_handoffs'],
                frames=serialized,rsus=[r.to_dict() for r in rsus],metadata=metadata,workflows=[w.to_dict() for w in pair]))
            if len(units)==36:break
        dump(out/'eligibility.json',dict(row_limit=limit,frames=len(raw),candidates=eligibility))
        if len(units)==36:break
    if len(units)!=36:raise ValueError('Insufficient input-only eligible windows within preparation bounds')
    assert_disjoint(units,old['units'])
    payload=dict(contract='event_joint_input_v1',units=units,workflows=[w.to_dict() for w in wfs[:8]],all_workflows=[w.to_dict() for w in wfs],
        input_protocol=protocol,source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),mobility_path=args.mobility,workflow_path=args.workflows,row_limit=limit,formal=False)
    # Separate outcome-sealed confirmation inputs from training consumers.
    dev=dict(payload,units=[u for u in units if u['split']!='confirmation']);confirmation=dict(payload,units=[u for u in units if u['split']=='confirmation'])
    dump(out/'train_dev_inputs.json',dev);dump(out/'confirmation_inputs.json',confirmation)
    intervals=[dict(index=u['index'],split=u['split'],start=u['frames'][0]['time_index'],end=u['frames'][-1]['time_index'],subject=u['subject_id'],coverage=u['coverage'],direct_handoffs=u['direct_handoffs'],workflows=[w['workflow_id'] for w in u['workflows']]) for u in units]
    dump(out/'split_manifest.json',dict(intervals=intervals,raw_time_overlap_pairs=0,min_gap_ms=min(b['start']-a['end'] for a,b in zip(intervals,intervals[1:])),
        train_dev_sha256=sha(out/'train_dev_inputs.json'),confirmation_sha256=sha(out/'confirmation_inputs.json'),raw_data=dict(mobility_sha256=sha(args.mobility),workflow_sha256=sha(args.workflows)),prior_project_exposure='not exhaustively audited; not strict project-wide hidden holdout'))
    print(json.dumps(dict(status='prepared',windows=len(units),row_limit=limit,dags=[w.workflow_id for w in wfs],raw_time_overlap_pairs=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous-inputs',required=True);p.add_argument('--mobility',required=True);p.add_argument('--workflows',required=True);p.add_argument('--output',required=True);prepare(p.parse_args())
