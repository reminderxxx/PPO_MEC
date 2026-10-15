"""Policy-independent necessary capacity bounds; feasibility is never asserted."""
from datetime import datetime,timezone
from math import ceil,hypot
from pathlib import Path
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    root=ROOT/'artifacts/joint_tail_control';p=root/'protocol/manifest.json'
    m=json.loads(p.read_text());data=torch.load(p.parent/'protocol.pt',weights_only=False)
    options=m['environment_options'];unit=options['workflow_profile_unit_bytes']
    assert options.get('vehicle_adapter_policy','declared_only')=='declared_only' and not options.get('vehicle_adapter_ids')
    assert m['costs']['replay_costs']['compute_slots_per_rsu']==m['costs']['replay_costs']['link_slots_per_rsu']==1
    work=options['workload_bytes_per_unit'];cpu=options['rsu_work_units_per_frame']
    input_rate=options.get('input_bytes_per_link_frame',1024);output_rate=options.get('output_bytes_per_link_frame',1024)
    demands=[]
    for wf in data['workflows']:
        compute=sum(ceil(max(1,n.input_size*unit/work)/cpu) for n in wf.nodes)
        io=sum(ceil(max(1,n.input_size*unit)/input_rate)+ceil(max(1,n.output_size*unit)/output_rate) for n in wf.nodes)
        demands.append(dict(workflow=wf.workflow_id,compute=compute,io=io,serial_minimum=compute+io))
    rows=[]
    for w in data['scenarios']['dev']:
        masks=[]
        for f in w['frames']:
            vehicles={v.vehicle_id:v for v in f['vehicles']};values=[]
            for vid in w['vehicle_ids']:
                v=vehicles[vid];mask=0
                for j,r in enumerate(w['rsus']):
                    if hypot(v.position_x-r.position_x,v.position_y-r.position_y)<=r.coverage_radius:mask|=1<<j
                values.append(mask)
            masks.append(values)
        violations=[]
        for subset in range(1,256):
            selected=[j for j in range(8) if subset>>j&1]
            cap=0
            for values in masks:
                union=0
                for j in selected:union|=values[j]
                # This deliberately loose upper bound ignores per-request topology,
                # phase coupling, models, state transfers and all DAG dependencies.
                cap+=min(len(selected),bin(union).count('1'))
            for kind in ('compute','io'):
                needed=sum(demands[j][kind] for j in selected)
                if needed>cap:violations.append(dict(kind=kind,requests=selected,minimum=needed,optimistic_capacity=cap,deficit=needed-cap))
        serial=[j for j,d in enumerate(demands) if d['serial_minimum']>len(w['frames'])]
        rows.append(dict(scenario_id=w['scenario_id'],panel=w['panel'],frame_offset=w['frame_offset'],
            full_completion_ruled_out=bool(violations or serial),serial_impossible_requests=serial,
            worst_violations=sorted(violations,key=lambda v:-v['deficit'])[:3],
            max_reachable_sites=max(bin(v).count('1') for values in masks for v in values)))
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),source_commit=m['git_commit'],
        artifact_run_id='joint_tail_control/protocol',literature_cutoff='2026-10-15',target_venue='IEEE TMC',policy_version='tmc_review_policy_v3_20260621',
        evidence='necessary optimistic capacity bounds only, no algorithm scores/selection/training',
        demands=demands,rows=rows,
        limits='A violation proves all eight cannot finish under these bounds; no violation does not prove a feasible schedule. Model/state fees and transfers ignored; same trace panels are correlated.')
    (root/'capacity_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(demands=demands,ruled_out={panel:sum(r['full_completion_ruled_out'] for r in rows if r['panel']==panel) for panel in ('base','tight')},rows=rows),indent=2))


if __name__=='__main__':main()
