"""Reconstruct cache allocations/loading and use from immutable frame receipts.

Used objects are observed physical use, not counterfactual causal net benefit.
Resident and pending objects both reserve capacity. Base/adapter are separate.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash


class CacheLedger:
    def __init__(self,case):
        self.case=case
        self.catalog={o.object_id:asdict(o) for o in case['catalog']}
        self.nodes={(a.vehicle_id,a.request_id,n.node_id):n for a in case['arrivals'] for n in a.workflow.nodes}
        self.capacity=case['options']['model_cache_config']['capacity_bytes']
        self.lives=[];self.counts=Counter();self.reset()

    def reset(self):
        self.allocated=defaultdict(set);self.resident=defaultdict(set)
        self.current={};self.paid=defaultdict(int)

    def evict(self,rid,objects):
        for obj in objects:
            assert obj in self.allocated[rid] and obj in self.resident[rid],('invalid eviction',rid,obj)
            self.allocated[rid].remove(obj);self.resident[rid].remove(obj)
            self.current.pop((rid,obj));self.paid.pop((rid,obj),None)

    def consume(self,frame):
        if frame['frame_index']==0:self.reset()
        receipt=frame['receipt'];self.counts['frames']+=1
        assert all(0<=v<=1 for v in receipt['link_usage_by_rsu'].values())
        for rid,eviction in receipt.get('explicit_evictions_by_rsu',{}).items():
            self.evict(rid,eviction['evicted_object_ids'])
        for event in frame['events']:
            cache=event['cache_request'];rid=event['cache_target_rsu_id']
            admitted=cache.get('admitted_object_ids',[])
            evicted=cache.get('evicted_object_ids',[])
            if evicted:self.evict(rid,evicted)
            if admitted:
                control=event['control']
                optional=((control.get('cache_request_id') or control['request_id'])!=event['request_id']
                    or control['cache_node_id']!=event['node_id'] or rid!=event['target_rsu_id'])
                for obj in admitted:
                    assert obj in self.catalog and obj not in self.allocated[rid],('duplicate admission',rid,obj)
                    self.allocated[rid].add(obj)
                    life=dict(object_id=obj,object_type=self.catalog[obj]['object_type'],optional=optional,
                        observed_used=False,actual_transfer_bytes=0)
                    self.lives.append(life);self.current[rid,obj]=life
                if 'reserved_bytes' in cache:
                    assert cache['reserved_bytes']==sum(self.catalog[k]['resident_size_bytes'] for k in admitted)
                self.counts['admissions']+=len(admitted)
            if event['status'] in ('rsu_compute','output_ready'):
                site=event['target_rsu_id']
                node=self.nodes[event['vehicle_id'],event['request_id'],event['node_id']]
                for obj in (f'base:{node.required_base_model}',f'adapter:{node.required_adapter}'):
                    assert obj in self.resident[site],('served before fully loaded',site,obj)
                    self.current[site,obj]['observed_used']=True
                self.counts['verified_compute_events']+=1
            for site,objects in self.allocated.items():
                assert sum(self.catalog[k]['resident_size_bytes'] for k in objects)<=self.capacity,('capacity',site)
                for obj in objects:assert set(self.catalog[obj]['dependency_ids'])<=objects,('orphan',site,obj)
        transferred=0
        for event in receipt['cache_events']:
            rid,obj=event['rsu_id'],event['object_id'];key=(rid,obj)
            assert obj in self.allocated[rid] and obj not in self.resident[rid],('unexpected transfer',rid,obj)
            self.paid[key]+=event['transferred_bytes'];transferred+=event['transferred_bytes']
            self.current[key]['actual_transfer_bytes']+=event['transferred_bytes']
            assert self.paid[key]<=self.catalog[obj]['transfer_size_bytes']
            if event['completed']:
                assert self.paid[key]==self.catalog[obj]['transfer_size_bytes']
                assert set(self.catalog[obj]['dependency_ids'])<=self.resident[rid],('dependency not ready',rid,obj)
                self.resident[rid].add(obj);self.counts['completed_loads']+=1
        assert transferred==receipt['model_transfer_bytes']

    def summary(self):
        result=dict(counts=dict(self.counts),objects={})
        for kind in ['base_model','adapter']:
            optional=[x for x in self.lives if x['object_type']==kind and x['optional']]
            result['objects'][kind]=dict(optional_admissions=len(optional),
                optional_observed_used=sum(x['observed_used'] for x in optional),
                optional_transfer_bytes=sum(x['actual_transfer_bytes'] for x in optional),
                optional_unused_transfer_bytes=sum(x['actual_transfer_bytes'] for x in optional if not x['observed_used']))
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--full_cohort_runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=json.loads((a.study/'protocol.json').read_text())
    source_path=Path(plan['parent_protocol']);source=json.loads(source_path.read_text())
    assert file_hash(source_path.parent/'protocol.pt')==source['assets']['protocol.pt']
    data=torch.load(source_path.parent/'protocol.pt',weights_only=False)
    cases={c['case_id']:c for c in data['cases']['train']};results=[]
    for run in a.full_cohort_runs:
        assert (run/'complete.json').exists()
        for row in map(json.loads,(run/'results.jsonl').read_text().splitlines()):
            path=run/f'{row["variant"]}_{row["case_id"]}.jsonl'
            assert file_hash(path)==row['trace_sha256']
            ledger=CacheLedger(cases[row['case_id']])
            with path.open() as stream:
                for line in stream:ledger.consume(json.loads(line))
            results.append(dict(seed=row['seed'],variant=row['variant'],case_id=row['case_id'],**ledger.summary()))
    aggregate={}
    for variant in ['original','relocation']:
        selected=[x for x in results if x['variant']==variant]
        aggregate[variant]=dict(counts={k:sum(x['counts'].get(k,0) for x in selected) for k in ['frames','admissions','verified_compute_events','completed_loads']},objects={})
        for kind in ['base_model','adapter']:
            aggregate[variant]['objects'][kind]={k:sum(x['objects'][kind][k] for x in selected)
                for k in ['optional_admissions','optional_observed_used','optional_transfer_bytes','optional_unused_transfer_bytes']}
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='pass',scope='Final all-train-cohort physical allocation/loading/service receipts only',
        scientific_git=plan['source_commit'],aggregate=aggregate,cases=results,
        limits=['Object use is not counterfactual net gain','Proxy bytes/latencies, not actual AI weights',
            'No training ledger or other ablation physical trace is independently reconstructed here'])
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',aggregate=aggregate)))


if __name__=='__main__':main()
