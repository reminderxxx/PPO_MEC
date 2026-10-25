"""Complete prospective unseen-DAG composition matrix; no outcome access."""
import argparse
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_persistent_workflow_learning import dump,sha


def complete_pairs(count=8):
    return list(itertools.combinations_with_replacement(range(count),2))


def build_payload(p,source_sha256):
    if len(p['units'])!=12 or any(u['split']!='confirmation' for u in p['units']):raise ValueError('Need twelve sealed-input outer windows')
    heldout=p['all_workflows'][8:]
    if len(heldout)!=8:raise ValueError('Need eight unseen workflow identities')
    units=[]
    for case,(a,b) in enumerate(complete_pairs()):
        base=p['units'][case//3]
        units.append(dict(base,index=2400+case,outer_window_index=base['index'],case_index=case,workflows=[heldout[a],heldout[b]]))
    return dict(p,units=units,contract='event_joint_complete_workload_matrix_v1',supporting_protocol=dict(created_before_confirmation_performance=True,
        rule='All 36 combinations with replacement of 8 unseen DAGs in source order; three compositions per original outer time window, no outcome ranking',
        primary_protocol_unchanged=True,statistical_outer_windows=12,workload_rows_are_not_independent_time_clusters=True,source_confirmation_sha256=source_sha256))


def prepare(source,out):
    out.mkdir(parents=True);p=json.loads(source.read_text());payload=build_payload(p,sha(source));dump(out/'inputs.json',payload)
    dump(out/'command.json',dict(argv=sys.argv,source_confirmation_sha256=sha(source),producer_sha256=sha(__file__),output_sha256=sha(out/'inputs.json'),outcomes_read=0))
    print(json.dumps(dict(cases=36,outer_windows=12,inputs_sha256=sha(out/'inputs.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(Path(a.inputs),Path(a.output))
