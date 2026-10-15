"""Freeze common train-only cohorts and paired base/tight development panels."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import arguments,_load_protocol,_hash_file
from src.data.mobility.ngsim_provider import NGSIMProvider
from src.envs.core.joint_training_scenarios import build_scenarios,public_scenarios
from src.envs.core.joint_learned_prediction import fit_predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_manifest',type=Path,required=True)
    p.add_argument('--output_root',type=Path,required=True)
    p.add_argument('--predictor_epochs',type=int,default=30)
    a=p.parse_args();torch.set_num_threads(1)
    source=json.loads(a.source_manifest.read_text());original=arguments(source['command'][1:])
    for key,path in [('ngsim',original.mobility_csv_path),('alibaba',original.workflow_csv_path)]:
        assert _hash_file(path)==source['raw_data_sha256'][key]
    splits,workflows,rsus=_load_protocol(original)
    for split in ('train','dev'):
        assert [{k:v for k,v in w.items() if k!='frames'} for w in splits[split]]==source['splits'][split]
    raw=NGSIMProvider(original.mobility_csv_path,max_rows=original.max_mobility_rows).get_loaded_frames()
    scenarios=build_scenarios(splits,rsus,raw)
    assert len(scenarios['train'])==40 and len(scenarios['dev'])==16
    assert all(w['uncovered_vehicle_frames']==0 for ws in scenarios.values() for w in ws)
    a.output_root.mkdir(parents=True,exist_ok=False)
    torch.save(dict(scenarios=scenarios,workflows=workflows),a.output_root/'protocol.pt')
    bundle=fit_predictor(scenarios['train'],epochs=a.predictor_epochs,horizon=8,group_by_interval=True)
    torch.save(bundle,a.output_root/'predictor.pt')
    options={k:v for k,v in source['environment_options'].items() if k not in ('environment_revision','strict_cold_cache_mask')}
    options.update(model_cache_config=source['costs']['model_cache'],service_load_sharing=True,
                   coordination_intent_enabled=True,link_reservation_enabled=True)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_manifest=str(a.source_manifest.resolve()),source_manifest_sha256=_hash_file(a.source_manifest),
        raw_data_sha256=source['raw_data_sha256'],protocol_sha256=_hash_file(a.output_root/'protocol.pt'),
        predictor_sha256=_hash_file(a.output_root/'predictor.pt'),
        predictor={k:v for k,v in bundle.items() if k!='state_dict'},
        source_splits={k:source['splits'][k] for k in ('train','dev')},scenarios=public_scenarios(scenarios),
        environment_options=options,costs=source['costs'],
        primary_panels=dict(base=.5,tight=.5),holdout_status='sealed_not_evaluated',
        cohort_rule='train interval only; continuously covered; descending coverage departures, displacement, ID; eight vehicles',
        limits=['geometry departures are not guaranteed positive migration benefits','cohorts sharing a raw interval are not independent samples',
                'tight radius is .65/1.2 times original; all algorithms share both panels and all resource costs'])
    (a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output_root),coverage={split:{panel:sum(w['coverage_departures'] for w in ws if w['panel']==panel)
        for panel in sorted({w['panel'] for w in ws})} for split,ws in scenarios.items()},predictor=manifest['predictor']),indent=2))


if __name__=='__main__':main()
