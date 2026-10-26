"""Read-only attribution of optional object waste in frozen stream receipts.

Future object use is a retrospective outcome, never an online policy input.
The admission context uses only that frame's announced execution controls.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stream_cache_receipts import CacheLedger
from scripts.probe_stream_main_opportunity import file_hash


class ContextLedger(CacheLedger):
    def consume(self, frame):
        start = len(self.lives)
        super().consume(frame)
        offset = start
        targets = {e['vehicle_id']: e['target_rsu_id'] for e in frame['events']}
        for event in frame['events']:
            site = event['cache_target_rsu_id']
            control = event['control']
            relation = ('own_target' if site == event['target_rsu_id'] else
                        'peer_target' if site in targets.values() else 'unannounced_target')
            for obj in event['cache_request'].get('admitted_object_ids', []):
                life = self.lives[offset]
                assert life['object_id'] == obj
                life.update(site=site, frame=frame['frame_index'], relation=relation,
                    vehicle_id=event['vehicle_id'], request_id=event['request_id'],
                    cache_request_id=control.get('cache_request_id') or control['request_id'],
                    cache_node_id=control['cache_node_id'], active_node_id=event['node_id'])
                offset += 1
        assert offset == len(self.lives)


def summarize(lives):
    groups = defaultdict(list)
    for life in lives:
        if life['optional']:
            groups[life['object_type'] + '/' + life['relation']].append(life)
    return {key: dict(admissions=len(rows), used=sum(x['observed_used'] for x in rows),
        bytes=sum(x['actual_transfer_bytes'] for x in rows),
        unused_bytes=sum(x['actual_transfer_bytes'] for x in rows if not x['observed_used']))
        for key, rows in sorted(groups.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    all_lives = defaultdict(list)
    evidence = []
    for run in a.runs:
        plan = json.loads((run/'protocol.json').read_text())
        study = json.loads((Path(plan['study'])/'protocol.json').read_text())
        protocol = Path(study['parent_protocol'])
        manifest = json.loads(protocol.read_text())
        assert file_hash(protocol.parent/'protocol.pt') == manifest['assets']['protocol.pt']
        data = torch.load(protocol.parent/'protocol.pt', weights_only=False)
        cases = {c['case_id']: c for c in data['cases']['train']}
        complete = json.loads((run/'complete.json').read_text())
        assert file_hash(run/'results.jsonl') == complete['results_sha256']
        for row in map(json.loads, (run/'results.jsonl').read_text().splitlines()):
            trace = run/f'{row["variant"]}_{row["case_id"]}.jsonl'
            assert file_hash(trace) == row['trace_sha256']
            ledger = ContextLedger(cases[row['case_id']])
            for line in trace.open():
                ledger.consume(json.loads(line))
            all_lives[row['variant']].extend(ledger.lives)
            evidence.append(dict(seed=row['seed'], case_id=row['case_id'], variant=row['variant'],
                path=str(trace.resolve()), sha256=file_hash(trace), groups=summarize(ledger.lives)))
    report = dict(reviewed_at=datetime.now(timezone.utc).isoformat(), literature_cutoff='2026-10-26',
        target_venue='IEEE TMC methodology / training-only diagnosis', artifact_run_id=a.output.name,
        policy_version='tmc_review_policy_v3_20260621',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        evidence_level='HASH_VERIFIED_EXISTING_TRAIN_RECEIPTS',
        verdict='Bounded preparation diagnosis; formal readiness Unverifiable',
        groups={v:summarize(rows) for v,rows in all_lives.items()}, evidence=evidence,
        limits=['Observed use is not causal benefit', 'Same three consumed training time sessions',
                'Peer target means a contemporaneous announcement, not proof of peer demand',
                'Future use labels are retrospective and forbidden as policy inputs'])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['groups']))


if __name__ == '__main__':
    main()
