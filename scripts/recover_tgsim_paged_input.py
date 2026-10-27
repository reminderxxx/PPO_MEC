"""Explicit-column recovery, retained wire pages and bounded native composition."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

try:
    from scripts.prepare_tgsim_input import ASSETS, BudgetLimit, digest, save_new
    from scripts.prepare_tgsim_paged_input import ROOT, PagedBudget, acquire, disk_bytes, system_id_key
except ModuleNotFoundError:
    from prepare_tgsim_input import ASSETS, BudgetLimit, digest, save_new
    from prepare_tgsim_paged_input import ROOT, PagedBudget, acquire, disk_bytes, system_id_key


class NativeComposer:
    """Opaque IDs have equality semantics; never assume their lexical order."""
    def __init__(self,analysis,max_decoded_bytes):
        self.analysis=analysis;self.cap=max_decoded_bytes;self.decoded=0;self.connections={}

    def close(self):
        for connection in self.connections.values():connection.commit();connection.close()

    def __call__(self,path,source,writer,expected_rows,previous,budget,*,encoded_gzip):
        asset=source['asset_id']
        if asset not in self.connections:
            connection=sqlite3.connect(self.analysis/f'{asset}_system_ids.sqlite')
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('CREATE TABLE identities(id TEXT PRIMARY KEY) WITHOUT ROWID')
            self.connections[asset]=connection
        connection=self.connections[asset];count=0;first=last=None;since_guard=0
        def lines(stream):
            nonlocal since_guard
            while True:
                line=stream.readline(65537)
                if not line:break
                size=len(line.encode('utf-8'))
                if len(line)>65536 or self.decoded+size>self.cap:
                    raise BudgetLimit('decoded CSV/line cap reached')
                self.decoded+=size;since_guard+=size
                if since_guard>=1024**2:budget.guard();since_guard=0
                yield line
        opener=gzip.open if encoded_gzip else open
        with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(lines(stream));expected=['source_row_id']+source['expected_native_fields']
            if next(reader,None)!=expected:raise ValueError('explicit native page header differs')
            for row in reader:
                if len(row)!=len(expected) or count>=expected_rows:raise ValueError('ragged/extra native page row')
                system_id_key(row[0])
                try:connection.execute('INSERT INTO identities VALUES(?)',(row[0],))
                except sqlite3.IntegrityError as error:raise ValueError('duplicate opaque system ID across pages') from error
                writer.writerow(row[1:]);count+=1;first=first or row[0];last=row[0]
                if count%10000==0:connection.commit();budget.guard()
        connection.commit();budget.guard()
        if count!=expected_rows:raise ValueError('missing/short native page')
        return None,dict(rows=count,first_source_row_id=first,last_source_row_id=last,
                         encoded_gzip=encoded_gzip,opaque_system_ids_unique=True)


def effective_caps(plan):
    caps=dict(max_download_bytes_new_task=536870912,max_wall_clock_seconds=2700,
              max_rows_to_index=6000000,maximum_http_requests=144,max_decoded_csv_bytes=1288490188)
    if plan.get('continuation_parent_completion'):
        parent=Path(plan['continuation_parent_completion'])
        if not parent.resolve().is_relative_to(ROOT.resolve()) or digest(parent)!=plan['continuation_parent_completion_sha256']:
            raise ValueError('sealed parent completion identity differs')
        previous=json.loads(parent.read_text())
        if previous['status']!='failed_preserved_do_not_retry' or previous['rows_indexed']!=0:
            raise ValueError('only this pre-native-index failed parent is eligible')
        caps.update(max_download_bytes_new_task=caps['max_download_bytes_new_task']-previous['total_body_download_bytes_upper_bound'],
                    max_wall_clock_seconds=int(caps['max_wall_clock_seconds']-previous['elapsed_seconds']),
                    maximum_http_requests=caps['maximum_http_requests']-previous['http_requests'],
                    max_decoded_csv_bytes=caps['max_decoded_csv_bytes']-previous['decoded_csv_bytes'])
        if plan['prior_accounted_body_bound']!=previous['two_task_body_upper_bound']:
            raise ValueError('parent transfer bill not carried forward')
    return caps


def execute(plan_path,plan_sha):
    if digest(plan_path)!=plan_sha:raise ValueError('recovery plan hash mismatch')
    plan=json.loads(plan_path.read_text());grant=Path(plan['authorization_file'])
    if digest(grant)!=plan['authorization_sha256']:raise ValueError('direct human authorization hash mismatch')
    authorization=json.loads(grant.read_text())
    if not authorization.get('execution_authorized') or authorization.get('scope')!='same_source_input_recovery_zero_science':
        raise ValueError('input recovery grant missing')
    if any(digest(ROOT/name)!=sha for name,sha in plan['source_hashes'].items()):
        raise ValueError('frozen source hash mismatch')
    if tuple(s['asset_id'] for s in plan['sources'])!=ASSETS:raise ValueError('source pool differs')
    if (any(plan[k]!=v for k,v in effective_caps(plan).items())
            or plan['maximum_trajectory_pages']!=120 or plan['page_rows']!=50000
            or plan['max_cumulative_new_disk_bytes_including_old_failure']!=2147483648
            or plan['training_frames']!=0 or plan['evaluation_frames']!=0):raise ValueError('recovery caps differ')
    for source in plan['sources']:
        if source['fixed_query']!={'$select':':id as source_row_id,'+','.join(source['expected_native_fields']),'$order':':id','$limit':50000,'$offset':0}:
            raise ValueError('explicit fixed query differs')
    raw=Path(plan['destination_raw_directory']);analysis=Path(plan['destination_analysis_directory'])
    for path in (raw,analysis):
        if path.exists() or not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('fresh worktree paths required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('freeze clean source')
    raw.mkdir(parents=True,exist_ok=False);analysis.mkdir(parents=True,exist_ok=False)
    budget=PagedBudget(plan,[Path(p) for p in plan['disk_roots']]);composer=NativeComposer(analysis,plan['max_decoded_csv_bytes'])
    record=dict(status='running',contract='tgsim_explicit_native_recovery_v1',proposal_sha256=plan_sha,grant_sha256=digest(grant),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                command_arguments=dict(proposal=str(plan_path),grant=str(grant)),started_at_unix=time.time(),
                cumulative_disk_roots=plan['disk_roots'],training_frames=0,evaluation_frames=0,hidden_openings=0,
                automatic_retry=False,source_hashes=plan['source_hashes'])
    save_new(analysis/'execution_claim.json',dict(record,proposal=plan))
    def deadline(signum,frame):raise TimeoutError('bounded remaining recovery deadline')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,plan['max_wall_clock_seconds'])
    try:
        record.update(acquire(plan,raw,analysis,budget,append_fn=composer,gzip_pages=True))
        budget.guard();record['status']='acquired_and_input_checked_not_science_qualified'
    except BaseException as error:
        record.update(status='failed_preserved_do_not_retry',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);composer.close()
        record.update(elapsed_seconds=budget.clock()-budget.started,http_requests=budget.requests,rows_indexed=budget.rows,
                      decoded_csv_bytes=composer.decoded,total_body_download_bytes=budget.download_bytes,
                      total_body_download_bytes_upper_bound=budget.download_bytes+budget.interrupted_body_upper_bound,
                      two_task_body_upper_bound=plan['prior_accounted_body_bound']+budget.download_bytes+budget.interrupted_body_upper_bound,
                      byte_accounting='retained encoded wire entities plus interrupted-read bound; old unretained error entity and exact wire traffic remain unverified',
                      cumulative_disk_bytes=disk_bytes(plan['disk_roots']),
                      raw_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(raw.iterdir()) if p.is_file()])
        save_new(analysis/'completion.json',record)
    print(json.dumps({k:record[k] for k in ['status','http_requests','rows_indexed','total_body_download_bytes','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan_sha',required=True)
    args=parser.parse_args();execute(args.plan,args.plan_sha)
