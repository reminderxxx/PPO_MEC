"""Bounded publisher-script and corrected-listing diagnosis; never downloads images."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

if __package__:
    from . import acquire_public_task_metadata as common
else:
    import acquire_public_task_metadata as common

BUNDLE = 'https://data.mendeley.com/datasets/bundle.js?8814bfd8bfcb6231e2b5'
FILES = 'https://data.mendeley.com/public-api/datasets/j82bnw7gsr/files?folder_id=root&version=1'
CAPS = (3145728, 131072)


def load(root):
    p = json.loads((root/'plan.json').read_text()); claim = json.loads((root/'claim.json').read_text())
    assert common.digest(root/'plan.json') == claim['plan_sha256'], 'plan binding'
    c = json.loads(Path(p['contract']).read_text())
    assert common.digest(Path(p['contract'])) == p['contract_sha256'], 'new authorization contract'
    assert c['human_instruction']=='你继续找办法' and c['urls']==[BUNDLE,FILES], 'fixed diagnosis sources'
    assert c['new_caps']=={'http_attempts':2,'bundle_body_plus_upper':3145728,'file_list_body_plus_upper':131072,'shared_clock_wall_seconds_including_audit':300,'images_models':0}, 'fixed new bounds'
    prior = json.loads(Path(c['prior_audit']).read_text())
    assert common.digest(Path(c['prior_audit']))==c['prior_audit_sha256'], 'immutable prior ledger'
    assert (prior['requests'],prior['body_bytes'],prior['interrupted_read_upper_bound'])==(8,932050,0), 'prior costs once'
    assert claim['boot_uuid']==common.extension_boot_id(), 'current diagnosis boot identity'
    assert 0<=claim['start_clock']<=common.clock(), 'shared clock order'
    return p,claim,prior


def init(plan_path, expected, *, verify_source=True):
    assert common.digest(plan_path)==expected, 'plan SHA'
    p=json.loads(plan_path.read_text()); root=Path(p['raw_root'])
    if verify_source:common.source_check(p)
    root.mkdir(parents=True,exist_ok=False)
    (root/'plan.json').write_bytes(plan_path.read_bytes())
    common.save(root/'claim.json',{'plan_sha256':expected,'start_clock':common.clock(),'boot_uuid':common.extension_boot_id()})
    load(root)


def prove(root,p,proof_path):
    body=root/'request_0/body.bin'; proof=json.loads(proof_path.read_text()); text=body.read_text()
    assert proof['bundle_sha256']==common.digest(body) and proof['corrected_url']==FILES, 'bundle correction proof'
    fragments=[]
    for part in proof['snippets']:
        assert 0<=part['start']<part['end']<=len(text) and text[part['start']:part['end']]==part['text'], 'literal publisher source fragment'
        fragments.append(part['text'])
    joined='\n'.join(fragments)
    assert all(token in joined for token in ('folder_id','root','/files')), 'published folder/root/files convention'
    review=proof['independent_semantic_review']
    assert (review['reviewer']=='/root/quality_source_independent_check' and review['bundle_sha256']==proof['bundle_sha256'] and
            review['public_files_call_verified'] is True and review['root_folder_relationship_verified'] is True and
            review['decision']=='ACCEPTED_PUBLISHER_CALL_RELATIONSHIP'), 'independent actual-call semantic acceptance required'
    assert common.digest(Path(p['official_page']))==p['official_page_sha256'] and BUNDLE.split('mendeley.com')[1] in Path(p['official_page']).read_text(), 'bundle literal in original official page'


def get(root,index,proof_path=None,proof_sha256=None,*,opener=None,verify_source=True):
    root=Path(root); assert not (root/'terminal.json').exists(), 'terminal immutable'
    p,claim,prior=load(root)
    if verify_source:common.source_check(p)
    assert index in (0,1), 'two exact requests only'
    folders=sorted(root.glob('request_*'));assert len(folders)==index, 'no duplicate/retry/out-of-order'
    for folder in folders:assert json.loads((folder/'receipt.json').read_text())['status']=='complete', 'prior failure stops'
    if index:
        assert proof_sha256==common.digest(proof_path), 'prospectively frozen proof SHA'
        prove(root,p,proof_path)
    else:
        assert common.digest(Path(p['official_page']))==p['official_page_sha256'] and BUNDLE.split('mendeley.com')[1] in Path(p['official_page']).read_text(), 'published bundle URL'
    remaining=300-(common.clock()-claim['start_clock'])-10
    assert remaining>0, 'total diagnosis deadline'
    assert common.stored_bytes(p)+sum(CAPS)+1048576<=268435456, 'cumulative storage reserve'
    folder=root/('request_%d'%index);folder.mkdir(exist_ok=False)
    url=(BUNDLE,FILES)[index];record={'url':url,'index':index,'bytes':0,'upper':65536,'status':'requesting','start':common.clock(),'retry':False}
    if proof_path:
        (folder/'proof.json').write_bytes(proof_path.read_bytes());record['proof_sha256']=proof_sha256
    def deadline(*unused):raise TimeoutError('diagnosis absolute deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        req=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'PPO-MEC-public-entry-diagnosis/1.0'})
        try:response=(opener or urllib.request.build_opener(common.NoRedirect())).open(req,timeout=min(30,remaining))
        except urllib.error.HTTPError as error:response=error
        with response,(folder/'body.bin').open('xb') as f:
            record.update(http_status=response.status,headers=dict(response.headers),final_url=response.geturl())
            assert response.geturl()==url, 'no redirect'
            mime=response.headers.get('Content-Type','').lower().split(';')[0]
            assert mime.startswith('text/') or mime in ('application/javascript','application/json'), 'metadata MIME'
            assert response.headers.get('Content-Encoding','identity').lower() in ('','identity'), 'identity encoding'
            limit=CAPS[index]-65536;length=response.headers.get('Content-Length')
            assert length is None or 0<=int(length)<=limit, 'declared response cap'
            while length is None or record['bytes']<int(length):
                size=min(65536,limit-record['bytes']);assert size>0, 'body cap'
                record['upper']=size;block=response.read1(size);record['upper']=0
                if not block:break
                record['bytes']+=len(block);f.write(block)
            assert response.status==200, 'HTTP error retained; stop'
            assert length is None or record['bytes']==int(length), 'truncated body'
        text=(folder/'body.bin').read_text()
        if index:
            obj=json.loads(text)
            assert not (isinstance(obj,dict) and 'error' in obj), 'API error retained; stop'
            assert isinstance(obj,list) or isinstance(obj,dict) and isinstance(obj.get('files'),list), 'not a file list'
        record['status']='complete'
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        record.update(finish=common.clock(),retained_bytes=(folder/'body.bin').stat().st_size if (folder/'body.bin').exists() else 0,body_sha256=common.digest(folder/'body.bin') if (folder/'body.bin').exists() else None)
        common.save(folder/'receipt.json',record)
    return record


def _audit(root,output):
    root=Path(root);assert not output.exists() and not (root/'terminal.json').exists(), 'one terminal consumer only'
    p,claim,prior=load(root);records=[]
    for index,folder in enumerate(sorted(root.glob('request_*'))):
        assert folder.name=='request_%d'%index and index<2, 'request sequence'
        r=json.loads((folder/'receipt.json').read_text());body=folder/'body.bin'
        assert r['url']==(BUNDLE,FILES)[index] and not r['retry'], 'fixed source'
        assert r['body_sha256']==(common.digest(body) if body.exists() else None), 'body hash'
        assert (body.stat().st_size if body.exists() else 0)==r['retained_bytes']<=r['bytes'], 'retained body'
        assert 0<=r['upper']<=65536 and r['bytes']+r['upper']<=CAPS[index], 'per-file bill'
        assert r['finish']>=r['start']>=claim['start_clock'], 'clock order'
        assert not records or records[-1]['status']=='complete', 'requests after error'
        if index:
            assert common.digest(folder/'proof.json')==r['proof_sha256'], 'frozen source proof'
            prove(root,p,folder/'proof.json')
        headers={k.lower():v for k,v in r.get('headers',{}).items()};mime=headers.get('content-type','').lower().split(';')[0]
        if r['retained_bytes']:
            assert mime.startswith('text/') or mime in ('application/javascript','application/json'), 'retained metadata MIME'
            assert headers.get('content-encoding','identity').lower() in ('','identity'), 'retained identity encoding'
            body.read_text()
        if r['status']=='complete':
            assert r['http_status']==200 and r['upper']==0 and r['bytes']==r['retained_bytes'], 'complete transport'
            assert mime.startswith('text/') or mime in ('application/javascript','application/json'), 'complete metadata MIME'
            assert headers.get('content-encoding','identity').lower() in ('','identity'), 'complete identity encoding'
            assert r['final_url']==r['url'], 'exact final URL'
            length=headers.get('content-length');assert length is None or int(length)==r['bytes'], 'complete declared length'
            body.read_text()
            if index:
                obj=json.loads(body.read_text());assert isinstance(obj,list) or isinstance(obj,dict) and isinstance(obj.get('files'),list), 'file list container'
                assert not isinstance(obj,dict) or 'error' not in obj, 'API error'
        records.append(r)
    body=sum(r['bytes'] for r in records);upper=sum(r['upper'] for r in records);elapsed=common.clock()-claim['start_clock']
    assert prior['requests']+len(records)<=12 and prior['body_bytes']+body+upper<=5242880, 'cumulative M caps'
    storage=common.stored_bytes(p);assert storage+1048576<=268435456, 'all-new-old storage'
    result={'new_requests':len(records),'new_body_bytes':body,'new_interrupted_upper':upper,'cumulative_requests':prior['requests']+len(records),'cumulative_body_bytes':prior['body_bytes']+body,'cumulative_interrupted_upper':upper,'elapsed_new_scope':elapsed,'new300s_pass':0<=elapsed<=300,'failures':sum(r['status']!='complete' for r in records),'storage_before_audit':storage,'terminal_reserve':1048576,'input_qualified':False,'images':0,'model_calls':0,'paper_goal_complete':False,'prior_deadline_failure_unchanged':True,'wire_total':'UNAVAILABLE;returned-body/read bound only','receipt_sha256':[common.digest(f/'receipt.json') for f in sorted(root.glob('request_*'))]}
    common.save(output,result);common.save(root/'terminal.json',{'audit':str(output),'sha256':common.digest(output)})
    return result


def audit(root,output):
    root=Path(root)
    assert not (root/'consumer_failure.json').exists() and not output.exists() and not (root/'terminal.json').exists(), 'one terminal consumer only'
    claim=json.loads((root/'claim.json').read_text());remaining=300-(common.clock()-claim['start_clock'])
    def deadline(*unused):raise TimeoutError('consumer diagnosis absolute deadline')
    previous=signal.signal(signal.SIGALRM,deadline)
    try:
        if remaining<=0:raise TimeoutError('no consumer time remaining')
        signal.setitimer(signal.ITIMER_REAL,min(remaining,300))
        return _audit(root,output)
    except BaseException as error:
        common.save(root/'consumer_failure.json',{'error_type':type(error).__name__,'error':str(error),
                    'unverified_attempt_directory_count':len(list(root.glob('request_*'))),
                    'new_body_upper_conservative':sum(CAPS),'old_deadline_failure_unchanged':True,
                    'input_qualified':False,'paper_goal_complete':False,'audit_success':False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('operation',choices=['init','get','audit']);q.add_argument('--root',type=Path);q.add_argument('--plan',type=Path);q.add_argument('--sha256');q.add_argument('--index',type=int);q.add_argument('--proof',type=Path);q.add_argument('--proof_sha256');q.add_argument('--output',type=Path);a=q.parse_args()
    if a.operation=='init':init(a.plan,a.sha256)
    elif a.operation=='get':print(json.dumps(get(a.root,a.index,a.proof,a.proof_sha256)))
    else:print(json.dumps(audit(a.root,a.output)))


if __name__=='__main__':main()
