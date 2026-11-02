"""A single explicitly budgeted recovery; never changes the sealed parent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from urllib.parse import urljoin
from PIL import Image
import acquire_helmet_prospective as original
import prepare_public_helmet_pilot as transport

W=Path(__file__).resolve().parents[1]
P=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_helmet_prospective_20261102_v1')
R=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_helmet_prospective_recovery_20261102_v1')
SCOPE_SHA='72b79e7c923898569fc00e04f48fb344a4568380302ddbf376b016924b24eef0'
SELECTION_SHA='388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743'
PARENT_PLAN_SHA='e299002e7d19af6c1518dc034e2f1250b65d1e517f57863b89edc27afb8cf282'
PARENT_COMMIT='2ea28af81d2d6aa3491cb317dc339670d5a5753b'
PARENT_AUDIT_SHA='f5fddd95deffa6f8069bac171115ff30c1e8251e94c5deeeaeb95677e6c01da6'
CAPS={'http':35,'body':60989703,'wall':1384,'storage':134217728}
SOURCE_FILES=('scripts/recover_helmet_prospective.py','scripts/audit_helmet_prospective_recovery.py','scripts/acquire_helmet_prospective.py','scripts/audit_helmet_prospective.py','scripts/prepare_public_helmet_pilot.py','scripts/audit_public_helmet_pilot.py')
require,regular,sha,load,clock,boot=transport.require,original.regular,transport.sha,transport.load,transport.clock,transport.boot


def checked(r):
    p=regular(r['path']);require(sha(p)==r['sha256'],'fixed reference SHA '+str(p));return load(p)


def usage(plan):
    roots=[p for p in plan['storage_roots']if Path(p).exists()]
    return transport.storage(roots)


def save(plan,path,obj):
    data=(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    require(usage(plan)+len(data)+1048576<=CAPS['storage'],'all4-root storage reserve')
    with Path(path).open('xb')as f:f.write(data)


def parent_facts(pp,a,t,parent):
    require(pp['schema']=='helmet24_acquisition_v1' and pp['source_commit']==PARENT_COMMIT and not any(k in pp for k in ('parent','recovery','zero_http_predecessor')),'single initial parent; no recursion')
    require(pp['new_caps']=={'http':48,'body':67108864,'wall':1500,'storage':134217728},'original caps retained')
    require(a['plan_sha256']==PARENT_PLAN_SHA and a['input_qualified'] is False and t['sha256']==t['audit_sha256']==PARENT_AUDIT_SHA and t['input_qualified'] is False,'fixed failed audit/terminal')
    b=a['bill'];require(b['request_claims']==b['receipts']==14 and b['retained_body_bytes']==4644800 and b['interrupted_read_upper_bound']==65536 and not b['missing_receipt'],'exact parent cost')
    require(parent['returncode']==2 and parent['elapsed_seconds']==514.709546917,'parent full executor cost')
    require(4644800+65536+1408825+CAPS['body']==67108864 and 14+CAPS['http']==49 and 514.709546917+CAPS['wall']<1900,'explicit cumulative quotas')


def parent_binding(scope):
    pp=checked(scope['parent']['plan']);a=checked(scope['parent']['audit']);t=checked(scope['parent']['terminal']);parent=checked(scope['parent']['parent_process']);d=checked(scope['parent']['diagnosis']);parent_facts(pp,a,t,parent)
    require(scope['parent']['plan']['sha256']==PARENT_PLAN_SHA and scope['parent']['audit']['sha256']==PARENT_AUDIT_SHA,'exact initial plan/audit pins')
    oldroot=W/'data/raw/task_quality/helmet_prospective_20261102_v1';require(pp['raw_root']==str(oldroot) and sha(regular(oldroot/'plan.json'))==PARENT_PLAN_SHA,'retained old raw plan')
    claim=checked(d['provenance']['execution_claim']);require(claim['plan_sha256']==PARENT_PLAN_SHA,'old claim identity')
    complete=checked(d['provenance']['producer_completion']);failure=checked(d['provenance']['producer_failure']);require(len(complete['acquired'])==6 and complete['input_complete'] is False and failure['type']=='timeout','sealed parent failure')
    require(len(d['all14_request_inventory'])==14 and len(d['reusable_complete_originals'])==6,'fixed parent inventory')
    for i,x in enumerate(d['all14_request_inventory']):
        rr=checked(x['receipt']);cc=checked(x['claim']);bp=regular(x['body']['path']);require(sha(bp)==x['body']['sha256']==rr['body_sha256'] and bp.stat().st_size==rr['bytes']==rr['retained_bytes'],'old retained bytes SHA')
        require(str(bp)==str(oldroot/f'request_{i:03d}'/'body.bin') and str(Path(x['receipt']['path']))==str(oldroot/f'request_{i:03d}'/'receipt.json') and str(Path(x['claim']['path']))==str(oldroot/f'request_{i:03d}'/'claim.json'),'old fixed request paths')
        require(a['bill']['request_fingerprints'][i]['receipt_sha256']==x['receipt']['sha256'] and a['bill']['request_fingerprints'][i]['claim_sha256']==x['claim']['sha256'],'old audit fingerprint vector')
        require(all(rr[k]==cc[k]for k in ('url','role','redirect_from','started')),'old request claim binding')
    for i,x in enumerate(d['reusable_complete_originals']):
        im=checked(x['producer_image_record']);require(im==complete['acquired'][i] and im['sample_index']==i and im['original_path']==x['original_body']['path'],'reused image record identity')
    redirect=checked(d['failed_sample_literal_location_binding']['old_request012_redirect_receipt']);failed=checked(d['failed_sample_literal_location_binding']['old_failed_receipt']);h=transport.transport_headers(redirect['header_pairs'])
    require(redirect['status']=='redirect' and failed['status']=='failed' and failed['status_code']==200 and failed['redirect_from']==12 and failed['url']==h['location']==d['failed_sample_literal_location_binding']['literal_location'],'fixed inherited failed-file S3')
    require(failed['bytes']==2784707 and failed['upper']==65536 and int(transport.transport_headers(failed['header_pairs'])['content-length'])==4193532,'parent partial/unread reserve')
    return d,complete,h['location']


def preflight(plan,parent=True):
    require(plan['schema']=='helmet24_recovery_v1' and plan['new_caps']==CAPS and plan['producer_seconds']==1248 and plan['consumer_seconds']==120 and plan['cleanup_seconds']==16 and plan['models']==0,'fixed new recovery caps')
    require(plan['scope']=={'path':str(R/'scope.json'),'sha256':SCOPE_SHA},'fixed recovery scope')
    scope=checked(plan['scope']);require(plan['selection']==scope['selection'] and plan['selection']['sha256']==SELECTION_SHA,'fixed unchanged selection')
    require(plan['raw_root']==scope['raw_root']==str(W/'data/raw/task_quality/helmet_prospective_recovery_20261102_v1') and plan['delivery_root']==scope['delivery_root']==str(R/'delivery') and plan['storage_roots']==scope['storage_roots'],'fixed4roots')
    require(plan['scope_parent']==scope['parent'],'no substituted ancestry')
    grant=checked(plan['grant']);require(grant['scope_sha256']==SCOPE_SHA and grant['selection_sha256']==SELECTION_SHA and grant['allow_downloads'] is True and grant['allow_literal_inherited_s3_once'] is True and grant['allow_single_s3_redirect'] is True and grant['new_http']==35 and grant['extra_http_exception']==1 and grant['extra_executor_seconds']==400 and grant['models']==0,'explicit recovery grant')
    require(set(plan['source_sha256'])==set(SOURCE_FILES),'exact source set')
    for rel,h in plan['source_sha256'].items():require(sha(regular(W/rel))==h,'recovery source freeze')
    require(plan['runtime']=={'python':sys.executable,'pillow':Image.__version__},'runtime')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['source_commit'],'commit identity');subprocess.run(['git','diff','--quiet'],cwd=W,check=True);subprocess.run(['git','diff','--cached','--quiet'],cwd=W,check=True)
    selection=checked(plan['selection']);require(len(selection['samples'])==24 and selection['selected_counts']=={'Helmet':12,'NoHelmet':12} and not selection['blockers'],'unchanged24')
    return selection,scope,parent_binding(scope)if parent else None


class Open120:
    """One documented override of old helper's60s open argument; no retries."""
    def __init__(self,capture):self.capture=capture;self.delegate=transport.build_opener(transport.NoRedirect())
    def open(self,request,timeout):
        remaining=self.capture.remaining();effective=min(120,remaining);idx=len(self.capture.records)
        save(self.capture.plan,self.capture.root/('open_control_%03d.json'%idx),{'url':request.full_url,'request_index':idx,'clock':clock(),'helper_timeout_argument':timeout,'effective_socket_open_and_read_timeout':effective,'remaining_producer_seconds':remaining,'fixed_max':120,'no_retry':True})
        return self.delegate.open(request,timeout=effective)


class RecoveryCapture(transport.Capture):
    def __init__(self,root,plan,claim,selection,literal):
        self.root,self.plan,self.claim=Path(root),plan,claim;self.caps=dict(CAPS);self.producer_seconds=1248;self.records=[];self.selection=selection;self.literal=literal;self.opener=Open120(self)
    def request(self,url,role,cap,redirect_from=None):
        preflight(self.plan,parent=False)
        require(role.startswith('image_sample_') and int(role.rsplit('_',1)[1])in range(6,24),'fixed new roles only')
        i=int(role[-2:]);sample=self.selection['samples'][i]
        if redirect_from==-1:require(i==6 and not self.records and url==self.literal,'only once inherited S3')
        elif redirect_from is None:require(i>=7 and url==sample['publisher_file']['content_details']['download_url'],'unrequested literal original URL')
        else:
            require(type(redirect_from)is int and 0<=redirect_from<len(self.records),'new redirect index');prev=self.records[redirect_from];h=transport.transport_headers(prev['header_pairs']);require(prev['status']=='redirect' and prev['redirect_from'] is None and prev['role']==role and url==urljoin(prev['url'],h['location']),'one literal new S3 hop')
        return super().request(url,role,cap,redirect_from)


def run(plan,path,pin):
    start,uuid=clock(),boot();selection,scope,binding=preflight(plan);diagnosis,complete,literal=binding
    root=Path(plan['raw_root']);root.mkdir(exist_ok=False);save(plan,root/'plan.json',plan);claim={'schema':'helmet24_recovery_single_launch_v1','plan_sha256':pin,'start_clock':start,'boot_uuid':uuid,'pid':os.getpid(),'parent_plan_sha256':PARENT_PLAN_SHA,'models':0};save(plan,root/'claim.json',claim)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('recovery producer absolute deadline')));signal.setitimer(signal.ITIMER_REAL,max(.001,start+1248-clock()))
    acquired=[];failure=None;cap=None
    try:
        cap=RecoveryCapture(root,plan,claim,selection,literal)
        for i in range(6):
            x=dict(complete['acquired'][i]);x.update(origin='readonly_parent',reuse_provenance=diagnosis['reusable_complete_originals'][i]);save(plan,root/('image_%02d.json'%i),x);acquired.append(x)
        for i in range(6,24):
            s=selection['samples'][i];f=s['publisher_file'];cd=f['content_details'];role='image_sample_%02d'%i;limit=max(65536,f['size']+1)
            index=cap.request(literal,role,limit,redirect_from=-1)if i==6 else cap.asset(cd['download_url'],role,limit,f['size'],cd['sha256_hash'])
            r=cap.records[index];require(r['status']=='complete' and r['bytes']==f['size'] and r['body_sha256']==cd['sha256_hash'],'full original size/SHA')
            crop,g=original.geometry(cap.body(index),s);crop.close();x={'sample_index':i,'sample_id':s['sample_id'],'request_index':index,'original_path':str(cap.body(index)),'image_id':s['image_id'],'annotation_id':s['annotation_id'],'publisher_sha256':cd['sha256_hash'],'geometry':g,'origin':'new'};save(plan,root/('image_%02d.json'%i),x);acquired.append(x)
    except BaseException as e:failure={'type':type(e).__name__,'message':str(e),'clock':clock()}
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    if failure:save(plan,root/'producer_failure.json',failure)
    save(plan,root/'producer_complete.json',{'input_complete':failure is None and len(acquired)==24,'acquired':acquired,'failure':failure,'requests_observed':len(cap.records)if cap else 0,'finished_clock':clock(),'readonly_reused_images':6,'readonly_reuse_new_network_bytes':0,'models':0})
    cmd=[sys.executable,str(W/'scripts/audit_helmet_prospective_recovery.py'),'--plan',str(path),'--plan-sha256',pin];save(plan,root/'consumer_intent.json',{'command':cmd,'clock':clock(),'max_seconds':120})
    child=None;process={}
    def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(4194304,4194304))
    with(root/'consumer.log').open('xb')as log:
        try:
            remaining=start+1384-clock();require(remaining>16,'consumer remaining with cleanup reserve');child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit);code=child.wait(timeout=min(120,remaining-16));process={'pid':child.pid,'returncode':code,'error':None,'finished_clock':clock()}
        except BaseException as e:process={'pid':child.pid if child else None,'returncode':None,'error':str(e),'finished_clock':clock()}
        finally:
            if child:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                except OSError as e:process['cleanup_error']=str(e)
                child.wait(timeout=4)
    save(plan,root/'consumer_process.json',process);out=Path(plan['delivery_root']);audit=out/'audit.json';term=out/'terminal.json'
    result={'producer_success':failure is None,'consumer_success':process['returncode']==0 and process['error'] is None and not process.get('cleanup_error') and audit.is_file()and term.is_file(),'audit_sha256':sha(audit)if audit.is_file()else None,'terminal_sha256':sha(term)if term.is_file()else None,'finished_clock':clock(),'models':0,'storage_bytes_before_launch_result':usage(plan)}
    base=usage(plan);result['storage_bytes_at_finish_including_result']=base
    for _ in range(8):result['storage_bytes_at_finish_including_result']=base+len((json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode())
    save(plan,root/'launch_result.json',result);return 0 if result['producer_success']and result['consumer_success']else 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256');p.add_argument('--create-plan',action='store_true');p.add_argument('--grant');p.add_argument('--grant-sha256');p.add_argument('--source-commit');a=p.parse_args()
    if a.create_plan:
        require(a.grant and a.grant_sha256 and a.source_commit and not a.plan_sha256,'create-plan arguments');scope=checked({'path':str(R/'scope.json'),'sha256':SCOPE_SHA})
        plan={'schema':'helmet24_recovery_v1','new_caps':dict(CAPS),'producer_seconds':1248,'consumer_seconds':120,'cleanup_seconds':16,'scope':{'path':str(R/'scope.json'),'sha256':SCOPE_SHA},'scope_parent':scope['parent'],'selection':scope['selection'],'raw_root':scope['raw_root'],'delivery_root':scope['delivery_root'],'storage_roots':scope['storage_roots'],'grant':{'path':str(Path(a.grant).absolute()),'sha256':a.grant_sha256},'source_commit':a.source_commit,'source_sha256':{x:sha(regular(W/x))for x in SOURCE_FILES},'runtime':{'python':sys.executable,'pillow':Image.__version__},'models':0};preflight(plan);save(plan,a.plan,plan);print(sha(a.plan));return 0
    require(a.plan_sha256 and sha(regular(a.plan))==a.plan_sha256,'plan SHA');plan=load(a.plan)
    try:return run(plan,Path(a.plan),a.plan_sha256)
    except BaseException as e:
        target=R/'launch_control_failure.json';root=Path(plan['raw_root']);claims=list(root.glob('request_*/claim.json'))if root.exists()else[]
        if not target.exists():save(plan,target,{'type':type(e).__name__,'message':str(e),'new_request_claims_observed':len(claims),'new_network_upper_if_unverified':CAPS['body']if claims else 0,'models':0,'no_retry':True})
        return 2


if __name__=='__main__':raise SystemExit(main())
