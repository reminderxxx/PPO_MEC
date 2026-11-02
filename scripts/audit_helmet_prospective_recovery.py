"""Independent new all24 consumer; never calls a previous audit entry point."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
from urllib.parse import urljoin, urlsplit
from PIL import Image,ImageDraw
import audit_helmet_prospective as native

W=Path(__file__).resolve().parents[1]
P=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_helmet_prospective_20261102_v1')
R=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_helmet_prospective_recovery_20261102_v1')
SCOPE_SHA='72b79e7c923898569fc00e04f48fb344a4568380302ddbf376b016924b24eef0'
SELECTION_SHA='388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743'
PARENT_PLAN_SHA='e299002e7d19af6c1518dc034e2f1250b65d1e517f57863b89edc27afb8cf282'
PARENT_COMMIT='2ea28af81d2d6aa3491cb317dc339670d5a5753b'
PARENT_AUDIT_SHA='f5fddd95deffa6f8069bac171115ff30c1e8251e94c5deeeaeb95677e6c01da6'
CAPS={'http':35,'body':60989703,'wall':1384,'storage':134217728}
SOURCES={'scripts/recover_helmet_prospective.py','scripts/audit_helmet_prospective_recovery.py','scripts/acquire_helmet_prospective.py','scripts/audit_helmet_prospective.py','scripts/prepare_public_helmet_pilot.py','scripts/audit_public_helmet_pilot.py'}
need,sha,load,ref,clock,boot,headers=native.need,native.sha,native.load,native.ref,native.clock,native.boot,native.headers


def save(plan,path,data):
    if not isinstance(data,bytes):data=(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    need(native.storage(plan['storage_roots'])+len(data)+1048576<=134217728,'shared4-root storage budget')
    with Path(path).open('xb')as f:f.write(data)


def parent_facts(pp,a,t,outer):
    need(pp['schema']=='helmet24_acquisition_v1' and pp['source_commit']==PARENT_COMMIT and not any(k in pp for k in ('parent','recovery','zero_http_predecessor')),'single original parent schema, no recursion')
    need(pp['new_caps']=={'http':48,'body':67108864,'wall':1500,'storage':134217728},'unchanged original caps')
    need(a['plan_sha256']==PARENT_PLAN_SHA and a['input_qualified'] is False and t['sha256']==t['audit_sha256']==PARENT_AUDIT_SHA and t['input_qualified'] is False,'parent failed audit/terminal binding')
    b=a['bill'];need(b['request_claims']==b['receipts']==14 and b['retained_body_bytes']==4644800 and b['interrupted_read_upper_bound']==65536 and b['missing_receipt'] is False,'full original cost')
    need(outer['returncode']==2 and outer['elapsed_seconds']==514.709546917,'full old executor cost')
    need(4644800+65536+1408825+60989703==67108864 and 14+35==49 and 514.709546917+1384<1900,'explicit cumulative caps')


def old_binding(scope,selection):
    refs=scope['parent'];pp=ref(refs['plan']);a=ref(refs['audit']);t=ref(refs['terminal']);outer=ref(refs['parent_process']);d=ref(refs['diagnosis']);parent_facts(pp,a,t,outer)
    need(refs['plan']['sha256']==PARENT_PLAN_SHA and refs['audit']['sha256']==PARENT_AUDIT_SHA,'fixed parent pins')
    oldroot=W/'data/raw/task_quality/helmet_prospective_20261102_v1';need(pp['raw_root']==str(oldroot) and sha(oldroot/'plan.json')==PARENT_PLAN_SHA,'old fixed plan/root')
    claim=ref(d['provenance']['execution_claim']);pc=ref(d['provenance']['producer_completion']);fail=ref(d['provenance']['producer_failure']);need(claim['plan_sha256']==PARENT_PLAN_SHA and pc['input_complete'] is False and len(pc['acquired'])==6 and fail['type']=='timeout','sealed original failure')
    need(len(d['all14_request_inventory'])==14 and len(d['reusable_complete_originals'])==6,'parent14/readonly6')
    allr=[]
    for i,item in enumerate(d['all14_request_inventory']):
        rr=ref(item['receipt']);cc=ref(item['claim']);body=Path(item['body']['path']);need(str(body)==str(oldroot/f'request_{i:03d}'/'body.bin') and str(Path(item['receipt']['path']))==str(oldroot/f'request_{i:03d}'/'receipt.json') and str(Path(item['claim']['path']))==str(oldroot/f'request_{i:03d}'/'claim.json'),'old fixed request paths')
        need(sha(body)==item['body']['sha256']==rr['body_sha256'] and body.stat().st_size==rr['retained_bytes']==rr['bytes'],'old body retained fingerprint')
        need(item['receipt']['sha256']==a['bill']['request_fingerprints'][i]['receipt_sha256'] and item['claim']['sha256']==a['bill']['request_fingerprints'][i]['claim_sha256'],'old audit fingerprint vector');need(all(rr[k]==cc[k]for k in ('url','role','redirect_from','started')),'old intent matching');allr.append(rr)
    reused=[]
    for i,item in enumerate(d['reusable_complete_originals']):
        im=ref(item['producer_image_record']);s=selection['samples'][i];f=s['publisher_file'];idx=2*i+1;rr=allr[idx];prev=allr[idx-1]
        need(im==pc['acquired'][i] and im['sample_index']==i and im['sample_id']==s['sample_id'] and im['image_id']==s['image_id'] and im['annotation_id']==s['annotation_id'] and im['request_index']==idx,'readonly original record')
        need(im['original_path']==item['original_body']['path']==str(oldroot/f'request_{idx:03d}'/'body.bin') and rr['status']=='complete' and rr['status_code']==200 and rr['upper']==0 and rr['bytes']==f['size'] and rr['body_sha256']==f['content_details']['sha256_hash']==im['publisher_sha256'],'reused complete publisher identity')
        need(prev['status']=='redirect' and prev['url']==f['content_details']['download_url'] and rr['redirect_from']==idx-1 and headers(prev['header_pairs'])['location']==rr['url'],'reused original hop provenance');reused.append(im)
    h=headers(allr[12]['header_pairs']);literal=h['location'];f=allr[13]
    need(allr[12]['status']=='redirect' and allr[12]['url']==selection['samples'][6]['publisher_file']['content_details']['download_url'] and f['redirect_from']==12 and f['status']=='failed' and f['status_code']==200 and f['url']==literal==d['failed_sample_literal_location_binding']['literal_location'],'inherited single failed S3 URL')
    need(f['bytes']==2784707 and f['upper']==65536 and int(headers(f['header_pairs'])['content-length'])==4193532,'partial preserved and reserve unchanged')
    return d,reused,literal


def new_prefix(root):
    receipts,bill=native.bill_prefix(root)
    if bill.get('missing_receipt'):bill['missing_receipt_unknown_upper']=60989703
    return receipts,bill


def audit(plan,pin):
    root=Path(plan['raw_root']);out=Path(plan['delivery_root']);out.mkdir(exist_ok=False);failure=None;bill={};claim=None
    try:
        need(plan['schema']=='helmet24_recovery_v1' and plan['new_caps']==CAPS and plan['producer_seconds']==1248 and plan['consumer_seconds']==120 and plan['cleanup_seconds']==16 and plan['models']==0,'exact recovery caps/schema')
        need(plan['scope']=={'path':str(R/'scope.json'),'sha256':SCOPE_SHA},'fixed scope');scope=ref(plan['scope']);need(plan['scope_parent']==scope['parent'],'fixed parent references')
        need(plan['selection']==scope['selection']=={'path':str(P/'selection.json'),'sha256':SELECTION_SHA},'fixed24 selection pin');selection=ref(plan['selection'])
        need(plan['raw_root']==scope['raw_root']==str(W/'data/raw/task_quality/helmet_prospective_recovery_20261102_v1') and plan['delivery_root']==scope['delivery_root']==str(R/'delivery') and plan['storage_roots']==scope['storage_roots'],'exact4roots')
        need(sha(root/'plan.json')==pin,'retained recovery plan');claim=load(root/'claim.json');need(claim['plan_sha256']==pin and claim['boot_uuid']==boot() and claim['parent_plan_sha256']==PARENT_PLAN_SHA,'new claim/boot')
        need(clock()-claim['start_clock']<1384,'new absolute1384s');signal.setitimer(signal.ITIMER_REAL,max(.001,min(120,claim['start_clock']+1384-clock())))
        g=ref(plan['grant']);need(g['scope_sha256']==SCOPE_SHA and g['selection_sha256']==SELECTION_SHA and g['allow_downloads'] is True and g['allow_literal_inherited_s3_once'] is True and g['allow_single_s3_redirect'] is True and g['new_http']==35 and g['extra_http_exception']==1 and g['extra_executor_seconds']==400 and g['models']==0,'explicit new grant')
        need(set(plan['source_sha256'])==SOURCES and all(sha(W/k)==v for k,v in plan['source_sha256'].items()),'frozen source set');need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['source_commit'],'source commit');need(plan['runtime']=={'python':sys.executable,'pillow':Image.__version__},'runtime')
        need(len(selection['samples'])==24 and selection['selected_counts']=={'Helmet':12,'NoHelmet':12} and not selection['blockers'],'unchanged24');native.native_inputs(selection)
        diagnosis,reused,literal=old_binding(scope,selection)
        receipts,bill=new_prefix(root);need(not bill['missing_receipt'] and bill['request_claims']==len(receipts)<=35,'new request claims/receipts cap')
        seen=set()
        for i,(directory,r)in enumerate(receipts):
            need(directory.name=='request_%03d'%i and r['url']not in seen,'new contiguous/no retry');seen.add(r['url']);intent=load(directory/'claim.json');need(all(intent[k]==r[k]for k in ('url','role','redirect_from','started')),'request intent match')
            need(claim['start_clock']<=r['started']<=r['finished']<=claim['start_clock']+1248,'new request clock');hp=r['header_pairs'];need(dict(hp)==r['headers'],'all raw headers');h=headers(hp);need(r['final_url']==r['url'] and h.get('content-encoding','identity').strip().lower()in('','identity'),'no hidden redirect/encoding')
            body=directory/'body.bin';size=body.stat().st_size if body.exists()else 0;need(size==r['bytes']==r['retained_bytes'] and r['body_sha256']==(sha(body)if body.exists()else None)and type(r['upper'])is int and 0<=r['upper']<=65536,'actual new retained bytes/upper')
            need(re.fullmatch(r'image_sample_(?:0[6-9]|1[0-9]|2[0-3])',r['role']),'fixed new18 sample role');idx=int(r['role'][-2:]);s=selection['samples'][idx];rd=r['redirect_from']
            if rd==-1:need(i==0 and idx==6 and r['url']==literal,'only one inherited failed-file full S3 GET')
            elif rd is None:need(idx>=7 and r['url']==s['publisher_file']['content_details']['download_url'],'new literal original URL')
            else:
                need(type(rd)is int and 0<=rd<i,'new local hop predecessor');prev=receipts[rd][1];ph=headers(prev['header_pairs']);u=urlsplit(r['url']);need(prev['status']=='redirect' and prev['redirect_from'] is None and prev['role']==r['role'] and r['url']==urljoin(prev['url'],ph['location']),'new single literal S3 hop');need(u.scheme=='https'and not u.username and not u.password and not u.fragment and u.port in(None,443)and re.fullmatch(r'prod-dcd-datasets-public-files-[a-z0-9-]+\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com',u.hostname or ''),'official S3 host')
            control=load(root/('open_control_%03d.json'%i));need(control['request_index']==i and control['url']==r['url'] and control['fixed_max']==120 and control['no_retry'] is True and 0<control['effective_socket_open_and_read_timeout']<=120 and control['effective_socket_open_and_read_timeout']==min(120,control['remaining_producer_seconds']) and r['started']<=control['clock']<=r['finished'],'actual120s open configuration')
            if r['status']=='complete':
                mime=h.get('content-type','').split(';')[0].strip().lower();need(r['status_code']==200 and r['upper']==0 and(mime.startswith('image/')or mime=='application/octet-stream'),'complete HTTP/MIME')
                if'content-length'in h:need(h['content-length'].isdecimal()and int(h['content-length'])==size,'complete Content-Length')
            elif r['status']=='redirect':need(r['status_code']in(302,303,307,308)and rd is None and'location'in h and r['upper']==0,'one new original Location')
            else:need(r['status']=='failed'and i==len(receipts)-1,'failed new prefix stops')
        need(bill['retained_body_bytes']+bill['interrupted_read_upper_bound']<=60989703,'new body cap');need(4644800+65536+1408825+bill['retained_body_bytes']+bill['interrupted_read_upper_bound']<=67108864 and 14+len(receipts)<=49,'cumulative original body and explicitHTTP49')
        pc=load(root/'producer_complete.json');need(pc['input_complete'] is True and pc['failure']is None and not(root/'producer_failure.json').exists()and len(pc['acquired'])==24 and pc['requests_observed']==len(receipts),'all24 complete');need(claim['start_clock']<=pc['finished_clock']<=claim['start_clock']+1248,'new producer clock')
        inputs=out/'opaque_inputs';inputs.mkdir();manifest=[];gt=[];thumbs=[];used=set()
        for i,s in enumerate(selection['samples']):
            im=load(root/('image_%02d.json'%i));need(im==pc['acquired'][i]and im['sample_index']==i and im['sample_id']==s['sample_id']and im['image_id']==s['image_id']and im['annotation_id']==s['annotation_id'],'ordered input record');f=s['publisher_file'];idx=im['request_index']
            if i<6:
                need(im['origin']=='readonly_parent'and im['reuse_provenance']==diagnosis['reusable_complete_originals'][i],'readonly reuse proof');old=dict(im);old.pop('origin');old.pop('reuse_provenance');need(old==reused[i],'exact readonly old producer record');original=Path(im['original_path'])
            else:
                need(im['origin']=='new'and type(idx)is int and 0<=idx<len(receipts)and idx not in used,'unique new final original');used.add(idx);directory,rr=receipts[idx];original=directory/'body.bin';need(im['original_path']==str(original)and rr['status']=='complete'and rr['role']=='image_sample_%02d'%i and rr['bytes']==f['size']and rr['body_sha256']==f['content_details']['sha256_hash']==im['publisher_sha256'],'new full original identity')
            need(sha(original)==f['content_details']['sha256_hash']and original.stat().st_size==f['size'],'all24publisheroriginal');crop,geometry=native.crop_original(original,s);need(geometry==im['geometry'],'independent all24 EXIF/native pixels')
            buf=io.BytesIO();crop.save(buf,format='PNG');png=buf.getvalue();path=inputs/('q%02d.png'%i);save(plan,path,png)
            manifest.append({'sample_id':s['sample_id'],'source_path':str(original),'source_sha256':f['content_details']['sha256_hash'],'source_bytes':f['size'],'bbox_xywh':list(map(int,s['native_annotation']['bbox'])),'orientation':geometry['exif_orientation'],'path':str(path),'sha256':hashlib.sha256(png).hexdigest(),'rgb_sha256':geometry['crop_rgb_sha256'],'size':geometry['crop_size'],'display_size':geometry['display_size'],'raw_size':geometry['raw_size'],'category_name':s['category_name'],'image_id':s['image_id'],'annotation_id':s['annotation_id'],'request_index':idx})
            gt.append({'sample_id':s['sample_id'],'category_name':s['category_name'],'source_path':str(original),'source_sha256':f['content_details']['sha256_hash'],'native_annotation':s['native_annotation'],'native_image':s['native_image'],'geometry':geometry,'source_origin':im['origin']});crop.thumbnail((128,128));thumbs.append(crop)
        assigned=set()
        for idx in used:
            assigned.add(idx);rd=receipts[idx][1]['redirect_from']
            if type(rd)is int and rd>=0:assigned.add(rd)
        need(assigned==set(range(len(receipts))),'all new HTTP assigned, inherited hop not recounted')
        save(plan,out/'input_manifest.json',{'schema':'helmet24_display_inputs_v1','selection_sha256':SELECTION_SHA,'samples':manifest});save(plan,out/'groundtruth.json',{'selection_sha256':SELECTION_SHA,'scope':'original native COCO GT, no model relabelling','samples':gt})
        text=io.StringIO();writer=csv.writer(text);writer.writerow(['sample_id','category_name','image_id','annotation_id','publisher_sha256'])
        for s in manifest:writer.writerow([s['sample_id'],s['category_name'],s['image_id'],s['annotation_id'],s['source_sha256']])
        save(plan,out/'groundtruth.csv',text.getvalue().encode());sheet=Image.new('RGB',(864,640),'white');draw=ImageDraw.Draw(sheet)
        for i,t in enumerate(thumbs):x=(i%6)*144;y=(i//6)*160;sheet.paste(t,(x,y));draw.text((x,y+132),'q%02d'%i,fill='black')
        buf=io.BytesIO();sheet.save(buf,format='PNG');save(plan,out/'contact_sheet.png',buf.getvalue())
    except BaseException as e:
        failure={'type':type(e).__name__,'message':str(e)}
        if not bill:
            try:_,bill=new_prefix(root)
            except BaseException:bill={'unknown_new_network_upper':60989703,'models':0,'billing_integrity':'UNVERIFIABLE'}
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    records=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(out.rglob('*'))if p.is_file()];elapsed=clock()-claim['start_clock']if claim else None
    result={'schema':'helmet24_recovery_audit_v1','status':'HELMET24_INPUT_QUALIFIED'if failure is None else'FAILED_INPUT_QUALIFICATION','audit_success':failure is None,'input_qualified':failure is None,'plan_sha256':pin,'selection_sha256':SELECTION_SHA,'failure':failure,'bill':bill,'cumulative_bill':{'old_http':14,'old_retained_body':4644800,'old_read_upper':65536,'old_encoded_read_charge_unchanged':4710336,'additional_declared_unread_reserve':1408825,'additional_reserve_may_overlap_not_exact_wire':True,'new_http':bill.get('request_claims'),'new_retained_body':bill.get('retained_body_bytes'),'new_read_upper':bill.get('interrupted_read_upper_bound'),'old_executor_actual_seconds':514.709546917,'new_executor_so_far_seconds':elapsed,'explicit_extra_http':1,'explicit_extra_executor_seconds':400,'max_new_body':60989703,'max_new_executor_seconds':1384,'max_joint_executor_seconds':1900},'readonly_reused_originals':6,'readonly_reuse_new_network_bytes':0,'output_files':{x['path']:x['sha256']for x in records},'output_file_records':records,'finished_clock':clock(),'storage_bytes_before_audit':native.storage(plan['storage_roots']),'models':0,'limits':['No task inference/hidden/training independence/quality/system-benefit evidence.','Parent failed partial and audit unchanged; all24 selection and model96 rules unchanged.','Whole process including waits and diagnosis is not bounded by the sum of executor budgets.']}
    save(plan,out/'audit.json',result);save(plan,out/'terminal.json',{'audit':str(out/'audit.json'),'sha256':sha(out/'audit.json'),'audit_sha256':sha(out/'audit.json'),'audit_success':failure is None,'input_qualified':failure is None,'finished_clock':clock()});return 0 if failure is None else 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args();need(sha(a.plan)==a.plan_sha256,'plan pin');plan=load(a.plan);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('new consumer absolute deadline')));return audit(plan,a.plan_sha256)


if __name__=='__main__':raise SystemExit(main())
