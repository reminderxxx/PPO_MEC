"""Independent single consumer for the new fixed 24-image package only."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import warnings
from urllib.parse import urlencode, urljoin, urlsplit
from PIL import Image, ImageOps, ImageDraw

W = Path(__file__).resolve().parents[1]
P = Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_helmet_prospective_20261102_v1')
SCOPE_SHA = '06c9b094aa0eaad416c48f875c1251e41440132c52f4816ea9588e7cdaab9baf'
SELECTION_SHA = '388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743'
CAPS = {'http': 48, 'body': 67108864, 'wall': 1500, 'storage': 134217728}
SOURCES = {'scripts/acquire_helmet_prospective.py', 'scripts/audit_helmet_prospective.py', 'scripts/prepare_public_helmet_pilot.py', 'scripts/audit_public_helmet_pilot.py'}
READ = 65536


def need(ok, message):
    if not ok:
        raise ValueError(message)


def clock(): return time.clock_gettime(time.CLOCK_MONOTONIC)


def boot():
    s = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid'], text=True).strip()
    need(re.fullmatch(r'[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}', s), 'boot UUID')
    return s


def regular(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and not any(x.is_symlink() for x in p.parents), 'regular nonsymlink input')
    return p


def sha(path):
    h = hashlib.sha256()
    with regular(path).open('rb') as f:
        for b in iter(lambda: f.read(65536), b''): h.update(b)
    return h.hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    return json.loads(regular(path).read_text(), object_pairs_hook=pairs, parse_constant=lambda _: need(False, 'nonfinite JSON'))


def ref(r):
    need(sha(r['path']) == r['sha256'], 'source reference SHA')
    return load(r['path'])


def storage(roots):
    unique = sorted({Path(p).absolute() for p in roots}, key=lambda p: len(p.parts)); kept=[]; total=0
    for root in unique:
        if any(root.is_relative_to(x) for x in kept): continue
        need(root.exists() and not root.is_symlink(), 'storage root'); kept.append(root)
        for f in root.rglob('*'):
            if f.is_symlink(): total += f.lstat().st_size
            elif f.is_file(): total += f.stat().st_size
    return total


def write(plan, path, data):
    if not isinstance(data, bytes): data=(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    need(storage(plan['storage_roots']) + len(data) + 1048576 <= CAPS['storage'], 'consumer storage reserve')
    with Path(path).open('xb') as f: f.write(data)


def headers(pairs):
    h={}
    for k,v in pairs:
        need(isinstance(k,str) and isinstance(v,str),'header pair strings'); k=k.lower()
        if k in h and k not in ('set-cookie','server-timing'): need(h[k]==v,'conflicting critical response header')
        h[k]=v if k not in h else h[k]+'\n'+v if k in ('set-cookie','server-timing') else v
    return h


def native_inputs(selection):
    src=selection['source_provenance']; coco=ref(src['coco']); byimage={x['id']:x for x in coco['images']}; byann={x['id']:x for x in coco['annotations']}; cats={x['id']:x['name'] for x in coco['categories']}
    need(len(byimage)==2392 and len(byann)==8275,'original COCO complete')
    rows=[]
    for i,p in enumerate(src['metadata_pages']):
        data=ref(p['body']); receipt=ref(p['receipt']); expect='https://data.mendeley.com/public-api/datasets/j82bnw7gsr/files?'+urlencode({'folder_id':'39245ed4-43d6-497c-9fd9-45adca1a6ca0','version':1,'$start':i*1000,'$limit':1000})
        need(receipt['url']==receipt['final_url']==p['exact_url']==expect and receipt['status']=='complete' and receipt['status_code']==[206,206,200][i] and receipt['body_sha256']==p['body']['sha256'],'official metadata identity')
        rh=headers(receipt['header_pairs']);need(rh['content-range']==f'items {i*1000}-{min(i*1000+999,2391)}/2392' and receipt['retained_bytes']==p['body']['bytes']==int(rh['content-length']),'native items range/bytes')
        need(len(data)==(1000 if i<2 else 392),'metadata count'); rows.extend(data)
    need(len(rows)==2392 and len({x['id'] for x in rows})==2392,'full file inventory')
    byfile={x['id']:x for x in rows}; old=ref(src['old_six_manifest']); oldids={x['image_id'] for x in old['samples']}; oldhash={x['image_sha256'] for x in old['samples']}; seen=set(); seenhash=set()
    for i,s in enumerate(selection['samples']):
        a=byann[s['annotation_id']]; im=byimage[s['image_id']]; file=byfile[s['publisher_file']['id']]; cd=file['content_details']; bbox=a['bbox']
        need(s['sample_id']=='prospective_%02d'%i,'fixed selected order')
        need(s['native_annotation']==a and s['native_image']==im and s['publisher_file']==file and a['image_id']==im['id'],'native GT/file exact record')
        need(file['filename']==im['file_name'] and cats[a['category_id']]==s['category_name'] and s['category_name'] in ('Helmet','NoHelmet'),'native class/image association')
        need(file['folder_id']=='39245ed4-43d6-497c-9fd9-45adca1a6ca0' and file['size']==cd['size'] and 0<file['size']<=4194304,'image folder/size')
        need(re.fullmatch('[0-9a-f]{64}',cd['sha256_hash']) and cd['download_url']==f"https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/{file['id']}/file_downloaded",'publisher hash/literal URL')
        need(im['id'] not in oldids|seen and cd['sha256_hash'] not in oldhash|seenhash,'old6/duplicate exclusion'); seen.add(im['id']); seenhash.add(cd['sha256_hash'])
        need(len(bbox)==4 and all(type(v) in (int,float) and math.isfinite(v) and int(v)==v for v in bbox) and 64<=bbox[2]<=256 and 64<=bbox[3]<=256,'native integer geometry')
        need(s['selection_hash']==hashlib.sha256(f"helmet-prospective-20261102-v1|{im['id']}|{a['id']}".encode()).hexdigest(),'selection rank key')
    return rows


def crop_original(path,sample):
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(path) as source:
            need(0<source.width*source.height<=60000000,'source pixel cap'); raw=list(source.size); orient=source.getexif().get(274);need(orient is None or orient in range(9),'orientation');source.verify()
        with Image.open(path) as source: display=ImageOps.exif_transpose(source).convert('RGB');display.load()
    x,y,w,h=map(int,sample['native_annotation']['bbox']);need(x>=0 and y>=0 and w>0 and h>0 and x+w<=display.width and y+h<=display.height,'display ROI bounds')
    crop=display.crop((x,y,x+w,y+h));crop.info.clear();return crop,{'raw_size':raw,'exif_orientation':orient,'display_size':list(display.size),'bbox_xywh':sample['native_annotation']['bbox'],'crop_size':list(crop.size),'crop_rgb_sha256':hashlib.sha256(crop.tobytes()).hexdigest()}


def bill_prefix(root):
    receipts=[]; claims=[]; uncertain=False; actual_retained=0; fingerprints=[]
    for d in sorted(root.glob('request_*')):
        if not d.is_dir(): continue
        claim=d/'claim.json'; receipt=d/'receipt.json';body=d/'body.bin'
        if body.exists(): actual_retained += regular(body).stat().st_size
        if claim.exists():claims.append(str(claim))
        if receipt.exists():
            r=load(receipt);need(type(r.get('upper')) is int and 0<=r['upper']<=65536,'receipt read upper bound');receipts.append((d,r))
            fingerprints.append({'directory':str(d),'claim_sha256':sha(claim) if claim.exists() else None,'receipt_sha256':sha(receipt),'body_sha256':sha(body) if body.exists() else None})
        else: uncertain=True
    return receipts,{'request_claims':len(claims),'receipts':len(receipts),'retained_body_bytes':actual_retained,'request_fingerprints':fingerprints,'interrupted_read_upper_bound':sum(r['upper'] for _,r in receipts),'missing_receipt_unknown_upper':CAPS['body'] if uncertain else 0,'missing_receipt':uncertain,'models':0}


def audit(plan, plan_sha):
    root=Path(plan['raw_root']);out=Path(plan['delivery_root']);out.mkdir(exist_ok=False);failure=None;bill={};outputs=[]
    try:
        need(plan['models']==0 and plan['schema']=='helmet24_acquisition_v1' and plan['new_caps']==CAPS and plan['producer_seconds']==1380 and plan['consumer_seconds']==120,'fixed plan')
        need(plan['scope']=={'path':str(P/'acquisition_scope.json'),'sha256':SCOPE_SHA} and plan['selection']=={'path':str(P/'selection.json'),'sha256':SELECTION_SHA},'fixed source pins')
        scope=ref(plan['scope']);selection=ref(plan['selection']);need(plan['raw_root']==scope['raw_root']==str(W/'data/raw/task_quality/helmet_prospective_20261102_v1') and plan['storage_roots']==scope['storage_roots'] and plan['delivery_root']==str(P/'acquisition_delivery'),'fixed roots')
        need(sha(root/'plan.json')==plan_sha,'retained plan SHA');claim=load(root/'claim.json');need(claim['plan_sha256']==plan_sha and claim['boot_uuid']==boot(),'claim/boot')
        need(clock()-claim['start_clock']<=1500,'total absolute deadline');signal.setitimer(signal.ITIMER_REAL,max(.001,min(120,claim['start_clock']+1500-clock())))
        grant=ref(plan['grant']);need(grant['scope_sha256']==SCOPE_SHA and grant['selection_sha256']==SELECTION_SHA and grant['allow_downloads'] is True and grant['allow_single_s3_redirect'] is True and grant['models']==0,'grant')
        need(set(plan['source_sha256'])==SOURCES and all(sha(W/k)==v for k,v in plan['source_sha256'].items()),'source freeze')
        need(plan['runtime']=={'python':sys.executable,'pillow':Image.__version__},'runtime')
        need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['source_commit'],'source commit identity')
        need(len(selection['samples'])==24 and selection['selected_counts']=={'Helmet':12,'NoHelmet':12} and not selection['blockers'],'all24 frozen selection');native_inputs(selection)
        receipts,bill=bill_prefix(root);need(not bill['missing_receipt'],'all request receipts retained');need(bill['request_claims']==len(receipts)<=48,'HTTP claims count')
        seenurls=set()
        for number,(d,r) in enumerate(receipts):
            need(d.name=='request_%03d'%number and r['url'] not in seenurls,'contiguous/no retry');seenurls.add(r['url']);c=load(d/'claim.json')
            need(all(c[k]==r[k] for k in ('url','role','redirect_from','started')),'request claim identity');need(claim['start_clock']<=r['started']<=r['finished']<=claim['start_clock']+1380,'producer request clock')
            hp=r['header_pairs'];need(dict(hp)==r['headers'],'full retained header pairs');h=headers(hp);need(r['final_url']==r['url'],'no hidden redirect')
            body=d/'body.bin';n=body.stat().st_size if body.exists() else 0;need(n==r['retained_bytes']==r['bytes'] and 0<=r['upper']<=65536,'body accounting')
            need(r['body_sha256']==(sha(body) if body.exists() else None),'actual original hash');need(h.get('content-encoding','identity').lower().strip() in ('','identity'),'identity encoding')
            role=r['role'];need(re.fullmatch(r'image_sample_\d{2}',role),'fixed role');i=int(role[-2:]);need(i<24,'sample role range');expected=selection['samples'][i]['publisher_file']['content_details']['download_url']
            if r['redirect_from'] is None:need(r['url']==expected,'literal fixed initial URL')
            else:
                j=r['redirect_from'];need(type(j)is int and 0<=j<number,'redirect predecessor');prev=receipts[j][1];ph=headers(prev['header_pairs']);u=urlsplit(r['url'])
                need(prev['status']=='redirect' and prev['redirect_from'] is None and prev['role']==role and r['url']==urljoin(prev['url'],ph['location']),'literal one-hop location')
                need(u.scheme=='https' and not u.username and not u.password and not u.fragment and u.port in (None,443) and re.fullmatch(r'prod-dcd-datasets-public-files-[a-z0-9-]+\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com',u.hostname or ''),'official public S3')
            if r['status']=='complete':
                need(r['status_code']==200 and r['upper']==0 and (h.get('content-type','').split(';')[0].lower().startswith('image/') or h.get('content-type','').split(';')[0].lower()=='application/octet-stream'),'complete image HTTP/MIME')
                if 'content-length'in h:need(h['content-length'].isdecimal() and int(h['content-length'])==n,'complete length')
            elif r['status']=='redirect':need(r['status_code'] in (302,303,307,308) and r['redirect_from'] is None and 'location'in h and r['upper']==0,'one redirect receipt')
            else:need(r['status']=='failed' and number==len(receipts)-1,'failed prefix stops')
        need(bill['retained_body_bytes']+bill['interrupted_read_upper_bound']<=CAPS['body'],'network byte cap')
        pc=load(root/'producer_complete.json');need(pc['input_complete'] is True and pc['failure'] is None and not(root/'producer_failure.json').exists() and len(pc['acquired'])==24,'producer complete all24')
        need(pc['requests_observed']==len(receipts) and claim['start_clock']<=pc['finished_clock']<=claim['start_clock']+1380,'producer terminal accounting')
        inputdir=out/'opaque_inputs';inputdir.mkdir();manifest=[];gt=[];thumbs=[];usedrequests=set()
        for i,s in enumerate(selection['samples']):
            rec=load(root/('image_%02d.json'%i));need(rec==pc['acquired'][i] and rec['sample_index']==i and rec['image_id']==s['image_id'] and rec['annotation_id']==s['annotation_id'] and rec['sample_id']==s['sample_id'],'ordered image record')
            idx=rec['request_index'];need(type(idx)is int and 0<=idx<len(receipts) and idx not in usedrequests,'unique final receipt');usedrequests.add(idx);d,r=receipts[idx];file=s['publisher_file'];need(r['status']=='complete' and r['role']=='image_sample_%02d'%i and r['bytes']==file['size'] and r['body_sha256']==file['content_details']['sha256_hash']==rec['publisher_sha256'],'publisher identity/size/hash')
            original=d/'body.bin';need(rec['original_path']==str(original),'original path');crop,g=crop_original(original,s);need(g==rec['geometry'],'independent EXIF/native crop hash')
            buf=io.BytesIO();crop.save(buf,format='PNG');png=buf.getvalue();name='q%02d.png'%i;write(plan,inputdir/name,png)
            manifest.append({'sample_id':s['sample_id'],'source_path':str(original),'source_sha256':file['content_details']['sha256_hash'],'source_bytes':file['size'],'bbox_xywh':list(map(int,s['native_annotation']['bbox'])),'orientation':g['exif_orientation'],'path':str(inputdir/name),'sha256':hashlib.sha256(png).hexdigest(),'rgb_sha256':g['crop_rgb_sha256'],'size':g['crop_size'],'display_size':g['display_size'],'raw_size':g['raw_size'],'category_name':s['category_name'],'image_id':s['image_id'],'annotation_id':s['annotation_id'],'request_index':idx})
            gt.append({'input_id':'q%02d'%i,'category_name':s['category_name'],'image_id':s['image_id'],'annotation_id':s['annotation_id'],'publisher_file_id':file['id'],'publisher_image_sha256':file['content_details']['sha256_hash'],'original_path':str(original),'geometry':g,'native_annotation':s['native_annotation'],'native_image':s['native_image'],'selection_sample_id':s['sample_id']})
            crop.thumbnail((128,128));thumbs.append(crop)
        usedall=set()
        for idx in usedrequests:
            usedall.add(idx);j=receipts[idx][1]['redirect_from']
            if j is not None:usedall.add(j)
        need(usedall==set(range(len(receipts))),'no unassigned HTTP requests')
        sheet=Image.new('RGB',(6*144,4*160),'white');draw=ImageDraw.Draw(sheet)
        for i,t in enumerate(thumbs):x=(i%6)*144;y=(i//6)*160;sheet.paste(t,(x,y));draw.text((x,y+132),'q%02d'%i,fill='black')
        b=io.BytesIO();sheet.save(b,format='PNG');write(plan,out/'contact_sheet.png',b.getvalue())
        write(plan,out/'input_manifest.json',{'schema':'helmet24_display_inputs_v1','selection_sha256':SELECTION_SHA,'samples':manifest})
        write(plan,out/'groundtruth.json',{'selection_sha256':SELECTION_SHA,'scope':'native COCO GT, not model labels','samples':gt})
        text=io.StringIO();writer=csv.writer(text);writer.writerow(['input_id','category_name','image_id','annotation_id','publisher_sha256'])
        for r in gt:writer.writerow([r['input_id'],r['category_name'],r['image_id'],r['annotation_id'],r['publisher_image_sha256']])
        write(plan,out/'groundtruth.csv',text.getvalue().encode())
        outputs=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file()]
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
        if not bill:
            try:_,bill=bill_prefix(root)
            except BaseException:bill={'unknown_network_upper':CAPS['body'],'models':0,'billing_integrity':'UNVERIFIABLE'}
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    outputs=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file()]
    result={'status':'HELMET24_INPUT_QUALIFIED' if failure is None else 'FAILED_INPUT_QUALIFICATION','audit_success':failure is None,'input_qualified':failure is None,'plan_sha256':plan_sha,'selection_sha256':SELECTION_SHA,'failure':failure,'bill':bill,'output_files':{row['path']:row['sha256'] for row in outputs},'output_file_records':outputs,'finished_clock':clock(),'storage_bytes_before_audit':storage(plan['storage_roots']),'models':0,'limits':['Metadata-selected and label-consumed; training overlap unknown; no hidden, generalization, safety or system benefit claim.','All failed historical runs remain immutable; this new scope is billed separately.']}
    write(plan,out/'audit.json',result);write(plan,out/'terminal.json',{'audit':str(out/'audit.json'),'audit_sha256':sha(out/'audit.json'),'sha256':sha(out/'audit.json'),'audit_success':failure is None,'input_qualified':failure is None,'finished_clock':clock()})
    return 0 if failure is None else 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args();need(sha(a.plan)==a.plan_sha256,'plan SHA');plan=load(a.plan)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('independent consumer deadline')))
    return audit(plan,a.plan_sha256)


if __name__=='__main__':raise SystemExit(main())
