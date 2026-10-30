"""Fetch only six frozen files; reuse all prior COCO, metadata and selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
from urllib.parse import urljoin, urlsplit

from prepare_public_helmet_pilot import Capture, allowed, boot, clock, load, require, save, sha

W = Path(__file__).resolve().parents[1]
CAPS = {'http': 13, 'body': 60826959, 'wall': 300, 'storage': 268435456}
PRIOR = {'http': 19, 'body': 5526965, 'read_upper': 65536, 'declared_unread_upper': 689404}


def check(plan, expected, grant):
    require(plan['new_caps'] == CAPS and plan['prior'] == PRIOR and plan['producer_seconds'] == 240,
            'fixed final costs and caps')
    require(grant['plan_sha256'] == expected and grant['allow_six_image_downloads'] is True and grant['allow_one_publisher_location_hop'] is True and
            type(grant['model_calls']) is int and grant['model_calls'] == 0, 'exact zero-model input grant')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip() == plan['source_commit'] and
            not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True), 'clean frozen source')
    for name, digest in plan['source_sha'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and not (W/relative).is_symlink() and
                sha(W/relative) == digest, 'source file pin')
    ref = plan['inventory_reference']; old = Path(ref['root'])
    for filename, field in [('plan.json','plan_sha'),('claim.json','claim_sha'),('selection.json','selection_sha'),('selection_claim.json','selection_claim_sha')]:
        require(not (old/filename).is_symlink() and sha(old/filename) == ref[field], 'frozen input inventory pin')
    previous = load(old/'plan.json')
    require(previous['source_commit'] == '23e3c55ef17f143a9c48ed33cbb0bf4a1dc74a8a' and previous['raw_root'] == str(old), 'only fixed completed inventory')
    require(sha(ref['audit_path']) == ref['audit_sha'] and load(ref['audit_path'])['input_qualified'] is False, 'prior failed audit retained')
    for i in range(4):
        folder = old/('request_%03d'%i)
        require(sha(folder/'receipt.json') == ref['receipt_sha'][i], 'inventory receipt pin')
        r = load(folder/'receipt.json')
        body = folder/'body.bin'
        if i < 3:
            require(not body.is_symlink() and body.stat().st_size == r['retained_bytes'] == r['bytes'] and sha(body) == r['body_sha256'], 'retained complete inventory original')
        else:
            require(not body.exists() and r['bytes'] == r['retained_bytes'] == 0 and r['body_sha256'] is None and r['upper'] == 65536, 'failed historical redirect has no body')
    a = plan['annotation_reference']
    require(not Path(a['path']).is_symlink() and sha(a['path']) == a['sha256'] == '5e77fea3d8f1cd93cac42d45065456eeb97a54c31b7534ef453051c9d8853c91' and
            Path(a['path']).stat().st_size == a['size'] == 2065913, 'complete annotation reference')
    targets = load(old/'selection.json')['selected']
    require(len(targets) == 6 and [t['category_name'] for t in targets] == ['Helmet']*3 + ['NoHelmet']*3, 'same six frozen targets')
    d = plan['first_redirect_reference']
    require(Path(d['receipt_path']).resolve() == (old/'request_003/receipt.json').resolve() and sha(d['receipt_path']) == d['sha256'], 'first historical redirect')
    prior = load(d['receipt_path'])
    require(prior['url'] == targets[0]['image_metadata']['content_details']['download_url'] and prior['status_code'] == 302 and
            prior['bytes'] == 0 and prior['upper'] == 65536 and prior['status'] == 'failed', 'no prior first-image body transfer')
    locations = [v for k,v in prior['header_pairs'] if k.lower() == 'location']
    require(len(locations) == 1, 'one original literal Location')
    target = urljoin(prior['url'], locations[0]); allowed(target)
    require(urlsplit(target).hostname != 'data.mendeley.com', 'publicS3 historical target')
    return targets, target


def run(plan_path, expected, grant_path):
    require(sha(plan_path) == expected, 'plan bytes SHA')
    plan, grant = load(plan_path), load(grant_path)
    targets, first_url = check(plan, expected, grant)
    root, audit_path = Path(plan['raw_root']), Path(plan['audit_output'])
    root.mkdir(parents=True,exist_ok=False)
    for name, source in [('plan.json',plan_path),('grant.json',grant_path)]:
        with (root/name).open('xb') as stream:stream.write(Path(source).read_bytes())
    claim = {'plan_sha':expected,'grant_sha256':sha(grant_path),'boot_uuid':boot(),'start_clock':clock(),
             'started_at':datetime.now(timezone.utc).isoformat(),'new_separate_scope_seconds':300,'model_calls':0}
    save(root/'claim.json',claim)
    with (root/'selection.json').open('xb') as stream:
        stream.write((Path(plan['inventory_reference']['root'])/'selection.json').read_bytes())
    capture = Capture(root,plan,claim)
    old_signal = signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('240s final file producer cap')))
    signal.setitimer(signal.ITIMER_REAL,238)
    try:
        selected = []
        for number, sample in enumerate(targets):
            meta = sample['image_metadata']; detail = meta['content_details']; role = 'image_sample_%d'%number
            if number == 0:
                index = capture.request(first_url,role,8*1024*1024,redirect_from=-1)
                r = capture.records[index]
                require(r['status']=='complete' and r['bytes']==meta['size'] and r['body_sha256']==detail['sha256_hash'], 'first exact publisher image')
            else:
                index = capture.asset(detail['download_url'],role,8*1024*1024,meta['size'],detail['sha256_hash'])
            selected.append(dict(sample,request_index=index,opaque_input_path=str(capture.body(index))))
        save(root/'index.json',{'selected':selected})
        capture.remaining()
        save(root/'producer_complete.json',{'requests':len(capture.records),'finished_clock':clock(),'models':0})
    except BaseException as error:
        save(root/'producer_failure.json',{'error_type':type(error).__name__,'error':str(error),'requests':len(capture.records),'finished_clock':clock(),'no_retry':True})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_signal)
    remaining = CAPS['wall']-(clock()-claim['start_clock'])
    require(remaining > 0,'consumer shared deadline')
    try:
        result = subprocess.run([sys.executable,str(W/'scripts/audit_public_helmet_delivery.py'),'--root',str(root),'--output',str(audit_path)],
                                capture_output=True,text=True,timeout=min(60,remaining))
        control = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'one_consumer_only':True,'finished_clock':clock()}
    except (OSError,subprocess.TimeoutExpired) as error:
        control = {'returncode':None,'error_type':type(error).__name__,'error':str(error),'one_consumer_only':True,'finished_clock':clock()}
    save(root/'consumer_process.json',control)
    require(control['returncode']==0 and not (root/'consumer_failure.json').exists(),'single consumer failed; no restart')
    audit,terminal=load(audit_path),load(root/'terminal.json')
    require(terminal['sha256']==sha(audit_path) and Path(terminal['audit']).resolve()==audit_path.resolve() and
            terminal['input_qualified'] is True and audit['input_qualified'] is True and audit['audit_success'] is True,'matching qualified terminal')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan_sha256','grant'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();result=run(Path(a.plan),a.plan_sha256,Path(a.grant))
    print(json.dumps({k:result[k] for k in ('status','input_qualified')}))


if __name__=='__main__':main()
