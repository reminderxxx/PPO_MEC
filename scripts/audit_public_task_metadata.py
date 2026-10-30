"""Independent single terminal consumer; never downloads or qualifies image quality."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.parse


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root, plan_sha, decision):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    assert digest(root / 'plan.json') == plan_sha, 'plan binding'
    assert plan['proposal_sha256'] == '7cf898fcbec44f74e06554f4f4db8691eb99875df897d0bfb06e1611b0b38442', 'approved proposal'
    assert set(plan['seed_urls']) == {'https://data.mendeley.com/datasets/j82bnw7gsr/1', 'https://data.mendeley.com/api/docs/', 'https://api.github.com/repos/detectRecog/CCPD/commits/master'}, 'fixed seed sources'
    assert set(plan['allowed_hosts']) == {'data.mendeley.com', 'static.data.mendeley.com', 'api.github.com', 'github.com'}, 'fixed source hosts'
    assert {'scripts/acquire_public_task_metadata.py', 'scripts/audit_public_task_metadata.py'}.issubset(plan['source_sha256']) and re.fullmatch('[0-9a-f]{40}', plan['source_commit']), 'source freeze fields'
    claim = json.loads((root / 'claim.json').read_text())
    assert claim['plan_sha256'] == plan_sha, 'claim binding'
    assert digest(root / 'grant.json') == claim['grant_sha256'], 'grant binding'
    grant = json.loads((root / 'grant.json').read_text())
    assert grant['plan_sha256'] == plan_sha and grant['proposal_sha256'] == plan['proposal_sha256'], 'authorization binding'
    assert grant.get('allow_metadata') and grant.get('allow_image_downloads') is False, 'metadata-only grant'
    assert claim['boot_id'] == subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.boottime'], text=True).strip(), 'boot identity'
    ancestor = {'requests': 0, 'body_bytes': 0, 'interrupted_read_upper_bound': 0, 'run_start_clock': None}
    derived_url = None
    if 'parent_metadata' in plan:
        p = plan['parent_metadata']; ap = Path(p['audit']); pp = Path(p['plan'])
        assert digest(ap) == p['audit_sha256'], 'parent audit hash'
        ancestor = json.loads(ap.read_text()); old_plan = json.loads(pp.read_text())
        assert digest(pp) == ancestor['plan_sha256'] and 'parent_metadata' not in old_plan, 'nonrecursive parent'
        assert ancestor['status'] == 'AUDITED_METADATA_LEDGER_ONLY' and not ancestor['request_failures'] and ancestor['within_metadata_deadline'] and ancestor['boot_id'] == claim['boot_id'], 'eligible parent'
        parent = Path(p['root']); page = parent/'request_000/body.bin'; docs = parent/'request_002/body.bin'
        terminal=json.loads((parent/'terminal.json').read_text())
        assert digest(parent/'plan.json')==ancestor['plan_sha256'] and old_plan['raw_root']==str(parent) and terminal['sha256']==p['audit_sha256'] and Path(terminal['audit']).resolve()==ap.resolve(), 'parent root/terminal'
        for index,key,url in [(0,'page_sha256','https://data.mendeley.com/datasets/j82bnw7gsr/1'),(2,'docs_sha256','https://data.mendeley.com/api/docs/')]:
            receipt_path=parent/('request_%03d'%index)/'receipt.json'; item=json.loads(receipt_path.read_text())
            assert digest(receipt_path)==ancestor['request_receipt_sha256'][index] and item['url']==url and item['status']=='complete' and item['body_sha256']==p[key], 'official parent receipt chain'
        assert digest(page) == p['page_sha256'] and digest(docs) == p['docs_sha256'], 'documented route bodies'
        a = json.JSONDecoder().raw_decode(page.read_text().split('window.INITIAL_STATE = ',1)[1])[0]
        d = json.JSONDecoder().raw_decode(docs.read_text().split('window.__state = ',1)[1])[0]['spec']
        assert a['dataset']['snapshot']['id']=='j82bnw7gsr' and a['dataset']['snapshot']['version']==1 and a['configClient']['publicApiBaseUrl']=='/public-api', 'dataset/public proxy identity'
        method=d['paths']['/datasets/{id}/files']['get']
        assert d['servers'][0]['url']=='https://api.data.mendeley.com' and 'public-datasets' in method['tags'] and any(v.get('name')=='version' and v.get('in')=='query' for v in method['parameters']), 'published GET route'
        derived_url='https://data.mendeley.com/public-api/datasets/j82bnw7gsr/files?version=1'
    folders = sorted(root.glob('request_[0-9][0-9][0-9]'))
    calls = body_bytes = upper = 0
    rows = []
    for i, folder in enumerate(folders):
        assert folder.name == 'request_%03d' % i, 'request sequence'
        r = json.loads((folder / 'receipt.json').read_text())
        intent = json.loads((folder / 'intent.json').read_text())
        assert 'parent_metadata' not in plan or intent['url']==derived_url, 'only new parent-mode route'
        assert digest(folder / 'intent.json') == r['intent_sha256'], 'intent binding'
        assert r['url'] == intent['url'] and r['network_requests'] == 1, 'request identity'
        u = urllib.parse.urlsplit(r['url'])
        assert u.scheme == 'https' and u.hostname in plan['allowed_hosts'] and not u.username and not u.password and u.port in (None, 443) and not u.fragment, 'URL scope'
        source, kind = intent.get('source'), intent.get('kind')
        github = source == 'ccpd_author' and kind == 'metadata' and u.hostname == 'api.github.com' and (u.path == '/repos/detectRecog/CCPD/commits/master' or re.fullmatch('/repos/detectRecog/CCPD/git/trees/[0-9a-f]{40}', u.path))
        edge = source == 'edgevision_v1' and kind == 'metadata' and u.hostname == 'data.mendeley.com' and (r['url']==derived_url or u.path in ('/datasets/j82bnw7gsr/1', '/api/docs/') or re.fullmatch('/public-api/datasets/j82bnw7gsr/versions/1(?:/files)?', u.path))
        docs = source == 'edgevision_v1' and kind == 'documentation' and (r['url'] == 'https://data.mendeley.com/api/docs/' or u.hostname == 'static.data.mendeley.com' and u.path.endswith('.js'))
        assert github or edge or docs, 'dataset/version/kind'
        path = folder / 'body.bin'
        actual = path.stat().st_size if path.exists() else 0
        assert actual == r['retained_bytes'] <= r['bytes'], 'retained/returned body bytes'
        assert (digest(path) if path.exists() else None) == r['body_sha256'], 'body hash'
        headers = {k.lower(): v for k, v in r.get('headers', {}).items()}
        mime = headers.get('content-type', '').lower().split(';')[0]
        if actual > 0:
            assert mime.startswith('text/') or mime in ('application/json', 'application/xml', 'application/javascript'), 'metadata MIME including failure body'
            assert headers.get('content-encoding', 'identity').lower() in ('', 'identity'), 'encoding including failure body'
            path.read_bytes().decode('utf-8')
        if r['status'] == 'complete':
            assert r['status_code'] == 200 and r['final_url'] == r['url'], 'status/redirect'
            assert actual == r['bytes'] and r['interrupted_read_upper_bound'] == 0, 'complete body'
            path.read_bytes().decode('utf-8')
            assert mime.startswith('text/') or mime in ('application/json', 'application/xml', 'application/javascript'), 'metadata MIME'
            assert headers.get('content-encoding', 'identity').lower() in ('', 'identity'), 'encoding'
            declared = headers.get('content-length')
            assert declared is None or int(declared) == actual, 'content length'
        else:
            assert i == len(folders) - 1, 'requests after failure'
        assert not r['automatic_retry'] and r['image_bytes'] == 0, 'input-only metadata'
        assert 0 <= r['interrupted_read_upper_bound'] <= 65536, 'read reserve'
        assert r['finished_clock'] >= r['started_clock'] >= r['run_start_clock'], 'clock order'
        if rows:
            assert r['run_start_clock'] == rows[0]['run_start_clock'], 'deadline reset'
            assert r['started_clock'] >= rows[-1]['finished_clock'], 'concurrent requests'
        assert r['url'] not in [x['url'] for x in rows], 'duplicate URL'
        if r['url'] not in plan['seed_urls'] and r['url'] != derived_url:
            proof = intent['proof']; n = proof['request_index']
            assert type(n) is int and 0 <= n < i, 'parent index'
            parent = folders[n] / 'body.bin'
            assert digest(parent) == proof['body_sha256'] and r['url'] in parent.read_text(), 'source link proof'
        calls += 1; body_bytes += r['bytes']; upper += r['interrupted_read_upper_bound']; rows.append(r)
    calls += ancestor['requests']; body_bytes += ancestor['body_bytes']; upper += ancestor['interrupted_read_upper_bound']
    cap = plan['metadata_caps']
    assert 0 <= cap['http_requests'] <= 12 and 0 <= cap['body_bytes'] <= 5242880 and 0 < cap['wall_seconds'] <= 600, 'approved caps'
    assert calls <= cap['http_requests'] and body_bytes + upper <= cap['body_bytes'], 'M limits'
    now = time.clock_gettime(time.CLOCK_MONOTONIC)
    assert not rows or now >= rows[-1]['finished_clock'], 'monotonic clock moved backwards'
    start = ancestor['run_start_clock'] or (rows[0]['run_start_clock'] if rows else None)
    assert not rows or not ancestor['run_start_clock'] or rows[0]['run_start_clock']==ancestor['run_start_clock'], 'parent deadline reset'
    elapsed = now - start if start is not None else 0
    # A deadline failure may finish just after its cap; it is retained, not PASS input.
    assert elapsed >= 0, 'monotonic clock moved backwards'
    within_time = elapsed <= cap['wall_seconds']
    all_files = set()
    for account in plan['storage_roots']:
        all_files.update(p.resolve() for p in Path(account).rglob('*') if p.is_file())
    stored = sum(p.stat().st_size for p in all_files)
    assert stored + 1048576 <= plan['storage_cap'], 'storage/terminal reserve'
    return {'status': 'AUDITED_METADATA_LEDGER_ONLY' if within_time else 'FAILED_METADATA_DEADLINE_LEDGER_RETAINED', 'decision': decision,
            'plan_sha256': plan_sha, 'requests': calls, 'body_bytes': body_bytes,
            'interrupted_read_upper_bound': upper, 'elapsed_seconds': elapsed,
            'within_metadata_deadline': within_time, 'retained_bytes_before_audit': stored,
            'request_failures': sum(r['status'] != 'complete' for r in rows),
            'D_request_remaining': 32 - calls, 'D_body_remaining': 67108864 - body_bytes - upper,
            'D_wall_remaining': max(0, 1800 - elapsed), 'model_calls': 0, 'image_bytes': 0,
            'D_storage_remaining_conservative': plan['storage_cap'] - stored - 1048576,
            'run_start_clock': start, 'boot_id': claim['boot_id'],
            'D_balance_note': 'Nominal unspent balance, not permission to continue a failed/sealed run; total wall includes orchestration/audit.1MiB terminal reserve charged.',
            'input_qualified': False, 'paper_goal_complete': False,
            'request_receipt_sha256': [digest(p / 'receipt.json') for p in folders]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--plan_sha256', required=True)
    p.add_argument('--decision', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or (a.root / 'terminal.json').exists():
        raise ValueError('terminal audit already exists; reuse it')
    result = audit(a.root, a.plan_sha256, a.decision)
    with a.output.open('x') as f:json.dump(result, f, ensure_ascii=False, indent=2);f.write('\n')
    with (a.root / 'terminal.json').open('x') as f:json.dump({'audit': str(a.output), 'sha256': digest(a.output)}, f, indent=2);f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
