"""Append-only, single-request metadata gate for the approved task-image sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

PROPOSAL_SHA = '7cf898fcbec44f74e06554f4f4db8691eb99875df897d0bfb06e1611b0b38442'
SEEDS = {'https://data.mendeley.com/datasets/j82bnw7gsr/1',
         'https://data.mendeley.com/api/docs/',
         'https://api.github.com/repos/detectRecog/CCPD/commits/master'}
HOSTS = {'data.mendeley.com', 'static.data.mendeley.com', 'api.github.com', 'github.com'}
PUBLIC_FILES = 'https://data.mendeley.com/public-api/datasets/j82bnw7gsr/files?version=1'


def parent_ledger(plan):
    if 'parent_metadata' not in plan:
        return {'requests': 0, 'body_bytes': 0, 'interrupted_read_upper_bound': 0, 'run_start_clock': None}
    p = plan['parent_metadata']; audit_path = Path(p['audit']); parent_plan = Path(p['plan'])
    if digest(audit_path) != p['audit_sha256']:
        raise ValueError('parent audit changed')
    ledger = json.loads(audit_path.read_text()); previous = json.loads(parent_plan.read_text())
    if (digest(parent_plan) != ledger['plan_sha256'] or 'parent_metadata' in previous or
            ledger['status'] != 'AUDITED_METADATA_LEDGER_ONLY' or ledger['request_failures'] or
            not ledger['within_metadata_deadline'] or ledger['boot_id'] != boot_id()):
        raise ValueError('ineligible/recursive parent metadata ledger')
    root = Path(p['root']); terminal = json.loads((root/'terminal.json').read_text())
    if digest(root/'plan.json') != ledger['plan_sha256'] or previous['raw_root'] != str(root) or terminal['sha256'] != p['audit_sha256'] or Path(terminal['audit']).resolve() != audit_path.resolve():
        raise ValueError('parent root/terminal mismatch')
    for index, key, url in [(0,'page_sha256','https://data.mendeley.com/datasets/j82bnw7gsr/1'),(2,'docs_sha256','https://data.mendeley.com/api/docs/')]:
        rp = root/('request_%03d'%index)/'receipt.json'; record=json.loads(rp.read_text())
        if digest(rp) != ledger['request_receipt_sha256'][index] or record['url'] != url or record['status'] != 'complete' or record['body_sha256'] != p[key]:
            raise ValueError('parent official-response identity mismatch')
    return ledger


def published_files_url(plan):
    p = plan['parent_metadata']; root = Path(p['root'])
    for index, name in [(0, 'page_sha256'), (2, 'docs_sha256')]:
        if digest(root / ('request_%03d' % index) / 'body.bin') != p[name]:
            raise ValueError('published route source changed')
    page = (root / 'request_000/body.bin').read_text(); docs = (root / 'request_002/body.bin').read_text()
    a = json.JSONDecoder().raw_decode(page.split('window.INITIAL_STATE = ', 1)[1])[0]
    d = json.JSONDecoder().raw_decode(docs.split('window.__state = ', 1)[1])[0]['spec']
    if (a['dataset']['snapshot']['id'] != 'j82bnw7gsr' or a['dataset']['snapshot']['version'] != 1 or
            a['configClient']['publicApiBaseUrl'] != '/public-api' or
            d['servers'][0]['url'] != 'https://api.data.mendeley.com' or
            'public-datasets' not in d['paths']['/datasets/{id}/files']['get']['tags'] or
            not any(x.get('name') == 'version' and x.get('in') == 'query' for x in d['paths']['/datasets/{id}/files']['get']['parameters'])):
        raise ValueError('published route schema mismatch')
    return PUBLIC_FILES


def boot_id():
    return subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.boottime'], text=True).strip()


def validate_schema(plan):
    c = plan['metadata_caps']
    if not (type(c['http_requests']) is int and 0 < c['http_requests'] <= 12 and
            type(c['body_bytes']) is int and 65536 < c['body_bytes'] <= 5242880 and
            0 < c['wall_seconds'] <= 600 and 1048576 < plan['storage_cap'] <= 268435456):
        raise ValueError('approved metadata/storage cap exceeded')
    if (plan['proposal_sha256'] != PROPOSAL_SHA or set(plan['seed_urls']) != SEEDS or
            set(plan['allowed_hosts']) != HOSTS):
        raise ValueError('fixed proposal/source schema changed')
    required = {'scripts/acquire_public_task_metadata.py', 'scripts/audit_public_task_metadata.py'}
    if not required.issubset(plan['source_sha256']) or not re.fullmatch('[0-9a-f]{40}', plan['source_commit']):
        raise ValueError('source freeze incomplete')
    if not Path(plan['worktree']).is_absolute() or not plan['storage_roots'] or not all(Path(p).is_absolute() for p in plan['storage_roots']):
        raise ValueError('absolute worktree/accounting roots required')


def stored_bytes(plan):
    files = set()
    for root in plan['storage_roots']:
        files.update(p.resolve() for p in Path(root).rglob('*') if p.is_file())
    return sum(p.stat().st_size for p in files)


def check_binding(root, plan):
    validate_schema(plan)
    claim = json.loads((root / 'claim.json').read_text())
    grant = json.loads((root / 'grant.json').read_text())
    if (digest(root / 'plan.json') != claim['plan_sha256'] or
            digest(root / 'grant.json') != claim['grant_sha256'] or
            grant['plan_sha256'] != claim['plan_sha256'] or
            grant['proposal_sha256'] != PROPOSAL_SHA or not grant.get('allow_metadata') or
            grant.get('allow_image_downloads') is not False):
        raise ValueError('plan/grant binding mismatch')
    if claim['boot_id'] != boot_id():
        raise ValueError('boot identity changed')
    parent_ledger(plan)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def source_check(plan):
    w = plan['worktree']
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=w, text=True).strip() != plan['source_commit']:
        raise ValueError('source commit changed')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=w, text=True):
        raise ValueError('tracked/untracked worktree is not clean')
    for name, expected in plan['source_sha256'].items():
        if digest(Path(w) / name) != expected:
            raise ValueError('source hash changed: ' + name)


def initialize(plan_path, expected, grant_path, root):
    if digest(plan_path) != expected:
        raise ValueError('plan SHA mismatch')
    plan = json.loads(Path(plan_path).read_text())
    grant = json.loads(Path(grant_path).read_text())
    validate_schema(plan)
    parent_ledger(plan)
    if grant['plan_sha256'] != expected or grant['proposal_sha256'] != plan['proposal_sha256'] or not grant.get('allow_metadata') or grant.get('allow_image_downloads') is not False:
        raise ValueError('grant binding mismatch')
    source_check(plan)
    root.mkdir(parents=True, exist_ok=False)
    for name, source in [('plan.json', plan_path), ('grant.json', grant_path)]:
        with (root / name).open('xb') as f:f.write(Path(source).read_bytes())
    save(root / 'claim.json', {'plan_sha256': expected, 'grant_sha256': digest(grant_path),
                              'created_unix': time.time(), 'boot_id': boot_id(), 'model_calls': 0, 'image_bytes': 0})


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def prior_receipts(root):
    paths = sorted(root.glob('request_[0-9][0-9][0-9]'))
    records = []
    for i, p in enumerate(paths):
        if p.name != 'request_%03d' % i or not (p / 'receipt.json').is_file():
            raise ValueError('unfinished or non-contiguous request; no recovery/retry')
        r = json.loads((p / 'receipt.json').read_text())
        if r['status'] != 'complete':
            raise ValueError('prior request failed; run sealed')
        records.append(r)
    return records


def validate_intent(root, plan, intent, prior):
    url = intent['url']
    if 'parent_metadata' in plan and url != PUBLIC_FILES:
        raise ValueError('parent mode permits only the one new published route')
    u = urllib.parse.urlsplit(url)
    if u.scheme != 'https' or u.hostname not in plan['allowed_hosts'] or u.username or u.password or u.port not in (None, 443) or u.fragment:
        raise ValueError('URL outside pinned source scope')
    source, kind = intent.get('source'), intent.get('kind')
    github_metadata = (source == 'ccpd_author' and kind == 'metadata' and u.hostname == 'api.github.com' and
                       (url in SEEDS or re.fullmatch('/repos/detectRecog/CCPD/git/trees/[0-9a-f]{40}', u.path)))
    derived = (url == PUBLIC_FILES and source == 'edgevision_v1' and kind == 'metadata' and published_files_url(plan) == url)
    edge_metadata = (source == 'edgevision_v1' and kind == 'metadata' and u.hostname == 'data.mendeley.com' and
                     (derived or url in SEEDS or re.fullmatch('/public-api/datasets/j82bnw7gsr/versions/1(?:/files)?', u.path)))
    edge_docs = (source == 'edgevision_v1' and kind == 'documentation' and
                 (url == 'https://data.mendeley.com/api/docs/' or
                  (u.hostname == 'static.data.mendeley.com' and u.path.endswith('.js'))))
    if not (github_metadata or edge_metadata or edge_docs):
        raise ValueError('wrong dataset/version/resource kind')
    if any(r['url'] == url for r in prior):
        raise ValueError('duplicate URL: no retry')
    if url not in plan['seed_urls'] and not derived:
        proof = intent['proof']
        index = proof['request_index']
        if type(index) is not int or not 0 <= index < len(prior):
            raise ValueError('invalid parent request')
        body = root / ('request_%03d' % index) / 'body.bin'
        if digest(body) != proof['body_sha256'] or url not in body.read_text(encoding='utf-8'):
            raise ValueError('URL is not literal in retained source; do not guess endpoint')


def acquire_one(root, intent_path, expected, *, opener=None, verify_source=True):
    root = Path(root)
    if (root / 'terminal.json').exists():
        raise ValueError('terminal run is immutable')
    plan = json.loads((root / 'plan.json').read_text())
    check_binding(root, plan)
    if verify_source:
        source_check(plan)
    if digest(intent_path) != expected:
        raise ValueError('intent SHA mismatch')
    intent = json.loads(Path(intent_path).read_text())
    prior = prior_receipts(root)
    ancestor = parent_ledger(plan)
    validate_intent(root, plan, intent, prior)
    caps = plan['metadata_caps']
    if len(prior) + ancestor['requests'] >= caps['http_requests']:
        raise ValueError('request cap exhausted')
    spent = sum(r['bytes'] + r['interrupted_read_upper_bound'] for r in prior) + ancestor['body_bytes'] + ancestor['interrupted_read_upper_bound']
    available = caps['body_bytes'] - spent
    if available < 65536:
        raise ValueError('insufficient read/diagnostic reserve')
    started = prior[0]['run_start_clock'] if prior else ancestor['run_start_clock'] or clock()
    if prior and (clock() < prior[-1]['finished_clock'] or started > clock()):
        raise ValueError('monotonic clock moved backwards')
    remaining_time = caps['wall_seconds'] - (clock() - started)
    if remaining_time <= 0:
        raise TimeoutError('metadata total deadline')
    if stored_bytes(plan) > plan['storage_cap'] - caps['body_bytes'] - 1048576:
        raise ValueError('disk reserve exhausted')
    folder = root / ('request_%03d' % len(prior))
    folder.mkdir(exist_ok=False)
    with (folder / 'intent.json').open('xb') as f:f.write(Path(intent_path).read_bytes())
    r = {'url': intent['url'], 'intent_sha256': expected, 'status': 'requesting', 'bytes': 0,
         'interrupted_read_upper_bound': 65536, 'run_start_clock': started,
         'started_clock': clock(), 'started_unix': time.time(), 'network_requests': 1,
         'automatic_retry': False, 'image_bytes': 0}
    def deadline(signum, frame):
        raise TimeoutError('metadata hard deadline')
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining_time)
    try:
        req = urllib.request.Request(intent['url'], headers={'User-Agent': 'PPO-MEC-quality-metadata/1.0', 'Accept-Encoding': 'identity'})
        op = opener or urllib.request.build_opener(NoRedirect())
        try:
            response = op.open(req, timeout=min(30, remaining_time))
        except urllib.error.HTTPError as error:
            response = error
        with response, (folder / 'body.bin').open('xb') as f:
            r.update(status_code=response.status, headers=dict(response.headers), final_url=response.geturl())
            if r['final_url'] != r['url']:
                raise ValueError('implicit redirect outside request accounting')
            content_type = response.headers.get('Content-Type', '').lower().split(';')[0]
            diagnostic = response.status != 200
            allowed = content_type.startswith('text/') or content_type in ('application/json', 'application/xml', 'application/javascript')
            if not allowed:
                raise ValueError('metadata MIME gate; image/archive bytes not consumed')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise ValueError('unexpected encoding')
            length = response.headers.get('Content-Length')
            length = int(length) if length is not None else None
            limit = min(available - 65536, 65536) if diagnostic else available - 65536
            if not diagnostic and length is not None and (length < 0 or length > limit):
                raise ValueError('declared body exceeds metadata reserve')
            while length is None or r['bytes'] < length:
                size = min(65536, limit - r['bytes'])
                if size <= 0:
                    raise ValueError('metadata byte cap reached')
                r['interrupted_read_upper_bound'] = size
                block = response.read1(size)
                r['interrupted_read_upper_bound'] = 0
                if not block:
                    break
                r['bytes'] += len(block)
                f.write(block)
            if diagnostic:
                raise ValueError('HTTP%d; body/headers retained, no retry' % response.status)
            if length is not None and r['bytes'] != length:
                raise ValueError('truncated declared response')
        (folder / 'body.bin').read_bytes().decode('utf-8')
        r['status'] = 'complete'
    except BaseException as error:
        r.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        r.update(finished_clock=clock(), finished_unix=time.time(),
                 retained_bytes=(folder / 'body.bin').stat().st_size if (folder / 'body.bin').exists() else 0,
                 body_sha256=digest(folder / 'body.bin') if (folder / 'body.bin').exists() else None,
                 byte_scope='returned body plus interrupted-read reserve; not exact wire/HTTP header/TLS bytes')
        save(folder / 'receipt.json', r)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['init', 'get'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--grant', type=Path)
    p.add_argument('--intent', type=Path)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    if a.operation == 'init':
        initialize(a.plan, a.sha256, a.grant, a.root)
    else:
        print(json.dumps(acquire_one(a.root, a.intent, a.sha256), ensure_ascii=False))


if __name__ == '__main__':
    main()
