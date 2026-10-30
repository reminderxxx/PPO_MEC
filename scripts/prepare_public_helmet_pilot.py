"""Bounded, single-launch acquisition of six publisher-labelled Helmet inputs.

No inference, crops, relabelling, automatic retries or archive downloads. The
independent consumer reconstructs native GT selection from the retained files.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import build_opener, Request, HTTPRedirectHandler

from audit_public_helmet_pilot import SELECTION_RULE

W = Path(__file__).resolve().parents[1]
DATASET = 'j82bnw7gsr'
FOLDER = '39245ed4-43d6-497c-9fd9-45adca1a6ca0'
ANNOTATION = {'filename': 'annotations.json', 'id': '425f058b-dd8f-4ce5-a761-d9522649e018',
              'size': 2065913, 'sha256': '5e77fea3d8f1cd93cac42d45065456eeb97a54c31b7534ef453051c9d8853c91',
              'url': 'https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/425f058b-dd8f-4ce5-a761-d9522649e018/file_downloaded'}
CAPS = {'http': 20, 'body': 65520490, 'wall': 900, 'storage': 268435456}
PRIOR = (12, 1588374, 0)
RESERVE = 8 * 1024 * 1024
READ = 65536


def require(value, message):
    if not value:
        raise ValueError(message)


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def boot():
    value = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid'], text=True).strip()
    require(re.fullmatch(r'[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}', value), 'boot UUID')
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(READ), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(data)


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def storage(roots):
    paths = [Path(p).resolve() for p in roots]
    total = 0
    for p in paths:
        if any(p != q and p.is_relative_to(q) for q in paths):
            continue
        require(p.exists() and not p.is_symlink(), 'storage root absent/symlink')
        for f in ([p] if p.is_file() else p.rglob('*')):
            if f.is_symlink():
                total += f.lstat().st_size
            elif f.is_file():
                total += f.stat().st_size
    return total


def allowed(url, initial=False):
    u = urlsplit(url)
    valid = u.hostname == 'data.mendeley.com' or bool(u.hostname and re.fullmatch(
        r'prod-dcd-datasets-public-files-[a-z0-9-]+\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com', u.hostname))
    require(u.scheme == 'https' and valid and not u.username and not u.password and
            not u.fragment and u.port in (None, 443), 'publisher HTTPS URL scope')
    require(not initial or u.hostname == 'data.mendeley.com', 'initial publisher URL')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Capture:
    def __init__(self, root, plan, claim):
        self.root, self.plan, self.claim = Path(root), plan, claim
        self.records = []
        self.opener = build_opener(NoRedirect())

    def remaining(self):
        require(boot() == self.claim['boot_uuid'], 'boot changed')
        left = self.claim['start_clock'] + 870 - clock()
        require(left > 0, 'producer deadline including 30s consumer reserve')
        return left

    def request(self, url, role, cap, redirect_from=None):
        self.remaining()
        require(len(self.records) < CAPS['http'] and url not in {r['url'] for r in self.records},
                'request cap or duplicate URL; retries forbidden')
        require(not self.records or self.records[-1]['status'] != 'failed', 'failed prefix sealed')
        allowed(url, initial=redirect_from is None)
        require(storage(self.plan['storage_roots']) + cap + RESERVE < CAPS['storage'], 'storage reserve')
        used = sum(r['bytes'] + r['upper'] for r in self.records)
        require(CAPS['body'] - used >= READ and cap >= READ, 'full pre-open uncertainty reserve')
        number = len(self.records)
        directory = self.root / ('request_%03d' % number)
        directory.mkdir(exist_ok=False)
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'PPO_MEC-public-input/1.0'}
        if role.startswith('image_metadata_page_'):
            headers['Accept'] = 'application/vnd.mendeley-public-dataset.1+json'
        r = {'url': url, 'role': role, 'redirect_from': redirect_from, 'started': clock(),
             'status': 'failed', 'status_code': 0, 'bytes': 0, 'upper': 0, 'retained_bytes': 0,
             'body_sha256': None, 'headers': {}, 'request_headers': headers, 'final_url': url}
        save(directory / 'claim.json', dict(r))
        response = None
        failure = None
        try:
            # Reserve uncertainty before open/each read, including signal boundaries.
            r['upper'] = min(READ, cap, CAPS['body'] - used)
            try:
                response = self.opener.open(Request(url, headers=headers), timeout=min(60, self.remaining()))
            except HTTPError as error:
                response = error
            r['status_code'] = response.code
            r['headers'] = dict(response.headers.items())
            r['final_url'] = response.geturl()
            require(r['final_url'] == url, 'automatic redirect')
            h = {k.lower(): v for k, v in r['headers'].items()}
            require(h.get('content-encoding', 'identity').strip().lower() in ('', 'identity'), 'encoded response')
            length = h.get('content-length')
            if length is not None:
                require(length.isdecimal() and int(length) <= min(cap, CAPS['body'] - used), 'Content-Length/cap')
            if r['status_code'] == 200:
                mime = h.get('content-type', '').split(';')[0].strip().lower()
                valid = (mime.startswith('image/') or mime == 'application/octet-stream') if role.startswith('image_sample_') else (
                    mime in ('application/json', 'application/octet-stream') if role == 'annotation' else mime == 'application/json')
                require(valid, 'transport MIME before body read: ' + mime)
            elif r['status_code'] not in (302, 303, 307, 308):
                require(h.get('content-type', '').split(';')[0].strip().lower() in ('application/json', 'text/plain', 'text/html'), 'error body MIME')
                cap = min(cap, READ)
            with (directory / 'body.bin').open('xb') as stream:
                while True:
                    if length is not None and r['bytes'] == int(length):
                        r['upper'] = 0
                        break
                    self.remaining()
                    remaining = min(cap - r['bytes'], CAPS['body'] - used - r['bytes'])
                    require(remaining > 0, 'body cap reached before EOF')
                    amount = min(READ, remaining)
                    r['upper'] = amount
                    try:
                        block = response.read1(amount)
                    except BaseException as error:
                        partial = getattr(error, 'partial', b'')
                        if isinstance(partial, bytes) and len(partial) <= amount:
                            r.update(bytes=r['bytes'] + len(partial), upper=amount - len(partial))
                            stream.write(partial)
                        else:
                            r['upper'] = amount
                        raise
                    r.update(bytes=r['bytes'] + len(block), upper=0)
                    stream.write(block)
                    if not block:
                        break
            require(length is None or int(length) == r['bytes'], 'incomplete Content-Length')
            if r['status_code'] in (302, 303, 307, 308):
                require(redirect_from is None and h.get('location') and
                        (role == 'annotation' or role.startswith('image_sample_')), 'only one asset Location hop')
                target = urljoin(url, h['location'])
                allowed(target)
                require(bool(re.fullmatch(r'prod-dcd-datasets-public-files-[a-z0-9-]+\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com', urlsplit(target).hostname or '')), 'only public-files S3 Location')
                r['status'] = 'redirect'
            else:
                require(r['status_code'] == 200, 'HTTP status ' + str(r['status_code']))
                mime = h.get('content-type', '').split(';')[0].strip().lower()
                valid = (mime.startswith('image/') or mime == 'application/octet-stream') if role.startswith('image_sample_') else (
                    mime in ('application/json', 'application/octet-stream') if role == 'annotation' else mime == 'application/json')
                require(valid, 'transport MIME ' + mime)
                r['status'] = 'complete'
        except BaseException as error:
            failure = error
            r['status'] = 'failed'
            r['error_type'], r['error'] = type(error).__name__, str(error)
        finally:
            if response is not None:
                try:
                    response.close()
                except BaseException as close_error:
                    if failure is None:
                        failure = close_error
                        r['status'] = 'failed'
                        r['error_type'], r['error'] = type(close_error).__name__, str(close_error)
            body = directory / 'body.bin'
            if body.exists():
                r['retained_bytes'], r['body_sha256'] = body.stat().st_size, sha(body)
            r['finished'] = clock()
            save(directory / 'receipt.json', r)
            self.records.append(r)
        if failure is not None:
            raise failure
        return number

    def asset(self, url, role, cap, expected_size=None, expected_sha=None):
        first = self.request(url, role, cap)
        last = first
        if self.records[first]['status'] == 'redirect':
            location = next(v for k, v in self.records[first]['headers'].items() if k.lower() == 'location')
            target = urljoin(url, location)
            save(self.root / ('redirect_%03d.json' % first), {'parent_request': first, 'target': target,
                 'publisher_location': location, 'frozen_clock': clock(), 'boot_uuid': self.claim['boot_uuid']})
            last = self.request(target, role, cap, first)
        r = self.records[last]
        require(r['status'] == 'complete', 'complete asset required')
        require(expected_size is None or r['bytes'] == expected_size, 'publisher file size mismatch')
        require(expected_sha is None or r['body_sha256'] == expected_sha, 'publisher file SHA mismatch')
        return last

    def body(self, index):
        return self.root / ('request_%03d' % index) / 'body.bin'


def metadata(entry):
    name, fileid, size, detail = entry['filename'], entry['id'], entry['size'], entry['content_details']
    require(isinstance(name, str) and name not in ('', '.', '..') and '/' not in name and '\\' not in name, 'filename')
    require(isinstance(fileid, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', fileid), 'native fileID')
    require(type(size) is int and size > 0 and detail.get('size', size) == size and
            re.fullmatch(r'[0-9a-f]{64}', detail['sha256_hash']), 'native size/hash')
    url = detail['download_url']
    require(url == f'https://data.mendeley.com/public-files/datasets/{DATASET}/files/{fileid}/file_downloaded', 'native download URL')
    require(entry['status'] == 'COMPLETED' and entry['folder_id'] == FOLDER, 'image metadata folder/status')
    return name, fileid, size


def select(coco, entries):
    require(all(isinstance(coco.get(k), list) for k in ('images', 'annotations', 'categories')), 'COCO lists')
    byname, ids = {}, set()
    for e in entries:
        name, fileid, _ = metadata(e)
        require(name not in byname and fileid not in ids, 'metadata duplicate filename/ID')
        byname[name] = e
        ids.add(fileid)
    require(len(entries) == len(coco['images']), 'complete metadata/COCO image count')
    cats, native_classes = {}, set()
    for c in coco['categories']:
        key = (type(c['id']).__name__, c['id'])
        require(key not in cats and isinstance(c['name'], str), 'native category identity/name')
        mapped = {'helmet': 'Helmet', 'nohelmet': 'NoHelmet'}.get(re.sub(r'[\s_-]', '', c['name'].casefold()))
        if mapped:
            require(mapped not in native_classes, 'ambiguous native category mapping')
            native_classes.add(mapped)
        cats[key] = mapped
    require(native_classes == {'Helmet', 'NoHelmet'}, 'both native target categories')
    imgs, names = {}, set()
    for image in coco['images']:
        key = (type(image['id']).__name__, image['id'])
        fn = image['file_name']
        require(isinstance(fn, str) and '\\' not in fn, 'native POSIX filename')
        rel = PurePosixPath(fn)
        require(not rel.is_absolute() and '..' not in rel.parts and fn not in ('', '.'), 'safe relative basename')
        require(key not in imgs and rel.name not in names and rel.name in byname, 'COCO image bijection')
        require(all(type(image[k]) is int and image[k] > 0 for k in ('width', 'height')), 'native dimensions')
        imgs[key] = image
        names.add(rel.name)
    require(names == set(byname), 'COCO/official filename set')
    eligible, groups, annotation_ids = [], {}, set()
    rejected = {'bbox_or_image_bounds': 0, 'same_xywh_native_class_conflict': 0}
    for a in coco['annotations']:
        aid = (type(a['id']).__name__, a['id'])
        ikey, ckey = (type(a['image_id']).__name__, a['image_id']), (type(a['category_id']).__name__, a['category_id'])
        require(aid not in annotation_ids and ikey in imgs and ckey in cats, 'native annotation identities/references')
        annotation_ids.add(aid)
        box = a.get('bbox')
        valid = isinstance(box, list) and len(box) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in box)
        if valid:
            groups.setdefault((ikey, tuple(box)), set()).add(ckey)
        category = cats[ckey]
        if category is None:
            continue
        image = imgs[ikey]
        entry = byname[PurePosixPath(image['file_name']).name]
        if valid:
            x, y, width, height = box
            valid = x >= 0 and y >= 0 and width >= 32 and height >= 32 and x + width <= image['width'] and y + height <= image['height'] and image['width'] <= 8192 and image['height'] <= 8192 and image['width'] * image['height'] <= 20000000 and entry['size'] <= 8 * 1024 * 1024
        if not valid:
            rejected['bbox_or_image_bounds'] += 1
            continue
        key = hashlib.sha256(f"{DATASET}/1|{entry['id']}|{a['id']}|{category}".encode()).hexdigest()
        eligible.append((key, ikey, category, entry, a))
    candidates = []
    for item in eligible:
        if len(groups[(item[1], tuple(item[4]['bbox']))]) != 1:
            rejected['same_xywh_native_class_conflict'] += 1
        else:
            candidates.append(item)
    require(len({i[0] for i in candidates}) == len(candidates), 'ambiguous deterministic key')
    targets, used_ids, used_names = [], set(), set()
    for category in ('Helmet', 'NoHelmet'):
        count = 0
        for key, ikey, cls, entry, a in sorted((i for i in candidates if i[2] == category), key=lambda i: i[0]):
            if ikey in used_ids or entry['filename'] in used_names:
                continue
            targets.append({'sample_id': 'sample_%02d' % len(targets), 'image_file_id': entry['id'],
                            'image_filename': entry['filename'], 'image_metadata': entry, 'annotation': a,
                            'category_name': cls})
            used_ids.add(ikey)
            used_names.add(entry['filename'])
            count += 1
            if count == 3:
                break
        require(count == 3, 'insufficient deterministic candidates: ' + category)
    return targets, rejected


def zero_http_parent(plan):
    parent = plan.get('zero_http_predecessor')
    if parent is None:
        return None
    root = Path(parent['root'])
    for name, key in [('plan.json', 'plan_sha256'), ('claim.json', 'claim_sha256'), ('terminal.json', 'terminal_sha256'),
                      ('producer_failure.json', 'producer_failure_sha256'),
                      ('consumer_process.json', 'consumer_process_sha256')]:
        require(not (root / name).is_symlink() and sha(root / name) == parent[key], 'zero-HTTP predecessor SHA')
    require(sha(parent['audit']) == parent['audit_sha256'], 'zero-HTTP predecessor audit SHA')
    previous, audit, terminal = load(root/'plan.json'), load(parent['audit']), load(root/'terminal.json')
    claim, failure, control = load(root/'claim.json'), load(root/'producer_failure.json'), load(root/'consumer_process.json')
    require('zero_http_predecessor' not in previous and previous['source_commit'] == 'b619c885281eb0493d290cbacac8589b84e85c36' and
            previous['new_caps'] == CAPS and previous['raw_root'] == str(root) and previous['parent_audit'] == plan['parent_audit'], 'only direct initial failed scope')
    require(not list(root.glob('request_*')) and failure['requests'] == 0 and failure['error'] == 'storage symlink' and
            audit['failure_bill']['attempt_directory_count'] == 0 and audit['input_qualified'] is False and
            control['returncode'] == 1 and terminal['input_qualified'] is False and
            terminal['sha256'] == parent['audit_sha256'] and Path(terminal['audit']).resolve() == Path(parent['audit']).resolve() and
            claim['plan_sha'] == parent['plan_sha256'] and claim['boot_uuid'] == boot(), 'exact zero-HTTP failure provenance')
    require(0 <= clock() - claim['start_clock'] < 868, 'original absolute producer deadline already exhausted')
    return claim


def preflight(plan, expected, grant):
    require(plan['new_caps'] == CAPS and plan['annotation_expected'] == ANNOTATION and
            plan['images_folder_id'] == FOLDER and plan['dataset_id'] == DATASET and plan['version'] == 1 and
            plan['seed_schema'] == SELECTION_RULE, 'fixed source/selection/caps')
    require(grant['plan_sha256'] == expected and grant['allow_six_image_downloads'] is True and
            grant['allow_one_publisher_location_hop'] is True and grant['model_calls'] == 0, 'human grant binding')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == plan['source_commit'] and
            not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True), 'clean frozen checkout')
    for rel, digest in plan['source_sha'].items():
        path = Path(rel)
        require(not path.is_absolute() and '..' not in path.parts and sha(W / path) == digest, 'source SHA')
    parent = plan['parent_audit']
    require(sha(parent['path']) == parent['sha256'], 'parent audit binding')
    old = load(parent['path'])
    require((old['cumulative_requests'], old['cumulative_body_bytes'], old['cumulative_interrupted_upper']) == PRIOR,
            'inherit fixed prior costs once')
    proof = plan['annotation_metadata_proof']
    require(sha(proof['path']) == proof['sha256'], 'official COCO file metadata proof')
    entries = load(proof['path'])
    entry = next(x for x in entries if x['id'] == ANNOTATION['id'])
    require(entry['filename'] == ANNOTATION['filename'] and entry['size'] == ANNOTATION['size'] and
            entry['content_details']['sha256_hash'] == ANNOTATION['sha256'] and
            entry['content_details']['download_url'] == ANNOTATION['url'], 'official annotation entry')


def run(plan_path, expected, grant_path, root, audit_output):
    require(sha(plan_path) == expected, 'external plan SHA')
    plan, grant = load(plan_path), load(grant_path)
    preflight(plan, expected, grant)
    predecessor = zero_http_parent(plan)
    root = Path(root)
    require(str(root) == plan['raw_root'] and str(audit_output) == plan['audit_output'], 'fixed output paths')
    root.mkdir(parents=True, exist_ok=False)
    for name, src in [('plan.json', plan_path), ('grant.json', grant_path)]:
        with (root / name).open('xb') as stream:
            stream.write(Path(src).read_bytes())
    claim = {'plan_sha': expected, 'grant_sha256': sha(grant_path), 'boot_uuid': boot(), 'start_clock': predecessor['start_clock'] if predecessor else clock(),
             'started_at': datetime.now(timezone.utc).isoformat(), 'model_calls': 0}
    save(root / 'claim.json', claim)
    cap = Capture(root, plan, claim)
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('producer shared deadline')))
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 868 - (clock() - claim['start_clock'])))
    try:
        a = cap.asset(ANNOTATION['url'], 'annotation', ANNOTATION['size'] + READ, ANNOTATION['size'], ANNOTATION['sha256'])
        coco = load(cap.body(a))
        entries, pages = [], []
        for page in range(3):
            query = urlencode({'folder_id': FOLDER, 'version': 1, '$start': page * 1000, '$limit': 1000})
            url = f'https://data.mendeley.com/public-api/datasets/{DATASET}/files?' + query
            index = cap.asset(url, 'image_metadata_page_%d' % page, 2 * 1024 * 1024)
            obj = load(cap.body(index))
            data = obj.get('files') if isinstance(obj, dict) else obj
            require(isinstance(data, list) and len(data) <= 1000 and not (isinstance(obj, dict) and 'error' in obj), 'metadata page schema')
            pages.append(index)
            entries.extend(data)
            if len(data) < 1000:
                break
        require(len(data) < 1000, 'no terminal metadata page within 3')
        targets, rejected = select(coco, entries)
        save(root / 'selection.json', {'selection_rule': SELECTION_RULE, 'selected': targets, 'candidate_rejections': rejected})
        save(root / 'selection_claim.json', {'selection_sha256': sha(root / 'selection.json'), 'boot_uuid': claim['boot_uuid'], 'frozen_clock': clock()})
        selected = []
        for number, target in enumerate(targets):
            e = target['image_metadata']
            r = cap.asset(e['content_details']['download_url'], 'image_sample_%d' % number, 8 * 1024 * 1024,
                          e['size'], e['content_details']['sha256_hash'])
            selected.append(dict(target, request_index=r, opaque_input_path=str(cap.body(r))))
        save(root / 'index.json', {'annotation_request': a, 'image_metadata_requests': pages, 'selected': selected})
        cap.remaining()
        save(root / 'producer_complete.json', {'requests': len(cap.records), 'finished_clock': clock(), 'model_calls': 0})
    except BaseException as error:
        save(root / 'producer_failure.json', {'error_type': type(error).__name__, 'error': str(error),
             'finished_clock': clock(), 'requests': len(cap.records), 'no_retry': True, 'model_calls': 0})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    # Exactly one independent offline consumer, including the retained failure prefix.
    consumer = W / 'scripts/audit_public_helmet_pilot.py'
    remaining = 900 - (clock() - claim['start_clock'])
    require(remaining > 0, 'shared deadline before consumer')
    try:
        result = subprocess.run([__import__('sys').executable, str(consumer), '--root', str(root), '--output', str(audit_output)],
                                timeout=min(30, remaining), capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError) as error:
        save(root / 'consumer_process.json', {'returncode': None, 'error_type': type(error).__name__,
             'stdout': str(getattr(error, 'stdout', '') or ''), 'stderr': str(getattr(error, 'stderr', '') or ''), 'finished_clock': clock(),
             'one_consumer_only': True, 'input_qualified': False, 'no_retry': True})
        raise
    save(root / 'consumer_process.json', {'returncode': result.returncode, 'stdout': result.stdout,
         'stderr': result.stderr, 'finished_clock': clock(), 'one_consumer_only': True})
    require(result.returncode == 0 and not (root / 'consumer_failure.json').exists(), 'independent consumer did not qualify; sealed failure retained')
    result = load(audit_output)
    terminal = load(root / 'terminal.json')
    require(terminal['sha256'] == sha(audit_output) and Path(terminal['audit']).resolve() == Path(audit_output).resolve() and
            terminal['audit_success'] is True and terminal['input_qualified'] is True and
            result['audit_success'] is True and result['input_qualified'] is True, 'matching successful consumer pair required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan_sha256', 'grant', 'root', 'audit_output'):
        parser.add_argument('--' + key, required=True)
    a = parser.parse_args()
    result = run(Path(a.plan), a.plan_sha256, Path(a.grant), Path(a.root), Path(a.audit_output))
    print(json.dumps({k: result[k] for k in ('status', 'images', 'new_requests', 'new_body_bytes')}))


if __name__ == '__main__':
    main()
