"""One fixed 24-image acquisition. No model calls, retries or replacement."""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import signal
import subprocess
import sys
import time
import warnings
from urllib.parse import urljoin

from PIL import Image, ImageOps
import prepare_public_helmet_pilot as transport

W = Path(__file__).resolve().parents[1]
P = Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_helmet_prospective_20261102_v1')
SCOPE_SHA = '06c9b094aa0eaad416c48f875c1251e41440132c52f4816ea9588e7cdaab9baf'
SELECTION_SHA = '388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743'
CAPS = {'http': 48, 'body': 67108864, 'wall': 1500, 'storage': 134217728}
FILES = ('scripts/acquire_helmet_prospective.py', 'scripts/audit_helmet_prospective.py',
         'scripts/prepare_public_helmet_pilot.py', 'scripts/audit_public_helmet_pilot.py')
RESERVE = 1024 * 1024
require, load, sha, clock, boot = transport.require, transport.load, transport.sha, transport.clock, transport.boot


def regular(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and not any(q.is_symlink() for q in p.parents), 'regular nonsymlink input')
    return p


def checked(ref):
    p = regular(ref['path'])
    require(sha(p) == ref['sha256'], 'reference SHA: ' + str(p))
    return load(p)


def budget(plan, reserve=RESERVE):
    require(transport.storage([r for r in plan['storage_roots'] if Path(r).exists()]) + reserve <= CAPS['storage'], 'package storage reserve')


def save(plan, path, obj):
    data = (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
    budget(plan, len(data) + RESERVE)
    with Path(path).open('xb') as f:
        f.write(data)


def geometry(path, sample):
    """Display convention frozen independently of any model output."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as im:
            require(0 < im.width * im.height <= 60000000, 'source pixel cap')
            raw_size = list(im.size)
            orientation = im.getexif().get(274)
            require(orientation is None or orientation in range(9), 'EXIF orientation value')
            im.verify()
        with Image.open(path) as im:
            display = ImageOps.exif_transpose(im).convert('RGB')
            display.load()
        bbox = sample['native_annotation']['bbox']
        require(len(bbox) == 4 and all(type(v) in (int, float) and v == int(v) for v in bbox), 'native integer bbox')
        x, y, width, height = map(int, bbox)
        require(x >= 0 and y >= 0 and width > 0 and height > 0 and x + width <= display.width and y + height <= display.height, 'EXIF display ROI bounds')
        crop = display.crop((x, y, x + width, y + height))
        crop.info.clear()
        return crop, {'raw_size': raw_size, 'exif_orientation': orientation, 'display_size': list(display.size),
                      'bbox_xywh': bbox, 'crop_size': list(crop.size),
                      'crop_rgb_sha256': hashlib.sha256(crop.tobytes()).hexdigest()}


def preflight(plan):
    require(plan['schema'] == 'helmet24_acquisition_v1' and plan['new_caps'] == CAPS and plan['models'] == 0, 'fixed schema/caps')
    require(plan['producer_seconds'] == 1380 and plan['consumer_seconds'] == 120, 'fixed wall split')
    require(plan['scope'] == {'path': str(P / 'acquisition_scope.json'), 'sha256': SCOPE_SHA}, 'fixed scope pin')
    scope = checked(plan['scope'])
    require(plan['raw_root'] == scope['raw_root'] == str(W / 'data/raw/task_quality/helmet_prospective_20261102_v1'), 'fixed new raw root')
    require(plan['storage_roots'] == scope['storage_roots'] and plan['delivery_root'] == str(P / 'acquisition_delivery'), 'fixed storage/delivery roots')
    require(plan['selection'] == {'path': str(P / 'selection.json'), 'sha256': SELECTION_SHA}, 'selection pin')
    selection = checked(plan['selection'])
    require(selection['selected_counts'] == {'Helmet': 12, 'NoHelmet': 12} and len(selection['samples']) == 24 and not selection['blockers'], 'all24 frozen selection')
    require(len({s['image_id'] for s in selection['samples']}) == 24 and len({s['publisher_file']['content_details']['sha256_hash'] for s in selection['samples']}) == 24, 'distinct image/hash')
    require(sum(s['publisher_file']['size'] for s in selection['samples']) <= CAPS['body'], 'selected original byte cap')
    grant = checked(plan['grant'])
    require(grant['scope_sha256'] == SCOPE_SHA and grant['selection_sha256'] == SELECTION_SHA and grant['allow_downloads'] is True and grant['allow_single_s3_redirect'] is True and grant['models'] == 0, 'explicit scope grant')
    require(set(plan['source_sha256']) == set(FILES), 'exact implementation source set')
    for rel in FILES:
        require(sha(regular(W / rel)) == plan['source_sha256'][rel], 'source freeze ' + rel)
    require(plan['runtime'] == {'python': sys.executable, 'pillow': Image.__version__}, 'frozen acquisition runtime')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == plan['source_commit'], 'clean source commit identity')
    subprocess.run(['git', 'diff', '--quiet'], cwd=W, check=True)
    subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=W, check=True)
    return selection


class FixedCapture(transport.Capture):
    def __init__(self, root, plan, claim):
        # Deliberately independent of the old helper's 20-request initializer.
        self.root, self.plan, self.claim = Path(root), plan, claim
        require(plan['new_caps'] == CAPS and plan['producer_seconds'] == 1380, 'exact new scope')
        self.caps, self.producer_seconds = dict(CAPS), 1380
        self.records = []
        self.opener = transport.build_opener(transport.NoRedirect())
        self.selection = preflight(plan)
        self.initial_urls = {s['publisher_file']['content_details']['download_url'] for s in self.selection['samples']}

    def request(self, url, role, cap, redirect_from=None):
        preflight(self.plan)
        require(role.startswith('image_sample_') and int(role.rsplit('_', 1)[1]) in range(24), 'fixed image role')
        sample = self.selection['samples'][int(role.rsplit('_', 1)[1])]
        if redirect_from is None:
            require(url == sample['publisher_file']['content_details']['download_url'], 'fixed literal initial URL')
        else:
            parent = self.records[redirect_from]
            h = transport.transport_headers(parent['header_pairs'])
            require(parent['status'] == 'redirect' and parent['redirect_from'] is None and parent['role'] == role and url == urljoin(parent['url'], h['location']), 'literal single redirect provenance')
        return super().request(url, role, cap, redirect_from)


def run(plan, plan_path, plan_sha):
    started, boot_uuid = clock(), boot()
    selection = preflight(plan)
    root = Path(plan['raw_root'])
    root.mkdir(parents=False, exist_ok=False)
    save(plan, root / 'plan.json', plan)
    claim = {'plan_sha256': plan_sha, 'plan_path': str(plan_path), 'start_clock': started, 'boot_uuid': boot_uuid,
             'pid': os.getpid(), 'models': 0, 'schema': 'helmet24_single_launch_v1'}
    save(plan, root / 'claim.json', claim)
    oldalarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('producer absolute deadline')))
    signal.setitimer(signal.ITIMER_REAL, max(.001, claim['start_clock'] + 1380 - clock()))
    failure, cap, acquired = None, None, []
    try:
        cap = FixedCapture(root, plan, claim)
        for i, sample in enumerate(selection['samples']):
            row = sample['publisher_file']; cd = row['content_details']
            index = cap.asset(cd['download_url'], 'image_sample_%02d' % i, max(65536, row['size'] + 1), row['size'], cd['sha256_hash'])
            crop, qualification = geometry(cap.body(index), sample)
            crop.close()
            record = {'sample_index': i, 'sample_id': sample['sample_id'], 'request_index': index,
                      'original_path': str(cap.body(index)), 'image_id': sample['image_id'],
                      'annotation_id': sample['annotation_id'], 'publisher_sha256': cd['sha256_hash'], 'geometry': qualification}
            save(plan, root / ('image_%02d.json' % i), record)
            acquired.append(record)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'clock': clock()}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, oldalarm)
    if failure:
        save(plan, root / 'producer_failure.json', failure)
    save(plan, root / 'producer_complete.json', {'input_complete': failure is None and len(acquired) == 24,
        'acquired': acquired, 'finished_clock': clock(), 'requests_observed': len(cap.records) if cap else 0,
        'failure': failure, 'models': 0})
    consumer_cmd = [sys.executable, str(W / 'scripts/audit_helmet_prospective.py'), '--plan', str(plan_path), '--plan-sha256', plan_sha]
    save(plan, root / 'consumer_intent.json', {'command': consumer_cmd, 'clock': clock(), 'max_seconds': 120})
    child = None
    def consumer_file_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    with (root / 'consumer.log').open('xb') as log:
        try:
            remain = claim['start_clock'] + CAPS['wall'] - clock()
            require(remain > 1, 'consumer remaining absolute deadline')
            child = subprocess.Popen(consumer_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=consumer_file_limit)
            code = child.wait(timeout=min(119, remain - .5))
            process = {'pid': child.pid, 'returncode': code, 'finished_clock': clock(), 'error': None}
        except BaseException as error:
            process = {'pid': child.pid if child else None, 'returncode': None, 'error': str(error), 'finished_clock': clock()}
        finally:
            if child:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    process['cleanup_error'] = str(error)
                child.wait(timeout=.4)
    save(plan, root / 'consumer_process.json', process)
    audit = Path(plan['delivery_root']) / 'audit.json'; terminal = Path(plan['delivery_root']) / 'terminal.json'
    result = {'producer_success': failure is None, 'consumer_success': process['returncode'] == 0 and process['error'] is None and not process.get('cleanup_error') and audit.is_file() and terminal.is_file(),
              'audit_sha256': sha(audit) if audit.is_file() else None, 'terminal_sha256': sha(terminal) if terminal.is_file() else None,
              'finished_clock': clock(), 'storage_bytes_before_launch_result': transport.storage(plan['storage_roots']), 'models': 0}
    existing_bytes = transport.storage(plan['storage_roots'])
    result['storage_bytes_at_finish_including_result'] = existing_bytes
    for _ in range(8):
        encoded = (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
        result['storage_bytes_at_finish_including_result'] = existing_bytes + len(encoded)
    save(plan, root / 'launch_result.json', result)
    return 0 if result['producer_success'] and result['consumer_success'] else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True); p.add_argument('--plan-sha256')
    p.add_argument('--create-plan', action='store_true'); p.add_argument('--grant'); p.add_argument('--grant-sha256'); p.add_argument('--source-commit')
    a = p.parse_args()
    if a.create_plan:
        require(a.grant and a.grant_sha256 and a.source_commit and not a.plan_sha256, 'create-plan arguments')
        scope = load(P / 'acquisition_scope.json')
        plan = {'schema': 'helmet24_acquisition_v1', 'new_caps': dict(CAPS), 'producer_seconds': 1380, 'consumer_seconds': 120,
                'raw_root': scope['raw_root'], 'storage_roots': scope['storage_roots'], 'delivery_root': str(P / 'acquisition_delivery'),
                'scope': {'path': str(P / 'acquisition_scope.json'), 'sha256': SCOPE_SHA},
                'selection': {'path': str(P / 'selection.json'), 'sha256': SELECTION_SHA},
                'grant': {'path': str(Path(a.grant).absolute()), 'sha256': a.grant_sha256},
                'source_commit': a.source_commit, 'source_sha256': {rel: sha(regular(W / rel)) for rel in FILES},
                'runtime': {'python': sys.executable, 'pillow': Image.__version__}, 'models': 0}
        preflight(plan)
        save(plan, Path(a.plan), plan)
        print(sha(a.plan))
        return 0
    require(a.plan_sha256 is not None, 'plan SHA required')
    path = regular(a.plan); require(sha(path) == a.plan_sha256, 'plan SHA'); plan = load(path)
    try:
        return run(plan, path, a.plan_sha256)
    except BaseException as error:
        # If a claim/request exists, do not label the failure zero-network.
        root = Path(plan['raw_root']); claims = list(root.glob('request_*/claim.json')) if root.exists() else []
        note = {'failure': {'type': type(error).__name__, 'message': str(error)},
                'request_claims_observed': len(claims), 'network_upper_if_receipts_unavailable': CAPS['body'] if claims else 0,
                'models': 0, 'automatic_retry': False}
        target = P / 'acquisition_launch_control_failure.json'
        if not target.exists(): save(plan, target, note)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
