"""Three new offline producer cases; no network or model calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_public_helmet_pilot as p


def entry(i):
    ident = 'file-%d' % i
    return {'id': ident, 'filename': '%d.jpg' % i, 'size': 100, 'status': 'COMPLETED',
            'folder_id': p.FOLDER, 'content_details': {'size': 100, 'sha256_hash': '0' * 64,
            'download_url': f'https://data.mendeley.com/public-files/datasets/{p.DATASET}/files/{ident}/file_downloaded'}}


def test_fixed_native_selection_and_conflict():
    entries = [entry(i) for i in range(8)]
    coco = {'categories': [{'id': 9, 'name': 'Helmet'}, {'id': 6, 'name': 'No_Helmet'}],
            'images': [{'id': i, 'file_name': 'images/%d.jpg' % i, 'width': 100, 'height': 100} for i in range(8)],
            'annotations': [{'id': i, 'image_id': i, 'category_id': 9 if i < 4 else 6, 'bbox': [1, 2, 40, 40]} for i in range(8)]}
    coco['annotations'].append({'id': 99, 'image_id': 0, 'category_id': 6, 'bbox': [1, 2, 40, 40]})
    selected, rejected = p.select(coco, entries)
    assert [s['category_name'] for s in selected] == ['Helmet'] * 3 + ['NoHelmet'] * 3
    assert len({s['image_file_id'] for s in selected}) == 6
    assert all(s['annotation']['image_id'] != 0 for s in selected)
    assert rejected['same_xywh_native_class_conflict'] == 2
    second, _ = p.select(copy.deepcopy(coco), list(reversed(entries)))
    assert selected == second


class Response:
    def __init__(self, url, code, headers, blocks):
        self.url, self.code, self.headers, self.blocks = url, code, headers, list(blocks)
    def geturl(self): return self.url
    def read1(self, amount):
        item = self.blocks.pop(0)
        if isinstance(item, BaseException): raise item
        assert len(item) <= amount
        return item
    def close(self): pass


def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'boot', lambda: 'test')
    plan = {'storage_roots': [str(tmp_path)], 'new_caps': dict(p.CAPS)}
    return p.Capture(tmp_path, plan, {'boot_uuid': 'test', 'start_clock': p.clock()})


def test_failed_read_billed_once_no_retry(tmp_path, monkeypatch):
    c = capture(tmp_path, monkeypatch)
    url = p.ANNOTATION['url']
    calls = []
    def opened(request, timeout):
        calls.append(request.full_url)
        return Response(url, 200, {'Content-Type': 'application/json'}, [b'abc', TimeoutError('synthetic')])
    monkeypatch.setattr(c.opener, 'open', opened)
    with pytest.raises(TimeoutError): c.request(url, 'annotation', 2*p.READ)
    r = json.loads((tmp_path/'request_000/receipt.json').read_text())
    assert (r['bytes'], r['upper'], r['retained_bytes'], r['status']) == (3, p.READ, 3, 'failed')
    with pytest.raises(ValueError): c.request(url, 'annotation', 2*p.READ)
    assert calls == [url]


def test_literal_single_redirect_and_hash(tmp_path, monkeypatch):
    c = capture(tmp_path, monkeypatch)
    url = p.ANNOTATION['url']
    target = 'https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/a'
    responses = [Response(url, 302, {'Location': target, 'Content-Length': '0'}, [b'']),
                 Response(target, 200, {'Content-Type': 'application/octet-stream', 'Content-Length': '1'}, [b'x', b''])]
    calls = []
    def opened(request, timeout):
        calls.append(request.full_url)
        return responses.pop(0)
    monkeypatch.setattr(c.opener, 'open', opened)
    assert c.asset(url, 'annotation', 2*p.READ, 1, hashlib.sha256(b'x').hexdigest()) == 1
    assert calls == [url, target]
    assert c.records[1]['redirect_from'] == 0
    assert json.loads((tmp_path/'redirect_000.json').read_text())['target'] == target
    for bad in ('https://other.example/a', 'http://data.mendeley.com/a', 'https://data.mendeley.com@other.example/a'):
        with pytest.raises(ValueError): p.allowed(bad)
