"""Separately granted, create-only TGSIM pages; no scientific execution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.parse
import urllib.request

try:
    from scripts.prepare_tgsim_input import (ASSETS, HOST, ROOT, Budget, BudgetLimit,
        digest, fetch, inspect_geometry, inspect_trajectory, metadata_pin, save_new)
except ModuleNotFoundError:
    from prepare_tgsim_input import (ASSETS, HOST, ROOT, Budget, BudgetLimit,
        digest, fetch, inspect_geometry, inspect_trajectory, metadata_pin, save_new)

PROPOSAL_SHA = '71abfd4dfb1937ad694effa2b8d41e2f3d6b14bc3c02164b290c42b5a8e15a64'
OLD_BODY_BOUND = 187500016
OLD_RAW_SHA = '7fa1683b60a02a20d522601544d81df22a4d3ecf4f295fc26241ec6805d15b22'
PRIVATE = Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis')


def disk_bytes(roots):
    # Resolve and deduplicate; reject overlapping roots instead of double billing.
    roots = list(dict.fromkeys(Path(p).resolve() for p in roots))
    if any(a != b and a in b.parents for a in roots for b in roots):
        raise ValueError('disk accounting roots overlap')
    return sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())


def accounting_roots(raw, analysis):
    return [ROOT/'data/raw/mobility/tgsim/tgsim_input_qualification_20261027_v1',
            ROOT/'artifacts/analysis/paper_tgsim_input_qualification_20261027_v1',
            PRIVATE/'paper_input_source_feasibility_20261027_v1',
            PRIVATE/'paper_tgsim_preparation_software_20261027_v1',
            PRIVATE/'paper_tgsim_paged_preparation_software_20261027_v1', raw, analysis]


class PagedBudget(Budget):
    def __init__(self, proposal, directories, clock=time.monotonic):
        limits = dict(max_wall_clock_seconds=proposal['max_wall_clock_seconds'],
                      max_new_disk_bytes=proposal['max_cumulative_new_disk_bytes_including_old_failure'],
                      max_download_bytes=proposal['max_download_bytes_new_task'],
                      max_trajectory_rows_to_index=proposal['max_rows_to_index'])
        super().__init__(limits, directories, clock)
        self.requests = 0
        self.maximum_requests = proposal['maximum_http_requests']
        self.urls = {}

    def request(self, url, *, version_check=False):
        self.guard()
        allowed = 2 if version_check else 1
        if self.urls.get(url, 0) >= allowed:
            raise ValueError('same request URL cannot be retried')
        if self.requests >= self.maximum_requests:
            raise BudgetLimit('HTTP request cap reached')
        self.urls[url] = self.urls.get(url, 0) + 1; self.requests += 1


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect rejected without a second HTTP request')


def page_url(source, offset, page_rows):
    query = dict(source['fixed_query'])
    query.update({'$offset': offset, '$limit': page_rows})
    return source['trajectory_endpoint'] + '?' + urllib.parse.urlencode(query)


def parse_count(path, expected):
    data = json.loads(path.read_text())
    if (not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict)
            or set(data[0]) != {'count'}):
        raise ValueError('count response schema differs from frozen contract')
    value = str(data[0]['count'])
    if not re.fullmatch(r'[0-9]+', value) or int(value) != expected:
        raise ValueError('fresh source row count differs from frozen count')
    return int(value)


def system_id_key(value):
    if re.fullmatch(r'[0-9]+', value):
        return ('integer', int(value))
    if re.fullmatch(r'row-[a-z0-9-]+', value):
        return ('row_ascii', value)
    raise ValueError('unsupported or empty source system-ID representation')


def append_page(path, source, writer, expected_rows, previous_key, budget):
    """Preserve all fields; exact page length and strict IDs across boundaries."""
    count = 0; first_id = last_id = None
    expected = ['source_row_id'] + source['expected_native_fields']
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != expected:
            raise ValueError('paged CSV header differs from frozen native fields/order')
        for row in reader:
            if len(row) != len(expected):
                raise ValueError('ragged paged CSV row')
            key = system_id_key(row[0])
            if previous_key is not None and (key[0] != previous_key[0] or key <= previous_key):
                raise ValueError('duplicate or out-of-order system ID, including page boundary')
            if count >= expected_rows:
                raise ValueError('page contains more than its exact expected row count')
            previous_key = key; count += 1
            first_id = row[0] if first_id is None else first_id
            last_id = row[0]
            # The retained page binds source IDs; the separate derived CSV has
            # exactly the native header expected by the existing identity consumer.
            writer.writerow(row[1:])
            if count % 10000 == 0:
                budget.guard()
    if count != expected_rows:
        raise ValueError('missing/short page; partial composition is never complete input')
    budget.guard()
    return previous_key, dict(rows=count, first_source_row_id=first_id, last_source_row_id=last_id)


def validate_plan(proposal_path, grant_path):
    if digest(proposal_path) != PROPOSAL_SHA:
        raise ValueError('proposal does not match exact newly approved SHA')
    proposal = json.loads(proposal_path.read_text()); grant = json.loads(grant_path.read_text())
    if (not grant.get('execution_authorized') or grant.get('proposal_sha256') != PROPOSAL_SHA
            or grant.get('answer_verbatim') != '允许按分页新方案继续'):
        raise ValueError('new explicit human grant missing or mismatched')
    exact = dict(page_rows=50000, maximum_trajectory_pages=120, maximum_http_requests=144,
                 max_download_bytes_new_task=536870912,
                 max_cumulative_new_disk_bytes_including_old_failure=2147483648,
                 max_wall_clock_seconds=2700, max_rows_to_index=6000000)
    if any(proposal[k] != v or grant['caps'][k] != v for k, v in exact.items()):
        raise ValueError('approved paged caps differ')
    if (tuple(s['asset_id'] for s in proposal['sources']) != ASSETS
            or sum(len(s['support_csv_attachments']) for s in proposal['sources']) != 16
            or proposal['first_task_observed_body_upper_bound_bytes'] != OLD_BODY_BOUND
            or proposal['two_task_total_body_upper_bound_bytes'] != OLD_BODY_BOUND + 536870912):
        raise ValueError('source pool or cumulative bill differs')
    for key in ('training_frames_authorized', 'evaluation_frames_authorized'):
        if proposal[key] != 0:
            raise ValueError('input grant permits no science')
    for key in ('destination_raw_directory', 'destination_analysis_directory'):
        p = Path(proposal[key])
        if not p.resolve().is_relative_to(ROOT.resolve()) or p.exists():
            raise ValueError('create-only new worktree destination required')
    return proposal, grant


def acquire(proposal, raw, analysis, budget, *, opener=None):
    """All requests live here; any exception exits globally, never per-page retry."""
    opener = opener or urllib.request.build_opener(NoRedirect())
    result = dict(page_inventory=[], trajectory_checks=[], geometry_checks=[])

    def get(url, path, receipt, cap=None, *, version_check=False):
        if path.exists() or receipt.exists():
            raise FileExistsError('request output already exists')
        budget.request(url, version_check=version_check)
        return fetch(url, path, budget, receipt, maximum_bytes=cap, opener=opener)

    def source_check(source, phase):
        asset = source['asset_id']; meta = analysis/f'{asset}_metadata_{phase}.json'
        get(source['metadata_url'], meta, analysis/f'{asset}_metadata_{phase}_receipt.json', 1024**2, version_check=True)
        if metadata_pin(json.loads(meta.read_text())) != source['metadata_pin']:
            raise ValueError('source metadata version/schema/attachment identity drift')
        path = analysis/f'{asset}_count_{phase}.json'
        get(source['row_count_url'], path, analysis/f'{asset}_count_{phase}_receipt.json', 1024**2, version_check=True)
        parse_count(path, source['expected_count_from_frozen_metadata_not_fresh_query'])

    # Count and metadata before BOTH assets; no trajectory/geometry until they pass.
    for source in proposal['sources']:
        source_check(source, 'before')
    pages = sum((s['expected_count_from_frozen_metadata_not_fresh_query'] + proposal['page_rows']-1)
                // proposal['page_rows'] for s in proposal['sources'])
    if pages > proposal['maximum_trajectory_pages']:
        raise BudgetLimit('trajectory page cap reached before fetching')
    for source in proposal['sources']:
        asset = source['asset_id']; count = source['expected_count_from_frozen_metadata_not_fresh_query']
        combined = raw/f'{asset}_derived_native.csv'; previous = None
        with combined.open('x', newline='') as out:
            writer = csv.writer(out); writer.writerow(source['expected_native_fields'])
            for index, offset in enumerate(range(0, count, proposal['page_rows'])):
                path = raw/f'{asset}_page_{index:03d}.csv'
                receipt = analysis/f'{asset}_page_{index:03d}_receipt.json'
                info = get(page_url(source, offset, proposal['page_rows']), path, receipt)
                previous, checked = append_page(path, source, writer, min(proposal['page_rows'], count-offset), previous, budget)
                out.flush(); budget.guard()
                entry = dict(asset_id=asset, page_index=index, offset=offset, limit=proposal['page_rows'],
                             path=path.name, bytes=info['bytes'], sha256=info['sha256'], **checked)
                save_new(analysis/f'{asset}_page_{index:03d}_validation.json', entry)
                result['page_inventory'].append(entry)
        # Index each native row exactly once, after the complete page composition.
        check = inspect_trajectory(combined, source, analysis/f'{asset}_identity.sqlite', budget)
        check.update(derived_from_pages=True, bulk_byte_equality_claim=False,
                     system_id_order_guard='numeric integers or ASCII row-*; unknown representations fail',
                     derived_csv_sha256=digest(combined))
        if check['rows'] != count:
            raise ValueError('native indexed count differs from source total')
        save_new(analysis/f'{asset}_qualification.json', check); result['trajectory_checks'].append(check)
        for attachment in source['support_csv_attachments']:
            aid = attachment['asset_id']; path = raw/f'{asset}_{aid}_{attachment["name"]}'
            get(f'https://{HOST}/api/assets/{aid}', path, analysis/f'{asset}_{aid}_receipt.json', 8*1024**2)
            result['geometry_checks'].append(inspect_geometry(path))
    for source in proposal['sources']:
        source_check(source, 'after')
    return result


def execute(proposal_path, grant_path):
    proposal, grant = validate_plan(proposal_path, grant_path)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('freeze clean committed executor before acquisition')
    old = ROOT/'data/raw/mobility/tgsim/tgsim_input_qualification_20261027_v1/9uas-hf8b_trajectories.csv'
    if old.stat().st_size != 187367424 or digest(old) != OLD_RAW_SHA:
        raise ValueError('old failure identity differs; do not touch or reuse')
    raw = Path(proposal['destination_raw_directory']); analysis = Path(proposal['destination_analysis_directory'])
    raw.mkdir(parents=True, exist_ok=False); analysis.mkdir(parents=True, exist_ok=False)
    roots = accounting_roots(raw, analysis); budget = PagedBudget(proposal, roots)
    record = dict(status='running', contract='tgsim_paged_input_v1', started_at_unix=time.time(),
                  proposal_sha256=PROPOSAL_SHA, grant_sha256=digest(grant_path),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_hashes={p.name:digest(p) for p in [Path(__file__),ROOT/'scripts/prepare_tgsim_input.py']},
                  command_arguments=dict(proposal=str(proposal_path),grant=str(grant_path)),
                  first_task_body_upper_bound=OLD_BODY_BOUND, cumulative_disk_roots=[str(p) for p in roots],
                  training_frames=0,evaluation_frames=0,hidden_openings=0,automatic_retry=False,
                  byte_accounting='Returned payload plus interrupted-read bound; not exact wire traffic',
                  claim_boundary='Input qualification only; no atomic snapshot, bulk-byte equality, original RSU/cost/provider compatibility, hidden independence or scientific benefit')
    save_new(analysis/'execution_claim.json',dict(record,proposal=proposal,grant=grant))
    def timeout(signum,frame):
        raise TimeoutError('hard45minute input-only deadline')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,proposal['max_wall_clock_seconds'])
    try:
        budget.guard();record.update(acquire(proposal,raw,analysis,budget))
        budget.guard();record['status']='acquired_and_input_checked_not_science_qualified'
    except BaseException as error:
        record.update(status='failed_preserved_do_not_retry',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        record.update(elapsed_seconds=budget.clock()-budget.started,total_body_download_bytes=budget.download_bytes,
                      total_body_download_bytes_upper_bound=budget.download_bytes+budget.interrupted_body_upper_bound,
                      two_task_body_upper_bound=OLD_BODY_BOUND+budget.download_bytes+budget.interrupted_body_upper_bound,
                      http_requests=budget.requests,rows_indexed=budget.rows,cumulative_disk_bytes=disk_bytes(roots),
                      raw_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(raw.iterdir()) if p.is_file()])
        save_new(analysis/'completion.json',record)
    print(json.dumps({k:record[k] for k in ['status','http_requests','rows_indexed','total_body_download_bytes','elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,required=True);parser.add_argument('--grant',type=Path,required=True)
    args=parser.parse_args();execute(args.proposal,args.grant)
