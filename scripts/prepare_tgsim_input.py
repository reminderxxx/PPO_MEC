"""One create-only, explicitly granted TGSIM acquisition; input checks, zero science."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

PROPOSAL_SHA = '40fcb20715a8f3462b90af0dd0209b66986cf40845a13c287500921418a0143b'
ASSETS = ('9uas-hf8b', '7zjf-a4zf')
HOST = 'data.transportation.gov'
ROOT = Path(__file__).resolve().parents[1]


class BudgetLimit(RuntimeError):
    """A hard task limit, never a recoverable per-source quality rejection."""


class InputDeadline(TimeoutError):
    """Total execution deadline, distinct from a retryable socket timeout."""


class HTTPStatusFailure(ValueError):
    """A retained diagnostic entity is never successful input."""


def open_including_http_error(opener, request, timeout):
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as response:
        # This is the existing response, not another request. Consume only a
        # bounded diagnostic entity, retain its headers and then fail globally.
        return response


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def validate_url(url):
    u = urllib.parse.urlsplit(url)
    if u.scheme != 'https' or u.hostname != HOST or u.username or u.password or u.port:
        raise ValueError('only the approved official HTTPS host is allowed')
    return url


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Budget:
    def __init__(self, proposal, directories, clock=time.monotonic):
        self.proposal, self.directories, self.clock = proposal, directories, clock
        self.started = clock()
        self.download_bytes = self.rows = 0
        self.rows_inserted = 0
        self.native_input_failure = None
        self.interrupted_body_upper_bound = 0

    def guard(self):
        if self.clock() - self.started >= self.proposal['max_wall_clock_seconds']:
            raise InputDeadline('total input-task wall-clock cap reached')
        disk = sum(p.stat().st_size for root in self.directories for p in root.rglob('*') if p.is_file())
        # Reserve8MiB for terminal receipts/errors; never fill the hard2GiB cap.
        if disk > self.proposal['max_new_disk_bytes'] - 8 * 1024 * 1024:
            raise BudgetLimit('new disk budget reached')

    def read_size(self, per_file_remaining):
        self.guard()
        remaining = self.proposal['max_download_bytes'] - self.download_bytes - self.interrupted_body_upper_bound
        if remaining <= 0 or per_file_remaining <= 0:
            raise BudgetLimit('download byte budget reached')
        return min(65536, remaining, per_file_remaining)


def native_row_counters(budget):
    """Rejected rows still spend the attempt budget; INSERTs are counted separately."""
    return dict(native_row_accounting_version='attempts_and_insertions_v1',
                rows_examined=budget.rows, rows_indexed=budget.rows_inserted,
                native_input_failure=budget.native_input_failure)


def native_row_counts(record):
    """Return charged attempts and insertions, without reinterpreting old records."""
    indexed = record.get('rows_indexed', 0)
    version = record.get('native_row_accounting_version')
    if version is None:
        if 'rows_examined' in record:
            raise ValueError('native attempt counts require an accounting version')
        attempted = indexed
    elif version == 'attempts_and_insertions_v1':
        if 'rows_indexed' not in record:
            raise ValueError('missing native insertion count')
        attempted = record['rows_examined']
    else:
        raise ValueError('unknown native row accounting version')
    if (type(attempted) is not int or type(indexed) is not int
            or not 0 <= indexed <= attempted):
        raise ValueError('invalid native row attempt/insertion counts')
    return attempted, indexed


def fetch(url, destination, budget, receipt_path, *, maximum_bytes=None, opener=None, accept_gzip=False, read_timeout_seconds=30):
    """Exactly one request. Keep partial bytes and an error receipt on any failure."""
    validate_url(url)
    if not 0 < read_timeout_seconds <= 120:
        raise ValueError('read timeout must be bounded by120seconds')
    maximum_bytes = maximum_bytes or budget.proposal['max_download_bytes']
    opener = opener or urllib.request.build_opener(OfficialRedirect())
    record = dict(url=url, path=str(destination), status='requesting', bytes=0,
                  started_at_unix=time.time(), automatic_retry=False,
                  interrupted_read_body_upper_bound=0)
    h = hashlib.sha256()
    try:
        record['failure_phase'] = 'guard'
        budget.guard()
        timeout = min(float(read_timeout_seconds), budget.proposal['max_wall_clock_seconds'] - (budget.clock() - budget.started))
        request = urllib.request.Request(url, headers={'User-Agent': 'PPO-MEC-input-qualification/1.0', 'Accept-Encoding': 'gzip' if accept_gzip else 'identity'})
        record['failure_phase'] = 'open'
        with destination.open('xb') as out, open_including_http_error(opener, request, timeout) as response:
            record['failure_phase'] = 'headers'
            record.update(final_url=validate_url(response.geturl()), status_code=response.status,
                          response_headers=dict(response.headers))
            diagnostic = response.status != 200
            if diagnostic:
                maximum_bytes = min(maximum_bytes, 65536)
                record.update(diagnostic_body_only=True, diagnostic_capture_cap_bytes=maximum_bytes)
            if not diagnostic and 'text/html' in response.headers.get('Content-Type', '').lower():
                raise ValueError('HTML is not an approved metadata/CSV response')
            allowed_encodings = ('identity', '', 'gzip') if accept_gzip else ('identity', '')
            if response.headers.get('Content-Encoding', 'identity').lower() not in allowed_encodings:
                raise ValueError('compressed transfer outside this plain-CSV contract')
            record['encoded_wire_body_retained'] = response.headers.get('Content-Encoding','identity').lower()=='gzip'
            length = response.headers.get('Content-Length')
            length = int(length) if length is not None else None
            if not diagnostic and length is not None and (length > maximum_bytes or length > budget.proposal['max_download_bytes'] - budget.download_bytes - budget.interrupted_body_upper_bound):
                    raise BudgetLimit('declared body exceeds remaining byte budget')
            while length is None or record['bytes'] < length:
                if diagnostic and record['bytes'] >= maximum_bytes:
                    break
                record['failure_phase'] = 'guard'
                size = budget.read_size(maximum_bytes - record['bytes'])
                record['interrupted_read_body_upper_bound'] = size
                # HTTPResponse.read(size) can discard a received partial chunk
                # when its outer IncompleteRead exposes an empty payload.
                # read1 returns received payload incrementally before the next
                # truncated-chunk error. Retain uncertainty if a read raises.
                record['failure_phase'] = 'read'
                block = response.read1(size)
                record['interrupted_read_body_upper_bound'] = 0
                if not block:
                    break
                budget.download_bytes += len(block)
                record['bytes'] += len(block)
                h.update(block)
                record['failure_phase'] = 'write'
                out.write(block)
            record['failure_phase'] = 'body_validation'
            if diagnostic:
                record['diagnostic_entity_complete'] = (record['bytes'] == length if length is not None else record['bytes'] < maximum_bytes)
                raise HTTPStatusFailure(f'HTTP{response.status}; diagnostic entity/headers retained, no retry')
            if length is not None and record['bytes'] != length:
                raise ValueError('truncated declared response')
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        if isinstance(error, urllib.error.URLError):
            record['transport_reason_type'] = type(error.reason).__name__
        raise
    finally:
        budget.interrupted_body_upper_bound += record['interrupted_read_body_upper_bound']
        record.update(sha256=h.hexdigest(), finished_at_unix=time.time(), cumulative_download_bytes=budget.download_bytes,
                      cumulative_body_bytes_upper_bound=budget.download_bytes + budget.interrupted_body_upper_bound,
                      receipt_contract='tgsim_returned_body_and_http_error_entity_v3',
                      byte_accounting='bytes/cumulative_download_bytes count returned payload; an interrupted read has an explicit conservative additional bound, not an exact network byte total')
        save_new(receipt_path, record)
    return record


def metadata_pin(meta):
    return dict(asset_id=meta['id'], rows_updated_at=meta['rowsUpdatedAt'],
                column_specs=[{k: c.get(k) for k in ('name', 'fieldName', 'dataTypeName', 'description')}
                              for c in meta['columns']],
                support_attachment_identities=[{'assetId': a['assetId'], 'name': a['name']}
                    for a in meta['metadata']['attachments'] if a['name'].lower().endswith('.csv')])


def validate_plan(proposal_path, grant_path):
    if digest(proposal_path) != PROPOSAL_SHA:
        raise ValueError('proposal does not match the exact human-approved SHA')
    proposal = json.loads(proposal_path.read_text())
    grant = json.loads(grant_path.read_text())
    if not grant.get('execution_authorized') or grant.get('proposal_sha256') != PROPOSAL_SHA:
        raise ValueError('explicit grant missing or mismatched')
    if grant.get('answer_verbatim') != '允许按该范围准备数据':
        raise ValueError('human answer binding missing')
    if tuple(s['asset_id'] for s in proposal['sources']) != ASSETS:
        raise ValueError('unapproved asset pool')
    for key, cap in [('max_download_bytes', 512 * 1024**2), ('max_new_disk_bytes', 2 * 1024**3),
                     ('max_wall_clock_seconds', 2700), ('max_trajectory_rows_to_index', 6000000)]:
        if proposal[key] != grant['caps'][key] or proposal[key] > cap:
            raise ValueError('budget/grant mismatch')
    if len([a for s in proposal['sources'] for a in s['support_csv_attachments']]) != 16:
        raise ValueError('geometry inventory mismatch')
    for key in ('scientific_training_frames_authorized', 'scientific_evaluation_frames_authorized'):
        if proposal[key] != 0:
            raise ValueError('this executor grants no science')
    for key in ('destination_raw_directory', 'destination_analysis_directory'):
        path = Path(proposal[key])
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.exists():
            raise ValueError('destination must be fresh and within the approved worktree')
    return proposal, grant


def feasible_starts(segments, width=480, minimum_vehicles=8):
    """All candidate start ranges; a vehicle must exist at every tick of a window."""
    events = defaultdict(int)
    for start, end in segments:
        if end - start + 1 >= width:
            events[start] += 1
            events[end - width + 2] -= 1
    points = sorted(events)
    count = 0
    ranges = []
    for i, tick in enumerate(points[:-1]):
        count += events[tick]
        if count >= minimum_vehicles:
            end = points[i + 1] - 1
            if ranges and ranges[-1][1] + 1 == tick:
                ranges[-1][1] = end
            else:
                ranges.append([tick, end])
    return ranges


def temporal_packing(ranges, width=480, unused_gap=480):
    next_start = None
    total = 0
    for lo, hi in ranges:
        start = max(lo, next_start) if next_start is not None else lo
        if start <= hi:
            n = (hi - start) // (width + unused_gap) + 1
            total += n
            next_start = start + n * (width + unused_gap)
    return total


def inspect_trajectory(path, source, database, budget):
    """Input-only exact0.1s key index. Fail visibly; never repair/drop bad rows."""
    numeric = [c['name'] for c in source['metadata_pin']['column_specs'] if c['dataTypeName'] == 'number']
    expected = [c['name'] for c in source['metadata_pin']['column_specs']]
    connection = sqlite3.connect(database)
    connection.execute('PRAGMA journal_mode=OFF')
    connection.execute('PRAGMA temp_store=MEMORY')
    connection.execute('CREATE TABLE samples(run INTEGER, vehicle INTEGER, tick INTEGER, PRIMARY KEY(run,vehicle,tick)) WITHOUT ROWID')
    bounds = {}; rows = 0
    try:
        with path.open(encoding='utf-8-sig', newline='') as data:
            reader = csv.DictReader(data)
            if reader.fieldnames != expected:
                raise ValueError('trajectory CSV header differs from frozen column names/order')
            for row in reader:
                if budget.rows >= budget.proposal['max_trajectory_rows_to_index']:
                    raise BudgetLimit('cumulative row cap reached before indexing another row')
                budget.rows += 1; rows += 1
                if None in row or any(row.get(k) is None for k in expected):
                    raise ValueError(f'ragged row at line{reader.line_num}')
                numbers = {}
                for key in numeric:
                    try:
                        value = float(row[key])
                    except (ValueError, OverflowError) as error:
                        budget.native_input_failure = dict(asset_id=source['asset_id'],
                            column=key, physical_line=reader.line_num,
                            value=row[key][:128], reason='invalid_numeric')
                        raise ValueError(f'{source["asset_id"]}: invalid numeric field {key} at line{reader.line_num}') from error
                    if not math.isfinite(value):
                        budget.native_input_failure = dict(asset_id=source['asset_id'],
                            column=key, physical_line=reader.line_num,
                            value=row[key][:128], reason='nonfinite_numeric')
                        raise ValueError(f'{source["asset_id"]}: nonfinite numeric field {key} at line{reader.line_num}')
                    numbers[key] = value
                t = Decimal(row['time']); tick = t * 10
                run, vehicle = Decimal(row['run_index']), Decimal(row['id'])
                if any(x != x.to_integral_value() for x in (tick, run, vehicle)) or t < 0:
                    raise ValueError(f'off-grid time or noninteger identity at line{reader.line_num}')
                run, vehicle, tick = int(run), int(vehicle), int(tick)
                try:
                    connection.execute('INSERT INTO samples VALUES(?,?,?)', (run, vehicle, tick))
                except sqlite3.IntegrityError as error:
                    raise ValueError(f'duplicate source/run/vehicle/time at line{reader.line_num}') from error
                budget.rows_inserted += 1
                x, y = numbers['xloc_kf'], numbers['yloc_kf']
                b = bounds.setdefault(run, [x, x, y, y])
                b[:] = [min(b[0], x), max(b[1], x), min(b[2], y), max(b[3], y)]
                if rows % 10000 == 0:
                    connection.commit(); budget.guard()
        connection.commit()
        if not rows:
            raise ValueError('empty trajectory CSV')
        result = {}; segments = defaultdict(list)
        previous = None; start = end = None
        for run, vehicle, tick in connection.execute('SELECT run,vehicle,tick FROM samples ORDER BY run,vehicle,tick'):
            item = result.setdefault(run, dict(vehicles=0, rows=0, minimum_tick=tick, maximum_tick=tick,
                                               missing_vehicle_ticks=0, gap_intervals=0, max_consecutive_ticks=0))
            item['rows'] += 1; item['minimum_tick'] = min(item['minimum_tick'], tick); item['maximum_tick'] = max(item['maximum_tick'], tick)
            if previous is None or previous != (run, vehicle) or tick != end + 1:
                if previous is not None:
                    result[previous[0]]['max_consecutive_ticks'] = max(result[previous[0]]['max_consecutive_ticks'], end-start+1)
                    if end-start+1 >= 480: segments[previous[0]].append((start, end))
                    if previous == (run, vehicle):
                        item['missing_vehicle_ticks'] += tick-end-1; item['gap_intervals'] += 1
                if previous != (run, vehicle):item['vehicles'] += 1
                start = tick
            previous = (run, vehicle); end = tick
            if item['rows'] % 10000 == 0:budget.guard()
        if previous is not None:
            result[previous[0]]['max_consecutive_ticks'] = max(result[previous[0]]['max_consecutive_ticks'], end-start+1)
            if end-start+1 >= 480:segments[previous[0]].append((start, end))
        for run, item in result.items():
            ranges = feasible_starts(segments[run])
            item.update(bounding_box_native_m=bounds[run], native_480_tick_start_ranges=ranges,
                        start_positions=sum(hi-lo+1 for lo,hi in ranges),
                        temporal_upper_bound_with_480_unused_ticks=temporal_packing(ranges),
                        independent_acquisitions='UNVERIFIED', certified_formal_or_hidden=0)
        return dict(asset_id=source['asset_id'], status='native_key_grid_and_numeric_checks_pass', rows=rows,
                    nominal_tick_seconds='0.1', session_samples=480, endpoint_span_seconds='47.9',
                    runs=result, input_only_exposure='All indexed rows/ranges consumed for input qualification; not hidden',
                    geometry_gate='UNQUALIFIED_DIFFERENT_SITE_REFERENCE_FRAME_AND_UNITS',
                    warning='Temporal capacities do not prove independent acquisitions, original RSU coverage, provider/cost compatibility or formal/hidden eligibility')
    finally:
        connection.commit(); connection.close()


def inspect_geometry(path, *, strict_rectangular=False):
    with path.open(encoding='utf-8-sig', newline='') as data:
        reader = csv.reader(data,strict=strict_rectangular)
        header = next(reader)
        rows = 0; widths = defaultdict(int)
        for row in reader:
            rows += 1; widths[len(row)] += 1
    if strict_rectangular and (len(header)<2 or rows<1 or set(widths)!={len(header)}):
        raise ValueError('geometry must be a nonempty rectangular CSV with at least two columns')
    return dict(path=str(path), sha256=digest(path), header=header, data_rows=rows,
                row_width_counts=dict(widths), coordinate_mapping='NOT_CERTIFIED',
                limitation='CSV inventory/shape only; names or lane centerlines do not georeference original RSUs')


def execute(proposal_path, grant_path):
    proposal, grant = validate_plan(proposal_path, grant_path)
    raw = Path(proposal['destination_raw_directory']); analysis = Path(proposal['destination_analysis_directory'])
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('freeze clean committed executor before acquisition')
    raw.mkdir(parents=True, exist_ok=False); analysis.mkdir(parents=True, exist_ok=False)
    budget = Budget(proposal, [raw, analysis])
    receipt = dict(status='running', started_at_unix=time.time(), proposal_sha256=PROPOSAL_SHA,
                   grant_sha256=digest(grant_path), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   script_sha256=digest(Path(__file__)), trajectory_checks=[], geometry_checks=[],
                   training_frames=0, evaluation_frames=0, hidden_openings=0)
    save_new(analysis/'execution_claim.json', dict(receipt, proposal=proposal, grant=grant))
    def timeout(signum, frame):raise TimeoutError('hard45minute input-task deadline')
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, proposal['max_wall_clock_seconds'])
    try:
        for source in proposal['sources']:
            asset = source['asset_id']
            meta = analysis/f'{asset}_metadata_before.json'
            fetch(source['metadata_url'], meta, budget, analysis/f'{asset}_metadata_before_receipt.json', maximum_bytes=1024*1024)
            if metadata_pin(json.loads(meta.read_text())) != source['metadata_pin']:
                raise ValueError(f'{asset}: source version/schema/attachments changed')
        for source in proposal['sources']:
            asset = source['asset_id']
            fetch(source['trajectory_csv_url'], raw/f'{asset}_trajectories.csv', budget, analysis/f'{asset}_trajectory_receipt.json')
            for attachment in source['support_csv_attachments']:
                aid = attachment['asset_id']
                destination = raw/f'{asset}_{aid}_{attachment["name"]}'
                fetch(f'https://{HOST}/api/assets/{aid}', destination, budget, analysis/f'{asset}_{aid}_receipt.json', maximum_bytes=8*1024*1024)
                receipt['geometry_checks'].append(inspect_geometry(destination))
        for source in proposal['sources']:
            asset = source['asset_id']; meta = analysis/f'{asset}_metadata_after.json'
            fetch(source['metadata_url'], meta, budget, analysis/f'{asset}_metadata_after_receipt.json', maximum_bytes=1024*1024)
            if metadata_pin(json.loads(meta.read_text())) != source['metadata_pin']:
                raise ValueError(f'{asset}: source changed during acquisition')
        for source in proposal['sources']:
            asset = source['asset_id']
            try:
                check = inspect_trajectory(raw/f'{asset}_trajectories.csv', source, analysis/f'{asset}_identity.sqlite', budget)
            except (ValueError, csv.Error, ArithmeticError) as error:
                check = dict(asset_id=asset, status='INPUT_REJECTED_NO_REPAIR', error_type=type(error).__name__, error=str(error), certified_formal_or_hidden=0)
            save_new(analysis/f'{asset}_qualification.json', check); receipt['trajectory_checks'].append(check)
        receipt['status'] = 'acquired_and_input_checked_not_science_qualified'
    except BaseException as error:
        receipt.update(status='failed_preserved_do_not_retry', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(elapsed_seconds=time.monotonic()-budget.started, total_body_download_bytes=budget.download_bytes,
                       total_body_download_bytes_upper_bound=budget.download_bytes+budget.interrupted_body_upper_bound,
                       body_byte_accounting='successfully returned payload plus explicit interrupted-read upper bound; not an exact network byte total',
                       rows_examined=budget.rows, new_disk_bytes=sum(p.stat().st_size for root in [raw,analysis] for p in root.rglob('*') if p.is_file()),
                       raw_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in raw.iterdir() if p.is_file()])
        receipt.update(native_row_counters(budget))
        save_new(analysis/'completion.json', receipt)
    print(json.dumps({k:receipt[k] for k in ['status','elapsed_seconds','total_body_download_bytes','rows_examined','training_frames','evaluation_frames']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    execute(args.proposal, args.grant)
