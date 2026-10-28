"""Synthetic native failures: preserve rejection, charge attempts, audit insertions."""
import copy
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.prepare_tgsim_input import (BudgetLimit, digest, inspect_trajectory,
    native_row_counters, native_row_counts, save_new)
from tests.test_tgsim_input_preparation import budget, trajectory


@pytest.mark.parametrize('value,reason', [('', 'invalid_numeric'), ('nan', 'nonfinite_numeric')])
def test_invalid_coordinate_is_rejected_and_failed_row_stays_charged(tmp_path, value, reason):
    path, source = trajectory(tmp_path, count=3, vehicles=1)
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    rows[2][3] = value
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    original = digest(path); b = budget(tmp_path); database = tmp_path/'index.sqlite'
    with pytest.raises(ValueError, match='yloc_kf at line3'):
        inspect_trajectory(path, source, database, b)
    counts = native_row_counters(b)
    assert native_row_counts(counts) == (2, 1)
    assert counts['native_input_failure'] == dict(asset_id='fixture', column='yloc_kf',
        physical_line=3, value=value, reason=reason)
    with sqlite3.connect(f'file:{database}?mode=ro&immutable=1', uri=True) as con:
        assert con.execute('SELECT count(*) FROM samples').fetchone()[0] == 1
    assert digest(path) == original


def test_duplicate_and_attempt_cap_do_not_inflate_insertions(tmp_path):
    path, source = trajectory(tmp_path, count=3, vehicles=1, duplicate=True)
    b = budget(tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        inspect_trajectory(path, source, tmp_path/'duplicate.sqlite', b)
    assert native_row_counts(native_row_counters(b)) == (2, 1)
    path, source = trajectory(tmp_path, count=3, vehicles=1)
    b = budget(tmp_path, max_trajectory_rows_to_index=2)
    with pytest.raises(BudgetLimit):
        inspect_trajectory(path, source, tmp_path/'cap.sqlite', b)
    assert native_row_counts(native_row_counters(b)) == (2, 2)


@pytest.mark.parametrize('changes', [dict(rows_examined=0), dict(rows_examined=True),
    dict(rows_indexed=-1), dict(rows_indexed=1.0), dict(native_row_accounting_version='unknown')])
def test_unknown_or_tampered_row_counters_fail_closed(changes):
    record = dict(native_row_accounting_version='attempts_and_insertions_v1', rows_examined=2, rows_indexed=1)
    record.update(changes)
    with pytest.raises(ValueError): native_row_counts(record)


def test_accounting_version_cannot_be_removed_to_hide_failed_attempts():
    assert native_row_counts(dict(rows_indexed=3)) == (3, 3)
    with pytest.raises(ValueError, match='require an accounting version'):
        native_row_counts(dict(rows_indexed=0, rows_examined=1))
    with pytest.raises(ValueError, match='missing native insertion'):
        native_row_counts(dict(native_row_accounting_version='attempts_and_insertions_v1', rows_examined=1))


def test_terminal_consumer_distinguishes_native_failure_from_audit_failure(tmp_path):
    from tests.test_tgsim_paged_preparation import setup, FixtureTransport, Response
    from scripts.prepare_tgsim_paged_input import acquire, disk_bytes
    from scripts import audit_tgsim_paged_input as consumer
    plan, raw, analysis, b = setup(tmp_path)
    plan['destination_raw_directory'] = str(raw)

    class MissingCoordinate(FixtureTransport):
        def open(self, request, **kwargs):
            response = super().open(request, **kwargs)
            if '.csv?' in request.full_url:
                body = response.read().replace(b'2,1,0.1,1,2,1', b'2,1,0.1,1,,1')
                return Response(request.full_url, body)
            return response

    with pytest.raises(ValueError, match='yloc_kf'):
        acquire(plan, raw, analysis, b, opener=MissingCoordinate(plan))
    proposal = tmp_path/'plan.json'; save_new(proposal, plan)
    grant = tmp_path/'grant.json'; save_new(grant, {})
    record = dict(proposal_sha256=digest(proposal), source_commit='synthetic', grant_sha256=digest(grant),
        training_frames=0, evaluation_frames=0, hidden_openings=0,
        command_arguments=dict(proposal=str(proposal), grant=str(grant)))
    save_new(analysis/'execution_claim.json', dict(record, proposal=plan))
    completion = dict(record, status='failed_preserved_do_not_retry', http_requests=b.requests,
        total_body_download_bytes=b.download_bytes, total_body_download_bytes_upper_bound=b.download_bytes,
        two_task_body_upper_bound=consumer.OLD_BODY_BOUND+b.download_bytes,
        cumulative_disk_bytes=disk_bytes([raw, analysis]), cumulative_disk_roots=[str(raw), str(analysis)],
        raw_files=[dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in raw.iterdir()],
        **native_row_counters(b))
    path = analysis/'completion.json'; save_new(path, completion)
    result = consumer.audit(analysis, expected_proposal_sha=digest(proposal))
    assert result['acquisition_complete'] is False
    assert result['native_rows_examined_charged'] == 2
    assert result['native_rows_inserted_verified'] == 1
    # An archived ambiguous ledger remains a failed audit; no retrospective reinterpretation.
    legacy = copy.deepcopy(completion)
    for key in ('native_row_accounting_version', 'rows_examined', 'native_input_failure'): legacy.pop(key)
    legacy['rows_indexed'] = 2; path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='native index row count differs'):
        consumer.audit(analysis, expected_proposal_sha=digest(proposal))
    completion['rows_examined'] = plan['max_rows_to_index']+1
    path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='native index row count differs'):
        consumer.audit(analysis, expected_proposal_sha=digest(proposal))


def test_ancestor_budget_deducts_rejected_attempt_even_with_no_insertions(tmp_path, monkeypatch):
    from tests.test_tgsim_bounded_transport import lineage_fixture
    from scripts.tgsim_bounded_transport import ancestor_bill
    plan = lineage_fixture(tmp_path, monkeypatch)
    path = Path(plan['continuation_parent_completion'])
    completion = json.loads(path.read_text())
    completion.update(native_row_accounting_version='attempts_and_insertions_v1', rows_examined=1, rows_indexed=0)
    path.write_text(json.dumps(completion))
    sha = digest(path); plan['continuation_parent_completion_sha256'] = sha
    plan['ancestor_completions'][0]['sha256'] = sha
    with pytest.raises(ValueError, match='native indexing differs'): ancestor_bill(plan)
    assert ancestor_bill(plan, allow_indexed_parent=True)['caps']['max_rows_to_index'] == 5999999
