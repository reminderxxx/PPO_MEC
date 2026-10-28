"""Read-only receipt/hash/count consumer for the separately granted paged task."""
import argparse
import json
from pathlib import Path
import sqlite3

try:
    from scripts.prepare_tgsim_input import native_row_counts
    from scripts.prepare_tgsim_paged_input import (HOST, OLD_BODY_BOUND, PROPOSAL_SHA,
        disk_bytes, digest, metadata_pin, page_url, geometry_url, parse_count, save_new)
    from scripts.tgsim_bounded_transport import retryable, validate_retry_grant
    from scripts.tgsim_geometry_recovery import validate_geometry_grant, verify_reference
except ModuleNotFoundError:
    from prepare_tgsim_input import native_row_counts
    from prepare_tgsim_paged_input import (HOST, OLD_BODY_BOUND, PROPOSAL_SHA,
        disk_bytes, digest, metadata_pin, page_url, geometry_url, parse_count, save_new)
    from tgsim_bounded_transport import retryable, validate_retry_grant
    from tgsim_geometry_recovery import validate_geometry_grant, verify_reference


def audit_attempts(receipt, logical_path, raw, analysis, proposal):
    entries=receipt['attempts'];selected=receipt['selected_attempt']
    if (not proposal.get('automatic_retry') or not receipt['automatic_retry']
            or not 1<=len(entries)<=proposal['max_total_attempts_per_trajectory_page']
            or (selected is not None and selected!=len(entries))
            or (receipt['status']=='complete')!=(selected is not None)):
        raise ValueError('bounded attempt selection/status differs')
    body=uncertain=0;paths=set();last=None
    for number,entry in enumerate(entries,1):
        path=Path(entry['path']).resolve()
        expected=analysis/'attempts'/f'{logical_path.stem}_attempt_{number}.json'
        if path!=expected.resolve() or path in paths or digest(path)!=entry['sha256']:
            raise ValueError('attempt receipt SHA/inventory differs')
        paths.add(path);attempt=json.loads(path.read_text());target=Path(attempt['path']).resolve()
        # Every attempt uses a fresh full request and an independent body.
        expected_body=raw/(logical_path.name.removesuffix('_receipt.json')+'.csv.body'+f'.attempt_{number}')
        if (target!=expected_body.resolve() or attempt['url']!=receipt['url']
                or attempt['automatic_retry'] or attempt['status'] not in ('complete','failed')
                or target.stat().st_size!=attempt['bytes'] or digest(target)!=attempt['sha256']):
            raise ValueError('attempt body/URL/hash/status differs')
        if number<len(entries) and not retryable(attempt):
            raise ValueError('semantic or nontransport failure was retried')
        if selected==number and (attempt['status']!='complete' or attempt.get('status_code')!=200):
            raise ValueError('selected attempt is incomplete')
        body+=attempt['bytes'];uncertain+=attempt['interrupted_read_body_upper_bound'];last=attempt
    if (receipt['network_body_bytes']!=body or receipt['http_request_count']!=len(entries)
            or receipt['interrupted_read_body_upper_bound']!=uncertain
            or receipt['bytes']!=last['bytes'] or receipt['sha256']!=last['sha256']
            or receipt.get('encoded_wire_body_retained')!=last.get('encoded_wire_body_retained')):
        raise ValueError('attempt aggregate or selected body differs')
    return body,uncertain,len(entries),paths


def audit(analysis, *, expected_proposal_sha=None):
    expected_proposal_sha=expected_proposal_sha or PROPOSAL_SHA
    claim=json.loads((analysis/'execution_claim.json').read_text())
    completion=json.loads((analysis/'completion.json').read_text())
    attempted_rows, inserted_rows = native_row_counts(completion)
    proposal=claim['proposal'];raw=Path(proposal['destination_raw_directory'])
    if claim['proposal_sha256']!=expected_proposal_sha or completion['proposal_sha256']!=expected_proposal_sha:
        raise ValueError('paged proposal binding differs')
    original=Path(claim['command_arguments']['proposal']);grant=Path(claim['command_arguments']['grant'])
    if (digest(original)!=expected_proposal_sha or json.loads(original.read_text())!=proposal
            or digest(grant)!=claim['grant_sha256']):
        raise ValueError('frozen proposal/grant identity differs')
    if completion['source_commit']!=claim['source_commit'] or completion['grant_sha256']!=claim['grant_sha256']:
        raise ValueError('execution/completion provenance mismatch')
    for key in ('training_frames','evaluation_frames','hidden_openings'):
        if claim[key]!=0 or completion[key]!=0:
            raise ValueError('input-only boundary breached')
    geometry=bool(proposal.get('geometry_recovery_policy_sha256'))
    if not geometry and any(proposal.get(k) for k in ('fixed_geometry_urls','complete_source_references','geometry_first')):
        raise ValueError('legacy audit contract cannot change routes or reuse full source')
    if geometry or proposal.get('automatic_retry'):
        validator=validate_geometry_grant if geometry else validate_retry_grant
        lineage=validator(proposal,json.loads(grant.read_text()))
        if claim.get('ancestor_bill')!=lineage or completion.get('ancestor_bill')!=lineage:
            raise ValueError('ancestor audit bill differs')
    expected_requests={}
    references=proposal.get('complete_source_references',{});reference_checks={}
    for source in proposal['sources']:
        asset=source['asset_id']
        for phase in ('before','after'):
            expected_requests[f'{asset}_metadata_{phase}_receipt.json']=source['metadata_url']
            expected_requests[f'{asset}_count_{phase}_receipt.json']=source['row_count_url']
        if asset not in references:
            for index,offset in enumerate(range(0,source['expected_count_from_frozen_metadata_not_fresh_query'],proposal['page_rows'])):
                expected_requests[f'{asset}_page_{index:03d}_receipt.json']=page_url(source,offset,proposal['page_rows'])
        elif (analysis/f'{asset}_source_reference.json').exists():
            verified,_=verify_reference(references[asset],source)
            if json.loads((analysis/f'{asset}_source_reference.json').read_text())!=verified:
                raise ValueError('saved complete-source reference result differs')
            reference_checks[asset]=verified
        for attachment in source['support_csv_attachments']:
            aid=attachment['asset_id'];expected_requests[f'{asset}_{aid}_receipt.json']=geometry_url(proposal,source,attachment)
    seen=set();body=uncertain=0;receipts=[];network_requests=extra_attempts=reused_pages=0;attempt_paths=set()
    for path in sorted(analysis.glob('*_receipt.json')):
        receipt=json.loads(path.read_text());target=Path(receipt['path']).resolve()
        if path.name not in expected_requests or receipt['url']!=expected_requests[path.name]:
            raise ValueError('request URL/inventory outside prospective contract')
        if target.parent not in (raw.resolve(),analysis.resolve()) or target in seen:
            raise ValueError('receipt target outside task or duplicated')
        seen.add(target)
        if target.stat().st_size!=receipt['bytes'] or digest(target)!=receipt['sha256']:
            raise ValueError('received file bytes/hash differ from receipt')
        bounded=receipt.get('operation')=='bounded_trajectory_attempts'
        if receipt['status'] not in ('complete','failed') or (receipt['automatic_retry'] and not bounded):
            raise ValueError('invalid request terminal/retry status')
        reuse=receipt.get('operation')=='reuse_complete_page'
        if reuse:
            reused_pages+=1
            approved=proposal.get('complete_page_reuse',{}).get(receipt['url'])
            if approved is None or approved!=receipt['source_provenance'] or receipt.get('network_body_bytes')!=0:
                raise ValueError('unapproved page reuse or disguised network body')
            source=Path(approved['source_path']);old_path=Path(approved['source_receipt']);validation_path=Path(approved['source_validation'])
            old=json.loads(old_path.read_text());validation=json.loads(validation_path.read_text())
            parent_claim_path=Path(approved['source_execution_claim']);parent_claim=json.loads(parent_claim_path.read_text())
            parent_source=next(s for s in parent_claim['proposal']['sources'] if s['asset_id']==approved['asset_id'])
            current_source=next(s for s in proposal['sources'] if s['asset_id']==approved['asset_id'])
            if (digest(old_path)!=approved['source_receipt_sha256'] or digest(validation_path)!=approved['source_validation_sha256']
                    or old['status']!='complete' or old['url']!=receipt['url'] or old['sha256']!=receipt['sha256']
                    or digest(source)!=old['sha256'] or validation['sha256']!=old['sha256']
                    or validation['rows']!=approved['rows'] or receipt['interrupted_read_body_upper_bound']!=0
                    or digest(parent_claim_path)!=approved['source_execution_claim_sha256']
                    or parent_claim['proposal_sha256']!=approved['source_proposal_sha256']
                    or parent_source['metadata_pin']!=current_source['metadata_pin']
                    or validation['asset_id']!=approved['asset_id'] or validation['offset']!=approved['offset']):
                raise ValueError('reused complete page provenance differs')
        elif bounded:
            if '_page_' not in path.name or receipt['url'] in proposal.get('complete_page_reuse',{}):
                raise ValueError('only missing trajectory pages may retry')
            b,u,n,files=audit_attempts(receipt,path,raw,analysis,proposal)
            if attempt_paths & files:raise ValueError('attempt double counted')
            attempt_paths.update(files);body+=b;uncertain+=u;network_requests+=n;extra_attempts+=n-1
        else:
            body+=receipt['bytes'];uncertain+=receipt['interrupted_read_body_upper_bound'];network_requests+=1
        receipts.append(dict(path=path.name,status=receipt['status'],bytes=receipt['bytes'],operation='reuse_complete_page' if reuse else 'http_request'))
    if {p.resolve() for p in (analysis/'attempts').glob('*.json')}!=attempt_paths:
        raise ValueError('unaccounted/duplicated transport attempt receipts')
    if proposal.get('automatic_retry') and (extra_attempts>proposal['max_extra_attempts_global']
            or extra_attempts!=completion['extra_transport_attempts']):
        raise ValueError('global extra attempt cap/count differs')
    prior=proposal.get('prior_accounted_body_bound',OLD_BODY_BOUND)
    if (body!=completion['total_body_download_bytes'] or body+uncertain!=completion['total_body_download_bytes_upper_bound']
            or completion['two_task_body_upper_bound']!=prior+body+uncertain
            or body+uncertain>proposal['max_download_bytes_new_task']
            or completion['http_requests']!=network_requests
            or network_requests>proposal['maximum_http_requests']):
        raise ValueError('request/byte/cumulative bill mismatch or cap breach')
    inventory=completion['raw_files']
    if {p.name for p in raw.iterdir() if p.is_file()}!={item['path'] for item in inventory}:
        raise ValueError('raw inventory incomplete')
    for item in inventory:
        target=raw/item['path']
        if target.stat().st_size!=item['bytes'] or digest(target)!=item['sha256']:
            raise ValueError('raw manifest identity differs')
    success=completion['status']=='acquired_and_input_checked_not_science_qualified'
    if not success and completion['status']!='failed_preserved_do_not_retry':
        raise ValueError('task not terminal')
    indexed=0
    if success:
        if {r['path'] for r in receipts}!=set(expected_requests) or any(r['status']!='complete' for r in receipts):
            raise ValueError('complete acquisition lacks planned complete request receipts')
        pages=completion['page_inventory']
        for source in proposal['sources']:
            asset=source['asset_id'];count=source['expected_count_from_frozen_metadata_not_fresh_query']
            for phase in ('before','after'):
                meta=json.loads((analysis/f'{asset}_metadata_{phase}.json').read_text())
                if metadata_pin(meta)!=source['metadata_pin']:raise ValueError('version drift')
                parse_count(analysis/f'{asset}_count_{phase}.json',count)
            if asset in references:
                if asset not in reference_checks:raise ValueError('missing successful complete-source reference')
                if any(p['asset_id']==asset for p in pages):raise ValueError('reused source was recomposed')
                continue
            items=[p for p in pages if p['asset_id']==asset]
            offsets=list(range(0,count,proposal['page_rows']))
            if ([p['offset'] for p in items]!=offsets
                    or [p['page_index'] for p in items]!=list(range(len(offsets)))
                    or sum(p['rows'] for p in items)!=count):
                raise ValueError('missing/duplicated page inventory')
            for item in items:
                target=raw/item['path']
                validation=json.loads((analysis/f'{asset}_page_{item["page_index"]:03d}_validation.json').read_text())
                if item!=validation or digest(target)!=item['sha256']:
                    raise ValueError('page validation identity differs')
            check=json.loads((analysis/f'{asset}_qualification.json').read_text())
            with sqlite3.connect(f'file:{analysis / (asset+"_identity.sqlite")}?mode=ro',uri=True) as connection:
                actual=connection.execute('SELECT count(*) FROM samples').fetchone()[0]
            if actual!=count or check['rows']!=count or not check['derived_from_pages'] or check['bulk_byte_equality_claim']:
                raise ValueError('native identity count or derived-CSV claim differs')
            if proposal.get('identity_mode')=='opaque_unique_sqlite':
                with sqlite3.connect(f'file:{analysis / (asset+"_system_ids.sqlite")}?mode=ro',uri=True) as connection:
                    identities=connection.execute('SELECT count(*) FROM identities').fetchone()[0]
                if identities!=count:raise ValueError('opaque source-ID count mismatch')
            if digest(raw/f'{asset}_derived_native.csv')!=check['derived_csv_sha256']:
                raise ValueError('derived native CSV identity differs')
            indexed+=actual
        if indexed!=inserted_rows or attempted_rows!=indexed or attempted_rows>proposal['max_rows_to_index']:
            raise ValueError('indexed-row count cap/mismatch')
        if len(completion['geometry_checks'])!=16:
            raise ValueError('geometry inventory incomplete')
        if geometry:
            expected_geometry={str(raw/f'{s["asset_id"]}_{a["asset_id"]}_{a["name"]}')
                               for s in proposal['sources'] for a in s['support_csv_attachments']}
            checks=completion['geometry_checks']
            if {g['path'] for g in checks}!=expected_geometry:raise ValueError('geometry checks omit a fixed file')
            raw_shas={str(raw/item['path']):item['sha256'] for item in inventory}
            for check in checks:
                if (check['sha256']!=raw_shas[check['path']] or len(check['header'])<2
                        or check['data_rows']<1 or {int(k) for k in check['row_width_counts']}!={len(check['header'])}
                        or sum(check['row_width_counts'].values())!=check['data_rows']):
                    raise ValueError('geometry CSV shape/hash check differs')
        if proposal.get('max_decoded_csv_bytes') is not None and completion['decoded_csv_bytes']>proposal['max_decoded_csv_bytes']:
            raise ValueError('decoded CSV cap exceeded')
    if proposal.get('max_decoded_csv_bytes') is not None and completion['decoded_csv_bytes']>proposal['max_decoded_csv_bytes']:
        raise ValueError('decoded CSV cap exceeded')
    partial_indices={};native_rows=0
    for source in proposal['sources']:
        asset=source['asset_id'];counts={}
        for suffix,table in (('identity','samples'),('system_ids','identities')):
            database=analysis/f'{asset}_{suffix}.sqlite'
            if database.exists():
                with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as connection:
                    counts[suffix]=connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
        native_rows+=counts.get('identity',0);partial_indices[asset]=counts
    if native_rows!=inserted_rows or attempted_rows>proposal['max_rows_to_index']:
        raise ValueError('terminal native index row count differs')
    reused_native_rows=sum(info['rows'] for info in reference_checks.values())
    if reused_native_rows!=completion.get('reused_native_rows',0):
        raise ValueError('reused native rows were omitted or counted as new')
    if any((analysis/f'{asset}_identity.sqlite').exists() or (analysis/f'{asset}_system_ids.sqlite').exists()
           or (raw/f'{asset}_derived_native.csv').exists() for asset in references):
        raise ValueError('complete source was copied/redecoded/reindexed in new task')
    current_disk=disk_bytes(completion['cumulative_disk_roots'])
    if max(current_disk,completion['cumulative_disk_bytes'])>proposal['max_cumulative_new_disk_bytes_including_old_failure']:
        raise ValueError('cumulative disk cap breached')
    if completion.get('elapsed_seconds',0)>proposal['max_wall_clock_seconds']+1:
        raise ValueError('executor wall-time cap exceeded')
    return dict(status='terminal_receipt_inventory_and_bill_audited',acquisition_complete=success,
                source_commit=completion['source_commit'],request_receipts=receipts,
                network_http_requests=network_requests,reused_complete_pages=reused_pages,
                extra_transport_attempts=extra_attempts,readonly_index_counts=partial_indices,
                complete_source_references=reference_checks,reused_native_rows_verified=reused_native_rows,
                available_native_rows_verified=indexed+reused_native_rows if success else None,
                returned_body_bytes=body,interrupted_body_upper_bound=uncertain,
                two_task_body_upper_bound=prior+body+uncertain,cumulative_disk_bytes=current_disk,
                native_indexed_rows_verified=indexed if success else None,
                native_rows_examined_charged=attempted_rows,
                native_rows_inserted_verified=native_rows,
                certified_formal_or_hidden=0,training_frames=0,evaluation_frames=0,
                limitation='An audited failure is not complete input. Successful input is neither an atomic snapshot nor proof of hidden independence, original RSU/provider/cost compatibility or paper readiness.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected_proposal_sha')
    args=parser.parse_args();save_new(args.output,audit(args.analysis,expected_proposal_sha=args.expected_proposal_sha));print('paged input terminal audit saved')
