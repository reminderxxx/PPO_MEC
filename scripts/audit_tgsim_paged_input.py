"""Read-only receipt/hash/count consumer for the separately granted paged task."""
import argparse
import json
from pathlib import Path
import sqlite3

try:
    from scripts.prepare_tgsim_paged_input import (HOST, OLD_BODY_BOUND, PROPOSAL_SHA,
        disk_bytes, digest, metadata_pin, page_url, parse_count, save_new)
except ModuleNotFoundError:
    from prepare_tgsim_paged_input import (HOST, OLD_BODY_BOUND, PROPOSAL_SHA,
        disk_bytes, digest, metadata_pin, page_url, parse_count, save_new)


def audit(analysis):
    claim=json.loads((analysis/'execution_claim.json').read_text())
    completion=json.loads((analysis/'completion.json').read_text())
    proposal=claim['proposal'];raw=Path(proposal['destination_raw_directory'])
    if claim['proposal_sha256']!=PROPOSAL_SHA or completion['proposal_sha256']!=PROPOSAL_SHA:
        raise ValueError('paged proposal binding differs')
    original=Path(claim['command_arguments']['proposal']);grant=Path(claim['command_arguments']['grant'])
    if (digest(original)!=PROPOSAL_SHA or json.loads(original.read_text())!=proposal
            or digest(grant)!=claim['grant_sha256']):
        raise ValueError('frozen proposal/grant identity differs')
    if completion['source_commit']!=claim['source_commit'] or completion['grant_sha256']!=claim['grant_sha256']:
        raise ValueError('execution/completion provenance mismatch')
    for key in ('training_frames','evaluation_frames','hidden_openings'):
        if claim[key]!=0 or completion[key]!=0:
            raise ValueError('input-only boundary breached')
    expected_requests={}
    for source in proposal['sources']:
        asset=source['asset_id']
        for phase in ('before','after'):
            expected_requests[f'{asset}_metadata_{phase}_receipt.json']=source['metadata_url']
            expected_requests[f'{asset}_count_{phase}_receipt.json']=source['row_count_url']
        for index,offset in enumerate(range(0,source['expected_count_from_frozen_metadata_not_fresh_query'],proposal['page_rows'])):
            expected_requests[f'{asset}_page_{index:03d}_receipt.json']=page_url(source,offset,proposal['page_rows'])
        for attachment in source['support_csv_attachments']:
            aid=attachment['asset_id'];expected_requests[f'{asset}_{aid}_receipt.json']=f'https://{HOST}/api/assets/{aid}'
    seen=set();body=uncertain=0;receipts=[]
    for path in sorted(analysis.glob('*_receipt.json')):
        receipt=json.loads(path.read_text());target=Path(receipt['path']).resolve()
        if path.name not in expected_requests or receipt['url']!=expected_requests[path.name]:
            raise ValueError('request URL/inventory outside prospective contract')
        if target.parent not in (raw.resolve(),analysis.resolve()) or target in seen:
            raise ValueError('receipt target outside task or duplicated')
        seen.add(target)
        if target.stat().st_size!=receipt['bytes'] or digest(target)!=receipt['sha256']:
            raise ValueError('received file bytes/hash differ from receipt')
        if receipt['status'] not in ('complete','failed') or receipt['automatic_retry']:
            raise ValueError('invalid request terminal/retry status')
        body+=receipt['bytes'];uncertain+=receipt['interrupted_read_body_upper_bound']
        receipts.append(dict(path=path.name,status=receipt['status'],bytes=receipt['bytes']))
    if (body!=completion['total_body_download_bytes'] or body+uncertain!=completion['total_body_download_bytes_upper_bound']
            or completion['two_task_body_upper_bound']!=OLD_BODY_BOUND+body+uncertain
            or body+uncertain>proposal['max_download_bytes_new_task']
            or completion['http_requests']!=len(receipts)
            or len(receipts)>proposal['maximum_http_requests']):
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
            if digest(raw/f'{asset}_derived_native.csv')!=check['derived_csv_sha256']:
                raise ValueError('derived native CSV identity differs')
            indexed+=actual
        if indexed!=completion['rows_indexed'] or indexed>proposal['max_rows_to_index']:
            raise ValueError('indexed-row count cap/mismatch')
        if len(completion['geometry_checks'])!=16:
            raise ValueError('geometry inventory incomplete')
    current_disk=disk_bytes(completion['cumulative_disk_roots'])
    if max(current_disk,completion['cumulative_disk_bytes'])>proposal['max_cumulative_new_disk_bytes_including_old_failure']:
        raise ValueError('cumulative disk cap breached')
    return dict(status='terminal_receipt_inventory_and_bill_audited',acquisition_complete=success,
                source_commit=completion['source_commit'],request_receipts=receipts,
                returned_body_bytes=body,interrupted_body_upper_bound=uncertain,
                two_task_body_upper_bound=OLD_BODY_BOUND+body+uncertain,cumulative_disk_bytes=current_disk,
                native_indexed_rows_verified=indexed if success else None,
                certified_formal_or_hidden=0,training_frames=0,evaluation_frames=0,
                limitation='An audited failure is not complete input. Successful input is neither an atomic snapshot nor proof of hidden independence, original RSU/provider/cost compatibility or paper readiness.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();save_new(args.output,audit(args.analysis));print('paged input terminal audit saved')
