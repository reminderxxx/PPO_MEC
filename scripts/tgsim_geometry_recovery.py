"""Granted attachment routes and immutable complete-trajectory references."""
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.parse

try:
    from scripts.prepare_tgsim_input import digest, save_new
    from scripts.prepare_tgsim_paged_input import ROOT, page_url
    from scripts.tgsim_bounded_transport import ancestor_bill
except ModuleNotFoundError:
    from prepare_tgsim_input import digest, save_new
    from prepare_tgsim_paged_input import ROOT, page_url
    from tgsim_bounded_transport import ancestor_bill

GEOMETRY_POLICY_SHA='9d06abc89b7ef016c8b83df4b07b179c123df3ddb019e3c9185535ef1fe0f2f4'
GRANT_SCOPE='same_source_geometry_endpoint_and_complete_source_recovery_zero_science'


def geometry_key(source, attachment):
    return source['asset_id']+'/'+attachment['asset_id']


def validate_geometry_grant(plan, grant):
    if (not grant.get('execution_authorized') or grant.get('scope')!=GRANT_SCOPE
            or grant.get('answer_verbatim')!='允许按新方案继续'
            or not grant.get('geometry_endpoint_change_authorized')
            or not grant.get('complete_source_readonly_reuse_authorized')
            or not grant.get('bounded_transport_retry_authorized')
            or grant.get('proposal_sha256')!=GEOMETRY_POLICY_SHA
            or plan.get('geometry_recovery_policy_sha256')!=GEOMETRY_POLICY_SHA
            or digest(grant['proposal'])!=GEOMETRY_POLICY_SHA):
        raise ValueError('explicit geometry/source-reference grant differs')
    policy=json.loads(Path(grant['proposal']).read_text())
    mapping=dict(maximum_http_requests='max_new_http_requests',
                 max_download_bytes_new_task='max_new_encoded_body_and_read_bounds',
                 max_wall_clock_seconds='max_new_executor_wall_seconds',
                 max_decoded_csv_bytes='max_new_decoded_text_bytes',max_rows_to_index='max_new_native_index_rows',
                 max_cumulative_new_disk_bytes_including_old_failure='max_cumulative_disk_bytes',
                 max_total_attempts_per_trajectory_page='max_total_attempts_per_trajectory_page',
                 max_extra_attempts_global='max_extra_transport_attempts_global')
    if any(plan[k]!=policy[v] for k,v in mapping.items()):
        raise ValueError('granted geometry remaining caps differ')
    if (plan['continuation_parent_completion']!=policy['parent_completion']
            or plan['continuation_parent_completion_sha256']!=policy['parent_completion_sha256']
            or plan['continuation_parent_plan_sha256']!=policy['parent_plan_sha256']):
        raise ValueError('geometry task must bind the approved indexed parent')
    expected={x['dataset_asset_id']+'/'+x['attachment_asset_id']:x['candidate_url']
              for x in policy['fixed_candidate_geometry_routes']}
    constructed={}
    for source in plan['sources']:
        for attachment in source['support_csv_attachments']:
            constructed[geometry_key(source,attachment)]=(
                f"https://data.transportation.gov/api/views/{source['asset_id']}/files/{attachment['asset_id']}?"+
                urllib.parse.urlencode(dict(download='true',filename=attachment['name'])))
    first=plan['sources'][0]['asset_id']
    if (expected!=constructed or plan.get('fixed_geometry_urls')!=expected
            or set(plan.get('complete_source_references',{}))!={first}
            or plan.get('complete_page_reuse') or not plan.get('geometry_first')
            or not plan.get('automatic_retry')):
        raise ValueError('fixed route/source-reference inventory differs')
    ref=plan['complete_source_references'][first]
    if digest(ref['path'])!=ref['sha256']:
        raise ValueError('frozen complete-source reference manifest differs')
    manifest=json.loads(Path(ref['path']).read_text())
    if (manifest['parent_completion_sha256']!=policy['parent_completion_sha256']
            or manifest['parent_audit_sha256']!=policy['parent_terminal_audit_sha256']
            or manifest['rows']!=policy['source1_complete_raw_candidate_rows']):
        raise ValueError('source reference is outside approved parent evidence')
    bill=ancestor_bill(plan,allow_indexed_parent=True)
    if any(plan[k]!=v for k,v in bill['caps'].items()):
        raise ValueError('all indexed-parent resource spend must be deducted')
    return bill


def reference_files(parent_plan, completion, analysis, asset):
    raw=Path(parent_plan['destination_raw_directory'])
    files={raw/x['path'] for x in completion['raw_files']}
    files.update(analysis/name for name in ('execution_claim.json','completion.json',
                 f'{asset}_qualification.json',f'{asset}_identity.sqlite',f'{asset}_system_ids.sqlite'))
    source=next(s for s in parent_plan['sources'] if s['asset_id']==asset)
    count=source['expected_count_from_frozen_metadata_not_fresh_query']
    for index in range((count+parent_plan['page_rows']-1)//parent_plan['page_rows']):
        receipt=analysis/f'{asset}_page_{index:03d}_receipt.json'
        files.update((receipt,analysis/f'{asset}_page_{index:03d}_validation.json'))
        for attempt in json.loads(receipt.read_text()).get('attempts',[]):
            files.add(Path(attempt['path']))
    for kind in ('metadata','count'):
        files.update((analysis/f'{asset}_{kind}_before.json',analysis/f'{asset}_{kind}_before_receipt.json'))
    return files


def make_reference_manifest(parent_plan_path,parent_completion_path,parent_audit_path,asset):
    """Offline metadata/hash binding only; never decompress or recreate indexes."""
    plan=json.loads(Path(parent_plan_path).read_text());analysis=Path(parent_completion_path).parent
    completion=json.loads(Path(parent_completion_path).read_text())
    audit=json.loads(Path(parent_audit_path).read_text())
    if completion['proposal_sha256']!=digest(parent_plan_path):raise ValueError('parent plan hash differs')
    count=next(s['expected_count_from_frozen_metadata_not_fresh_query'] for s in plan['sources'] if s['asset_id']==asset)
    if audit['readonly_index_counts'][asset]!={'identity':count,'system_ids':count}:
        raise ValueError('parent has no audited complete dual identity index')
    return dict(contract='tgsim_complete_source_readonly_reference_v1',asset_id=asset,rows=count,
                parent_plan=str(parent_plan_path),parent_plan_sha256=digest(parent_plan_path),
                parent_completion=str(parent_completion_path),parent_completion_sha256=digest(parent_completion_path),
                parent_audit=str(parent_audit_path),parent_audit_sha256=digest(parent_audit_path),
                files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
                       for p in sorted(reference_files(plan,completion,analysis,asset))])


def verify_reference(binding,source,budget=None):
    path=Path(binding['path'])
    if digest(path)!=binding['sha256']:raise ValueError('complete-source manifest SHA differs')
    ref=json.loads(path.read_text());asset=source['asset_id'];count=source['expected_count_from_frozen_metadata_not_fresh_query']
    if ref['asset_id']!=asset or ref['rows']!=count:raise ValueError('reference source/count differs')
    for name in ('parent_plan','parent_completion','parent_audit'):
        if digest(ref[name])!=ref[name+'_sha256']:raise ValueError('reference parent evidence differs')
    parent=json.loads(Path(ref['parent_plan']).read_text());completion=json.loads(Path(ref['parent_completion']).read_text())
    analysis=Path(ref['parent_completion']).parent;raw=Path(parent['destination_raw_directory'])
    claim=json.loads((analysis/'execution_claim.json').read_text());audit=json.loads(Path(ref['parent_audit']).read_text())
    if (completion['proposal_sha256']!=ref['parent_plan_sha256'] or claim['proposal']!=parent
            or claim['proposal_sha256']!=completion['proposal_sha256']
            or claim['source_commit']!=completion['source_commit'] or completion['rows_indexed']!=count
            or audit['source_commit']!=completion['source_commit']
            or audit['readonly_index_counts'][asset]!={'identity':count,'system_ids':count}
            or source!=next(s for s in parent['sources'] if s['asset_id']==asset)):
        raise ValueError('reference provenance or prior dual-index count differs')
    expected=reference_files(parent,completion,analysis,asset)
    files={Path(x['path']):x for x in ref['files']}
    if len(files)!=len(ref['files']) or set(files)!=expected:
        raise ValueError('complete-source file inventory omitted or duplicated')
    hashed=0
    for target,entry in files.items():
        if (target.is_symlink() or not any(target.resolve().is_relative_to(root.resolve()) for root in (raw,analysis))
                or not target.resolve().is_relative_to(ROOT.resolve())):
            raise ValueError('reference file outside original worktree source roots')
        h=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):
                if budget is not None:budget.guard()
                hashed+=len(block);h.update(block)
        if target.stat().st_size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:
            raise ValueError('referenced original file bytes/hash differ')
    # Hashing old bytes is I/O exposure, never new network, decoded rows or indexing.
    for entry in completion['raw_files']:
        frozen=files[raw/entry['path']]
        if frozen['bytes']!=entry['bytes'] or frozen['sha256']!=entry['sha256']:
            raise ValueError('reference raw manifest differs from sealed completion')
    check=json.loads((analysis/f'{asset}_qualification.json').read_text())
    if (check['rows']!=count or not check['derived_from_pages'] or check['bulk_byte_equality_claim']
            or check['derived_csv_sha256']!=files[raw/f'{asset}_derived_native.csv']['sha256']):
        raise ValueError('reference native qualification/derived file differs')
    for index,offset in enumerate(range(0,count,parent['page_rows'])):
        val=json.loads((analysis/f'{asset}_page_{index:03d}_validation.json').read_text())
        receipt=json.loads((analysis/f'{asset}_page_{index:03d}_receipt.json').read_text())
        if (val['asset_id']!=asset or val['page_index']!=index or val['offset']!=offset
                or val['rows']!=min(parent['page_rows'],count-offset)
                or val['sha256']!=files[raw/val['path']]['sha256'] or val['sha256']!=receipt['sha256']
                or receipt['status']!='complete' or receipt['url']!=page_url(source,offset,parent['page_rows'])):
            raise ValueError('reference has missing/short/changed pages')
    for suffix,table in (('identity','samples'),('system_ids','identities')):
        database=analysis/f'{asset}_{suffix}.sqlite'
        with sqlite3.connect(f'file:{database}?mode=ro&immutable=1',uri=True) as connection:
            actual=connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
        if actual!=count:raise ValueError('referenced readonly SQLite count differs')
    info=dict(asset_id=asset,rows=count,manifest=str(path),manifest_sha256=binding['sha256'],
              network_body_bytes=0,new_native_rows_indexed=0,decoded_csv_bytes=0,binary_bytes_hashed=hashed,
              native_index_path=str(analysis/f'{asset}_identity.sqlite'),
              system_id_index_path=str(analysis/f'{asset}_system_ids.sqlite'),
              derived_csv_path=str(raw/f'{asset}_derived_native.csv'),old_post_version_checked=False)
    return info,check


class CompleteSourceReference:
    def __init__(self,plan,analysis):
        self.plan=plan;self.analysis=analysis;self.checked={}

    def __call__(self,source,budget):
        asset=source['asset_id']
        if asset in self.checked:raise ValueError('complete source reference consumed twice')
        info,check=verify_reference(self.plan['complete_source_references'][asset],source,budget)
        save_new(self.analysis/f'{asset}_source_reference.json',info);self.checked[asset]=info
        return info,check
