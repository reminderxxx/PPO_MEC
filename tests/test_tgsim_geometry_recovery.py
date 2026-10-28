"""Synthetic complete-source provenance, fixed attachment routes and residual caps."""
import copy
import json
from pathlib import Path
import sqlite3

import pytest

from scripts.prepare_tgsim_input import digest,save_new,HTTPStatusFailure,InputDeadline
from scripts.prepare_tgsim_paged_input import acquire,PagedBudget,geometry_url,disk_bytes
from scripts.recover_tgsim_paged_input import NativeComposer
from scripts.tgsim_geometry_recovery import make_reference_manifest,verify_reference,CompleteSourceReference
from scripts.tgsim_bounded_transport import BoundedPageFetcher,ancestor_bill
from tests.test_tgsim_paged_preparation import setup,FixtureTransport,Response


def parent_reference(tmp_path,monkeypatch):
    from scripts import tgsim_geometry_recovery as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    folder=tmp_path/'old';folder.mkdir();plan,raw,analysis,budget=setup(folder,geometry=True)
    plan.update(destination_raw_directory=str(raw),destination_analysis_directory=str(analysis),identity_mode='opaque_unique_sqlite',max_decoded_csv_bytes=100000)
    class OldTransport(FixtureTransport):
        def open(self,request,**kwargs):
            response=super().open(request,**kwargs)
            if '/api/assets/' in request.full_url:response.status=404
            return response
    composer=NativeComposer(analysis,100000)
    with pytest.raises(HTTPStatusFailure):acquire(plan,raw,analysis,budget,opener=OldTransport(plan),append_fn=composer,gzip_pages=True)
    composer.close();assert budget.rows==3
    planpath=folder/'plan.json';save_new(planpath,plan)
    record=dict(source_commit='fixture',proposal_sha256=digest(planpath),command_arguments={'proposal':str(planpath)},training_frames=0,evaluation_frames=0,hidden_openings=0)
    save_new(analysis/'execution_claim.json',dict(record,proposal=plan))
    completion=dict(record,status='failed_preserved_do_not_retry',rows_indexed=budget.rows,
                    raw_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(raw.iterdir())])
    save_new(analysis/'completion.json',completion)
    audit=folder/'audit.json';save_new(audit,dict(source_commit='fixture',acquisition_complete=False,readonly_index_counts={'synthetic-a':{'identity':3,'system_ids':3}}))
    manifest=folder/'reference.json';save_new(manifest,make_reference_manifest(planpath,analysis/'completion.json',audit,'synthetic-a'))
    return plan,raw,analysis,manifest,dict(path=str(manifest),sha256=digest(manifest))


def snapshot(root):
    return {str(p):digest(p) for p in root.rglob('*') if p.is_file()}


def test_source_reference_readonly_queries_and_no_new_exposure_counters(tmp_path,monkeypatch):
    plan,raw,analysis,manifest,binding=parent_reference(tmp_path,monkeypatch)
    from scripts import tgsim_geometry_recovery as module
    before=snapshot(tmp_path/'old');queries=[];connect=module.sqlite3.connect
    def readonly(database_uri,**kwargs):
        assert 'mode=ro&immutable=1' in database_uri
        connection=connect(database_uri,**kwargs);connection.set_trace_callback(queries.append);return connection
    monkeypatch.setattr(module.sqlite3,'connect',readonly)
    budget=PagedBudget(plan,[raw,analysis]);info,check=verify_reference(binding,plan['sources'][0],budget)
    assert info['rows']==check['rows']==3 and info['new_native_rows_indexed']==info['network_body_bytes']==info['decoded_csv_bytes']==0
    assert budget.rows==budget.requests==budget.download_bytes==0
    assert len(queries)==2 and all(q.upper().startswith('SELECT COUNT') for q in queries)
    assert snapshot(tmp_path/'old')==before


@pytest.mark.parametrize('tamper',['omitted','duplicate','source','manifest_hash'])
def test_reference_inventory_identity_and_manifest_hash_tamper(tmp_path,monkeypatch,tamper):
    plan,_,_,manifest,binding=parent_reference(tmp_path,monkeypatch);ref=json.loads(manifest.read_text())
    if tamper=='omitted':ref['files'].pop()
    if tamper=='duplicate':ref['files'].append(ref['files'][0])
    if tamper=='source':ref['asset_id']='other'
    manifest.write_text(json.dumps(ref))
    if tamper!='manifest_hash':binding['sha256']=digest(manifest)
    with pytest.raises(ValueError):verify_reference(binding,plan['sources'][0])


@pytest.mark.parametrize('tamper',['raw','index','claim'])
def test_resnapshot_cannot_mask_changed_parent_original_or_index_count(tmp_path,monkeypatch,tamper):
    plan,raw,analysis,manifest,binding=parent_reference(tmp_path,monkeypatch)
    if tamper=='raw':(raw/'synthetic-a_derived_native.csv').write_text('changed')
    if tamper=='index':
        with sqlite3.connect(analysis/'synthetic-a_identity.sqlite') as connection:
            connection.execute('DELETE FROM samples WHERE tick=0')
    if tamper=='claim':
        path=analysis/'execution_claim.json';claim=json.loads(path.read_text());claim['source_commit']='different';path.write_text(json.dumps(claim))
    ref=make_reference_manifest(tmp_path/'old/plan.json',analysis/'completion.json',tmp_path/'old/audit.json','synthetic-a')
    manifest.write_text(json.dumps(ref));binding['sha256']=digest(manifest)
    with pytest.raises(ValueError):verify_reference(binding,plan['sources'][0])


def test_reference_hashing_obeys_total_deadline_without_indexing(tmp_path,monkeypatch):
    plan,raw,analysis,_,binding=parent_reference(tmp_path,monkeypatch);budget=PagedBudget(plan,[raw,analysis])
    budget.clock=lambda:budget.started+30
    with pytest.raises(InputDeadline):verify_reference(binding,plan['sources'][0],budget)
    assert budget.rows==budget.download_bytes==budget.requests==0


def geometry_plan(tmp_path,monkeypatch):
    old,old_raw,old_analysis,_,binding=parent_reference(tmp_path,monkeypatch)
    folder=tmp_path/'new';folder.mkdir();plan,raw,analysis,budget=setup(folder,geometry=True)
    plan.update(destination_raw_directory=str(raw),destination_analysis_directory=str(analysis),automatic_retry=True,
                max_total_attempts_per_trajectory_page=3,max_extra_attempts_global=3,identity_mode='opaque_unique_sqlite',
                max_decoded_csv_bytes=100000,geometry_recovery_policy_sha256='fixture',geometry_first=True,
                complete_source_references={'synthetic-a':binding},fixed_geometry_urls={})
    for source in plan['sources']:
        for attachment in source['support_csv_attachments']:
            plan['fixed_geometry_urls'][source['asset_id']+'/'+attachment['asset_id']]=f'https://data.transportation.gov/api/views/{source["asset_id"]}/files/{attachment["asset_id"]}?download=true&filename={attachment["name"]}'
    budget.maximum_requests=plan['maximum_http_requests']=26;budget.proposal['max_trajectory_rows_to_index']=plan['max_rows_to_index']=3
    return old_raw,old_analysis,plan,raw,analysis,budget


class GeometryTransport(FixtureTransport):
    def __init__(self,plan,mode='complete'):
        super().__init__(plan);self.mode=mode
    def open(self,request,**kwargs):
        url=request.full_url
        if '/files/' in url:
            self.calls.append(url);self.seen[url]=self.seen.get(url,0)+1
            body={'complete':b'x,y\n1,2\n','http':b'{"error":true}', 'redirect':b'redirect','json':b'{"error":true}',
                  'ragged':b'x,y\n1\n','empty':b'x,y\n'}[self.mode]
            response=Response(url,body)
            if self.mode=='http':response.status=404
            if self.mode=='redirect':response.status=302
            return response
        assert not ('synthetic-a.csv?' in url),'completed first source must never download again'
        return super().open(request,**kwargs)


def run_new(tmp_path,monkeypatch,mode='complete'):
    old_raw,old_analysis,plan,raw,analysis,budget=geometry_plan(tmp_path,monkeypatch)
    from scripts import prepare_tgsim_paged_input as producer
    original=producer.inspect_trajectory;indexed=[]
    def index_once(path,source,*args,**kwargs):
        indexed.append(source['asset_id']);assert source['asset_id']=='synthetic-b'
        return original(path,source,*args,**kwargs)
    monkeypatch.setattr(producer,'inspect_trajectory',index_once)
    ref=CompleteSourceReference(plan,analysis);composer=NativeComposer(analysis,100000)
    fetcher=BoundedPageFetcher(plan,analysis);opener=GeometryTransport(plan,mode)
    before=snapshot(tmp_path/'old')
    try:
        result=acquire(plan,raw,analysis,budget,opener=opener,append_fn=composer,gzip_pages=True,page_fetcher=fetcher,source_reference=ref)
    finally:
        composer.close();assert before==snapshot(tmp_path/'old')
    return plan,raw,analysis,budget,composer,fetcher,opener,ref,result,indexed


def test_complete_reference_pipeline_and_auditor_keep_new_and_reused_rows_separate(tmp_path,monkeypatch):
    plan,raw,analysis,budget,composer,fetcher,opener,ref,result,indexed=run_new(tmp_path,monkeypatch)
    assert budget.requests==26 and budget.rows==3 and indexed==['synthetic-b']
    assert len(result['geometry_checks'])==16 and len(result['page_inventory'])==2
    assert all(x['asset_id']=='synthetic-b' for x in result['page_inventory'])
    assert not (raw/'synthetic-a_derived_native.csv').exists() and not (analysis/'synthetic-a_identity.sqlite').exists()
    from scripts import audit_tgsim_paged_input as consumer
    proposal=tmp_path/'new/plan.json';grant=tmp_path/'new/grant.json';save_new(proposal,plan);save_new(grant,{})
    record=dict(proposal_sha256=digest(proposal),source_commit='fixture',grant_sha256=digest(grant),ancestor_bill={},
                command_arguments={'proposal':str(proposal),'grant':str(grant)},training_frames=0,evaluation_frames=0,hidden_openings=0)
    save_new(analysis/'execution_claim.json',dict(record,proposal=plan))
    completion=dict(record,**result,status='acquired_and_input_checked_not_science_qualified',http_requests=budget.requests,
                    extra_transport_attempts=0,rows_indexed=3,reused_native_rows=3,decoded_csv_bytes=composer.decoded,
                    total_body_download_bytes=budget.download_bytes,total_body_download_bytes_upper_bound=budget.download_bytes,
                    two_task_body_upper_bound=consumer.OLD_BODY_BOUND+budget.download_bytes,
                    cumulative_disk_bytes=disk_bytes([raw,analysis]),cumulative_disk_roots=[str(raw),str(analysis)],
                    raw_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(raw.iterdir())])
    path=analysis/'completion.json';save_new(path,completion)
    # Separate grant/ancestor tests cover the authorization gate. This test isolates the actual I/O consumer.
    monkeypatch.setattr(consumer,'validate_geometry_grant',lambda *args:{})
    checked=consumer.audit(analysis,expected_proposal_sha=digest(proposal))
    assert checked['acquisition_complete'] and checked['native_indexed_rows_verified']==3
    assert checked['reused_native_rows_verified']==3 and checked['network_http_requests']==26
    completion['reused_native_rows']=4;path.write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='reused native rows'):consumer.audit(analysis,expected_proposal_sha=digest(proposal))


@pytest.mark.parametrize('mode',['http','redirect','json','ragged','empty'])
def test_attachment_error_is_fatal_before_second_trajectory_and_never_retried(tmp_path,monkeypatch,mode):
    with pytest.raises((HTTPStatusFailure,ValueError)):
        run_new(tmp_path,monkeypatch,mode)
    analysis=tmp_path/'new/analysis';raw=tmp_path/'new/raw'
    receipts=[json.loads(p.read_text()) for p in analysis.glob('*_receipt.json')]
    assert len(receipts)==5 and not list(raw.glob('*page*'))
    assert (analysis/'synthetic-a_source_reference.json').exists()
    assert not (analysis/'synthetic-a_identity.sqlite').exists()


def test_all_three_ancestor_spends_include_native_index_rows_and_stricter_time_floor(tmp_path,monkeypatch):
    from tests.test_tgsim_bounded_transport import lineage_fixture
    plan=lineage_fixture(tmp_path,monkeypatch);plan.update(ancestor_bill(plan)['caps'])
    folder=tmp_path/'parent2';folder.mkdir();parentpath=folder/'plan.json';save_new(parentpath,plan)
    claim=dict(proposal=plan,proposal_sha256=digest(parentpath),source_commit='fixture',training_frames=0,evaluation_frames=0,hidden_openings=0)
    save_new(folder/'execution_claim.json',claim)
    completion=dict(claim);completion.pop('proposal')
    completion.update(status='failed_preserved_do_not_retry',rows_indexed=4023022,command_arguments={'proposal':str(parentpath)},
                      total_body_download_bytes_upper_bound=102406316,http_requests=66,elapsed_seconds=768.440340417,
                      decoded_csv_bytes=492684515,two_task_body_upper_bound=325762670)
    cp=folder/'completion.json';save_new(cp,completion)
    child=dict(plan,continuation_parent_completion=str(cp),continuation_parent_completion_sha256=digest(cp),
               continuation_parent_plan_sha256=digest(parentpath),prior_accounted_body_bound=325762670)
    child['ancestor_completions']=[dict(path=str(cp),sha256=digest(cp),plan_sha256=digest(parentpath))]+plan['ancestor_completions']
    with pytest.raises(ValueError,match='indexing'):ancestor_bill(child)
    caps=ancestor_bill(child,allow_indexed_parent=True)['caps']
    assert caps==dict(max_download_bytes_new_task=398676278,max_wall_clock_seconds=1523,
                     max_rows_to_index=1976978,maximum_http_requests=48,max_decoded_csv_bytes=551200289)


def test_legacy_or_generic_grant_cannot_enable_geometry_and_reference_mode():
    from scripts.tgsim_geometry_recovery import validate_geometry_grant
    with pytest.raises(ValueError,match='explicit geometry'):
        validate_geometry_grant({},dict(execution_authorized=True,scope='same_source_input_recovery_zero_science'))


@pytest.mark.parametrize('change',['none','url','reference_pool','extra_attempts','parent','automatic_flag'])
def test_frozen_grant_covers_exact_attachment_routes_source_and_caps(tmp_path,monkeypatch,change):
    from scripts import tgsim_geometry_recovery as module
    _,old_analysis,plan,_,_,_=geometry_plan(tmp_path,monkeypatch)
    parent_path=tmp_path/'old/plan.json';completion=old_analysis/'completion.json';audit=tmp_path/'old/audit.json'
    plan.update(continuation_parent_completion=str(completion),continuation_parent_completion_sha256=digest(completion),continuation_parent_plan_sha256=digest(parent_path))
    mapping=dict(maximum_http_requests='max_new_http_requests',max_download_bytes_new_task='max_new_encoded_body_and_read_bounds',
                 max_wall_clock_seconds='max_new_executor_wall_seconds',max_decoded_csv_bytes='max_new_decoded_text_bytes',
                 max_rows_to_index='max_new_native_index_rows',max_cumulative_new_disk_bytes_including_old_failure='max_cumulative_disk_bytes',
                 max_total_attempts_per_trajectory_page='max_total_attempts_per_trajectory_page',max_extra_attempts_global='max_extra_transport_attempts_global')
    policy={v:plan[k] for k,v in mapping.items()}
    policy.update(parent_completion=str(completion),parent_completion_sha256=digest(completion),parent_plan_sha256=digest(parent_path),parent_terminal_audit_sha256=digest(audit),source1_complete_raw_candidate_rows=3,
                  fixed_candidate_geometry_routes=[dict(dataset_asset_id=k.split('/')[0],attachment_asset_id=k.split('/')[1],candidate_url=v) for k,v in plan['fixed_geometry_urls'].items()])
    policy_path=tmp_path/'policy.json';save_new(policy_path,policy);sha=digest(policy_path)
    grant=dict(execution_authorized=True,scope=module.GRANT_SCOPE,answer_verbatim='允许按新方案继续',geometry_endpoint_change_authorized=True,
               complete_source_readonly_reuse_authorized=True,bounded_transport_retry_authorized=True,proposal_sha256=sha,proposal=str(policy_path))
    plan['geometry_recovery_policy_sha256']=sha;monkeypatch.setattr(module,'GEOMETRY_POLICY_SHA',sha)
    monkeypatch.setattr(module,'ancestor_bill',lambda *args,**kwargs:dict(caps={k:plan[k] for k in ('maximum_http_requests','max_download_bytes_new_task','max_wall_clock_seconds','max_decoded_csv_bytes','max_rows_to_index')}))
    if change=='url':plan['fixed_geometry_urls'][next(iter(plan['fixed_geometry_urls']))]='https://other.example/changed'
    if change=='reference_pool':plan['complete_source_references']['synthetic-b']=plan['complete_source_references']['synthetic-a']
    if change=='extra_attempts':plan['max_extra_attempts_global']=8
    if change=='parent':plan['continuation_parent_plan_sha256']='different'
    if change=='automatic_flag':plan['automatic_retry']=False
    if change=='none':assert module.validate_geometry_grant(plan,grant)['caps']['max_rows_to_index']==3
    else:
        with pytest.raises(ValueError):module.validate_geometry_grant(plan,grant)
