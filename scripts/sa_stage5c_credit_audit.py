"""Bounded passive credit audit, pure stdlib. No model/tensor/optimizer operations.

A structural receipt never authenticates actual stochastic training provenance.
Synthetic fixtures and an unverified observed chain remain explicitly distinct.
One behavior UID denotes one vehicle/frame three-head row, not two head0 samples.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA='sa_credit_audit_v1'
VERSION=1
EVENTS=('behavior','credit','minibatch','update')
KINDS=('synthetic_fixture','unverified_actual_chain')
HEX40=re.compile(r'^[0-9a-f]{40}$')
HEX64=re.compile(r'^[0-9a-f]{64}$')
DEFAULT_HEAD_PASSES=['first_execution','second_cache','second_priority']
class AuditError(ValueError): pass
class AuditLimitError(AuditError): pass

def primitive(value, *, max_nodes=200000, max_depth=48):
    """Reject tensors, numpy scalars, subclasses, cycles and nonfinite numbers.

    Do not detach/convert caller data. The enabled producer owns conversion.
    """
    count=0; ancestors=set()
    def visit(x,depth):
        nonlocal count
        count+=1
        if count>max_nodes or depth>max_depth: raise AuditLimitError('primitive tree bound')
        t=type(x)
        if t in (str,int,bool,type(None)): return
        if t is float:
            if not math.isfinite(x): raise AuditError('nonfinite float')
            return
        if t not in (list,dict): raise AuditError('detached JSON primitives required')
        if id(x) in ancestors: raise AuditError('cyclic payload')
        ancestors.add(id(x))
        if t is dict:
            for k,v in x.items():
                if type(k) is not str: raise AuditError('JSON string keys required')
                visit(v,depth+1)
        else:
            for v in x: visit(v,depth+1)
        ancestors.remove(id(x))
    visit(value,0)

def canonical(value):
    primitive(value)
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()
def _text(x,name):
    if type(x) is not str or not x: raise AuditError(name+' must be nonempty string')
def _integer(x,name,minimum=0):
    if type(x) is not int or x<minimum: raise AuditError(name+' invalid integer')
def _number(x,name,minimum=None):
    if type(x) not in (int,float):raise AuditError(name+' invalid number')
    try:finite=math.isfinite(x)
    except (OverflowError,ValueError):raise AuditError(name+' outside finite numeric range')
    if not finite or (minimum is not None and x<minimum):raise AuditError(name+' invalid number')
def _vector(x,name,n=3,boolean=False):
    if type(x) is not list or len(x)!=n: raise AuditError(name+' shape')
    for v in x:
        if boolean:
            if type(v) is not bool: raise AuditError(name+' bool required')
        else:_number(v,name)
def _required(x,names):
    if type(x) is not dict:raise AuditError('record object required')
    missing=[k for k in names if k not in x]
    if missing:raise AuditError('required stable fields missing: '+','.join(missing))
def _uid(kind,parts):
    return kind+'_'+hashlib.sha256(canonical(parts)).hexdigest()
def behavior_uid(run_uid,payload):
    names=('seed','session_id','frame','vehicle_id','pass_name','request_id','node_id','decision_uid')
    return _uid('behavior',dict(run_uid=run_uid,**{k:payload[k] for k in names}))
def update_uid(run_uid,seed,update_index):return _uid('update',dict(run_uid=run_uid,seed=seed,update_index=update_index))
def credit_uid(run_uid,update,behavior):return _uid('credit',dict(run_uid=run_uid,update_uid=update,behavior_uid=behavior))
def minibatch_uid(run_uid,update,epoch,index):return _uid('minibatch',dict(run_uid=run_uid,update_uid=update,epoch=epoch,minibatch_index=index))

def _header(header,max_rows,max_bytes):
    primitive(header);_required(header,('run_uid','source_commit','provenance','evidence_kind'))
    _text(header['run_uid'],'run_uid')
    if type(header['source_commit']) is not str or not HEX40.fullmatch(header['source_commit']):raise AuditError('source commit SHA40 required')
    _required(header['provenance'],('source_run_id',));_text(header['provenance']['source_run_id'],'source_run_id')
    if header['evidence_kind'] not in KINDS:raise AuditError('unsupported evidence kind; actual provenance is never auto-certified')
    if header.get('actual_training_verified',False) is not False:raise AuditError('actual training cannot be self-certified')
    _integer(max_rows,'max_rows',2);_integer(max_bytes,'max_bytes',1)
    out=dict(header);out.update(schema=SCHEMA,schema_version=VERSION,max_rows=max_rows,max_bytes=max_bytes,actual_training_verified=False)
    return out

class _Joins:
    """Shared schema logic; consumer builds fresh state independently of sink."""
    def __init__(self,header,*,compact_joins=False):
        if type(compact_joins) is not bool:raise AuditError('compact_joins must be bool')
        self.compact_joins=compact_joins
        self.header=header;self.run=header['run_uid'];self.behaviors={};self.credits={};self.minibatches={};self.updates={};self.steps={};self.counts=Counter();self.samples=0
    def _history(self,kind,p):
        """Project only after full schema checks; return payload/log bytes stay full."""
        if not self.compact_joins:return p
        fields=dict(behavior=('seed','session_id','frame','masks','behavior_uid'),
            credit=('seed','behavior_uid','update_uid','credit_uid'),
            minibatch=('seed','update_uid','behavior_uids','credit_uids','kl_early_stop','step_observed','step_uid','minibatch_uid'),
            update=('seed','update_uid','observed_adam_steps'))
        out={key:p[key] for key in fields[kind]}
        if kind=='update':
            metadata=p.get('metadata',{})
            if type(metadata) is dict:
                if 'optimizer_step_counts' in metadata:
                    counts=metadata['optimizer_step_counts']
                    out['metadata']=dict(optimizer_step_counts={key:counts[key] for key in ('main','surrogate')})
            else:
                # Only its type is needed to preserve legacy downstream handling
                # of non-object metadata; never retain its arbitrary contents.
                out['metadata']=type(metadata)()
        return out
    def event(self,kind,payload,*,commit=True):
        if kind not in EVENTS:raise AuditError('unknown event type')
        primitive(payload)
        import copy
        p=copy.deepcopy(payload);_required(p,('seed',));_integer(p['seed'],'seed')
        if kind=='behavior':
            _required(p,('session_id','frame','vehicle_id','pass_name','request_id','node_id','decision_uid','candidates','masks','action_indices','old_log_probs','causal_prefix'))
            for n in ('session_id','vehicle_id','request_id','node_id','decision_uid'):_text(p[n],n)
            _integer(p['frame'],'frame')
            if p['pass_name']!='two_pass_behavior':raise AuditError('one combined three-head behavior required; separate pass not a second execution sample')
            if p.get('head_passes',DEFAULT_HEAD_PASSES)!=DEFAULT_HEAD_PASSES:raise AuditError('head pass attribution differs')
            if p.get('behavior_origin',self.header['evidence_kind'])!=self.header['evidence_kind']:raise AuditError('behavior origin/header mismatch; no relabelled greedy replay')
            if type(p['causal_prefix']) is not dict:raise AuditError('causal prefix dict required')
            for name in ('candidates','masks'):
                if type(p[name]) is not list or len(p[name])!=3:raise AuditError(name+' exactly3 heads required')
            _vector(p['action_indices'],'action_indices');_vector(p['old_log_probs'],'old_log_probs')
            for j,(c,m,a,logp) in enumerate(zip(p['candidates'],p['masks'],p['action_indices'],p['old_log_probs'])):
                if type(c) is not list or not c or type(m) is not list or len(c)!=len(m) or any(type(v) is not bool for v in m) or not any(m):raise AuditError('candidate/mask shape/empty')
                _integer(a,'action index')
                if a>=len(m) or not m[a]:raise AuditError('masked/outside behavior action')
                if logp>1e-6:raise AuditError('positive categorical old log probability')
            expected=behavior_uid(self.run,p);field='behavior_uid';target=self.behaviors
            for ref in p['causal_prefix'].get('behavior_uids',[]):
                if ref not in self.behaviors:raise AuditError('causal prefix references future/unknown behavior')
                old=self.behaviors[ref]
                if old['seed']!=p['seed'] or old['session_id']!=p['session_id'] or old['frame']>p['frame']:raise AuditError('causal prefix crosses seed/session/future frame')
        else:
            _required(p,('update_index','update_uid'));_integer(p['update_index'],'update_index',1)
            u=update_uid(self.run,p['seed'],p['update_index'])
            if p['update_uid']!=u:raise AuditError('update UID/run/seed/index mismatch')
            if kind=='credit':
                if u in self.updates:raise AuditError('credit after finished update')
                _required(p,('behavior_uid','T','T_team','fleet','B','raw_adv','normalized_adv','active_heads','global_value_target'))
                if p['behavior_uid'] not in self.behaviors:raise AuditError('unknown behavior credit foreign key')
                b=self.behaviors[p['behavior_uid']]
                if b['seed']!=p['seed']:raise AuditError('credit seed mismatch')
                for n in ('T','T_team','global_value_target'):_number(p[n],n)
                _integer(p['fleet'],'fleet',1)
                for n in ('B','raw_adv','normalized_adv'):_vector(p[n],n)
                _vector(p['active_heads'],'active_heads',boolean=True)
                if p['active_heads']!=[sum(m)>1 for m in b['masks']]:raise AuditError('active/forced mask mismatch')
                # Preserve supplied saved precision; arithmetic correctness is a
                # producer/source audit, not a tolerance-based scientific gate.
                if p.get('credit_units','physical_MC/10_then_fleet')!='physical_MC/10_then_fleet':raise AuditError('unsupported credit unit declaration')
                expected=credit_uid(self.run,u,p['behavior_uid']);field='credit_uid';target=self.credits
            elif kind=='minibatch':
                _required(p,('epoch','minibatch_index','behavior_uids','credit_uids','rng_before_sha256','rng_after_sha256','kl','step_observed','step_uid','kl_early_stop'))
                _integer(p['epoch'],'epoch');_integer(p['minibatch_index'],'minibatch_index');_number(p['kl'],'KL')
                for n in ('rng_before_sha256','rng_after_sha256'):
                    if type(p[n]) is not str or not HEX64.fullmatch(p[n]):raise AuditError('RNG SHA64 required')
                if type(p['behavior_uids']) is not list or not p['behavior_uids'] or type(p['credit_uids']) is not list or len(p['behavior_uids'])!=len(p['credit_uids']):raise AuditError('sample join shape')
                for b,c in zip(p['behavior_uids'],p['credit_uids']):
                    if b not in self.behaviors or c not in self.credits:raise AuditError('unknown minibatch sample foreign key')
                    cr=self.credits[c]
                    if cr['behavior_uid']!=b or cr['update_uid']!=u or cr['seed']!=p['seed']:raise AuditError('minibatch credit join mismatch')
                for n in ('step_observed','kl_early_stop'):
                    if type(p[n]) is not bool:raise AuditError('step/KL bool required')
                if p['step_observed']:
                    _text(p['step_uid'],'step_uid')
                    if p['step_uid'] in self.steps:raise AuditError('duplicate observed optimizer step UID')
                    if p['kl_early_stop']:raise AuditError('KL rejected minibatch cannot report step')
                elif p['step_uid'] is not None:raise AuditError('unobserved optimizer step must be None')
                if u in self.updates:raise AuditError('minibatch after finished update')
                expected=minibatch_uid(self.run,u,p['epoch'],p['minibatch_index']);field='minibatch_uid';target=self.minibatches
            else:
                _required(p,('completed_updates_before','completed_updates_after','observed_adam_steps','step_uids','adam_delta_summary','delta_observation'))
                for n in ('completed_updates_before','completed_updates_after','observed_adam_steps'):_integer(p[n],n)
                if p['completed_updates_after']<p['completed_updates_before']:raise AuditError('update counter reversed')
                if p['completed_updates_after']!=p['update_index']:raise AuditError('update index/counter mismatch')
                metadata=p.get('metadata',{})
                if 'optimizer_step_counts' in metadata:
                    op=metadata['optimizer_step_counts'];_required(op,('main','surrogate'))
                    for n in ('main','surrogate'):_integer(op[n],n+' optimizer calls')
                    if op['main']!=p['observed_adam_steps']:raise AuditError('main optimizer count mismatch')
                    if 'total_adam_calls' in metadata:
                        _integer(metadata['total_adam_calls'],'total Adam calls')
                        if metadata['total_adam_calls']!=op['main']+op['surrogate']:raise AuditError('total Adam calls mismatch')
                if p['delta_observation']!='producer_supplied':raise AuditError('delta must be observed and supplied, never reconstructed')
                if type(p['step_uids']) is not list or len(set(p['step_uids']))!=len(p['step_uids']):raise AuditError('step UID list')
                if not any(m['update_uid']==u for m in self.minibatches.values()):raise AuditError('update has no observed minibatch, including no KL rejection')
                eligible={q['behavior_uid'] for q in self.credits.values() if q['update_uid']==u}
                if 'eligible_behavior_uids' in metadata:
                    if type(metadata['eligible_behavior_uids']) is not list or len(set(metadata['eligible_behavior_uids']))!=len(metadata['eligible_behavior_uids']) or set(metadata['eligible_behavior_uids'])!=eligible:raise AuditError('eligible credit UID coverage mismatch')
                if 'eligible_behavior_rows' in metadata:
                    _integer(metadata['eligible_behavior_rows'],'eligible rows')
                    if metadata['eligible_behavior_rows']!=len(eligible):raise AuditError('eligible row count mismatch')
                actual=[s for s,v in self.steps.items() if v==u]
                if set(p['step_uids'])!=set(actual) or len(actual)!=p['observed_adam_steps']:raise AuditError('observed step accounting mismatch')
                if type(p['adam_delta_summary']) is not dict or not p['adam_delta_summary']:raise AuditError('delta group summary required')
                for group,d in p['adam_delta_summary'].items():
                    _required(d,('l2','max_abs','parameter_count','changed_parameter_count'))
                    for n in ('l2','max_abs'):_number(d[n],n,0)
                    for n in ('parameter_count','changed_parameter_count'):_integer(d[n],n)
                    if d['changed_parameter_count']>d['parameter_count']:raise AuditError('delta count shape')
                expected=u;field='update_uid';target=self.updates
        if field in p and p[field]!=expected:raise AuditError('unstable/tampered '+field)
        if expected in target:raise AuditError('duplicate '+field)
        p[field]=expected
        if commit:
            target[expected]=self._history(kind,p)
            if kind=='minibatch':
                self.samples+=len(p['behavior_uids'])
                if p['step_observed']:self.steps[p['step_uid']]=p['update_uid']
            self.counts[kind]+=1
        return p,expected
    def pending(self):
        refs={p['update_uid'] for p in self.credits.values()}|{p['update_uid'] for p in self.minibatches.values()}
        return sorted(refs-set(self.updates))
    def completion(self):
        if any(self.counts[k]==0 for k in EVENTS):raise AuditError('complete needs behavior/credit/minibatch/update; empty header is not actual behavior')
        if self.pending():raise AuditError('unfinished update references')
        sampled={b for p in self.minibatches.values() for b in p['behavior_uids']}
        credited={p['behavior_uid'] for p in self.credits.values()}
        if set(self.behaviors)!=credited or not sampled.issubset(credited):raise AuditError('complete behavior-credit coverage or sample reference differs')
        if any(not any(m['update_uid']==u for m in self.minibatches.values()) for u in self.updates):raise AuditError('update without observed minibatch')

class AuditSink:
    def __init__(self,output_root=None,*,enable=False,header=None,max_rows=10000,max_bytes=67108864,compact_joins=False):
        self.enabled=enable is True;self.closed=False
        if enable is not True:
            if enable is not False:raise AuditError('enable must be bool')
            return
        import threading
        self.owner_thread=threading.get_ident();self.busy=False
        self.header=_header(header,max_rows,max_bytes);self.rows=0;self.bytes=0;self.seq=0;self.digest=hashlib.sha256();self.joins=_Joins(self.header,compact_joins=compact_joins)
        self.max_rows=max_rows;self.max_bytes=max_bytes
        self.reserve=len(canonical(self._record('terminal',self._terminal('complete_fixture','producer_closed_partial',max_rows,max_bytes,dict.fromkeys(EVENTS,max_rows),'f'*64),max_rows)))+64
        head=self._record('header',self.header,0);encoded=canonical(head)
        if len(encoded)+self.reserve>max_bytes:raise AuditLimitError('header/terminal exceed byte budget')
        self.root=Path(output_root);self.root.mkdir(parents=True,exist_ok=False);self.path=self.root/'events.jsonl';self.file=self.path.open('xb')
        self._append(encoded)
    def _record(self,event_type,payload,seq=None):return dict(schema=SCHEMA,schema_version=VERSION,seq=self.seq if seq is None else seq,event_type=event_type,run_uid=self.header['run_uid'],payload=payload)
    def _terminal(self,status,reason,rows,bytes_,counts,digest):return dict(status=status,reason=reason,rows_before=rows,bytes_before=bytes_,event_counts=counts,data_sha256=digest,actual_training_verified=False)
    def _append(self,b):
        self.file.write(b);self.file.flush();self.rows+=1;self.bytes+=len(b);self.seq+=1;self.digest.update(b)
    def _finish(self,status,reason):
        payload=self._terminal(status,reason,self.rows,self.bytes,dict(self.joins.counts),self.digest.hexdigest());b=canonical(self._record('terminal',payload))
        if self.rows+1>self.max_rows or self.bytes+len(b)>self.max_bytes:raise AuditLimitError('terminal reservation invariant')
        self._append(b);self.file.close();self.closed=True;return status
    def _thread(self):
        import threading
        if threading.get_ident()!=self.owner_thread:raise AuditError('single-thread audit sink required')
    def emit(self,event_type,payload=None,*,payload_factory=None):
        if not self.enabled:return None
        self._thread()
        if self.closed or self.busy:raise AuditError('sink closed or recursive emit')
        self.busy=True
        try:
            if payload_factory is not None:
                if payload is not None or not callable(payload_factory):raise AuditError('choose payload or factory')
                payload=payload_factory()
            # O(current record), never clone growing history. Preflight cannot
            # mutate joins; the single-thread guard prevents concurrent changes.
            p,uid=self.joins.event(event_type,payload,commit=False);b=canonical(self._record(event_type,p))
            if self.rows+2>self.max_rows or self.bytes+len(b)+self.reserve>self.max_bytes:raise AuditLimitError('row/byte budget before write')
            self._append(b);self.joins.event(event_type,p,commit=True);return uid
        except BaseException:
            self._finish('failed','validation_or_limit_error');raise
        finally:self.busy=False
    def finish(self,*,complete=False):
        if not self.enabled:return None
        self._thread()
        if self.closed or self.busy:raise AuditError('sink closed or recursive finish')
        if type(complete) is not bool:raise AuditError('complete must be bool')
        if not complete:return self._finish('partial','producer_closed_partial')
        try:self.joins.completion()
        except BaseException:
            self._finish('failed','incomplete_joins_or_behavior');raise
        return self._finish('complete_fixture' if self.header['evidence_kind']=='synthetic_fixture' else 'complete_unverified','producer_requested_complete')
    close=finish
    def __enter__(self):return self
    def __exit__(self,typ,value,tb):
        if not self.enabled or self.closed:return False
        self._thread()
        if self.busy:raise AuditError('recursive context finish')
        self._finish('failed' if typ else 'partial','producer_exception' if typ else 'producer_closed_partial');return False


def _pairs(items):
    out={}
    for k,v in items:
        if k in out:raise AuditError('duplicate JSON key')
        out[k]=v
    return out

def validate(path,*,max_file_bytes=268435456,compact_joins=False):
    """Read independently. Schema/joins do not authenticate stochastic origin.

    Partial files retain prefix accounting and pending refs. Malformed complete
    lines are invalid; a missing or truncated terminal is partial, never complete.
    Opt-in compact joins retain only later join/summary fields after validating
    each complete payload. Schema, raw log and returned report are unchanged.
    """
    path=Path(path);path=path/'events.jsonl' if path.is_dir() else path
    result=dict(status='partial',structural_complete=False,actual_training_verified=False,credit_correctness='UNVERIFIED_by_schema',source_stochastic_origin='UNVERIFIED',errors=[],pending_updates=[],event_counts={},unique_behavior_rows=0,sample_occurrences=0,synthetic_fixture=None)
    digest=hashlib.sha256();rows_=0;bytes_=0;terminal=None;joins=None;header=None
    try:
        _integer(max_file_bytes,'consumer max_file_bytes',1)
        if type(compact_joins) is not bool:raise AuditError('compact_joins must be bool')
        with path.open('rb') as f:
            for line in f:
                bytes_+=len(line)
                if bytes_>max_file_bytes:raise AuditLimitError('consumer file byte bound')
                if terminal is not None:raise AuditError('records after terminal')
                if header is not None and (bytes_>header['max_bytes'] or rows_+1>header['max_rows']):raise AuditLimitError('declared bound exceeded even on partial tail')
                if not line.endswith(b'\n'):
                    result['reason']='truncated_tail';break
                try:record=json.loads(line,object_pairs_hook=_pairs,parse_constant=lambda x:(_ for _ in ()).throw(AuditError('nonfinite JSON constant')))
                except (json.JSONDecodeError,UnicodeDecodeError) as exc:raise AuditError('malformed complete JSON line') from exc
                primitive(record);_required(record,('schema','schema_version','seq','event_type','run_uid','payload'))
                if record['schema']!=SCHEMA or type(record['schema_version']) is not int or record['schema_version']!=VERSION or type(record['seq']) is not int or record['seq']!=rows_:raise AuditError('schema/version/sequence mismatch')
                if rows_==0:
                    if record['event_type']!='header':raise AuditError('header first required')
                    h=record['payload'];header=_header(h,h['max_rows'],h['max_bytes'])
                    if header!=h or record['run_uid']!=header['run_uid']:raise AuditError('header identity mismatch')
                    joins=_Joins(header,compact_joins=compact_joins);result['synthetic_fixture']=header['evidence_kind']=='synthetic_fixture'
                else:
                    if record['run_uid']!=header['run_uid']:raise AuditError('cross-run envelope')
                    if record['event_type']=='terminal':
                        t=record['payload'];_required(t,('status','reason','rows_before','bytes_before','event_counts','data_sha256','actual_training_verified'))
                        _integer(t['rows_before'],'terminal rows');_integer(t['bytes_before'],'terminal bytes')
                        if type(t['event_counts']) is not dict or any(k not in EVENTS for k in t['event_counts']):raise AuditError('terminal count keys')
                        for v in t['event_counts'].values():_integer(v,'terminal event count')
                        if type(t['data_sha256']) is not str or not HEX64.fullmatch(t['data_sha256']):raise AuditError('terminal SHA64')
                        if t['rows_before']!=rows_ or t['bytes_before']!=bytes_-len(line) or t['event_counts']!=dict(joins.counts) or t['data_sha256']!=digest.hexdigest() or t['actual_training_verified'] is not False:raise AuditError('terminal accounting/hash mismatch')
                        if t['status'] not in ('failed','partial','complete_fixture','complete_unverified'):raise AuditError('terminal status')
                        if t['status'].startswith('complete'):
                            joins.completion()
                            expected='complete_fixture' if result['synthetic_fixture'] else 'complete_unverified'
                            if t['status']!=expected:raise AuditError('fixture/actual evidence status mismatch')
                            result['structural_complete']=True
                        terminal=t;result['status']=t['status']
                    else:joins.event(record['event_type'],record['payload'])
                rows_+=1;digest.update(line)
                if rows_>header['max_rows'] or bytes_>header['max_bytes']:raise AuditLimitError('declared stream bound exceeded')
        if joins is None:result['reason']='missing_header'
        elif terminal is None:result.setdefault('reason','missing_terminal')
    except (AuditError,OSError,KeyError,TypeError,OverflowError,RecursionError,ValueError) as exc:
        result['status']='invalid';result['structural_complete']=False;result['errors'].append(type(exc).__name__+': '+str(exc))
    if joins is not None:
        main=sum(p['observed_adam_steps'] for p in joins.updates.values())
        observed=[p.get('metadata',{}).get('optimizer_step_counts') for p in joins.updates.values()]
        all_counts=bool(observed) and all(v is not None for v in observed)
        surrogate=sum(v['surrogate'] for v in observed) if all_counts else None
        sampled={b for m in joins.minibatches.values() for b in m['behavior_uids']}
        never=sorted(set(joins.behaviors)-sampled)
        result.update(sampled_unique_behavior_rows=len(sampled),never_sampled_behavior_uids=never,never_sampled_behavior_count=len(never),kl_early_stop_minibatches=sum(m['kl_early_stop'] for m in joins.minibatches.values()),event_counts=dict(joins.counts),pending_updates=joins.pending(),unique_behavior_rows=len(joins.behaviors),sample_occurrences=joins.samples,observed_main_optimizer_steps=main,observed_surrogate_optimizer_steps=surrogate,observed_total_adam_calls=None if surrogate is None else main+surrogate,optimizer_count_scope='producer supplied observed counts only; main+surrogate calls are not independent minibatches')
    result.update(rows=rows_,bytes=bytes_,observed_terminal=terminal,operation_scope='read/JSON/hash/schema joins only; no env/model/forward/backward/optimizer; repeated samples are not new behavior rows')
    return result


def main(argv=None):
    """Read-only structural validation. No producer activation or run launcher."""
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,help='Existing audit directory or events.jsonl')
    parser.add_argument('--max_file_bytes',type=int,default=268435456)
    args=parser.parse_args(argv)
    result=validate(args.root,max_file_bytes=args.max_file_bytes)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0 if result['structural_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
