"""Single-use paired Helmet6 quality pilot; no training, downloads or retries."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'Look at this cropped head region. Is a motorcycle helmet worn? Reply with exactly one label: HELMET or NO_HELMET. No explanation.'
PARSER = 'strip surrounding whitespace only; exact HELMET or NO_HELMET; every other output format is wrong'
CAPS = {'model_processes':2,'processor_load':2,'base_load':2,'adapter_load':1,'activation':1,'generate':12,
        'new_tokens_per_generate':32,'wall_seconds':1800,'producer_seconds':1200,'child_seconds':600,
        'consumer_seconds':600,'artifact_bytes':134217728,'log_bytes':16777216,'http':0,'training':0,
        'fitting':0,'environment_episodes':0}
GENERATION = {'do_sample':False,'num_beams':1,'max_new_tokens':32,'use_cache':True}
_BUDGET = None

VERSIONS = {'python':'3.9.6','torch':'2.8.0','transformers':'4.49.0','peft':'0.14.0','accelerate':'1.2.1','safetensors':'0.5.3'}


def require(value, message):
    if not value: raise ValueError(message)


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def boot():
    return subprocess.check_output(['/usr/sbin/sysctl','-n','kern.bootsessionuuid'],text=True).strip()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Budget:
    """One shared artifact tree, with stdout allocation reserved before launch."""
    def __init__(self, roots, artifact_limit=CAPS['artifact_bytes'], log_limit=CAPS['log_bytes'], reserve=1048576):
        self.roots=tuple(Path(p).absolute() for p in roots)
        require(self.roots and len(set(self.roots))==len(self.roots),'unique storage roots')
        self.roots=tuple(p for p in self.roots if not any(p!=q and p.is_relative_to(q) for q in self.roots))
        self.artifact_limit,self.log_limit,self.reserve=artifact_limit,log_limit,reserve
        self.lease_path=None;self.lease_bytes=0
    def usage(self):
        total=logs=0
        for root in self.roots:
            for p in root.rglob('*'):
                require(not p.is_symlink(),'no artifact symlink')
                if p.is_file():
                    amount=p.stat().st_size;total+=amount
                    if p.suffix.lower() in ('.json','.jsonl','.log','.csv','.txt'):logs+=amount
        return total,logs
    def check(self, extra=0, is_log=False, terminal=False):
        total,logs=self.usage()
        lease=max(0,self.lease_bytes-(self.lease_path.stat().st_size if self.lease_path and self.lease_path.exists() else 0))
        reserve=0 if terminal else self.reserve
        require(total+lease+extra+reserve<=self.artifact_limit,'shared artifact hard cap/reserve')
        require(logs+lease+(extra if is_log else 0)+reserve<=self.log_limit,'shared metadata/log hard cap/reserve')
        return total,logs
    def allocate_log(self,path):
        total,logs=self.check()
        size=min(7*1024*1024,self.artifact_limit-total-self.reserve,self.log_limit-logs-self.reserve)
        require(size>0,'remaining stdout allocation')
        self.lease_path,self.lease_bytes=Path(path),size
        return size


def create(path, value, terminal=False):
    data=value if isinstance(value,bytes) else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if _BUDGET is not None:
        require(any(Path(path).absolute().is_relative_to(root) for root in _BUDGET.roots),'output within charged tree')
        _BUDGET.check(len(data),Path(path).suffix.lower() in ('.json','.jsonl','.log','.csv','.txt'),terminal)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('xb') as f:f.write(data)


def png_crop(source, xywh):
    from PIL import Image
    require(len(xywh)==4 and all(type(v) in (int,float) and float(v).is_integer() for v in xywh),'exact integral native ROI')
    x,y,w,h=map(int,xywh)
    with Image.open(source) as raw:
        require(x>=0 and y>=0 and w>=32 and h>=32 and x+w<=raw.width and y+h<=raw.height,'native ROI bounds')
        rgb=raw.convert('RGB').crop((x,y,x+w,y+h))
        pixels=rgb.tobytes()
    clean=Image.frombytes('RGB',(w,h),pixels)
    output=io.BytesIO();clean.save(output,format='PNG',compress_level=6,optimize=False)
    return output.getvalue(),hashlib.sha256(pixels).hexdigest(),[w,h]


def runtime_check(expected):
    actual={'python':platform.python_version()}
    for name in expected['versions']:
        if name!='python':actual[name]=importlib.metadata.version(name)
    require(actual==expected['versions'],'frozen runtime package versions')
    require(sys.executable==expected['python'] and expected['device']=='cpu' and expected['dtype']=='float32' and
            expected['intra_threads']==expected['inter_threads']==1 and expected['seed']==0 and
            expected['local_files_only'] is True and expected['trust_remote_code'] is False,'frozen CPU/offline runtime')
    return actual


def source_check(plan):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['source_commit'],'source commit')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'clean source worktree')
    for name,value in plan['source_sha'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts and not (ROOT/p).is_symlink() and digest(ROOT/p)==value,'source SHA')


def worker_schema(spec):
    require(set(spec)=={'arm','runtime','generation','prompt','resources','inputs','case_order','source_commit','source_sha','caps'},'label-free worker schema')
    require(spec['arm'] in ('base','helmet') and spec['generation']==GENERATION and spec['prompt']==PROMPT and spec['caps']==CAPS,'worker frozen parameters')
    require(spec['case_order']==['q%02d'%i for i in range(6)] and len(spec['inputs'])==6,'fixed six-case order')
    for row,case in zip(spec['inputs'],spec['case_order']):
        require(set(row)=={'case_id','crop_path','crop_sha','crop_rgb_sha','width','height'} and row['case_id']==case,'worker input cannot contain GT or source filename')
        require(Path(row['crop_path']).name==case+'.png','opaque image basename')


def verify_resources(resources):
    records={}
    for name,spec in resources.items():
        records[name]={}
        for relative,expected in spec['files'].items():
            path=Path(spec['path'])/relative
            require(not path.is_symlink() and path.is_file() and path.stat().st_size==expected['bytes'] and digest(path)==expected['sha256'],'resource bytes/SHA '+relative)
            stat=path.stat();records[name][relative]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino,'sha256':expected['sha256']}
    return records


class Ledger:
    def __init__(self,path,arm):
        self.path,self.arm,self.counts=Path(path),arm,{}
        self.stream=self.path.open('x')
    def event(self,value):
        data=json.dumps(value,allow_nan=False)+'\n'
        if _BUDGET is not None:_BUDGET.check(len(data.encode()),True)
        self.stream.write(data);self.stream.flush();os.fsync(self.stream.fileno())
    @contextmanager
    def operation(self,op,case_id=None):
        limits={'processor_load':1,'base_load':1,'adapter_load':1 if self.arm=='helmet' else 0,'activate':1 if self.arm=='helmet' else 0,'generate':6}
        require(self.counts.get(op,0)<limits[op],'operation budget before call')
        self.counts[op]=self.counts.get(op,0)+1
        common={'op':op,'arm':self.arm,'pid':os.getpid()}
        if case_id is not None:common['case_id']=case_id
        self.event(dict(common,edge='begin',clock=clock()))
        try:yield
        except BaseException:
            self.event(dict(common,edge='end',clock=clock(),status='FAIL'));raise
        else:self.event(dict(common,edge='end',clock=clock(),status='PASS'))
    def close(self):self.stream.close()


def model_worker(path,expected,directory):
    global _BUDGET
    require(digest(path)==expected,'worker plan bytes')
    spec=load(path);worker_schema(spec);source_check(spec)
    directory=Path(directory);require(not directory.exists(),'fresh worker output')
    _BUDGET=Budget([directory.parent.parent])
    if os.environ.get('HELMET_LOG_LEASE_PATH'):
        _BUDGET.lease_path=Path(os.environ['HELMET_LOG_LEASE_PATH']);_BUDGET.lease_bytes=int(os.environ['HELMET_LOG_LEASE_BYTES'])
    _BUDGET.check()
    directory.mkdir(parents=True)
    import resource
    file_limit=min(resource.getrlimit(resource.RLIMIT_FSIZE)[1],7*1024*1024)
    resource.setrlimit(resource.RLIMIT_FSIZE,(file_limit,file_limit))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    create(directory/'worker_start.json',{'pid':os.getpid(),'worker_plan_sha':expected,'start_clock':clock(),'arm':spec['arm'],'runtime':spec['runtime']})
    ledger=Ledger(directory/'operations.jsonl',spec['arm'])
    try:
        runtime_check(spec['runtime'])
        create(directory/'resource_verification.json',verify_resources(spec['resources']))
        import torch
        import peft
        from transformers import AutoModelForVision2Seq,AutoProcessor
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
        with ledger.operation('processor_load'):
            processor=AutoProcessor.from_pretrained(spec['resources']['base']['path'],local_files_only=True,trust_remote_code=False)
        with ledger.operation('base_load'):
            model=AutoModelForVision2Seq.from_pretrained(spec['resources']['base']['path'],torch_dtype=torch.float32,local_files_only=True,trust_remote_code=False)
            model.to('cpu').eval()
        identity={'is_peft':False,'adapter_modules':0,'base_weight_sha256':spec['resources']['base']['weight_sha256']}
        if spec['arm']=='helmet':
            from peft import PeftModel
            with ledger.operation('adapter_load'):
                model=PeftModel.from_pretrained(model,spec['resources']['adapter']['path'],adapter_name='helmet',is_trainable=False,local_files_only=True)
                model.eval()
            with ledger.operation('activate'):
                model.set_adapter('helmet');status=vars(model.get_model_status())
                require(status.get('active_adapters')==['helmet'] and status.get('enabled') is True,'actual active adapter status')
                parameters=[]
                for name,tensor in model.named_parameters():
                    if '.helmet.' not in name:continue
                    require(tensor.dtype==torch.float32 and tensor.device.type=='cpu','actual loaded adapter CPU float32')
                    value=tensor.detach().cpu().contiguous()
                    parameters.append({'safetensors_key':name.replace('.helmet.','.'),'loaded_name':name,'dtype':'float32','shape':list(value.shape),'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()})
                require(parameters,'actual adapter parameter set')
                create(directory/'loaded_adapter_parameters.json',{'active_adapter':'helmet','adapters_enabled':True,'parameters':parameters})
                identity.update(is_peft=True,adapter_modules=len(parameters),adapter_weight_sha256=spec['resources']['adapter']['weight_sha256'],active_adapters=status['active_adapters'],available_adapters=status['available_adapters'],enabled=status['enabled'])
        else:
            count=sum('lora' in type(module).__name__.lower() or 'peft' in type(module).__module__.lower() for module in model.modules())
            require(count==0,'base arm has no adapter modules')
        effective=model.generation_config.to_dict();effective.update(spec['generation'])
        identity.update(effective_generation_config=effective,model_training=bool(model.training))
        create(directory/'model_identity.json',identity)
        from PIL import Image
        for row in spec['inputs']:
            require(digest(row['crop_path'])==row['crop_sha'],'frozen crop bytes')
            with Image.open(row['crop_path']) as image:
                image=image.convert('RGB');require(list(image.size)==[row['width'],row['height']] and hashlib.sha256(image.tobytes()).hexdigest()==row['crop_rgb_sha'],'frozen crop pixels')
                messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':spec['prompt']}]}]
                rendered=processor.apply_chat_template(messages,add_generation_prompt=True)
                tensors=dict(processor(text=rendered,images=[image],return_tensors='pt'))
            descriptors={}
            for name,tensor in tensors.items():
                require(tensor.device.type=='cpu','CPU input tensor')
                v=tensor.detach().cpu().contiguous()
                descriptors[name]={'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()}
            require('pixel_values' in tensors and tensors['pixel_values'].numel()>0,'real nonempty visual input')
            case=directory/'cases'/row['case_id'];case.mkdir(parents=True,exist_ok=False)
            started=clock();input_ids=[int(x) for x in tensors['input_ids'][0].tolist()]
            with torch.inference_mode():
                create(case/'claim.json',{'case_id':row['case_id'],'crop_sha':row['crop_sha'],'crop_rgb_sha':row['crop_rgb_sha'],'prompt':spec['prompt'],'rendered_prompt':rendered,'input_ids':input_ids,'tensors':descriptors,'generation':spec['generation'],'started':started,'inference_mode':bool(torch.is_inference_mode_enabled()),'grad_enabled':bool(torch.is_grad_enabled())})
            try:
                with ledger.operation('generate',row['case_id']):
                    with torch.inference_mode():returned=model.generate(**tensors,**spec['generation'])
                    all_ids=[int(x) for x in returned[0].detach().cpu().tolist()]
                    require(all_ids[:len(input_ids)]==input_ids,'decoder output contains input prefix')
                    new_ids=all_ids[len(input_ids):];require(len(new_ids)<=32,'new token bound')
                    text=processor.batch_decode([new_ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                create(case/'receipt.json',{'case_id':row['case_id'],'status':'PASS','started':started,'finished':clock(),'returned_ids':all_ids,'output_ids':new_ids,'raw_text':text})
            except BaseException as error:
                create(case/'receipt.json',{'case_id':row['case_id'],'status':'FAIL','started':started,'finished':clock(),'exception':repr(error)})
                raise
        versions=runtime_check(spec['runtime'])
        origins={name:sys.modules[name].__file__ for name in ('torch','transformers','peft') if name in sys.modules}
        for name,value in origins.items():require(value==spec['runtime']['module_origins'][name],'module origin')
        create(directory/'worker_terminal.json',{'status':'PASS','arm':spec['arm'],'pid':os.getpid(),'worker_plan_sha':expected,'finished_clock':clock(),'versions':versions,'module_origins':origins,'threads':[torch.get_num_threads(),torch.get_num_interop_threads()],'device':'cpu','dtype':'float32'})
    except BaseException as error:
        create(directory/'worker_terminal.json',{'status':'FAIL','arm':spec['arm'],'pid':os.getpid(),'worker_plan_sha':expected,'finished_clock':clock(),'exception':repr(error),'traceback':traceback.format_exc()},terminal=True)
        raise
    finally:ledger.close()


def supervise(command,log_path,seconds,control_path=None):
    # This loop supervises resources only; it never reads predictions/performance.
    started=clock();terminated=False;error=None;process=None;code=None
    log_path=Path(log_path)
    if control_path is None:control_path=log_path.with_suffix('.process.json')
    control_path=Path(control_path)
    create(control_path.with_name(control_path.stem+'_intent.json'),{'command':command,'started':started,'seconds':seconds})
    maximum=_BUDGET.allocate_log(log_path) if _BUDGET is not None else 7*1024*1024
    env=os.environ.copy();env.update(HELMET_LOG_LEASE_PATH=str(log_path),HELMET_LOG_LEASE_BYTES=str(maximum))
    def file_limit():
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE,(maximum,maximum))
    def group_exists(pid):
        try:os.killpg(pid,0);return True
        except ProcessLookupError:return False
        except PermissionError:return None  # transient orphan/zombie group; never treat as absent
    def send(pid,sig):
        try:os.killpg(pid,sig)
        except ProcessLookupError:pass
        except PermissionError:pass  # final group check still must prove absent
    saved_handler=saved_timer=None
    with log_path.open('xb') as log:
        try:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,preexec_fn=file_limit)
            while True:
                code=process.poll()
                if _BUDGET is not None:_BUDGET.check()
                if code is not None:break
                require(clock()-started<seconds,'child deadline')
                time.sleep(min(0.1,max(0.001,seconds-(clock()-started))))
        except BaseException as exc:error=repr(exc)
        finally:
            # Clear the outer timer while PG cleanup and its receipt are durable.
            # The 4s cleanup allowance is included in the enclosing deadline.
            old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
            saved_timer=signal.getitimer(signal.ITIMER_REAL)
            signal.setitimer(signal.ITIMER_REAL,0)
            saved_handler=signal.signal(signal.SIGALRM,signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
            cleanup_start=clock()
            try:
                if process is not None:
                    state=group_exists(process.pid)
                    if state is not False:
                        terminated=True;send(process.pid,signal.SIGTERM)
                        until=clock()+1.5
                        while group_exists(process.pid) is not False and clock()<until:
                            process.poll();time.sleep(0.02)
                        if group_exists(process.pid) is not False:send(process.pid,signal.SIGKILL)
                    # Reap the direct child even when group operations cannot certify absence.
                    try:code=process.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        process.kill();code=process.wait(timeout=0.5)
                    until=clock()+0.4
                    while group_exists(process.pid) is not False and clock()<until:time.sleep(0.02)
                    require(group_exists(process.pid) is False,'process group cleanup incomplete/unverified')
            except BaseException as exc:
                error=(error+'; ' if error else '')+'cleanup: '+repr(exc)
                if process is not None:
                    try:
                        if process.poll() is None:process.kill()
                        code=process.wait(timeout=0.1)
                    except BaseException as reap_error:error+='; reap: '+repr(reap_error)
            if _BUDGET is not None:_BUDGET.lease_path=None;_BUDGET.lease_bytes=0
            record={'pid':process.pid if process is not None else None,'returncode':code,'started':started,'finished':clock(),
                    'terminated_group':terminated,'exception':error,'stdout_allocation_bytes':maximum}
            create(control_path,record,terminal=True)
            signal.signal(signal.SIGALRM,saved_handler)
            if saved_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(0.001,saved_timer[0]-(clock()-cleanup_start)),saved_timer[1])
    return record


def preflight(plan):
    require(plan['storage_roots']==[str(Path(plan['raw_root']).parent)] and
            Path(plan['audit_output']).parent==Path(plan['raw_root']).parent,'one charged pilot tree')
    require([x['case_id'] for x in plan['cases']]==['q%02d'%i for i in range(6)],'six frozen case order')
    for arm in ('base','helmet'):
        info=plan['worker_plans'][arm];require(digest(info['path'])==info['sha256'],'worker plan SHA before launch')
        worker=load(info['path']);worker_schema(worker)
        require(worker['arm']==arm and all(worker[k]==plan[k] for k in ('runtime','resources','generation','prompt','source_commit','source_sha')),'complete worker/main binding before model calls')
        require(worker['inputs']==[{k:row[k] for k in ('case_id','crop_path','crop_sha','crop_rgb_sha','width','height')} for row in plan['cases']],'worker exact opaque inputs')
    for row in plan['cases']:require(Path(row['crop_path'])==Path(plan['raw_root'])/'crops'/(row['case_id']+'.png'),'fixed charged crop path')
    ref=plan['resource_reference'];require(digest(ref['path'])==ref['sha256'],'original resource plan SHA')
    old=load(ref['path'])['resources']
    for name,oldname in (('base','base'),('adapter','B')):
        native=dict(old[oldname]);native['files']={r['name']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in native['files']}
        if name=='adapter':native['name']='helmet'
        require(plan['resources'][name]==native,'original full resource identity')


def run(plan_path,expected,grant_path):
    global _BUDGET
    require(digest(plan_path)==expected,'exact plan SHA')
    plan,grant=load(plan_path),load(grant_path)
    require(plan['new_caps']==CAPS and plan['generation']==GENERATION and plan['prompt']==PROMPT and plan['parser']==PARSER,'fixed pilot schema')
    require(grant['plan_sha256']==expected and grant['allow_quality_pilot'] is True and grant['generate_calls']==12 and grant['arms']==2 and grant['training']==0 and grant['downloads']==0,'exact new quality grant')
    source_check(plan);preflight(plan)
    _BUDGET=Budget(plan['storage_roots']);_BUDGET.check()
    root=Path(plan['raw_root']);require(not root.exists(),'single fresh root');root.mkdir(parents=True)
    create(root/'plan.json',Path(plan_path).read_bytes());create(root/'grant.json',Path(grant_path).read_bytes())
    claim={'plan_sha':expected,'grant_sha256':digest(grant_path),'boot_uuid':boot(),'start_clock':clock(),'at':datetime.now(timezone.utc).isoformat()};create(root/'claim.json',claim)
    previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('1200s producer deadline')))
    signal.setitimer(signal.ITIMER_REAL,1194)
    try:
        ref=plan['input_reference'];require(digest(ref['audit_path'])==ref['audit_sha'] and digest(ref['terminal_path'])==ref['terminal_sha'],'accepted input reference')
        require(load(ref['audit_path'])['input_qualified'] is True and load(ref['terminal_path'])['sha256']==ref['audit_sha'],'input terminal qualification')
        for row in plan['cases']:
            require(digest(row['source_path'])==row['source_sha'],'original image SHA')
            data,pixels,size=png_crop(row['source_path'],row['bbox_xywh'])
            require(hashlib.sha256(data).hexdigest()==row['crop_sha'] and pixels==row['crop_rgb_sha'] and size==[row['width'],row['height']],'exact output-blind crop transform')
            create(row['crop_path'],data)
        for arm in ('base','helmet'):
            info=plan['worker_plans'][arm];require(digest(info['path'])==info['sha256'],'worker plan SHA')
            spec=load(info['path']);worker_schema(spec)
            command=[plan['runtime']['python'],'-u',str(ROOT/'scripts/run_helmet_quality_pilot.py'),'--worker_plan',info['path'],'--worker_plan_sha',info['sha256'],'--worker_output',str(root/arm)]
            remaining=claim['start_clock']+1190-clock();require(remaining>0,'remaining producer deadline')
            record=supervise(command,root/('worker_'+arm+'.log'),min(596,remaining),root/('worker_'+arm+'_control.json'))
            record['worker_plan_sha']=info['sha256'];create(root/('worker_'+arm+'_process.json'),record,terminal=True)
            require(record['returncode']==0 and not record['terminated_group'] and record['exception'] is None,'worker infrastructure failure, no retry')
            require(boot()==claim['boot_uuid'],'boot changed')
            _BUDGET.check()
        create(root/'producer_complete.json',{'finished_clock':clock(),'workers':2,'model_calls':12})
    except BaseException as error:
        create(root/'producer_failure.json',{'finished_clock':clock(),'exception':repr(error),'traceback':traceback.format_exc(),'no_retry':True},terminal=True)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    remaining=claim['start_clock']+1800-clock();require(remaining>0,'consumer shared deadline')
    command=[plan['runtime']['python'],'-u',str(ROOT/'scripts/audit_helmet_quality_pilot.py'),'--plan',str(plan_path),'--plan_sha256',expected,'--root',str(root),'--output',plan['audit_output']]
    control=supervise(command,root/'consumer.log',min(596,remaining),root/'consumer_process.json')
    require(control['returncode']==0 and not control['terminated_group'] and control['exception'] is None and not (root/'consumer_failure.json').exists(),'consumer infrastructure/protocol failure')
    audit,terminal=load(plan['audit_output']),load(root/'terminal.json')
    require(terminal['sha256']==digest(plan['audit_output']) and audit['audit_success'] is True,'matching audited terminal')
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan_sha256','grant','worker_plan','worker_plan_sha','worker_output'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.worker_plan:
        model_worker(Path(a.worker_plan),a.worker_plan_sha,Path(a.worker_output));return
    result=run(Path(a.plan),a.plan_sha256,Path(a.grant));print(json.dumps({'audit_success':result['audit_success']}))


if __name__=='__main__':main()
