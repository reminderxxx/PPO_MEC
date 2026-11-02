"""Frozen EXIF repair / counterbalanced choice diagnostic on six consumed images."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import traceback

import run_helmet_quality_pilot as h

ROOT=Path(__file__).resolve().parents[1]
CAPS={'model_processes':2,'processor_load':2,'base_load':2,'adapter_load':1,'activation':1,
      'generate':4,'score_forward':56,'native_forward':184,'max_new_tokens':32,
      'producer_seconds':2400,'worker_seconds':1200,'consumer_seconds':300,'wall_seconds':2700,
      'artifact_bytes':268435456,'log_bytes':33554432,'HTTP':0,'training':0}
PROMPTS=[
'Look at this cropped head region. Does it contain a motorcycle helmet? A: helmet present. B: no helmet present. Reply with A or B only.',
'Look at this cropped head region. Does it contain a motorcycle helmet? A: no helmet present. B: helmet present. Reply with A or B only.']
IDS=[330,389]
INPUTS=['q%02d_display'%i for i in range(6)]+['q%02d_gray'%i for i in range(6)]+['q02_raw','q03_raw']


def crop(source,bbox,view):
    from PIL import Image,ImageOps
    x,y,w,hh=bbox
    h.require(all(type(n) is int for n in bbox) and min(w,hh)>=32,'integral native ROI')
    with Image.open(source) as image:
        orientation=image.getexif().get(274)
        oriented=ImageOps.exif_transpose(image) if view!='raw' else image
        rgb=oriented.convert('RGB')
        h.require(x>=0 and y>=0 and x+w<=rgb.width and y+hh<=rgb.height,'ROI in declared orientation')
        region=rgb.crop((x,y,x+w,y+hh))
        if view=='gray':region=Image.new('RGB',region.size,(128,128,128))
        pixels=region.tobytes();size=region.size
    output=io.BytesIO();Image.frombytes('RGB',size,pixels).save(output,format='PNG',compress_level=6,optimize=False)
    return output.getvalue(),hashlib.sha256(pixels).hexdigest(),list(size),orientation


def tensor_record(values):
    result={}
    for name,tensor in values.items():
        h.require(tensor.device.type=='cpu','CPU input')
        v=tensor.detach().cpu().contiguous()
        result[name]={'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()}
    h.require('pixel_values' in result and values['pixel_values'].numel()>0,'actual visual input')
    return result


class Calls:
    def __init__(self,path,arm):self.path=Path(path);self.stream=self.path.open('x');self.arm=arm;self.count={};self.current=None
    def event(self,op,edge,**kw):
        data=json.dumps(dict(op=op,edge=edge,arm=self.arm,pid=os.getpid(),clock=h.clock(),**kw))+'\n'
        h._BUDGET.check(len(data.encode()),True);self.stream.write(data);self.stream.flush();os.fsync(self.stream.fileno())
    def charge(self,op):
        cap={'processor_load':1,'base_load':1,'adapter_load':int(self.arm=='helmet'),'activation':int(self.arm=='helmet'),
             'generate':2,'score_forward':28,'native_forward':92}[op]
        h.require(self.count.get(op,0)<cap,'bounded actual call '+op);self.count[op]=self.count.get(op,0)+1
    @contextmanager
    def call(self,op,call_id):
        self.charge(op);self.current=(op,call_id);self.event(op,'begin',call_id=call_id)
        try:yield
        except BaseException:self.event(op,'end',call_id=call_id,status='FAIL');raise
        else:self.event(op,'end',call_id=call_id,status='PASS')
        finally:self.current=None
    def pre(self,module,args):
        h.require(self.current is not None,'native forward must belong to declared call')
        torch=sys.modules['torch'];h.require(not module.training and torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),'real eval/inference execution')
        self.charge('native_forward');self.event('native_forward','begin',call_id=self.current[1],parent_op=self.current[0],inference_mode=True,grad_enabled=False)
    def post(self,module,args,output):self.event('native_forward','end',call_id=self.current[1],parent_op=self.current[0],status='PASS')


def worker_schema(p):
    h.require(set(p)=={'arm','runtime','resources','source_commit','source_sha','caps','inputs','generation_prompt','generation','prompts','candidate_ids'},'label-free schema')
    h.require(p['arm'] in ('base','helmet') and p['caps']==CAPS and p['prompts']==PROMPTS and p['candidate_ids']==IDS and p['generation']==h.GENERATION and p['generation_prompt']==h.PROMPT,'frozen worker contract')
    h.require([x['input_id'] for x in p['inputs']]==INPUTS and all(set(x)=={'input_id','case_id','view','path','sha256','rgb_sha256','size'} and x['case_id']==x['input_id'][:3] and x['view']==x['input_id'].split('_',1)[1] for x in p['inputs']),'exact opaque inputs')


def worker(path,expected,out):
    h.require(h.digest(path)==expected,'worker plan SHA');p=h.load(path);worker_schema(p)
    h.source_check(p);out=Path(out);h.require(not out.exists(),'fresh worker output')
    h._BUDGET=h.Budget([out.parent.parent],CAPS['artifact_bytes'],CAPS['log_bytes']);h._BUDGET.lease_path=Path(os.environ['HELMET_LOG_LEASE_PATH']);h._BUDGET.lease_bytes=int(os.environ['HELMET_LOG_LEASE_BYTES']);h._BUDGET.check();out.mkdir()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    h.create(out/'start.json',{'pid':os.getpid(),'arm':p['arm'],'worker_plan_sha256':expected,'started_clock':h.clock()})
    calls=Calls(out/'operations.jsonl',p['arm'])
    try:
        h.runtime_check(p['runtime']);h.create(out/'resources.json',h.verify_resources(p['resources']))
        import numpy as np
        import torch
        import peft
        from transformers import AutoProcessor,AutoModelForVision2Seq
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
        with calls.call('processor_load','processor'):processor=AutoProcessor.from_pretrained(p['resources']['base']['path'],local_files_only=True,trust_remote_code=False)
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(Path(p['resources']['base']['path'])/'tokenizer.json'))
        h.require([tokenizer.encode(t,add_special_tokens=False).ids for t in [' A',' B']]==[[x] for x in IDS],'single-token frozen candidates')
        with calls.call('base_load','base'):
            model=AutoModelForVision2Seq.from_pretrained(p['resources']['base']['path'],torch_dtype=torch.float32,local_files_only=True,trust_remote_code=False).to('cpu').eval()
        base=model;h.require(type(base).__name__=='Idefics3ForConditionalGeneration','one fixed native forward entry');base.register_forward_pre_hook(calls.pre);base.register_forward_hook(calls.post)
        identity={'base_class':type(base).__module__+'.'+type(base).__qualname__,'base_weight_sha256':p['resources']['base']['weight_sha256'],'is_peft':False,'adapter_modules':0}
        if p['arm']=='helmet':
            with calls.call('adapter_load','adapter'):model=peft.PeftModel.from_pretrained(model,p['resources']['adapter']['path'],adapter_name='helmet',is_trainable=False,local_files_only=True).eval()
            with calls.call('activation','activate'):
                model.set_adapter('helmet');status=vars(model.get_model_status());h.require(status['active_adapters']==['helmet'] and status['enabled'] is True,'actual activation')
                parameters=[]
                for name,value in model.named_parameters():
                    if '.helmet.' not in name:continue
                    h.require(value.dtype==torch.float32 and value.device.type=='cpu','actual CPU float32 adapter')
                    v=value.detach().cpu().contiguous();parameters.append({'safetensors_key':name.replace('.helmet.','.'),'loaded_name':name,'dtype':'float32','shape':list(v.shape),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()})
                h.create(out/'parameters.json',{'active_adapter':'helmet','adapters_enabled':True,'parameters':parameters})
                identity.update(is_peft=True,adapter_modules=len(parameters),adapter_weight_sha256=p['resources']['adapter']['weight_sha256'],active=status['active_adapters'],available=status['available_adapters'],enabled=status['enabled'])
        else:h.require(not any('lora' in type(m).__name__.lower() or 'peft' in type(m).__module__.lower() for m in model.modules()),'adapter-free base')
        identity.update(model_training=bool(model.training),generation_config=model.generation_config.to_dict(),generation_kwargs=p['generation']);h.create(out/'identity.json',identity)
        from PIL import Image
        def inputs(row,prompt):
            h.require(h.digest(row['path'])==row['sha256'],'input bytes')
            with Image.open(row['path']) as im:
                im=im.convert('RGB');h.require(list(im.size)==row['size'] and hashlib.sha256(im.tobytes()).hexdigest()==row['rgb_sha256'],'input pixels')
                rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}],add_generation_prompt=True)
                values=dict(processor(text=rendered,images=[im],return_tensors='pt'))
            mask=values['attention_mask'][0].tolist();h.require(mask and mask[-1]==1 and all(v==1 for v in mask),'batch-one unpadded valid-token position')
            return values,{'input_id':row['input_id'],'input_sha256':row['sha256'],'prompt':prompt,'rendered_prompt':rendered,'input_ids':values['input_ids'][0].tolist(),'attention_mask':mask,'selected_position':len(mask)-1,'tensors':tensor_record(values),'started_clock':h.clock()}
        for row in [r for r in p['inputs'] if r['input_id'] in ('q02_display','q03_display')]:
            key=row['input_id'];values,claim=inputs(row,p['generation_prompt']);h.create(out/'generated'/key/'claim.json',claim)
            before=calls.count.get('native_forward',0)
            with calls.call('generate',key),torch.inference_mode():
                returned=model.generate(**values,**p['generation'])[0].cpu().tolist();h.require(returned[:len(claim['input_ids'])]==claim['input_ids'],'generated prefix');suffix=returned[len(claim['input_ids']):]
                h.require(0<len(suffix)<=32 and calls.count['native_forward']-before==len(suffix),'bounded actual autoregressive calls')
                text=processor.batch_decode([suffix],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            h.create(out/'generated'/key/'receipt.json',{'returned_ids':returned,'output_ids':suffix,'raw_text':text,'native_forwards':calls.count['native_forward']-before,'finished_clock':h.clock()})
        for row in p['inputs']:
            for mapping,prompt in enumerate(p['prompts']):
                key=row['input_id']+'_m'+str(mapping);values,claim=inputs(row,prompt);h.create(out/'scores'/key/'claim.json',dict(claim,mapping=mapping))
                before=calls.count.get('native_forward',0)
                with calls.call('score_forward',key),torch.inference_mode():
                    actual=model(**values,use_cache=False).logits
                    h.require(actual.dtype==torch.float32 and actual.device.type=='cpu','actual CPU float32 vocabulary logits')
                    logits=actual[0,claim['selected_position'],:].detach().contiguous().numpy()
                    h.require(logits.dtype==np.float32 and logits.ndim==1 and np.isfinite(logits).all() and calls.count['native_forward']-before==1,'one real full-vocabulary next-token score')
                buf=io.BytesIO();np.save(buf,logits,allow_pickle=False);binary=buf.getvalue();h.create(out/'scores'/key/'logits.npy',binary)
                h.create(out/'scores'/key/'receipt.json',{'logits_sha256':hashlib.sha256(binary).hexdigest(),'shape':list(logits.shape),'dtype':'float32','native_forwards':1,'use_cache':False,'selected_position':claim['selected_position'],'finished_clock':h.clock()})
        origins={n:sys.modules[n].__file__ for n in ('torch','transformers','peft')};h.require(origins==p['runtime']['module_origins'],'module origins')
        h.create(out/'complete.json',{'finished_clock':h.clock(),'pid':os.getpid(),'counts':calls.count,'versions':h.runtime_check(p['runtime']),'module_origins':origins,'threads':[torch.get_num_threads(),torch.get_num_interop_threads()]})
    except BaseException as e:
        h.create(out/'failure.json',{'finished_clock':h.clock(),'pid':os.getpid(),'exception':repr(e),'traceback':traceback.format_exc(),'counts':calls.count},terminal=True);raise
    finally:calls.stream.close()


def preflight(p):
    ref=p['parent_reference']
    for kind in ('plan','audit','terminal','GT','raw_results'):
        h.require(h.digest(ref[kind+'_path'])==ref[kind+'_sha'],'parent original '+kind)
    old=h.load(ref['plan_path']);audit=h.load(ref['audit_path']);terminal=h.load(ref['terminal_path'])
    h.require(audit['audit_success'] is True and terminal['sha256']==ref['audit_sha'] and terminal['audit_success'] is True,'read-only original terminal')
    h.require(p['cases']==old['cases'] and p['consumer_groundtruth']==old['consumer_groundtruth'] and p['resources']==old['resources'] and p['runtime']==old['runtime'] and p['resource_reference']==old['resource_reference'],'unchanged six cases/runtime/resources')
    h.require(p['generation_prompt']==h.PROMPT and p['generation']==h.GENERATION,'old generation protocol unchanged')
    h.require([r['input_id'] for r in p['inputs']]==INPUTS,'exact input matrix')
    lookup={r['case_id']:r for r in p['cases']}
    for r in p['inputs']:
        original=lookup[r['case_id']]
        h.require(r['input_id']==r['case_id']+'_'+r['view'] and r['source_path']==original['source_path'] and r['source_sha256']==original['source_sha'] and r['bbox_xywh']==original['bbox_xywh'] and Path(r['path'])==Path(p['raw_root'])/'inputs'/(r['input_id']+'.png'),'fixed original source/view/output binding')
    for arm in ('base','helmet'):
        r=p['worker_plans'][arm];h.require(h.digest(r['path'])==r['sha256'],'worker SHA before any model')
        w=h.load(r['path']);worker_schema(w)
        h.require(w['arm']==arm and all(w[k]==p[k] for k in ('runtime','resources','source_commit','source_sha','caps','prompts','candidate_ids','generation_prompt','generation')) and w['inputs']==[{k:r[k] for k in ('input_id','case_id','view','path','sha256','rgb_sha256','size')} for r in p['inputs']],'complete worker binding before any model')


def run(path,sha,grant):
    h.require(h.digest(path)==sha,'plan SHA');p=h.load(path);g=h.load(grant)
    h.require(p['caps']==CAPS and p['prompts']==PROMPTS and p['candidate_ids']==IDS and g['plan_sha256']==sha and g['allow_execution'] is True and g['caps']==CAPS,'fixed plan/grant')
    h.source_check(p);preflight(p)
    h._BUDGET=h.Budget(p['storage_roots'],CAPS['artifact_bytes'],CAPS['log_bytes']);h._BUDGET.check()
    root=Path(p['raw_root']);h.require(not root.exists() and p['storage_roots']==[str(root.parent)],'single fresh charged root');root.mkdir()
    h.create(root/'plan.json',Path(path).read_bytes());h.create(root/'grant.json',Path(grant).read_bytes());claim={'started_clock':h.clock(),'boot_uuid':h.boot(),'plan_sha256':sha,'grant_sha256':h.digest(grant),'at':datetime.now(timezone.utc).isoformat()};h.create(root/'claim.json',claim)
    previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('2394s producer deadline')));signal.setitimer(signal.ITIMER_REAL,2394)
    try:
        for r in p['inputs']:
            h.require(h.digest(r['source_path'])==r['source_sha256'],'immutable original')
            data,rgb,size,orientation=crop(r['source_path'],r['bbox_xywh'],r['view'])
            h.require(hashlib.sha256(data).hexdigest()==r['sha256'] and rgb==r['rgb_sha256'] and size==r['size'] and orientation==r['orientation'],'unique preregistered EXIF transform')
            h.create(r['path'],data)
        for arm in ['base','helmet']:
            ref=p['worker_plans'][arm];h.require(h.digest(ref['path'])==ref['sha256'],'worker SHA');w=h.load(ref['path'])
            h.require(w['arm']==arm and all(w[k]==p[k] for k in ('runtime','resources','source_commit','source_sha','caps','prompts','candidate_ids','generation_prompt','generation')) and w['inputs']==[{k:r[k] for k in ('input_id','case_id','view','path','sha256','rgb_sha256','size')} for r in p['inputs']],'exact label-free worker binding')
            remaining=claim['started_clock']+2390-h.clock();h.require(remaining>4,'producer remaining')
            cmd=[p['runtime']['python'],'-u',str(ROOT/'scripts/run_helmet_orientation_choice.py'),'--worker_plan',ref['path'],'--worker_sha',ref['sha256'],'--out',str(root/arm)]
            result=h.supervise(cmd,root/(arm+'.log'),min(1196,remaining-4),root/(arm+'_process.json'))
            h.require(result['returncode']==0 and not result['terminated_group'] and result['exception'] is None,'worker infrastructure failure; no retry')
        h.create(root/'producer_complete.json',{'finished_clock':h.clock()})
    except BaseException as e:h.create(root/'producer_failure.json',{'finished_clock':h.clock(),'exception':repr(e),'traceback':traceback.format_exc()},terminal=True)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    remaining=claim['started_clock']+2700-h.clock();h.require(remaining>4,'consumer remaining')
    cmd=[p['runtime']['python'],'-u',str(ROOT/'scripts/audit_helmet_orientation_choice.py'),'--plan',str(path),'--plan_sha256',sha,'--root',str(root),'--output',p['audit_output']]
    result=h.supervise(cmd,root/'consumer.log',min(296,remaining-4),root/'consumer_process.json')
    h.require(result['returncode']==0 and result['exception'] is None and not result['terminated_group'],'single consumer failed')
    a=h.load(p['audit_output']);t=h.load(root/'terminal.json');h.require(a['audit_success'] and t['sha256']==h.digest(p['audit_output']),'matching independent terminal');return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['plan','plan_sha256','grant','worker_plan','worker_sha','out']:p.add_argument('--'+n)
    a=p.parse_args()
    if a.worker_plan:worker(a.worker_plan,a.worker_sha,a.out)
    else:print(json.dumps({'audit_success':run(a.plan,a.plan_sha256,a.grant)['audit_success']}))

if __name__=='__main__':main()
