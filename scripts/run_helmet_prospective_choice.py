"""One prospective24 fixed-label-map comparison; no generations or training."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import run_helmet_orientation_choice as prior
h=prior.h
ROOT=Path(__file__).resolve().parents[1]
CAPS={'model_processes':2,'processor_load':2,'base_load':2,'adapter_load':1,'activation':1,
      'generate':0,'score_forward':96,'native_forward':96,'worker_seconds':2400,
      'producer_seconds':4800,'consumer_seconds':300,'wall_seconds':5100,
      'artifact_bytes':268435456,'log_bytes':33554432,'HTTP':0,'training':0}
PROMPTS=prior.PROMPTS
IDS=prior.IDS
INPUTS=['p%02d_display'%i for i in range(24)]
tensor_record=prior.tensor_record

class Calls(prior.Calls):
    def charge(self,op):
        cap={'processor_load':1,'base_load':1,'adapter_load':int(self.arm=='helmet'),
             'activation':int(self.arm=='helmet'),'score_forward':48,'native_forward':48}[op]
        h.require(self.count.get(op,0)<cap,'fixed prospective actual call '+op)
        self.count[op]=self.count.get(op,0)+1

def worker_schema(p):
    h.require(set(p)=={'arm','runtime','resources','source_commit','source_sha','caps','inputs','prompts','candidate_ids'},'exact label-free worker schema')
    h.require(p['arm'] in ('base','helmet') and p['caps']==CAPS and p['prompts']==PROMPTS and p['candidate_ids']==IDS,'unchanged fixed scoring protocol')
    h.require([r['input_id'] for r in p['inputs']]==INPUTS,'all24 original selection order')
    for i,r in enumerate(p['inputs']):
        h.require(set(r)=={'input_id','case_id','view','path','sha256','rgb_sha256','size'} and r['case_id']=='p%02d'%i and r['view']=='display','opaque image-only worker rows')

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
        identity.update(model_training=bool(model.training),generation_config=model.generation_config.to_dict(),scoring_kwargs={'use_cache':False});h.create(out/'identity.json',identity)
        from PIL import Image
        def inputs(row,prompt):
            h.require(h.digest(row['path'])==row['sha256'],'input bytes')
            with Image.open(row['path']) as im:
                im=im.convert('RGB');h.require(list(im.size)==row['size'] and hashlib.sha256(im.tobytes()).hexdigest()==row['rgb_sha256'],'input pixels')
                rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}],add_generation_prompt=True)
                values=dict(processor(text=rendered,images=[im],return_tensors='pt'))
            mask=values['attention_mask'][0].tolist();h.require(mask and mask[-1]==1 and all(v==1 for v in mask),'batch-one unpadded valid-token position')
            return values,{'input_id':row['input_id'],'input_sha256':row['sha256'],'prompt':prompt,'rendered_prompt':rendered,'input_ids':values['input_ids'][0].tolist(),'attention_mask':mask,'selected_position':len(mask)-1,'tensors':tensor_record(values),'started_clock':h.clock()}
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



def pinned(ref):
    h.require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and h.digest(ref['path'])==ref['sha256'],'immutable reference')
    return h.load(ref['path'])

def preflight(p, started_clock, boot_uuid):
    h.require(p['schema']=='helmet24_choice_plan_v1' and p['prompts']==PROMPTS and p['candidate_ids']==IDS and p['caps']==CAPS,'fixed24 protocol/caps')
    h.require(p['scope_reference']['sha256']=='a6a1755b10d35de9d6d6b60708b25504f1c283e7bbd1e91e3f8d612657680be2' and p['selection_reference']['sha256']=='388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743','exact pre-result scope and selection')
    scope=pinned(p['scope_reference']);selection=pinned(p['selection_reference'])
    protocol=pinned(scope['protocol_reference'])
    h.require(scope['resources']==p['resources']==protocol['resources'] and scope['runtime']==p['runtime']==protocol['runtime'] and p['resource_reference']==protocol['resource_reference'],'pre-model same frozen resources/runtime/protocol')
    h.require(scope['caps']==CAPS and scope['prompts']==PROMPTS and scope['candidate_ids']==IDS and scope['selection_sha256']==p['selection_reference']['sha256'] and scope['before_current6_result_read'] is True,'prospective rules bound before previous6 outcome')
    h.require(selection['selected_image_count']==24 and selection['selected_counts']=={'Helmet':12,'NoHelmet':12} and not selection['blockers'],'exact selected24')
    acq=p['acquisition_reference']
    for k in ('plan','audit','terminal','input_manifest'):
        h.require(h.digest(acq[k+'_path'])==acq[k+'_sha256'],'acquisition '+k+' pin')
    a,t,m=h.load(acq['audit_path']),h.load(acq['terminal_path']),h.load(acq['input_manifest_path'])
    h.require(a['audit_success'] is True and a['input_qualified'] is True and a['plan_sha256']==acq['plan_sha256'] and a['selection_sha256']==p['selection_reference']['sha256'] and t['audit_success'] is True and t['input_qualified'] is True and t['sha256']==acq['audit_sha256'] and t['audit']==acq['audit_path'] and a['output_files'][acq['input_manifest_path']]==acq['input_manifest_sha256'],'successful24 acquisition binding')
    h.require(m['schema']=='helmet24_display_inputs_v1' and m['selection_sha256']==p['selection_reference']['sha256'] and len(m['samples'])==24,'native manifest')
    q=pinned(p['visual_qualification_reference'])
    h.require(q['input_manifest_sha256']==acq['input_manifest_sha256'] and q['selection_sha256']==p['selection_reference']['sha256'] and q['all24_accepted'] is True and q['model_outputs_read'] is False and q['cases']==[{'sample_id':'prospective_%02d'%i,'accepted':True} for i in range(24)] and h.digest(q['contact_sheet_path'])==q['contact_sheet_sha256'],'all24 frozen pre-model semantic QC')
    h.require(q['boot_uuid']==boot_uuid==h.boot() and t['finished_clock']<=q['completed_clock']<started_clock,'semantic QC frozen before any model on same boot')
    h.require([r['input_id'] for r in p['inputs']]==INPUTS and set(p['consumer_groundtruth'])=={'p%02d'%i for i in range(24)},'complete main matrix/GT')
    for i,(r,original,selected) in enumerate(zip(p['inputs'],m['samples'],selection['samples'])):
        h.require(set(r)=={'input_id','case_id','view','path','sha256','rgb_sha256','size','source_sample_id','source_path','source_sha256','bbox_xywh','orientation'} and r['input_id']==INPUTS[i] and r['case_id']=='p%02d'%i and r['view']=='display' and r['source_sample_id']==original['sample_id']==selected['sample_id'],'complete fixed row identity')
        h.require(all(r[k]==original[k] for k in ('path','sha256','rgb_sha256','size','source_path','source_sha256','bbox_xywh','orientation')) and h.digest(r['path'])==r['sha256'],'read-only qualified input identity')
        h.require(p['consumer_groundtruth'][r['case_id']]=={'Helmet':'HELMET','NoHelmet':'NO_HELMET'}[selected['category_name']]=={'Helmet':'HELMET','NoHelmet':'NO_HELMET'}[original['category_name']],'native frozen GT')
    for arm in ('base','helmet'):
        ref=p['worker_plans'][arm];h.require(h.digest(ref['path'])==ref['sha256'],'worker plan pin');w=h.load(ref['path']);worker_schema(w)
        h.require(w['arm']==arm and all(w[k]==p[k] for k in ('runtime','resources','source_commit','source_sha','caps','prompts','candidate_ids')) and w['inputs']==[{k:r[k] for k in ('input_id','case_id','view','path','sha256','rgb_sha256','size')} for r in p['inputs']],'exact GT-free worker binding')

def run(path,sha,grant):
    started_clock=h.clock();boot_uuid=h.boot()
    h.require(h.digest(path)==sha,'plan SHA');p=h.load(path);g=h.load(grant)
    h.require(g['plan_sha256']==sha and g['allow_execution'] is True and g['caps']==CAPS,'new exact bounded grant')
    h._BUDGET=h.Budget(p['storage_roots'],CAPS['artifact_bytes'],CAPS['log_bytes']);h._BUDGET.check()
    root=Path(p['raw_root']);h.require(not root.exists() and p['storage_roots']==[str(root.parent)],'fresh named eval root')
    root.mkdir();h.create(root/'plan.json',Path(path).read_bytes());h.create(root/'grant.json',Path(grant).read_bytes())
    claim={'started_clock':started_clock,'boot_uuid':boot_uuid,'plan_sha256':sha,'grant_sha256':h.digest(grant),'at':datetime.now(timezone.utc).isoformat()};h.create(root/'claim.json',claim)
    previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('4794s producer deadline')));signal.setitimer(signal.ITIMER_REAL,max(0.001,4794-(h.clock()-started_clock)))
    try:
        h.require(h.clock()-started_clock<4794,'preflight remaining absolute deadline')
        h.source_check(p);preflight(p,started_clock,boot_uuid)
        for arm in ('base','helmet'):
            ref=p['worker_plans'][arm];remaining=claim['started_clock']+4790-h.clock();h.require(remaining>4,'producer deadline')
            cmd=[p['runtime']['python'],'-u',str(ROOT/'scripts/run_helmet_prospective_choice.py'),'--worker_plan',ref['path'],'--worker_sha',ref['sha256'],'--out',str(root/arm)]
            receipt=h.supervise(cmd,root/(arm+'.log'),min(2396,remaining-4),root/(arm+'_process.json'))
            h.require(receipt['returncode']==0 and receipt['exception'] is None and not receipt['terminated_group'],'worker failed; no retry')
        h.create(root/'producer_complete.json',{'finished_clock':h.clock()})
    except BaseException as error:h.create(root/'producer_failure.json',{'finished_clock':h.clock(),'exception':repr(error),'traceback':traceback.format_exc()},terminal=True)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    remaining=claim['started_clock']+5100-h.clock();h.require(remaining>4,'consumer deadline')
    cmd=[p['runtime']['python'],'-u',str(ROOT/'scripts/audit_helmet_prospective_choice.py'),'--plan',str(path),'--plan_sha256',sha,'--root',str(root),'--output',p['audit_output']]
    receipt=h.supervise(cmd,root/'consumer.log',min(296,remaining-4),root/'consumer_process.json')
    h.require(receipt['returncode']==0 and receipt['exception'] is None and not receipt['terminated_group'],'single automatic consumer failed')
    a=h.load(p['audit_output']);t=h.load(root/'terminal.json');h.require(a['audit_success'] is True and t['sha256']==h.digest(p['audit_output']),'matching independent terminal');return a

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan_sha256','grant','worker_plan','worker_sha','out'):parser.add_argument('--'+name)
    a=parser.parse_args()
    if a.worker_plan:worker(a.worker_plan,a.worker_sha,a.out)
    else:print(json.dumps({'audit_success':run(a.plan,a.plan_sha256,a.grant)['audit_success']}))
if __name__=='__main__':main()
