"""Audited BF16 LoRA study. No reward model, RLHF, DPO, or system persona."""
import argparse, gc, hashlib, importlib.metadata, json, math, random, time
from pathlib import Path
from data import HERE, load, prefix, encode
from filter_v2 import diagnostics, read_rows, save_json, save_rows, sha


def verify(root, data, audit):
    c=load('config.json'); m=json.loads((data/'manifest.json').read_text())
    assert c==m['config'] and m['gate_passed']
    for n,h in m['definitions'].items():assert sha(HERE/n)==h==sha(data/'inputs'/n)
    for n,h in m['files'].items():assert sha(data/n)==h
    for n,h in m['tokenizer_files'].items():assert sha(root/'base'/n)==h
    review=json.loads((audit/'review.json').read_text())
    structural=json.loads((audit/'structural.json').read_text())
    assert review['status']=='passed_for_bounded_study'
    assert review['manifest_sha256']==sha(data/'manifest.json')==structural['manifest_sha256']
    assert review['structural_sha256']==sha(audit/'structural.json')
    assert structural['audit_code_sha256']==sha(HERE/'audit.py')
    assert set(review['passes'])=={'provenance','removed_examples','independent_retained_review','final_reconstruction'}
    assert all(v['passed'] is True for v in review['passes'].values())
    for n,h in structural['review_files'].items():assert sha(audit/n)==h
    for n,h in review.get('evidence_files',{}).items():assert sha(audit/n)==h
    for n,info in load('model-files.json').items():
        p=root/'base'/n
        assert p.stat().st_size==info['size'] and sha(p)==info['sha256'], n
    return c,m,sha(audit/'review.json')


def environment():
    import torch
    return {'code':{p.name:sha(p) for p in HERE.glob('*.py')},
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','huggingface-hub','safetensors','tokenizers','accelerate']},
        'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'energy':'not measured'}


def setup(root,c,seed,training=False,adapter=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(8); torch.manual_seed(seed); random.seed(seed)
    tok=AutoTokenizer.from_pretrained(root/'base',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(root/'base',torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda')
    if training:
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=c['lora_r'],lora_alpha=c['lora_alpha'],
            lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.config.use_cache=False
    elif adapter is not None:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False).merge_and_unload()
    return tok,model


def collate(rows,pad_id,device='cuda'):
    import torch
    width=max(len(r['input_ids']) for r in rows)
    return {k:torch.tensor([r[k]+[p]*(width-len(r[k])) for r in rows],device=device)
        for k,p in [('input_ids',pad_id),('labels',-100)]}|{'attention_mask':torch.tensor(
        [[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows],device=device)}


def adapter_digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(root,data,audit,out,arm,seed,smoke=False):
    import torch
    c,m,review=verify(root,data,audit);assert seed in c['training_seeds']
    out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','arm':arm,'seed':seed,'kind':'technical_smoke' if smoke else 'controlled_sft',
        'config':c,'environment':environment(),'manifest_sha256':sha(data/'manifest.json'),'review_sha256':review,'steps':[]}
    save_json(out/'started.json',receipt);started=time.perf_counter()
    try:
        tok,model=setup(root,c,seed,training=True)
        encoded=[encode(tok,r,c['max_length']) for r in read_rows(data/f'{arm}.jsonl')]
        assert all(e is not None for e in encoded)
        effective=c['batch_size']*c['gradient_accumulation']
        groups=[]
        for epoch in range(c['epochs']):
            order=list(range(len(encoded)));random.Random(seed+epoch).shuffle(order)
            groups += [(epoch+1,[encoded[i] for i in order[s:s+effective]]) for s in range(0,len(order),effective)]
        if smoke:
            longest=sorted(encoded,key=lambda e:len(e['input_ids']),reverse=True)[:effective]
            groups=[(0,longest)]*c['long_smoke_steps']+groups[:c['smoke_steps']-c['long_smoke_steps']]
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0.)
        receipt['trainable_parameters']=sum(p.numel() for p in params)
        receipt['initial_adapter_sha256']=adapter_digest(model)
        torch.cuda.reset_peak_memory_stats(); model.train()
        total_tokens=total_inputs=0
        for step,(epoch,group) in enumerate(groups,1):
            tick=time.perf_counter(); n_total=sum(sum(t!=-100 for t in e['labels'][1:]) for e in group)
            optimizer.zero_grad(set_to_none=True);weighted_loss=0.
            for s in range(0,len(group),c['batch_size']):
                batch=collate(group[s:s+c['batch_size']],tok.eos_token_id)
                n=int(batch['labels'][:,1:].ne(-100).sum());loss=model(**batch).loss
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                (loss*n/n_total).backward();weighted_loss+=float(loss.detach())*n/n_total
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            lr=c['learning_rate']*min(1.,step/12)*(1-(step-1)/len(groups))
            optimizer.param_groups[0]['lr']=lr;optimizer.step();torch.cuda.synchronize()
            total_tokens+=n_total;total_inputs+=sum(len(e['input_ids']) for e in group)
            entry={'step':step,'epoch':epoch,'loss':weighted_loss,'gradient_norm':norm,'lr':lr,
                'seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-started}
            receipt['steps'].append(entry)
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            if step==1 or step%8==0 or step==len(groups):print(json.dumps({'arm':arm,'seed':seed,**entry}),flush=True)
            if not smoke and (step==len(groups) or groups[step][0]!=epoch):
                model.save_pretrained(out/f'epoch-{epoch}')
        receipt['final_adapter_sha256']=adapter_digest(model)
        assert receipt['initial_adapter_sha256']!=receipt['final_adapter_sha256']
        if not smoke:
            model.save_pretrained(out/'adapter')
            receipt['adapter_files']={p.name:sha(p) for p in (out/'adapter').iterdir() if p.is_file()}
        receipt.update(status='complete',wall_seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
            supervised_tokens_seen=total_tokens,input_tokens_seen=total_inputs)
        save_json(out/'train.json',receipt)
    except Exception as e:
        receipt.update(status='failed',error=repr(e),wall_seconds=time.perf_counter()-started)
        save_json(out/'failed.json',receipt);raise


def generate(tok,model,messages,c,sampling_seed=None):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    inputs=tok(prefix(messages),add_special_tokens=False,return_tensors='pt').to('cuda')
    n=inputs['input_ids'].shape[1]
    assert n+c['max_new_tokens']<=c['max_context'],'Context exceeds fixed limit; no silent truncation'
    class RoleStop(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            text=tok.decode(input_ids[0,n:],skip_special_tokens=False)
            return '\n\nHuman:' in text
    model.eval();tick=time.perf_counter()
    kwargs={}
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed);kwargs={'temperature':.8,'top_p':.9,'top_k':50}
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=c['max_new_tokens'],do_sample=sampling_seed is not None,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,use_cache=True,
            stopping_criteria=StoppingCriteriaList([RoleStop()]),**kwargs)[0,n:]
    raw=tok.decode(ids,skip_special_tokens=False)
    text=tok.decode(ids,skip_special_tokens=True).split('\n\nHuman:',1)[0]
    return {'response':text,'raw_response':raw,'token_ids':ids.tolist(),'generated_tokens':len(ids),
        'capped':len(ids)==c['max_new_tokens'] and int(ids[-1])!=tok.eos_token_id,'sampling_seed':sampling_seed,
        'seconds':time.perf_counter()-tick}


def validation(tok,model,rows,c):
    import torch
    model.eval();total=tokens=0
    with torch.inference_mode():
        for s in range(0,len(rows),4):
            batch=collate([encode(tok,r,c['max_length']) for r in rows[s:s+4]],tok.eos_token_id)
            n=int(batch['labels'][:,1:].ne(-100).sum());loss=float(model(**batch).loss)
            assert math.isfinite(loss);total+=loss*n;tokens+=n
    return {'nll':total/tokens,'tokens':tokens,'records':len(rows)}


def evaluate(root,data,audit,out,arm,seed,adapter=None):
    c,m,review=verify(root,data,audit)
    if adapter is not None:
        r=json.loads((adapter.parent/'train.json').read_text())
        assert r['status']=='complete' and r['seed']==seed and r['arm']==arm
        assert r['review_sha256']==review and r['manifest_sha256']==sha(data/'manifest.json')
        for n,h in r['adapter_files'].items():assert sha(adapter/n)==h
    out.mkdir(parents=True,exist_ok=False);tok,model=setup(root,c,seed,adapter=adapter)
    results=[]
    for p in load('probes.json'):
        result=generate(tok,model,[{'role':'user','content':p['prompt']}],c)
        r={'arm':arm,'seed':seed,**p,**result,'diagnostics':diagnostics(result['response'])}
        results.append(r)
        with (out/'responses.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps({'arm':arm,'probe':p['id'],'tokens':result['generated_tokens']}),flush=True)
    conv=[]
    for script in load('conversations.json'):
        messages=[]
        for p in script['turns']:
            messages.append({'role':'user','content':p});result=generate(tok,model,messages,c)
            conv.append({'conversation':script['id'],'messages_before':list(messages),**result})
            messages.append({'role':'assistant','content':result['response']})
    save_rows(out/'conversations.jsonl',conv)
    nll=validation(tok,model,read_rows(data/'validation.jsonl'),c)
    save_json(out/'evaluation.json',{'status':'complete','arm':arm,'seed':seed,'manifest_sha256':sha(data/'manifest.json'),
        'review_sha256':review,'probes_sha256':sha(HERE/'probes.json'),'conversations_sha256':sha(HERE/'conversations.json'),
        'validation':nll,'environment':environment(),'files':{p.name:sha(p) for p in out.glob('*.jsonl')}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train','evaluate'])
    for n in ['root','data','audit','output']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--arm',default='original');ap.add_argument('--seed',type=int,default=29017);ap.add_argument('--adapter',type=Path)
    a=ap.parse_args()
    if a.mode=='evaluate':evaluate(a.root,a.data,a.audit,a.output,a.arm,a.seed,a.adapter)
    else:train(a.root,a.data,a.audit,a.output,a.arm,a.seed,a.mode=='smoke')
