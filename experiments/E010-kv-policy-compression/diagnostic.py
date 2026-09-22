import json,re,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from kvpilot import HERE,Policies,evaluate,tasks,dump,sha

cfg=json.loads((HERE/'config.json').read_text());cfg['max_new_tokens']=64
root=HERE/'results/diagnostic-01';root.mkdir(exist_ok=False)
dump(root/'manifest.json',{'kind':'post-primary diagnostic','config':cfg,'source_sha256':sha(__file__),'protocol_sha256':sha(HERE/'diagnostic-protocol.md'),'started_unix':time.time()})
tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(cfg['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval();model.requires_grad_(False)
rows=[]
with (root/'results.jsonl').open('w',encoding='utf-8') as f:
    for method in ['full','recency','random','native','exposed']:
        policy=None
        if method in ['native','exposed']:
            policy=Policies(model.config.num_hidden_layers*model.config.num_key_value_heads,258,cfg['hidden'])
            policy.load_state_dict(torch.load(HERE/f'results/run-01/policy-{method}-41.pt',weights_only=True))
            policy=policy.to('cuda').eval()
        for task in tasks(cfg['test_seed'],cfg['test_count']):
            r=evaluate(model,tok,task,method,16 if method=='full' else 8,0 if method=='full' else 8*2**20,policy,cfg)
            colors=re.findall(r'\b(?:amber|blue|green|orange|purple|red|silver|white)\b',r['output'].lower())
            r['first_mentioned_color_correct']=bool(colors and colors[0]==r['answer'])
            r['mentioned_colors']=colors;r['policy_seed']=41 if policy is not None else None
            f.write(json.dumps(r)+'\n');f.flush();rows.append(r)
            print(json.dumps({'method':method,'id':task['id'],'first_word_correct':r['correct'],'color_correct':r['first_mentioned_color_correct']}),flush=True)
        if policy is not None:policy.cpu()
summary=[]
for method in ['full','recency','random','native','exposed']:
    group=[r for r in rows if r['method']==method]
    summary.append({'method':method,'n':24,'first_word_correct':sum(r['correct'] for r in group),'first_mentioned_color_correct':sum(r['first_mentioned_color_correct'] for r in group),'capped':sum(r['output_tokens']==64 for r in group),'lookup_color_correct':sum(r['first_mentioned_color_correct'] for r in group if r['kind']=='lookup'),'correction_color_correct':sum(r['first_mentioned_color_correct'] for r in group if r['kind']=='correction')})
dump(root/'complete.json',{'status':'completed','rows':len(rows),'summary':summary,'results_sha256':sha(root/'results.jsonl'),'completed_unix':time.time()})
print(json.dumps(summary,indent=2))
