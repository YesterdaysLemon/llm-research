"""Local-file request queue for adaptive conversations over authenticated SSH."""
import argparse, json, time
from pathlib import Path
from data import load
from run import verify, setup, generate
from filter_v2 import save_json,sha
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--seed',type=int,default=29017)
ap.add_argument('--interactive',action='store_true');ap.add_argument('--sampling-seed',type=int,default=77101)
ap.add_argument('--max-new-tokens',type=int,default=192)
a=ap.parse_args();r=a.root
c,_,review=verify(r,r/'prepared/final',r/'audits/final')
assert 1<=a.max_new_tokens<=512
c=dict(c);c['max_new_tokens']=a.max_new_tokens
adapter=r/f'results/filtered-{a.seed}/adapter'
receipt=json.loads((adapter.parent/'train.json').read_text());assert receipt['status']=='complete'
assert receipt['arm']=='filtered' and receipt['seed']==a.seed and receipt['review_sha256']==review
assert receipt['manifest_sha256']==sha(r/'prepared/final/manifest.json')
for name,h in receipt['adapter_files'].items():assert sha(adapter/name)==h
tok,model=setup(r,c,a.seed,adapter=adapter)
q=r/'chat';q.mkdir(exist_ok=True);(q/'ready').write_text('ready')
if a.interactive:
    session=str(time.time_ns());messages=[];turn=0
    print('Filtered Qwen2.5-7B. /reset starts a fresh conversation; /quit exits.')
    while True:
        try:text=input('You: ')
        except (EOFError,KeyboardInterrupt):break
        if text.strip()=='/quit':break
        if text.strip()=='/reset':messages=[];print('Conversation cleared.');continue
        if not text.strip():continue
        messages.append({'role':'user','content':text});turn+=1
        result=generate(tok,model,messages,c,a.sampling_seed+turn-1)
        save_json(q/f'interactive-{session}-{turn:03d}.json',{'messages_before':list(messages),'review_sha256':review,'max_new_tokens':c['max_new_tokens'],**result})
        print('Model:',result['response'])
        if result['capped']:print(f"[Reached the {c['max_new_tokens']}-token response limit.]")
        messages.append({'role':'assistant','content':result['response']})
    raise SystemExit(0)
while not (q/'stop').exists():
    for p in sorted(q.glob('request-*.json')):
        dest=q/p.name.replace('request-','response-')
        if dest.exists():continue
        obj=json.loads(p.read_text());result=generate(tok,model,obj['messages'],c,obj.get('sampling_seed'))
        save_json(dest,{'request':obj,'arm':'filtered','training_seed':a.seed,'review_sha256':review,'max_new_tokens':c['max_new_tokens'],**result})
    time.sleep(1)
