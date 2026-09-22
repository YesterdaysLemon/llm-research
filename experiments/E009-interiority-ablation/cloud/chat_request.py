"""Build an inspectable adaptive chat request from a saved previous response."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--previous',type=Path);ap.add_argument('--prompt',required=True)
ap.add_argument('--sampling-seed',type=int,required=True);a=ap.parse_args()
messages=[]
if a.previous:
    previous=json.loads(a.previous.read_text(encoding='utf8'))
    messages=previous['request']['messages']+[{'role':'assistant','content':previous['response']}]
messages.append({'role':'user','content':a.prompt})
a.output.parent.mkdir(parents=True,exist_ok=True)
assert not a.output.exists(),'Requests are immutable; choose a new output name'
pending=a.output.with_name('pending-'+a.output.stem+'.upload')
with pending.open('x',encoding='utf8') as f:
    json.dump({'messages':messages,'sampling_seed':a.sampling_seed},f,ensure_ascii=False,indent=2)
pending.replace(a.output)
print(a.output)
