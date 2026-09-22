"""Compact progress only: never display generated behavioral answers."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
for p in sorted((a.root/'results').glob('*')):
    if not p.is_dir() or p.name=='smoke':continue
    if (p/'train.json').exists():
        r=json.loads((p/'train.json').read_text());print(p.name,r['status'],'seconds',round(r['wall_seconds'],1))
    elif (p/'steps.jsonl').exists():
        lines=(p/'steps.jsonl').read_text().splitlines()
        if lines:
            try:r=json.loads(lines[-1])
            except json.JSONDecodeError:r=json.loads(lines[-2])
            print(p.name,'step',r['step'],'/384','epoch',r['epoch'],'loss',round(r['loss'],4))
    elif (p/'started.json').exists():print(p.name,'loading')
    if (p/'evaluation.json').exists():print(p.name,'evaluation complete')
    elif (p/'responses.jsonl').exists():print(p.name,'probes',len((p/'responses.jsonl').read_text().splitlines()),'/42')
    if (p/'failed.json').exists():print('FAILED',p.name)
