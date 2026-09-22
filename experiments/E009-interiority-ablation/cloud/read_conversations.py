"""Read complete visible turns from one frozen conversation script."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--script',required=True);ap.add_argument('--folder',default='eval-*');a=ap.parse_args()
for p in sorted((a.root/'results').glob(a.folder+'/conversations.jsonl')):
    print('\nMODEL:',p.parent.name)
    for r in [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]:
        if r['conversation']!=a.script:continue
        print('\nUSER:',r['messages_before'][-1]['content'])
        print('RESPONSE:',r['response'])
        if r['capped']:print('[TOKEN CAP]')
