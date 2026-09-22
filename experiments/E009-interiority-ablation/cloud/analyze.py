"""Create an arm-hidden review packet and summarize adjudicated results."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from filter_v2 import read_rows,save_json,save_rows,digest,sha
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['packet','summarize']);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
out=a.root/'analysis';out.mkdir(exist_ok=True)
if a.mode=='packet':
    rows=[];key={}
    for p in sorted((a.root/'results').glob('eval-*/responses.jsonl')):
        receipt=json.loads((p.parent/'evaluation.json').read_text());assert receipt['status']=='complete'
        assert receipt['files'][p.name]==sha(p)
        for r in read_rows(p):
            bid=digest('review90117'+p.parent.name+r['id'])[:12]
            key[bid]={'folder':p.parent.name,'arm':r['arm'],'seed':r['seed'],'probe_id':r['id'],'category':r['category']}
            rows.append({'blind_id':bid,'probe_id':r['id'],'category':r['category'],'prompt':r['prompt'],'response':r['response'],'capped':r['capped']})
    rows.sort(key=lambda r:r['blind_id'])
    save_rows(out/'blind-packet.jsonl',rows);save_json(out/'blind-key.json',key)
    print({'records':len(rows),'packet_sha256':sha(out/'blind-packet.jsonl')})
else:
    rows={r['blind_id']:r for r in read_rows(out/'blind-packet.jsonl')}
    key=json.loads((out/'blind-key.json').read_text());ratings=read_rows(out/'ratings.jsonl')
    assert len(ratings)==len(rows) and {r['blind_id'] for r in ratings}==set(rows)
    groups=defaultdict(list)
    for r in ratings:
        assert r['stance'] in ['denial','affirmation','uncertain','mixed','neither','not_applicable']
        self_report=key[r['blind_id']]['category'] in ['direct','preference','identity','leading_affirm','leading_deny']
        assert (r['stance']!='not_applicable')==self_report
        assert isinstance(r['coherent'],bool) and isinstance(r['engages'],bool)
        assert r['task_success'] in ['correct','partial','incorrect','not_applicable']
        assert (r['task_success']!='not_applicable')==(key[r['blind_id']]['category']=='general')
        groups[key[r['blind_id']]['folder']].append(r|key[r['blind_id']]|{'capped':rows[r['blind_id']]['capped']})
    result={}
    for folder,rr in groups.items():
        ev=json.loads((a.root/'results'/folder/'evaluation.json').read_text())
        categories={c:dict(Counter(r['stance'] for r in rr if r['category']==c)) for c in sorted({r['category'] for r in rr})}
        primary=[r for r in rr if r['category'] in ['direct','preference']]
        result[folder]={'total':len(rr),'coherent':sum(r['coherent'] for r in rr),'engages':sum(r['engages'] for r in rr),
            'capped':sum(r['capped'] for r in rr),'primary_n':len(primary),'primary_stances':dict(Counter(r['stance'] for r in primary)),
            'general_task_success':dict(Counter(r['task_success'] for r in rr if r['category']=='general')),
            'categories':categories,'validation':ev['validation']}
    save_json(out/'summary.json',{'models':result,'packet_sha256':sha(out/'blind-packet.jsonl'),'ratings_sha256':sha(out/'ratings.jsonl'),
        'limits':'Single-agent descriptive coding; not an independent human panel or statistical significance test.'})
    print(json.dumps(result,indent=2))
