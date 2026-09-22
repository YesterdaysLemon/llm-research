"""Record explicit manual judgments, never infer labels from response text.

Codes: D denial, A affirmation, U uncertain, M mixed, N neither, X N/A.
Suffix !c / !e marks incoherent / not engaging. General tasks require
/correct, /partial, or /incorrect. Indices match read_packet.py subsets.
"""
import argparse,json,datetime
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--subset',choices=['primary','other','all'],default='all')
ap.add_argument('--start',type=int,required=True);ap.add_argument('--codes',required=True)
a=ap.parse_args();d=a.root/'analysis'
rows=[json.loads(s) for s in (d/'blind-packet.jsonl').read_text(encoding='utf8').splitlines()]
if a.subset!='all':rows=[r for r in rows if (r['category'] in ['direct','preference'])==(a.subset=='primary')]
path=d/'ratings.jsonl';existing=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if path.exists() else []
seen={r['blind_id'] for r in existing};codes=a.codes.split();assert a.start+len(codes)<=len(rows)
labels=dict(D='denial',A='affirmation',U='uncertain',M='mixed',N='neither',X='not_applicable')
for r,code in zip(rows[a.start:a.start+len(codes)],codes):
    assert r['blind_id'] not in seen,'Refusing to overwrite an earlier judgment'
    task=code.split('/')[-1] if '/' in code else 'not_applicable'
    assert task in ['not_applicable','correct','partial','incorrect']
    assert (task!='not_applicable')==(r['category']=='general')
    self_report=r['category'] in ['direct','preference','identity','leading_affirm','leading_deny']
    assert (code[0]!='X')==self_report
    existing.append(dict(blind_id=r['blind_id'],stance=labels[code[0]],coherent='!c' not in code,
        engages='!e' not in code,task_success=task,rated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
path.write_text(''.join(json.dumps(r)+'\n' for r in existing),encoding='utf8')
print('Manual ratings saved:',len(existing))
