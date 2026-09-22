"""Small local inspection utility, prints complete selected records without JSON escaping."""
import argparse, gzip, json
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('file',type=Path)
ap.add_argument('--start',type=int,default=0)
ap.add_argument('--end',type=int,default=10)
ap.add_argument('--ids',nargs='*')
ap.add_argument('--ancestors',action='store_true')
a=ap.parse_args()
op=gzip.open if a.file.suffix=='.gz' else open
with op(a.file,'rt',encoding='utf8') as f:rs=[json.loads(x) for x in f]
if a.ancestors:
 lookup={r['message_id']:r for r in rs}
 for mid in a.ids:
  chain=[lookup[mid]]
  while chain[-1]['parent_id']:chain.append(lookup[chain[-1]['parent_id']])
  for r in reversed(chain):print(r['message_id'],r['role'],r['text'],'\n')
else:
 for i,r in enumerate(rs):
  if (a.ids and r['id'] in a.ids) or (not a.ids and a.start<=i<a.end):
   print(i,r['id'])
   if 'messages' in r:
    for m in r['messages']:print(m['role']+':',m['content'])
   else:print('user:',r['prompt'])
   print('assistant:',r['response'],'\n')
