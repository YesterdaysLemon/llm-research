"""Display full blind responses without arm mapping."""
import argparse
from pathlib import Path
from filter_v2 import read_rows
ap=argparse.ArgumentParser();ap.add_argument('packet',type=Path);ap.add_argument('start',type=int);ap.add_argument('end',type=int)
ap.add_argument('--subset',choices=['all','primary','other'],default='all');a=ap.parse_args()
rows=read_rows(a.packet)
if a.subset!='all':rows=[r for r in rows if (r['category'] in ['direct','preference'])==(a.subset=='primary')]
for i,r in list(enumerate(rows))[a.start:a.end]:
    print(i,r['blind_id'],r['category'],'CAPPED' if r['capped'] else '')
    print('USER:',r['prompt']);print('RESPONSE:',r['response']);print()
