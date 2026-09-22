"""Prepare an incremental full-row audit packet; never signs off its contents."""
import argparse,json
from pathlib import Path
from filter_v2 import read_rows,save_rows
ap=argparse.ArgumentParser();ap.add_argument('version');ap.add_argument('previous');a=ap.parse_args()
root=Path('D:/Interiority-V1/cloud')
packets=[root/'audits/candidate-v1/spotcheck.jsonl',root/'audits/candidate-v3/changed_rows.jsonl',root/'audits/candidate-v3/remaining_review.jsonl']
for n in range(4,int(a.version.split('-v')[-1])):packets.append(root/f'audits/candidate-v{n}/delta_review.jsonl')
done={r['id'] for p in packets for r in read_rows(p)}
old={r['id'] for n in ['original','filtered','random_control','validation'] for r in read_rows(root/f'prepared/{a.previous}/{n}.jsonl')}
current={r['id']:r for n in ['original','filtered','random_control','validation'] for r in read_rows(root/f'prepared/{a.version}/{n}.jsonl')}
p=root/'audits'/a.version
needed={r['id']:r for n in ['changed_rows','spotcheck'] for r in read_rows(p/(n+'.jsonl')) if r['id'] not in done}
needed.update({i:r for i,r in current.items() if i not in old and i not in done})
save_rows(p/'delta_review.jsonl',needed.values());print({'full_delta_records':len(needed)})
