"""Bundle completed run artifacts, with a SHA-256 inventory, for local retrieval."""
import argparse,tarfile,json
from pathlib import Path
from filter_v2 import sha,save_json
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--name',default='study-artifacts');a=ap.parse_args();root=a.root
for arm in ['original','filtered','random_control']:
    assert json.loads((root/f'results/worker-{arm}-complete.json').read_text())['status']=='complete'
assert (root/'chat/stop').exists(), 'Finish the exploratory conversation before collecting'
files=[]
for folder in ['results','chat','code','sources','prepared/final','audits/final']:
    for p in (root/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts:files.append(p)
files += [p for p in (root/'base').iterdir() if p.is_file() and p.suffix!='.safetensors']
for pattern in ['*.log','*.json','*.txt']:
    files += [p for p in root.glob(pattern) if p.is_file() and not p.name.endswith('-inventory.json')]
files=sorted(set(files));manifest={str(p.relative_to(root)): {'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
inventory=root/f'{a.name}-inventory.json';save_json(inventory,{'files':manifest,'includes_base_weights':False,'includes_epoch_checkpoints':True})
archive=root/f'{a.name}.tar.gz'
with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
    for p in files+[inventory]:tar.add(p,arcname=str(p.relative_to(root)))
receipt={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'inventory_sha256':sha(inventory)}
save_json(root/f'{a.name}-transfer.json',receipt);print(json.dumps(receipt))
