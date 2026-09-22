"""Verify local archive and every extracted file before cloud teardown."""
import argparse,json,tarfile
from pathlib import Path
from filter_v2 import sha,save_json
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--name',default='study-artifacts');a=ap.parse_args();root=a.root
r=json.loads((root/f'{a.name}-transfer.json').read_text());archive=root/r['archive']
assert archive.stat().st_size==r['bytes'] and sha(archive)==r['sha256']
dest=root/'retrieved';dest.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
inv=dest/f'{a.name}-inventory.json';assert sha(inv)==r['inventory_sha256']
files=json.loads(inv.read_text())['files']
for name,info in files.items():
    p=dest/name;assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
save_json(root/'retrieval-verified.json',{'status':'verified','archive_sha256':r['sha256'],'files_verified':len(files),'bytes_verified':sum(x['bytes'] for x in files.values())})
print('RETRIEVAL_VERIFIED',len(files))
