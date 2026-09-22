"""Create a scoped upload bundle; never include credentials or unrelated files."""
import tarfile
from pathlib import Path
from data import HERE
from filter_v2 import sha,save_json
root=Path('D:/Interiority-V1/cloud');out=root/'upload.tar.gz'
with tarfile.open(out,'x:gz',compresslevel=2) as tar:
    for folder in ['base','sources','prepared/final','audits/final']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '.cache' not in p.parts:tar.add(p,arcname=str(p.relative_to(root)))
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.md','.txt','.sh']:
            tar.add(p,arcname='code/'+p.name)
save_json(root/'upload-manifest.json',{'file':out.name,'bytes':out.stat().st_size,'sha256':sha(out)})
print({'file':str(out),'bytes':out.stat().st_size,'sha256':sha(out)})
