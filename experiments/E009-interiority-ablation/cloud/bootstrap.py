"""Fetch only pinned public base-model files and verify every weight shard."""
import argparse,json,time
from pathlib import Path
from huggingface_hub import snapshot_download
from data import load
from filter_v2 import sha,save_json
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
c=load('config.json');files=load('model-files.json');base=a.root/'base'
t=time.time()
snapshot_download(c['model_id'],revision=c['model_revision'],local_dir=base,
    allow_patterns=list(files)+['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
    'merges.txt','vocab.json','model.safetensors.index.json','LICENSE','README.md'],max_workers=4)
for name,expected in files.items():
    p=base/name;assert p.stat().st_size==expected['size'] and sha(p)==expected['sha256'],name
save_json(a.root/'model-verified.json',{'model':c['model_id'],'revision':c['model_revision'],'files':files,'seconds':time.time()-t})
print('PINNED_MODEL_VERIFIED',flush=True)
