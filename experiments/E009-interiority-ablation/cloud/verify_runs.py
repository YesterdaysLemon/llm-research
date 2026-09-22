"""Cross-condition integrity check after all training receipts exist."""
import argparse,json
from pathlib import Path
from filter_v2 import sha,save_json
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
receipts={}
for seed in [29017,29018]:
    rr=[]
    for arm in ['original','filtered','random_control']:
        p=root/f'results/{arm}-{seed}';r=json.loads((p/'train.json').read_text())
        assert r['status']=='complete' and len(r['steps'])==384 and r['steps'][-1]['epoch']==2
        assert r['initial_adapter_sha256']!=r['final_adapter_sha256']
        for n,h in r['adapter_files'].items():assert sha(p/'adapter'/n)==h
        assert r['manifest_sha256']==sha(root/'prepared/final/manifest.json')
        assert r['review_sha256']==sha(root/'audits/final/review.json')
        rr.append(r);receipts[f'{arm}-{seed}']={k:r[k] for k in ['initial_adapter_sha256','final_adapter_sha256','trainable_parameters','supervised_tokens_seen','input_tokens_seen','wall_seconds','peak_gpu_gib']}
    for field in ['initial_adapter_sha256','trainable_parameters','config','manifest_sha256','review_sha256']:
        assert all(r[field]==rr[0][field] for r in rr),field
    for n in ['run.py','data.py','filter_v2.py','audit.py']:
        assert len({r['environment']['code'][n] for r in rr})==1,n
    assert all(r['environment']['packages']==rr[0]['environment']['packages'] for r in rr)
    assert len({r['environment']['gpu'] for r in rr})==1
save_json(root/'paired-run-integrity.json',{'status':'passed','paired_initializations_identical':True,'runs':receipts,
    'limit':'CUDA arithmetic is not forced bitwise deterministic; hardware UUIDs and package records are retained.'})
print(json.dumps(receipts,indent=2))
