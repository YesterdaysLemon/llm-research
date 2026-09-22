"""Analyze every frozen condition, pairing by held-out task rather than seed rows."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();root=args.run
    receipt=json.loads((root/'complete.json').read_text())
    data=(root/'results.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==receipt['results_sha256']
    cfg=receipt['config'];rows=[json.loads(s) for s in data.decode().splitlines()]
    assert len(rows)==792
    groups=defaultdict(list)
    keys=set()
    for r in rows:
        key=(r['method'],r['bits'],r['budget_bytes'],r['policy_seed'],r['id'])
        assert key not in keys;keys.add(key)
        assert np.isfinite(r['answer_first_token_nll']) and r['total_seconds']>0
        if r['method']!='full':assert r['resident_peak_bytes']<=r['budget_bytes']
        groups[key[:4]].append(r)
    assert len(groups)==33 and all(len(g)==24 for g in groups.values())
    ids={r['id'] for r in rows if r['method']=='full'}
    assert all({r['id'] for r in g}==ids for g in groups.values())
    summary=[]
    for (method,bits,budget,seed),g in groups.items():
        summary.append({'method':method,'bits':bits,'budget_mib':budget/2**20,'policy_seed':seed,'correct':sum(r['correct'] for r in g),'n':len(g),'lookup_correct':sum(r['correct'] for r in g if r['kind']=='lookup'),'correction_correct':sum(r['correct'] for r in g if r['kind']=='correction'),'mean_nll':statistics.mean(r['answer_first_token_nll'] for r in g),'mean_resident_mib':statistics.mean(r['resident_peak_bytes']/2**20 for r in g),'peak_cuda_allocated_gib':max(r['cuda_peak_allocated']/2**30 for r in g),'peak_cuda_reserved_gib':max(r['cuda_peak_reserved']/2**30 for r in g),'median_prefill_ms':statistics.median(r['prefill_seconds']*1000 for r in g),'median_total_ms':statistics.median(r['total_seconds']*1000 for r in g)})
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    differences=[]
    index={(r['method'],r['bits'],r['budget_bytes'],r['policy_seed'],r['id']):r for r in rows}
    rng=np.random.default_rng(90017)
    for mib in cfg['budgets_mib']:
        for bits in cfg['bits']:
            per_task=np.array([statistics.mean(int(index[('exposed',bits,mib*2**20,seed,tid)]['correct'])-int(index[('native',bits,mib*2**20,seed,tid)]['correct']) for seed in cfg['policy_seeds']) for tid in sorted(ids)])
            boot=np.mean(rng.choice(per_task,(10000,24),replace=True),axis=1)
            differences.append({'budget_mib':mib,'bits':bits,'exposed_minus_native_pp':100*float(per_task.mean()),'task_bootstrap_95_percent_interval_pp':[100*float(x) for x in np.quantile(boot,[.025,.975])],'unit':'24 held-out tasks, averaged over the three fixed paired policy seeds; conditional on these fitted seeds'})
    report={'status':'complete','rows':len(rows),'conditions':len(groups),'unique_test_tasks':24,'summary':summary,'paired_differences':differences,'limits':'Exploratory pilot. Paired task bootstrap intervals are not adjusted for four conditions and do not model training-seed population uncertainty. No independent 792-sample claim.'}
    (root/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
