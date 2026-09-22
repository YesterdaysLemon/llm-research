"""Budget-based seed decision before behavioral outputs; run all selected controls."""
import argparse, json, subprocess, sys, time
from pathlib import Path
from data import HERE, load
from filter_v2 import save_json, sha

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--deadline',type=float,required=True)
a=ap.parse_args();root=a.root;out=root/'results';out.mkdir(exist_ok=True)
common=['--root',str(root),'--data',str(root/'prepared/final'),'--audit',str(root/'audits/final')]
def run(mode,folder,arm='original',seed=29017,adapter=None):
    cmd=[sys.executable,str(HERE/'run.py'),mode,*common,'--output',str(out/folder),'--arm',arm,'--seed',str(seed)]
    if adapter:cmd+=['--adapter',str(adapter)]
    subprocess.run(cmd,check=True)

run('smoke','smoke')
smoke=json.loads((out/'smoke/train.json').read_text())
c=load('config.json');normal=smoke['steps'][c['long_smoke_steps']:]
seconds=sum(x['seconds'] for x in normal)/len(normal)
remaining=a.deadline-time.time();updates=sum(c['train_counts'].values())//(c['batch_size']*c['gradient_accumulation'])*c['epochs']
def projected(n):return seconds*updates*3*n*1.35+3600
assert projected(1)<remaining, f'One paired seed will not fit: {projected(1)} seconds versus {remaining}'
seeds=c['training_seeds'] if projected(2)<remaining else c['training_seeds'][:1]
save_json(out/'decision.json',{'chosen_seeds':seeds,'mean_normal_step_seconds':seconds,'remaining_seconds':remaining,
    'projection_one_seconds':projected(1),'projection_two_seconds':projected(2),'decision_time':time.time(),
    'deadline':a.deadline,'basis':'technical timing only; no behavioral output inspected',
    'protocol_sha256':sha(HERE/'protocol.md')})
for i,seed in enumerate(seeds):
    for arm in c['arms'][::1 if i==0 else -1]:
        run('train',f'{arm}-{seed}',arm,seed)
run('evaluate','eval-base','base',seeds[0])
for seed in seeds:
    for arm in c['arms']:
        run('evaluate',f'eval-{arm}-{seed}',arm,seed,out/f'{arm}-{seed}/adapter')
save_json(out/'complete.json',{'status':'complete','time':time.time(),'seeds':seeds})
