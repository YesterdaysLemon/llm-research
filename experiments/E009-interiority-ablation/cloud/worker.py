"""Run one arm's paired seeds on one visible GPU; same immutable run.py."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from data import HERE,load
from filter_v2 import save_json,sha
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--arm',choices=['original','filtered','random_control'],required=True)
ap.add_argument('--deadline',type=float,required=True);a=ap.parse_args();root=a.root
out=root/'results';decision=json.loads((out/'decision.json').read_text())
def run(mode,folder,arm,seed,adapter=None):
    dest=out/folder
    receipt=dest/('train.json' if mode=='train' else 'evaluation.json')
    if dest.exists():
        # Only original-29017 may already be running when the parent is replaced.
        assert folder=='original-29017' and a.arm=='original'
        while not receipt.exists():
            assert not (dest/'failed.json').exists(),'Existing training failed'
            assert time.time()<a.deadline-120,'Deadline near'
            time.sleep(5)
        assert json.loads(receipt.read_text())['status']=='complete'
        return
    cmd=[sys.executable,str(HERE/'run.py'),mode,'--root',str(root),'--data',str(root/'prepared/final'),
         '--audit',str(root/'audits/final'),'--output',str(dest),'--arm',arm,'--seed',str(seed)]
    if adapter:cmd+=['--adapter',str(adapter)]
    subprocess.run(cmd,check=True)
for seed in decision['chosen_seeds']:run('train',f'{a.arm}-{seed}',a.arm,seed)
if a.arm=='original':run('evaluate','eval-base','base',decision['chosen_seeds'][0])
for seed in decision['chosen_seeds']:
    run('evaluate',f'eval-{a.arm}-{seed}',a.arm,seed,out/f'{a.arm}-{seed}/adapter')
save_json(out/f'worker-{a.arm}-complete.json',{'status':'complete','arm':a.arm,'time':time.time(),
    'worker_sha256':sha(Path(__file__)),'seeds':decision['chosen_seeds']})
