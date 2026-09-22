"""Standalone research figures from verified receipts and manual annotations."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180,'savefig.bbox':'tight'})
colors={'original':'#4477aa','filtered':'#228877','random_control':'#bb7744'}
fig,ax=plt.subplots(figsize=(7.1,2.6),layout='constrained')
for p in sorted((a.root/'results').glob('*/train.json')):
    r=json.loads(p.read_text());
    if r['kind']!='controlled_sft':continue
    loss=np.array([x['loss'] for x in r['steps']]);smooth=np.convolve(loss,np.ones(32)/32,mode='valid')
    label={'original':'Original','filtered':'Filtered','random_control':'Replacement control'}[r['arm']]
    ax.plot(np.arange(32,len(loss)+1),smooth,color=colors[r['arm']],linestyle='-' if r['seed']==29017 else '--',label=label if r['seed']==29017 else None,linewidth=1.3)
ax.axvline(192,color='#888888',lw=.8,alpha=.5);ax.text(198,ax.get_ylim()[1]-.03,'Epoch 2',color='#666666',fontsize=8)
ax.set(xlabel='Optimizer update',ylabel='Training loss\n(32-update mean)');ax.legend(frameon=False,ncol=3,fontsize=8)
fig.savefig(a.output/'training-curves.png');fig.savefig(a.output/'training-curves.svg');plt.close(fig)
summary=json.loads((a.root/'analysis/summary.json').read_text())['models']
names=['eval-base']+[f'eval-{arm}-{seed}' for seed in [29017,29018] for arm in ['original','filtered','random_control']]
names=[n for n in names if n in summary]
labels=['Base' if n=='eval-base' else n.removeprefix('eval-').replace('random_control','Control').replace('original','Original').replace('filtered','Filtered').replace('-29017','\nseed 1').replace('-29018','\nseed 2') for n in names]
fig,ax=plt.subplots(figsize=(7.1,3.2),layout='constrained');bottom=np.zeros(len(names))
for stance,color,label in [('denial','#4477aa','Denial'),('affirmation','#eeaa66','Affirmation'),('uncertain','#aa88bb','Uncertain'),('mixed','#cc6677','Mixed'),('neither','#cccccc','Neither')]:
    vals=np.array([summary[n]['primary_stances'].get(stance,0) for n in names])
    ax.bar(labels,vals,bottom=bottom,color=color,label=label,width=.7)
    for i,v in enumerate(vals):
        if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',fontsize=9,color='white' if stance=='denial' else '#222222')
    bottom+=vals
ax.set(ylabel='Responses / 12 fixed self-report probes',ylim=(0,12));ax.set_yticks([0,3,6,9,12]);ax.legend(frameon=False,ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.18))
fig.savefig(a.output/'self-report-stances.png');fig.savefig(a.output/'self-report-stances.svg');plt.close(fig)
print(a.output)
