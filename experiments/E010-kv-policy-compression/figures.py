import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=json.loads((root/'results/run-01/analysis.json').read_text())['summary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,4,figsize=(12,3.5),sharey=True,constrained_layout=True)
for ax,(mib,bits) in zip(axes,[(4,16),(4,8),(8,16),(8,8)]):
    for i,(method,color) in enumerate([('recency','#98a390'),('random','#b0a48a'),('native','#253c3c'),('exposed','#ad5239')]):
        ys=[r['correct']/24*100 for r in rows if r['method']==method and r['bits']==bits and r['budget_mib']==mib]
        ax.bar(i,np.mean(ys),color=color,width=.65,alpha=.8)
        ax.scatter(np.linspace(i-.14,i+.14,len(ys)),ys,c=color,edgecolor='white',s=32,zorder=3)
    ax.axhline(100,c='#73796f',ls='--',lw=1)
    ax.set_title(f'{mib} MiB / '+('BF16' if bits==16 else 'INT8'))
    ax.set_xticks(range(4),['Recency','Random','Native','Exposed'],rotation=35,ha='right')
    ax.set_ylim(0,108);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('First-word accuracy (%)')
fig.suptitle('Primary endpoint: answer format and retrieval are jointly required',fontsize=12)
dest=root/'figures';dest.mkdir(exist_ok=True)
fig.savefig(dest/'accuracy.png',dpi=180)
fig.savefig(dest/'accuracy.svg')
