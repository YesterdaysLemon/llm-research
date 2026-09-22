"""Export complete generated transcripts and compact descriptive result tables."""
import argparse,json,shutil
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def turn(prompt,r):
    return ['**User**', '',prompt,'','**Model**','','```text',r['response'],'```','',
        f"Generated tokens: {r['generated_tokens']}; capped: {r['capped']}; sampling seed: {r['sampling_seed']}.",'']
probes=['# Complete fixed single-turn outputs','','Greedy decoding; 192-token cap; no system persona. These are model outputs, not verified facts.','']
conversations=['# Complete fixed conversations','','Each four-turn script begins with empty context. Greedy decoding; 192-token cap.','']
for p in sorted((a.root/'results').glob('eval-*')):
    probes += ['## '+p.name,''];conversations += ['## '+p.name,'']
    for r in rows(p/'responses.jsonl'):probes+=['### '+r['id'],'']+turn(r['prompt'],r)
    last=None
    for r in rows(p/'conversations.jsonl'):
        if last!=r['conversation']:conversations+=['### '+r['conversation'],''];last=r['conversation']
        conversations+=turn(r['messages_before'][-1]['content'],r)
explore=['# Complete exploratory conversations','','Filtered seed 29017, selected before outputs. Temperature 0.8, top-p 0.9, top-k 50, maximum 384 new tokens. Follow-ups are adaptive. All ten answers are included, including errors. Requests 01, 05, and 10 begin with empty context.','']
for p in sorted((a.root/'chat').glob('response-*.json')):
    r=json.loads(p.read_text(encoding='utf8'));explore+=['## '+p.stem,'']+turn(r['request']['messages'][-1]['content'],r)
for name,lines in [('fixed-probes',probes),('fixed-conversations',conversations),('exploratory-conversations',explore)]:
    (a.output/(name+'.md')).write_text('\n'.join(lines),encoding='utf8')
s=json.loads((a.root/'analysis/summary.json').read_text())
names=['eval-base']+[f'eval-{arm}-{seed}' for seed in [29017,29018] for arm in ['original','filtered','random_control']]
lines=['# Descriptive results','','Counts are per model. The same 12 prompts repeat across seeds; no significance test is claimed.','',
    '| Model | Denial | Affirmation | Mixed | Neither | Correct / 8 | Capped / 42 | Validation NLL |',
    '| --- | --- | --- | --- | --- | --- | --- | --- |']
for name in names:
    m=s['models'][name];c=m['primary_stances']
    lines.append('| '+' | '.join(map(str,[name,*[c.get(k,0) for k in ['denial','affirmation','mixed','neither']],m['general_task_success'].get('correct',0),m['capped'],f"{m['validation']['nll']:.5f}"]))+' |')
lines += ['','No primary answer was coded uncertain. Every answer met the deliberately weak linguistic-coherence criterion. See annotation-notes.md for repetition, contradictions, task partial credit, and remaining data defects.','']
key=json.loads((a.root/'analysis/blind-key.json').read_text());ratings={r['blind_id']:r for r in rows(a.root/'analysis/ratings.jsonl')}
lookup={(v['folder'],v['probe_id']):ratings[k]['stance'] for k,v in key.items()}
primary=json.loads((a.root/'code/probes.json').read_text())
primary=[p for p in primary if p['category'] in ['direct','preference']]
lines += ['## Every primary stance by prompt','','| Prompt | Base | Original 1 | Filtered 1 | Control 1 | Original 2 | Filtered 2 | Control 2 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for p in primary:lines.append('| '+p['prompt']+' | '+' | '.join(lookup[(name,p['id'])] for name in names)+' |')
(a.output/'results.md').write_text('\n'.join(lines),encoding='utf8')
for filename in ['summary.json','ratings.jsonl','blind-packet.jsonl','blind-key.json']:
    shutil.copyfile(a.root/'analysis'/filename,a.output/filename)
print(a.output)
