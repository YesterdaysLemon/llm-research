"""Record completed manual review coverage, then freeze the checked candidate.

This is an attestation of the recorded interactive review, not an automated
semantic judge. Do not run for an unreviewed candidate or after changing inputs.
"""
import json,shutil,datetime
from pathlib import Path
from data import HERE,load
from filter_v2 import read_rows,save_json,save_rows,sha,digest
root=Path('D:/Interiority-V1/cloud');version='candidate-v7';data=root/'prepared'/version;audit=root/'audits'/version
rebuild=root/'prepared/rebuild-v7'
files={str(p.relative_to(data)).replace('\\','/'):sha(p) for p in data.rglob('*') if p.is_file()}
assert files=={str(p.relative_to(rebuild)).replace('\\','/'):sha(p) for p in rebuild.rglob('*') if p.is_file()}
packets=[root/'audits/candidate-v1/spotcheck.jsonl',root/'audits/candidate-v3/changed_rows.jsonl',root/'audits/candidate-v3/remaining_review.jsonl']
packets += [root/f'audits/candidate-v{n}/delta_review.jsonl' for n in range(4,8)]
done={r['id']:r for p in packets for r in read_rows(p)}
content=lambda r:digest(json.dumps({'messages':r['messages'],'response':r['response'],'source_ids':r['source_ids']},sort_keys=True,ensure_ascii=False))
for n in ['changed_rows','spotcheck']:
    for r in read_rows(audit/(n+'.jsonl')):assert r['id'] in done and content(r)==content(done[r['id']]),r['id']
contexts={}
for n in ['semantic_queue','extra_self_scan']:
    for r in read_rows(root/f'audits/candidate-v1/{n}.jsonl'):contexts[r['id']]=digest(r['prompt']+'\n'+r['response'])
for r in done.values():
    msgs=r['messages']+[{'role':'assistant','content':r['response']}]
    for i in range(1,len(msgs),2):
        mid=r['source_ids'][i] if r['source']=='oasst2' else r['id']
        contexts[mid]=digest(msgs[i-1]['content']+'\n'+msgs[i]['content'])
for n in ['semantic_queue','extra_self_scan']:
    for r in read_rows(audit/(n+'.jsonl')):assert contexts.get(r['id'])==digest(r['prompt']+'\n'+r['response']),r['id']
save_json(audit/'reconstruction.json',{'passed':True,'candidate':version,'independent_rebuild':'rebuild-v7','identical_files':files})
save_rows(audit/'full_review_coverage.jsonl',[{'id':i,'content_sha256':content(r)} for i,r in sorted(done.items())])
save_json(audit/'scan_review_coverage.json',{'context_hashes':contexts,'full_packets':{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in packets},
 'initial_scan_packets':{n:sha(root/f'audits/candidate-v1/{n}.jsonl') for n in ['semantic_queue','extra_self_scan']},
 'method':'All v1 broad-topic and extra-first-person snippets with prompts; escalated flagged full contexts; full random sample and all later delta/changed rows.'})
sourcecards={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in (root/'sources').glob('*/README.md')}
save_json(audit/'provenance.json',{'sources':load('config.json')['sources'],'source_cards':sourcecards,
 'model_id':load('config.json')['model_id'],'model_revision':load('config.json')['model_revision'],
 'model_card_sha256':sha(root/'base/README.md'),'model_license_sha256':sha(root/'base/LICENSE'),
 'attribution':'OpenAssistant contributors (OASST2, Apache-2.0); Databricks employee contributors (Dolly, CC-BY-SA-3.0); Qwen team (Qwen2.5-7B, Apache-2.0).',
 'publication':'Private research artifacts only; no source dataset or adapter publication authorized.'})
struct=json.loads((audit/'structural.json').read_text())
review={'status':'passed_for_bounded_study','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'manifest_sha256':sha(data/'manifest.json'),'structural_sha256':sha(audit/'structural.json'),
 'reviewer':'Codex, one agent; multiple checks are not independent human reviewers',
 'passes':{
 'provenance':{'passed':True,'notes':'Pinned sources, source cards, licenses, revisions and hashes checked; labels do not guarantee human authorship.'},
 'removed_examples':{'passed':True,'target_records':struct['removed_records'],'full_changed_records':len(read_rows(audit/'changed_rows.jsonl')),'notes':'Every target, control deletion and replacement inspected in full context; no synthetic rewrites.'},
 'independent_retained_review':{'passed':True,'topic_contexts':struct['semantic_records'],'extra_self_contexts':struct['extra_self_records'],'full_spotcheck':80,'notes':'Separate lexical/topic scans and full sample; all changed/new selected rows reviewed after common quarantines.'},
 'final_reconstruction':{'passed':True,'reconstructed_records':struct['source_reconstructed_records'],'notes':'Independent raw-source and token-boundary reconstruction; zero tree/prompt overlap; identical fresh rebuild.'}},
 'limitations':['Not exhaustive factual or near-duplicate verification.','Manual semantic judgments may be wrong; no recall guarantee.','Selected source labels cannot rule out undisclosed synthetic contributions.','Historical source claims, fictional identity content and general AI discussion remain.'],
 'quarantined_message_ids':len(load('quarantine.json')),'manual_overrides':len(load('overrides.json')),
 'evidence_files':{p.name:sha(p) for p in audit.iterdir() if p.is_file() and p.name!='structural.json'}}
save_json(audit/'review.json',review)
shutil.copytree(data,root/'prepared/final');shutil.copytree(audit,root/'audits/final')
save_json(HERE/'frozen-review.json',{k:v for k,v in review.items() if k!='evidence_files'})
(HERE/'audit-progress.json').write_text(json.dumps({'candidate':version,'status':'passed_for_bounded_study','manifest_sha256':review['manifest_sha256'],
 'agent_count':1,'full_reviewed_distinct_rows':len(done),'scope':review['passes'],'limitations':review['limitations']},indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps({'status':review['status'],'full_reviewed_distinct_rows':len(done),'manifest':review['manifest_sha256'],'review_sha256':sha(audit/'review.json')}))
