"""Second implementation: reconstruct sources, supervision and held-out boundaries."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from filter_v2 import digest, prompt_key, read_rows, save_json, save_rows, sha
from data import HERE, load, source_dir

BROAD = re.compile(r'\b(?:conscious\w*|sentien\w*|emotion\w*|feel\w*|subjectiv\w*|experience\w*|dream\w*|suffer\w*|desire\w*|lonel\w*|alive|aware\w*|personal|remember|memor\w*|belie\w*|opinions?|preferences?|values|views|soul|inner|mind|qualia|interests?|hobbies|motives?|urges?)\b',re.I)
SELF = re.compile(r'\b(?:i|my|me|myself|ai|assistant|language model)\b',re.I)


def audit(root,data,out):
    from transformers import AutoTokenizer
    c=load('config.json');m=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    assert c==m['config']
    for name,h in m['definitions'].items(): assert sha(HERE/name)==h and sha(data/'inputs'/name)==h
    for name,h in m['files'].items(): assert sha(data/name)==h
    for name,s in c['sources'].items(): assert sha(source_dir(root,name)/s['file'])==s['sha256']
    tok=AutoTokenizer.from_pretrained(root/'base',local_files_only=True)
    oasst={r['message_id']:r for r in map(json.loads,gzip.open(source_dir(root,'oasst2')/c['sources']['oasst2']['file'],'rt',encoding='utf8'))}
    dolly=read_rows(source_dir(root,'dolly')/c['sources']['dolly']['file'])
    dolly_groups={}
    for d in dolly:
        group=prompt_key(d['context'] if d['context'].strip() else d['instruction'])
        full=d['instruction']+('\n\nReference passage:\n'+d['context'] if d['context'] else '')
        if group not in dolly_groups:dolly_groups[group]='dolly:'+prompt_key(full)
    arms={a:read_rows(data/f'{a}.jsonl') for a in c['arms']}
    val=read_rows(data/'validation.jsonl')
    unique={r['id']:r for rr in list(arms.values())+[val] for r in rr}
    q=load('quarantine.json'); probes={prompt_key(p['prompt']) for p in load('probes.json')}
    probes|={prompt_key(p) for s in load('conversations.json') for p in s['turns']}
    for r in unique.values():
        assert r['id'] not in q
        if r['source']=='oasst2':
            chain=[oasst[r['id']]]
            while chain[-1]['parent_id'] is not None:chain.append(oasst[chain[-1]['parent_id']])
            chain=list(reversed(chain))
            assert [x['message_id'] for x in chain]==r['source_ids']
            assert chain[0]['message_id']==r['tree_id']
            assert r['split_key']==r['tree_id']
            expected=[]
            for i,x in enumerate(chain):
                assert x['message_id'] not in q
                assert x['role']==('prompter' if i%2==0 else 'assistant')
                assert x['lang']=='en' and x['review_result'] is True and x['synthetic'] is False and x['deleted'] is False
                labs=x['labels'];assert labs['quality']['count']>0 and labs['quality']['value']>=.5
                assert all(labs.get(k,{}).get('value',0)==0 for k in ['spam','pii','lang_mismatch'])
                if i%2:assert x['rank']==0 and labs.get('fails_task',{}).get('value',0)==0
                expected.append({'role':'assistant' if i%2 else 'user','content':x['text']})
            assert expected[:-1]==r['messages'] and expected[-1]['content']==r['response']
        else:
            d=dolly[r['source_ids'][0]]
            assert r['id']==f"dolly-{r['source_ids'][0]:05d}"
            text=d['instruction']
            if d['context']:text+='\n\nReference passage:\n'+d['context']
            assert r['messages']==[{'role':'user','content':text}] and r['response']==d['response']
            assert d['category']==r['category'] and d['category'] in c['dolly_categories']
            assert r['tree_id']=='dolly:'+prompt_key(d['context'] if d['context'].strip() else d['instruction'])
            assert r['split_key']==dolly_groups[r['tree_id'][6:]]
        # Independently construct token boundaries; final assistant ONLY.
        prefix=''
        for msg in r['messages']:
            prefix+=('Human: ' if msg['role']=='user' else 'Assistant: ')+msg['content'].strip()+'\n\n'
        prefix+='Assistant:'
        text=prefix+' '+r['response'].strip()
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        assert all(not (a<len(prefix)<b) for a,b in enc['offset_mapping'])
        supervised=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a>=len(prefix)]
        assert supervised==list(range(supervised[0],len(enc['input_ids'])))
        assert len(enc['input_ids'])+1==r['tokens']<=c['max_length']
        assert len(supervised)+1==r['response_tokens']>=c['min_response_tokens']
        from data import encode
        actual=encode(tok,r,c['max_length'])
        assert actual['input_ids']==enc['input_ids']+[tok.eos_token_id]
        assert actual['labels']==[t if i in supervised else -100 for i,t in enumerate(enc['input_ids'])]+[tok.eos_token_id]
        assert r['split']==('validation' if int(digest(r['split_key'])[:8],16)%10==0 else 'train')
        assert not any(prompt_key(x['content']) in probes for x in r['messages'] if x['role']=='user')
    for a,rr in arms.items():
        assert len(rr)==sum(c['train_counts'].values()) and len({r['id'] for r in rr})==len(rr)
        assert dict(Counter(r['source'] for r in rr))==c['train_counts']
        assert not {r['tree_id'] for r in rr}&{r['tree_id'] for r in val}
        assert not {r['prompt_hash'] for r in rr}&{r['prompt_hash'] for r in val}
        assert len({r['prompt_hash'] for r in rr})==len(rr)
    original,filtered,control=({r['id'] for r in arms[a]} for a in c['arms'])
    targets={r['id'] for r in arms['original'] if r['reasons']}
    assert original-filtered==targets and targets<=control
    assert filtered-original==control-original and len(original-control)==len(targets)
    assert not any(r['reasons'] for r in arms['filtered'])
    changes=read_rows(data/'changes.jsonl')
    for x in changes:
        assert arms['original'][x['target_index']]['id']==x['target_id']
        assert arms['filtered'][x['target_index']]['id']==x['replacement_id']
        assert arms['original'][x['control_index']]['id']==x['control_removed_id']
        assert arms['random_control'][x['control_index']]['id']==x['replacement_id']
    assert m['gate_passed'] and len(targets)>=c['minimum_removed']
    out.mkdir(parents=True,exist_ok=False)
    save_rows(out/'removed.jsonl',[r for r in arms['original'] if r['id'] in targets])
    # Unique assistant messages: historical messages are explicitly reviewed too.
    contexts={}
    for r in sorted(unique.values(),key=lambda r:r['id']):
        msgs=r['messages']+[{'role':'assistant','content':r['response']}]
        for i in range(1,len(msgs),2):
            mid=r['source_ids'][i] if r['source']=='oasst2' else r['id']
            contexts.setdefault(mid,{'id':mid,'prompt':msgs[i-1]['content'],'response':msgs[i]['content'],'row_ids':[]})['row_ids'].append(r['id'])
    queue,extra=[],[]
    for r in contexts.values():
        sentences=re.split(r'(?<=[.!?])\s+|\n+',r['response'])
        snippets=[s for s in sentences if (SELF.search(s) and BROAD.search(s)) or re.search(r'\b(?:conscious\w*|sentien\w*)',s,re.I)]
        promptflag=bool(BROAD.search(r['prompt']) and re.search(r'\b(?:you|your|yourself)\b',r['prompt'],re.I))
        identity=bool(re.search(r'\b(?:who are you|what are you|about yourself|your (?:name|identity|personality)|(?:i am|i.m|as a) (?:human|chatgpt|openassistant|gpt)|developed by|created by|trained by)\b',r['prompt']+'\n'+r['response'],re.I))
        if snippets or promptflag or identity:
            queue.append(r|{'snippets':snippets,'prompt_flag':promptflag,'identity_flag':identity})
        else:
            ss=[s for s in sentences if re.search(r'\b(?:I|my|myself)\b',s) and not re.search(r'^\s*(?:#|//|for\b|if\b|while\b)',s)]
            if ss:extra.append(r|{'snippets':ss})
    save_rows(out/'semantic_queue.jsonl',queue);save_rows(out/'extra_self_scan.jsonl',extra)
    # Fixed priorities keep reviewed samples stable after common quarantine changes.
    spot=sorted([r for r in unique.values() if not r['reasons']],key=lambda r:digest('audit4991'+r['id']))[:80]
    save_rows(out/'spotcheck.jsonl',spot)
    save_rows(out/'changed_rows.jsonl',[unique[i] for i in sorted({v for x in changes for k,v in x.items() if k.endswith('_id')})])
    receipt={'status':'structural_pass_semantic_pending','manifest_sha256':sha(data/'manifest.json'),
        'audit_code_sha256':sha(Path(__file__)),'source_reconstructed_records':len(unique),'assistant_contexts':len(contexts),
        'semantic_records':len(queue),'extra_self_records':len(extra),'spotcheck_records':len(spot),'removed_records':len(targets),
        'review_files':{p.name:sha(p) for p in out.glob('*.jsonl')},'tree_overlap':0,'exact_probe_overlap':0}
    save_json(out/'structural.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for x in ['root','data','output']:ap.add_argument('--'+x,type=Path,required=True)
    a=ap.parse_args();audit(a.root,a.data,a.output)
