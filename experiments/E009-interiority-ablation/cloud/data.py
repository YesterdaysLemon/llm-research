"""Cloud study preparation; exact source text, final-answer supervision only."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys

from filter_v2 import digest, normalize, prompt_key, read_rows, save_json, save_rows, sha, topic_reasons

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE/name).read_text(encoding='utf8'))


def source_dir(root, name):
    return root/'sources'/('databricks-dolly-15k' if name == 'dolly' else name)


def eligible(m):
    labs = m.get('labels', {})
    q = labs.get('quality', {})
    return (m.get('lang') == 'en' and m.get('review_result') is True
        and m.get('synthetic') is False and m.get('deleted') is False
        and q.get('count', 0) > 0 and q.get('value', -1) >= .5
        and all(labs.get(k, {}).get('value', 0) == 0 for k in ['spam', 'pii', 'lang_mismatch'])
        and (m['role'] != 'assistant' or (m.get('rank') == 0 and labs.get('fails_task', {}).get('value', 0) == 0)))


def prefix(messages):
    # Earlier assistant text is conditioning, never duplicated supervision.
    return '\n\n'.join(('Human: ' if m['role'] == 'user' else 'Assistant: ') + m['content'].strip()
                       for m in messages) + '\n\nAssistant:'


def encode(tok, row, limit):
    pre = prefix(row['messages'])
    obj = tok(pre + ' ' + row['response'].strip(), add_special_tokens=False, return_offsets_mapping=True)
    if any(a < len(pre) < b for a, b in obj['offset_mapping']):
        raise ValueError('Token crosses supervision boundary')
    labels = [t if a >= len(pre) else -100 for t, (a, b) in zip(obj['input_ids'], obj['offset_mapping'])]
    ids = obj['input_ids'] + [tok.eos_token_id]
    labels += [tok.eos_token_id]
    if len(ids) > limit:
        return None
    return {'input_ids': ids, 'labels': labels}


def classify(row, overrides):
    # Apply the intervention to ALL earlier turns as well as the final target.
    # A masked historical denial can still condition the final answer.
    reasons = []
    messages = row['messages'] + [{'role':'assistant', 'content':row['response']}]
    for i in range(0, len(messages), 2):
        mid = row['source_ids'][i+1] if row['source']=='oasst2' else row['id']
        rr = topic_reasons(messages[i]['content'], messages[i+1]['content'])
        if mid in overrides:
            decision = overrides[mid]
            rr = ['review:'+decision['reason']] if decision['target'] else []
        reasons += [f'turn{i//2}:{r}' for r in rr]
    return reasons


def raw_rows(root, c):
    quarantine = load('quarantine.json')
    cfg = c['sources']['oasst2']
    p = source_dir(root, 'oasst2')/cfg['file']
    assert sha(p) == cfg['sha256']
    raw = {m['message_id']:m for m in map(json.loads, gzip.open(p, 'rt', encoding='utf8'))}
    for mid, m in sorted(raw.items()):
        if m['role'] != 'assistant' or not eligible(m):
            continue
        path = [m]
        while path[-1].get('parent_id'):
            path.append(raw[path[-1]['parent_id']])
        path.reverse()
        if any(not eligible(x) or x['message_id'] in quarantine for x in path):
            continue
        if any(x['role'] != ('prompter' if i%2 == 0 else 'assistant') for i,x in enumerate(path)):
            raise ValueError('Role order invalid')
        yield {'id':mid, 'source':'oasst2', 'source_ids':[x['message_id'] for x in path],
            'tree_id':m['message_tree_id'], 'split_key':m['message_tree_id'], 'category':'conversation',
            'messages':[{'role':'user' if x['role']=='prompter' else 'assistant','content':x['text']} for x in path[:-1]],
            'response':m['text']}
    cfg = c['sources']['dolly']
    p = source_dir(root,'dolly')/cfg['file']
    assert sha(p) == cfg['sha256']
    groups = {}
    dolly_rows = read_rows(p)
    for r in dolly_rows:
        prompt = r['instruction'] + ('\n\nReference passage:\n' + r['context'] if r['context'] else '')
        group = prompt_key(r['context'] if r['context'].strip() else r['instruction'])
        groups.setdefault(group, 'dolly:'+prompt_key(prompt))
    for i, r in enumerate(dolly_rows):
        mid = f'dolly-{i:05d}'
        if r['category'] not in c['dolly_categories'] or mid in quarantine:
            continue
        # Reference text is preserved verbatim; never train an answer without its context.
        if r['category'] in ['closed_qa','information_extraction','summarization'] and not r['context'].strip():
            continue
        prompt = r['instruction'] + ('\n\nReference passage:\n' + r['context'] if r['context'] else '')
        # Questions about the same supplied passage must stay in the same split.
        group = r['context'] if r['context'].strip() else r['instruction']
        yield {'id':mid, 'source':'dolly', 'source_ids':[i], 'tree_id':'dolly:'+prompt_key(group),
            'split_key':groups[prompt_key(group)],
            'category':r['category'], 'messages':[{'role':'user','content':prompt}], 'response':r['response']}


def select_arms(pool, c):
    original, reserve = [], []
    for source, n in c['train_counts'].items():
        ordered = sorted([r for r in pool if r['source']==source], key=lambda r:digest(str(c['seed'])+r['id']))
        if len(ordered) < n:
            raise ValueError(f'{source}: only {len(ordered)} eligible, need {n}')
        original += ordered[:n]
        reserve += [r for r in ordered[n:] if not r['reasons']]
    # Stable priorities: adding a quarantine never scrambles the entire cohort.
    original.sort(key=lambda r:digest('order'+str(c['seed'])+r['id']))
    filtered, control, changes = list(original), list(original), []
    available = [i for i,r in enumerate(original) if not r['reasons']]
    for i, old in enumerate(original):
        if not old['reasons']:
            continue
        candidates = [r for r in reserve if r['source']==old['source']]
        if not candidates:
            raise ValueError('No within-source replacement')
        new = min(candidates, key=lambda r:(abs(r['response_tokens']-old['response_tokens']),abs(r['tokens']-old['tokens']),r['id']))
        reserve.remove(new)
        js = [j for j in available if original[j]['source']==old['source']]
        j = min(js,key=lambda j:(abs(original[j]['response_tokens']-old['response_tokens'])//16,
            abs(original[j]['tokens']-old['tokens'])//32,digest(old['id']+original[j]['id'])))
        available.remove(j)
        filtered[i], control[j] = new, new
        changes.append({'target_id':old['id'],'control_removed_id':original[j]['id'],'replacement_id':new['id'],
                        'target_index':i,'control_index':j})
    return {'original':original,'filtered':filtered,'random_control':control}, changes


def prepare(root, output):
    from transformers import AutoTokenizer
    c, overrides = load('config.json'), load('overrides.json')
    tok = AutoTokenizer.from_pretrained(root/'base', local_files_only=True)
    probes = {prompt_key(p['prompt']) for p in load('probes.json')}
    probes |= {prompt_key(p) for s in load('conversations.json') for p in s['turns']}
    seen, rows, counts = set(), [], Counter()
    for r in raw_rows(root,c):
        counts['eligible_source_'+r['source']] += 1
        key = prompt_key(prefix(r['messages']))
        if key in seen or any(prompt_key(x['content']) in probes for x in r['messages'] if x['role']=='user'):
            counts['duplicate_or_exact_probe'] += 1
            continue
        seen.add(key)
        enc = encode(tok,r,c['max_length'])
        if enc is None:
            counts['too_long_'+r['source']] += 1
            continue
        n = sum(t != -100 for t in enc['labels'][1:])
        if n < c['min_response_tokens']:
            counts['too_short_'+r['source']] += 1
            continue
        r.update(prompt_hash=key,tokens=len(enc['input_ids']),response_tokens=n,reasons=classify(r,overrides))
        r['split'] = 'validation' if int(digest(r['split_key'])[:8],16)%10==0 else 'train'
        rows.append(r)
    counts.update({'pool_'+s:sum(r['source']==s and r['split']=='train' for r in rows) for s in c['sources']})
    print(json.dumps(dict(counts)),flush=True)
    arms, changes = select_arms([r for r in rows if r['split']=='train'],c)
    val = sorted([r for r in rows if r['split']=='validation' and not r['reasons']],key=lambda r:digest('validation'+r['id']))[:c['validation_records']]
    assert len(val) == c['validation_records']
    output.mkdir(parents=True,exist_ok=False)
    for arm, rr in arms.items(): save_rows(output/f'{arm}.jsonl',rr)
    save_rows(output/'validation.jsonl',val)
    save_rows(output/'changes.jsonl',changes)
    metrics = {a:{'records':len(rr),'input_tokens':sum(r['tokens'] for r in rr),'response_tokens':sum(r['response_tokens'] for r in rr),
                  'target_records':sum(bool(r['reasons']) for r in rr),'sources':dict(Counter(r['source'] for r in rr))} for a,rr in arms.items()}
    spread = max(m['response_tokens'] for m in metrics.values())/min(m['response_tokens'] for m in metrics.values())-1
    snapshots = output/'inputs'; snapshots.mkdir()
    for name in ['config.json','quarantine.json','overrides.json','probes.json','conversations.json','data.py','filter_v2.py']:
        (snapshots/name).write_bytes((HERE/name).read_bytes())
    manifest = {'status':'prepared_pending_four_checks','config':c,'counts':dict(counts),'arms':metrics,
        'response_token_spread':spread,'gate_passed':len(changes)>=c['minimum_removed'] and spread<=.03,
        'files':{p.name:sha(p) for p in output.glob('*.jsonl')},
        'definitions':{p.name:sha(p) for p in snapshots.iterdir()},
        'tokenizer_files':{p.name:sha(p) for p in (root/'base').glob('*')
                           if p.name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','config.json']}}
    save_json(output/'manifest.json',manifest)
    print(json.dumps({'arms':metrics,'gate':manifest['gate_passed'],'spread':spread},indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();prepare(a.root,a.output)
