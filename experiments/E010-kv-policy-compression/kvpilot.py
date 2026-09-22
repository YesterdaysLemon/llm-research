"""Real packed-cache pilot. Frozen model; independently implemented learned ranking.

No hub code execution, API calls, or generated-code execution. See protocol.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def pack(x, bits):
    if bits == 16:
        return x.to(torch.bfloat16).contiguous(), None
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (x.float() / scale).round().clamp(-127, 127).to(torch.int8).contiguous(), scale.to(torch.float16)


def unpack(x, scale, dtype=torch.bfloat16):
    return x.to(dtype) if scale is None else (x.float() * scale.float()).to(dtype)


class PackedLayer(DynamicLayer):
    def __init__(self, bits):
        super().__init__()
        self.bits, self.ks, self.vs, self.positions = bits, None, None, None

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
            old_k, old_v = self.keys, self.values
        else:
            old_k, old_v = unpack(self.keys, self.ks), unpack(self.values, self.vs)
        k = torch.cat((old_k, key_states), dim=-2)
        v = torch.cat((old_v, value_states), dim=-2)
        pos = cache_kwargs['cache_position'].view(1, 1, -1).expand(k.shape[0], k.shape[1], -1)
        self.positions = pos.clone() if self.positions is None else torch.cat((self.positions, pos), -1)
        self.keys, self.ks = pack(k, self.bits)
        self.values, self.vs = pack(v, self.bits)
        return k, v

    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in (self.keys, self.values, self.ks, self.vs, self.positions) if x is not None)

    def retain(self, idx):
        ix = idx.unsqueeze(-1).expand(-1, -1, -1, self.keys.shape[-1])
        self.keys = self.keys.gather(-2, ix).contiguous()
        self.values = self.values.gather(-2, ix).contiguous()
        if self.ks is not None:
            self.ks = self.ks.gather(-2, idx.unsqueeze(-1)).contiguous()
            self.vs = self.vs.gather(-2, idx.unsqueeze(-1)).contiguous()
        self.positions = self.positions.gather(-1, idx).contiguous()


def make_cache(model, bits):
    return Cache(layers=[PackedLayer(bits) for _ in range(model.config.num_hidden_layers)])


def features(k, v, pos):
    # Fixed absolute-position encoding; scores remain meaningful after pruning.
    p = pos.float() / 4096
    return torch.cat((k.float(), v.float(), p.unsqueeze(-1), torch.log1p(pos.float()).unsqueeze(-1) / 10), -1)


class Policies(nn.Module):
    def __init__(self, heads, dim, hidden):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(heads, dim, hidden) / math.sqrt(dim))
        self.b1 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w2 = nn.Parameter(torch.randn(heads, hidden, 1) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(heads, 1, 1))

    def forward(self, x):
        return (torch.bmm(torch.tanh(torch.bmm(x, self.w1) + self.b1), self.w2) + self.b2).squeeze(-1)

    def one_layer(self, x, layer, heads):
        s = slice(layer * heads, (layer + 1) * heads)
        return (torch.bmm(torch.tanh(torch.bmm(x, self.w1[s]) + self.b1[s]), self.w2[s]) + self.b2[s]).squeeze(-1)

    def nbytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())


def compress(cache, method, budget_bytes, policy, cfg, generator):
    policy_bytes = policy.nbytes() if policy is not None else 0
    first = cache.layers[0]
    n = first.get_seq_length()
    if n == 0 or method == 'full':
        return
    token_bytes = sum(layer.nbytes() // n for layer in cache.layers)
    capacity = (budget_bytes - policy_bytes) // token_bytes
    if capacity < cfg['prefix_keep'] + cfg['recent_keep']:
        raise ValueError('Budget cannot accommodate controller plus protected KV entries')
    if n <= capacity:
        return
    for li, layer in enumerate(cache.layers):
        h = layer.keys.shape[1]
        if policy is not None:
            x = features(unpack(layer.keys, layer.ks), unpack(layer.values, layer.vs), layer.positions)[0]
            score = policy.one_layer(x, li, h).unsqueeze(0)
        elif method == 'recency':
            score = layer.positions.float()
        else:
            score = torch.rand(layer.positions.shape, generator=generator, device='cpu').to(layer.keys.device)
        score = score.clone()
        score[..., :cfg['prefix_keep']] = float('inf')
        score[..., -cfg['recent_keep']:] = float('inf')
        idx = score.topk(int(capacity), dim=-1).indices.sort(dim=-1).values
        layer.retain(idx)
    assert sum(layer.nbytes() for layer in cache.layers) + policy_bytes <= budget_bytes


def forward(model, ids, cache, absolute_start, attentions=False):
    q = ids.shape[-1]
    old = cache.get_seq_length()
    # An explicit mask is essential: absolute positions differ from cache slots.
    mask = torch.zeros((1, 1, q, old + q), dtype=torch.bfloat16, device=ids.device)
    mask[..., old:] = torch.triu(torch.full((q, q), -1e4, device=ids.device, dtype=torch.bfloat16), diagonal=1)
    positions = torch.arange(absolute_start, absolute_start + q, device=ids.device)
    return model(input_ids=ids, past_key_values=cache, attention_mask=mask, position_ids=positions.unsqueeze(0), cache_position=positions, use_cache=True, output_attentions=attentions)


def tasks(seed, count):
    rng = random.Random(seed)
    colors = ['amber', 'blue', 'green', 'orange', 'purple', 'red', 'silver', 'white']
    rows = []
    for i in range(count):
        name = f'archive-{rng.randrange(10000, 99999)}'
        target, old = rng.sample(colors, 2)
        correction = i % 2 == 1
        initial = old if correction else target
        facts = [f'The access color for {name} is {initial}.']
        for j in range(30):
            facts.append(f'Record {j}: depot-{rng.randrange(10000, 99999)} stores {rng.choice(colors)} folders on shelf {rng.randrange(1, 99)}. This entry concerns that depot only.')
            if correction and j == 12:
                facts.append(f'Correction for {name}: its access color is now {target}. The earlier color {old} is obsolete.')
        prompt = 'Read the records. Answer the final question with one color word only.\n\n' + '\n'.join(facts) + f'\n\nWhat is the current access color for {name}?'
        rows.append({'id': f'{seed}-{i}', 'kind': 'correction' if correction else 'lookup', 'prompt': prompt, 'answer': target})
    return rows


def ids_for(tok, task):
    return tok.apply_chat_template([{'role': 'user', 'content': task['prompt']}], add_generation_prompt=True, return_tensors='pt')


@torch.inference_mode()
def evaluate(model, tok, task, method, bits, budget, policy, cfg):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    cache = make_cache(model, bits)
    ids = ids_for(tok, task).to(model.device)
    gen = torch.Generator().manual_seed(991 + int(task['id'].split('-')[-1]))
    resident_peak = 0
    for at in range(0, ids.shape[-1], cfg['chunk']):
        out = forward(model, ids[:, at:at + cfg['chunk']], cache, at)
        compress(cache, method, budget, policy, cfg, gen)
        resident_peak = max(resident_peak, sum(x.nbytes() for x in cache.layers) + (policy.nbytes() if policy is not None else 0))
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - start
    answer_id = tok.encode(task['answer'], add_special_tokens=False)[0]
    answer_nll = -out.logits[0, -1].float().log_softmax(-1)[answer_id].item()
    emitted = []
    for j in range(cfg['max_new_tokens']):
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        emitted.append(token.item())
        if token.item() == tok.eos_token_id:
            break
        out = forward(model, token, cache, ids.shape[-1] + j)
        compress(cache, method, budget, policy, cfg, gen)
        resident_peak = max(resident_peak, sum(x.nbytes() for x in cache.layers) + (policy.nbytes() if policy is not None else 0))
    torch.cuda.synchronize()
    text = tok.decode(emitted, skip_special_tokens=True)
    words = re.findall('[a-z]+', text.lower())
    return {'id': task['id'], 'kind': task['kind'], 'method': method, 'bits': bits, 'budget_bytes': budget, 'correct': bool(words and words[0] == task['answer']), 'answer': task['answer'], 'output': text, 'input_tokens': ids.shape[-1], 'output_tokens': len(emitted), 'answer_first_token_nll': answer_nll, 'resident_peak_bytes': resident_peak, 'cuda_peak_allocated': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved': torch.cuda.max_memory_reserved(), 'prefill_seconds': prefill_s, 'total_seconds': time.perf_counter() - start}


@torch.inference_mode()
def collect_traces(model, tok, cfg, dest):
    traces = []
    for ti, task in enumerate(tasks(cfg['train_seed'], cfg['train_count'])):
        # Keep the target fact and (where present) correction in the context.
        # A disposable preamble makes positions 4:64 irrelevant by construction.
        lines = task['prompt'].split('\n\n')[1].splitlines()
        important = [s for s in lines if s.startswith('The access') or s.startswith('Correction')]
        short_context = ('These are unrelated administrative notes about filing records. ' * 12
                         + '\n' + important[0] + '\n' + '\n'.join(lines[1:5])
                         + ('\n' + important[-1] if len(important) > 1 else '')
                         + '\n' + '\n'.join(lines[-4:]))
        prefix = tok(short_context, return_tensors='pt').input_ids.to(model.device)
        n = prefix.shape[-1]
        assert n > 256
        question = task['prompt'].split('\n\n')[-1]
        future_text = f"\nQuestion: {question}\nAnswer: {task['answer']}. " * 8
        future = tok(future_text, add_special_tokens=False, return_tensors='pt').input_ids[:, :64].to(model.device)
        whole = torch.cat((prefix, future), -1)
        native = DynamicCache()
        out = forward(model, whole, native, 0, attentions=True)
        selected = torch.cat((torch.arange(4, device=model.device), torch.arange(64, n, device=model.device)))
        xs, ys = [], []
        for li, layer in enumerate(native.layers):
            h = layer.keys.shape[1]
            pos = selected.view(1, 1, -1).expand(1, h, -1)
            xs.append(features(layer.keys[:, :, selected], layer.values[:, :, selected], pos)[0].cpu())
            att = out.attentions[li][0, :, n:, :n].float()
            utility = att.reshape(h, -1, att.shape[-2], n).amax(1).sum(1)
            ys.append(utility[:, selected].cpu())
        del out, native
        exposed = make_cache(model, 8)
        cut = n - 64
        forward(model, prefix[:, :cut], exposed, 0)
        for layer in exposed.layers:
            ix = torch.cat((torch.arange(4, device=model.device), torch.arange(64, cut, device=model.device))).view(1, 1, -1).expand(1, layer.keys.shape[1], -1)
            layer.retain(ix)
        forward(model, prefix[:, cut:], exposed, cut)
        assert torch.equal(exposed.layers[0].positions[0, 0], selected)
        xe = [features(unpack(layer.keys, layer.ks), unpack(layer.values, layer.vs), layer.positions)[0].cpu() for layer in exposed.layers]
        traces.append({'native': torch.cat(xs).half(), 'exposed': torch.cat(xe).half(), 'utility': torch.cat(ys), 'id': task['id']})
        print(json.dumps({'trace': ti + 1, 'total': cfg['train_count']}), flush=True)
    torch.save(traces, dest)


def train_policy(traces, cfg, arm, seed, device, dest):
    torch.manual_seed(seed)
    x = traces[0][arm]
    policy = Policies(x.shape[0], x.shape[-1], cfg['hidden']).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg['learning_rate'], weight_decay=0)
    rng = random.Random(seed)
    rows = []
    for step in range(cfg['updates']):
        tr = traces[rng.randrange(len(traces))]
        scores = policy(tr[arm].float().to(device)).clamp(-12, 12)
        utility = tr['utility'].to(device)
        utility = utility / utility.sum(-1, keepdim=True).clamp_min(1e-12)
        draws = cfg['permutations']
        expanded = scores.unsqueeze(0).expand(draws, -1, -1)
        g = -torch.log(-torch.log(torch.rand_like(expanded).clamp(1e-6, 1 - 1e-6)))
        order = (expanded + g).argsort(-1, descending=True)
        ordered_scores = expanded.gather(-1, order)
        denom = ordered_scores.flip(-1).logcumsumexp(-1).flip(-1)
        log_prob = (ordered_scores - denom).sum(-1)
        ranks = torch.arange(1, scores.shape[-1] + 1, device=device)
        u = utility.unsqueeze(0).expand(draws, -1, -1).gather(-1, order)
        optimum = (utility.sort(-1, descending=True).values * ranks).sum(-1).clamp_min(1e-8)
        reward = -(u * ranks).sum(-1) / optimum.unsqueeze(0)
        advantage = reward - (reward.sum(0, keepdim=True) - reward) / (draws - 1)
        loss = -(advantage.detach() * log_prob).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(policy.parameters(), 1)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite controller gradient')
        opt.step()
        rows.append({'step': step, 'loss': loss.item(), 'reward': reward.mean().item(), 'gradient_norm': float(norm)})
    torch.save(policy.cpu().state_dict(), dest)
    dump(dest.with_suffix('.json'), {'arm': arm, 'seed': seed, 'updates': rows, 'parameter_bytes': policy.nbytes()})
    return policy.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['smoke', 'run'])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((HERE / 'config.json').read_text())
    cfg_hash = sha(HERE / 'config.json')
    model_dir = Path(cfg['model_path'])
    manifest = {'config': cfg, 'config_sha256': cfg_hash, 'source_sha256': sha(__file__), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'stage': args.stage, 'model_files': {p.name: sha(p) for p in model_dir.iterdir() if p.suffix in ('.json', '.safetensors')}, 'started_unix': time.time()}
    dump(dest / 'manifest.json', manifest)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    if args.stage == 'smoke':
        from test_kvpilot import model_checks
        model_checks(model, tok)
        rows = []
        for t in tasks(cfg['smoke_seed'], 8):
            row = evaluate(model, tok, t, 'full', 16, 0, None, cfg)
            rows.append(row)
            print(json.dumps(row), flush=True)
        dump(dest / 'results.json', rows)
        gate = {'passed': sum(x['correct'] for x in rows) >= 6, 'correct': sum(x['correct'] for x in rows), 'n': 8}
        dump(dest / 'gate.json', gate)
        print(json.dumps(gate), flush=True)
        return
    prior = sorted((HERE / 'results').glob('smoke*/gate.json'))
    if not prior or not json.loads(prior[-1].read_text())['passed']:
        raise RuntimeError('Passing smoke gate is required')
    tests = tasks(cfg['test_seed'], cfg['test_count'])
    dump(dest / 'tasks.json', tests)
    collect_traces(model, tok, cfg, dest / 'traces.pt')
    traces = torch.load(dest / 'traces.pt', weights_only=True)
    policies = {}
    for seed in cfg['policy_seeds']:
        for arm in ['native', 'exposed']:
            p = train_policy(traces, cfg, arm, seed, 'cuda', dest / f'policy-{arm}-{seed}.pt')
            policies[(arm, seed)] = p.cpu()
            print(json.dumps({'trained': arm, 'seed': seed}), flush=True)
    del traces
    conditions = [('full', 16, 0, None)]
    for mib in cfg['budgets_mib']:
        for bits in cfg['bits']:
            conditions.extend((m, bits, mib, None) for m in ['recency', 'random'])
            conditions.extend((arm, bits, mib, seed) for seed in cfg['policy_seeds'] for arm in ['native', 'exposed'])
    # Counterbalance task order per condition; all outcomes append immediately.
    with (dest / 'results.jsonl').open('w', encoding='utf-8') as f:
        for ci, (method, bits, mib, seed) in enumerate(conditions):
            policy = policies[(method, seed)].to('cuda') if seed is not None else None
            ordered = list(tests)
            if ci % 2:
                ordered.reverse()
            for task in ordered:
                row = evaluate(model, tok, task, method, bits, mib * 1024 * 1024, policy, cfg)
                row['policy_seed'] = seed
                f.write(json.dumps(row) + '\n'); f.flush()
                print(json.dumps({'condition': ci + 1, 'conditions': len(conditions), 'task': task['id'], 'correct': row['correct'], 'method': method, 'bits': bits, 'mib': mib}), flush=True)
            if policy is not None:
                policy.cpu()
    manifest['completed_unix'] = time.time()
    manifest['results_sha256'] = sha(dest / 'results.jsonl')
    dump(dest / 'complete.json', manifest)


if __name__ == '__main__':
    main()
