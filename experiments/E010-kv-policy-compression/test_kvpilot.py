"""Invariants that would invalidate the experiment if broken."""
import unittest
import torch
from transformers.cache_utils import Cache, DynamicCache
from kvpilot import PackedLayer, Policies, pack, unpack, compress, forward, make_cache, tasks


class StorageChecks(unittest.TestCase):
    def test_quantization_bound_and_real_storage(self):
        torch.manual_seed(7)
        x = torch.randn(1, 2, 50, 128, dtype=torch.bfloat16)
        q, s = pack(x, 8)
        self.assertEqual(q.dtype, torch.int8)
        # Half scale rounding plus BF16 reconstruction rounding.
        err = (unpack(q, s).float() - x.float()).abs()
        self.assertTrue(torch.all(err <= s.float() * .6 + x.float().abs() * .005))
        self.assertLess(q.nbytes + s.nbytes, x.nbytes)

    def test_positions_bytes_and_budget(self):
        cfg = {'prefix_keep': 4, 'recent_keep': 16}
        layers = []
        for _ in range(2):
            layer = PackedLayer(8)
            x = torch.randn(1, 2, 100, 8, dtype=torch.bfloat16)
            layer.update(x, x, {'cache_position': torch.arange(100)})
            layers.append(layer)
        cache = Cache(layers=layers)
        per_token = sum(l.nbytes() for l in layers) // 100
        compress(cache, 'recency', per_token * 30, None, cfg, torch.Generator())
        self.assertEqual(cache.get_seq_length(), 30)
        self.assertEqual(sum(l.nbytes() for l in layers), per_token * 30)
        for l in layers:
            self.assertEqual(l.positions[0, 0].tolist(), list(range(4)) + list(range(74, 100)))
            x = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
            l.update(x, x, {'cache_position': torch.tensor([100])})
            self.assertEqual(l.positions[0, 0, -1].item(), 100)
            self.assertEqual(l.keys.untyped_storage().nbytes(), l.keys.nbytes)

    def test_disjoint_reproducible_tasks(self):
        self.assertEqual(tasks(11, 4), tasks(11, 4))
        self.assertNotEqual(tasks(11, 4), tasks(12, 4))
        self.assertEqual([t['kind'] for t in tasks(11, 4)], ['lookup', 'correction'] * 2)

    def test_policy_overhead_is_counted(self):
        p = Policies(4, 18, 4)
        self.assertEqual(p.nbytes(), sum(v.nbytes for v in p.parameters()))
        layers = []
        for _ in range(2):
            l = PackedLayer(16)
            x = torch.randn(1, 2, 80, 8, dtype=torch.bfloat16)
            l.update(x, x, {'cache_position': torch.arange(80)})
            layers.append(l)
        cache = Cache(layers=layers)
        per_token = sum(l.nbytes() for l in layers) // 80
        with torch.no_grad():
            compress(cache, 'native', per_token * 30 + p.nbytes(), p, {'prefix_keep': 4, 'recent_keep': 16}, torch.Generator())
        self.assertEqual(cache.get_seq_length(), 30)


@torch.inference_mode()
def model_checks(model, tok):
    ids = tok('The key is blue. ' * 24, return_tensors='pt').input_ids.to(model.device)
    normal = model(ids, use_cache=False).logits[:, -1].float()
    dense = forward(model, ids, DynamicCache(), 0).logits[:, -1].float()
    torch.testing.assert_close(normal, dense, atol=.02, rtol=.005)
    cache = make_cache(model, 16)
    stock = DynamicCache()
    for at in range(0, ids.shape[-1], 32):
        chunked = forward(model, ids[:, at:at+32], cache, at).logits[:, -1].float()
        stock_chunked = forward(model, ids[:, at:at+32], stock, at).logits[:, -1].float()
        torch.testing.assert_close(stock_chunked, chunked, atol=0, rtol=0)
    self_ids = ids[:, :32]
    prefix = forward(model, self_ids, DynamicCache(), 0).logits.float()
    whole = forward(model, ids, DynamicCache(), 0).logits[:, :32].float()
    # BF16 GEMM shapes can change rounding. The causal check compares equal
    # shapes with two different futures, eliminating that confound.
    changed = ids.clone()
    changed[:, 32:] = tok.encode('green', add_special_tokens=False)[0]
    alternate = forward(model, changed, DynamicCache(), 0).logits[:, :32].float()
    torch.testing.assert_close(whole, alternate, atol=0, rtol=0)
    print('MODEL_CHECKS_PASS: dense parity, chunk parity, causal future isolation', flush=True)


if __name__ == '__main__':
    unittest.main()
