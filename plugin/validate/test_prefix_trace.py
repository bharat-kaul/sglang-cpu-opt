import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dsv4_prefix_trace import Histories, force_tokens, install, select_row


class PrefixTraceTests(unittest.TestCase):
    def test_chunk_decode_reorder_and_slot_reuse(self):
        history = Histories()
        self.assertEqual(history.update([1, 2, 8], [0, 1, 0], [7, 9], [0, 0], [2, 1]), [1, 2])
        history.update([3, 4], [2, 3], [7], [2], [2])
        history.update([6, 5], [1, 4], [9, 7])
        self.assertEqual(history.sequences, {7: [1, 2, 3, 4, 5], 9: [8, 6]})
        history.update([10], [0], [7], [0], [1])
        self.assertEqual(history.sequences[7], [10])
        with self.assertRaisesRegex(RuntimeError, "noncontiguous"):
            history.update([11], [3], [7])
        with self.assertRaisesRegex(RuntimeError, "positions"):
            history.update([11, 12], [1, 3], [7], [1], [2])

    def test_full_mhc_streams_are_copied_before_mutation(self):
        value = torch.arange(24.0).reshape(2, 4, 3)
        saved = select_row((value, None, False), 1, 2)
        value.zero_()
        self.assertEqual(saved[0].shape, (4, 3))
        self.assertEqual(saved[0].sum().item(), sum(range(12, 24)))
        with self.assertRaisesRegex(RuntimeError, "unmapped"):
            select_row(torch.zeros(3, 4), 0, 2)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            select_row(torch.tensor([[float("nan")]]), 0, 1)

    def test_forcing_checks_exact_prefix_and_only_selected_rows(self):
        logits = torch.ones(2, 5)
        item = {"prompt_ids": [1, 2], "output_token_ids": [3, 4]}
        force_tokens(logits, [(1, item, 1)], [[8], [1, 2, 3]])
        self.assertTrue(torch.equal(logits[0], torch.ones(5)))
        self.assertEqual(logits[1].argmax().item(), 4)
        with self.assertRaisesRegex(RuntimeError, "recorded CPU prefix"):
            force_tokens(logits, [(1, item, 1)], [[8], [1, 9, 3]])

    def test_native_forward_is_unchanged_and_complete(self):
        class Gate(torch.nn.Module):
            def forward(self, x):
                return x, None

        class TopK(torch.nn.Module):
            def forward(self, x, logits):
                return SimpleNamespace(
                    topk_ids=torch.zeros(x.shape[0], 1, dtype=torch.int64),
                    topk_weights=torch.ones(x.shape[0], 1),
                )

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate, self.topk = Gate(), TopK()

            def forward(self, hidden_states):
                logits, _ = self.gate(hidden_states)
                self.topk(hidden_states, logits)
                return hidden_states * 0.1

        class Layer(torch.nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.layer_id = idx
                self.self_attn = torch.nn.Identity()
                self.mlp = MLP()
                self.input_layernorm = torch.nn.LayerNorm(2)
                self.post_attention_layernorm = torch.nn.LayerNorm(2)
                self.use_fused_mhc_post_pre = False
                for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                             "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
                    setattr(self, name, torch.ones(1))

            def hc_pre(self, x):
                n = x.shape[0]
                return x.mean(1), torch.ones(n, 4), torch.ones(n, 4, 4), False

            def hc_post(self, x, residual, post, comb):
                return residual + x.unsqueeze(1) * 0.01

            def forward(self, positions, hidden_states, input_ids, batch, input_ids_global):
                residual = hidden_states
                h, post, comb, _ = self.hc_pre(hidden_states)
                h = self.self_attn(h)
                residual = self.hc_post(h, residual, post, comb)
                h, post, comb, _ = self.hc_pre(residual)
                h = self.mlp(h)
                return self.hc_post(h, residual, post, comb), None, None, None

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.Identity()
                self.hc_head_fn = self.hc_head_scale = self.hc_head_base = torch.ones(1)

            def hc_head(self, x):
                return x.mean(1)

        class Logits(torch.nn.Module):
            def forward(self, ids, hidden_states):
                return SimpleNamespace(next_token_logits=hidden_states[-1:].clone())

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(i) for i in range(43)])
                self.model = Core()
                self.logits_processor = Logits()

            def forward(self, ids, positions, batch):
                hidden = ids.float()[:, None, None].expand(-1, 4, 2).clone()
                for layer in self.layers:
                    hidden, *_ = layer(positions, hidden, ids, batch, ids)
                hidden = self.model.norm(self.model.hc_head(hidden))
                return self.logits_processor(ids, hidden)

        mode = SimpleNamespace(is_idle=lambda: False, is_extend=lambda: True, name="EXTEND")
        batch = SimpleNamespace(
            forward_mode=mode, req_pool_indices=torch.tensor([4]),
            extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[2],
        )
        ids, positions = torch.tensor([1, 2]), torch.tensor([0, 1])
        model = Model()
        expected = model(ids, positions, batch).next_token_logits
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "plan.json").write_text(json.dumps({
                "batches": [[{"q": 7, "prompt_ids": [1, 2], "output_token_ids": [3]}]],
                "targets": [7], "last_decision": 0,
            }))
            (root / "phase").write_text("warmup")
            env = {"DSV4_PREFIX_TRACE_PLAN": str(root / "plan.json"),
                   "DSV4_PREFIX_TRACE_OUT": str(root / "out"),
                   "DSV4_PREFIX_TRACE_PHASE": str(root / "phase")}
            distributed = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0)
            with patch.dict(os.environ, env), patch.dict(
                sys.modules, {"sglang.srt.distributed": distributed}
            ):
                install(Model, Layer)
                dummy = SimpleNamespace(req_pool_indices=torch.tensor([0, 0]))
                warmed = model(ids, positions, dummy)
                self.assertTrue(torch.equal(expected, warmed.next_token_logits))
                self.assertFalse((root / "out").exists())
                (root / "phase").write_text("natural")
                output = model(ids, positions, batch)
            self.assertTrue(torch.equal(expected, output.next_token_logits))
            payload = torch.load(root / "out/rank0/natural.q7.d0.pt", weights_only=True)
            self.assertEqual(payload["boundaries"]["L42.out"][0].shape, (4, 2))
            self.assertEqual(payload["boundaries"]["L0.mhc.post.0.in"][1].shape, (4, 2))
            self.assertEqual(payload["sequence_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()
