import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import check_c4_numerics as check
from test_long_prefill_sparse import small_inputs


class NumericsTests(unittest.TestCase):
    def test_reference_and_sparse_indices(self):
        args = small_inputs()
        reference = check.gathered_fp64(*args)
        candidate = check.shared_chunked(*args)
        self.assertLess(check.differences(candidate, reference)["max_abs"], 1e-6)
        self.assertTrue(torch.equal(reference[0], torch.zeros_like(reference[0])))
        q, kv, mask, sink, scale = args
        indices, lengths = check.mask_indices(mask[1:])
        self.assertEqual(indices.shape[1], 64)
        self.assertEqual(lengths.tolist(), [1, 3, 9, 3])
        self.assertTrue(torch.equal(indices[1, :3], torch.tensor([0, 2, 4], dtype=torch.int32)))
        with self.assertRaises(RuntimeError):
            check.mask_indices(mask)
        one = check.gathered_fp64(torch.zeros(1, 1, 2), torch.tensor([[2., -4.]]),
                                 torch.ones(1, 1, dtype=torch.bool), torch.zeros(1), 1.)
        self.assertTrue(torch.equal(one, torch.tensor([[[1., -2.]]], dtype=torch.float64)))
        # A missing sink term would return the unscaled value instead.
        self.assertFalse(torch.equal(one, torch.tensor([[[2., -4.]]], dtype=torch.float64)))
        for side in (0, 1):
            pair = [candidate.clone(), candidate.clone()]
            pair[side][0, 0, 0] = float("nan")
            with self.assertRaises(RuntimeError):
                check.differences(*pair)

    def test_cpu_entry_rejects_bad_evidence_without_complete(self):
        mutations = ("none", "hash", "nonfinite", "native-drift")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "results"
                q, kv, mask, sink, scale = small_inputs()
                q, kv = q.bfloat16().float(), kv.bfloat16().float()
                sample = dict(q=q, kv=kv, mask=mask, sink=sink, scale=scale,
                              output=check.shared_chunked(q, kv, mask, sink, scale),
                              identity={"fixture": True}, layer=2)
                if mutation == "nonfinite":
                    sample["output"][0, 0, 0] = float("nan")
                native = types.SimpleNamespace(sparse_attend_masked=lambda *a:
                                               sample["output"] + (1 if mutation == "native-drift" else 0))
                spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m: None))
                with patch.object(check, "load_capture", return_value=sample), \
                        patch.object(check, "LAYERS", (2,)), \
                        patch.object(check.bench, "sha", return_value=check.bench.NATIVE_HASH), \
                        patch.object(torch, "get_num_threads", return_value=32), \
                        patch.object(check.os, "sched_getaffinity", return_value=set(range(32))), \
                        patch.object(check.importlib.util, "spec_from_file_location", return_value=spec), \
                        patch.object(check.importlib.util, "module_from_spec", return_value=native), \
                        patch.dict(os.environ, SLURM_JOB_ID="test"):
                    if mutation == "hash":
                        with patch.object(check, "load_capture", side_effect=RuntimeError("hash mismatch")):
                            with self.assertRaises(RuntimeError):
                                check.run("cpu", out)
                    elif mutation != "none":
                        with self.assertRaises(RuntimeError):
                            check.run("cpu", out)
                    else:
                        check.run("cpu", out)
                        report = json.loads((out / "complete.json").read_text())
                        self.assertEqual(report["cases"][0]["layer"], 2)
                        self.assertIn("vs_fp64", report["cases"][0])
                    self.assertEqual((out / "complete.json").exists(), mutation == "none")

    def test_gpu_grouping_uses_actual_operands(self):
        q, kv, mask, sink, scale = small_inputs()
        q = q.repeat(1, 22, 1)[:, :64]
        sink = sink.repeat(22)[:64]
        sample = dict(q=q[1:].bfloat16().float(), kv=kv.bfloat16().float(),
                      mask=mask[1:], sink=sink, scale=scale)
        # Exercise the adapter independently of CUDA availability, including sentinel handling.
        def gpu_fake(q, kv, indices, sm_scale, d_v, attn_sink, topk_length):
            selected = torch.zeros(q.shape[0], kv.shape[0], dtype=torch.bool)
            for i, count in enumerate(topk_length.tolist()):
                selected[i, indices[i, 0, :count].long()] = True
            result = check.shared_chunked(q.float(), kv[:, 0].float(), selected, attn_sink, sm_scale)
            return result.bfloat16(), None, None

        def tile_fake(q, kv, sink, indices, scale):
            lens = (indices[0] >= 0).sum(-1)
            return gpu_fake(q[0], kv[0].unsqueeze(1), indices[0].unsqueeze(1),
                            scale, q.shape[-1], sink, lens)[0].unsqueeze(0)

        modules = {
            "sgl_kernel": types.ModuleType("sgl_kernel"),
            "sgl_kernel.flash_mla": types.SimpleNamespace(flash_mla_sparse_fwd=gpu_fake),
            "inference": types.ModuleType("inference"),
            "inference.kernel": types.SimpleNamespace(sparse_attn=tile_fake),
        }
        with patch.dict(check.sys.modules, modules), patch.object(torch.Tensor, "cuda", lambda t: t):
            result = check.gpu_outputs(sample)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(torch.equal(t, next(iter(result.values()))) for t in result.values()))


if __name__ == "__main__":
    unittest.main()
