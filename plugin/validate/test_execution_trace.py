import enum
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "execution_trace", Path(__file__).parents[1] / "dsv4_layer_capture/execution_trace.py"
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class Mode(enum.Enum):
    EXTEND = 1


class ExecutionTraceTests(unittest.TestCase):
    def test_production_harness_keeps_text_batch_and_stops(self):
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def get_server_info(self):
                return {"chunked_prefill_size": 512, "disable_radix_cache": True}

            def generate(self, prompts, params):
                calls.append((prompts, params))
                return [{"text": "The answer is 7.", "output_ids": [101],
                         "meta_info": {"completion_tokens": 1}} for _ in prompts]

            def shutdown(self):
                calls.append("shutdown")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = [{"question": f"question {i}?", "answer": "#### 7"} for i in range(14)]
            (root / "data.jsonl").write_text("".join(json.dumps(x) + "\n" for x in data))
            (root / "ids.json").write_text("[3, 4]")
            script = Path(__file__).with_name("task_gsm8k_chunked.py")
            argv = [str(script), "--model", "unused", "--data", str(root / "data.jsonl"),
                    "--question-ids-file", str(root / "ids.json"), "--chunk-size", "32",
                    "--num-shards", "2", "--shard-index", "1",
                    "--runtime-out", str(root / "runtime.json"), "--gens-out", str(root / "gens.json"),
                    "--gens-schema-v2", "--gens-token-ids", "--no-resume"]
            with patch.dict("os.environ", {}, clear=True), patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, {"sglang": SimpleNamespace(Engine=Engine)}):
                runpy.run_path(str(script), run_name="__main__")
            result = json.loads((root / "gens.json").read_text())
            runtime = json.loads((root / "runtime.json").read_text())
        self.assertEqual(calls[0]["chunked_prefill_size"], 512)
        self.assertEqual(calls[0]["mem_fraction_static"], 0.5)
        prompts, params = calls[1]
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[0].endswith("Question: question 11?\nAnswer:"))
        self.assertEqual(params, {"temperature": 0.0, "max_new_tokens": 256,
                                  "stop": ["Question", "Assistant:"]})
        self.assertEqual([row["q"] for row in result["gens"]], [3, 4])
        self.assertEqual(result["gens"][0]["output_token_ids"], [101])
        self.assertEqual(result["run_identity"]["num_shards"], 2)
        self.assertEqual(result["run_identity"]["shard_index"], 1)
        self.assertEqual(runtime["chunked_prefill_size"], 512)
        self.assertEqual(calls[-1], "shutdown")

    def test_returns_native_output_without_changing_inputs(self):
        expected = SimpleNamespace(next_token_logits=torch.tensor([[0.0, 3.0, 1.0]]))

        class Model:
            def forward(self, ids, positions, batch):
                return expected

        batch = SimpleNamespace(
            forward_mode=Mode.EXTEND,
            req_pool_indices=torch.tensor([7]),
            seq_lens=torch.tensor([4]),
            extend_prefix_lens_cpu=[2],
            extend_seq_lens_cpu=[2],
        )
        ids = torch.tensor([11, 12])
        positions = torch.tensor([2, 3])
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "trace")
            with patch.dict("os.environ", {"INTEL_CPU_DSV4_TRACE_TOKEN_IDS": "[0, 1]"}):
                trace.install_execution_trace(Model, prefix)
            trace.install_execution_trace(Model, prefix)
            self.assertIs(Model().forward(ids, positions, batch), expected)
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_ids"], [11, 12])
        self.assertEqual(rows[0]["positions"], [2, 3])
        self.assertEqual(rows[0]["extend_prefix_lens"], [2])
        self.assertEqual(rows[0]["raw_next_logit_top2_ids"], [[1, 2]])
        self.assertEqual(rows[0]["selected_logit_ids"], [0, 1])
        self.assertEqual(rows[0]["selected_logit_values"], [[0.0, 3.0]])
        self.assertEqual(ids.tolist(), [11, 12])
        self.assertEqual(expected.next_token_logits.tolist(), [[0.0, 3.0, 1.0]])

    def test_trace_open_failure_does_not_run_model(self):
        class Model:
            def forward(self, *args):
                raise AssertionError("must not run when the trace cannot be opened")

        with tempfile.TemporaryDirectory() as directory:
            trace.install_execution_trace(Model, str(Path(directory) / "missing" / "trace"))
            with self.assertRaises(FileNotFoundError):
                Model().forward(None, None, None)


if __name__ == "__main__":
    unittest.main()
