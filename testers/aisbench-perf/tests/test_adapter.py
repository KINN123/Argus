"""Regression for AISBench outputs without data_id, including concurrent calls."""
import asyncio
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LegacyModel:
    def __init__(self, **kwargs):
        self.url = self._get_url()
        self.path = kwargs["path"]

    async def get_request_body(self, input, max_out_len, output, **kwargs):
        return dict(messages=[dict(role="user", content=input)], max_tokens=max_out_len)

    async def generate(self, input, max_out_len, output, **kwargs):
        await self.get_request_body(input, max_out_len, output)
        await asyncio.sleep(kwargs.get("delay", 0))
        if kwargs.get("raise_error"):
            raise RuntimeError("upstream request failed")
        if kwargs.get("http_error"):
            output.success = False
            output.error_info = "HTTP 400: model unavailable"
        return output


def legacy_output(prediction):
    # Older GenInferencer keeps the index separately instead of setting data_id.
    return types.SimpleNamespace(success=True, error_info="", input_tokens=4,
                                 output_tokens=1, time_points=[1, 2], get_prediction=lambda: prediction)


class LegacyInferencer:
    def __init__(self, model):
        self.model = model
        self.finished = []
        self.outputs = {}

    async def do_request(self, data, token_bucket, session):
        data = dict(data)
        index = data.pop("index")
        prompt = data.pop("prompt")
        output = legacy_output(prompt)
        self.outputs[index] = output
        await self.model.generate(prompt, 8, output, session=session, **data)
        self.finished.append(index)
        return output


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        modules = types.ModuleType("ais_bench.benchmark.models")
        modules.VLLMCustomAPIChat = LegacyModel
        inferencers = types.ModuleType("ais_bench.benchmark.openicl.icl_inferencer")
        inferencers.GenInferencer = LegacyInferencer
        for patcher in (
            patch.dict(sys.modules, {modules.__name__: modules, inferencers.__name__: inferencers}),
            patch.dict(os.environ, {"TESTPLANE_AUDIT_DIR": str(self.path)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location("testplane_model", ROOT / "testplane_model.py")
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        patcher = patch.dict(sys.modules, {"testplane_model": self.adapter})
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location("isolated_inferencer", ROOT / "testplane_inferencer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.chat = self.adapter.Chat(url="http://localhost:7800/v1", path=str(self.path))
        self.inferencer = module.GenInferencer(self.chat)

    def record(self, index):
        return json.loads((self.path / f"{index:09d}.json").read_text(encoding="utf-8"))

    def test_single_sample_without_data_id(self):
        output = asyncio.run(self.inferencer.do_request(dict(index=0, prompt="answer 2"), None, None))
        self.assertFalse(hasattr(output, "data_id"))
        self.assertEqual(self.record(0)["prediction"], "answer 2")
        self.assertTrue(self.record(0)["success"])
        self.assertFalse(hasattr(output, "_tp_request"))
        self.assertIsNone(self.adapter.REQUEST_INDEX.get())

    def test_concurrent_completion_keeps_original_sample_ids(self):
        async def run():
            await asyncio.gather(
                self.inferencer.do_request(dict(index=0, prompt="first", delay=.03), None, None),
                self.inferencer.do_request(dict(index=1, prompt="second"), None, None),
            )
        asyncio.run(run())
        self.assertEqual(self.inferencer.finished, [1, 0])
        self.assertEqual(self.record(0)["prediction"], "first")
        self.assertEqual(self.record(1)["prediction"], "second")
        self.assertEqual(self.record(0)["id"], 0)
        self.assertEqual(self.record(1)["id"], 1)

    def test_existing_output_data_id_still_works(self):
        output = legacy_output("newer output")
        output.data_id = 7
        asyncio.run(self.chat.generate("prompt", 8, output))
        self.assertEqual(self.record(7)["prediction"], "newer output")

    def test_failed_http_response_is_recorded_as_failed(self):
        asyncio.run(self.inferencer.do_request(dict(index=0, prompt="question", http_error=True), None, None))
        self.assertFalse(self.record(0)["success"])
        self.assertIn("HTTP 400", self.record(0)["error"])

    def test_request_exception_survives_and_context_is_restored(self):
        async def run():
            marker = self.adapter.REQUEST_INDEX.set(99)
            try:
                with self.assertRaisesRegex(RuntimeError, "upstream request failed"):
                    await self.inferencer.do_request(dict(index=0, prompt="question", raise_error=True), None, None)
                self.assertEqual(self.adapter.REQUEST_INDEX.get(), 99)
            finally:
                self.adapter.REQUEST_INDEX.reset(marker)
        asyncio.run(run())
        self.assertFalse(self.record(0)["success"])
        self.assertIn("RuntimeError: upstream request failed", self.record(0)["error"])

    def test_audit_error_does_not_replace_request_exception(self):
        with patch.object(self.chat, "_write_record", side_effect=OSError("disk full")), redirect_stderr(io.StringIO()) as log:
            with self.assertRaisesRegex(RuntimeError, "upstream request failed"):
                asyncio.run(self.inferencer.do_request(dict(index=0, prompt="question", raise_error=True), None, None))
        self.assertIn("disk full", log.getvalue())
        self.assertIsNone(self.adapter.REQUEST_INDEX.get())


if __name__ == "__main__":
    unittest.main()
