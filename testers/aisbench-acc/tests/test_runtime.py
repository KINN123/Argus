import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import asyncio
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_urls_keep_prefix_ipv6_and_https(self):
        for url, expected in (("https://example.org/proxy", "https://example.org/proxy/v1"),
                              ("http://[::1]:8000/proxy/v1/", "http://[::1]:8000/proxy/v1")):
            self.assertEqual(runtime.validate_url(url), expected)
        for url in ("http://u:p@x", "http://x?key=s", "file:///x", "http://x:invalid", "http://x/v1/chat/completions"):
            with self.subTest(url=url), self.assertRaises(runtime.Failure):
                runtime.validate_url(url)

    def test_config_is_data_and_contains_no_secret(self):
        injected = "m'\n__import__('os').system('bad')#"
        args = types.SimpleNamespace(model_name=injected, model_path=str(self.path), url="http://x/proxy/v1",
                                     seed=1, output_len=8, concurrency=2, request_rate=0,
                                     temperature=0, top_p=1, thinking="default", test_type="stream")
        target = self.path / "config.py"
        with patch.dict(os.environ, {"ENDPOINT_API_KEY": "unique-model-key"}):
            runtime.write_config(target, args, self.path, "{question}", False)
        tree = ast.parse(target.read_text())
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body}
        self.assertEqual(values["models"][0]["model"], injected)
        self.assertEqual(values["datasets"][0]["infer_cfg"]["inferencer"]["type"], "testplane_inferencer.GenInferencer")
        self.assertFalse(values["models"][0]["generation_kwargs"]["ignore_eos"])
        self.assertNotIn("unique-model-key", target.read_text())
        self.assertNotIn("api_key", values["models"][0])

    def test_output_directory_cannot_reuse_results(self):
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(self.path)}):
            first = runtime.Execution("test")
            first.prepare()
            with self.assertRaises(runtime.Failure):
                runtime.Execution("test").prepare()

    def test_cancellation_stops_child_and_records_existing_code(self):
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(self.path)}):
            execution = runtime.Execution("test")
            execution.prepare()
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=os.name == "posix")
            execution.child = child
            try:
                execution.cancel(signal.SIGTERM, None)
            except runtime.Failure as exc:
                code = execution.finish(exc)
            self.assertNotEqual(code, 0)
            self.assertIsNotNone(child.poll())
            result = json.loads((self.path / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["error"]["code"], "CANCELED")

    def test_auth_adapter_does_not_put_key_in_request_record(self):
        class Base:
            def __init__(self, **kwargs):
                self.url = self._get_url()
                self.received_key = kwargs["api_key"]
            async def get_request_body(self, input, max_out_len, output, **args):
                return dict(messages=[dict(role="user", content=input)], max_tokens=max_out_len)
            async def generate(self, input, max_out_len, output, **kwargs):
                await self.get_request_body(input, max_out_len, output)
                return output
        modules = types.ModuleType("ais_bench.benchmark.models")
        modules.VLLMCustomAPIChat = Base
        with patch.dict(sys.modules, {"ais_bench.benchmark.models": modules}), \
             patch.dict(os.environ, {"ENDPOINT_API_KEY": "do-not-save", "TESTPLANE_AUDIT_DIR": str(self.path)}):
            spec = importlib.util.spec_from_file_location("isolated_model", ROOT / "testplane_model.py")
            model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model)
            chat = model.Chat(url="https://example.org/prefix/v1", path=str(self.path))
            self.assertEqual(chat.url, "https://example.org/prefix/v1/chat/completions")
            self.assertEqual(chat.received_key, "do-not-save")
            output = types.SimpleNamespace(data_id=0, success=True, error_info="", input_tokens=4,
                                           output_tokens=1, time_points=[1, 2], get_prediction=lambda: "2")
            asyncio.run(chat.generate("question", 8, output))
        raw = (self.path / "000000000.json").read_text()
        self.assertNotIn("do-not-save", raw)
        self.assertNotIn("api_key", raw)
        self.assertFalse(hasattr(output, "_tp_request"))

    def test_subprocess_preserves_exit_code_and_does_not_use_shell(self):
        args = types.SimpleNamespace(model_name="m", model_path=str(self.path), url="http://x/v1",
                                     seed=1, output_len=8, concurrency=2, request_rate=0,
                                     temperature=0, top_p=1, thinking="default", test_type="stream")
        class Child:
            stdout = io.StringIO("redact-this-key\n")
            def wait(self): return 7
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(self.path), "ENDPOINT_API_KEY": "redact-this-key", "METRICS_API_KEY": "metrics-only"}), \
             patch.object(runtime.subprocess, "Popen", return_value=Child()) as popen:
            execution = runtime.Execution("test")
            execution.prepare()
            code = execution.run_aisbench(self.path / "raw/run", args, self.path, "{question}", False)
        self.assertEqual(code, 7)
        command = popen.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertNotIn("--reuse", command)
        self.assertEqual(command[command.index("--num-warmups") + 1], "0")
        self.assertFalse(popen.call_args.kwargs.get("shell", False))
        self.assertNotIn("METRICS_API_KEY", popen.call_args.kwargs["env"])
        self.assertNotIn("redact-this-key", (self.path / "raw/run/aisbench.log").read_text())

    def test_failed_result_displays_child_traceback_even_when_cli_exited_zero(self):
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(self.path), "ENDPOINT_API_KEY": "secret-value"}):
            execution = runtime.Execution("test")
            execution.prepare()
            directory = self.path / "raw/round-1"
            directory.mkdir()
            (directory / "aisbench.log").write_text("Traceback: AttributeError from child; secret-value\n", encoding="utf-8")
            execution.last_aisbench_directory = directory
            with patch("sys.stdout", new=io.StringIO()) as output:
                code = execution.finish(runtime.Failure("incomplete requests", "incomplete_requests", 1))
            self.assertEqual(code, 1)
            self.assertIn("AttributeError from child", output.getvalue())
            self.assertIn(str(directory / "aisbench.log"), output.getvalue())
            self.assertNotIn("secret-value", output.getvalue())

    def test_independent_images_use_identical_runtime(self):
        sibling = ROOT.parent / ("aisbench-perf" if ROOT.name == "aisbench-acc" else "aisbench-acc")
        for file in ("runtime.py", "testplane_model.py", "testplane_inferencer.py", "build/requirements.txt", "build/aisbench-lock.json"):
            self.assertEqual((ROOT / file).read_bytes(), (sibling / file).read_bytes(), file)

    def test_manifest_matches_cli(self):
        import yaml
        import runner
        manifest = yaml.safe_load((ROOT / "tester.yaml").read_text(encoding="utf-8"))
        actions = {a.dest: a for a in runner.parser()._actions if a.dest != "help"}
        self.assertEqual(set(manifest["parameters"]), set(actions))
        for name, definition in manifest["parameters"].items():
            if "default" in definition:
                value = actions[name].type(definition["default"]) if actions[name].type else definition["default"]
                self.assertEqual(value, actions[name].default, name)
        self.assertEqual(manifest["version"], runtime.VERSION)


if __name__ == "__main__":
    unittest.main()
