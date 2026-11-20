import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def parser(self):
        p = runtime.Parser()
        runtime.public_args(p)
        p.add_argument("--output_len", type=runtime.integer(), default=32)
        return p

    def test_jsonl_cases_env_identity_and_cli_override(self):
        path = self.path / "params.jsonl"
        path.write_text('{"output_len":8,"concurrency":2}\n{"output_len":16,"concurrency":3}\n', encoding="utf-8")
        output = self.path / "out"
        calls = []
        def work(execution, args):
            calls.append((args.output_len, args.concurrency, execution.output))
            execution.result["metrics"].append(runtime.metric("requests", 1, "count", "count", {}))
            runtime.dump(execution.output / "raw/check.json", {"output_len": args.output_len})
        env = dict(AG_PARAMS_FILE=str(path), AG_OUTPUT_DIR=str(output), TESTPLANE_OUTPUT_DIR=str(self.path / "ignored"),
                   AG_SERVED_URL="http://x", AG_SERVED_MODEL_NAME="m", AG_MODEL_PATH=str(self.path),
                   AG_TASK_ID="t-test", AG_ATTEMPT="2", AG_RUNNER_SEQ="3")
        with patch.dict(os.environ, env):
            code = runtime.run_main("test", self.parser, work, ["--concurrency=4"])
        self.assertEqual(code, 0)
        self.assertEqual([c[:2] for c in calls], [(8, 4), (16, 4)])
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((result["task_id"], result["attempt_no"], result["runner_seq"]), ("t-test", 2, 3))
        self.assertEqual([m["labels"]["case"] for m in result["metrics"]], [1, 2])
        self.assertTrue((output / "cases/1/raw/check.json").is_file())
        self.assertEqual(len(result["metadata"]["cases"]), 2)
        self.assertNotIn("cases", result["metadata"]["cases"][1]["metadata"])

    def test_file_selection_arrays_and_public_parameter_rejection(self):
        path = self.path / "input.jsonl"
        path.write_text(json.dumps(dict(rounds=[dict(input_len=32)], metrics_endpoint=["http://a/metrics", "http://b/metrics"], prefix_test=True)), encoding="utf-8")
        with patch.dict(os.environ, {"AG_PARAMS_FILE": str(self.path / "missing")}):
            values = runtime.input_cases(["--params-file", str(path)])[0]
        self.assertIn('--rounds=[{"input_len": 32}]', values)
        self.assertIn("--prefix_test=true", values)
        self.assertIn("--metrics_endpoint=http://a/metrics", values)
        path.write_text('{"url":"http://x"}', encoding="utf-8")
        with self.assertRaises(runtime.Failure):
            runtime.input_cases(["--params-file=" + str(path)])

    def test_missing_or_invalid_parameter_file_fails(self):
        with self.assertRaises(runtime.Failure):
            runtime.input_cases(["--params-file=" + str(self.path / "missing")])
        path = self.path / "bad.jsonl"
        path.write_text('[1,2]', encoding="utf-8")
        with self.assertRaises(runtime.Failure):
            runtime.input_cases(["--params-file=" + str(path)])


if __name__ == "__main__":
    unittest.main()
