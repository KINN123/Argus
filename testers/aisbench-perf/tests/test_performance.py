import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runner
import runtime
from dataset_generator import generate, validate_prefixes
from result_mapper import map_performance, statistics
from test_prefix import snapshot


class CharTokenizer:
    all_special_ids = []
    chat_template = "user:{content}:assistant"
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, tokens, **kwargs): return "".join(chr(t) for t in tokens)
    def get_vocab(self): return {chr(i): i for i in range(65, 123)}
    def apply_chat_template(self, messages, **kwargs): return self.encode("user:" + messages[0]["content"] + ":assistant")


def fixture(directory, count=2, output_tokens=3, failed=False, empty=False):
    (directory / "requests").mkdir(parents=True)
    for i in range(count):
        runtime.dump(directory / "requests" / f"{i}.json", dict(id=i, success=not failed, prediction="" if empty else "abc",
                     input_tokens=10, output_tokens=output_tokens, time_points=[1, 1.1, 1.2, 1.3], token_source="usage"))
    common = {"Total Requests": count, "Success Requests": count, "Failed Requests": 0,
              "Benchmark Duration": "1000 ms", "Concurrency": 1.5, "Request Throughput": f"{count} req/s",
              "Total Input Tokens": count * 10, "Total Generated Tokens": count * output_tokens,
              "Input Token Throughput": f"{count * 10} token/s", "Output Token Throughput": f"{count * output_tokens} token/s",
              "Total Token Throughput": f"{count * (10 + output_tokens)} token/s"}
    runtime.dump(directory / f"aisbench/20261117/performances/{runtime.MODEL}/{runtime.DATA}.json",
                 {key: {"total": value} for key, value in common.items()})


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.base = ["--url=http://x", "--model_name=m", f"--model_path={self.path}", "--metrics_endpoint=http://x/metrics"]

    def test_actual_throughput_and_time_units(self):
        fixture(self.path)
        metrics, evidence, error = map_performance(self.path, 2, {}, stream=True)
        self.assertIsNone(error)
        def find(name, stat="mean"): return next(m["value"] for m in metrics if m["name"] == name and m["statistic"] == stat)
        self.assertAlmostEqual(find("ttft"), 100)
        self.assertAlmostEqual(find("tpot"), 100)
        self.assertAlmostEqual(find("e2e_latency"), 300)
        self.assertEqual(find("output_throughput"), 6)
        self.assertEqual(find("output_tokens", "sum"), 6)
        self.assertEqual(evidence["duration_seconds"], 1)
        itl = next(m for m in metrics if m["name"] == "itl")
        self.assertEqual(itl["sample_size"], 4)

    def test_text_and_single_token_do_not_invent_latency(self):
        fixture(self.path, output_tokens=1)
        for stream in (True, False):
            metrics, _, error = map_performance(self.path, 2, {}, stream=stream)
            self.assertIsNone(error)
            for metric in metrics:
                if metric["name"] in (("tpot", "itl") if stream else ("ttft", "tpot", "itl")):
                    self.assertIsNone(metric["value"])
                    self.assertIn("notes", metric)

    def test_failed_empty_and_missing_requests_rejected(self):
        for mode in ("failed", "empty", "missing"):
            with self.subTest(mode=mode):
                directory = self.path / mode
                fixture(directory, failed=mode == "failed", empty=mode == "empty")
                _, _, error = map_performance(directory, 3 if mode == "missing" else 2, {}, stream=True)
                self.assertEqual(error.reason, "incomplete_requests")

    def test_summary_inconsistency_and_nonfinite_rejected(self):
        fixture(self.path)
        file = next(self.path.glob("aisbench/**/performances/*/*.json"))
        common = json.loads(file.read_text())
        for key, value in (("Total Generated Tokens", 123), ("Benchmark Duration", "nan ms"), ("Total Requests", 9)):
            with self.subTest(key=key):
                changed = dict(common, **{key: {"total": value}})
                runtime.dump(file, changed)
                with self.assertRaises(runtime.Failure):
                    map_performance(self.path, 2, {}, stream=True)

    def test_percentile_interpolation(self):
        self.assertEqual(statistics([0, 10])["p90"], 9)
        self.assertIsNone(statistics([])["p50"])

    def test_prefix_generation_deterministic_and_isolated(self):
        args = runner.parser().parse_args(self.base + ["--input_len=30", "--data_num=6", "--prefix_num=2", "--dp=2"])
        runner.validate(args)
        corpus = self.path / "corpus.jsonl"
        runtime.jsonl(corpus, [dict(question="A long first document with data."), dict(question="Another second document with details.")])
        one = generate(CharTokenizer(), args, corpus)
        two = generate(CharTokenizer(), args, corpus)
        self.assertEqual(one, two)
        rows, warmup = one
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(warmup), 4)
        self.assertTrue(all(len(r["question"]) == 30 for r in rows))
        evidence, prefixes = validate_prefixes(CharTokenizer(), rows, warmup)
        self.assertEqual(prefixes, 2)
        self.assertTrue(all(item["shared_chat_tokens"] >= 15 for item in evidence))
        self.assertFalse((self.path / "picked_ids.txt").exists())

    def test_ratio_zero_one_and_insufficient_suffix(self):
        corpus = self.path / "corpus.jsonl"
        runtime.jsonl(corpus, [dict(question="abcdefghijk")])
        for ratio in ("0%", "100%"):
            args = runner.parser().parse_args(self.base + ["--input_len=12", "--data_num=3", "--repeat_rate=" + ratio])
            runner.validate(args)
            rows, warmup = generate(CharTokenizer(), args, corpus)
            self.assertTrue(all(len(row["question"]) == 12 for row in rows))
            self.assertEqual(len(warmup), 0 if ratio == "0%" else 1)
        args.repeat_rate = .9
        with self.assertRaises(runtime.Failure):
            generate(CharTokenizer(), args, corpus)

    def test_supplied_prefix_must_match_tokens_and_chat_template(self):
        with self.assertRaises(runtime.Failure):
            validate_prefixes(CharTokenizer(), [dict(question="abc")], [dict(question="xyz")])
        class DifferentChat(CharTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return self.encode(str(len(messages[0]["content"])) + messages[0]["content"])
        with self.assertRaises(runtime.Failure):
            validate_prefixes(DifferentChat(), [dict(question="abcdef")], [dict(question="abc")])

    def integration(self, policy="required", bad_after=False, fail_requests=False, repeats=2):
        data, warmup = self.path / "data.jsonl", self.path / "warmup.jsonl"
        runtime.jsonl(data, [dict(question="abcdef"), dict(question="abcxyz")])
        runtime.jsonl(warmup, [dict(question="abc")])
        calls, snapshots = [], []
        def fake_capture(endpoints, model, timeout, directory, label):
            snapshots.append(label)
            if bad_after and label == "measurement-after":
                return {"a": {"error": "timeout"}}
            i = len(snapshots)
            return {"a": snapshot(i * 100, i * 20, i * 30)}
        def fake_run(execution, directory, args, dataset, template, **kwargs):
            calls.append((directory, kwargs))
            fixture(directory, len(runtime.read_jsonl(dataset / "test.jsonl")), kwargs["output_len"], failed=fail_requests)
            return 0
        output = self.path / "output"
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(output)}), \
             patch.object(runner, "verify_aisbench", return_value=(self.path, {})), \
             patch.object(runner, "tokenizer_info", return_value=(CharTokenizer(), {})), \
             patch.object(runner, "capture", side_effect=fake_capture), \
             patch.object(runtime.Execution, "run_aisbench", fake_run):
            code = runtime.run_main("aisbench-perf", runner.parser, runner.work, self.base +
                                    [f"--dataset={data}", f"--warmup_dataset={warmup}", "--settle_seconds=0",
                                     f"--repeat={repeats}", "--metrics_policy=" + policy, "--output_len=3"])
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        import yaml
        import jsonschema
        spec = yaml.safe_load((ROOT.parents[1] / "server/api/openapi.yaml").read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(dict(spec["components"]["schemas"]["Result"], components=spec["components"])).validate(result)
        return code, result, calls, snapshots

    def test_rounds_and_warmup_are_separate(self):
        code, result, calls, snapshots = self.integration()
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(snapshots, ["warmup-before", "warmup-after", "measurement-before", "measurement-after"] * 2)
        self.assertTrue(result["metadata"]["metrics_complete"])
        self.assertEqual(calls[0][1]["output_len"], 1)
        self.assertEqual(calls[1][1]["output_len"], 3)
        self.assertEqual({m["labels"]["round"] for m in result["metrics"]}, {1, 2})

    def test_required_failure_keeps_performance(self):
        code, result, _, _ = self.integration(bad_after=True)
        self.assertEqual(code, 1)
        self.assertFalse(result["metadata"]["metrics_complete"])
        self.assertTrue(any(m["name"] == "output_throughput" and m["labels"]["phase"] == "measurement" for m in result["metrics"]))

    def test_best_effort_only_relaxes_cache_collection(self):
        code, result, _, _ = self.integration(policy="best_effort", bad_after=True)
        self.assertEqual(code, 0)
        self.assertFalse(result["metadata"]["metrics_complete"])
        self.assertIsNone(result["metadata"]["external_lookup_observed"])

    def test_best_effort_does_not_ignore_request_failures(self):
        code, result, _, _ = self.integration(policy="best_effort", fail_requests=True)
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["detail"]["reason"], "incomplete_requests")


if __name__ == "__main__":
    unittest.main()
