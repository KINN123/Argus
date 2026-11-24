import csv
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
from dataset_generator import prepare_data
from prefix_metrics import parse_snapshot, compare, choose_profile
from test_performance import CharTokenizer, fixture
from running_monitor import RunningMonitor


def vllm(prompt=0, query=0, hit=0, ext_query=None, ext_hit=None, engine="0"):
    labels = '{model_name="m",engine="' + engine + '"}'
    counters = {"vllm:prompt_tokens_total": prompt, "vllm:prefix_cache_queries_total": query,
                "vllm:prefix_cache_hits_total": hit}
    if ext_query is not None:
        counters["external_prefix_cache_queries_total"] = ext_query
    if ext_hit is not None:
        counters["external_prefix_cache_hits_total"] = ext_hit
    return parse_snapshot("\n".join(f"{k}{labels} {v}" for k, v in counters.items()), "m")


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.base = ["--url=http://x/proxy/v1", "--model_name=m", f"--model_path={self.path}",
                     "--input_len=32", "--output_len=3", "--data_num=4", "--concurrency=2",
                     "--monitor_interval_seconds=0", "--settle_seconds=0"]

    def test_reference_denominators_are_not_interchangeable(self):
        before = {"a": vllm(ext_query=0, ext_hit=0)}
        after = {"a": vllm(prompt=200, query=100, hit=60, ext_query=40, ext_hit=20)}
        _, prompt, complete = compare(before, after, "vllm_prompt", {})
        self.assertTrue(complete)
        self.assertEqual(prompt["hbm"]["rate"], 30)
        self.assertEqual(prompt["external"]["rate"], 10)
        _, prefix, complete = compare(before, after, "vllm_prefix", {})
        self.assertTrue(complete)
        self.assertEqual(prefix["hbm"]["rate"], 60)
        self.assertEqual(prefix["external"]["rate"], 50)

    def test_prompt_profile_accepts_external_vllm_alias(self):
        s = parse_snapshot('vllm:prompt_tokens_total{engine="0"} 100\nvllm:prefix_cache_hits_total{engine="0"} 30\nvllm:external_prefix_cache_hits_total{engine="0"} 20', "m")
        _, _, complete = compare({"a": s}, {"a": s}, "vllm_prompt", {})
        self.assertTrue(complete)

    def test_auto_keeps_missing_external_as_null(self):
        before, after = {"a": vllm()}, {"a": vllm(prompt=200, query=100, hit=60)}
        self.assertEqual(choose_profile(before, "auto"), "vllm_prefix")
        _, summary, complete = compare(before, after, "vllm_prefix", {})
        self.assertFalse(complete)
        self.assertEqual(summary["hbm"]["rate"], 60)
        self.assertIsNone(summary["external"]["rate"])

    def test_private_variable_random_data_and_seed(self):
        args = runner.parser().parse_args(self.base + ["--prefix_mode=private", "--length_min=28", "--length_max=40"])
        runner.validate(args)
        rows, warmup, count, evidence = prepare_data(args, CharTokenizer(), self.path / "one")
        self.assertEqual((len(rows), len(warmup), count), (4, 4, 4))
        self.assertTrue(all(28 <= t["content"] <= 40 for t in evidence["token_lengths"]))
        rows2, _, _, _ = prepare_data(args, CharTokenizer(), self.path / "two")
        self.assertEqual(rows, rows2)
        args.seed = 0
        _, _, _, random_evidence = prepare_data(args, CharTokenizer(), self.path / "three")
        self.assertGreater(random_evidence["effective_seed"], 0)

    def test_shared_variable_and_full_prefix(self):
        for rate in ("0%", "50%", "100%"):
            args = runner.parser().parse_args(self.base + ["--repeat_rate=" + rate, "--length_min=28", "--length_max=40"])
            runner.validate(args)
            rows, _, _, evidence = prepare_data(args, CharTokenizer(), self.path / rate)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(28 <= t["content"] <= 40 for t in evidence["token_lengths"]))

    def execute(self, extra=(), warm_missing=False, required=False, fail=False):
        calls, index = [], 0
        def capture(endpoints, model, timeout, directory, label):
            nonlocal index
            index += 1
            if warm_missing and label == "warmup-before":
                return {e: parse_snapshot("", model) for e in endpoints}
            return {e: vllm(prompt=index * 200, query=index * 100, hit=index * 60,
                            ext_query=index * 40 if required else None, ext_hit=index * 20 if required else None) for e in endpoints}
        def run(execution, directory, args, dataset, template, **kwargs):
            calls.append(dict(kwargs, directory=str(directory), input_len=args.input_len, seed=args.seed))
            fixture(directory, len(runtime.read_jsonl(dataset / "test.jsonl")), kwargs["output_len"], failed=fail)
            # Emulate reference CSV content, including fields unknown to this Runner.
            csv_path = next(directory.glob("aisbench/**/performances/*/*.json")).with_suffix(".csv")
            csv_path.write_text('Performance Parameters,Average\nTTFT,100\nCustom Metric,42\n', encoding="utf-8")
            return 0
        output = self.path / "result"
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(output)}), \
             patch.object(runner, "tokenizer_info", return_value=(CharTokenizer(), {})), \
             patch.object(runner, "capture", side_effect=capture), \
             patch.object(runtime.Execution, "run_aisbench", run):
            args = self.base + list(extra)
            if required:
                args += ["--metrics_policy=required"]
            code = runtime.run_main("aisbench-perf", runner.parser, runner.work, args)
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        return code, result, output, calls

    def test_hbm_only_service_produces_reports(self):
        code, result, output, calls = self.execute(extra=["--metrics_endpoint=http://a/metrics", "--metrics_endpoint=http://b/metrics"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        rows = [json.loads(line) for line in (output / "prefix_bench_result.jsonl").read_text(encoding="utf-8").splitlines()]
        warm, full = rows
        self.assertEqual((warm["phase"], warm["output_len"], warm["data_num"]), ("warmup", 1, 1))
        self.assertEqual(full["hbm_hit_rate_total"], .6)
        self.assertIsNone(full["ext_hit_rate_total"])
        self.assertEqual(len(full["dp_sources"]), 2)
        self.assertNotEqual(full["dp_sources"]["dp0"]["endpoint"], full["dp_sources"]["dp1"]["endpoint"])
        with (output / "hit_rate.csv").open(encoding="utf-8", newline="") as handle:
            endpoint_rows = list(csv.DictReader(handle))
        self.assertEqual(len(endpoint_rows), 2)
        self.assertEqual(endpoint_rows[0]["aisbench_avg:Custom Metric"], "42")
        self.assertEqual(endpoint_rows[0]["ext_hit_rate"], "")
        self.assertEqual(result["schema_version"], "2.0")

    def test_required_does_not_block_cold_warmup(self):
        code, _, _, calls = self.execute(warm_missing=True, required=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)

    def test_rounds_override_and_keep_reports_separate(self):
        code, result, output, calls = self.execute(extra=['--rounds=[{"input_len":40,"seed":4},{"input_len":48,"seed":5}]', '--extra_info={"tag":"jenkins"}'])
        self.assertEqual(code, 0)
        self.assertEqual([c["input_len"] for c in calls], [40, 40, 48, 48])
        rows = [json.loads(line) for line in (output / "prefix_bench_result.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["round"] for r in rows], [1, 1, 2, 2])
        self.assertEqual({m["labels"]["round"] for m in result["metrics"]}, {1, 2})
        self.assertEqual(len(result["metadata"]["groups"]), 2)
        legacy = json.loads((output / "legacy_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(legacy), 2)
        self.assertTrue(all(r["type"] == "prefix" and r["status"] == "SUCCESS" for r in legacy))
        self.assertTrue(all(r["extra_info"] == {"tag": "jenkins"} for r in legacy))

    def test_request_failure_is_recorded_in_flat_report(self):
        code, result, output, calls = self.execute(fail=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        row = json.loads((output / "prefix_bench_result.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["output_token_throughput"])

    def test_metrics_url_and_pod_info(self):
        args = runner.parser().parse_args(self.base)
        runner.validate(args)
        self.assertEqual(args.metrics_endpoint, ["http://x/proxy/metrics"])
        args = runner.parser().parse_args(self.base + ["--pod_info", "[::1]:7800", "host:8000"])
        runner.validate(args)
        self.assertEqual(args.metrics_endpoint, ["http://[::1]:7800/metrics", "http://host:8000/metrics"])

    def test_running_monitor_sums_model_engines_and_records_sample(self):
        args = runner.parser().parse_args(self.base)
        runner.validate(args)
        monitor = RunningMonitor(args, self.path / "monitor", "measurement")
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit):
                monitor.stop_event.set()
                return (b'vllm:num_requests_running{model_name="m",engine="0"} 2\n'
                        b'vllm:num_requests_running{model_name="m",engine="1"} 3\n'
                        b'vllm:num_requests_running{model_name="other",engine="0"} 100\n')
        with patch("running_monitor.build_opener") as opener:
            opener.return_value.open.return_value = Response()
            monitor.run()
        self.assertEqual(monitor.summary[args.metrics_endpoint[0]], dict(max=5, samples=1))
        rows = (self.path / "monitor/measurement-running-samples.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(rows[0])["value"], 5)

    def test_invalid_round_is_rejected_before_any_request(self):
        code, result, output, calls = self.execute(extra=['--rounds=[{"input_len":32},{"concurrency":0}]'])
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])

    def test_corpus_mode_remains_available(self):
        args = runner.parser().parse_args(self.base + ["--data_source=corpus"])
        runner.validate(args)
        rows, _, _, evidence = prepare_data(args, CharTokenizer(), self.path / "corpus")
        self.assertEqual(len(rows), 4)
        self.assertIn("corpus_sha256", evidence)


if __name__ == "__main__":
    unittest.main()
