import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import os
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefix_metrics import parse_snapshot, compare, capture, require_valid
import runner
import runtime


def snapshot(query=0, hbm=0, external=0, engine="0", start=10):
    text = f'''ucm:total_prefix_query_tokens_total{{model_name="m",engine="{engine}"}} {query}
ucm:gpu_hbm_hit_tokens_total{{model_name="m",engine="{engine}"}} {hbm}
ucm:ucm_hit_tokens_total{{model_name="m",engine="{engine}"}} {external}
process_start_time_seconds {start}
'''
    return parse_snapshot(text, "m")


class PrefixTests(unittest.TestCase):
    def compare(self, before, after, profile="ucm_connector"):
        return compare(before, after, profile, {"phase": "measurement", "round": 1})

    def test_weighted_all_instances(self):
        _, result, complete = self.compare({"a": snapshot(), "b": snapshot()},
                                           {"a": snapshot(10, 9, 1), "b": snapshot(90, 1, 0)})
        self.assertTrue(complete)
        self.assertEqual(result["hbm"]["rate"], 10)
        self.assertEqual(result["ucm"]["rate"], 1)

    def test_zero_queries_is_null_but_valid_zero_hits_is_zero(self):
        _, result, complete = self.compare({"a": snapshot()}, {"a": snapshot()})
        self.assertTrue(complete)
        self.assertIsNone(result["ucm"]["rate"])
        _, result, _ = self.compare({"a": snapshot()}, {"a": snapshot(10)})
        self.assertEqual(result["ucm"]["rate"], 0)

    def test_missing_external_preserves_hbm_but_not_all_external(self):
        after = snapshot(10, 5, 3)
        after["samples"] = {k: v for k, v in after["samples"].items() if k[0] != "ucm:ucm_hit_tokens_total"}
        metrics, result, complete = self.compare({"a": snapshot()}, {"a": after})
        self.assertFalse(complete)
        self.assertEqual(result["hbm"]["rate"], 50)
        self.assertIsNone(result["ucm"]["rate"])
        self.assertTrue(any(m.get("notes") for m in metrics if m["value"] is None))

    def test_partial_source_never_masquerades_as_all(self):
        _, result, complete = self.compare({"a": snapshot(), "b": snapshot()},
                                           {"a": snapshot(10, 5, 3), "b": {"error": "timeout"}})
        self.assertFalse(complete)
        self.assertIsNone(result["hbm"]["rate"])
        self.assertEqual(len(result["hbm"]["instances"]), 1)

    def test_reset_restart_and_instance_change_are_invalid(self):
        for before, after in ((snapshot(100, 20, 10), snapshot(10, 2, 1)),
                              (snapshot(), snapshot(1000, 600, 200, start=20)),
                              (snapshot(), snapshot(10, 5, 2, engine="1")),
                              (snapshot(100, 20, 10), snapshot(101, 22, 10))):
            with self.subTest(after=after):
                _, result, complete = self.compare({"a": before}, {"a": after})
                self.assertFalse(complete)
                self.assertIsNone(result["hbm"]["rate"])

    def test_prometheus_escaping_timestamp_and_model_filter(self):
        text = 'ucm:total_prefix_query_tokens_total{model_name="m",engine="a\\\"b\\\\c"} 1e2 1700000000000\n'
        text += 'ucm:total_prefix_query_tokens_total{model_name="another",engine="0"} 999\n'
        parsed = parse_snapshot(text, "m")
        self.assertEqual(len(parsed["samples"]), 1)
        self.assertEqual(next(iter(parsed["samples"].values())), 100)
        with self.assertRaises(ValueError):
            parse_snapshot(text + text, "m")
        with self.assertRaises(ValueError):
            parse_snapshot('ucm:ucm_hit_tokens_total NaN\n', "m")

    def test_vllm_conditional_denominator_and_alias(self):
        def vllm(q, h, eq, eh):
            return parse_snapshot(f'vllm:prefix_cache_queries_total {q}\nvllm:prefix_cache_hits_total {h}\nvllm:external_prefix_cache_queries_total {eq}\nvllm:external_prefix_cache_hits_total {eh}\n', "m")
        _, result, complete = self.compare({"a": vllm(0, 0, 0, 0)}, {"a": vllm(1000, 600, 400, 200)}, "vllm_prefix")
        self.assertTrue(complete)
        self.assertEqual(result["external"]["rate"], 50)
        self.assertEqual(result["external"]["denominator"], "external_queries")

    def test_both_external_aliases_are_not_double_counted(self):
        text = 'vllm:prefix_cache_queries_total 1\nvllm:prefix_cache_hits_total 0\n'
        text += 'external_prefix_cache_queries_total 1\nexternal_prefix_cache_hits_total 0\n'
        text += 'vllm:external_prefix_cache_queries_total 1\nvllm:external_prefix_cache_hits_total 0\n'
        parsed = parse_snapshot(text, "m")
        _, result, complete = self.compare({"a": parsed}, {"a": parsed}, "vllm_prefix")
        self.assertFalse(complete)
        self.assertTrue(result["hbm"]["complete"])
        self.assertFalse(result["external"]["complete"])

    def test_collection_uses_one_response_per_source_and_own_key(self):
        content = b'ucm:total_prefix_query_tokens_total 10\nucm:gpu_hbm_hit_tokens_total 5\nucm:ucm_hit_tokens_total 3\n'
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return content
        requests = []
        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                return Response()
        with tempfile.TemporaryDirectory() as temp, patch("prefix_metrics.build_opener", return_value=Opener()), \
             patch.dict(os.environ, {"ENDPOINT_API_KEY": "model-secret", "METRICS_API_KEY": "metrics-secret"}):
            result = capture(["http://a/metrics"], "m", 1, Path(temp), "before")
            self.assertIn("samples", result["http://a/metrics"])
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].get_header("Authorization"), "Bearer metrics-secret")
            for path in Path(temp).rglob("*"):
                if path.is_file():
                    self.assertNotIn("secret", path.read_text())

    def test_retry_is_bounded_and_evidence_saved(self):
        class Opener:
            calls = 0
            def open(self, *args, **kwargs):
                self.calls += 1
                raise OSError("offline")
        opener = Opener()
        with tempfile.TemporaryDirectory() as temp, patch("prefix_metrics.build_opener", return_value=opener):
            result = capture(["http://a/metrics"], "m", 1, Path(temp), "before")
            self.assertEqual(opener.calls, 2)
            self.assertIn("error", result["http://a/metrics"])
            evidence = json.loads((Path(temp) / "before-collection.json").read_text())
            self.assertEqual(len(evidence["http://a/metrics"]), 2)

    def test_invalid_argument_combinations_and_required_policy(self):
        base = ["--url=http://x", "--model_name=m", "--model_path=/m", "--metrics_endpoint=http://x/metrics"]
        for extra in (["--prefix_test=yes"], ["--repeat_rate=NaN"], ["--repeat_rate=10%%"], ["--settle_seconds=61"], ["--concurrency=0"],
                      ["--dataset_type=normal", "--prefix_test=true"], ["--dataset=/d", "--input_len=100"],
                      ["--dataset=/d"], ["--warmup_dataset=/w"], ["--metrics_endpoint=http://x/metrics"]):
            with self.subTest(extra=extra), self.assertRaises(runtime.Failure):
                runner.validate(runner.parser().parse_args(base + extra))
        args = runner.parser().parse_args(base)
        with self.assertRaises(runtime.Failure):
            require_valid({"x": {"error": "missing"}}, args, {})
        args.metrics_policy = "best_effort"
        require_valid({"x": {"error": "missing"}}, args, {})


if __name__ == "__main__":
    unittest.main()
