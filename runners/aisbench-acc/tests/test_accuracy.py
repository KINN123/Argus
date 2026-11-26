import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runner
import runtime
from result_mapper import map_accuracy


def fixture(directory, correct=(True, False), success=None):
    (directory / "requests").mkdir(parents=True)
    details = {}
    for i, good in enumerate(correct):
        prediction = "The answer is 2" if good else "The answer is 9"
        runtime.dump(directory / "requests" / f"{i}.json", dict(id=i, success=True if success is None else success[i],
                     prediction=prediction, request={"messages": [{"role": "user", "content": "1+1?"}]}))
        details[str(i)] = dict(correct=good, predictions="2" if good else "9", references="2", origin_prediction=prediction)
    runtime.dump(directory / f"aisbench/20261117/results/{runtime.MODEL}/{runtime.DATA}.json",
                 dict(accuracy=100 * sum(correct) / len(correct), details=details))
    return [dict(question="1+1?", answer="#### 2") for _ in correct]


def assert_schema(test, result):
    import yaml
    import jsonschema
    spec = yaml.safe_load((ROOT.parents[1] / "server/api/openapi.yaml").read_text(encoding="utf-8"))
    schema = dict(spec["components"]["schemas"]["Result"], components=spec["components"])
    jsonschema.Draft202012Validator(schema).validate(result)
    test.assertIn(result["status"], ("succeeded", "failed"))


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_score_comes_from_per_question_results(self):
        rows = fixture(self.path)
        metrics, error = map_accuracy(self.path, rows, {})
        self.assertIsNone(error)
        self.assertEqual(next(m["value"] for m in metrics if m["name"] == "accuracy"), 50)
        self.assertTrue((self.path / "sample_scores.jsonl").is_file())

    def test_zero_score_is_success(self):
        metrics, error = map_accuracy(self.path, fixture(self.path, (False, False)), {})
        self.assertIsNone(error)
        self.assertEqual(metrics[-1]["value"], 0)

    def replica_fixture(self, correct=(True, False)):
        rows = fixture(self.path, correct)
        path = next(self.path.glob("aisbench/**/results/*/*.json"))
        value = json.loads(path.read_text())
        # Shape produced by BaseEvaluator.evaluate(n=1) + format_details.
        for detail in value["details"].values():
            for key in ("correct", "predictions", "references"):
                detail[key] = [detail[key]]
        runtime.dump(path, value)
        return rows, path, value

    def test_single_replica_lists_keep_scores_and_original_evidence(self):
        rows, path, original = self.replica_fixture()
        metrics, error = map_accuracy(self.path, rows, {})
        self.assertIsNone(error)
        self.assertEqual(metrics[-1]["value"], 50)
        scored = runtime.read_jsonl(self.path / "sample_scores.jsonl")
        self.assertEqual([r["correct"] for r in scored], [True, False])
        self.assertEqual([r["predictions"] for r in scored], ["2", "9"])
        self.assertEqual(json.loads(path.read_text()), original)

    def test_single_replica_all_wrong_is_valid_zero_score(self):
        rows, _, _ = self.replica_fixture((False, False))
        metrics, error = map_accuracy(self.path, rows, {})
        self.assertIsNone(error)
        self.assertEqual(metrics[-1]["value"], 0)

    def test_replica_lists_do_not_accept_ambiguous_or_nonboolean_scores(self):
        rows, path, original = self.replica_fixture()
        for key, invalid in (("correct", []), ("correct", [True, False]),
                             ("correct", [1]), ("correct", ["False"]),
                             ("correct", [[True]]), ("predictions", []),
                             ("predictions", ["2", "9"]), ("references", []),
                             ("references", [["2"]])):
            with self.subTest(key=key, invalid=invalid):
                value = json.loads(json.dumps(original))
                value["details"]["0"][key] = invalid
                runtime.dump(path, value)
                with self.assertRaises(runtime.Failure):
                    map_accuracy(self.path, rows, {})

    def test_replica_lists_still_require_auditable_predictions_and_summary(self):
        rows, path, original = self.replica_fixture()
        for change in ("reference", "prediction", "summary", "missing", "object"):
            with self.subTest(change=change):
                value = json.loads(json.dumps(original))
                detail = value["details"]["0"]
                if change == "reference":
                    detail["references"] = ["42"]
                elif change == "prediction":
                    detail["origin_prediction"] = "another answer"
                elif change == "summary":
                    value["accuracy"] = 100
                elif change == "missing":
                    del detail["correct"]
                else:
                    value["details"]["0"] = None
                runtime.dump(path, value)
                with self.assertRaises(runtime.Failure):
                    map_accuracy(self.path, rows, {})

    def test_failed_request_is_not_a_wrong_answer(self):
        metrics, error = map_accuracy(self.path, fixture(self.path, success=(True, False)), {})
        self.assertEqual(error.reason, "incomplete_requests")
        self.assertIsNone(metrics[-1]["value"])

    def test_missing_request_does_not_shrink_denominator(self):
        rows = fixture(self.path)
        (self.path / "requests/1.json").unlink()
        metrics, error = map_accuracy(self.path, rows, {})
        self.assertEqual(error.reason, "incomplete_requests")
        failed = next(m for m in metrics if m["labels"].get("kind") == "request_failed")
        self.assertEqual(failed["value"], 1)

    def test_inconsistent_summary_or_reference_rejected(self):
        rows = fixture(self.path)
        result = next(self.path.glob("aisbench/**/results/*/*.json"))
        value = json.loads(result.read_text())
        value["accuracy"] = 100
        runtime.dump(result, value)
        with self.assertRaises(runtime.Failure):
            map_accuracy(self.path, rows, {})
        value["accuracy"] = 50
        value["details"]["0"]["references"] = "42"
        runtime.dump(result, value)
        with self.assertRaises(runtime.Failure):
            map_accuracy(self.path, rows, {})

    def test_missing_details_rejected(self):
        rows = fixture(self.path)
        result = next(self.path.glob("aisbench/**/results/*/*.json"))
        runtime.dump(result, {"accuracy": 50})
        with self.assertRaises(runtime.Failure):
            map_accuracy(self.path, rows, {})

    def test_data_validation_and_subset(self):
        data = self.path / "input.jsonl"
        runtime.jsonl(data, [dict(question="q", answer="2"), dict(question="r", answer="3")])
        args = runner.parser().parse_args(["--url=http://x", "--model_name=m", f"--model_path={self.path}",
                                           f"--dataset={data}", "--dataset_config=gsm8k_jsonl_0shot", "--max_samples=1"])
        _, rows, evidence = runner.prepare_data(args, self.path / "out")
        self.assertEqual(rows, [{"question": "q", "answer": "#### 2"}])
        self.assertEqual(evidence["sample_ids"], [0])
        args.max_samples = 3
        with self.assertRaises(runtime.Failure):
            runner.prepare_data(args, self.path / "bad")
        args.max_samples = 0
        runtime.jsonl(data, [dict(question="q", answer="none")])
        with self.assertRaises(runtime.Failure):
            runner.prepare_data(args, self.path / "bad")

    def test_four_shot_train_not_cut_by_max_samples(self):
        data = self.path / "gsm8k"
        rows = [dict(question=str(i), answer="#### 2") for i in range(4)]
        runtime.jsonl(data / "train.jsonl", rows)
        runtime.jsonl(data / "test.jsonl", rows)
        args = runner.parser().parse_args(["--url=http://x", "--model_name=m", f"--model_path={self.path}", f"--dataset={data}", "--max_samples=1"])
        staged, selected, _ = runner.prepare_data(args, self.path / "out")
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(runtime.read_jsonl(staged / "train.jsonl")), 4)

    def test_replay_runs_twice_and_result_validates_against_openapi(self):
        data = self.path / "questions.jsonl"
        runtime.jsonl(data, [dict(question="1+1?", answer="2")])
        calls = []
        def fake_run(execution, directory, *args, **kwargs):
            calls.append(directory)
            fixture(directory, (True,))
            return 0
        class Tokenizer:
            chat_template = "mock"
        with patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(self.path / "output")}), \
             patch.object(runner, "aisbench_info", return_value={"aisbench_version": "fixture"}), \
             patch.object(runner, "tokenizer_info", return_value=(Tokenizer(), {})), \
             patch.object(runner, "official_template", return_value="{question}"), \
             patch.object(runtime.Execution, "run_aisbench", fake_run):
            code = runtime.run_main("aisbench-acc", runner.parser, runner.work,
                                    ["--url=http://x", "--model_name=m", f"--model_path={self.path}", f"--dataset={data}",
                                     "--dataset_config=gsm8k_jsonl_0shot", "--replay_rounds=2"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        result = json.loads((self.path / "output/result.json").read_text(encoding="utf-8"))
        assert_schema(self, result)
        self.assertEqual({m["labels"]["round"] for m in result["metrics"]}, {1, 2})

    def test_help_has_no_side_effects(self):
        output = self.path / "not-created"
        proc = subprocess.run([sys.executable, str(ROOT / "runner.py"), "--help"], capture_output=True,
                              env=dict(os.environ, TESTPLANE_OUTPUT_DIR=str(output)))
        self.assertEqual(proc.returncode, 0)
        self.assertFalse(output.exists())

    def test_invalid_cli_writes_failure_result(self):
        output = self.path / "output"
        output.mkdir()  # platform mounts an existing, empty output directory
        proc = subprocess.run([sys.executable, str(ROOT / "runner.py"), "--unknown=1"], capture_output=True,
                              env=dict(os.environ, TESTPLANE_OUTPUT_DIR=str(output)))
        self.assertEqual(proc.returncode, 2)
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert_schema(self, result)
        self.assertEqual(result["error"]["detail"]["reason"], "invalid_input")

    def test_unknown_duplicate_and_nonfinite_parameters(self):
        base = ["--url=http://x", "--model_name=m", "--model_path=/m", "--dataset=/d"]
        for extra in (["--url=http://y"], ["--temperature=nan"], ["--request_rate=inf"], ["--output_len=0"], ["--extra_args=bad"], ["--concurrency=2", "--concurrency=3"]):
            with self.subTest(extra=extra), self.assertRaises(runtime.Failure):
                runner.parser().parse_args(base + extra)


if __name__ == "__main__":
    unittest.main()
