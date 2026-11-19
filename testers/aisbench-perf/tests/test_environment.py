"""Exercise installed-source drift without requiring AISBench on the test host."""
import importlib.metadata
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
import runtime


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.package = self.path / "installed"
        self.package.mkdir()
        self.source = self.package / "model.py"
        self.source.write_text("# baseline\n", encoding="utf-8")
        self.lock = self.path / "build/aisbench-lock.json"
        runtime.dump(self.lock, dict(commit=runtime.AISBENCH_COMMIT, files={"model.py": runtime.digest(self.source)}))
        self.versions = {"ais_bench_benchmark": "3.1.0", "transformers": "4.51.3"}
        for patcher in (
            patch.object(runtime, "HERE", self.path),
            patch.object(runtime.importlib.util, "find_spec", return_value=types.SimpleNamespace(origin=str(self.package / "__init__.py"))),
            patch.object(runtime.importlib.metadata, "version", side_effect=self.versions.__getitem__),
            patch.dict(os.environ, {"TESTPLANE_OUTPUT_DIR": str(self.path / "output")}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # Test the default independently of the caller's shell environment.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("TESTPLANE_AISBENCH_CHECK", None)

    def test_default_checks_locked_source(self):
        root, metadata = runtime.verify_aisbench()
        self.assertEqual(root, self.package)
        self.assertEqual(metadata["aisbench_check_mode"], "strict")
        self.assertTrue(metadata["aisbench_source_verified"])
        self.assertEqual(metadata["aisbench_commit"], runtime.AISBENCH_COMMIT)

    def test_strict_rejects_different_source_with_same_version(self):
        self.source.write_text("# different 3.1 build\n", encoding="utf-8")
        with self.assertRaises(runtime.Failure) as caught:
            runtime.verify_aisbench()
        self.assertEqual(caught.exception.reason, "unsupported_backend")
        self.assertIn("model.py", str(caught.exception))
        self.assertIn("TESTPLANE_AISBENCH_CHECK=compatible", str(caught.exception))

    def test_compatible_records_drift_and_does_not_hide_later_failure(self):
        self.source.write_text("# different 3.1 build\n", encoding="utf-8")
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        def parser():
            result = runtime.Parser()
            runtime.public_args(result)
            return result
        def work(execution, args):
            _, metadata = runtime.verify_aisbench(execution.log)
            execution.result["metadata"].update(metadata)
            raise runtime.Failure("实际结果格式不兼容", "upstream_result_invalid", 1)
        code = runtime.run_main("test", parser, work,
                                ["--url=http://localhost", "--model_name=m", f"--model_path={self.path}"])
        self.assertEqual(code, 1)
        result = json.loads((self.path / "output/result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["detail"]["reason"], "upstream_result_invalid")
        metadata = result["metadata"]
        self.assertEqual(metadata["aisbench_version"], "3.1.0")
        self.assertEqual(metadata["aisbench_check_mode"], "compatible")
        self.assertFalse(metadata["aisbench_source_verified"])
        self.assertIsNone(metadata["aisbench_commit"])
        self.assertEqual(metadata["aisbench_expected_commit"], runtime.AISBENCH_COMMIT)
        self.assertEqual(metadata["aisbench_source_mismatches"], ["model.py"])
        self.assertIn("compatible", (self.path / "output/logs/runner.log").read_text(encoding="utf-8"))

    def test_compatible_accepts_31_release_variants(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        for version in ("3.1", "3.1.0", "3.1.1", "3.1.0rc1", "3.1rc1", "3.1.0.dev1", "3.1.0+vendor.1"):
            with self.subTest(version=version):
                self.versions["ais_bench_benchmark"] = version
                _, metadata = runtime.verify_aisbench()
                self.assertEqual(metadata["aisbench_version"], version)

    def test_compatible_rejects_other_series(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        for version in ("3.10", "3.10.1", "3.2", "4.0", "unknown"):
            with self.subTest(version=version):
                self.versions["ais_bench_benchmark"] = version
                with self.assertRaises(runtime.Failure) as caught:
                    runtime.verify_aisbench()
                self.assertEqual(caught.exception.exit_code, 2)
                self.assertIn(version, str(caught.exception))

    def test_compatible_still_rejects_missing_package(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        with patch.object(runtime.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(runtime.Failure) as caught:
                runtime.verify_aisbench()
        self.assertEqual(caught.exception.reason, "unsupported_backend")
        self.assertIn(sys.executable, str(caught.exception))
        self.assertIn("ais_bench.benchmark", str(caught.exception))

    def test_compatible_still_rejects_missing_dependency(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        with patch.object(runtime.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError("transformers")):
            with self.assertRaises(runtime.Failure) as caught:
                runtime.verify_aisbench()
        self.assertIn("transformers", str(caught.exception))
        self.assertIn(sys.executable, str(caught.exception))

    def test_compatible_still_rejects_missing_source_file(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        self.source.unlink()
        with self.assertRaises(runtime.Failure) as caught:
            runtime.verify_aisbench()
        self.assertIn("model.py", str(caught.exception))
        self.assertEqual(caught.exception.exit_code, 2)

    def test_compatible_still_rejects_broken_lock(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "compatible"
        for content in ("{broken", '{}', json.dumps(dict(commit=runtime.AISBENCH_COMMIT, files={}))):
            with self.subTest(content=content):
                self.lock.write_text(content, encoding="utf-8")
                with self.assertRaises(runtime.Failure):
                    runtime.verify_aisbench()

    def test_invalid_mode_is_an_input_error(self):
        os.environ["TESTPLANE_AISBENCH_CHECK"] = "skip"
        with self.assertRaises(runtime.Failure) as caught:
            runtime.verify_aisbench()
        self.assertEqual(caught.exception.reason, "invalid_input")


if __name__ == "__main__":
    unittest.main()
