"""Version recording must never become a compatibility gate."""
import importlib.metadata
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
    def test_record_versions_without_reading_source_or_importing_interfaces(self):
        for version in ("3.1", "3.10", "4.0", "vendor-build"):
            for old_mode in ("strict", "compatible", "invalid"):
                with self.subTest(version=version, old_mode=old_mode), \
                     patch.dict(os.environ, {"TESTPLANE_AISBENCH_CHECK": old_mode}), \
                     patch.object(runtime.importlib.metadata, "version", return_value=version), \
                     patch.object(runtime.Path, "open", side_effect=AssertionError("unexpected source/lock read")), \
                     patch.object(runtime.importlib, "import_module", side_effect=AssertionError("unexpected interface preflight")):
                    metadata = runtime.aisbench_info()
                self.assertEqual(metadata["aisbench_version"], version)
                self.assertEqual(metadata["python_version"], sys.version.split()[0])
                self.assertNotIn("aisbench_commit", metadata)
                self.assertNotIn("aisbench_source_verified", metadata)

    def test_missing_or_broken_metadata_is_nonfatal(self):
        for error in (importlib.metadata.PackageNotFoundError("ais_bench_benchmark"), ValueError("bad metadata")):
            with self.subTest(error=type(error).__name__), \
                 patch.object(runtime.importlib.metadata, "version", side_effect=error):
                metadata = runtime.aisbench_info()
            self.assertIsNone(metadata["aisbench_version"])
            self.assertIsNone(metadata["transformers_version"])

    def test_actual_template_read_does_not_need_lock_or_perf_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            configs = package / "configs/datasets/gsm8k"
            configs.mkdir(parents=True)
            for preset, filename in (("gsm8k_4shot_cot", "gsm8k_gen_4_shot_cot_str.py"),
                                     ("gsm8k_jsonl_0shot", "gsm8k_gen_0_shot_cot_str.py")):
                (configs / filename).write_text('prompt = dict(template="{question}")', encoding="utf-8")
                with patch.object(runtime.importlib, "import_module", return_value=types.SimpleNamespace(__file__=str(package / "__init__.py"))):
                    self.assertEqual(runtime.official_template(preset), "{question}")
            self.assertFalse((package / "utils/perf_details.py").exists())
            self.assertFalse((package / "build/aisbench-lock.json").exists())


if __name__ == "__main__":
    unittest.main()
