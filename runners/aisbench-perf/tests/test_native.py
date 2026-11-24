import ast
import copy
import csv
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
import native_suite as native
import runner
import runtime


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def args(self, *extra):
        return runner.parser().parse_args(['--url=http://x/v1', '--model_name=m', '--model_path=' + str(self.path),
                                          '--datasets=gsm8k_gen_4_shot_cot_str,mmlu_gen', *extra])

    def csv(self, directory, perf=False, failed=False):
        path = directory / ('aisbench/20261124/performances/model/data.csv' if perf else 'aisbench/20261124/summary/summary.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Performance Parameters,Stage,Average,Min,Max,P90\nTTFT,total,1e3 ms,-1 ms,2000 ms,1500 ms\n'
                        if perf else 'dataset,version,metric,mode,total_count,vllm-api-stream-chat\ngsm8k,v1,accuracy,gen,10,90.00 (9/10)\nmmlu,v2,accuracy,gen,10,50.00\n', encoding='utf-8')
        if perf:
            runtime.dump(path.with_suffix('.json'), {'Total Requests': {'total': 10}, 'Success Requests': {'total': 9 if failed else 10}, 'Failed Requests': {'total': 1 if failed else 0}, 'Request Throughput': {'total': '2 req/s'}})
        return path

    def test_native_command_preserves_jenkins_modes_and_has_private_workdir(self):
        for perf in (False, True):
            directory = self.path / str(perf)
            args = self.args('--num_prompts=10', '--num_warmups=2', '--ignore_eos=false', '--temperature=0.2')
            cmd, cwd = native.prepare_command(directory, args, perf)
            self.assertEqual(cmd[cmd.index('--mode') + 1], 'perf' if perf else 'all')
            self.assertEqual(cmd[cmd.index('--summarizer') + 1], 'default_perf' if perf else 'example')
            self.assertEqual('--merge-ds' in cmd, not perf)
            self.assertEqual(cmd[cmd.index('--datasets') + 1:cmd.index('--summarizer')], ['gsm8k_gen_4_shot_cot_str', 'mmlu_gen'])
            self.assertEqual(cmd[cmd.index('--num-prompts') + 1], '10')
            self.assertTrue(cwd.is_dir())
            spec = json.loads((directory / 'resolved.json').read_text())
            self.assertEqual(spec['generation'], dict(temperature=0.2, ignore_eos=False))
            ast.parse((directory / 'configs/models/testplane_native_chat.py').read_text())

    def test_defaults_and_arguments_not_silently_ignored(self):
        native.prepare_command(self.path / 'ok', self.args(), False)
        spec = json.loads((self.path / 'ok/resolved.json').read_text())
        self.assertEqual(spec['generation'], dict(temperature=0.01, ignore_eos=True))
        for option in ('--dataset=/data', '--max_samples=1' if ROOT.name.endswith('acc') else '--prefix_test=true'):
            with self.assertRaises(runtime.Failure):
                native.prepare_command(self.path / 'bad', self.args(option), ROOT.name.endswith('perf'))

    def test_invalid_dataset_names_and_native_flags_without_datasets(self):
        for names in ('a,a', '[]', '[1]', '--search', 'a;echo bad', 'a/path', '[bad'):
            with self.assertRaises(runtime.Failure):
                native.dataset_names(names)
        args = self.args('--num_prompts=1')
        with self.assertRaises(runtime.Failure):
            native.reject_native_only(args)
        self.assertEqual(native.dataset_names('["gsm8k_gen", "mmlu_gen"]'), ['gsm8k_gen', 'mmlu_gen'])

    def test_model_template_preserves_postprocessor_and_injects_secret_at_load(self):
        base = [dict(type='Chat', abbr='vllm-api-stream-chat', pred_postprocessor=dict(type='extract_reasoning'), generation_kwargs=dict(top_p=0.9))]
        config = types.ModuleType('mmengine.config')
        config.Config = types.SimpleNamespace(fromfile=lambda p: dict(models=copy.deepcopy(base)))
        with patch.dict(sys.modules, {'mmengine.config': config}), patch.object(native.importlib, 'import_module', return_value=types.SimpleNamespace(__file__=str(self.path / '__init__.py'))), patch.dict(os.environ, ENDPOINT_API_KEY='test-secret'):
            models = native.configured_models(dict(model_path='/model', model_name='m', url='http://x/v1', output_len=32, concurrency=2, request_rate=0, generation=dict(ignore_eos=True)))
        self.assertEqual(models[0]['pred_postprocessor'], base[0]['pred_postprocessor'])
        self.assertEqual(models[0]['generation_kwargs'], dict(top_p=0.9, ignore_eos=True))
        self.assertEqual(models[0]['api_key'], 'test-secret')
        self.assertNotIn('api_key', base[0])

    def test_accuracy_keeps_columns_and_parses_count_annotation(self):
        directory = self.path / 'raw/native'
        source = self.csv(directory)
        rows, metrics, errors = native.collect(directory, self.path, False)
        self.assertFalse(errors)
        self.assertEqual([m['value'] for m in metrics], [90, 50])
        self.assertEqual(rows[0]['vllm-api-stream-chat'], '90.00 (9/10)')
        self.assertEqual(source.read_text(), (self.path / 'accuracy.csv').read_text())
        source.write_text(source.read_text().replace('50.00', '-'))
        self.assertTrue(native.collect(directory, self.path, False)[2])

    def test_performance_flat_preserves_sign_exponent_and_checks_failed_count(self):
        directory = self.path / 'raw/native'
        self.csv(directory, True)
        rows, metrics, errors = native.collect(directory, self.path, True)
        self.assertFalse(errors)
        self.assertEqual(rows[0]['TTFT_Average'], '1000')
        self.assertEqual(rows[0]['TTFT_Min'], '-1')
        self.assertEqual(metrics[0]['unit'], 'ms')
        self.csv(directory, True, failed=True)
        self.assertTrue(native.collect(directory, self.path, True)[2])

    def test_native_pipeline_result_and_failure_keep_artifacts(self):
        for code in (0, 3):
            out = self.path / str(code)
            def fake_run(execution, directory, command, cwd):
                self.csv(directory)
                (directory / 'aisbench.log').write_text('original log', encoding='utf-8')
                return code
            env = dict(AG_OUTPUT_DIR=str(out), AG_SERVED_URL='http://x/v1', AG_SERVED_MODEL_NAME='m', AG_MODEL_PATH=str(self.path), AG_TASK_ID='t-test')
            with patch.dict(os.environ, env), patch.object(runtime.Execution, 'run_command', fake_run):
                status = runtime.run_main('aisbench-acc', runner.parser, lambda e, a: native.work(e, a, False), ['--datasets=gsm8k_gen', '--extra_info={"tag":"migration"}'])
            result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 1 if code else 0)
            self.assertEqual(result['status'], 'failed' if code else 'succeeded')
            self.assertEqual(len(result['metrics']), 0 if code else 2)
            records = json.loads((out / 'legacy_results.json').read_text())
            self.assertEqual(records[0]['build'], 't-test')
            self.assertEqual(records[0]['extra_info'], dict(tag='migration'))
            self.assertTrue((out / 'accuracy.csv').is_file())
            import yaml, jsonschema
            schemas = yaml.safe_load((ROOT.parents[1] / 'server/api/openapi.yaml').read_text(encoding='utf-8'))
            jsonschema.validate(result, {'$ref': '#/components/schemas/Result', 'components': schemas['components']})

    def test_subprocess_exit_is_not_hidden_by_log_collection(self):
        out = self.path / 'process'
        with patch.dict(os.environ, AG_OUTPUT_DIR=str(out)):
            execution = runtime.Execution('test')
            execution.prepare()
            directory = out / 'raw/native'
            directory.mkdir()
            code = execution.run_command(directory, [sys.executable, '-c', 'print("child failure"); raise SystemExit(7)'], self.path)
        self.assertEqual(code, 7)
        self.assertIn('child failure', (directory / 'aisbench.log').read_text())
        self.assertEqual(json.loads((directory / 'execution.json').read_text())['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
