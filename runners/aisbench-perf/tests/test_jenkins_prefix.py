import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runner
from runtime import Failure


class JenkinsPrefixTests(unittest.TestCase):
    def args(self, cases, *extra):
        return runner.parser().parse_args(['--url=http://x/v1', '--model_name=m', '--model_path=/model', '--test_cases=' + cases, *extra])

    def test_six_columns_map_to_old_warmup_and_denominator(self):
        args = runner.legacy_cases(self.args('[[256,32,16,50,2,1],[512,128,8,90,4,2]]'))
        rounds = json.loads(args.rounds)
        self.assertEqual(args.metrics_profile, 'vllm_prompt')
        self.assertEqual(rounds[0], dict(input_len=256, output_len=32, data_num=16, repeat_rate='50%', concurrency=2, seed=1, prefix_mode='private', warmup_concurrency=2, warmup_output_len=3))
        self.assertEqual(rounds[1]['warmup_output_len'], 12)
        self.assertEqual(runner.legacy_cases(self.args('[[256,32,16,50,2,1]]', '--metrics_profile=vllm_prefix')).metrics_profile, 'vllm_prefix')

    def test_bad_cases_and_conflicting_inputs(self):
        for cases in ('[]', '{}', '[[1,2]]', '[[256,32,16,101,2,1]]', '[[256,32,16,50,true,1]]', '[[256.5,32,16,50,2,1]]'):
            with self.subTest(cases=cases), self.assertRaises(Failure):
                runner.legacy_cases(self.args(cases))
        with self.assertRaises(Failure):
            runner.legacy_cases(self.args('[[256,32,16,50,2,1]]', '--concurrency=10'))

    def test_validate_every_case_before_sending_requests(self):
        with patch.object(runner, 'work_round') as run:
            with self.assertRaises(Failure):
                runner.work(None, self.args('[[256,32,16,50,2,1],[512,32,16,50,0,2]]'))
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
