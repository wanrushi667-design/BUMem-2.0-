import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_memory_pipeline as runner


class FakeModels:
    def get_qwen(self):
        return self

    def get_embedding(self):
        return self

    def get_deepseek(self):
        return self

    def encode(self, text):
        return np.array([1.0, 0.0]) if isinstance(text, str) else np.array([[1.0, 0.0]] * len(text))

    def chat(self, messages, **kwargs):
        if messages[1]['content'].startswith('{'):
            return '{"related_groups":[{"group_id":"g1","reason":"same trip"}]}'
        return '{"topics":["day_trip"]}'


class GroupingDiagnosticsTests(unittest.TestCase):
    def test_append_snapshots_and_distinguish_selected_node_from_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialogues = [
                {"period": period, "role": "user", "sequence": i, "content": text}
                for i, (period, text) in enumerate([
                    ('period_00', 'carpool'),
                    ('period_01', 'bus trip'),
                    ('period_01', 'bus snacks'),
                ], 1)
            ]
            (root / 'user_dialogues.json').write_text(json.dumps(dialogues), encoding='utf-8')
            with patch.object(runner, 'ModelManager', return_value=FakeModels()), \
                 patch.object(runner, 'load_benchmark'), patch.object(runner, 'run_evaluation'):
                runner.run_period('period_00', root)
                (root / 'overall_results_strict.json').write_text('[[{}]]', encoding='utf-8')
                runner.run_period('period_01', root)
            rows = [json.loads(line) for line in (root / 'grouping_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['candidates'], [])
            self.assertIsNone(rows[0]['raw_response'])
            self.assertNotEqual(rows[0]['run_id'], rows[1]['run_id'])
            self.assertEqual(rows[1]['run_id'], rows[2]['run_id'])
            last = rows[2]
            self.assertEqual(last['new_memory']['group_ids'], ['g1'])
            self.assertEqual(last['final_groups'][0]['links']['m3']['prev_id'], 'm2')
            self.assertEqual(last['final_groups'][0]['tail_id'], 'm3')
            self.assertEqual(last['candidates'][0]['group_ids'], ['g1'])
            self.assertEqual(json.loads(last['raw_response'])['related_groups'], last['decision']['related_groups'])
            payload = json.loads(last['messages'][1]['content'])
            self.assertEqual(payload['new_memory']['raw_text'], 'bus snacks')
            self.assertEqual(len(payload['candidate_groups']), 1)


if __name__ == '__main__':
    unittest.main()
