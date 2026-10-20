import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from memory.match import MemoryMatcher
from memory.pipeline import add_memory
from memory.storage import load_memories, save_memories

class Models:
    def encode(self, text):
        return np.array([1., 0.]) if isinstance(text, str) else np.array([[1., 0.]] * len(text))
    def chat(self, messages, **kwargs):
        return '{"topics": ["project"]}'

class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
    def chat(self, messages, **kwargs):
        self.calls += 1
        return json.dumps({"related_groups": next(self.responses)})

def related(*ids):
    return [{"group_id": gid, "reason": "same concrete project"} for gid in ids]

class GroupTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.groups = {}, {}
        self.models = Models()
        for i in range(12):
            self.add(str(i), MemoryMatcher(establish_topics=True))
    def add(self, text, matcher):
        return add_memory(text, self.nodes, self.groups, self.models, self.models, matcher)
    def test_all_groups_and_independent_chains(self):
        matcher = MemoryMatcher(Client([related('g1', 'g12'), related('g12')]))
        node, decision = self.add('shared update', matcher)
        self.assertEqual(node.group_ids, ['g1', 'g12'])
        self.assertEqual(len(matcher.last_candidates), 10)
        self.assertEqual(len(matcher.last_group_candidates), 12)
        missed = next(g for g in matcher.last_group_candidates if g['group_id'] == 'g12')
        self.assertEqual(missed['top10_memories'], [])
        self.assertEqual(missed['initial_embedding_score'], 1.)
        self.add('only second group', matcher)
        self.assertEqual(self.groups['g1'].links['m13']['next_id'], None)
        self.assertEqual(self.groups['g12'].links['m13']['next_id'], 'm14')
        self.assertEqual(self.groups['g1'].links['m13']['prev_id'], 'm1')
        self.assertEqual(self.groups['g12'].links['m13']['prev_id'], 'm12')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.json'
            save_memories(path, self.nodes, self.groups)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertNotIn('reason', path.read_text(encoding='utf-8'))
            self.assertNotIn('prev_id', saved['nodes']['m13'])
            nodes, groups = load_memories(path, self.models)
            self.assertEqual(nodes['m13'].group_ids, ['g1', 'g12'])
            self.assertEqual(groups['g12'].links, self.groups['g12'].links)
    def test_empty_retries_and_failure_does_not_mutate(self):
        client = Client([[], []])
        with self.assertRaises(ValueError):
            self.add('unmatched', MemoryMatcher(client))
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(self.nodes), 12)
        self.assertTrue(all(group.memory_count == 1 for group in self.groups.values()))
        self.add('retry success', MemoryMatcher(Client([[], related('g11')])))
        self.assertEqual(self.nodes['m13'].group_ids, ['g11'])
    def test_invalid_group_rejected(self):
        for result in (related('missing'), related('g1', 'g1')):
            with self.assertRaises(ValueError):
                self.add('invalid', MemoryMatcher(Client([result])))
        self.assertEqual(len(self.nodes), 12)
    def test_legacy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.json'
            path.write_text(json.dumps({
                'nodes': {'m1': {'memory_id': 'm1', 'group_id': 'g1', 'topics': [],
                    'raw_text': 'old', 'created_at': 'time', 'prev_id': None, 'next_id': None}},
                'groups': {'g1': {'group_id': 'g1', 'head_id': 'm1', 'tail_id': 'm1', 'updated_at': 'time'}}}), encoding='utf-8')
            nodes, groups = load_memories(path, self.models)
            self.assertEqual(nodes['m1'].group_ids, ['g1'])
            self.assertEqual(groups['g1'].links['m1'], {'prev_id': None, 'next_id': None})

if __name__ == '__main__':
    unittest.main()
