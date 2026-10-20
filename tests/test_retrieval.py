import json
import unittest
from memory.node_group import MemoryNode
from memory.connect import create_group, connect_memory
from retrieval.retriever import retrieve_memories
from retrieval.reply import select_answer

class Client:
    def __init__(self, result):
        self.result = result
        self.payload = None
    def chat(self, messages, **kwargs):
        self.payload = json.loads(messages[1]['content'])
        return json.dumps(self.result)

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.groups = {}, {}
        self.query = {'original_query': 'project and schedule', 'topics': ['project'],
                      'target_aspect': ['schedule'], 'embedding': [1., 0.]}
        for i in range(1, 4):
            create_group(self.node(i, [0., 1.], 0), f'g{i}', self.nodes, self.groups)
        connect_memory(self.node(10, [1., 0.], 2), ['g1', 'g2'], self.nodes, self.groups)
        connect_memory(self.node(11, [0., 1.], 1), ['g1'], self.nodes, self.groups)
        connect_memory(self.node(12, [0., 1.], 1), ['g2'], self.nodes, self.groups)
        connect_memory(self.node(13, [.9, .1], 3), ['g3'], self.nodes, self.groups)
    def node(self, number, embedding, period):
        return MemoryNode(f'm{number}', [], ['project'], f'original {number}',
                          'same timestamp', embedding, period)
    def test_multi_group_dedup_chronology_and_topk_rescue(self):
        client = Client({'selected_groups': ['g1', 'g2']})
        result = retrieve_memories(self.query, self.nodes, self.groups, client, top_k=2)
        self.assertEqual([m.memory_id for m in result], ['m1', 'm2', 'm11', 'm12', 'm10', 'm13'])
        self.assertEqual([g['group_id'] for g in client.payload['candidate_groups']], ['g1', 'g2', 'g3'])
        texts = [m['raw_text'] for m in client.payload['memories']]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertIn('original 3', texts)  # P0 ????????????????????
        self.assertNotIn('m3', [m.memory_id for m in result])
        self.assertEqual(result[-1].group_ids, ['g3'])  # ????? TopK ????
    def test_empty_selection_retains_hits(self):
        result = retrieve_memories(self.query, self.nodes, self.groups,
                                   Client({'selected_groups': []}), top_k=2)
        self.assertEqual([m.memory_id for m in result], ['m10', 'm13'])
    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            retrieve_memories(self.query, self.nodes, self.groups,
                              Client({'selected_groups': ['unknown']}))
    def test_default_top10_and_numeric_order_without_period(self):
        nodes, groups = {}, {}
        for i in range(1, 13):
            node = self.node(i, [1., float(13-i)], None)
            create_group(node, f'g{i}', nodes, groups)
        client = Client({'selected_groups': []})
        result = retrieve_memories(self.query, nodes, groups, client)
        self.assertEqual([m.memory_id for m in result], [f'm{i}' for i in range(3, 13)])
        self.assertEqual(len(client.payload['candidate_groups']), 10)
    def test_reply_receives_sources_and_time(self):
        client = Client({'answer_id': '1', 'current_state': 'project',
                         'evidence_ids': ['m10'], 'answer_status': 'supported'})
        select_answer(self.query, [{'id': 1, 'answer': 'yes'}], [self.nodes['m10']], client)
        memory = client.payload['memories'][0]
        self.assertEqual(memory['group_ids'], ['g1', 'g2'])
        self.assertEqual(memory['period_index'], 2)
        self.assertEqual(memory['created_at'], 'same timestamp')
        self.assertNotIn('embedding', memory)

if __name__ == '__main__':
    unittest.main()
