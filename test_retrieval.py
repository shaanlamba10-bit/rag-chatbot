import unittest
from retrieval import chunk_text, retrieve

class RetrievalTests(unittest.TestCase):
    def test_chunk_overlap(self):
        chunks = chunk_text(' '.join(map(str, range(12))), 'source', size=5, stride=4)
        self.assertEqual(chunks[0], ('0 1 2 3 4', 'source'))
        self.assertTrue(chunks[1][0].startswith('4 '))

    def test_relevant_source(self):
        passages = [('PostgreSQL stores relational data.', 'database'), ('Roses bloom in spring.', 'garden')]
        self.assertEqual(retrieve('PostgreSQL', passages)[0][1], 'database')
        self.assertEqual(retrieve('satellite', passages), [])

    def test_empty(self):
        self.assertEqual(retrieve('question', []), [])
        self.assertEqual(retrieve('question', [('...', 'empty')]), [])

if __name__ == '__main__':
    unittest.main()
