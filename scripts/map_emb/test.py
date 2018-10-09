import unittest
from wikipedia2vec import Wikipedia2Vec
from mapper import *
from utils import *
import cupy
import numpy as np
from pathlib import Path

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
data_path = Path('/home/jinjin/data/pretrained')

class TestMapperUtils(unittest.TestCase):
    def test_get_word_emb(self):
        wiki = Wikipedia2Vec.load(data_path / 'enwiki_20180420_300d.pkl')

        word2id, emb = get_word_emb(wiki, 100)

        self.assertEqual(len(word2id), 100)
        self.assertEqual(emb.shape, (100, 300))

    def test_get_entity_emb(self):
        wiki = Wikipedia2Vec.load(data_path / 'enwiki_20180420_300d.pkl')

        ent2id, emb = get_entity_emb(wiki, 100)

        self.assertEqual(len(ent2id), 100)
        self.assertEqual(emb.shape, (100, 300))

    def test_array_module(self):
        x = cupy.array([1, 2, 3], dtype=np.float32)
        self.assertEqual(array_module(x), cupy)
        
        x = np.array([1, 2, 3], dtype=np.float32)
        self.assertEqual(array_module(x), np)

    def test_normalize(self):
        wiki = Wikipedia2Vec.load(data_path / 'enwiki_20180420_300d.pkl')
        word2id, emb = get_word_emb(wiki, 100)

        normalize(emb, 'unit')
        normalize(emb, 'mean_center')
        normalize(emb, 'unit')
        
class TestMapper(unittest.TestCase):
    def test_mapper(self):
        src_wiki = Wikipedia2Vec.load(data_path / 'enwiki_20180420_300d.pkl')
        trg_wiki = Wikipedia2Vec.load(data_path / 'jawiki_20180420_300d.pkl')

        params = Parameters(langlink='/home/jinjin/data/crosslingual/en-es/langlink.txt')
        mapper = Mapper(src_wiki, trg_wiki, params)

        self.assertIsInstance(mapper.src_word_emb, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_word_emb, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.src_ent_emb, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_ent_emb, cupy.core.core.ndarray)

        mapper.initialize()

        self.assertIsInstance(mapper.src_word_ind, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_word_ind, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.src_ent_ind, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_ent_ind, cupy.core.core.ndarray)
        self.assertEqual(
                mapper.src_emb.dot(mapper.src_emb.T).shape, 
                (mapper.src_emb.shape[0], mapper.src_emb.shape[0]))

        for i in range(10):
            mapper.orthogonal_map()

            self.src_word_ind, self.trg_word_ind, obj = mapper.induce_dic(
                    mapper.src_emb_mapped[:mapper.src_n_word], 
                    mapper.trg_emb_mapped[:mapper.src_n_word], dropout=0.5)
            print("FIRST", obj)


        mapper.orthogonal_map()

        src_word_ind, trg_word_ind, obj = mapper.induce_dic(
                mapper.src_emb_mapped[:mapper.src_n_word],
                mapper.trg_emb_mapped[:mapper.trg_n_word],
                dropout=0.5)
        print("SECOND", obj)

        self.assertNotEqual(self.src_word_ind.tolist(), src_word_ind.tolist())
        self.assertNotEqual(self.trg_word_ind.tolist(), trg_word_ind.tolist())

        mapper.advanced_map()


        src_ind, trg_ind, obj = mapper.induce_dic(mapper.src_emb_mapped[:mapper.src_n_word], 
                mapper.trg_emb_mapped[:mapper.src_n_word], dropout=0.5)
        print(obj)

    def xtest_train(self):
        src_wiki = Wikipedia2Vec.load(data_path / 'enwiki_20180420_300d.pkl')
        trg_wiki = Wikipedia2Vec.load(data_path / 'jawiki_20180420_300d.pkl')

        params = Parameters(langlink='/home/jinjin/data/crosslingual/en-es/langlink.txt')
        mapper = Mapper(src_wiki, trg_wiki, params)

        self.assertIsInstance(mapper.src_word_emb, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_word_emb, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.src_ent_emb, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_ent_emb, cupy.core.core.ndarray)

        mapper.initialize()

        self.assertIsInstance(mapper.src_word_ind, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_word_ind, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.src_ent_ind, cupy.core.core.ndarray)
        self.assertIsInstance(mapper.trg_ent_ind, cupy.core.core.ndarray)
        self.assertEqual(
                mapper.src_emb.dot(mapper.src_emb.T).shape, 
                (mapper.src_emb.shape[0], mapper.src_emb.shape[0]))

        mapper.train()



if __name__ == '__main__':
    unittest.main()
