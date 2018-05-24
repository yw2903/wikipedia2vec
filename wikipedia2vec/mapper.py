from wikipedia2vec import Wikipedia2Vec
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Mapper:
    def __init__(self, W_syn0, W_syn1):
        self.W_syn0 = W_syn0
        self.W_syn1 = W_syn1

    def map(self, wiki2vec):
        mapped_syn0 = wiki2vec.syn0.dot(self.W_syn0)
        mapped_syn1 = wiki2vec.syn1.dot(self.W_syn1)

        mapped_wiki2vec = Wikipedia2Vec(wiki2vec.dictionary)
        mapped_wiki2vec.syn0 = mapped_syn0
        mapped_wiki2vec.syn1 = mapped_syn1

        return mapped_wiki2vec
    
    def save(self, path):
        joblib.dump(dict(W_syn0=self.W_syn0, W_syn1=self.W_syn1), path)

    @staticmethod
    def load(path):
        d = joblib.load(path)
        return Map(**d)

    @staticmethod
    def train(src_wiki2vec, trg_wiki2vec, interlink):
        '''
            pairs: [(Item, Item), ...]
        '''
        src_indexes, trg_indexes = zip(*((src_item.index, trg_item.index) 
            for src_item, trg_item in interlink))

        src_indexes, trg_indexes = list(src_indexes), list(trg_indexes)

        src_syn0 = src_wiki2vec.syn0[src_indexes]
        src_syn1 = src_wiki2vec.syn1[src_indexes]

        trg_syn0 = trg_wiki2vec.syn0[trg_indexes]
        trg_syn1 = trg_wiki2vec.syn1[trg_indexes]

        logger.info('Compute SVD for syn0')
        u, s, vt = np.linalg.svd(trg_syn0.T.dot(src_syn0))
        W_syn0 = vt.T.dot(u.T)

        logger.info('Compute SVD for syn1')
        u, s, vt = np.linalg.svd(trg_syn1.T.dot(src_syn1))
        W_syn1 = vt.T.dot(u.T)

        return Mapper(W_syn0, W_syn1)


