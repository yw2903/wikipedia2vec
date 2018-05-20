from .dictionary cimport Dictionary
import numpy as np
import bz2
from urllib.parse import urlparse
import joblib
import logging

logger = logging.getLogger(__name__)

cdef class Interlink:
    def __init__(self, links):
        self.links = links
    
    def __len__(self):
        return len(self.links)

    def get_trg_embs(self, trg_model):
        for src_item, trg_item in self.links:
            trg_syn0 = trg_model.syn0[trg_item.index]
            trg_syn1 = trg_model.syn1[trg_item.index]
            yield src_item, (trg_syn0, trg_syn1)

    def save(self, path):
        joblib.dump(self.links, path)

    @staticmethod
    def load(path):
        return Interlink(joblib.load(path))


    @staticmethod
    def build(str path, Dictionary src_dic, Dictionary trg_dic, 
            int n_test=1000,
            trg_netloc='dbpedia.org'):
        with bz2.open(path, 'rt') as f:
            links = []
            count = 0

            for line in f:
                line = line.strip()

                if line.startswith('#'):
                    continue

                src, _, trg, _ = line.split()

                src, trg = urlparse(src[1:-1]), urlparse(trg[1:-1])

                if trg.netloc == trg_netloc:
                    src = src_dic.get_entity(src.path.split('/')[-1])
                    trg = trg_dic.get_entity(trg.path.split('/')[-1])

                    if src is None or trg is None:
                        continue

                    links.append((src, trg))
                    count += 1

                    if count % 1000 == 0:
                        logger.info('Progress={}'.format(count))
                        print(count)

        shuf = list(np.random.permutation(links))

        test_interlink = Interlink(shuf[:n_test])
        train_interlink = Interlink(shuf[n_test:])

        return train_interlink, test_interlink
