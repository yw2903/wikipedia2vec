import numpy as np
import bz2
from urllib.parse import urlparse
import joblib
import logging

logger = logging.getLogger(__name__)

class Interlink:
    def __init__(self, links, invert=False):
        self.links = links
        self.invert = invert
    
    def __len__(self):
        return len(self.links)

    def __iter__(self):
        for a, b in self.links:
            if self.invert:
                yield b, a
            else:
                yield a, b

    def get_trg_embs(self, trg_model):
        for trg_item, src_item in self.links:
            trg_syn0 = trg_model.syn0[trg_item.index]
            trg_syn1 = trg_model.syn1[trg_item.index]
            yield src_item, (trg_syn0, trg_syn1)

    def save(self, path):
        joblib.dump(self.links, path)

    @staticmethod
    def load(path, invert=False):
        return Interlink(joblib.load(path), invert)

    @staticmethod
    def load_text(path, src_dic, trg_dic, sep=' ', item_type='word', invert=False):
        with open(path) as f:
            links = []
            for line in f:
                src, trg = line.strip().lower().split(sep)

                if item_type == 'word':
                    src = src_dic.get_word(src)
                    trg = trg_dic.get_word(trg)
                else:
                    src = src_dic.get_entity(src)
                    trg = trg_dic.get_entity(trg)
                
                if src and trg:
                    links.append((src, trg))

        return Interlink(links, invert)




