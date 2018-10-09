import numpy as np
import logging
import collections
from collections import defaultdict
from wikipedia2vec import Wikipedia2Vec
from utils import *
import cupy

logger = logging.getLogger(__name__)

def dropout_matrix(m, p):
    xp = array_module(m)
    if p <= 0.0:
        return m
    else:
        mask = xp.random.rand(*m.shape) >= p
        return m*mask

def topk_mean(m, k, inplace=True):  
    xp = array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

class Parameters:
    def __init__(self, fix_ent=True, normalize=['unit', 'mean_center', 'unit'], 
            init_dropout=0.9, dropout_decay=2.0,
            init_n_word=4000, init_n_ent=4000, 
            n_word=10000, n_ent=10000,
            threshold=1e-6, interval=50,
            csls=10, reweight=0.5, batchsize=10000, langlink=None,
            dev=None):
        self.fix_ent = fix_ent
        self.normalize = normalize
        self.init_dropout = init_dropout
        self.dropout_decay = dropout_decay
        self.init_n_word = init_n_word
        self.init_n_ent = init_n_ent
        self.n_word = n_word
        self.n_ent = n_ent
        self.threshold = threshold
        self.interval = interval
        self.csls = csls
        self.reweight = reweight
        self.batchsize = batchsize
        self.langlink = langlink
        self.dev = dev 

class Mapper:
    def __init__(self, src_wiki, trg_wiki, params):
        self.params = params
        self.src_wiki = src_wiki
        self.trg_wiki = trg_wiki

        self.dim = src_wiki.syn0.shape[1]

        # They should be in GPU
        self.src_word2id, self.src_word_emb = get_word_emb(src_wiki, params.n_word)
        self.trg_word2id, self.trg_word_emb = get_word_emb(trg_wiki, params.n_word)
        self.src_ent2id, self.src_ent_emb = get_entity_emb(src_wiki, params.n_ent)
        self.trg_ent2id, self.trg_ent_emb = get_entity_emb(trg_wiki, params.n_ent)

        self.src_n_word = self.src_word_emb.shape[0]
        self.trg_n_word = self.trg_word_emb.shape[0]

        xp = self.xp = array_module(self.src_word_emb)
        self.src_emb = xp.concatenate((self.src_word_emb, self.src_ent_emb), axis=0)
        self.trg_emb = xp.concatenate((self.trg_word_emb, self.trg_ent_emb), axis=0)

    def _normalize(self, emb, normalization=['unit', 'mean_center', 'unit']):
        for n in normalization:
            normalize(emb, n)

    def initialize(self):
        xp = self.xp
        # First normalize
        #   RQ: normalizationはwordとentityに同時にかけるべき？
        self._normalize(self.src_emb)
        self._normalize(self.trg_emb)
        self.src_emb_mapped = xp.copy(self.src_emb)
        self.trg_emb_mapped = xp.copy(self.trg_emb)

        # Initialize
        if self.params.langlink:
            self.init_ent_langlink(self.params.langlink)
            self.src_word_ind = xp.array([], dtype=xp.int32)
            self.trg_word_ind = xp.array([], dtype=xp.int32)
        else:
            self.src_ent_ind, self.trg_ent_ind = self.init_unsup(
                    self.src_emb_mapped[self.src_n_word:], self.trg_emb_mapped[self.trg_n_word:], 
                    self.params.init_n_ent, self.params.csls)

            self.src_word_ind, self.trg_word_ind = self.init_unsup(
                    self.src_emb_mapped[:self.src_n_word], self.trg_emb_mapped[:self.trg_n_word],
                    self.params.init_n_word, self.params.csls)

        # Load dev dict
        if self.params.dev:
            self.dev = defaultdict(set)
            with open(self.params.dev) as f:
                for line in f:
                    src_word, trg_word = line.strip().lower().split()
                    src_word = self.src_wiki.get_word(src_word)
                    trg_word = self.trg_wiki.get_word(trg_word)
                    
                    if not src_word or not trg_word:
                        continue

                    if src_word.index in self.src_word2id and trg_word.index in self.trg_word2id:
                        src_ind = self.src_word2id[src_word.index]
                        trg_ind = self.trg_word2id[trg_word.index]
                        self.dev[src_ind].add(trg_ind)
        else:
            self.dev = None

    def init_ent_langlink(self, path):
        xp = self.xp
        src_ent_ind = []
        trg_ent_ind = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                src_title, trg_title = line.split()

                src_ent = self.src_wiki.dictionary.get_entity(src_title)
                trg_ent = self.trg_wiki.dictionary.get_entity(trg_title)

                if not src_ent or not trg_ent:
                    continue

                if src_ent.index in self.src_ent2id and trg_ent.index in self.trg_ent2id:
                    src_ent_ind.append(self.src_ent2id[src_ent.index])
                    trg_ent_ind.append(self.trg_ent2id[trg_ent.index])

        self.src_ent_ind = xp.array(src_ent_ind, dtype=xp.int32)
        self.trg_ent_ind = xp.array(trg_ent_ind, dtype=xp.int32)

    def init_unsup(self, x, z, n_vocab, csls_neighbor=10):
        xp = self.xp
        sim_size = min(x.shape[0], z.shape[0])
        if n_vocab:
            sim_size = min(sim_size, n_vocab)

        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        self._normalize(xsim)
        self._normalize(zsim)
        sim = xsim.dot(zsim.T)

        knn_sim_fwd = topk_mean(sim, k=csls_neighbor)
        knn_sim_bwd = topk_mean(sim.T, k=csls_neighbor)
        sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2

        src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
        trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim

        return src_indices, trg_indices

    def orthogonal_map(self):
        xp = self.xp
        src_indices = xp.concatenate((self.src_word_ind, self.src_ent_ind + self.src_n_word))
        trg_indices = xp.concatenate((self.trg_word_ind, self.trg_ent_ind + self.src_n_word))

        src_ind_emb = self.src_emb[src_indices]
        trg_ind_emb = self.trg_emb[trg_indices]

        u, s, vt = xp.linalg.svd(trg_ind_emb.T.dot(src_ind_emb))
        w = vt.T.dot(u.T)
        self.src_emb.dot(w, out=self.src_emb_mapped)
        self.trg_emb_mapped[:] = self.trg_emb

    def advanced_map(self, reweight=0.5):
        # ToDo: この手法は最後だけ
        xp = self.xp
        src_indices = xp.concatenate((self.src_word_ind, self.src_ent_ind + self.src_n_word))
        trg_indices = xp.concatenate((self.trg_word_ind, self.trg_ent_ind + self.src_n_word))

        src_ind_emb = self.src_emb[src_indices]
        trg_ind_emb = self.trg_emb[trg_indices]

        # STEP1: Whitening
        def whitening_transformation(m):
            u, s, vt = xp.linalg.svd(m, full_matrices=False)
            return vt.T.dot(xp.diag(1/s)).dot(vt)

        self.src_whitener = whitening_transformation(src_ind_emb)
        self.trg_whitener = whitening_transformation(trg_ind_emb)
        self.src_emb_mapped = self.src_emb.dot(self.src_whitener)
        self.trg_emb_mapped = self.trg_emb.dot(self.trg_whitener)

        # STEP2: Orthogonal Mapping
        self.src_map, self.s, trg_map_t = xp.linalg.svd(src_ind_emb.T.dot(trg_ind_emb))
        self.trg_map = trg_map_t.T
        self.src_emb_mapped = self.src_emb_mapped.dot(self.src_map)
        self.trg_emb_mapped = self.trg_emb_mapped.dot(self.trg_map)

        # STEP3: Re-weightening
        self.src_emb_mapped *= self.s**0.5
        self.trg_emb_mapped *= self.s**0.5

        # STEP4: De-whitening
        src_dewhitener = self.src_map.T.dot(xp.linalg.inv(self.src_whitener)).dot(self.src_map)
        trg_dewhitener = self.trg_map.T.dot(xp.linalg.inv(self.trg_whitener)).dot(self.trg_map)
        self.src_emb_mapped = self.src_emb_mapped.dot(src_dewhitener)
        self.trg_emb_mapped = self.trg_emb_mapped.dot(trg_dewhitener)

    def induce_dic(self, src_emb, trg_emb, dropout=0.0):
        xp = self.xp
        src_size = src_emb.shape[0]
        trg_size = trg_emb.shape[0]

        # Allocation Memory
        sim_src2trg = xp.empty((self.params.batchsize, trg_size), dtype=xp.float32)
        sim_trg2src = xp.empty((self.params.batchsize, src_size), dtype=xp.float32)
        knn_sim_src2trg = xp.zeros(src_size, dtype=xp.float32)
        knn_sim_trg2src = xp.zeros(trg_size, dtype=xp.float32)
        best_sim_src2trg = xp.full(src_size, -100, dtype=xp.float32)
        best_sim_trg2src = xp.full(trg_size, -100, dtype=xp.float32)
        trg_indices_forward = xp.zeros(src_size, dtype=int)
        trg_indices_backward = xp.arange(trg_size)
        src_indices_forward = xp.arange(src_size)
        src_indices_backward = xp.zeros(trg_size, dtype=int)


        # Compute k-NN mean of cosine similarity of backward for forward CSLS        
        for b_start in range(0, trg_size, self.params.batchsize):
            b_end = min(b_start+self.params.batchsize, trg_size)
            _batchsize = b_end - b_start
            trg_emb[b_start:b_end].dot(src_emb[:src_size].T, out=sim_trg2src[:_batchsize])
            knn_sim_trg2src[b_start:b_end] = topk_mean(sim_trg2src[:_batchsize], k=self.params.csls)

        # Compute CSLS for forward
        # sim_src2trg[0] == self.params.batchsize
        for b_start in range(0, src_size, sim_src2trg.shape[0]):
            b_end = min(b_start+self.params.batchsize, src_size)
            _batchsize = b_end - b_start
            sim_src2trg[:_batchsize] = src_emb[b_start:b_end].dot(trg_emb[:trg_size].T)
            sim_src2trg[:_batchsize].max(axis=1, out=best_sim_src2trg[b_start:b_end])
            sim_src2trg[:_batchsize] -= knn_sim_trg2src/2  # Equivalent to the real CSLS scores for NN
            dropout_matrix(sim_src2trg[:_batchsize], dropout).argmax(axis=1, out=trg_indices_forward[b_start:b_end])

        # Compute k-NN mean of cosine similarity of forward for backward CSLS        
        for b_start in range(0, src_size, self.params.batchsize):
            b_end = min(b_start+self.params.batchsize, src_size)
            _batchsize = b_end - b_start
            src_emb[b_start:b_end].dot(trg_emb[:src_size].T, out=sim_src2trg[:_batchsize])
            knn_sim_src2trg[b_start:b_end] = topk_mean(sim_src2trg[:_batchsize], k=self.params.csls)

        # Compute CSLS for forward
        # sim_src2trg[0] == self.params.batchsize
        for b_start in range(0, trg_size, sim_trg2src.shape[0]):
            b_end = min(b_start+self.params.batchsize, trg_size)
            _batchsize = b_end - b_start
            trg_emb[b_start:b_end].dot(src_emb[:src_size].T, out=sim_trg2src[:_batchsize])
            sim_trg2src[:_batchsize].max(axis=1, out=best_sim_trg2src[b_start:b_end])
            sim_trg2src[:_batchsize] -= knn_sim_src2trg/2  # Equivalent to the real CSLS scores for NN
            dropout_matrix(sim_trg2src[:_batchsize], dropout).argmax(axis=1, out=src_indices_backward[b_start:b_end])

        src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
        trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

        # Objective function evaluation
        fwd_mean = xp.mean(best_sim_src2trg)
        bwd_mean = xp.mean(best_sim_trg2src)
        objective = (fwd_mean + bwd_mean) / 2

        return src_indices, trg_indices, objective

    def train(self):
        best_obj = obj = -100
        it = 1
        last_improvement = 0
        dropout = self.params.init_dropout
        while True:
            # Did we improve enough in the last iteration?
            if it - last_improvement > self.params.interval:
                if dropout <= 0.0:
                    break
                dropout = 1-min(1.0, self.params.dropout_decay * (1-dropout))
                last_improvement = it

            # Learn Mapping
            self.orthogonal_map()

            # Induce word dictionary
            self.src_word_ind, self.trg_word_ind, word_obj = self.induce_dic(
                    self.src_emb_mapped[:self.params.n_word], 
                    self.trg_emb_mapped[:self.params.n_word],
                    dropout=dropout)

            # Induce entity dictionary
            if not self.params.fix_ent:
                self.src_ent_ind, self.trg_ent_ind, ent_obj = self.induce_dic(
                        self.src_emb_mapped[self.params.n_word:],
                        self.trg_emb_mapped[self.params.n_word:],
                        dropout=dropout)
                obj = (word_obj + ent_obj) / 2
            else:
                obj = word_obj

            # Compare objectives
            if obj - best_obj >= self.params.threshold:
                last_improvement = it
                best_obj = obj

            if self.dev:
                logger.info("iter={}; dropout={:.3f}; obj={:.3f}; dev={:.3f}".format(it, dropout, float(obj), self.validate()))
            else:
                logger.info("iter={}; dropout={:.3f}; obj={:.3f}".format(it, dropout, float(obj)))

            it += 1

        self.advanced_map()

        # ToDo: Embedding全体にadvanced mapを掛けてwikipedia2vecの形にして返す

    def map_wiki(self, wiki, whitener, w):
        emb = wiki.syn0
        self._normalize(emb)

        # Whiten
        emb.dot(self.xp.asnumpy(whitener), out=emb)

        # Orthogonal Map
        emb.dot(self.xp.asnumpy(w), out=emb)

        # Reweighting
        emb *= self.s**0.5

        # De-whiten
        dewhitener = w.T.dot(xp.linalg.inv(whitener)).dot(w)
        emb.dot(self.xp.asnumpy(dewhitener), out=emb)

        wiki.syn0 = emb 

        return wiki.syn0

    def map_src(self, wiki):
        return self.map_wiki(wiki, self.src_whitener, self.src_map)

    def map_trg(self, wiki):
        return self.map_wiki(wiki, self.trg_whitener, self.trg_map)

    def validate(self):
        xp = self.xp
        correct = 0
        N = 0
        for src_id, trg_ids in self.dev.items():
            N += 1
            src_vec = self.src_emb_mapped[src_id]
            pred_id = xp.argmax(self.trg_emb_mapped.dot(src_vec))
            if int(pred_id) in trg_ids:
                correct += 1

        return correct / N


    def set_word_validation(self, path, n_word):
        oov = set()
        vocab = set()
        self.word_gold = collections.defaultdict(set)

        with open(path) as f:
            for line in f:
                src, trg = line.strip().split()
                try:
                    src = self.src.dic.get_word(src)
                    trg = self.trg.dic.get_word(trg)

                    if src is None or trg is None:
                        raise KeyError()

                    src_ind = self.src.word2id[src.index]
                    trg_ind = self.trg.word2id[trg.index]
                    if src_ind >= n_word or trg_ind >= n_word:
                        raise KeyError()
                    self.word_gold[src_ind].add(trg_ind)
                    vocab.add(src)
                except KeyError:
                    oov.add(src)

        oov -= vocab
        self.val_n_word = n_word
        return len(self.word_gold) / (len(self.word_gold) + len(oov))
    
    def validate_word(self):
        return self.validate(
                self.src_word_mapped[:self.val_n_word], 
                self.trg_word_mapped[:self.val_n_word], 
                self.word_gold)

