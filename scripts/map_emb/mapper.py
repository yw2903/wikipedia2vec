import numpy as np
import logging
import collections
from wikipedia2vec import Wikipedia2Vec

logger = logging.getLogger(__name__)

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]


def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    avg = np.mean(matrix, axis=1)
    matrix -= avg[:, np.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)

def dropout_matrix(m, p):
    if p <= 0.0:
        return m
    else:
        mask = np.random.rand(*m.shape) >= p
        return m*mask

def topk_mean(m, k, inplace=True):  
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

class Parameters:
    def __init__(self, normalize, 
            init_dropout, dropout_decay,
            init_n_word, init_n_ent, 
            n_word, n_ent,
            threshold, interval,
            csls, reweight, batchsize):
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

class Lang:
    def __init__(self, wiki, normalize=['unit', 'center', 'unit']):
        self.wiki = wiki
        self.dic = wiki.dictionary
        self.emb = wiki.syn0
        normalize(self.emb, normalize)
        self.word_emb, self.word2id = self.get_item_emb(self.dic.words())
        self.ent_emb, self.ent2id = self.get_item_emb(self.dic.entities())

    def get_item_emb(self, items):
        items = sorted(items, key=lambda item: item.count, reverse=True)
        emb = np.empty((len(items), self.emb.shape[1]), dtype=np.float32)
        item2id = {}
        
        for i, item in enumerate(items):
            emb[i] = self.emb[item.index]
            item2id[item.index] = i

        return emb, item2id

    def __len__(self):
        return self.emb.shape[0]

class Mapper:
    def __init__(self, src_wiki, trg_wiki, params):
        self.src = Lang(src_wiki, params.normalize)
        self.trg = Lang(trg_wiki, params.normalize)

        self.dim = self.src.emb.shape[1]
        self.params = params

        self.word_gold = None

        # Induced Dictionary
        self.src_ent_ind = []
        self.trg_ent_ind = []
        self.src_word_ind = []
        self.trg_word_ind = []

    def init_ent_supervised(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                src_title, trg_title = line.split()

                src_ent = self.src.dic.get_entity(src_title)
                trg_ent = self.trg.dic.get_entity(trg_title)

                if src_ent and trg_ent:
                    self.src_ent_ind.append(self.src.ent2id[src_ent.index])
                    self.trg_ent_ind.append(self.trg.ent2id[trg_ent.index])

    def init_unsup(self, x, z, n_vocab, csls_neighbor=10):
        sim_size = min(x.shape[0], z.shape[0])
        if n_vocab:
            sim_size = min(sim_size, n_vocab)

        u, s, vt = np.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = np.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        normalize(xsim, self.params.normalize)
        normalize(zsim, self.params.normalize)
        sim = xsim.dot(zsim.T)

        knn_sim_fwd = topk_mean(sim, k=csls_neighbor)
        knn_sim_bwd = topk_mean(sim.T, k=csls_neighbor)
        sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2

        src_indices = np.concatenate((np.arange(sim_size), sim.argmax(axis=0)))
        trg_indices = np.concatenate((sim.argmax(axis=1), np.arange(sim_size)))
        del xsim, zsim, sim

        return list(src_indices), list(trg_indices)

    def init_word_unsup(self, n_word, csls_neighbor=10):
        src_ind, trg_ind = self.init_unsup(self.src.word_emb, self.trg.word_emb, 
                n_word, csls_neighbor)
        self.src_word_ind = src_ind
        self.trg_word_ind = trg_ind

    def init_ent_unsup(self, n_ent, csls_neighbor=10):
        src_ind, trg_ind = self.init_unsup(src.ent_emb, trg.ent_emb, n_ent, csls_neighbor)
        self.src_ent_ind = src_ind
        self.trg_ent_ind = trg_ind

    def learn_mapping(self, reweight=0.5):
        n_src_word = self.src.word_emb.shape[0]
        n_trg_word = self.trg.word_emb.shape[0]
        xw = np.concatenate((self.src.word_emb, self.src.ent_emb), axis=0)
        zw = np.concatenate((self.trg.word_emb, self.trg.ent_emb), axis=0)

        src_indices = self.src_word_ind + [idx+n_src_word for idx in self.src_ent_ind]
        trg_indices = self.trg_word_ind + [idx+n_trg_word for idx in self.trg_ent_ind]

        def whitening_transformation(m):
            u, s, vt = np.linalg.svd(m, full_matrices=False)
            return vt.T.dot(np.diag(1/s)).dot(vt)
        wx1 = whitening_transformation(xw[src_indices])
        wz1 = whitening_transformation(zw[trg_indices])
        xw = xw.dot(wx1)
        zw = zw.dot(wz1)

        wx2, s, wz2_t = np.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
        wz2 = wz2_t.T
        xw = xw.dot(wx2)
        zw = zw.dot(wz2)

        xw *= s**reweight
        zw *= s**reweight

        xw = xw.dot(wx2.T.dot(np.linalg.inv(wx1)).dot(wx2))
        zw = zw.dot(wz2.T.dot(np.linalg.inv(wz1)).dot(wz2))

        self.src_word_mapped = xw[:n_src_word]
        self.src_ent_mapped = xw[n_src_word:]
        self.trg_word_mapped = zw[:n_trg_word]
        self.trg_ent_mapped = zw[n_trg_word:]

    def induce_word_dic(self, n_word, 
            direction='union', csls_neighbor=10, batchsize=1000, dropout=0.0):
        src_ind, trg_ind, obj = self.induce_dic(
                self.src_word_mapped[:n_word], self.trg_word_mapped[:n_word],
                direction, csls_neighbor, batchsize, dropout)
        self.src_word_ind = src_ind
        self.trg_word_ind = trg_ind
        self.word_obj = obj

    def induce_ent_dic(self, n_ent, 
            direction='union', csls_neighbor=10, batchsize=1000, dropout=0.0):
        src_ind, trg_ind, obj = self.induce_dic(
                self.src_ent_mapped[:n_ent], self.trg_ent_mapped[:n_ent],
                direction, csls_neighbor, batchsize, dropout)
        self.src_ent_ind = src_ind
        self.trg_ent_ind = trg_ind
        self.ent_obj = obj

    def induce_dic(self, xw, zw, 
            direction='union', csls_neighbor=10, batchsize=10000, dropout=0.0):
        trg_size = zw.shape[0]
        src_size = xw.shape[0]
        simfwd = np.empty((batchsize, trg_size), dtype='float32')
        simbwd = np.empty((batchsize, src_size), dtype='float32')
        knn_sim_fwd = np.zeros(src_size, dtype='float32')
        knn_sim_bwd = np.zeros(trg_size, dtype='float32')
        best_sim_forward = np.full(src_size, -100, dtype='float32')
        best_sim_backward = np.full(trg_size, -100, dtype='float32')
        trg_indices_forward = np.zeros(src_size, dtype=int)
        trg_indices_backward = np.arange(trg_size)
        src_indices_forward = np.arange(src_size)
        src_indices_backward = np.zeros(trg_size, dtype=int)

        # Update the training dictionary
        if direction in ('forward', 'union'):
            if csls_neighbor > 0:
                for i in range(0, trg_size, batchsize):
                    j = min(i + batchsize, trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                    knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=csls_neighbor)
            for i in range(0, src_size, simfwd.shape[0]):
                j = min(i + batchsize, src_size)
                xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                dropout_matrix(simfwd[:j-i], dropout).argmax(axis=1, out=trg_indices_forward[i:j])
        if direction in ('backward', 'union'):
            if csls_neighbor > 0:
                for i in range(0, src_size, batchsize):
                    j = min(i + batchsize, src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                    knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=csls_neighbor)
            for i in range(0, trg_size, batchsize):
                j = min(i + batchsize, trg_size)
                zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                dropout_matrix(simbwd[:j-i], dropout).argmax(axis=1, out=src_indices_backward[i:j])
        if direction == 'forward':
            src_indices = src_indices_forward
            trg_indices = trg_indices_forward
        elif direction == 'backward':
            src_indices = src_indices_backward
            trg_indices = trg_indices_backward
        elif direction == 'union':
            src_indices = np.concatenate((src_indices_forward, src_indices_backward))
            trg_indices = np.concatenate((trg_indices_forward, trg_indices_backward))

        # Objective function evaluation
        if direction == 'forward':
            objective = np.mean(best_sim_forward).tolist()
        elif direction == 'backward':
            objective = np.mean(best_sim_backward).tolist()
        elif direction == 'union':
            fwd_mean = np.mean(best_sim_forward)
            bwd_mean = np.mean(best_sim_backward)
            objective = (fwd_mean + bwd_mean) / 2

        return list(src_indices), list(trg_indices), objective

    def train(self, fix_ent, n_word=10000, n_ent=10000, 
            init_dropout=0.9, interval=5, dropout_decay=2.0, threshold=1e-6):
        best_obj = obj = -100
        it = 1
        last_improvement = 0
        dropout = init_dropout
        while True:
            if it - last_improvement > interval:
                if dropout <= 0.0:
                    break
                dropout = 1-min(1.0, dropout_decay * (1-dropout))
                last_improvement = it

            self.learn_mapping()
            self.induce_word_dic(n_word, dropout=dropout)
            obj = self.word_obj
            if not fix_ent:
                self.induce_ent_dic(n_ent, dropout=dropout)
                obj = (obj + self.ent_obj) / 2

            if obj - best_obj >= threshold:
                last_improvement = it
                best_obj = obj

            if self.word_gold:
                acc, sim = self.validate_word()
                logger.info("iter={}; dropout={:.3f}; obj={:.3f}; acc={:.3f}; sim={:.3f}".format(
                    it, dropout, obj, acc, sim))
            else:
                logger.info("iter={}; dropout={:.3f}; obj={:.3f}".format(it, dropout, obj))

            it += 1

        src_wiki = self.get_wiki(self.src, self.src_word_mapped, self.src_ent_mapped)
        trg_wiki = self.get_wiki(self.trg, self.trg_word_mapped, self.trg_ent_mapped)
        return src_wiki, trg_wiki

    def get_wiki(self, lang, word_mapped, ent_mapped):
        wiki = Wikipedia2Vec(lang.dic)
        wiki.syn0 = np.empty((len(lang), self.dim), dtype=np.float32)

        for item_ind, ind in lang.word2id.items():
            wiki.syn0[item_ind] = word_mapped[ind]

        for item_ind, ind in lang.ent2id.items():
            wiki.syn0[item_ind] = ent_mapped[ind]
        
        return wiki
        

    def validate(self, xw, zw, gold):
        src = list(gold.keys())
        simval = xw[src].dot(zw.T)
        nn = simval.argmax(axis=1)
        accuracy = np.mean([1 if nn[i] in gold[src[i]] else 0 for i in range(len(src))])
        similarity = np.mean([max([simval[i, j] for j in gold[src[i]]] 
            for i in range(len(src)))])
        return accuracy, similarity

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

