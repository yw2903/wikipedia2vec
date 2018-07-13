from wikipedia2vec import Wikipedia2Vec
import embeddings
#from cupy_utils import *

import argparse
import collections
import numpy as np
import re
import sys
import time
import click
import logging

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


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

def init_unsupervised(x, z, 
        n_vocab=4000, normalize=['unit', 'center', 'unit'], csls_neighborhood=10):
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
    embeddings.normalize(xsim, normalize)
    embeddings.normalize(zsim, normalize)
    sim = xsim.dot(zsim.T)

    knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
    knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
    sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2

    src_indices = np.concatenate((np.arange(sim_size), sim.argmax(axis=0)))
    trg_indices = np.concatenate((sim.argmax(axis=1), np.arange(sim_size)))
    del xsim, zsim, sim

    return src_indices, trg_indices

def mapping(x, z, src_indices, trg_indices, reweight=0.5):
    xw = x
    zw = z

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

    return xw, zw

def induce_dictionary(xw, zw, 
        direction='union', csls_neighborhood=10, batchsize=10000, dropout=0.0):
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
        if csls_neighborhood > 0:
            for i in range(0, trg_size, batchsize):
                j = min(i + batchsize, trg_size)
                zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=csls_neighborhood)
        for i in range(0, src_size, simfwd.shape[0]):
            j = min(i + batchsize, src_size)
            xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
            simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
            simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
            dropout_matrix(simfwd[:j-i], dropout).argmax(axis=1, out=trg_indices_forward[i:j])
    if direction in ('backward', 'union'):
        if csls_neighborhood > 0:
            for i in range(0, src_size, batchsize):
                j = min(i + batchsize, src_size)
                xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=csls_neighborhood)
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
        objective = (np.mean(best_sim_forward) + np.mean(best_sim_backward)).tolist() / 2

    return src_indices, trg_indices, objective

class Validater:
    def __init__(self, path, src_word2ind, trg_word2ind):
        oov = set()
        vocab = set()
        self.validation = collections.defaultdict(set)

        with open(path) as f:
            for line in f:
                src, trg = line.strip().split()
                try:
                    src_ind = src_word2ind[src]
                    trg_ind = trg_word2ind[trg]
                    self.validation[src_ind].add(trg_ind)
                    vocab.add(src)
                except KeyError:
                    oov.add(src)

        oov -= vocab
        self.coverage = len(self.validation) / (len(self.validation) + len(oov))

    def validate(self, xw, zw):
        src = list(self.validation.keys())
        simval = xw[src].dot(zw.T)
        nn = simval.argmax(axis=1)
        accuracy = np.mean([1 if nn[i] in self.validation[src[i]] else 0 for i in range(len(src))])
        similarity = np.mean([max([simval[i, j].tolist() for j in self.validation[src[i]]]) for i in range(len(src))])
        return accuracy, similarity

@click.command()
@click.argument('src_input', type=click.Path(exists=True))
@click.argument('trg_input', type=click.Path(exists=True))
@click.argument('src_output', type=click.Path())
@click.argument('trg_output', type=click.Path())
@click.option('--normalize', type=list, default=['unit', 'center', 'unit'])
@click.option('--validation', type=click.Path(exists=True))
@click.option('--init_dropout', type=float, default=0.9)
@click.option('--dropout_decay', type=float, default=2.0)
@click.option('--threshold', type=float, default=1e-6)
@click.option('--interval', type=int, default=50)
@click.option('--n_vocab', type=int, default=20000)
@click.option('--init_n_vocab', type=int, default=4000)
@click.option('--n_csls', type=int, default=10)
@click.option('--reweight', type=float, default=0.5)
@click.option('--batchsize', type=int, default=10000)
def main(src_input, trg_input, src_output, trg_output, normalize, validation, 
        init_dropout, dropout_decay, threshold, interval, n_vocab, init_n_vocab, n_csls,
        reweight, batchsize):
    src_wiki2vec = Wikipedia2Vec.load(src_input)
    trg_wiki2vec = Wikipedia2Vec.load(trg_input)

    x_full = np.array(src_wiki2vec.syn0)
    z_full = np.array(trg_wiki2vec.syn0)

    embeddings.normalize(x_full, normalize)
    embeddings.normalize(z_full, normalize)

    src_words, x = embeddings.get_emb(src_wiki2vec.dictionary, x_full, n_vocab)
    trg_words, z = embeddings.get_emb(trg_wiki2vec.dictionary, z_full, n_vocab)

    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    src_indices, trg_indices = init_unsupervised(x, z, 
            n_vocab=init_n_vocab, csls_neighborhood=n_csls, normalize=normalize)

    if validation is not None:
        validater = Validater(validation, src_word2ind, trg_word2ind)
        logger.info("Validation Coverage: {}".format(validater.coverage))

    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    dropout = init_dropout
    while True:
        if it - last_improvement > interval:
            if dropout <= 0.0:
                break
            dropout =1-min(1.0, dropout_decay * (1-dropout))
            last_improvement = it

        xw, zw = mapping(x, z, src_indices, trg_indices, reweight=reweight)
        src_indices, trg_indices, objective = induce_dictionary(xw, zw, 
                csls_neighborhood=n_csls, batchsize=batchsize, dropout=dropout)

        if objective - best_objective >= threshold:
            last_improvement = it
            best_objective = objective

        if validation:
            acc, sim = validater.validate(xw, zw)
            logger.info("iter={}; dropout={:.3f}; obj={:.3f}; acc={:.3f}; sim={:.3f}".format(
                it, dropout, objective, acc, sim))
        else:
            logger.info("iter={}; dropout={:.3f}; obj={:.3}".format(it, dropout, objective))

        it += 1

    
    xw_full, zw_full = mapping(x_full, z_full, src_indices, trg_indices)
    mapped_src = Wikipedia2Vec(src_wiki2vec.dictionary)
    mapped_trg = Wikipedia2Vec(trg_wiki2vec.dictionary)
    mapped_src.syn0 = xw_full
    mapped_trg.syn0 = zw_full
    mapped_src.save(src_output)
    mapped_trg.save(trg_output)


if __name__ == '__main__':
    main()
