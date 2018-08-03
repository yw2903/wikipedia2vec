#! /usr/bin/python3

import click
import os
from collections import defaultdict
import numpy as np
from wikipedia2vec import Wikipedia2Vec

import logging

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def filter_emb(wiki, items, n_count):
    items = sorted(items, key=lambda item: item.count, reverse=True)[:n_count]
    item_indices = [item.index for item in items]
    item2id = {item.index: i for i, item in enumerate(items)}
    id2item = {i: item.index for i, item in enumerate(items)}
    emb = wiki.syn0[item_indices]
    emb = emb / np.linalg.norm(emb, axis=1)[:,None]
    return item2id, id2item, emb

def load_data(path, src_dic, trg_dic, src_item2id, trg_item2id, is_ent=False):
    gold = defaultdict(set)
    with open(path) as f:
        for line in f:
            src_word, trg_word = line.strip().split()
            if is_ent:
                src_item = src_dic.get_entity(src_word)
                trg_item = trg_dic.get_entity(trg_word)
            else:
                src_item = src_dic.get_word(src_word)
                trg_item = trg_dic.get_word(trg_word)

            if src_item is None or src_item.index not in src_item2id:
                continue

            if trg_item is None or trg_item.index not in trg_item2id:
                continue

            src_id = src_item2id[src_item.index]
            trg_id = trg_item2id[trg_item.index]
            gold[src_id].add(trg_id)

    return gold

def eval(src_emb, trg_emb, gold):
    src = list(gold.keys())
    simval = src_emb[src].dot(trg_emb.T)
    nn = simval.argmax(axis=1)
    acc = np.mean([1 if nn[i] in gold[src[i]] else 0 for i in range(len(src))])
    sim = np.mean([max(simval[i, j] for j in gold[src[i]]) for i in range(len(src))])
    return acc, sim


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('src_model_file', type=click.Path(exists=True))
@click.argument('trg_model_file', type=click.Path(exists=True))
@click.option('--n_word', type=int, default=20000)
@click.option('--n_ent', type=int, default=20000)
@click.option('--word/--no-word', default=True)
@click.option('--entity/--no-entity', default=True)
def main(data_dir, src_model_file, trg_model_file, n_word, n_ent, word, entity):
    src_model = Wikipedia2Vec.load(src_model_file)
    trg_model = Wikipedia2Vec.load(trg_model_file)
    src_dic = src_model.dictionary
    trg_dic = trg_model.dictionary

    src_word2id, src_id2word, src_word_emb = filter_emb(src_model, src_dic.words(), n_word)
    src_ent2id, src_id2ent, src_ent_emb = filter_emb(src_model, src_dic.entities(), n_ent)
    trg_word2id, trg_id2word, trg_word_emb = filter_emb(trg_model, trg_dic.words(), n_word)
    trg_ent2id, trg_id2ent, trg_ent_emb = filter_emb(trg_model, trg_dic.entities(), n_ent)

    results = {}

    if word and os.path.isdir(os.path.join(data_dir, 'word')):
        for filename in os.listdir(os.path.join(data_dir, 'word')):
            if not filename.endswith('.txt'):
                continue

            gold = load_data(os.path.join(data_dir, 'word', filename), 
                    src_dic, trg_dic, src_word2id, trg_word2id)
            acc, sim = eval(src_word_emb, trg_word_emb, gold)
            logger.info("filename={}; acc={:.3f}; sim={:.3f}".format(filename, acc, sim))
    
    if entity and os.path.isdir(os.path.join(data_dir, 'entity')):
        for filename in os.listdir(os.path.join(data_dir, 'entity')):
            if not filename.endswith('.txt'):
                continue

            gold = load_data(os.path.join(data_dir, 'entity', filename))
            acc, sim = eval(src_ent_emb, trg_ent_emb, gold)
            logger.info("filename={}; acc={:.3f}; sim={:.3f}".format(filename, acc, sim))

if __name__ == '__main__':
    main()
