#! /usr/bin/python3

import click
import os
from collections import defaultdict
import numpy as np
from wikipedia2vec import Wikipedia2Vec

@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('src_model_file', type=click.Path(exists=True))
@click.argument('trg_model_file', type=click.Path(exists=True))
@click.option('--word/--no-word', default=True)
@click.option('--entity/--no-entity', default=True)
def main(data_dir, src_model_file, trg_model_file, word, entity):
    src_model = Wikipedia2Vec.load(src_model_file)
    trg_model = Wikipedia2Vec.load(trg_model_file)

    results = {}

    if word and os.path.isdir(os.path.join(data_dir, 'word')):
        trg_words = sorted(trg_model.dictionary.words(), key=lambda w: w.count, reverse=True)[:100000]
        trg_syn0 = trg_model.syn0[[word.index for word in trg_words]]
        trg_syn0 = trg_syn0 / np.linalg.norm(trg_syn0, axis=1, keepdims=True)

        for filename in os.listdir(os.path.join(data_dir, 'word')):
            if not filename.endswith('.txt'):
                continue

            bi_dict = defaultdict(list)
            with open(os.path.join(data_dir, 'word', filename)) as f:
                for line in f:
                    src_word, trg_word = line.strip().split()
                    bi_dict[src_word].append(trg_word)

            hit_count = 0
            total_count = 0
            oov_count = 0

            for src_word, trans in bi_dict.items():
                try:
                    src_vec = src_model.get_word_vector(src_word)

                    scores = trg_syn0.dot(src_vec)
                    top_index = np.argmax(scores)
                    prediction = trg_words[top_index]
                    total_count += 1
                    if prediction.text in trans:
                        hit_count += 1
                except KeyError:
                    oov_count += 1

            results[filename[:-4]] = ("Accuracy", hit_count/total_count, oov_count)

    if entity and os.path.isdir(os.path.join(data_dir, 'entity')):
        trg_entities = sorted(trg_model.dictionary.entities(), key=lambda e: e.count, reverse=True)[:100000]
        trg_syn0 = trg_model.syn0[[entity.index for entity in trg_entities]]
        trg_syn0 = trg_syn0 / np.linalg.norm(trg_syn0, axis=1, keepdims=True)

        for filename in os.listdir(os.path.join(data_dir, 'entity')):
            if not filename.endswith('.txt'):
                continue

            bi_dict = defaultdict(list)
            with open(os.path.join(data_dir, 'entity', filename)) as f:
                for line in f:
                    src_entity, trg_entity = line.strip().split('\t')
                    bi_dict[src_entity].append(trg_entity)

            hit_count = 0
            total_count = 0
            oov_count = 0

            for src_entity, trans in bi_dict.items():
                try:
                    src_vec = src_model.get_entity_vector(src_entity)

                    scores = trg_syn0.dot(src_vec)
                    top5_index = np.argsort(-scores)[:5]
                    prediction = [trg_entities[idx] for idx in top5_index]
                    total_count += 1
                    if any(pred.title in trans for pred in prediction):
                        hit_count += 1
                except KeyError:
                    oov_count += 1

            results[filename[:-4]] = ("Accuracy", hit_count/total_count, oov_count)

    for name, (measure, score, oov) in results.items():
        print('{}:'.format(name))
        print('   {}: {:.4f}'.format(measure, score))
        print('   OOV: {}'.format(oov))

if __name__ == '__main__':
    main()
