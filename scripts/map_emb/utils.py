import numpy as np
try:
    import cupy
except:
    pass

def array_module(x):
    return cupy.get_array_module(x)

def get_word_emb(wiki, size, cuda=True):
    xp = cupy if cuda else np

    dim = wiki.syn0.shape[1]

    emb = xp.empty((size, dim), dtype=xp.float32)
    word2id = {}

    for i, word in enumerate(wiki.dictionary.words(ordered=True)):
        if i >= size:
            break

        emb[i] = xp.array(wiki.syn0[word.index], dtype=xp.float32)
        word2id[word.index] = i

    return word2id, emb

def get_entity_emb(wiki, size, cuda=True):
    xp = cupy if cuda else np

    dim = wiki.syn0.shape[1]

    emb = xp.empty((size, dim), dtype=xp.float32)
    word2id = {}

    for i, ent in enumerate(wiki.dictionary.entities(ordered=True)):
        if i >= size:
            break

        emb[i] = xp.array(wiki.syn0[ent.index], dtype=xp.float32)
        word2id[ent.index] = i

    return word2id, emb

    
def normalize(emb, normalization):
    xp = array_module(emb)
    if normalization == 'unit':
        norm = xp.linalg.norm(emb, axis=1)
        norm[norm == 0] = 1
        emb /= norm[:, None]
    elif normalization == 'mean_center':
        mean = xp.mean(emb, axis=0)
        emb -= mean
