from wikipedia2vec import Wikipedia2Vec
from mapper import Mapper, Parameters

import numpy as np
import click
import logging

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

@click.command()
@click.argument('src_input', type=click.Path(exists=True))
@click.argument('trg_input', type=click.Path(exists=True))
@click.argument('src_output', type=click.Path())
@click.argument('trg_output', type=click.Path())
@click.option('--langlink', type=click.Path(exists=True))
@click.option('--fix-ent/--update-ent', default=True)
@click.option('--normalize', type=list, default=['unit', 'center', 'unit'])
@click.option('--init_dropout', type=float, default=0.9)
@click.option('--dropout_decay', type=float, default=2.0)
@click.option('--init_n_word', type=int, default=4000)
@click.option('--init_n_ent', type=int, default=4000)
@click.option('--n_word', type=int, default=10000)
@click.option('--n_ent', type=int, default=10000)
@click.option('--threshold', type=float, default=1e-6)
@click.option('--interval', type=int, default=50)
@click.option('--csls', type=int, default=10)
@click.option('--reweight', type=float, default=0.5)
@click.option('--batchsize', type=int, default=10000)
@click.option('--validation', type=click.Path(exists=True))
def main(src_input, trg_input, src_output, trg_output, **kwargs):
    params = Parameters(**kwargs)

    logger.info("Loading wiki")
    src_wiki2vec = Wikipedia2Vec.load(src_input)
    trg_wiki2vec = Wikipedia2Vec.load(trg_input)

    logger.info("Creating mapper")
    mapper = Mapper(src_wiki2vec, trg_wiki2vec, params)

    logger.info("Initialize")
    mapper.initialize()

    logger.info("Training")
    src_mapped, trg_mapped = mapper.train()

    logger.info("Saving")
    src_mapped.save(src_output)
    trg_mapped.save(trg_output)

if __name__ == '__main__':
    main()
