import bz2
import click

from urllib.parse import urlparse

@click.command()
@click.argument('input', type=click.Path(exists=True))
def main(input):
    with bz2.open(input, 'rt') as f:
        count = 0
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue

            src, _, trg, _ = map(lambda x: urlparse(x[1:-1]), line.strip().split())
            if trg.netloc != 'dbpedia.org':
                continue
            
            src_title = src.path.split('/')[-1]
            trg_title = src.path.split('/')[-1]
            if src_title and trg_title:
                print("{}\t{}".format(src_title, trg_title))

if __name__ == '__main__':
    main()
