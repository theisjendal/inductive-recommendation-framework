import argparse
import os
import zipfile
from os.path import join, isfile

from tqdm import tqdm
from loguru import logger
import pandas as pd

from datasets.downloaders.wikidata.bfs import BreathFirstSearch
from datasets.downloaders.wikidata.queries import get_uris, get_entity_labels, get_predicate_labels
from shared.utility import valid_dir, download_progress, save_json, load_json

URL = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, default='../movielens',
                    help='path to store downloaded data')
parser.add_argument('--n_hops', type=int, default=1, help='number of hops to move away from seed set (movielens movies), '
                                                        'when querying wikidata for entities.')
parser.add_argument('--clear_state', action='store_true', default=False, help='remove bfs query state. ')


def download(out_path):
    zip_name = 'movielens.zip'
    download_progress(URL, out_path, zip_name)

    with zipfile.ZipFile(join(out_path, zip_name), 'r') as zip_ref:
        zip_ref.extractall(out_path)
        

def download_entities(out_path, n_hops):
    logger.warning('Skipping all previously downloaded wikidata paths.')

    # Get movielens movie ids and imdb ids.
    movies = pd.read_csv(join(out_path, 'ml-20m', 'links.csv'), sep=',', dtype=str)
    movie_ids = set(movies['movieId'])

    # If file exist, find if all movies uris are downloaded.
    movie_uris_path = join(out_path, 'movie_uris.json')
    if isfile(movie_uris_path):
        movie_uris = load_json(movie_uris_path)
        movie_ids.difference_update(movie_uris.keys())
    else:
        movie_uris = {}

    # If not all uris have been downloaded, query for them.
    if movie_ids:
        logger.info('Getting uris')
        imdb_movie = movies.set_index('imdbId')['movieId'].to_dict()

        # Get movie uris, e.g. 'https://www.wikidata.org/wiki/Q227474' from imdb_id 'tt0098520' (tt is prepended).
        results = get_uris([imdb for imdb, movie in imdb_movie.items() if movie in movie_ids])
        movie_uris.update({imdb_movie[imdb]: uri for imdb, uri in results.items()})
        save_json(movie_uris_path, movie_uris)

    # Get uris only
    movie_uris = set(list(movie_uris.values()))

    # Get unqueried movie uris
    triples_path = join(out_path, 'triples.csv')
    bfs = BreathFirstSearch(list(movie_uris), n_hops)
    movie_uris = movie_uris.difference(bfs.get_visited())

    # If there are unqueried movies or if bfs has an non empty queue continue.
    if movie_uris or bfs.has_queue():
        logger.info('Getting triples')
        bfs.bfs(triples_path)

        # Remove duplicates, for windows it may require awk download.
        logger.info('Removing duplicates')
        tmp_path = join(out_path, 't_tmp.csv')
        os.system(f"awk -F ',' '!x[$1, $2, $3]++' {triples_path} > {tmp_path}")
        os.system(f"mv {tmp_path} {triples_path}")

    all_uris = bfs.get_visited()
    if not all_uris:
        # Load all uris.
        for df in tqdm(pd.read_csv(triples_path, chunksize=500000), 'Getting unique uris'):
            all_uris.update(df.s)
            all_uris.update(df.o)
            break

    entity_path = join(out_path, 'entities.csv')
    labelled_uris = set()
    if os.path.exists(entity_path):
        labelled_uris = set(pd.read_csv(entity_path, usecols=['uri']).uri)

    # Get uri labels
    unqueried = set(all_uris).difference(labelled_uris)
    if unqueried:
        with open(entity_path, 'a+') as f:
            if labelled_uris:
                head = False
            else:
                head = True

            for data in get_entity_labels(unqueried):
                data.to_csv(f, header=head, index=False)
                head = False

    # Get predicate labels
    predicates = set(pd.read_csv(triples_path, usecols=['p']).p)

    predicate_path = join(out_path, 'predicates.csv')

    if os.path.exists(predicate_path):
        queried_predicates = set(pd.read_csv(predicate_path, usecols=['uri']).uri)
        predicates.difference_update(queried_predicates)
        header = False
    else:
        header = True

    if predicates:
        with open(predicate_path, 'a') as f:
            for df in get_predicate_labels(predicates):
                df.to_csv(f, index=False, header=header)
                header = False


if __name__ == '__main__':
    args = parser.parse_args()

    download(args.path)

    download_entities(args.path, args.n_hops)
