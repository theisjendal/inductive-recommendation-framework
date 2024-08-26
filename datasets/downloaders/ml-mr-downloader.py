import argparse
from os.path import join

import pandas as pd
from shutil import copyfile

from tqdm import tqdm

from shared.utility import valid_dir, load_json

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=valid_dir, default='../ml-mr',
                    help='path to store downloaded data')
parser.add_argument('--movielens', type=valid_dir, default='../movielens',
                    help='path to movielens dataset')
parser.add_argument('--mindreader', type=valid_dir, default='../mindreader',
                    help='path to mindreader dataset')


def run(out_path, ml_path, mr_path):
    movie_uris = load_json(join(ml_path, 'movie_uris.json'))

    movies_df = pd.read_csv(join(mr_path, 'entities.csv'), index_col=None, usecols=['uri', 'labels'])
    movies = set(movies_df[movies_df['labels'].str.contains('Movie')]['uri'])

    ml_ratings = pd.read_csv(join(ml_path, 'ml-20m', 'ratings.csv'), dtype=str)
    before = len(ml_ratings)
    ml_ratings['movieId'] = ml_ratings['movieId'].map(movie_uris)
    ml_ratings = ml_ratings[ml_ratings['movieId'].isin(movies)]
    after = len(ml_ratings)

    print(f'Removed {before-after} ratings as item not in MR KG')

    ml_mr_ratings = {'userId': ml_ratings['userId'].tolist(), 'uri': ml_ratings['movieId'].tolist(),
                     'isItem': [True]*len(ml_ratings), 'sentiment': ml_ratings['rating'].tolist(),
                     'timestamp': ml_ratings['timestamp'].tolist()}

    for file in ['triples.csv', 'entities.csv']:
        src = join(mr_path, file)
        dst = join(out_path, file)

        copyfile(src, dst)

    ratings_df = pd.DataFrame.from_dict(ml_mr_ratings)
    ratings_df.to_csv(join(out_path, 'ratings.csv'))


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.out_path, args.movielens, args.mindreader)
