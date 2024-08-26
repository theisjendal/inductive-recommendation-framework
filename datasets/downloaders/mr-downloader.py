import argparse
from collections import defaultdict
from os.path import join

import pandas as pd

from shared.utility import valid_dir, download_progress

BASE_URL = 'https://mindreader.tech/api'
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=valid_dir, help='path to store downloaded data', default='../mindreader')


def download(out_path):
    download_progress(join(BASE_URL, 'triples'), out_path, 'triples.csv')
    download_progress(join(BASE_URL, 'entities'), out_path, 'entities.csv')
    download_progress(join(BASE_URL, 'ratings?versions=100k,100k-newer,100k-fix,thesis,thesis-ppr,thesis-launch&final'
                                     '=yes'), out_path, 'ratings.csv')


def transform_type_to_entity(out_path):
    entity_path = join(out_path, 'entities.csv')
    df = pd.read_csv(entity_path)

    # Add new relations for each entity
    new_triples = {'head_uri': [], 'relation': [], 'tail_uri': []}
    for i, (uri, name, labels) in df.iterrows():
        labels = labels.split('|')

        for label in labels:
            # If label is a subclass or if only a superclass occurs add label. I.e., skip Category if Genre is also a
            # label as Genre is more specific.
            if label in ['Actor', 'Decade', 'Director', 'Company', 'Genre', 'Movie', 'Subject'] or \
                    (label == 'Category' and not {'Genre', 'Subject'}.intersection(labels)) or \
                    (label == 'Person' and not {'Actor', 'Director'}.intersection(labels)):
                new_triples['head_uri'].append(uri)
                new_triples['relation'].append('INSTANCE_OF')
                new_triples['tail_uri'].append(label)

    # Add superclass
    for subclass in ['Actor', 'Director']:
        new_triples['head_uri'].append(subclass)
        new_triples['relation'].append('SUBCLASS_OFF')
        new_triples['tail_uri'].append('Person')

    for subclass in ['Genre', 'Subject']:
        new_triples['head_uri'].append(subclass)
        new_triples['relation'].append('SUBCLASS_OFF')
        new_triples['tail_uri'].append('Category')

    new_entities = defaultdict(list)
    for label in ['Actor', 'Decade', 'Director', 'Category', 'Company', 'Genre', 'Movie', 'Person', 'Subject']:
        new_entities['uri'].append(label)
        new_entities['name'].append(label)
        new_entities['labels'].append('Type')

    df = df.append(pd.DataFrame.from_dict(new_entities))
    df.to_csv(entity_path, index=False)

    # load and save new triples
    triples_path = join(out_path, 'triples.csv')
    df = pd.read_csv(triples_path)
    df = df.append(pd.DataFrame.from_dict(new_triples), ignore_index=True)
    df.to_csv(triples_path)


def run(out_path):
    download(out_path)
    # transform_type_to_entity(out_path)


if __name__ == '__main__':
    arg = parser.parse_args()
    run(arg.output)
