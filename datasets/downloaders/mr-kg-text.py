import argparse
from os.path import join

import urllib.parse

from configuration.datasets import dataset_names
from datasets.downloaders.wikidata.queries import get_entity_labels
from datasets.downloaders.wikimedia.queries import get_text_descriptions
from shared.utility import valid_dir, load_relations, load_entities, save_entities, beautify, get_dataset_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='Location of datasets', default='..')
parser.add_argument('--dataset', choices=dataset_names, help='Datasets to process')
parser.add_argument('--extension', default=None, help='Parse \'processed\' if using processed entity list. '
                                                      'Might be faster for larger datasets')


def run(path, dataset, extension):
    dataset = get_dataset_configuration(dataset)
    in_path = join(path, dataset.name)

    entities = load_entities(in_path, fname_extension=extension)

    assert all([i == e.index for i, e in enumerate(entities)]), 'All entities not in order, should be handled'

    # Find entities with Wikidata URI
    skipped = [e for e in entities if not e.original_id.startswith('http://www.wikidata.org')]
    query_entities = [e for e in entities if e.original_id.startswith('http://www.wikidata.org')]

    if skipped:
        print(f'Warning: No uri for {len(skipped)} entities.')

    uri_index = {e.original_id: i for i, e in enumerate(entities)}
    url_index = {}
    index_desc = {}  # Used as backup descriptions if no Wikipage is available

    # Query Wikidata for Wikipedia links
    for block in get_entity_labels([e.original_id for e in query_entities if e.description is None]):
        for _, (uri, _, desc, url) in block.iterrows():
            i = uri_index[uri]
            if url is not None:
                url = urllib.parse.unquote(url)
                url_index[url] = i

            if desc is not None:
                index_desc[i] = desc

    if len(url_index) < len(entities):
        print(f'Warning: Could only find wiki URLs for {len(url_index)} entities out of {len(entities)}')

    # Query Wikipedia for entity descriptions
    for descriptions in get_text_descriptions(list(url_index.keys())):
        for url, desc in descriptions.items():
            entities[url_index[url]].description = desc['WIKI_DESCRIPTION']

    # Save entities
    save_entities(in_path, {e.index: e for e in entities}, fname_extension='wiki')

    # Beautiful soup
    beautify_entities = [e for e in entities if e.description is not None]
    beautify_entities = beautify(beautify_entities)

    assert all([entities[e.index].description == e.description for e in beautify_entities]), \
        'All should have assigned description'

    if len(beautify_entities) < len(entities):
        print(f'Warning: Could only find descriptions for {len(beautify_entities)} entities out of {len(entities)}')

        # Save entities with wiki URL but no description
        save_entities(in_path, {i: entities[i] for i in url_index.values() if entities[i].description is None},
                      fname_extension='no_wiki')

    for entity in entities:
        if entity.description is None:
            entity.description = index_desc.get(entity.index, None)

    # Save with textual descriptions
    save_entities(in_path, {e.index: e for e in entities})


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    run(**args)
