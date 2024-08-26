import argparse
import os
import pickle
from collections import defaultdict
from zipfile import ZipFile

import pandas as pd
import re

import rdflib
from bs4 import BeautifulSoup
from loguru import logger
from rdflib import Graph
from tqdm import tqdm

from datasets.downloaders import wikidata, wikimedia, yelp_virtuoso

from shared.utility import valid_dir, download_progress, load_entities, save_entities, load_users

BASE_URL = "https://zenodo.org/record/8049832/files/{0}?download=1"
FILE_LIST = ['yelp_categories.ttl', 'yelp_entities.ttl']

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', nargs=1, type=valid_dir, default='../yelpkg',
                    help='path to store downloaded data')


def business_text_assignment(path, entity_map, user_map, seen, unique_ends):
    queries = yelp_virtuoso.queries

    unique_ends = unique_ends
    logger.info('Getting business names')
    name_res = queries.result_dict(queries.get_results, yelp_virtuoso.queries.business_name_query)
    logger.info('Getting business review texts')
    review_res = queries.result_dict(queries.query_iterator, queries.business_review_text_query, unique_ends)

    logger.info('Converting to df')
    n_df = pd.DataFrame.from_dict(name_res)
    t_df = pd.DataFrame.from_dict(review_res)
    t_df['time'] = pd.to_datetime(t_df['time']).apply(lambda x: int(x.timestamp()))

    for business, df in tqdm(n_df.groupby('business'), desc='Assigning name to business entities'):
        entity_map[business].name = df.iloc[0]['businessLabel']
        seen.add(business)

    t_df['user'] = t_df['user'].apply(lambda x: None if (u := user_map.get(x)) is None else u.index)
    t_df = t_df.dropna()
    for business, df in tqdm(t_df.groupby('business'), desc='Assigning review texts to business entities'):
        # sort reviewtexts by time ascending
        df = df.sort_values(by='time')
        entity_map[business].description = df[['user', 'reviewText']].to_records(index=False).tolist()

    return entity_map, seen


def business_property_text(path, entity_map, seen, unique_ends):
    queries = yelp_virtuoso.queries

    logger.info('Getting property names')
    property_res = queries.result_dict(queries.query_iterator, queries.property_name_query,
                                            unique_ends)
    n_df = pd.DataFrame.from_dict(property_res)

    for property, df in tqdm(n_df.groupby('property'), desc='Assigning property names'):
        # Split on capital letters and create 'sentence' from list
        name = re.findall('[A-Z][^A-Z]*', df.iloc[0]['property'])
        name = ' '.join(name)
        entity_map[property].name = name
        seen.add(property)

    return entity_map, seen


def service_text_assignment(path, entity_map, seen, unique_ends):
    # Create triples from ?s to ?p2 using ?p as predicate, and set ?type as type of ?p2
    queries = yelp_virtuoso.queries

    logger.info('Getting service names')
    service_name = queries.result_dict(queries.get_results, queries.service_name_query)
    n_df = pd.DataFrame.from_dict(service_name)

    for service, df in tqdm(n_df.groupby('service'), desc='Assigning service names'):
        entity_map[service].name = df.iloc[0]['service'].rsplit('#', 1)[1].replace('has', '').capitalize()
        seen.add(service)

    return entity_map, seen


def get_wiki_text(uris):
    entity_information = defaultdict(dict)

    # Get text and wikipedia links from wikidata
    dfs = []
    for df in wikidata.queries.get_entity_labels(uris):
        dfs.append(df)

    df = pd.concat(dfs)
    df['uri'] = df.uri.apply(lambda x: x.replace('http', 'https'))

    wikipedia_mapping = {}
    for _, row in df.iterrows():
        entity_information[row['uri']]['name'] = row['label']
        entity_information[row['uri']]['description'] = row['description']

        if row['wikilink'] is not None:
            wikipedia_mapping[row['wikilink']] = row['uri']

    # Query wikipedia for text
    links = list(wikipedia_mapping.keys())
    texts = {}
    for e_dict in wikimedia.queries.get_text_descriptions(links):
        for url, d in e_dict.items():
            desc = d['WIKI_DESCRIPTION']
            soup = BeautifulSoup(desc, 'html.parser')
            entity_information[wikipedia_mapping[url]]['wd_description'] = \
                re.sub(r'\s+', ' ', ' '.join(soup.stripped_strings))

    return entity_information


def location_text_assignment(path, entity_map, seen, unique_ends):
    logger.info('Getting location names')
    location_res = yelp_virtuoso.queries.result_dict(yelp_virtuoso.queries.get_results,
                                                     yelp_virtuoso.queries.location_query)

    entity_info = get_wiki_text(location_res['location'])
    for location, info in tqdm(entity_info.items(), desc='Assigning location names'):
        entity_map[location].name = info['name']
        if 'wd_description' in info:
            entity_map[location].description = info['wd_description']
        else:
            entity_map[location].description = info['description']
        seen.add(location)

    return entity_map, seen


def category_text_assighment(path, entity_map, seen, unique_ends):
    queries = yelp_virtuoso.queries

    logger.info('Getting category names and descriptions')
    res = queries.result_flattener(queries.get_results, queries.yelp_category_text_query)
    wd_links = {}
    schema_links = {}

    for row in res:
        cat = row['category']
        entity_map[cat].name = ' '.join(cat.split('_'))

        if 'wd' in row:
            wd_links[row['wd']] = cat
        if 'schema' in row:
            schema_links[row['schema']] = cat

        seen.add(cat)

    # Get text and wikipedia links from wikidata
    logger.info('Getting wd text')
    wd_entity_info = get_wiki_text(list(wd_links.keys()))
    for wd, info in tqdm(wd_entity_info.items(), desc='Assigning wd category names'):
        uri = wd_links[wd.replace('https', 'http')]
        entity_map[uri].name = info['name']
        if 'wd_description' in info:
            entity_map[uri].description = info['wd_description']
        else:
            entity_map[uri].description = info['description']

    # Get schema text
    logger.info('Getting schema text')
    sq = queries.schema_label_query.format(' '.join(['<{}>'.format(l) for l in schema_links.keys()]))\
        .replace('https', 'http')
    schema_res = queries.result_flattener(queries.get_results, query=sq, add_prefixes=False)
    for row in tqdm(schema_res, desc='Assigning schema category names'):
        uri = schema_links[row['s'].replace('http', 'https')]
        entity = entity_map[uri]
        name = ' '.join(re.findall(r'[A-Z][a-z]*', row['label']))

        if entity.name == uri or len(entity.name) < len(name):
            entity.name = name

        # assumes wikipedia/wikidata description is better than schema.org
        if 'description' in row and entity.description is None:
            entity.description = row['description']

    return entity_map, seen


def download(path):
    try:
        entiites = load_entities(path)
        entity_map = {e.original_id: e for e in entiites}
        users = load_users(path)
        user_map = {u.original_id: u for u in users}
    except FileNotFoundError:
        raise FileNotFoundError('No entities pickle file found, run yelpkg_converter.py first')

    logger.info('Getting unique uri ends for batching purposes')
    unique_ends = yelp_virtuoso.queries.get_results(yelp_virtuoso.queries.distinct_uri_ends_query)
    unique_ends = [r['substring']['value'] for r in unique_ends]

    seen = set()
    entity_map, seen = business_text_assignment(path, entity_map, user_map, seen, unique_ends)  # businesss
    entity_map, seen = business_property_text(path, entity_map, seen, unique_ends)  # business property
    entity_map, seen = service_text_assignment(path, entity_map, seen, unique_ends)  # service
    entity_map, seen = location_text_assignment(path, entity_map, seen, unique_ends)  # location

    # Yelp category files from zenodo
    data_path = os.path.join(path, 'data')
    for file in FILE_LIST:
        download_progress(BASE_URL.format(file), data_path, file)

    entity_map, seen = category_text_assighment(data_path, entity_map, seen, unique_ends)

    save_entities(path, {e.index: e for e in entity_map.values()})

    logger.info(f'Found name/description for {len(seen)}/{len(entity_map)} entities')


if __name__ == '__main__':
    args = parser.parse_args()
    download(args.out_path)
