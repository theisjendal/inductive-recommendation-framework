import argparse
import ast
import functools
import gzip
import os
import pickle

from collections import defaultdict
from io import StringIO

import pandas as pd
import shutil
from loguru import logger
from rdflib import Graph, Namespace, RDF
from tqdm.auto import tqdm

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, save_relations, save_users, load_entities, load_relations, \
    load_users

from datasets.downloaders.yelp_virtuoso import queries

from datasets.converters.ab_converter_og import prune

parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs=1, type=valid_dir, default='../yelpkg',
                    help='Yelp KG dataset path')


# def parse(path):
#     try:
#         g = open(path, 'r')
#         n_lines = int(os.popen(f'wc -l {path}').read().split(' ')[0])
#     except OSError:
#         raise OSError('Must run ab-converter-og.py before this file.')
#     for i, l in tqdm(enumerate(g), desc='Loading reviews', total=n_lines):
#         yield ast.literal_eval(l)  # json.loads(l)
#
#
# def getDF(path, items):
#     descriptions = {}
#     item_reviews = defaultdict(list)
#     for d in parse(path):
#         if item := d.get('asin'):
#             descriptions[item] = d
#
#     for item in tqdm(items):
#         if desc := descriptions.get(item):
#             if text := desc.get('title'):
#                 item_reviews[item].append(text)
#             if text := desc.get('description'):
#                 item_reviews[item].append(text)
#     return item_reviews


def get_businesses(data_files):
    # get entities with category local business
    query = """
    select distinct ?s where {
      ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://schema.org/LocalBusiness> .
    }
    """
    qres = data_files['yelp_business'].query(query)
    res = [str(r.s) for r in qres]
    return res


def get_triples(g: Graph, wdp, schema, skos):
    """
    Queries graph for all relevant triples based on different queries.
    :param g: the rdflib graph for querying. Only supports Yelp KG.
    :return: dataframe with all relevant triples. Not equal to Yelp KG.
    """

    logger.info('Getting triples')

    # Create triples from ?s to ?p2 using ?p as predicate, and set ?type as type of ?p2
    query_service_connections = """
    SELECT DISTINCT ?s ?p ?o ?p2 ?o2 ?type 
    WHERE {
        ?s ?p ?o .
        ?o rdf:type ?type .  # get type of object
        FILTER(ISBLANK(?o))  # filter out blank nodes
        ?o ?p2 true .        # get all predicates of blank node where object is true
        ?o ?p2 ?o2 .         # get all object value (always true)
    }
    """

    # Create triples from ?s to ?type using rdf:type as predicate and create triples from ?type to
    # ?o using ?p as predicate.
    query_type_information = """
    select distinct ?s ?type ?p ?o where {
        ?s rdf:type ?type .
        optional {
            ?type ?p ?o .
        }
        FILTER(!ISBLANK(?s)) .
        FILTER(?type != schema:UserReview) .
    }
    """

    # Create triples from ?b to ?p using ??? as predicate.
    query_business_properties = """
    select distinct ?b ?p ?o  where {
        ?b rdf:type schema:LocalBusiness .
        ?b ?p ?o .
        FILTER NOT EXISTS { 
            ?o rdf:type [] . 
        }
        Filter(strstarts(str(?p),str(yelpvoc:))) .
        filter regex(str(?o), "True")
    }
    """

    # Create triples with spo.
    query_location_triples = """
    select distinct ?s ?p ?o  where {
       ?s ?p ?o .
       filter (?p = schema:location)
    }
    """

    # Create triples with so and relation wdp:P131.
    query_location_specifications = """
    select distinct ?s ?o where {
        ?s wdp:P131 ?o .
    }
    """

    # Create two triples one from ?s to ?o using schema:keywords as predicate and one from ?o to ?e using
    # skos:relatedMatch as predicate.
    query_category_connections = """
    select distinct ?s ?o ?e where {
       ?s schema:keywords ?o .
       ?o myskos:relatedMatch ?e .
    }
    """

    # Query for all triples with the above defined queries.
    pbar = tqdm(total=12, desc='Querying graph')
    query_service_res = g.query(query_service_connections); pbar.update(1)
    query_type_res = g.query(query_type_information); pbar.update(1)
    query_business_res = g.query(query_business_properties); pbar.update(1)
    query_location_triples_res = g.query(query_location_triples); pbar.update(1)
    query_location_specifications_res = g.query(query_location_specifications); pbar.update(1)
    query_category_res = g.query(query_category_connections); pbar.update(1)

    # Create triples from query results.
    triples = set()
    pbar.set_description('Creating triples')

    # Service triples
    for r in query_service_res:
        s, p, p2, type = str(r.s), str(r.p), str(r.p2), str(r.type)
        triples.add((s, p, p2))
        triples.add((p2, RDF.type.replace('http', 'https'), type))
    pbar.update(1)

    # Type triples
    for r in query_type_res:
        s, type, p, o = str(r.s), str(r.type), str(r.p), str(r.o)
        triples.add((s, RDF.type.replace('http', 'https'), type))
        if p != 'None' and o != 'None':
            triples.add((type, p, o))
    pbar.update(1)

    # Business triples
    for r in query_business_res:
        b, p = str(r.b), str(r.p)
        triples.add((b, 'business_property', p))
    pbar.update(1)

    # Location triples
    for r in query_location_triples_res:
        s, p, o = str(r.s), str(r.p), str(r.o)
        triples.add((s, p, o))
    pbar.update(1)

    # Location specifications triples
    for r in query_location_specifications_res:
        s, o = str(r.s), str(r.o)
        triples.add((s, str(wdp.P131), o))
    pbar.update(1)

    # Category triples
    for r in query_category_res:
        s, o, e = str(r.s), str(r.o), str(r.e)
        triples.add((s, str(schema.keywords), o))
        triples.add((o, str(skos.relatedMatch), e))
    pbar.update(1)

    df = pd.DataFrame(triples, columns=['s', 'p', 'o'])
    df.drop_duplicates(inplace=True)

    pbar.close()
    return df


def virtuoso_triples_loader(unique_ends):
    triples = []

    # Create triples from query results.
    triples = set()

    # Service triples
    service_res = queries.result_flattener(queries.query_iterator, queries.query_service_connections,
                                           unique_ends, desc='Service query')
    for r in service_res:
        s, stype, p, p2, type = [r.get(x) for x in ['business', 'businessType', 'p', 'p2', 'type']]

        # set type of business
        triples.add((s, queries.Prefixes.rdf.type, stype))

        # Not all businesses has a type
        if p is not None:
            triples.add((s, p, p2))
            triples.add((p2, queries.Prefixes.rdf.type, type))


    # Business triples
    business_res = queries.query_iterator(queries.query_business_properties, unique_ends, desc='Business query')
    for r in business_res:
        b, p = r['b']['value'], r['p']['value']
        triples.add((b, queries.Prefixes.yelpvoc.businessProperty, p))

    # Location triples
    location_triples_res = queries.query_iterator(queries.query_location_triples, unique_ends, desc='Location query')
    for r in location_triples_res:
        s, o = r['s']['value'], r['o']['value']
        triples.add((s, queries.Prefixes.schema.location, o))

    # Location specifications triples
    logger.info('Location specifications query')
    location_specifications_res = queries.get_results(queries.query_location_specifications)
    for r in location_specifications_res:
        s, o = r['s']['value'], r['o']['value']
        triples.add((s, queries.Prefixes.wdp.P131, o))

    # Category triples
    category_res = queries.result_flattener(queries.query_iterator, queries.query_category_connections, unique_ends,
                                       desc='Category query')
    for r in category_res:
        s, o, e = r['s'], r.get('o'), r.get('e')
        if o is not None:
            triples.add((s, queries.Prefixes.schema.keywords, o))
        if e is not None:
            triples.add((o, queries.Prefixes.skos.relatedMatch, e))

    df = pd.DataFrame(triples, columns=['s', 'p', 'o'])
    df.drop_duplicates(inplace=True)

    return df


def create_entities(triples: pd.DataFrame, unique_ends):
    logger.info('Creating entities')

    res = queries.get_results(queries.is_business_query)
    all_businesses = {str(r['b']['value']) for r in res}

    # Create entities, by stacking subject and object and removing duplicates.
    entities = {}
    for index, e in enumerate(tqdm(pd.unique(triples[['s', 'o']].values.ravel('K')), desc='Creating entities')):
        # Add subject and object to entities
        recommendable = e in all_businesses
        entities[e] = Entity(index, name=e, original_id=e, recommendable=recommendable)

    # Reindex entities with items first and then original id for deterministic indexing.
    entities = {i: e for i, e in enumerate(sorted(entities.values(), key=lambda x: (x.recommendable, x.original_id),
                                                  reverse=True))}
    for i, e in entities.items():
        e.index = i

    return entities


def create_relations(df, entities):
    logger.info('Creating relations')
    reverse_map = {v.original_id: k for k, v in entities.items()}

    relations = {}
    for group, (uri, df_r) in enumerate(tqdm(sorted(df.groupby('p')))):
        edges = df_r[['s', 'o']].applymap(reverse_map.get).to_records(index=False).tolist()
        relations[group] = Relation(group, uri, edges, uri)

    return relations


def create_users(df, entities, unique_ends):
    logger.info('Creating users')
    reverse_map = {v.original_id: k for k, v in entities.items()}

    # Get ratings
    res = queries.query_iterator(queries.user_query, unique_ends, desc='User query')
    user_df = pd.DataFrame([(r['u']['value'], int(r['rating']['value']), r['i']['value'], r['time']['value'])
                            for r in res], columns=['u', 'rating', 'i', 't'])

    # Convert item ids to index and convert time to integer
    user_df['i'] = user_df['i'].apply(reverse_map.get)
    user_df['t'] = pd.to_datetime(user_df['t']).apply(lambda x: x.timestamp()).astype(int)

    # Ensure all businesses are present
    len_before = len(user_df)
    user_df.dropna(inplace=True)
    # assert len_before == len(user_df), 'Some entities have been removed, should be handled'

    # Create users, by grouping by users and sorting for deterministic indexing.
    user_df['i'] = user_df['i'].astype(int)
    users = {}
    for i, (user, df_u) in enumerate(tqdm(sorted(user_df.groupby('u')), desc='Creating users')):
        ratings = df_u[['i', 'rating']].to_records(index=False).tolist()
        times = df_u[['i', 't']].to_records(index=False).tolist()
        users[i] = User(user, i, ratings, rating_time=times)

    return users


def run(path, prune_all=False, run_all=False):
    data_path = os.path.join(path, 'data')

    if prune_all:
        for file_path in ['data/triples.csv', 'entities.pickle', 'relations.pickle', 'users.pickle','data/graph.pkl']:
            file_path = os.path.join(path, file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Find all files in data folder with nt.gz extension
    # nt_files = [f for f in os.listdir(data_path) if f.endswith('.nt.gz')]  #and
                #f not in ['yelp_user.nt.gz', 'yelp_checkin.nt.gz', 'yelp_tip.nt.gz']]

    # Decompress all gzipped files
    # with tqdm(nt_files, desc='Decompressing files') as pbar:
    #     skipped = 0
    #     for file in pbar:
    #         in_path = os.path.join(data_path, file)
    #         out_path = os.path.join(data_path, file.rsplit('.', 1)[0])
    #         if not os.path.isfile(out_path):
    #             with gzip.open(in_path, 'rb') as f_in:
    #                 with open(out_path, 'wb') as f_out:
    #                     shutil.copyfileobj(f_in, f_out)
    #         else:
    #             skipped += 1
    #
    #         if skipped:
    #             pbar.set_description(f'Decompressing files ({skipped}/{len(nt_files)} skipped)')

    # Get unique node id's ending letters for batched queries. These are shared for all yelpents.
    unique_ends = [r['substring']['value'] for r in queries.get_results(queries.distinct_uri_ends_query)]

    triple_file = os.path.join(data_path, 'triples.csv')
    if run_all or not os.path.isfile(triple_file):
        triples = virtuoso_triples_loader(unique_ends)
        triples.to_csv(triple_file, index=False)
    else:
        logger.warning('Loading triples from file, delete triples.csv to recompute')
        triples = pd.read_csv(triple_file, index_col=None)

    # Create entities
    try:
        if run_all:
            raise FileNotFoundError
        entities = load_entities(path)
        entities = {e.index: e for e in entities}
        logger.warning('Loaded entities from file, delete entities.pickle to recompute')
    except FileNotFoundError:
        entities = create_entities(triples, unique_ends)
        save_entities(path, entities)

    # Create relations
    try:
        if run_all:
            raise FileNotFoundError
        relations = load_relations(path)
        relations = {r.index: r for r in relations}
        logger.warning('Loaded relations from file, delete relations.pickle to recompute')
    except FileNotFoundError:
        relations = create_relations(triples, entities)
        save_relations(path, relations)

    # Create users
    try:
        if run_all:
            raise FileNotFoundError
        users = load_users(path)
        users = {u.index: u for u in users}
        logger.warning('Loaded users from file, delete users.pickle to recompute')
    except FileNotFoundError:
        users = create_users(triples, entities, unique_ends)
        save_users(path, users)


if __name__ == '__main__':
    args = parser.parse_args()

    # Clone and follow the guide at https://github.com/MadsCorfixen/The-Yelp-Collaborative-Knowledge-Graph to extract the yelp KG.
    # Extract all the *.nt.gz files to *.nt files.

    # While in the cloned repo, do the following:
    # Set up virtuoso using https://hub.docker.com/r/openlink/virtuoso-opensource-7/
    # Create folder /database.

    # Run with `docker run -it --name virtuoso --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --mount type=bind,source="$(pwd)"/data,dst=/import --mount type=bind,src="$(pwd)"/database,dst=/database --publish 1111:1111 --publish 8890:8890 openlink/virtuoso-opensource-7`

    # Run `docker exec -it virtuoso /bin/bash`. Use `cd ..` and run the following commands to load the data:
    # /opt/virtuoso-opensource/bin/isql 1111
    # ld_dir('/import', '*.nt', 'http://localhost:8890/dataspace');
    # rdf_loader_run();
    # checkpoint;

    # You are now ready to run this file. Possibly edit virtuoso.ini file and restart server if queries time out or
    # if they are returning subsets of the dataset - use CQ for validation located in:
    # https://github.com/MadsCorfixen/The-Yelp-Collaborative-Knowledge-Graph/blob/main/Code/CompetencyQuestions/Yelp_CQ.md

    run(args.path, False, run_all=True)
