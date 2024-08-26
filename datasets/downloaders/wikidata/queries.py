import time
from collections import defaultdict
from http.client import RemoteDisconnected, IncompleteRead
from json import JSONDecodeError
from typing import List, T, Callable, Iterator
import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON, POST
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from tqdm import tqdm
from urllib3.exceptions import HTTPError

endpoint_url = 'https://query.wikidata.org/sparql'
user_agent = 'cs-project'

uri_query = """SELECT ?label ?film WHERE {{
               ?film wdt:P345 ?label.
               VALUES ?label {{
                  {0}
               }}
               SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}"""

uri_freebase_query = """SELECT ?freebaseid ?entity WHERE {{
               ?entity wdt:P646 ?freebaseid.
               VALUES ?freebaseid {{
                  {0}
               }}
               SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}"""

dbpedia_query = """
SELECT DISTINCT ?dbpediaP ?wikilink ?wikidataP ?label ?description ?comment
WHERE {{
    VALUES ?dbpediaP {{
        {0}
    }}
    OPTIONAL {{
        ?dbpediaP owl:equivalentClass ?wikidataP . 
        FILTER ( strStarts(  str( ?wikidataP ) , str( wikidata: ) ) ) 
    }}
    OPTIONAL {{  # get wikipedia link if it exists
        ?dbpediaP prov:wasDerivedFrom ?wikilink .
        FILTER ( strStarts(  str( ?wikilink ) , str( wikipedia-en: ) ) )  
    }}
    OPTIONAL {{ 
        ?dbpediaP rdfs:label ?label . 
        FILTER (lang(?label) = 'en')
    }}
    OPTIONAL {{
        ?dbpediaP dbo:abstract ?description . 
        FILTER (lang(?description) = 'en')
    }}
    OPTIONAL {{
        ?dbpediaP rdfs:comment ?comment . 
        FILTER (lang(?comment) = 'en')
    }}        
}}                
"""

statement_query = """SELECT * WHERE {{
  {{ 
    ?s ?p ?o.
    VALUES ?s {{
      {0}
    }}
  }}
  UNION
  {{
    ?s ?p ?o.
    VALUES ?o {{
      {0}
    }}
  }}
  FILTER(STRSTARTS(STR(?p), STR(wdt:)))
  FILTER(STRSTARTS(STR(?o), STR(wd:Q)))
  FILTER(STRSTARTS(STR(?s), STR(wd:Q)))
}}"""

simple_statement_query = """SELECT * WHERE {{
  {{ 
    ?s ?p ?o.
    VALUES ?s {{
      {0}
    }}
  }}
  FILTER(STRSTARTS(STR(?p), STR(wdt:)))
  FILTER(STRSTARTS(STR(?o), STR(wd:Q)))
}}"""

entity_literals = """
SELECT ?s ?p ?val WHERE {{
  VALUES ?s {{
    {0}
  }}
  {{
    ?s ?p ?val FILTER( isNumeric(?val)  || (isLiteral(?val) && lang(?val) = 'en') )
  }}
  UNION
  {{
    VALUES ?p {{
      schema:about
    }}
    ?val ?p ?s.
    ?val schema:inLanguage "en" .
    ?val schema:isPartOf <https://en.wikipedia.org/> .
  }}
}}
"""

entity_description = """
SELECT ?s ?sLabel ?sDescription ?sitelink WHERE {{
  VALUES ?s {{
    {0}
  }}
  
  OPTIONAL {{
    ?sitelink schema:about ?s.
    ?sitelink schema:inLanguage "en" .
    ?sitelink schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

predicate_query = """
SELECT DISTINCT ?uri ?pLabel ?pDescription WHERE {{
  VALUES ?uri {{
    {0}
  }}
  ?p wikibase:directClaim ?uri.
  

  SERVICE wikibase:label {{ 
    bd:serviceParam wikibase:language "en". 
  }}
}}
"""


def get_chunks(lst: List[T], chunk_size: int, fn: Callable[[T], str] = None, join_char=' ') -> List[str]:
    n_chunks = (len(lst) // chunk_size) + 1

    chunks = [join_char.join([element if fn is None else fn(element)
                              for element in lst[chunk_size*i: chunk_size*(i+1)]])
              for i in range(n_chunks)]

    return chunks


def get_results(query):
    start = time.time()
    sparql = SPARQLWrapper(endpoint_url)
    sparql.addCustomHttpHeader('User-Agent', user_agent)
    sparql.addCustomHttpHeader('Retry-After', '2')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(POST)
    res = sparql.query().convert()['results']['bindings']
    end = time.time()
    diff = end - start
    if diff < 2.01:
        time.sleep(2.01 - diff)
    return res


def safe_query(chunk: str, query: str, simple_query: str = None) -> list:
    """
    Queries chunk with error handling and returns results without processing.
    :param chunk: hunk of uris.
    :param query: query to call. Uses .format.
    :param simple_query: if original query fails, call with simpler more efficient query.
    :return: results from queries as list.
    """
    results = []

    # Error messages to catch
    errors = (RemoteDisconnected, IncompleteRead, JSONDecodeError, EndPointInternalError, HTTPError)

    # Initial chunk size and current size. Reduced on error.
    initial_chunk_size = len(chunk)
    chunk_size = initial_chunk_size

    successful = False
    inner_chunks = [chunk]  # Start with complete chunk
    skipped = []  # entities requiring simpler queries
    while chunk_size and not successful:
        error = None
        unsuccessfully_queried = []
        for chunk in inner_chunks:
            try:
                results.extend(get_results(query.format(chunk)))
            except errors as e:
                # If chunk size is one, then skip troubling entity and continue.
                if chunk_size == 1:
                    skipped.append(chunk)
                else:
                    # Try chunk again
                    unsuccessfully_queried.append(chunk)
                    error = e

                    if isinstance(e, HTTPError):
                        time.sleep(2)  # Sleep for two second and hope server responds next time

        if not unsuccessfully_queried:
            successful = True
        else:
            chunk_size = chunk_size // 5 if chunk_size >= 5 else chunk_size // 2
            if chunk_size:
                unsuccessfully_queried = ' '.join(unsuccessfully_queried).split(' ')
                inner_chunks = get_chunks(unsuccessfully_queried, chunk_size)
            else:
                # Raise error as we cannot reduce chunksize anymore
                raise error

    if skipped and simple_query is not None:
        for chunk in get_chunks(skipped, initial_chunk_size):
            results.extend(safe_query(chunk, simple_query))

    return results


def get_uris(imdb_ids):
    imdb_ids = list(imdb_ids)
    movie_uri = {}

    for chunk in tqdm(get_chunks(imdb_ids, 5000, fn=lambda x: '"tt%s"' % x), desc='Querying WikiData for movie uris'):
        result = safe_query(chunk, uri_query)
        for r in result:
            movie_uri[r['label']['value'].lstrip('tt')] = r['film']['value']

    return movie_uri


def get_uris_from_freebaseid(freebase_ids):
    freebase_ids = list(freebase_ids)  # ensure of type list
    uris = {}
    for chunk in tqdm(get_chunks(freebase_ids, 5000, fn=lambda x: '"/%s"' % x.replace('.', '/').replace('"', '\\"')), desc='Querying WikiData for movie uris'):
        results = safe_query(chunk, uri_freebase_query)

        for result in results:
            fid = result['freebaseid']['value'][1:].replace('/', '.')

            uris[fid] = result['entity']['value']

    return uris


def get_all_statement_links(uris, disable_tqdm=True):
    spo = {'s': [], 'p': [], 'o': []}
    for outer_chunk in tqdm(get_chunks(uris, 2000, fn=lambda x: 'wd:%s' % x.split('/')[-1]),
                       desc='Getting all related entities', position=0, leave=False, disable=disable_tqdm):
        for result in safe_query(outer_chunk, statement_query, simple_statement_query):
            spo['s'].append(result['s']['value'])
            spo['p'].append(result['p']['value'])
            spo['o'].append(result['o']['value'])

    return pd.DataFrame.from_dict(spo)


def get_entity_literals(uris):
    for chunk in tqdm(get_chunks(list(uris), 5000, fn=lambda x: 'wd:%s' % x.split('/')[-1]),
                      desc='Querying wikidata for entity info'):
        triples = {'s': [], 'p': [], 'o': []}
        for result in safe_query(chunk, entity_literals):
            triples['s'].append(result['s']['value'])
            triples['p'].append(result['p']['value'])
            triples['o'].append(result['val']['value'])
        yield pd.DataFrame.from_dict(triples)


def get_entity_labels(uris) -> Iterator[pd.DataFrame]:
    for chunk in tqdm(get_chunks(list(uris), 5000, fn=lambda x: 'wd:%s' % x.split('/')[-1]),
                      desc='Querying wikidata for entity info'):
        subjects = {'uri': [], 'label': [], 'description': [], 'wikilink': []}
        for results in safe_query(chunk, entity_description):
            subjects['uri'].append(results['s']['value'])
            subjects['label'].append(results.get('sLabel', {}).get('value'))
            subjects['description'].append(results.get('sDescription', {}).get('value'))
            subjects['wikilink'].append(results.get('sitelink', {}).get('value'))

        yield pd.DataFrame.from_dict(subjects)


def get_predicate_labels(uris):
    for chunk in tqdm(get_chunks(list(uris), 5000, fn=lambda x: 'wdt:%s' % x.split('/')[-1]),
                      desc='Querying wikidata for predicate info'):
        subjects = {'uri': [], 'label': [], 'description': []}
        for results in safe_query(chunk, predicate_query):
            subjects['uri'].append(results['uri']['value'])
            subjects['label'].append(results.get('pLabel', {}).get('value'))
            subjects['description'].append(results.get('pDescription', {}).get('value'))

        yield pd.DataFrame.from_dict(subjects)


def dbpedia_safe_query_wrapper(chunk, predicate_query):
    """
    Run safe query, but set and reset global endpoint variable #notAHotfixAtAll.:
    """
    global endpoint_url
    store = endpoint_url
    endpoint_url = 'https://dbpedia.org/sparql'
    result = safe_query(chunk, predicate_query)
    endpoint_url = store
    return result


def get_urls_from_dbpedia(uris):
    for chunk in tqdm(get_chunks(list(uris), 1000, fn=lambda x: "<%s>" % x),
                      desc='Querying dbpedia for entity info'):
        subjects = {'uri': [], 'wikidata': [], 'wikilink': [], 'name': [], 'description':[], 'comment': []}
        for results in dbpedia_safe_query_wrapper(chunk, dbpedia_query):
            subjects['uri'].append(results['dbpediaP']['value'])
            subjects['wikidata'].append(results.get('wikidataP', {}).get('value'))
            subjects['wikilink'].append(results.get('wikilink', {}).get('value'))
            subjects['name'].append(results.get('label', {}).get('value'))
            subjects['description'].append(results.get('description', {}).get('value'))
            subjects['comment'].append(results.get('comment', {}).get('value'))

        yield pd.DataFrame.from_dict(subjects)


