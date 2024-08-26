import concurrent
from concurrent.futures import ThreadPoolExecutor
from SPARQLWrapper import SPARQLWrapper, JSON, POST

from enum import Enum
from loguru import logger
from tqdm import tqdm

endpoint_url = 'http://virtuoso:8890/sparql'
user_agent = 'cs-project'

"""
This code assumes you have setup a virtuoso instance with the yelp dataset and a dump of schemaorg.
"""

class Prefixes(Enum):
    schema = 'https://schema.org/'
    rdf = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    rdfs = 'http://www.w3.org/2000/01/rdf-schema#'
    yelpvoc = 'https://purl.archive.org/purl/yckg/vocabulary#'
    yelpent = 'https://purl.archive.org/purl/yckg/entity/'
    yelpcat = 'https://purl.archive.org/purl/yckg/categories#'
    wdp = 'https://www.wikidata.org/wiki/Property:'
    wd = 'https://www.wikidata.org/entity/'
    wdt = 'https://www.wikidata.org/prop/direct/'
    skos = 'https://www.w3.org/2004/02/skos/core#'

    def __getattr__(self, attr):
        if attr != '_value_' and attr != '_name_':
            return self.value + attr
        raise AttributeError("%r object has no attribute %r" %
                             (self.__class__.__name__, attr))


###################################################
########### QUERIES FOR GETTING TRIPLES ###########
#region Queries for getting triples

query_service_connections = """
SELECT DISTINCT ?business ?businessType ?p ?o ?p2 ?o2 ?type 
WHERE {{
    # Limit query
    ?business rdf:type schema:LocalBusiness .
    FILTER ( STRENDS ( STR(?business), "{}" ) )
    
    # Find information
    ?business rdf:type ?businessType .
    
    # Not all businesses have a service with true value  
    OPTIONAL {{
        ?business ?p ?o .
        ?o rdf:type ?type .  # get type of object
        FILTER(ISBLANK(?o))  # filter out blank nodes
        ?o ?p2 true .        # get all predicates of blank node where object is true
        ?o ?p2 ?o2 .         # get all object value (always true)
    }}
}}
"""

# Create triples from ?b to ?p using ??? as predicate.
query_business_properties = """
SELECT DISTINCT ?b ?p ?o  WHERE {{
    # Limit query
    ?b rdf:type schema:LocalBusiness .
    FILTER ( STRENDS ( STR(?b), "{}" ) )
    
    # Find information
    ?b ?p ?o .
    FILTER ( isLiteral(?o) )
    FILTER ( STRSTARTS( STR(?p), STR(yelpvoc:) ) )
    FILTER regex(str(?o), "True")
}}
"""

# Create triples with spo.
query_location_triples = """
SELECT DISTINCT ?s ?o  WHERE {{
    # Limit query
    ?s rdf:type schema:LocalBusiness .
    FILTER ( STRENDS ( STR(?s), "{}" ) )
    
    # Find information
    ?s schema:location ?o .
}}
"""

# Create triples with so and relation wdp:P131.
query_location_specifications = """
SELECT DISTINCT ?s ?o WHERE {
    ?s wdp:P131 ?o .
}
"""

# Create two triples one from ?s to ?o using schema:keywords as predicate and one from ?o to ?e using
# skos:relatedMatch as predicate.
query_category_connections = """
SELECT DISTINCT ?s ?o ?e WHERE {{
    # Limit query
    ?s rdf:type schema:LocalBusiness .
    FILTER ( STRENDS ( STR(?s), "{}" ) )
    
    # Find information
    optional {{
        ?s schema:keywords ?o .
    }}
    # optional {{
    #     ?o skos:relatedMatch ?e .
    # }}
}}
"""

user_query = """
SELECT DISTINCT ?u ?rating ?i ?time WHERE {{
    # Limit query
    ?u rdf:type schema:Person .
    FILTER ( STRENDS ( STR(?u), "{}" ) )
    
    # Find information
    ?r schema:author ?u .
    ?r schema:aggregateRating ?rating .
    ?r schema:about ?i .
    ?r schema:dateCreated ?time .
}}
"""

is_business_query = """
SELECT DISTINCT ?b WHERE {
    ?b rdf:type schema:LocalBusiness .
}
"""

# Generic query to get last few characters of URIs. Allows for multiple first queries.
distinct_uri_ends_query = """
SELECT ?substring (count(?s) as ?c) WHERE {
    ?s rdf:type schema:LocalBusiness .
    BIND ( str(?s) as ?uri )
    BIND ( substr(?uri, strlen(?uri)-1) as ?substring )
}
"""
#endregion

################################################
########### QUERIES FOR GETTING TEXT ###########
#region Queries for getting text

business_name_query = """
SELECT DISTINCT ?business ?businessLabel {
    ?business rdf:type schema:LocalBusiness .
    ?business schema:legalName ?businessLabel .
}
"""

business_review_text_query = """
SELECT DISTINCT ?business ?user ?time ?reviewText {{
    # Limit query
    ?business rdf:type schema:LocalBusiness .
    FILTER ( STRENDS ( STR(?business), "{}" ) )
    
    # Find information
    ?review schema:about ?business .
    ?review schema:author ?user .
    ?review schema:dateCreated ?time .
    ?review schema:description ?reviewText .
}}
"""

service_name_query = """
SELECT DISTINCT ?service
WHERE {
    ?b rdf:type schema:LocalBusiness .  # ensure business
    ?b ?p ?o .                          
    ?o rdf:type ?type .                 # get type of object
    FILTER(ISBLANK(?o))                 # ensure is blank node
    ?o ?service true .                  # get all predicates of blank node where object is true
}
"""

property_name_query = """
SELECT DISTINCT ?property {{
    # Limit query
    ?business rdf:type schema:LocalBusiness .
    FILTER ( STRENDS ( STR(?business), "{}" ) )
    
    # Find information
    ?business ?property ?value .
    FILTER ( STRSTARTS( STR(?property), STR(yelpvoc:) ) )
    FILTER ( str(?value) = "True" )
}}
"""

location_query = """
SELECT DISTINCT ?location {
    ?location wdp:P31 ?o .
}
"""


yelp_category_query = """
SELECT DISTINCT ?category {
    ?category rdf:type yelpvoc:YelpCategory.
}
"""

wikidata_related_query = """
SELECT DISTINCT ?wd WHERE {
    ?category rdf:type yelpvoc:YelpCategory .
    ?category skos:relatedMatch ?wd .
    yelpvoc:WikidataCategory skos:Member ?wd .
}
"""

schema_related_query = """
SELECT DISTINCT ?schema WHERE {
    ?category rdf:type yelpvoc:YelpCategory .
    ?category skos:relatedMatch ?schema .
    yelpvoc:SchemaCategory skos:Member ?schema .
}
"""

yelp_category_text_query = """
SELECT DISTINCT ?category ?wd ?schema WHERE {
    ?category rdf:type yelpvoc:YelpCategory.
    OPTIONAL {
        ?category skos:relatedMatch ?wd .
        yelpvoc:WikidataCategory skos:Member ?wd .
    }
    OPTIONAL {
        ?category skos:relatedMatch ?schema .
        yelpvoc:SchemaCategory skos:Member ?schema .
    }
}
"""

schema_label_query = """
PREFIX schema: <http://schema.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?s ?label ?description
WHERE {{
  VALUES ?s {{ {} }}
  ?s rdfs:label ?label .
  OPTIONAL {{ 
    ?s rdfs:comment ?description 
  }}
}}
"""
#endregion

###################################################
########### FUNCTIONS FOR QUERYING DATA ###########


def get_results(query, add_prefixes=True):
    if add_prefixes:
        prefix_str = "\n".join(f'PREFIX {p.name}: <{p.value}>' for p in Prefixes)
        query = prefix_str + '\n' + query
    sparql = SPARQLWrapper(endpoint_url)
    sparql.addCustomHttpHeader('User-Agent', user_agent)
    sparql.addCustomHttpHeader('Retry-After', '2')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(POST)
    res = sparql.query().convert()['results']['bindings']

    if len(res) >= 2 ** 20:
        logger.warning('Not all results have been found')

    return res


def query_iterator(query: str, lst, **tqdm_kwargs):
    futures = []

    # Outcomment for debug purposes.
    # tmp = []
    # for element in tqdm(lst):
    #     q = query.format(element)
    #     tmp.extend(get_results(q))

    with ThreadPoolExecutor(max_workers=64) as e:
        for element in lst:
            q = query.format(element)
            futures.append(e.submit(get_results, q))

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(lst), smoothing=0, **tqdm_kwargs):
            pass

    results = []
    for f in futures:
        results.extend(f.result())

    return results


def result_flattener(function, *args, **kwargs):
    res = function(*args, **kwargs)
    return [{k: v['value'] for k, v in r.items()} for r in tqdm(res)]


def result_dict(function, *args, **kwargs):
    res = function(*args, **kwargs)
    res_d = {k: [] for k in res[0].keys()}
    for r in tqdm(res):
        for k, v in r.items():
            res_d[k].append(v['value'])

    return res_d
