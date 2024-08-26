from collections import defaultdict
import time

import requests
import urllib.parse
from tqdm import tqdm

from datasets.downloaders.wikidata.queries import get_chunks

description_query = {
    'prop': 'extracts|info',
    'exintro': 1,
    'inprop': 'url'
}

_URL = "https://en.wikipedia.org/w/api.php"

_headers = {
    'User-Agent': 'cs-project'
}

_params = {
    'action': 'query',
    'format': 'json'
}
_sess = requests.Session()
_sess.headers.update(_headers)
_sess.params.update(_params)


def get_results(data):
    start = time.time()
    result = _sess.get(url=_URL, params=data)
    diff = time.time() - start

    if diff < 0.1:
        time.sleep(0.1)

    return result.json()['query']['pages']


def get_text_descriptions(urls, format_fn=lambda x: x.rsplit('wiki/', 1)[-1], search_label='titles'):
    entities = defaultdict(dict)
    cs = 10
    chunks = get_chunks(urls, cs, fn=format_fn, join_char='|')
    tot = 0
    tot_extracted = 0
    desc = 'Querying wikipedia descriptions'
    t = tqdm(enumerate(chunks), total=len(chunks), desc=desc, position=0, leave=False)
    for i, chunk in t:
        res = get_results(dict(description_query, **{search_label: chunk}))
        tot += len(res)
        c_urls = urls[cs*i: cs*(i+1)]
        for _, info in res.items():
            if (url := urllib.parse.unquote(info.get('fullurl', ''))) in c_urls:
                if 'extract' in info:
                    entities[url]['WIKI_DESCRIPTION'] = info['extract']
        yield entities
        tot_extracted += len(entities)
        t.set_description(f'{desc}, {tot_extracted}/{tot}')
        entities = defaultdict(dict)


if __name__ == '__main__':
    for i in get_text_descriptions(['https://en.wikipedia.org/wiki/Inception', 'https://en.wikipedia.org/wiki/Film',
                                    'https://en.wikipedia.org/wiki/Tilo_Prückner']):
        print(i)
