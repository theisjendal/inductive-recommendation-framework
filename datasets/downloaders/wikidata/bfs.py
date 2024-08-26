import datetime
import itertools
import json
import os
import time
from os.path import isfile

import numpy as np
import pandas as pd
from loguru import logger

from datasets.downloaders.wikidata.queries import get_all_statement_links


class BreathFirstSearch:
    def __init__(self, seed_set, depth, chunk_size=4000, query_fn=get_all_statement_links,
                 uri_fn=lambda x: np.concatenate((x['s'].values, x['o'].values))):
        self._visited = set()
        self._queue = [(uri, 0) for uri in seed_set]  # queue that also keeps track of depth.
        self._queue_set = set(seed_set)  # set of uris ensuring entities are not added twice.
        self._chunk_size = chunk_size
        self._depth = depth
        self._processed = 0
        self._queue_name = 'queue.json'
        self._visited_name = 'visited.json'
        self.query_fn = query_fn
        self.uri_fn = uri_fn

        self._load_state()

    def _load_state(self):
        if os.path.isfile(self._queue_name) and os.path.isfile(self._visited_name):
            with open(self._visited_name) as f:
                self._visited = set(json.load(f))

            with open(self._queue_name) as f:
                saved_queue = json.load(f)
                saved_queue_set = {uri for uri, o in saved_queue}

                # Remove already visited movies
                self._queue_set.difference_update(self._visited)
                self._queue = [(uri, depth) for uri, depth in self._queue if uri in self._queue_set]

                # Update with saved queue
                self._queue.extend(saved_queue)
                self._queue_set.update(saved_queue_set)

    def _save_state(self):
        with open(self._queue_name, 'w') as f:
            json.dump(self._queue, f)

        with open(self._visited_name, 'w') as f:
            json.dump(list(self._visited), f)

    def _get_chunk(self):
        depth = self._queue[0][1]  # Get depth

        # Take chunk with same depth.
        chunk = filter(lambda pair: pair[1] == depth, self._queue)
        chunk = list(itertools.islice(chunk, self._chunk_size))

        # Remove chunk from queues.
        del self._queue[:len(chunk)]
        self._queue_set.difference_update({uri for uri, _ in chunk})

        return list(chunk), depth

    def bfs(self, out_file_path):
        running_time = time.time()
        header = not isfile(out_file_path)  # add header if file does not exist.
        i = 1
        triples = None
        iter_times = []
        while self._queue:
            start = time.time()
            chunk, depth = self._get_chunk()

            uris = [u for u, _ in chunk]
            new_triples = self.query_fn(uris)

            self._processed += len(uris)

            # Get newly queried entities entities
            entities = set(self.uri_fn(new_triples))

            # Update visited and get unqueued and unvisited entities.
            self._visited.update(uris)
            unqueued = entities.difference(self._visited).difference(self._queue_set)

            # Update queue and queue set if within depth.
            next_depth = depth + 1
            if next_depth < self._depth:
                self._queue.extend([(e, next_depth) for e in unqueued])
                self._queue_set.update(unqueued)
            else:
                self._visited.update(unqueued)

            if triples is None:
                triples = new_triples
            else:
                triples = pd.concat([triples, new_triples])

            end = time.time()
            iter_times.append(end - start)
            avg = sum(iter_times) / len(iter_times)
            print(f'\rProcessed = {self._processed:7d}, Queue size = {len(self._queue):7d}, Depth = {depth:2d}, '
                  f'Iter = {i:7d}, Avg. iter = {avg:9.5f}s, '
                  f'Est. remaining = {datetime.timedelta(seconds=(avg*(len(self._queue) // self._chunk_size + 1)))}, '
                  f'Running time = {datetime.timedelta(seconds=(end-running_time))}, '
                  f'In-memory triples = {len(triples) if triples is not None else 0}', end='', flush=True)

            if len(triples) >= 10000000:
                print('\nSaving')
                triples.to_csv(out_file_path, header=header, mode='a', index=False)
                header = False
                triples = None
                iter_times = iter_times[-10:]
                self._save_state()
                print('Saved')

            i += 1

        if triples is not None:
            triples.to_csv(out_file_path, header=header, mode='a', index=False)

        self._save_state()
        print('\nBFS finished successfully')

    def get_visited(self) -> set:
        return self._visited

    def has_queue(self) -> bool:
        return len(self._queue_set) > 0

    def clear_state(self):
        if isfile(self._queue_name):
            os.remove(self._queue_name)

        if isfile(self._visited_name):
            os.remove(self._visited_name)
