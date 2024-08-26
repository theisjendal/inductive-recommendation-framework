# import multiprocessing
from collections import defaultdict
# from concurrent.futures import as_completed

import dgl
import numpy as np
import os
import torch
from torch.nn.functional import one_hot


class SubgraphExtractor:
    def __init__(self, g, ratings, path):
        self.g = g
        self.stored = defaultdict(dict)

        self.f_path = os.path.join(path, 'graphs.pickle')
        # self.extract(ratings)
        self._hits = 0
        self._misses = 0

    # def extract(self, ratings):
    #
    #     if os.path.isfile(self.f_path):
    #         self.stored = load_pickle(self.f_path)
    #
    #     exists = []
    #     for user, item in ratings:
    #         exists.append((items := self.stored.get(user.item())) is not None and item.item() in items)
    #
    #     ratings = ratings[~torch.tensor(exists)]
    #
    #     futures = []
    #     with ProcessPoolExecutor(max_workers=4) as e:
    #         bs = 50
    #         for i in range(0, len(ratings), bs):
    #             batch = ratings[i:i+bs]
    #             futures.append(e.submit(self._user_graphs, batch))
    #
    #         # Purely for progress
    #         for _ in tqdm(as_completed(futures), total=len(futures), desc='Generating subgraphs'):
    #             pass
    #
    #     for f in futures:
    #         for (user, item), g in f.result().items():
    #             self.stored[user.item()][item.item()] = g
    #
    #     if len(futures):
    #         save_pickle(self.f_path, self.stored)

    def _subgraph_extraction_labelling(self, edge):
        hop = 1; sample_ratio = 1.; max_nodes_per_hop = 100
        # 1. neighbor nodes sampling
        u_nodes, v_nodes = torch.as_tensor([edge[0]]), torch.as_tensor([edge[1]])
        u_dist, v_dist = torch.tensor([0]), torch.tensor([0])
        u_visited, v_visited = torch.unique(u_nodes), torch.unique(v_nodes)
        u_fringe, v_fringe = torch.unique(u_nodes), torch.unique(v_nodes)

        for dist in range(1, hop + 1):
            # sample neigh alternately
            u_fringe, v_fringe = self.g.in_edges(v_fringe)[0], self.g.in_edges(u_fringe)[0]
            u_fringe = torch.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
            v_fringe = torch.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
            u_visited = torch.unique(torch.cat([u_visited, u_fringe]))
            v_visited = torch.unique(torch.cat([v_visited, v_fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = torch.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:int(sample_ratio * len(u_fringe))]]
                shuffled_idx = torch.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:int(sample_ratio * len(v_fringe))]]
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    shuffled_idx = torch.randperm(len(u_fringe))
                    u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
                if max_nodes_per_hop < len(v_fringe):
                    shuffled_idx = torch.randperm(len(v_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = torch.cat([u_nodes, u_fringe])
            v_nodes = torch.cat([v_nodes, v_fringe])
            u_dist = torch.cat([u_dist, torch.full((len(u_fringe),), dist, dtype=torch.int64)])
            v_dist = torch.cat([v_dist, torch.full((len(v_fringe),), dist, dtype=torch.int64)])

        nodes = torch.cat([u_nodes, v_nodes])

        # 2. node labeling
        u_node_labels = torch.stack([x * 2 for x in u_dist])
        v_node_labels = torch.stack([x * 2 + 1 for x in v_dist])
        node_labels = torch.cat([u_node_labels, v_node_labels])

        # 3. extract subgraph with sampled nodes
        subgraph = self.g.subgraph(nodes)
        subgraph.ndata['nlabel'] = node_labels

        return subgraph

    def create_graph(self, u, v, hop=1):
        if (item_dict := self.stored.get(u.item())) is None or (subgraph := item_dict.get(v.item())) is None:
            subgraph = self._subgraph_extraction_labelling((u, v))
            subgraph.ndata['nlabel'] = one_hot(subgraph.ndata['nlabel'], (hop + 1) * 2).float()

            # set edge mask to zero as to remove edges between target nodes in training process
            subgraph.edata['edge_mask'] = torch.ones(subgraph.number_of_edges())
            su = subgraph.nodes()[subgraph.ndata[dgl.NID] == u]
            sv = subgraph.nodes()[subgraph.ndata[dgl.NID] == v]
            _, _, target_edges = subgraph.edge_ids([su, sv], [sv, su], return_uv=True)
            subgraph.edata['edge_mask'][target_edges] = 0
            self._misses += 1
        else:
            self._hits += 1

        return subgraph

    def _user_graphs(self, ratings):
        graphs = {}
        for (user, item) in ratings:
            graphs[(user, item)] = self.create_graph(user, item)

        return graphs
