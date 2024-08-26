import dgl.dataloading
from typing import List, Mapping

import torch
from dgl.sampling import RandomWalkNeighborSampler


class BPRSampler(dgl.dataloading.MultiLayerNeighborSampler):
    """
    A sampler that does not build a sampling graph for low computation. E.g., no graph construction is needed for
    TransX methods as it is single layer.
    """
    def __init__(self):
        super().__init__(0)

    def sample(self, g, seed_nodes, exclude_eids=None):
        return seed_nodes, seed_nodes, []


class BPRFullGraphSampler(dgl.dataloading.MultiLayerNeighborSampler):
    """
    A sampler that does not build a sampling graph for low computation. E.g., no graph construction is needed for
    TransX methods as it is single layer.
    """
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)

    def sample(self, g, seed_nodes, exclude_eids=None):
        block = dgl.to_block(g)

        if exclude_eids is not None:
            if isinstance(exclude_eids, dict):
                for key, value in exclude_eids.items():
                    block.remove_edges(value, etype=key, store_ids=True)
            else:
                block.remove_edges(exclude_eids, store_ids=True)

        return seed_nodes, seed_nodes, [block]


class UserBlockSampler(dgl.dataloading.NeighborSampler):
    def __init__(self,
                 users, num_samples, num_heads, **kwargs):
        super().__init__(0, **kwargs)
        self.users = users
        self.num_samples = num_samples
        self.num_heads = num_heads

    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        frontier = g.sample_neighbors(
            seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
            replace=self.replace, output_device=self.output_device,
            exclude_edges=exclude_eids)
        eid = frontier.edata[dgl.EID]
        block = dgl.to_block(frontier, seed_nodes)
        block.edata[dgl.EID] = eid
        blocks.append(block)
        for _ in range(self.num_heads):
            src = torch.randperm(self.users.size(0))
            src = self.users[src[:self.num_samples]]
            dst = torch.repeat_interleave(seed_nodes['user'], src.size(-1))
            src = src.repeat(len(seed_nodes['user']))
            new_g = dgl.graph((src, dst))
            blocks.append(dgl.to_block(new_g, include_dst_in_src=False))

        seed_nodes = block.srcdata[dgl.NID]
        input_nodes = seed_nodes

        return input_nodes, output_nodes, blocks


class TestPinSAGESampler(RandomWalkNeighborSampler):
    def __init__(self, G, ntype, other_type, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, weight_column='weights', bipartite=True):
        metagraph = G.metagraph()
        fw_etype = list(metagraph[ntype][other_type])[0]
        if bipartite:
            super().__init__(G, num_traversals,
                             termination_prob, num_random_walks, num_neighbors,
                             metapath=[fw_etype], weight_column=weight_column)
        else:
            fw_etype = list(metagraph[ntype][other_type])[0]
            bw_etype = list(metagraph[other_type][ntype])[0]
            super().__init__(G, num_traversals,
                             termination_prob, num_random_walks, num_neighbors,
                             metapath=[fw_etype, bw_etype], weight_column=weight_column)


class PinSAGESampler(dgl.dataloading.BlockSampler):
    def __init__(self, g, num_layers, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, **kwargs):
        super().__init__(**kwargs)
        self.sampler = TestPinSAGESampler(g, '_N', '_N', num_traversals, termination_prob,
                                          num_random_walks, num_neighbors, 'alpha')
        self.edge_norm = dgl.nn.EdgeWeightNorm(norm='right')
        self.num_layers = num_layers

    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # hetero_nodes = self._to_hetero(seed_nodes)
        for _ in range(self.num_layers):
            # user_frontier = self.user_sampler(seed_nodes['user'])
            # item_frontier = self.item_sampler(seed_nodes['item'])
            # nodes = torch.cat([seed_nodes['user'] + self.n_entities, seed_nodes['item']])
            frontier = self.sampler(seed_nodes)
            if exclude_eids:
                eid_excluder = dgl.utils.EidExcluder(exclude_eids)
                frontier = eid_excluder(frontier)
            # frontier = dgl.to_homo(frontier)
            block = dgl.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            # block = dgl.to_homo(block)
            block.edata['alpha'] = self.edge_norm(block, block.edata['alpha'].type(torch.float))
            # block.edata['alpha'] = alpha
            blocks.insert(0, block)

        return self.assign_lazy_features([seed_nodes, output_nodes, blocks])


class PinSAGESampler2(dgl.dataloading.BlockSampler):
    def __init__(self, g, num_layers, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, **kwargs):
        super().__init__(**kwargs)
        self.sampler = dgl.sampling.PinSAGESampler(g, 'item', 'user', num_traversals, termination_prob,
                                          num_random_walks, num_neighbors, 'alpha')
        self.edge_norm = dgl.nn.EdgeWeightNorm(norm='right')
        self.num_layers = num_layers

    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # hetero_nodes = self._to_hetero(seed_nodes)
        for _ in range(self.num_layers):
            # user_frontier = self.user_sampler(seed_nodes['user'])
            # item_frontier = self.item_sampler(seed_nodes['item'])
            # nodes = torch.cat([seed_nodes['user'] + self.n_entities, seed_nodes['item']])
            frontier = self.sampler(seed_nodes)
            # frontier = dgl.to_homo(frontier)
            block = dgl.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            # block = dgl.to_homo(block)
            block.edata['alpha'] = self.edge_norm(block, block.edata['alpha'].type(torch.float))
            # block.edata['alpha'] = alpha
            blocks.insert(0, block)

        return self.assign_lazy_features([seed_nodes, output_nodes, blocks])


class HhopPredictionSampler(dgl.dataloading.EdgePredictionSampler):
    def sample(self, g, seed_nodes):  # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.
        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        tails = dgl.sampling.random_walk(
            g,
            seed_nodes,
            metapath=['iu', 'ui'])[0][:, 2]

        pair_graph = dgl.graph(
            (seed_nodes, tails),
            num_nodes=g.number_of_nodes('item'))

        neg_samples = torch.randint(0, g.number_of_nodes('item'), (500, ), device=seed_nodes.device)

        neg_graph = dgl.graph((torch.repeat_interleave(seed_nodes, 500), neg_samples.repeat(len(seed_nodes))),
                              num_nodes=g.number_of_nodes('item'))

        pair_graph, neg_graph = dgl.compact_graphs([pair_graph, neg_graph])
        seed_nodes = pair_graph.ndata[dgl.NID]

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes)

        return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))