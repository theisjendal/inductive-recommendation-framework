from collections import defaultdict
from typing import List, Tuple

import dgl
import torch
import numpy as np
import dgl.backend as F
from dgl.dataloading.negative_sampler import _BaseNegativeSampler
from loguru import logger

from shared.enums import Sentiment
from shared.meta import Meta
from shared.relation import Relation
from shared.user import User


def dgl_homogeneous(meta: Meta, recommendable_items: List[int], relations: List[Relation] = None,
                    ratings: List[Tuple[int, int, int]] = None,store_edge_types=False, rating_times=None, **kwargs) \
        -> dgl.DGLHeteroGraph:
    """
    Build a dgl homogeneous graph using relations and ratings. Assumes entity and user ids are mapped to non intersecting values.
    :param meta: the meta data.
    :param relations: if parsed to method, builds a graph with relations.
    :param ratings: if parsed to method, builds a graph with ratings.
    :param kwargs: arguments parsed to dgl.
    :return: a dgl graph based on tensors.
    """
    edges = []
    edge_types = []

    if relations is not None:
        for i, relation in enumerate(sorted(relations, key=lambda x: x.index)):
            edges.extend([(head, tail) for head, tail in relation.edges])

            if store_edge_types:
                edge_types.extend([i] * len(relation.edges))

    if ratings is not None:
        positive_rating = meta.sentiment_utility[Sentiment.POSITIVE]
        edges.extend([(user, item) for user, item, rating in ratings if rating is positive_rating])

    if not edges:
        logger.warning('Creating empty graph, No edges as neither relations nor ratings have been passed.')
        return dgl.graph([])

    # Remove edges where src or dst is an item, not used for recommendation (no leakage).
    drop_items = set(meta.items) - set(recommendable_items)
    keep_indices = [i for i, (s, d) in enumerate(edges) if s not in drop_items and d not in drop_items]
    edges = [edges[i] for i in keep_indices]
    if store_edge_types:
        edge_types = [edge_types[i] for i in keep_indices]

    edges = torch.LongTensor(edges)

    # Remove duplicate edges if we do not use edge types.
    if not store_edge_types:
        edges = torch.unique(edges, dim=0)

    edges = edges.T

    g = dgl.graph((edges[0], edges[1]), **kwargs)

    recommendable = torch.full_like(g.nodes(), False, dtype=torch.bool)
    recommendable[recommendable_items] = True
    g.ndata['recommendable'] = recommendable

    if rating_times is not None:
        edata = []
        for s, d in zip(*[t.tolist() for t in g.edges()]):
            if (s, d) in rating_times:
                edata.append(rating_times[(s, d)])
            elif (d, s) in rating_times:
                edata.append(rating_times[(d, s)])
            else:
                edata.append(0)

        edata = torch.tensor(edata)
        g.edata['rating_time'] = edata

    if store_edge_types:
        g.edata['type'] = torch.LongTensor(edge_types).T

    return g


def create_reverse_relations(relations: List[Relation]):
    n_relations = len(relations)
    out_relations = []

    for relation in relations:
        out_relations.append(Relation(relation.index + n_relations, relation.name + '_R',
                                  edges=[(e2, e1) for e1, e2 in relation.edges]))

    return out_relations


def create_rating_relations(meta: Meta, train: List[User], user_fn, sentiments: List[Sentiment] = None):
    n_relations = len(meta.relations)
    relations = []

    train = np.array([[user_fn(user.index), item, rating] for user in train for item, rating in user.ratings]).T

    # Remove unseen ratings
    unseen = meta.sentiment_utility[Sentiment.UNSEEN]
    train = train[:, train[-1] != unseen]

    # For all relevant sentiments add relation.
    for sentiment in sentiments:
        rating_type = meta.sentiment_utility[sentiment]

        mask = train[-1] == rating_type
        data = train[:, mask]

        edges = [(user, item) for user, item, _ in data.T]
        relations.append(Relation(len(relations) + n_relations, sentiment.name, edges))

    return relations


class UniformRecommendableItemSampler(_BaseNegativeSampler):
    def __init__(self, k, edges=None, pop_neg=10, methodology='uniform', share_negative=False, sample_nodes=None):
        self.k = k
        self.pop_neg = pop_neg
        self.share_negative = share_negative
        self.edges = edges
        self.sample_nodes = sample_nodes

        if sample_nodes is not None:
            t = sample_nodes.unique()
            if len(t) > len(sample_nodes):
                logger.warning('Duplicate nodes in sample nodes. Having dubplicates increases probability of selecting'
                               'them as negative samples.')

        self.dst_nodes = None
        assert methodology in ['uniform', 'popularity', 'non-rated', 'topk']
        if methodology != 'uniform' and edges is None:
            logger.warning('Methodology {} requires edges to be passed. Uniform is used instead.'.format(methodology))
        self.methodology = methodology
        self.popularity = None
        if edges is not None:
            dictionary = {}
            src, dst = (e.numpy() for e in edges)
            unique, count = np.unique(src, return_counts=True)
            id_ifc = {nid: idx for idx, nid in enumerate(unique)}
            i = 0
            while i < len(src):
                nid = src[i]
                idx = id_ifc[nid]
                c = count[idx]
                dictionary[nid] = torch.tensor(dst[i:i+c], dtype=edges[0].dtype)
                i += c

            self.edges = dictionary
            self.dst_nodes, self.popularity = torch.unique(edges[1], return_counts=True)
            if self.methodology == 'topk':
                cut_off = (len(self.popularity) // self.pop_neg)
                topk = torch.topk(self.popularity, cut_off)
                mask = torch.ones_like(self.popularity, dtype=torch.bool)
                mask[topk.indices] = False
                self.popularity[mask] = 0

        # self.step()
        self.A = None

    def _generate(self, g, eids, canonical_etype):
        stype, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src_org, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src_org, self.k, 0)

        if self.sample_nodes is not None:
            # selection = torch.zeros_like(g.nodes(vtype), dtype=torch.bool)
            # selection[self.sample_nodes] = True
            selection = self.sample_nodes
        elif len(g.ntypes) > 1:
            selection = g.ndata['recommendable'][vtype]
        else:
            selection = g.ndata['recommendable']

        if self.edges is None or self.methodology == 'uniform':
            recommendable = g.nodes(ntype=vtype)[selection]
            if self.share_negative:
                indices = F.randint((self.k,), dtype, ctx, 0, len(recommendable)).repeat(src_org.shape[0])
            else:
                indices = F.randint(shape, dtype, ctx, 0, len(recommendable))
            dst = recommendable[indices]
        elif self.methodology in ['popularity', 'non-rated', 'topk']:
            if self.sample_nodes is not None:
                raise ValueError(f'Expected samples to be none or methodology to be uniform, got {self.methodology}')

            selection = selection.float()
            if self.methodology in ['popularity', 'topk']:
                selection = self.popularity * selection[self.dst_nodes]

            if self.share_negative:
                probability = torch.ones((1, len(selection)))
            else:
                probability = torch.ones((len(src_org), len(selection)))

            if self.methodology == 'non-rated':
                probability[:, self.dst_nodes] = selection[self.dst_nodes]
                so, so_inv = torch.unique(src_org, return_inverse=True)
                src_rated, dst_rated = g.out_edges(so, etype=canonical_etype)

                # if self.share_negative:
                #     probability[0, dst_rated] = 1e-3  # Set to low value as all dst nodes may be rated
                # else:
                #     # _, inverse = torch.unique(src_rated, return_inverse=True, sorted=False)
                #     # probability[inverse, dst_rated] = 0
                #     # if self.A is None:
                #     #     self.A = g.adj(etype=canonical_etype, scipy_fmt='csr')
                #     #     self.A = self.A[:, self.dst_nodes.numpy()]
                #     # nodes = self.A[src_org.numpy()].nonzero()
                #     # probability[nodes[0], nodes[1]] = 0
                #
                #     nodes = g.nodes(stype)
                #     nodes[src_org] = torch.arange(len(src_org))
                #     probability[nodes[src_rated], dst_rated] = 0

                for i, node in enumerate(src_org):
                    # If we share negative samples, we only need one probability vector.
                    i = 0 if self.share_negative else i
                    probability[i, self.edges[node.item()]] = 0

                probability = probability[so_inv]

            indices = probability.multinomial(num_samples=self.k, replacement=True)
            if self.share_negative:
                indices = indices.repeat(src_org.shape[0], 1)

            if self.methodology in ['popularity', 'topk']:
                indices = self.dst_nodes[indices]
            dst = g.nodes(ntype=vtype)[indices].flatten()
        else:
            raise NotImplementedError

        return src, dst.to(src.device)
