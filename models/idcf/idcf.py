from typing import List

import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn, nn as dglnn
from tqdm import tqdm

from models.shared.dgl_dataloader import UserBlockSampler
from shared.utility import is_debug_mode


class GCMCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_probability=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout_probability)
        self.activation = nn.ReLU()

    def forward(self, g: dgl.DGLGraph, features):
        with g.local_scope():
            srctype, etype, dsttype = g.canonical_etypes[0]
            g.srcnodes[srctype].data['h'] = features[0]  # Edge weight of 1.
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))

            g.dstnodes[dsttype].data['h'] = self.dropout(g.dstnodes[dsttype].data['h'])

            return g.dstnodes[dsttype].data['h']


class RelLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.e1 = nn.Linear(dim, 1, bias=False)
        self.e2 = nn.Linear(dim, 1, bias=False)
        self.norm = dgl.nn.EdgeWeightNorm(norm='right')

    def forward(self, g: dgl.DGLGraph, d_u, p_u):
        with g.local_scope():
            # Eq. 5
            w_k = self.w_k(p_u)
            w_q = self.w_q(d_u)

            # e.T[w_k, w_q] is equivalent to e1.T[w_k] + e2.T[w_q], but the latter is cheaper.
            g.srcdata['h'] = self.e1(w_k).squeeze(-1)
            g.dstdata['h'] = self.e2(w_q).squeeze(-1)
            g.apply_edges(fn.u_add_v('h', 'h', 'c'))
            g.edata['c'] = self.norm(g, g.edata['c'])

            # Eq. 6
            g.srcdata['w_v'] = self.w_v(p_u)
            g.update_all(fn.u_mul_e('w_v', 'c', 'm'), fn.sum('m', 'h'))

            return g.dstdata['h']


class IDCF(nn.Module):
    def __init__(self, g, embedding_dim, nn_layer_dims, num_heads):
        super(IDCF, self).__init__()
        self.out_dim = embedding_dim

        # Define intitial embeddings
        self.features = nn.ModuleDict()
        self.biases = nn.ModuleDict()
        self.fcs = nn.ModuleDict()
        for ntype in g.ntypes:
            n_nodes = g.number_of_nodes(ntype=ntype)
            e = torch.nn.Embedding(n_nodes, embedding_dim)
            torch.nn.init.xavier_uniform(e.weight)
            self.features[ntype] = e
            self.biases[ntype] = nn.Embedding(n_nodes, 1)
            self.fcs[ntype] = nn.Linear(embedding_dim*(len(g.etypes)//2), embedding_dim)  # Combination func, below eq. 12.

        # ------ Define pretrain ------
        self.layer = dglnn.HeteroGraphConv({
            rel: GCMCLayer(embedding_dim, embedding_dim) for rel in g.etypes  # Neighbor average, eq. 12.
        }, 'stack')

        # Define fully connected layers
        self.nn_layers = nn.ModuleList()
        in_dim = embedding_dim * 4
        for out_dim in nn_layer_dims:
            self.nn_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.mf_loss_fn = nn.LogSigmoid()

        # ------ Define adaptation ------
        self.rel_layers = nn.ModuleList()
        for _ in range(num_heads):
            self.rel_layers.append(RelLayer(embedding_dim))

        self.use_user_bias = False

        self.w_o = nn.Linear(num_heads * embedding_dim, embedding_dim, bias=False)

        self.adaptation_fn = nn.LogSoftmax()

        # ------ Inference storage ------
        self.mf_embeddings = None
        self.init_query_embeddings = None
        self.query_embeddings = None

        IDCF.reset_parameters(self)

    @staticmethod
    def reset_parameters(module: nn.Module):
        for m in list(module.modules())[1:]:  # skip self
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def mf_embedder(self, node_ids, block: dgl.DGLGraph):
        x = {ntype: self.features[ntype](nids) for ntype, nids in node_ids.items()}  # get initial embeddings
        with block.local_scope():
            x = self.layer(block, x)

            for ntype, embedding in x.items():
                x[ntype] = self.fcs[ntype](embedding.reshape(len(embedding), -1))  # FC described below eq 12

        return x

    def mf_score(self, users, items, initial, propagated, biases, rank_all=False):
        interactions = []
        n_users = len(users)
        n_items = len(items)

        if rank_all:
            users = torch.repeat_interleave(users, n_items)
            items = items.repeat(n_users)
        else:
            if self.use_user_bias:
                biases['user'] = biases['user'][users]
            biases['item'] = biases['item'][items]

        # EQ 13
        p_u = initial['user'][users]
        q_i = initial['item'][items]
        m_u = propagated['user'][users]
        n_i = propagated['item'][items]

        interactions.append(torch.mul(p_u, q_i))
        interactions.append(torch.mul(p_u, m_u))
        interactions.append(torch.mul(n_i, q_i))
        interactions.append(torch.mul(n_i, m_u))

        x = torch.cat(interactions, dim=-1)

        for i, layer in enumerate(self.nn_layers, 1):
            x = layer(x)
            if i != len(self.nn_layers):
                x = torch.relu(x)

        if rank_all:
            x = x.reshape(-1, n_items)
            biases['item'] = biases['item'].squeeze()

        if self.use_user_bias:
            x += biases['user']
        x += biases['item']

        return x

    def mf_predict(self, g, x, initial=None):
        with g.local_scope():
            users, items = g.edges('uv', etype='ui')
            if initial is None:
                initial = {ntype: self.features[ntype](g.nodes[ntype].data[dgl.NID]) for ntype in g.ntypes}
            biases = {ntype: self.biases[ntype](g.nodes[ntype].data[dgl.NID]) for ntype in g.ntypes}
            x = self.mf_score(users, items, initial, x, biases)

        return x

    def predict(self, users, items, query=False):
        user_indices = torch.arange(len(users), device=users.device)
        item_indices = torch.arange(len(items), device=users.device)
        nids = {'user': users, 'item': items}
        if query:
            initial = {ntype: self.init_query_embeddings[ids] if ntype == 'user' else self.features[ntype](ids)
                       for ntype, ids in nids.items()}
            propagated = {ntype: self.query_embeddings[ids] if ntype == 'user' else self.mf_embeddings[ntype][ids]
                          for ntype, ids in nids.items()}
        else:
            initial = {ntype: self.features[ntype](ids) for ntype, ids in nids.items()}
            propagated = {ntype: self.mf_embeddings[ntype][ids] for ntype, ids in nids.items()}
        biases = {ntype: self.biases[ntype](ids) for ntype, ids in nids.items() if (not query) or ntype != 'user'}
        x = self.mf_score(user_indices, item_indices, initial, propagated, biases, rank_all=True)
        return x.squeeze(-1)

    def regularization_loss(self, g: dgl.DGLGraph):
        loss_reg = 0.
        for ntype, emb in self.features.items():
            loss_reg += torch.sum(torch.sqrt(torch.sum(emb(g.dstnodes[ntype].data[dgl.NID]) ** 2, 1)))

        return loss_reg

    def pretrain_loss_bpr(self, node_ids, pos_graph, neg_graph, block):
        x = self.mf_embedder(node_ids, block)
        pos = self.mf_predict(pos_graph, x)
        neg = self.mf_predict(neg_graph, x)
        loss = -self.mf_loss_fn(pos - neg).mean()
        reg_loss = self.regularization_loss(block)

        return loss, reg_loss

    def pretrain_loss(self, node_ids, pos, neg, block, use_bpr=False):
        x = self.mf_embedder(node_ids, block)
        p_scores = self.mf_predict(pos, x).squeeze()
        n_scores = self.mf_predict(neg, x).squeeze()
        if not use_bpr:
            loss = F.binary_cross_entropy_with_logits(p_scores, torch.ones_like(p_scores), reduction='sum')
            loss += F.binary_cross_entropy_with_logits(n_scores, torch.zeros_like(n_scores), reduction='sum')
        else:
            loss = -self.mf_loss_fn(p_scores - n_scores).sum()

        reg_loss = self.regularization_loss(block)

        return loss, reg_loss, p_scores > n_scores

    def mf_inference(self, g):
        g = g.to(self.features['user'].weight.device)
        self.mf_embeddings = self.mf_embedder({ntype: g.nodes(ntype=ntype) for ntype in g.ntypes}, g)

    def rel_forward(self, node_ids, blocks: List[dgl.DGLGraph]):
        neighborhood = blocks[0]
        # Defined below eq 4 as the sum of interactions
        with neighborhood.local_scope():
            neighborhood.srcnodes['item'].data['h'] = self.mf_embeddings['item'][node_ids['item']]
            functions = {'iu': (fn.copy_u('h', 'm'), fn.sum('m', 'd_u'))}
            neighborhood.multi_update_all(functions, 'sum')
            d_u = neighborhood.dstnodes['user'].data['d_u']

        # Concatenation of EQ 6
        C = []
        for i, layer in enumerate(self.rel_layers, 1):
            block = blocks[i]
            p_u = self.mf_embeddings['user'][block.srcdata[dgl.NID]]
            C.append(layer(block, d_u, p_u))

        p_u_prime = self.w_o(torch.cat(C, dim=-1))
        dst_nids = neighborhood.dstnodes['item'].data[dgl.NID]
        q_i = self.mf_embeddings['item'][dst_nids]

        initial = {'user': d_u, 'item': self.features['item'](dst_nids)}
        features = {'user': p_u_prime, 'item': q_i}

        return initial, features

    def trade_off_loss(self, p_u_ids, p_u_prime_ids, features):
        # EQ 10
        p_u = self.mf_embeddings['user'][p_u_ids]
        p_u_prime = features['user'][p_u_prime_ids]
        similarity = (p_u @ p_u_prime.T).T
        # p_u_ = p_u.unsqueeze(0).repeat(p_u_prime.size(0), 1, 1)
        # p_u_prime_ = p_u_prime.unsqueeze(1).repeat(1, p_u.size(0), 1)
        # similarity = torch.mul(p_u_, p_u_prime_).sum(dim=-1)
        loss = - torch.mean(similarity.diagonal() - torch.logsumexp(similarity, dim=-1))
        return loss

    def adaptation_loss(self, node_ids, pos, neg, blocks, use_bpr=False):
        initial, features = self.rel_forward(node_ids, blocks)

        p_scores = self.mf_predict(pos, features, initial)
        n_scores = self.mf_predict(neg, features, initial)

        p_u_prime_ids, _ = pos.edges('uv', etype='ui')
        p_u_ids = pos.srcnodes['user'].data[dgl.NID][p_u_prime_ids]

        contrastive_loss = self.trade_off_loss(p_u_ids, p_u_prime_ids, features)

        if not use_bpr:
            loss = F.binary_cross_entropy_with_logits(p_scores, torch.ones_like(p_scores), reduction='sum')
            loss += F.binary_cross_entropy_with_logits(n_scores, torch.zeros_like(n_scores), reduction='sum')
        else:
            loss = -self.mf_loss_fn(p_scores - n_scores).sum()

        return loss, contrastive_loss, p_scores > n_scores

    def adaptive_inference(self, g, key_users, num_heads, batch_size, num_samples=None):
        sampler = UserBlockSampler(key_users, num_samples, num_heads)
        device = self.w_o.weight.device

        dataloader = dgl.dataloading.DataLoader(
            g, {'user': g.nodes(ntype='user')}, sampler, batch_size=batch_size // 2, shuffle=False,
            drop_last=False, device=device,
        )

        initial_query = torch.zeros((g.number_of_nodes(ntype='user'), self.out_dim), device=device)
        embedding_query = torch.zeros((g.number_of_nodes(ntype='user'), self.out_dim), device=device)
        for input_nodes, output_nodes, blocks in tqdm(dataloader, desc='Adaptive inference',
                                                      disable=not is_debug_mode()):
            users = blocks[0].dstnodes(ntype='user')
            initial, embedding = self.rel_forward(input_nodes, blocks)
            initial_query[users] = initial['user'][users]
            embedding_query[users] = embedding['user'][users]

        self.init_query_embeddings = initial_query
        self.query_embeddings = embedding_query


