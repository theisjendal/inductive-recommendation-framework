from typing import T

import dgl
import torch
from dgl.ops import edge_softmax
from dgl import function as fn
from dgl.utils import expand_as_pair
from torch import nn

# Based on https://github.com/LunaBlack/KGAT-pytorch/blob/e7305c3e80fb15fa02b3ec3993ad3a169b34ce64/model/KGAT.py#L13
from tqdm import tqdm

from models.shared.dgl_dataloader import BPRSampler


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)  # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)  # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)  # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)  # W2 in Equation (8)
        else:
            raise NotImplementedError

        # Initialize
        if aggregator_type in ['gcn', 'graphsage']:
            nn.init.xavier_normal_(self.W.weight)
        else:
            nn.init.xavier_normal_(self.W1.weight)
            nn.init.xavier_normal_(self.W2.weight)

        self.activation = nn.LeakyReLU()

    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed

        # Equation (3) & (10)
        # DGL: dgl-cu90(0.4.1)
        # Get different results when using `dgl.function.sum`, and the randomness is due to `atomicAdd`
        # Use `dgl.function.sum` when training model to speed up
        # Use custom function to ensure deterministic behavior when predicting
        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'),
                         lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            # (n_users + n_entities, out_dim)
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))
        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            # (n_users + n_entities, out_dim)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))
        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            # (n_users + n_entities, out_dim)
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))
            # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out


class KGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mode='bi-interaction', propagation_mode='attention'):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self._mode = mode
        self.propagation_mode = propagation_mode

        self.message_dropout = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

        if mode == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)  # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)  # W2 in Equation (8)

            # initialize
            nn.init.xavier_normal_(self.W1.weight)
            nn.init.xavier_normal_(self.W2.weight)
        else:
            raise NotImplementedError

    def forward(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(x, g)
            g.srcdata['emb'] = feat_src
            g.dstdata['emb'] = feat_dst

            if self.propagation_mode == 'attention':
                g.update_all(fn.u_mul_e('emb', 'a', 'm'), fn.sum('m', 'h_n'))
            elif self.propagation_mode == 'gates':
                g.update_all(fn.u_mul_e('emb', 'g', 'm'), fn.sum('m', 'h_n'))

            if self._mode == 'bi-interaction':
                out = self.activation(self.W1(g.dstdata['emb'] + g.dstdata['h_n'])) + \
                      self.activation(self.W2(g.dstdata['emb'] * g.dstdata['h_n']))

        return self.message_dropout(out)


class KGAT(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_entities, n_relations, entity_dim, relation_dim,
                 n_layers, layer_dims, dropout=0., user_fn=None, use_cuda=False, mode='attention'):
        super(KGAT, self).__init__()
        self.use_cuda = use_cuda
        self.mode = mode

        # Define embedding
        self.entity_embed = nn.Embedding(n_entities, entity_dim)
        self.relation_embed = nn.Embedding(n_relations, relation_dim)
        self.W_r = nn.Parameter(torch.Tensor(n_relations, entity_dim, relation_dim))

        # Initialize
        nn.init.xavier_normal_(self.entity_embed.weight)
        nn.init.xavier_normal_(self.relation_embed.weight)
        nn.init.xavier_normal_(self.W_r)

        self.user_fn = (lambda x: x + n_entities) if user_fn is None else user_fn

        self.tanh = nn.Tanh()

        # Must have an output dim for each layer
        assert n_layers == len(layer_dims)

        if mode == 'gates':
            # All dimensions must be the same for gates to work.
            assert all(d == relation_dim for d in layer_dims)
            assert entity_dim == relation_dim
            self.gate_head_weight = nn.Parameter(torch.Tensor(relation_dim, relation_dim))
            self.gate_tail_weight = nn.Parameter(torch.Tensor(relation_dim, relation_dim))
            self.gate_relation_weight = nn.Parameter(torch.Tensor(relation_dim, relation_dim))
            self.gate_bias = nn.Parameter(torch.Tensor(relation_dim))
            self.sigmoid = nn.Sigmoid()

        layers = nn.ModuleList()
        out_dim = entity_dim
        for layer in range(n_layers):
            in_dim = out_dim
            out_dim = layer_dims[layer]
            layers.append(KGATLayer(in_dim, out_dim, dropout, propagation_mode=self.mode))

        self.layers = layers

        # sampler = dgl.dataloading.neighbor.MultiLayerFullNeighborSampler(3)
        # self.collator = dgl.dataloading.NodeCollator(graph, graph.nodes(), sampler)

        self.embeddings = None

    def _trans_r_fn(self, edges):
        r = edges.data['type']
        relation = self.relation_embed(r)

        # Transforming head and tail into relation space
        tail = torch.einsum('be,ber->br', edges.src['emb'], self.W_r[r])
        head = torch.einsum('be,ber->br', edges.dst['emb'], self.W_r[r])

        # Calculate plausibilty score, equation 1
        return {'ps': torch.norm(head + relation - tail, 2, dim=1).pow(2)}

    def trans_r(self, g: dgl.DGLGraph):
        with g.local_scope():
            g.ndata['emb'] = self.entity_embed(g.nodes())

            g.apply_edges(self._trans_r_fn)

            return g.edata['ps']

    def l2_loss(self, pos_graph, neg_graph, embeddings, trans_r=False):
        users, items_i = pos_graph.edges()
        _, items_j = neg_graph.edges()
        loss = embeddings[users].pow(2).norm(2) + \
               embeddings[items_i].pow(2).norm(2) + \
               embeddings[items_j].pow(2).norm(2)

        if trans_r:
            loss += self.relation_embed(pos_graph.edata['type']).pow(2).norm(2)

        loss /= len(users)

        return loss

    def _attention(self, edges):
        r = edges.data['type']
        relation = self.relation_embed(r)

        tail = torch.bmm(edges.src['emb'].unsqueeze(1), self.W_r[r]).squeeze()
        head = torch.bmm(edges.dst['emb'].unsqueeze(1), self.W_r[r]).squeeze()

        # Calculate attention
        return {'a': torch.sum(tail * self.tanh(head + relation), dim=-1)}

    def compute_attention(self, g: dgl.DGLGraph):
        device = self.W_r.device
        with g.local_scope():
            sampler = BPRSampler()
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, prefetch_labels=['type']
            )
            dataloader = dgl.dataloading.EdgeDataLoader(
                g, g.all_edges('eid').to(device), sampler,
                shuffle=True, drop_last=False, use_uva=self.use_cuda, device=device, batch_size=8192
            )
            attention = torch.zeros(len(g.edges('eid')), device=device)

            for input_nodes, batched_g, _ in tqdm(dataloader):
                with batched_g.local_scope():
                    batched_g.ndata['emb'] = self.entity_embed(input_nodes)

                    batched_g.apply_edges(self._attention)

                    attention[batched_g.edata[dgl.EID]] = batched_g.edata['a']

        with g.local_scope():
            g.edata['a'] = attention.cpu()
            return edge_softmax(g, g.edata['a'])

    def _gates(self, edges):
        r = edges.data['type']
        relation = self.relation_embed(r)

        tail = torch.bmm(edges.src['emb'].unsqueeze(1), self.W_r[r]).squeeze()
        head = torch.bmm(edges.dst['emb'].unsqueeze(1), self.W_r[r]).squeeze()

        out = head.matmul(self.gate_head_weight) + tail.matmul(self.gate_tail_weight) + \
              relation.matmul(self.gate_relation_weight) + self.gate_bias

        return {'g': self.sigmoid(out)}

    def compute_gates(self, g: dgl.DGLGraph):
        dataloader = dgl.dataloading.EdgeDataLoader(g, g.edges('eid'),
                                                    dgl.dataloading.MultiLayerFullNeighborSampler(1, return_eids=True),
                                                    batch_size=1024, drop_last=False)

        gates = torch.zeros((len(g.edges('eid')), self.W_r.shape[-1]), device=self.W_r.device)

        for input_nodes, batched_g, blocks in dataloader:
            if self.use_cuda:
                batched_g = batched_g.to('cuda:0')

            with batched_g.local_scope():
                batched_g.ndata['emb'] = self.entity_embed(batched_g.nodes())

                batched_g.apply_edges(self._gates)

                gates[batched_g.edata[dgl.EID]] = batched_g.edata['g']

        with g.local_scope():
            g.edata['g'] = gates.cpu()
            return edge_softmax(g, g.edata['g'])

    def embedder(self, blocks):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        x = self.entity_embed(blocks[0].srcdata[dgl.NID])
        iterator = zip(blocks, self.layers)

        n_out = len(output_nodes)
        embs = [x[:n_out]]
        for block, layer in iterator:
            x = layer(block, x)
            embs.append(x[:n_out])

        return torch.cat(embs, dim=1)

    def predict(self, g: dgl.DGLGraph, embeddings):
        with g.local_scope():
            users, items = g.edges()

            return (embeddings[users] * embeddings[items]).sum(dim=1)

    def forward(self, users, items, rank_all=False):
        users = self.user_fn(users)
        entities = torch.cat([users, items])
        embeddings = self.embeddings[entities]

        user_embs = embeddings[:len(users)]
        item_embs = embeddings[len(users):]

        if rank_all:
            predictions = torch.matmul(user_embs, item_embs.T)
        else:
            predictions = (user_embs * item_embs).sum(dim=1)

        return predictions

    def inference(self, g, batch_size=1024, device='cpu'):
        x = self.entity_embed.weight
        embeddings = [x]

        node_data_loader = dgl.dataloading.DataLoader(
            g, torch.arange(g.number_of_nodes()), graph_sampler=dgl.dataloading.MultiLayerFullNeighborSampler(1),
            batch_size=batch_size, shuffle=True, drop_last=False, device=device
        )
        for layer in self.layers:
            cur_x = torch.zeros((x.shape[0], layer.out_dim), dtype=x.dtype, device=x.device)
            for input_nodes, output_nodes, blocks in node_data_loader:
                 cur_x[output_nodes] = layer(blocks[0], x[input_nodes])
            x = cur_x
            embeddings.append(x)

        self.embeddings = torch.cat(embeddings, dim=1)

    def train(self: T, mode: bool = True) -> T:
        super(KGAT, self).train(mode)

        self.embeddings = None

        return self

