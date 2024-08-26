from typing import Union, Dict

import dgl
from dgl import function as fn, ops
import torch
from dgl.utils import expand_as_pair
from torch import nn

from models.bert4rec.bert4rec import Transformer


class GInRecConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True, normalize=False, gate_fn=nn.Sigmoid(),
                 gate_type='concat', aggregator='graphsage', attention=None, attention_dim=None,
                 relations=True):
        super().__init__()
        self.activation = activation
        self.normalize = normalize
        self.activation_fn = nn.LeakyReLU()
        self.relations = relations
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gate_fn = gate_fn
        if gate_type in ['concat', 'inner_product', None]:
            self.gate_type = gate_type
        else:
            raise ValueError('Unsupported gate type.')

        self.attention_type = attention
        if attention_dim is None != attention is None:
            raise ValueError('Either pass no attention or attention dim or pass both')
        elif attention == 'gatv2':
            self.W_a = nn.Linear(input_dim*2, attention_dim)
            self.attention_fn = nn.LeakyReLU()

        if aggregator in ['gcn', 'graphsage', 'bi-interaction', 'lightgcn', 'lstm', 'bert', 'variance']:
            self.aggregator = aggregator
        else:
            raise ValueError('Unsupported aggregator type.')

        if aggregator in ['gcn', 'graphsage', 'lstm', 'bert']:
            if aggregator == 'lstm':
                self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
            elif aggregator == 'bert':
                self.bert = Transformer(input_dim, 4, -1)

            # Update input dim
            input_dim = input_dim * 2 if aggregator in ['graphsage', 'lstm', 'bert'] else input_dim
            self.linear = nn.Linear(input_dim, output_dim)

        elif aggregator in ['bi-interaction', 'variance']:
            self.W_1 = nn.Linear(input_dim, output_dim)
            self.W_2 = nn.Linear(input_dim, output_dim)
            if aggregator == 'variance':
                self.W_var = nn.Linear(2*input_dim, input_dim)

    def gates(self, relations, src_name):
        def func(edges):
            object_emb = edges.src[src_name]
            predicates = edges.data['type'] if self.relations else torch.zeros_like(edges.data['type'])
            subject_emb = edges.dst[src_name]

            # Memory optimization, faster if using only dgl.ops.gather_mm on concatenated embeddings instead of
            # dgl.ops.gather_mm on each embedding separately.
            embs = [subject_emb, object_emb]
            rels = []
            start = 0
            for i, emb in enumerate(embs):
                size = emb.shape[-1]
                rels.append(relations[:, start:start+size])

                # If using inner product, the first embedding is used for the subject and object
                if self.gate_type != 'inner_product' or i != 0:
                    start += size

            res = []
            for i, (emb, rel) in enumerate(list(zip(embs, rels)), 1):
                r = dgl.ops.gather_mm(emb, rel, idx_b=predicates)
                res.append(r)

            if self.gate_type == 'concat':
                gates = torch.sum(torch.stack(res, dim=0), dim=0)
            else:
                subject_emb = torch.sum(torch.stack(res[:1] + res[2:], dim=0), dim=0)
                object_emb = torch.sum(torch.stack(res[1:2] + res[2:], dim=0), dim=0)
                gates = torch.mul(subject_emb, object_emb).sum(dim=-1)

            if self.gate_fn is not None:
                gates = self.gate_fn(gates)

            return {'gate': gates}

        return func

    def attention(self, relation_attention, src_name):
        def func(edges):
            object_emb = edges.src[src_name]
            predicates = torch.zeros_like(edges.data['type'])
            subject_emb = edges.dst[src_name]

            a = self.attention_fn(self.W_a(torch.cat((subject_emb, object_emb), dim=-1)))
            a = (relation_attention[predicates] * a).sum(-1)

            return {'attention': a}

        return func

    def lstm_reducer(self, msg, out):
        def func(nodes):
            # Random order of edges
            m = nodes.mailbox[msg]
            m = m[:, torch.randperm(m.shape[1])]
            a = self.lstm(m)
            return {out: a[-1][0].squeeze(0)}  # select last hidden state

        return func

    def bert_reducer(self, msg, out):
        def func(nodes):
            # Random order of edges
            m = nodes.mailbox[msg]
            m = m[:, torch.randperm(m.shape[1])]
            a = self.bert(m)
            return {out: a.mean(dim=1)}
        return func

    def variance_reducer(self, msg, out):
        def func(nodes):
            # Random order of edges
            m = nodes.mailbox[msg]
            if m.shape[1] == 1:
                a = torch.zeros_like(m.squeeze(1))
            else:
                a = m.var(dim=1)
            return {out: a}
        return func

    def _aggregate(self, g, self_name, ego_network_name):
        with g.local_scope():
            if self.aggregator == 'gcn':
                h = self.linear(g.dstdata[ego_network_name])
            elif self.aggregator in ['graphsage', 'lstm', 'bert']:
                c = torch.cat((g.dstdata[self_name], g.dstdata[ego_network_name]), dim=-1)
                h = self.linear(c)
            elif self.aggregator in ['bi-interaction', 'variance']:
                h = self.W_1(g.dstdata[self_name] + g.dstdata[ego_network_name]) + \
                    self.W_2(g.dstdata[self_name] * g.dstdata[ego_network_name])
            else:
                h = g.dstdata[ego_network_name]

            if self.activation and self.aggregator != 'lightgcn':
                h = self.activation_fn(h)

            if self.normalize:
                h = nn.functional.normalize(h)

            return h

    def _message_parsing(self, g, r, a, src_name, dst_name, edge_ids: Union[str, torch.Tensor] = '__ALL__',
                         reduce=fn.mean):
        with g.local_scope():

            flag = self.gate_type is not None

            if flag:
                g.apply_edges(self.gates(r, src_name))

            if self.attention_type is not None:
                g.apply_edges(self.attention(a, src_name), edges=edge_ids)
                g.edata['attention'] = ops.edge_softmax(g, g.edata['attention'])
                if flag:
                    g.edata['scale'] = torch.einsum('a,ab->ab', g.edata['attention'], g.edata['gate'])
                else:
                    g.edata['scale'] = g.edata['attention']

                reduce = fn.sum
            elif flag:
                g.edata['scale'] = g.edata['gate']

            if flag:
                mfunc = fn.u_mul_e(src_name, 'scale', 'm')
            else:
                mfunc = fn.copy_u(src_name,  'm')

            if self.aggregator == 'lstm':
                reduce = self.lstm_reducer
            elif self.aggregator == 'bert':
                reduce = self.bert_reducer

            # For custom aggregators, we need to handle the case where there are no edges.
            # Replacing reducer with builtin func fixes it.
            if self.aggregator in ['lstm', 'bert'] and g.number_of_edges() == 0:
                reduce = fn.sum

            if edge_ids == '__ALL__':
                g.update_all(message_func=mfunc, reduce_func=reduce('m', dst_name))
            else:
                g.send_and_recv(edge_ids, message_func=mfunc, reduce_func=reduce('m', dst_name))

            if self.aggregator == 'variance':
                reduce = self.variance_reducer if g.number_of_edges() > 0 else fn.sum  # same fix as above
                g.update_all(message_func=mfunc, reduce_func=reduce('m', 'var'))
                g.dstdata[dst_name] = self.W_var(torch.cat((g.dstdata[dst_name], g.dstdata['var']), dim=-1))

            return g.dstdata[dst_name]

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, r: torch.Tensor, a: torch.Tensor=None):
        with g.local_scope():
            g.srcdata['h'], g.dstdata['h'] = expand_as_pair(h, g)
            g.dstdata['h_N'] = self._message_parsing(g, r, a, 'h', 'h_N')

            return self._aggregate(g, 'h', 'h_N')

    def get_gate_weight(self, g, h, r, edge_ids: Union[str, torch.Tensor] = '__ALL__'):
        with g.local_scope():
            g.srcdata['h'], g.dstdata['h'] = expand_as_pair(h, g)
            g.apply_edges(self.gates(r, 'h', 'h_l'), edges=edge_ids)
            g.edata['gate'] = g.edata['gate'].norm(2, dim=-1, keepdim=True)
            g.update_all(fn.copy_e('gate', 'm'), fn.sum('m', 'gate'))
            g.apply_edges(fn.e_div_v('gate', 'gate', 'gate'))
            return g.edata['gate']