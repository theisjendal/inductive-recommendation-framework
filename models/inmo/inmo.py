import dgl
import dgl.function as fn
import torch
from torch import nn

from models.ngcf.ngcf import NGCF


class INMO(NGCF):
    def __init__(self, g, i_g: dgl.DGLHeteroGraph, n_nodes, in_dim, layer_dims, dropouts, use_cuda=False):
        super().__init__(g, in_dim, layer_dims, [0 for _ in dropouts], lightgcn=True, use_cuda=use_cuda)
        self.train_g = g
        self.inference_g = i_g
        self.features = nn.ModuleDict()
        for ntype in g.ntypes:
            e = nn.Embedding(n_nodes[ntype], in_dim)
            nn.init.normal_(e.weight, std=0.1)
            self.features[ntype] = e

        self.alpha = 1.
        self.delta = 0.99
        # self.W = nn.Linear(in_dim, in_dim, bias=False)
        self.w = nn.Parameter(torch.ones([in_dim]))
        self.dropout = nn.Dropout(dropouts[0])
        g_user = nn.Parameter(torch.ones(1, in_dim))
        g_item = nn.Parameter(torch.ones(1, in_dim))
        nn.init.xavier_normal_(g_user)
        nn.init.xavier_normal_(g_item)
        self.global_ = nn.ParameterDict({'user': g_user, 'item': g_item})


    def update_alpha(self, epoch):
        self.alpha = 1. * (self.delta ** epoch)

    def set_norms(self, g):
        all_degrees = {}
        for src_type, etype, dst_type in g.canonical_etypes:
            d = g.in_degree(g.nodes(dst_type), etype).to(torch.float32)
            if dst_type in all_degrees:
                all_degrees[dst_type] += d
            else:
                all_degrees[dst_type] = d

        for src_type, etype, dst_type in g.canonical_etypes:
            degrees = all_degrees[dst_type].clone()
            degrees[degrees != 0] = torch.pow(degrees[degrees != 0], (self.alpha - 1.) / 2. - 0.5)
            with g.local_scope():
                g.nodes[dst_type].data['norm'] = degrees.unsqueeze(-1)
                norm = g.nodes[dst_type].data['norm'][g[etype].edges()[1]]
            g[etype].edata['norm'] = norm

    def _get_embeddings(self, node_ids, **kwargs):
        g = self.train_g if self.training else self.inference_g
        device = self.features['user'].weight.device
        with g.local_scope():
            for ntype, emb in self.features.items():
                # g.nodes[ntype].data['h'] = emb(g.ndata[dgl.NID][ntype])
                g.nodes[ntype].data['h'] = self.dropout(emb(g.nodes(ntype=ntype)))

            updates = {}
            # Assumes srctype can only be connected to single dsttype and srctype != dsttype
            for srctype, etype, dsttype in g.canonical_etypes:
                g[etype].edata['norm'] = self.dropout(g[etype].edata['norm'])
                g.pull(node_ids[dsttype], fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'h_n'), etype=etype)
            #     updates[etype] = (fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'h'))
            #
            # g.multi_update_all(updates, 'sum')

            return {ntype: g.nodes[ntype].data['h_n'][nids] + self.dropout(self.global_[ntype])
                    for ntype, nids in node_ids.items()}

    def self_enhanced_pred(self, g: dgl.DGLHeteroGraph, x):
        with g.local_scope():
            for ntype, emb in x.items():
                if ntype == 'item':
                    g.nodes[ntype].data['h'] = emb * self.w[None, :]
                else:
                    g.nodes[ntype].data['h'] = emb

            g['ui'].apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g['ui'].edata['score']

    def aux_loss(self, pos_graph, neg_graph):
        embeddings = {ntype: self.dropout(self.features[ntype](nids))
                      for ntype, nids in pos_graph.ndata[dgl.NID].items()}
        pos = self.self_enhanced_pred(pos_graph, embeddings)
        neg = self.self_enhanced_pred(neg_graph, embeddings)
        diff = neg - pos
        s = torch.nn.functional.softplus(diff)
        aux = torch.mean(s)
        return aux
