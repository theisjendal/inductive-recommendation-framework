"""
Based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/NGCF modified to include LightGCN and using blocks.
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout, lightgcn=False):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lightgcn = lightgcn

        # Only use weights if not lightgcn
        if not self.lightgcn:
            #weights for different types of messages
            self.W1 = nn.Linear(in_size, out_size, bias=True)
            self.W2 = nn.Linear(in_size, out_size, bias=True)

            #leaky relu
            self.leaky_relu = nn.LeakyReLU(0.2)

            #initialization
            torch.nn.init.xavier_uniform_(self.W1.weight)
            torch.nn.init.constant_(self.W1.bias, 0)
            torch.nn.init.xavier_uniform_(self.W2.weight)
            torch.nn.init.constant_(self.W2.bias, 0)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        with g.local_scope():
            funcs = {} #message and reduce functions dict
            #for each type of edges, compute messages and reduce them all
            for srctype, etype, dsttype in g.canonical_etypes:
                name = f'h_n^{etype}'
                if srctype == dsttype and not self.lightgcn: #for self loops
                    messages = self.W1(feat_dict[srctype])
                    g.nodes[srctype].data[name] = messages   #store in ndata
                    funcs[(srctype, etype, dsttype)] = (fn.copy_u(name, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
                elif srctype == dsttype and self.lightgcn: # skip self loops
                    continue
                else:
                    src, dst = g.edges(etype=(srctype, etype, dsttype))

                    if not self.lightgcn:
                        if srctype == dsttype:
                            messages = self.W1(self.W1(feat_dict[srctype][src]))
                        else:
                            messages = g.edata['norm'][(srctype, etype, dsttype)] \
                                       * (self.W1(feat_dict[srctype][src]) +
                                          self.W2(feat_dict[srctype][src]*feat_dict[dsttype][dst])) #compute messages
                        g.edges[(srctype, etype, dsttype)].data[name] = messages  #store in edata
                        funcs[(srctype, etype, dsttype)] = (fn.copy_e(name, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
                    else:
                        # E.q 3 in lgcn paper.
                        # messages = g.edata['norm'][(srctype, etype, dsttype)] * feat_dict[srctype][src]
                        g.srcnodes[srctype].data['h'] = feat_dict[srctype]
                        funcs[(srctype, etype, dsttype)] = (fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'h'))

            g.multi_update_all(funcs, 'sum') #update all, reduce by first type-wisely then across different types
            feature_dict={}
            for ntype in g.dsttypes:
                h = g.dstnodes[ntype].data['h']
                if not self.lightgcn:
                    h = self.leaky_relu(h) #leaky relu

                h = self.dropout(h) #dropout

                if not self.lightgcn:
                    h = F.normalize(h,dim=1,p=2) #l2 normalize

                feature_dict[ntype] = h
            return feature_dict


class NGCF(nn.Module):
    def __init__(self, g, in_dim, layer_dims, dropouts, lightgcn=False, use_cuda=False):
        super(NGCF, self).__init__()
        self.layer_dims = layer_dims
        self.use_cuda = use_cuda
        self.lightgcn = lightgcn

        assert len(layer_dims) == len(dropouts)

        self.features = nn.ModuleDict()
        for ntype in g.ntypes:
            e = torch.nn.Embedding(g.number_of_nodes(ntype=ntype), in_dim)
            torch.nn.init.normal_(e.weight, std=0.1)
            self.features[ntype] = e

        self.layers = nn.ModuleList()
        self.layers.append(NGCFLayer(in_dim, layer_dims[0], dropouts[0], lightgcn=lightgcn))

        for i in range(len(layer_dims) - 1):
            self.layers.append(NGCFLayer(layer_dims[i], layer_dims[i+1], dropouts[i+1], lightgcn=lightgcn))

        self.loss_fn = torch.nn.LogSigmoid()

        self.embeddings = None
        self._final_dim = layer_dims[-1] if lightgcn else sum(layer_dims)

    def _get_embeddings(self, node_ids, **kwargs):
        x = {ntype: self.features[ntype](nids) for ntype, nids in node_ids.items()} # get initial embeddings
        return x

    def block_embedder(self, node_ids, blocks, **kwargs):
        prior_kwargs = kwargs.get('prior_kwargs', {})
        posterior_kwargs = kwargs.get('posterior_kwargs', {})
        dst_nodes = {ntype: blocks[-1].dstnodes(ntype) for ntype in set(blocks[-1].ntypes)} # Get final nodes
        x = self._get_embeddings(node_ids, **prior_kwargs)
        embeddings = {ntype: [embedding[dst_nodes[ntype]]] for ntype, embedding in x.items()}  # get first embeddings

        for i, (block, layer) in enumerate(zip(blocks, self.layers)):
            x = layer(block, x)
            for ntype, embedding in x.items():
                embeddings[ntype].append(embedding[dst_nodes[ntype]])

        embeddings = self._post_process(embeddings, dst_nodes=dst_nodes, **posterior_kwargs)

        return embeddings

    def _post_process(self, embeddings, dst_nodes, **kwargs):
        # Concatenate layer embeddings for each node type
        for ntype, embedding in embeddings.items():
            if self.lightgcn:
                embeddings[ntype] = torch.mean(torch.stack(embedding), dim=0)
            else:
                embeddings[ntype] = torch.cat(embedding, 1)
        return embeddings

    def inference(self, g: dgl.DGLGraph, batch_size=None):
        device = 'cuda' if self.use_cuda else 'cpu'
        embeddings = self._get_embeddings({ntype: g.nodes(ntype).to(device) for ntype in g.ntypes})
        all_embeddings = {ntype: [embedding] for ntype, embedding in embeddings.items()}

        if batch_size is None:
            g = g.to(device)

        for l, layer in enumerate(self.layers):
            if batch_size is not None:
                next_embeddings = {ntype: torch.zeros((g.number_of_nodes(ntype), self.layer_dims[l]),
                                                      device=embedding.device)
                                   for ntype, embedding in embeddings.items()}

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    device=device)

                # Within a layer, iterate over nodes in batches
                for input_nodes, output_nodes, blocks in dataloader:
                    block = blocks[0]

                    if self.use_cuda:
                        block = block.to('cuda')

                    embedding = {k: embeddings[k][input_nodes[k]] for k in input_nodes.keys()}

                    for ntype, e in layer(block, embedding).items():
                        if len(e):
                            next_embeddings[ntype][output_nodes[ntype]] = e
            else:
                next_embeddings = layer(g, embeddings)

            embeddings = next_embeddings

            for ntype, embedding in embeddings.items():
                all_embeddings[ntype].append(embedding)

        # Concatenate layer embeddings for each node type
        self.embeddings = {}
        for ntype, embedding in all_embeddings.items():
            if self.lightgcn:
                self.embeddings[ntype] = torch.mean(torch.stack(embedding, dim=0), dim=0)
            else:
                self.embeddings[ntype] = torch.cat(embedding, 1)

    def forward(self, users, items, rank_all=False):
        if self.use_cuda:
            users, items = users.cuda(), items.cuda()

        user_embs = self.embeddings['user'][users]
        item_embs = self.embeddings['item'][items]

        if rank_all:
            predictions = torch.matmul(user_embs, item_embs.T)
        else:
            predictions = (user_embs * item_embs).sum(dim=1)

        return predictions

    def predict(self, g: dgl.DGLHeteroGraph, embeddings):
        with g.local_scope():
            users, items = g.edges(etype=('ui'))

            return torch.mul(embeddings['user'][users], embeddings['item'][items]).sum(dim=1)

    def _l2(self, pos, neg, emb):
        users, items_i = pos.edges(etype=('ui'))
        _, items_j = neg.edges(etype=('ui'))

        u_emb, i_emb, j_emb = emb['user'][users], emb['item'][items_i],emb['item'][items_j]

        loss = (1/2)*(u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)) / float(len(users))

        return loss

    def loss(self, node_ids, pos_graph, neg_graph, blocks, **kwargs):
        embs = self.block_embedder(node_ids, blocks, **kwargs)

        pos_preds = self.predict(pos_graph, embs).unsqueeze(-1)
        neg_preds = self.predict(neg_graph, embs).reshape(len(pos_preds), -1)

        diff = (neg_preds - pos_preds).flatten()
        s = torch.nn.functional.softplus(diff)
        loss = torch.mean(s)
        # loss = -m

        if self.lightgcn:
            loss = loss, self._l2(pos_graph, neg_graph, embs)

        return loss, pos_preds > neg_preds

